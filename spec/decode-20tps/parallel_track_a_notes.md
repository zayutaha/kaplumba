## Appendix: Parallel Track A: static audit of the 126 vs 199 GB/s gap

Static audit only (no GPU touched). All claims quoted from
`spec/decode-25tok-peak-bandwidth/implementation_notes.md` (§16-18),
`tasks.md` T5.10/T5.11, `crates/engine/tests/gemv_bandwidth.rs`,
`crates/kernels/src/linear.rs`, `crates/engine/src/model.rs`.

### A.1 Why production halves the microbenchmark: the three stated causes

**(a) The microbenchmark's own validation anchors are the disputed numbers.**
§16.3 says the microbenchmark "reproduces" production: *"`colmajor_2col_sk2` at K=21504, N=5376 measures **199.3 GB/s**, against 200 GB/s derived from production for `down_proj`. And at K=8192 it measures 120.7 GB/s, against ~100-120 for `o_proj`. It reproduces both production anchors, so its *absolute* numbers are trustworthy."* But the §16.2 "200 GB/s" anchor comes from the `own_ms` profiler table, and §16.2 itself confesses: *"Calibrating it against `head/gemv` ... gives a scale factor of 1.71 between its `own_ms` and real ms."* §18.2, from the distortion-free ablation harness, restates the same projections at `gate_up 129 GB/s`, `down 126 GB/s`, `qkv 125 GB/s`. The §16.3 "reproduction" compares against the miscalibrated profiler numbers; the ablation says production is really ~126. The microbenchmark's ~199 only "matches production" if production is 200 — which §18.2 says it is not. This internal inconsistency is never reconciled in the notes.

**(b) The microbenchmark does not see production's working set.** §16.3:
*"Production streams 10.78 GB of distinct weights per token; re-reading one matrix repeatedly can be served from L2 and reports bandwidth the real workload never sees."* And §16.3 again: *"The isolated benchmark re-reads a small set of buffers back to back, which is not the interleaved 10.78 GB stream production sees."* The test code confirms: `MAX_WORKING_SET = 768 << 20`, `MIN_WORKING_SET = 192 << 20`, `nbuf` clamped to `(2, 8)` — at most ~768 MiB of distinct packed weights, re-read for tens of ms. Production touches 10.777 GB of distinct weight bytes per token (§16.4's own byte count). The rotating-buffer fix in the microbench shrinks L2 reuse but does not approach the 10.78 GB production stream, so a ~73 GB/s L2/streaming delta is fully plausible.

**(c) The microbenchmark inverts variant rankings — it does not transfer.** §16.3:
*"The microbenchmark says sk4 at K=8192 is 52% faster than sk2 and that no-split-K is 41% faster; production says the opposite for both ... Confirmed by profiling `o_proj` directly: production `o_proj` own_ms is **172.9 on sk2 and 211.1 on sk4**, while the microbenchmark had sk4 far ahead."* §18.2: *"...the layer GEMVs are running at 62% of the measured peak (126/203) — inefficient for decode, but changing variants hasn't moved them (T5.11 exhausted all seven SK/no-SK/grid-capping combinations; microbenchmark's variant ranking is inverted relative to production)."* T5.11 table adds: *"Capping the grid is monotonically worse ... kills the 'fixed per-threadgroup cost at short K' explanation."* So scheduling is exhausted; the 7 scheduling variants all failed in production.

### A.2 Threadgroup-count arithmetic (does occupancy starve production?)

Dispatch equations from `linear.rs`:
- `encode_gemv_raw_sk2`: `groups = ((out_features / 16) / 2) * 2` → `groups * 256` threads, 256/TG.
- `encode_gemv_raw_sk4`: `pairs = (out/16)/2; groups = pairs * 4`.
- `encode_gemv_raw` 2col branch: `pairs = out_features / 32; groups = pairs.min(gemv_max_groups())` (u64::MAX default → uncapped).
- `encode_gemv_fused2_mlp` / `encode_gemv_fused2` (2col branch): `groups = (tiles_a + tiles_b) / 2`.

Production decode shapes (hidden 5376, q 8192, k/v 4096, intermediate 21504):

| projection | path | TG equation | TGs launched | TGs per 20 cores |
|---|---|---|---|---|
| o_proj (K=8192, N=5376) | `encode_gemv_raw_sk2` | (5376/16/2)·2 | **336** | 16.8 |
| down (K=21504, N=5376) | `encode_gemv_raw_sk2` | (5376/16/2)·2 | **336** | 16.8 |
| qkv (q N=8192 + k N=4096) | `encode_gemv_fused2` 2col | (8192/16 + 4096/16)/2 | **384** | 19.2 |
| gate_up (N=21504 ×2) | `encode_gemv_fused2_mlp` 2col | (21504/16 + 21504/16)/2 | **1344** | 67.2 |
| head (N=262144) | `encode_gemv_raw` 2col | 262144/32 | **8192** | 409.6 |

Reading: o_proj/down launch only 336 TGs; T5.11 already showed capping below that (168, 84) is monotonically *worse*, and SK4 (672 TGs) does not help in production (`15.781 vs 15.816`). gate_up's 1344 TGs ≈ 8.4× more, yet it runs at the same ~129 GB/s. Head launches 8192 TGs and reaches 253 GB/s — but it is 6 bpw (192 B/tile vs 96 B/tile at 3 bpw, §16.4), so its better GB/s is attributed to tile size, not TG count alone. Verdict: TG count/occupancy arithmetic is a real constraint for o_proj/down (336 TGs ≈ 2 waves of ~160 on 20 cores), but it does **not** explain why gate_up at 1344 TGs also sits at 129 GB/s, nor why all 7 variants fail. Per-TG streaming efficiency, not TG count, is the limiter for the long-K GEMVs.

### A.3 If all seven variants failed: what the 199 GB/s can and cannot mean

T5.11 + §18.2: every scheduling knob (SK2/SK4/no-SK/contig4/1col/grid-capping) reproduces ~126–150 GB/s in production. Therefore the 199 GB/s from `gemv_bandwidth.rs` is an artifact of the harness conditions, not a transferable property of the kernel+shape. The two remaining candidate real mechanisms:

1. **Working-set / cache effect (the microbenchmark's rotating buffers hide it).** §16.3 itself: the microbench rotates over ≤768 MiB while production streams 10.78 GB; L2 on M4 Pro serves part of any re-read ≤ last-level cache. The microbench's 199 could be "what DRAM gives when the same ~175 MiB cycle through" — not what a cold 10.78 GB stream delivers.
2. **Production-side serialization / dispatch-gap accumulation.** §18.2: *"Dispatch gaps cost 0.7 us each (T5.10), so removing ~180 dispatches buys +0.031 tok/s — dispatch count is not the lever."* Breakdown of that arithmetic: 180 dispatches × 0.7 us ≈ 126 us ≈ 0.126 ms against a 63.1 ms step — negligible, and both the microbench (`set_max_dispatches(u32::MAX)`) and production (`set_max_dispatches(2500)`, model.rs:3820) now run one rolling command buffer, so MAX_DISPATCHES_PER_BUFFER=50 chunking (T5.10-era) no longer applies. This mechanism is largely already excluded by the notes; encoding more bubbles between GEMVs (0.7 us each) cannot account for 40.34 ms vs 26 ms on gate_up.

So the only candidate with a plausible 73 GB/s (57%) scale is a memory-system effect — L2 residency, DRAM page/bank behavior across the true 10.78 GB interleaved stream, or clock behavior — that the ≤768 MiB microbench working set does not reproduce.

### A.4 One decisive experiment (needs the GPU; do NOT run while the other agent holds it)

Modify `crates/engine/tests/gemv_bandwidth.rs` reproduction (do not commit): at the exact down_proj anchor `colmajor_2col_sk2, K=21504, N=5376, rate 3`, sweep **one variable only** — the rotation span `MAX_WORKING_SET` — and hold everything else at the §16.3 conditions (GPU clock timing, `set_max_dispatches(u32::MAX)`, same TARGET_MS):

- run A: current `768 MiB` cap (expect ~199 GB/s, reproducing §16.3);
- run B: cap raised so each rep touches a distinct buffer and the total working set is ≥ 10.78 GB (i.e. ~60 buffers × 179.6 MiB/layer-shaped, cycled once per pass) — expect ~126 GB/s if mechanism 1 is real.

Decision rule: if B collapses to ~126, the gap is a working-set/cache artifact and production's 126 is the true DRAM rate for this kernel — the 199 was never achievable in production and §16.4's 17.2 tok/s ceiling stands as *optimistic*, worth re-deriving from B. If B stays ~199, the loss is elsewhere in production and the next discriminator is inserting production-like inter-kernel dispatch spacing (and per-layer interleaving of qkv/o/gate_up/down buffers at real offsets) into the microbench.

Current microbench allocation (`run` in gemv_bandwidth.rs:124-135): `nbuf = (MAX_WORKING_SET / shape_bytes).clamp(2, 8)` and each buffer is `shape_bytes/2` bytes — for down_proj that is nbuf ≈ 8 × 21.7 MB ≈ 173 MiB. Run B needs only this constant changed (and the `clamp(2, 8)` raised).
