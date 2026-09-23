# Implementation Notes: Decode to 20 t/s

Running log. Append as work proceeds. See AGENTS.md §5.

## 2026-09-23 — P0 stream bandwidth probe (lands use TBD)

- `bench_stream_bw_slow` (gated): contig 512 MB = **211 GB/s**; gemv-strided
  (in 2048 × out 1024, twords 24, 192 MB) nsplit=1 = **214**, nsplit=2 = **334**,
  nsplit=4 = **377**, nsplit=8 = **381 GB/s**.
- **Verdict: the decode gemv's ~79 GB/s (168 Gw/s gate) is an in-kernel
  compute/occupancy wall, not a memory wall.** The identical access pattern
  reads at 214+ GB/s with zero decode; splitting the in-tile range across more
  threadgroups raises read throughput to ~380 GB/s (memory-level parallelism).
  P1-T1 (128-thread/8-acc) has ~2.7–4.8× headroom; plain decode at 20 t/s
  (224 GB/s) is credible without speculation. P3 remains the safety net.
- Probe cleanups: `dispatch_thread_groups` clamps grids past 65535/axis and
  under-runs SILENTLY (first contig kernel used 16.7M groups → 0.027 ms "20 TB/s";
  mystery numbers). Use `dispatch_threads` (nonuniform) via flat indices.
  Per-thread output slots beat one-hot atomically-shared u64 for a read bench.
## 2026-09-23 — P1-T1 v12-parity gemv: NEGATIVE, v1 stays (plus ALU-roof finding)

- Shipped `exl3_gemv4` alongside `exl3_gemv` + `gemv_pass_v4` + parity test +
  A/B columns. Production untouched (revertable one-file-away; run `bench_passes_slow`
  vs `bench_gemv_split_sweep_slow` to re-judge).
- **A/B (real artifacts, min-of-12 ms, best nsplit each)**: v4 loses at every
  shape/split. gate 0.683 vs 0.709; down 0.697 vs 0.709; o 0.368 vs 0.377; q
  tie (0.361 vs 0.369); k 0.272 vs 0.275; v 0.271 tie; lm_head 5.847 vs 6.186.
  Quadrupling ILP + 4-SG in-tile stride (zero hot-loop barriers) changes
  nothing → **not dependency/latency bound. The v12 shape does not transfer to
  M4 Pro / Gemma 4 3.0 bpw shapes** (PonyExl3 runs it on M3/M5 Max at k-high
  shapes; our k=3 artifacts have 1344–16384 out-tiles, low per-tile flop).
- **The real wall is decode ALU** (arithmetic-estimate):
  - Per tile (16 weights, k=3): 2 ld + u64 funnel combine/shift ~8 ops;
    extraction 8 × (u64 shift + and) ≈ 16–24; `exl3_decode_f` ≈ 7 ops × 8 ≈ 56;
    16 fma; 4 shuffles; loop ~4 → **~100+ ops / 16 weights ≈ 6–7 ops/weight**.
  - Whole model ≈ 31G weight-evals × 6.5 ≈ 200G ALU ops/token. M4 Pro GPU
    int/fp32 ALU ≈ 1.4–2.0T ops/s (20 cores × 32 lanes × ~2.2–3.1 GHz approx).
    → **~100–140 ms/token ≈ 7–10 t/s ALU ceiling for plain decode. Measured
    142 ms / 7.03 t/s sits essentially ON the roof.**
- **Consequence**: P0's "20 t/s plain decode is credible (224 GB/s)" is
  FALSE — the bus was never the limit; the roof is ALU at ~7–10 t/s. 20 t/s
  plain would need ~3× ALU reduction (only f16x2 packed FMAC could do it,
  blocked by the no-quality-loss constraint at this bpw). **P3 (verify-gated
  speculation) is no longer the fallback — it is the primary path**, and the
  mechanism that makes it hit 20 t/s is the **row-batched GEMM verify**
  (`exl3_gemm_sg`, RGT≤8 rows): decode ALU is spent once per weight tile and
  the simdgroup tensor units do the per-row MACs, so per-position ALU drops to
  ~decode/8 + 1 MAC × 31G ≈ 2 ops/weight/position → ~25–40 ms/batch-step for
  N≤8 draft positions. Draft quality determines accepted tokens/step (~2.5–3×
  real speedup ⇒ ~17–25 t/s).
- Rejected in-kernel "cheaper decode" ideas (documented for the P3/p4 review):
  constant-buffer codebook gather `tbl[cw&0xff]` — data-dependent warp gather on
  the mix output is not reliably cheaper than the 7-op hash-mix PonyExl3 uses;
  dense-nibble repack — the artifact trellis format is fixed (RW bronze), can't
  change offline layout.
- Metric targeted: tg/s (end-to-end). This phase did not move it; it moved the
  ERROR BAR on what plain decode can reach.

## 2026-09-23 — P1-T2 nsplit sweep-argmax chooser

- `gemv_splits` now has a measured argmax override table (v1 geometry):
  `{1344 → 2 (gate), 336 → 8 (down + o share out_tiles=336), 512 → 2 (q),
  256 → 8 (k/v)}`, formula fallback elsewhere. Memory note: hash of the
  bench day — sweep argmax ordering was gate n2 ≈ n4 within noise, o n8 beats
  n4 by 2%, q n2 beats n8 by 3.5%, k/v n8 by ~7% over n2 (reference formula's
  128-floor would have picked n2 — overrides it).
- Policy test updated; kernels 21 + engine fast 117 green. Committed before
  any E2E re-verify (P2).
