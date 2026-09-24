# Implementation notes: exl3-load-latency

Running log. Append as work proceeds, not at the end.

## 2026-09-24 — spec opened; the control experiment that changes the model

Ran the previously-never-executed control, `bench_stream_bw_slow`
(`crates/engine/src/metal_linear.rs:2573`), on the M4 Pro. 512 MB synthetic u32 buffer:

```
P0 contig 512MB: min 2.558 med 2.625 ms -> 210 GB/s
P0 strided nsplit=1: min 0.949 med 1.006 ms (192 MB) -> 212 GB/s
P0 strided nsplit=2: min 0.624 med 0.664 ms (192 MB) -> 323 GB/s
P0 strided nsplit=4: min 0.544 med 0.561 ms (192 MB) -> 370 GB/s
P0 strided nsplit=8: min 0.548 med 0.560 ms (192 MB) -> 368 GB/s
```

`stream_sum_strided` replicates the `exl3_gemv` grid (`grid.x = out_tiles * nsplit`,
32 lanes/TG) and walks tiles with stride `out_tiles * twords` words — i.e. exactly the
access pattern the "336 tiny straws 21 KB apart" story blamed for a ~68 GB/s ceiling.

**The strided walk is not slower than a fully linear read** (212 vs 210 GB/s at nsplit=1).
Consequences:

1. The `spec/trellis-dense-repack` premise is **falsified**. Its W1/W4 side-car layouts were
   built to remove a stride that hardware handles natively. This explains, after the fact,
   why both candidates measured *slower* than the plain word layout in
   `bench_dense_ab_slow` (e.g. gate W1 41.4 vs word 60.6 GB/s; down W4 40.3 vs word 53.2).
   Repacking was solving a non-problem.
2. The `exl3_gemv_probe` verdict ("per-symbol decode ALU is effectively free; m0 ≈ m2")
   is **withdrawn as a control**. That probe kept the production per-lane gather structure
   and the production grid, so mode 0 and mode 2 were both pinned at the same non-DRAM
   ceiling. It measured the kernel, not the memory system. The "14 ops/weight = 113 ms
   of ALU" model derived alongside it is not supported by this data either way — T3
   measures it directly instead.
3. The real gap is **inside** `exl3_gemv_spec`: 30–95 GB/s achieved vs 212 GB/s available
   for the same walk with no math. That is the ~3× this spec exists to decompose.

Caveats recorded up front (do not skip these when reading the T4 table):
- 370 GB/s exceeds the 273 GB/s nominal bus figure. Almost certainly SLC residency and/or
  boost-clock variance on a 192 MB working set. The defensible comparison point is the
  **nsplit=1, 212 GB/s** number, which is same-shape against the contiguous control.
- The 210 GB/s contiguous figure is one launch over 512 MB. Real linears are 16–87 MB and
  may sit partly in the SLC. T3/T4 therefore measure at real per-linear sizes and report
  cold vs warm separately rather than quoting a single "ceiling".

## 2026-09-24 — decisions taken with the user before opening this spec
- Drop the uncommitted T4 fused RMSNorm+Hadamard work (targets dispatch overhead, which
  the control says is not the dominant term).
- **Keep** 19.5 t/s as the acceptance bar even though it implies 242.6 GB/s sustained,
  above the 210 GB/s measured linear-read figure (~17.8 t/s at 11.833 GB/token). The gate
  stays unflipped and is expected to stay red. Accepted explicitly.
- Attack `exl3_gemv_spec` first (the K-split path serving gate/up/q/k/v — 5 of 7 linears).
- Delete the dense W1/W4 kernels; keep `stash@{0}` (holds the RT2c host plumbing, now moot).
- Bisect before optimizing; new kernel alongside rather than in-place replacement.
- Session scope: deletion + bisection, then report and stop.

## Environment facts
- No `build.rs` in `crates/kernels`. MSL is `include_str!`'d and compiled at **runtime** by
  `new_library_with_source`, which compiles the *entire* file. `cargo build` cannot catch a
  Metal syntax error; any engine GPU test that constructs `MetalLinear::new` will. Corollary
  for T3: the new bench MSL string only gets syntax-checked when the bench actually runs.
- Test suites at `4b3708f` (re-measured after the T0 revert, release profile):
  engine lib **132 passed / 0 failed / 12 ignored**; kernels **21 passed / 0 failed**.
  (An earlier reading of 133 included the now-discarded T4 test
  `gpu_rms_had_in_batched_matches_separate_passes`; 132 is the correct `4b3708f` count.)
- Gated benches need `KAPLUMBA_RUN_SLOW=1` **and** `--ignored`.
- `spec/bandwidth-saturate` T3 (`gemv_splits(336) = 1`) was committed at `707714e` and then
  reverted inside `4b3708f` after it measured 2.2x slower on `down` — a second instance of
  a plausible idea losing to measurement. Precedent: measure before believing.
