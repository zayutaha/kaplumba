# Tasks: EXL3 decode throughput 1.4 -> 20 tok/s

- [~] **T1** Establish the practical bandwidth ceiling. A synthetic streaming
  probe was written and abandoned: its kernel hung in `wait_until_completed`
  while the GPU was demonstrably healthy (the CLI ran fine in the same state), so
  the probe was buggy, not the hardware. **Resolved empirically instead** — T8
  measures achieved GB/s for the real kernel on the real shapes, which is the
  number that actually matters. If the new kernel plateaus far below 273 GB/s,
  the remaining bottleneck is the strided trellis access pattern, not arithmetic
  (see T13).

- [ ] **T2** Add `exl3_decode8` to `linear.metal`: eight decoded values from one
  window setup, sharing the `lo` / `a` / `c` arithmetic already in
  `exl3_code_at`.
  - Files: `crates/kernels/src/linear.metal`
  - Test: unit test against the CPU decoder for rates 1..8, all 8 values each

- [ ] **T3** Add `linear_gemv_exl3` with the reference thread mapping: one thread
  per trellis lane, threadgroup = 256 (8 simdgroups), grid =
  `(out_features/16)*256`, simdgroups striding tile-rows.
  - Files: `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`
  - Test: correctness on a small matrix before wiring the pipeline

- [ ] **T4** Reuse the 4 `x` values (`r0`, `r0+1`, `r0+8`, `r0+9`) for both
  column groups instead of reading 16 values once each.

- [ ] **T5** Reduction via `simd_shuffle_xor` (1, then 2) within the simdgroup,
  then a `threadgroup float red[8][vecs][16]` sum across simdgroups. No atomics.

- [ ] **T6** Wire the new kernel into `LinearKernels`: route integer rates with
  the MCG codebook to it; leave fractional/mul1 on the old path.

- [ ] **T7** Correctness matrix: rates 1..8 × {Plain, Mcg}, square and
  non-square, several `out_features/16` values, compare against the CPU reference
  within float tolerance.
  - Files: `crates/engine/tests/linear_metal.rs`

- [ ] **T8** Measure. `bench_gemv` old vs new on all four real shapes. Target
  ≥ 40% of 273 GB/s on `q_proj` and `up_proj` (from 6–9%).

- [ ] **T9** End-to-end: decode tok/s at a 512-token prompt, greedy. Gate: 10
  tok/s. Stretch: 20.

- [ ] **T10** Confirm prefill did not regress (10 tok/s at 418 tokens today).

- [ ] **T11** Confirm `output_identity` still passes. If float tolerance is
  needed rather than exact text, convert that test and say why.

- [ ] **T12** Retire `linear_gemv_fast1..8` once the generic kernel measures
  faster or equal. Keeping 8 copies of a 16-call inner loop is what created the
  defect.

- [ ] **T13** If T8 plateaus well below 40%, the bottleneck is the strided trellis
  access pattern, not arithmetic. Write the follow-up spec for a tile-swizzled
  layout that lets a simdgroup read contiguous bytes. This requires repacking the
  checkpoint, so it is out of scope here.
