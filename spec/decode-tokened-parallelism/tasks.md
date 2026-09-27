# Tasks: decode token-1 parallelism

- [ ] **T1** Time `online_attention` in `elem_bench` at the real decode shape
  (`queries=1, keys=200, q_heads=8, kv_heads=4, head_dim=512`, and the sliding
  variant with `keys=1024`). This is the only large cost still inferred rather
  than measured, and it sizes the second half of the work.
  - Files: `crates/engine/examples/elem_bench.rs`
  - Test: run, record µs

- [ ] **T2** Parallel `rms_norm_weighted` for small `rows`. One threadgroup per
  row, 256 threads, chunked accumulation plus a `threadgroup_barrier` tree
  reduction. Keep the existing one-thread-per-row path for large `rows` so
  prefill is untouched.
  - Files: `crates/kernels/src/elements.metal`, `crates/kernels/src/elements.rs`
  - Test: `elem_bench` under 20 µs at `rows=1`; correctness vs the current
    kernel at `rows` in {1, 2, 7, 64, 134} and widths {128, 5376}

- [ ] **T3** Same for `rms_norm_unweighted` (`pre_ffn_norm`).

- [ ] **T4** Parallel `online_attention` over keys. Split the key range across
  threadgroups; each writes a partial `(max, sum, acc[head_dim])`; a second pass
  combines partials with `exp(m_i - m_global)` rescaling. No fp32 atomics.
  - Files: `crates/kernels/src/attention.metal`, `crates/kernels/src/attention.rs`
  - Test: correctness vs the current kernel for `queries` in {1, 8}, `keys` in
    {1, 200, 1024}, GQA ratios {1, 2, 8}, full and sliding windows

- [ ] **T5** Remove the per-thread `accum[512]` where the grid is small; stage in
  `threadgroup` memory or restructure so occupancy is not capped at 1–2
  threadgroups per core.

- [ ] **T6** Extend `elem_bench` to cover every element and attention kernel on
  the decode path, so a future serial regression is visible as a microsecond
  number.

- [ ] **T7** Re-measure the layer stack with `KAPLUMBA_PROFILE_PHASES`. Gate:
  under 250 ms, from 783 ms.

- [ ] **T8** End-to-end decode. Gate 5 tok/s, stretch 10.

- [ ] **T9** Confirm prefill did not regress (8.4 tok/s at 134 tokens today); the
  new code paths are only selected for small `rows`/`queries`, so this should be
  a no-op, but measure it.

- [ ] **T10** `output_identity` within float tolerance; full suite green.
