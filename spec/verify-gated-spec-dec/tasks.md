# Tasks: Verify-Gated Speculative Decoding to ≥ 20.0 t/s

- [x] **T1: Row-batched target verify latency probe (R1)**
  - **Files**: `crates/engine/src/gpu_decode.rs`, `crates/kernels/src/exl3.rs`
  - **Intent**: Write a gated test `bench_verify_batch_latency_slow` measuring wall-clock time of a 2-row and 4-row target forward pass. Gate `exl3_gemm_sg` on `rows >= 32` so small row batches ($M \le 8$) avoid the 64-row tensor core padding penalty.
  - **Testing**: `env KAPLUMBA_RUN_SLOW=1 cargo test -p kaplumba-engine --release --lib bench_verify_batch_latency_slow -- --ignored --nocapture --test-threads=1`.
  - **Result**: Done. 1-row plain decode = 141.7 ms (7.06 t/s). 2-row batched verify = 249.0 ms (124.5 ms/tok, 8.03 t/s, 1.14x speedup). 4-row batched verify = 374.9 ms (93.7 ms/tok, 10.67 t/s, 1.51x speedup). Validated decode ALU amortization.
  - **Commit**: TBD

- [x] **T2: KV cache rewind/truncate support for speculative draft (R3)**
  - **Files**: `crates/engine/src/kv_cache.rs`
  - **Intent**: Implement safe truncation / rewind in `KvCache::truncate` to rollback unaccepted speculative draft tokens without allocations or memory leaks.
  - **Testing**: `cargo test -p kaplumba-engine truncate_rolls_back_speculative_tokens` passed, all 118 fast engine tests green.
  - **Result**: Done. Truncate correctly resets `l.len` and sequence `self.len`, properly handles sliding window boundaries, rejects invalid lengths > `self.len`, and allows seamless appending at the truncated position.
  - **Commit**: TBD

- [ ] **T3.5: Small-M verify latency probe — decide T4 kernel path (R3 pre-work)**
  - **Files**: `crates/engine/src/gpu_decode.rs` (test only), possibly `crates/kernels/src/exl3.rs`
  - **Intent**: T1 measured 4-row verify at 374.9 ms (93.7 ms/row) — ~8× the 12.5–14.4 ms/row prefill marginal cost (`spec/prefill-scratch-fix`) and ~8× the 12.4 ms/row tensor-core compute floor. The gap: M=2..31 takes `exl3_gemm` (RGT=8, ~0.7 TFLOP/s) while M≥32 takes `exl3_gemm_sg` (BM=64, ~5 TFLOP/s). Probe verify wall-clock for M ∈ {1,2,4,8,16,32,64} with **amortised timing** (one commit+wait per M, never per-linear — the `exl3-load-latency` measurement bug), on real weights, reporting ms/row. This decides whether T4 targets 20 t/s (needs ≤~100 ms for 4 rows) or whether the spec re-baselines to ~12 t/s.
  - **Testing**: `KAPLUMBA_RUN_SLOW=1 cargo test -p kaplumba-engine --release --lib bench_verify_rowsweep_slow -- --ignored --nocapture --test-threads=1`.
  - **Rationale for placement**: plan.md's amortization math assumes ~40 ms for K=3 verify, which T1's own measurement contradicts by ~8×. Measure the real curve before writing T3/T4 code.

- [ ] **T3: Implement shallow-layer target drafter (R2)**
  - **Files**: `crates/engine/src/gpu_decode.rs`
  - **Intent**: Add shallow-layer drafting method (`draft_step`) that runs $L_{\text{draft}} \in [2, 4]$ layers and projects logits to propose speculative tokens.
  - **Testing**: Test shallow draft generation on synthetic and real prompts, measuring per-token draft latency (target $\le 10$ ms).

- [ ] **T4: Implement row-batched target verification & acceptance logic (R3)**
  - **Files**: `crates/engine/src/gpu_decode.rs`
  - **Intent**: Implement `verify_draft_batch` in `GpuDecoder` running all 60 layers for $K$ rows with `exl3_gemm_sg`, computing logits for all draft positions, and applying greedy argmax matching + cache rewind on mismatch.
  - **Testing**: Verify acceptance and rollback correctness against single-step greedy target decoder.

- [ ] **T5: End-to-end speculative generation wiring & coherence verification (R4)**
  - **Files**: `crates/engine/src/generate.rs`, `crates/engine/src/gemma4.rs`
  - **Intent**: Wire speculative decoding into the generation loop. Verify greedy-identical generation on test fixtures and standard prompts ("The capital of France is Paris.").
  - **Testing**: `cargo test -p kaplumba-engine --release --lib trace_chat_prompt_steps_slow -- --nocapture --test-threads=1`.

- [ ] **T6: Tune draft length $K$ and depth $L_{\text{draft}}$ for maximum throughput (R5)**
  - **Files**: `crates/engine/src/gpu_decode.rs`, `crates/engine/src/gemma4.rs`
  - **Intent**: Measure throughput across $K \in [2, 4]$ and $L_{\text{draft}} \in [2, 3]$ to find the optimal operating point on M4 Pro.
  - **Testing**: Benchmark t/s and acceptance rate across prompt lengths.

- [ ] **T7: Enable gate assertion in `decode_throughput_slow` ≥ 19.5 t/s (R5)**
  - **Files**: `crates/engine/src/gemma4.rs`
  - **Intent**: Update `decode_throughput_slow` assertion from `> 0.5` to `≥ 19.5 t/s`.
  - **Testing**: Two consecutive stable runs of `decode_throughput_slow` passing $\ge 19.5$ t/s without swap-file growth.
