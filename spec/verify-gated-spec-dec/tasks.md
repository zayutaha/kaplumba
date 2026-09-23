# Tasks: Verify-Gated Speculative Decoding to ≥ 20.0 t/s

- [ ] **T1: Row-batched target verify latency probe (R1)**
  - **Files**: `crates/engine/src/metal_linear.rs`, `crates/engine/src/gpu_decode.rs`
  - **Intent**: Write a gated test `bench_verify_batch_latency_slow` measuring wall-clock time of a 2-row and 4-row target forward pass using `exl3_gemm_sg`. Verify that $K=2$ and $K=4$ rows execute in $\le 45$ ms total, confirming the decode ALU amortization on M4 Pro.
  - **Testing**: `env KAPLUMBA_RUN_SLOW=1 cargo test -p kaplumba-engine --release --lib bench_verify_batch_latency_slow -- --ignored --nocapture --test-threads=1`.

- [ ] **T2: KV cache rewind/truncate support for speculative draft (R3)**
  - **Files**: `crates/engine/src/gemma4.rs` (`KvCache`)
  - **Intent**: Implement safe truncation / rewind in `KvCache` to rollback unaccepted speculative draft tokens without allocations or memory leaks.
  - **Testing**: Unit tests verifying cache length and content before and after rewind match non-speculative sequential evaluation.

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
