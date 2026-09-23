# Spec: Verify-Gated Speculative Decoding to ≥ 20.0 t/s

## Goal

Achieve end-to-end Gemma 4 31B EXL3 generation speed of **≥ 20.0 t/s** (thermal/gated limit **≥ 19.5 t/s** in `decode_throughput_slow`) on Apple M4 Pro (20 GPU cores, 273 GB/s bandwidth) while maintaining **≤ 14.6 GB resident memory** and **100% bit-exact / greedy-identical output** to the plain target model. We accomplish this by pairing a shallow-layer target draft ($L_{\text{draft}} \in [2, 4]$ layers using the model's existing weights) with row-batched target verification via the simdgroup tensor-core GEMM (`exl3_gemm_sg`), amortizing the decode ALU roofline across draft positions.

## Non-goals / Restrictions

- **No quality loss** (standing project constraint, AGENTS.md §8): Speculative decoding must be strictly verify-gated. Under greedy decoding ($T=0$), draft tokens are accepted if and only if they match the target model's argmax prediction exactly. If a draft token diverges, it is rejected, earlier matching tokens are kept, and the target's true argmax is emitted. Output tokens must match plain greedy decode 100%.
- **No KV cache quantization** (standing constraint).
- **No Python in product runtime** (standing constraint): 100% pure Rust + Metal MSL.
- **Zero additional model weights**: The draft model uses the target model's own resident weights (shallow layer prefix + head), adding 0 bytes of weight memory to remain comfortably within the 14.6 GB resident memory budget (currently ~12.06 GB resident GPU weights).
- **No speculative KV corruption**: On partial draft acceptance or rejection, any speculative KV entries added for unaccepted tokens must be cleanly rolled back or trimmed without memory leaks or race conditions.
- **Never land red**: The `≥ 19.5 t/s` assertion in `decode_throughput_slow` is enabled and flipped only after verified passing runs.

## Requirements

1. **R1 (2-Row Batched Verify Probe)**: Measure and prove that a 2-row and 4-row target forward pass using `exl3_gemm_sg` amortizes the decode ALU roof on M4 Pro, verifying that $K$-row batched target execution achieves substantially lower per-token latency than $K \times$ single-token plain decode.
2. **R2 (Shallow-Layer Target Draft Engine)**: Implement a fast shallow-layer target draft pass in `gpu_decode.rs` using the model's own first $L_{\text{draft}}$ layers ($L_{\text{draft}} \in [2, 4]$). Evaluate per-token draft latency and draft acceptance rate on real prompts.
3. **R3 (Batched Target Verification & Rejection Rollback)**: Implement row-batched target verification in `GpuDecoder`:
   - Propose $K$ draft tokens ($K \in [2, 5]$).
   - Execute target forward across all 60 layers for the $K$ positions in a single row-batched pass with `exl3_gemm_sg`.
   - Verify draft tokens against target argmax: accept $m$ matching tokens ($0 \le m \le K$) plus 1 bonus target token.
   - Synchronize/trim the KV cache to the accepted position ($base\_pos + m + 1$).
4. **R4 (Greedy Parity & Coherence Preservation)**: Verify that generation output on benchmark fixtures and prompts (including "What is the capital of France? -> The capital of France is Paris.") produces the exact same token sequences as plain greedy decoding.
5. **R5 (Throughput Gate ≥ 19.5 t/s)**: Demonstrate `decode_throughput_slow` achieving ≥ 19.5 t/s (target ≥ 20.0 t/s) across consecutive stable runs without swap memory growth.

## Acceptance Criteria

1. `cargo test -p kaplumba-engine --release --lib decode_throughput_slow -- --ignored --nocapture --test-threads=1` passes with measured decode throughput $\ge 19.5$ t/s.
2. `trace_chat_prompt_steps_slow` and greedy generation tests pass with 100% identical token IDs to plain decode.
3. Total process resident memory remains $\le 14.6$ GB (with resident GPU weights at 12.058 GB).
4. All existing kernel and engine tests pass with no regressions.

## Open Questions

None. Confirmed by user: shallow-layer target draft, greedy-identical verify gating, target $\ge 20.0$ t/s.
