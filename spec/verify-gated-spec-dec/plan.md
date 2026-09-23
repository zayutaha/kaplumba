# Plan: Verify-Gated Speculative Decoding to ≥ 20.0 t/s

## Architecture and Data Flow

Plain decode on Gemma 4 31B EXL3 is ALU-roofed at ~144 ms/token (~7.0 t/s) because single-token GEMV requires ~6.5 scalar ALU ops per decoded weight across 31B parameters (~200G ALU ops/token vs M4 Pro's ~1.4–2.0T ops/s GPU ALU capacity).

To break this roof and reach $\ge 20.0$ t/s without quality loss, we implement verify-gated speculative decoding:

```
                  +----------------------------------------------+
                  | Current Context & KV Cache (length = pos)   |
                  +----------------------------------------------+
                                         |
                                         v
               +----------------------------------------------------+
               | Step 1: Draft Stage (Shallow-Layer Target Draft)    |
               | - Evaluates L_draft layers (e.g. layers 0..1 or 2)  |
               | - Predicts K speculative draft tokens (K = 2..4)   |
               | - Latency: ~6-10 ms per draft token                |
               +----------------------------------------------------+
                                         |
                                         | Proposes [t_1, t_2, ..., t_K]
                                         v
               +----------------------------------------------------+
               | Step 2: Target Verify Stage (exl3_gemm_sg)          |
               | - Row-batched forward of K tokens across all 60 L  |
               | - Trellis tiles decoded ONCE; f16 tensor core MMA  |
               | - Evaluates logits at all K positions in parallel  |
               | - Latency: ~30-45 ms total for the entire batch!   |
               +----------------------------------------------------+
                                         |
                                         | Target predictions [p_1, ..., p_{K+1}]
                                         v
               +----------------------------------------------------+
               | Step 3: Verify & Accept Logic (Greedy Identical)   |
               | - Compare t_j with p_j for j = 1..K                |
               | - Find first mismatch m (or m = K if all match)    |
               | - Emit accepted tokens t_1..t_{m-1} + target p_m   |
               | - Trim/rewind KV cache to accepted pos + m + 1     |
               +----------------------------------------------------+
```

### Amortization Math
- Batched target forward of $K=3$ tokens: ~40 ms.
- Draft forward of 3 tokens ($L_{\text{draft}}=2$): $3 \times 8 \text{ ms} = 24 \text{ ms}$.
- Total cycle time: $\approx 64 \text{ ms}$.
- If average acceptance is $2.0$ tokens per cycle:
  $$2.0 \text{ tokens} / 0.064 \text{ s} \approx 31.2 \text{ t/s}!$$
- Even if average acceptance is $1.35$ tokens per cycle:
  $$1.35 / 0.064 \approx 21.0 \text{ t/s}!$$
- With greedy argmax matching, the output is mathematically identical to running plain 60-layer decode token by token.

## File/Module Touchpoints

1. `crates/engine/src/gpu_decode.rs`:
   - Extend `GpuDecoder` to support:
     - `encode_draft_tokens`: runs $L_{\text{draft}}$ layers on the current token and returns draft token predictions.
     - `encode_verify_batch`: runs all 60 layers for $K$ rows using `exl3_gemm_sg`, producing multi-row logits via `lm_head`.
     - `speculative_decode_step`: orchestrates the draft $\to$ verify $\to$ accept/trim loop.
2. `crates/engine/src/gemma4.rs`:
   - Integrate speculative decode into `generate()` or add `generate_speculative()`.
   - Update `decode_throughput_slow` to measure and assert $\ge 19.5$ t/s.
3. `crates/engine/src/cache.rs` (or KV cache in `gemma4.rs`):
   - Add safe rewind/truncate support for speculative rows in `KvCache` if partial draft rejected.
4. `crates/kernels/src/exl3.rs` / `metal_linear.rs`:
   - Ensure `lm_head` and `encode_linear_with_scratch` smoothly handle $m \in [2, 8]$ rows.

## Sequencing & Dependencies

- **Task T1**: Micro-benchmark & E2E probe measuring 2-row and 4-row target forward latency (`bench_verify_batch_latency_slow`). Verify that batched forward executes in $\le 45$ ms for $m \in [2, 4]$.
- **Task T2**: KV cache reservation & rewind mechanism for speculative draft tokens. Ensure `KvCache::truncate(valid_len)` works cleanly on device memory.
- **Task T3**: Implement shallow-layer target draft pass in `gpu_decode.rs` ($L_{\text{draft}} = 2$ or $3$).
- **Task T4**: Implement batched target verification with `exl3_gemm_sg` in `gpu_decode.rs` and greedy acceptance logic.
- **Task T5**: Wire speculative decoding into `generate()` and verify bit-exact parity on test fixtures and France prompt.
- **Task T6**: Profile and tune draft length $K \in [2, 4]$ and draft depth $L_{\text{draft}}$ for maximum throughput on M4 Pro.
- **Task T7**: Flip `decode_throughput_slow` gate to assert $\ge 19.5$ t/s and verify consecutive stable runs without swap growth.

## Risks and Fallbacks

1. **Risk**: `lm_head` multi-row latency might be higher if not taking `gemm_sg`.
   - *Mitigation*: `lm_head` has $N = 262144$ columns. For $M=1$ it is 5.5 ms. For $M=4$, with tensor cores (`exl3_gemm_sg`), the trellis tile decode is shared across all 4 rows, keeping latency well under ~8 ms.
2. **Risk**: Shallow-layer draft acceptance might vary across prompts.
   - *Mitigation*: Even at a modest acceptance rate (e.g. 1.25–1.5 accepted tokens per cycle), the cycle amortization keeps throughput above 20 t/s. If needed, $L_{\text{draft}}$ can be adjusted (e.g., $L=2, 3,$ or $4$).
3. **Risk**: Rejection rollback might corrupt KV cache indices.
   - *Mitigation*: Write comprehensive unit tests for `KvCache::truncate` verifying that keys, values, and rotary indices match the non-speculative baseline exactly.

## Requirement → Task Map

- R1 (2-Row Batched Verify Probe) $\to$ Task T1
- R2 (Shallow-Layer Target Draft Engine) $\to$ Task T3
- R3 (Batched Target Verification & Rejection Rollback) $\to$ Tasks T2, T4
- R4 (Greedy Parity & Coherence Preservation) $\to$ Task T5
- R5 (Throughput Gate $\ge 19.5$ t/s) $\to$ Tasks T6, T7
