# Specification: Bandwidth Saturation (≥ 20.0 t/s Pure Decode)

## 1. Goal

Achieve ≥ 20.0 tokens/second (≤ 50.0 ms/token forward decode) in pure autoregressive single-token decode for Gemma 4 31B EXL3 (~11.20 GB weights) on the Apple M4 Pro (20 GPU cores, 273 GB/s peak bandwidth) while maintaining 100% greedy token-by-token bit-exactness (predicting token 818 "The" in `trace_chat_prompt_steps_slow`).

## 2. Non-goals / Restrictions

- **No Speculative Decoding**: Decode must remain 100% pure autoregressive single-token greedy generation.
- **No Weight Repacking / Format Changes**: Trellises on disk and in memory remain in their original EXL3 memory-mapped positions without copying or alterations.
- **No KV Cache Quantization**: KV cache remains full-precision f16/bf16.
- **No Quality Loss**: Bit-identical greedy token output on all validation prompts.
- **No Extra VRAM / Scratch Growth**: Resident memory must stay within limits; no dynamic allocations on the hot path.
- **Pure Rust + Metal Only**: No Python runtime dependencies.

## 3. Requirements

1. **R1 (Vectorized EXL3 Inner Loop)**: Vectorize the `exl3_decode_f` inner loop using Metal `uint4` SIMD vector arithmetic and `dot(float4, float4)` hardware intrinsics, reducing ALU instruction count per weight from ~7.0 down to ~1.75 instructions/weight.
2. **R2 (Single-Dispatch GEMV for O and Down)**: Replace the 3-dispatch K-split (`had_in` + `gemv_spec` [nsplit=8] + `gemv_reduce_post`) on `o_proj` and `down_proj` with direct `exl3_gemv_post` (nsplit=1), eliminating intermediate DRAM partials write/read.
3. **R3 (Fused RMSNorm + Hadamard Prologue)**: Fuse `input_layernorm` into `had_in_batched` (QKV) and `pre_feedforward_layernorm` into `had_in_batched` (Gate/Up), eliminating standalone RMSNorm dispatches and intermediate `xn` buffers.
4. **R4 (Fused GELU + Hadamard Prologue)**: Fuse `gelu_mul` into the `had_in` prologue for `down_proj`, eliminating standalone GELU dispatch and intermediate `gu` buffer.
5. **R5 (Fused Attention Post-Processing / QKV Prep)**: Fuse Q/K/V norms, RoPE rotation, and KV-cache write into a unified `qkv_prep` kernel, reducing 6 dispatches per layer down to 1.
6. **R6 (End-to-End Acceptance)**: Forward decode throughput in `decode_throughput_slow` must measure ≥ 20.0 t/s (≤ 50.0 ms/token, ≥ 224 GB/s effective weight stream), with `trace_chat_prompt_steps_slow` passing bit-identically (predicting token 818).

## 4. Acceptance Criteria

1. `trace_chat_prompt_steps_slow` passes bit-for-bit (predicting token 818 on step 19).
2. `KAPLUMBA_RUN_SLOW=1 cargo test --release -p kaplumba-engine decode_throughput_slow` measures:
   - `forward-only` latency ≤ 50.0 ms/token (≥ 20.0 t/s).
   - `implied weight stream` ≥ 224.0 GB/s.
3. All 143 existing kernel and engine tests pass without regression.

## 5. Open Questions

None.
