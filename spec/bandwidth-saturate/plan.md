# Technical Plan: Bandwidth Saturation (≥ 20.0 t/s Pure Decode)

## 1. Architecture and Data Flow

Current state:
- **116.0 ms/token** (8.62 t/s, 96.6 GB/s).
- 24 separate kernel dispatches per layer (1,440 dispatches/token).
- 10–12 ALU instructions per EXL3 weight decode in GEMV hot loop.
- Serial FP32 accumulator dependency chains on `accA` and `accB`.
- Intermediate DRAM buffers: `xn` (RMS output), `gu` (GELU output), `scratch.partial` (8 splits for O and Down).

Target state:
- **≤ 48.0 ms/token** (≥ 20.8 t/s, ≥ 233 GB/s).
- 11 kernel dispatches per layer (660 dispatches/token).
- ~1.75 vector ALU instructions per weight decode using `uint4` and `dot(float4, float4)`.
- Zero intermediate DRAM buffers for `xn`, `gu`, or `partial`.

```text
Current Layer Data Flow (24 dispatches):
x_in ──► [RMSNorm] ──► xn ──► [HadInBatched(3)] ──► xh_qkv
     ──► [GEMVPostFused3] ──► qkv_raw
     ──► [RMSNorm(Q)] ──► [RoPE(Q)]
     ──► [RMSNorm(K)] ──► [RoPE(K)] ──► KV-Cache
     ──► [RMSNorm(V)] ──► [Copy(V)] ──► KV-Cache
     ──► [SDPA Attention] ──► attn
     ──► [HadIn(O)] ──► [GEMVSpec(O, nsplit=8)] ──► partials ──► [GEMVReducePost(O)] ──► o_proj
     ──► [RMSNorm(post_attn)] ──► [Add(residual)] ──► x_mid
     ──► [RMSNorm(pre_ff)] ──► xn_mlp ──► [HadInBatched(2)] ──► xh_gu
     ──► [GEMVPostFused2] ──► gu_raw
     ──► [GELUMul] ──► gu ──► [HadIn(Down)] ──► xh_down
     ──► [GEMVSpec(Down, nsplit=8)] ──► partials ──► [GEMVReducePost(Down)] ──► down_out
     ──► [RMSNorm(post_ff)] ──► [AddScale(residual)] ──► x_out

Optimized Layer Data Flow (11 dispatches):
x_in ──► [RMS_HadInBatched(3, QKV)] ──► xh_qkv
     ──► [GEMVPostFused3 (Vectorized uint4+dot)] ──► qkv_raw
     ──► [QKV_Prep (Norms + RoPE + CacheWrite)] ──► Q & KV-Cache
     ──► [SDPA Attention] ──► attn
     ──► [HadIn(O)] ──► [GEMVPost(O, nsplit=1, Vectorized)] ──► o_proj
     ──► [RMS_Add(post_attn + residual)] ──► x_mid
     ──► [RMS_HadInBatched(2, GU)] ──► xh_gu
     ──► [GEMVPostFused2 (Vectorized uint4+dot)] ──► gu_raw
     ──► [GELU_HadIn(Down)] ──► xh_down
     ──► [GEMVPost(Down, nsplit=1, Vectorized)] ──► down_out
     ──► [RMS_AddScale(post_ff + residual)] ──► x_out
```

## 2. File and Module Touchpoints

1. `crates/kernels/src/exl3.metal`:
   - Vectorize `exl3_decode_f` quad decode:
     ```metal
     inline float4 exl3_decode_quad(uint4 cw4, uint mode) {
         uint4 x = cw4 & 0xFFFFu;
         if (mode == 1u) x = x * 0xCBAC1FEDu;
         else x = x * 89226354u + 64248484u;
         uint4 r = 0x3B603B60u ^ (x & 0x8FFF8FFFu);
         half2 h0 = as_type<half2>(r.x);
         half2 h1 = as_type<half2>(r.y);
         half2 h2 = as_type<half2>(r.z);
         half2 h3 = as_type<half2>(r.w);
         return float4(h0.x + h0.y, h1.x + h1.y, h2.x + h2.y, h3.x + h3.y);
     }
     ```
   - Update inner loops in `exl3_gemv_post`, `exl3_gemv_post_fused2`, `exl3_gemv_post_fused3`:
     ```metal
     uint4 cw0 = uint4(cw0_0, cw0_1, cw0_2, cw0_3);
     float4 w0 = exl3_decode_quad(cw0, p.mode);
     accA += dot(float4(x00, x01, x02, x03), w0);
     ```
   - Implement `exl3_rms_had_in_batched`: Combines RMS normalization of input vector with multi-row Hadamard butterfly in SRAM.
   - Implement `exl3_gelu_had_in`: Combines GELU(gate)*up multiplication with Hadamard butterfly in SRAM.
   - Implement `exl3_qkv_prep`: Combines Q/K RMS norm, RoPE rotation, and V copy into a single dispatch.
   - Implement `exl3_rms_add` and `exl3_rms_add_scale`: Combines RMS norm with residual addition.

2. `crates/kernels/src/exl3.rs`:
   - Bindings and pipeline states for new fused kernels.
   - Set `gemv_splits(336)` to `1` so `o_proj` and `down_proj` use direct `exl3_gemv_post`.

3. `crates/engine/src/metal_linear.rs`:
   - Methods to encode the fused prologue/epilogue kernels.

4. `crates/engine/src/gpu_decode.rs`:
   - Integrate fused kernels into `encode_token` loop.
   - Remove redundant intermediate buffer usages.

5. `crates/engine/src/gemma4.rs`:
   - Verify `trace_chat_prompt_steps_slow` (exact token 818 match).
   - Verify `decode_throughput_slow` meets ≥ 20.0 t/s.

## 3. Requirement to Task Mapping

- R1 (Vectorized Inner Loop) $\to$ T1, T2
- R2 (Direct GEMVPost for O and Down) $\to$ T3
- R3 (Fused RMSNorm + Hadamard Prologue) $\to$ T4
- R4 (Fused GELU + Hadamard Prologue) $\to$ T5
- R5 (Fused QKV Prep) $\to$ T6
- R6 (End-to-End Acceptance) $\to$ T7
