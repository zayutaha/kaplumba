# Technical Plan: Sampled Decode 15.8 tok/s

## Architecture and Data Flow

### 1. High-Occupancy GEMV Kernels
The autoregressive decode loop is bound by DRAM read bandwidth on weight matrices (~12 GB of 3 bpw weights per token). On the Apple M4 Pro (20 GPU cores, 273 GB/s peak), reaching 15.8 tok/s requires sustained memory bandwidth $> 195\text{ GB/s}$ ($1000 / 63\text{ ms} \approx 15.87\text{ tok/s}$).

The 4-trip unroll with `decode_and_accum_4tile_2col` blew register usage to $>128$ registers per thread, slashing threadgroup occupancy per core in half and starving DRAM latency hiding. By restoring the lean 2-trip burst loop with single `float v[8]` array reuse, register usage returns to ~28 registers/thread, allowing maximum threadgroup occupancy (1024 active threads/core) and full memory bandwidth.

### 2. Pipelined Device Sampling in Generation
In `crates/engine/src/generate.rs`:
- Prefill computes initial logits; GPU device argmax/sample produces token 0.
- For each step $N$:
  1. CPU encodes forward step $N+1$ in deferred submission mode using `HeadPick::Sample(params)` and rolls bounded command buffers while GPU executes step $N$.
  2. CPU waits on step $N$, reads `token = read_host_token(step - 1)`.
  3. If `token == EOS`, CPU discards unsubmitted step $N+1$ buffers and exits.
  4. CPU stages the exact 10.7 KB embedding row from mmap into shared scratch.
  5. CPU commits step $N+1$.
This completely removes the 512 KB logit copy to host and CPU float conversions from the hot path.

### 3. Multi-Input-Vector Support for Fused Projections (fused2_mlp and fused3)
Gxma-repacked assigns distinct Hadamard/sign vectors (`suh`) to `gate_proj`, `up_proj`, `q_proj`, `k_proj`, and `v_proj`. Fused kernels previously assumed a single shared input activation vector, forcing decode to fall back to separate `encode_linear` calls.
By adding `stride_x` to `linear_gemv_ishizuki_k3_fused2_2col` and `fused3_2col`:
- Threadgroups processing matrix A read $x$, threadgroups for B read $x + \text{stride\_x}$, and C read $x + 2 \times \text{stride\_x}$.
- `encode_input_transform` prepares each projection's transformed input vector at distinct offsets in `scratch.transformed`.
- This eliminates 6 redundant kernel dispatches per layer and restores fused `gate_up_gelu` and fused `qkv_proj` with 100% bit-exact correctness.

## File Touchpoints

1. `crates/kernels/src/linear_repacked.metal`:
   - `linear_gemv_ishizuki_k3_colmajor_2col`
   - `linear_gemv_ishizuki_k3_colmajor_2col_sk2`
   - `linear_gemv_ishizuki_k3_fused2_2col`
   - `linear_gemv_ishizuki_k3_fused3_2col`
2. `crates/engine/src/generate.rs`:
   - Parameterize pipelined generation over `HeadPick` (`Argmax` vs `Sample(params)`).
   - Enforce `top_k <= 1024` error check.
3. `crates/engine/src/chat.rs`:
   - Enforce `top_k <= 1024` error check.
4. `crates/engine/tests/`:
   - Regression and bit-exactness verification.

## Risks & Fallbacks

- **Risk:** Register allocation variance in Metal compiler.
  - *Mitigation:* `gemv_bandwidth` and `live_kernel_bit_exact` verify occupancy and bandwidth immediately after editing the metal kernels.
- **Risk:** Out-of-bounds `top_k` behavior.
  - *Mitigation:* Explicit unit test verifying `top_k = 1025` returns `SamplerError::DeviceTopKTooLarge`.
