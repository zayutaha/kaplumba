# Tasks: Pure Autoregressive Decode $\ge 20\text{ tok/s}$

- [x] **T1: Vectorize and Pipeline EXL3 Dequantization Inner Loop**
  - **Files touched:** `crates/kernels/src/linear_repacked.metal`
  - **Intent:** Unroll the MCG dequantization evaluation in the production GEMV kernels (`linear_gemv_ishizuki_k3_fused2_2col`, `linear_gemv_ishizuki_k3_fused3_2col`, `linear_gemv_ishizuki_k3_colmajor_2col`) across 4 tiles using the ILP pattern verified in `alu_boost.rs` (`arm_decode_u4`).
  - **Testing:**
    1. Sabotage test: Perturb vector constants, confirm `cargo test -p kaplumba-engine --test live_kernel_bit_exact` fails loudly with non-zero diff.
    2. Revert sabotage, run bit-exactness test: confirm 100% bit-exact match across all shapes.
    3. Run `scripts/gpuq try -- cargo run --release -p kaplumba-engine --example crux3_decode_algebra` to confirm GEMV throughput reaches $\ge 240\text{ GB/s}$.
  - **Commit target:** `spec/pure-decode-20tps: T1 vectorize and pipeline EXL3 dequantization inner loop`


- [ ] **T2: Single-Threadgroup Kernel Grid Multi-Core Parallelization**
  - **Files touched:** `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`
  - **Intent:** Widen single-threadgroup grid configurations (`dispatch_groups(..., 1, 1024)`) in:
    - `encode_finalize_o_proj_and_pre_ffn_transform_sk2`
    - `encode_cross_layer_finalize_and_norm`
    - `encode_input_norm_and_qkv_transform`
    Partition the 5376 elements across multiple workgroups (e.g. 42 or 168 workgroups matching 1 full wave) to utilize all 20 GPU cores instead of idling 19 cores.
  - **Testing:**
    1. Run `cargo test -p kaplumba-engine --test perf_probe_criticalpath_latency`.
    2. Run `cargo test -p kaplumba-engine --test perf_probe_dispatch_census` to verify grid widening without breaking layer invariants.
    3. Sabotage test: alter grid element indexing, verify unit test failure, revert.
  - **Commit target:** `spec/pure-decode-20tps: T2 widen single-threadgroup grids across all GPU cores`

- [ ] **T3: Wire Pipelined Fast-Path Sampling into `generate_gpu` for Temp > 0**
  - **Files touched:** `crates/engine/src/generate.rs`
  - **Intent:** `chat.rs` already has the pipelined decode loop supporting `HeadPick::Sample` for `temperature > 0` and `top_k <= 64`, achieving 14+ tok/s. `generate_gpu` in `generate.rs` was falling back to the synchronous unpipelined `forward_buffer_into_vec` loop for `temperature > 0`. Wire the pipelined decode loop into `generate_gpu` when `temperature > 0` and `top_k <= 64`.
  - **Testing:**
    1. Verify bit-exact consistency and RNG adherence: `cargo test -p kaplumba-engine --test probe_sampler`.
    2. Sabotage test: modify sampled token pass-through, verify failure, revert.
  - **Commit target:** `spec/pure-decode-20tps: T3 wire pipelined sampling into generate_gpu for temp > 0`

- [ ] **T4: Command Buffer Rolling Cadence and Lookahead Tuning**
  - **Files touched:** `crates/kernels/src/encoder.rs`, `crates/engine/src/model.rs`
  - **Intent:** Eliminate inter-buffer submission bubbles during decode execution. Ensure command encoder lookahead prevents GPU queue stalls while keeping in-flight buffers strictly within `MAX_IN_FLIGHT`.
  - **Testing:**
    1. Measure command buffer count: `KAPLUMBA_COUNT_BUFFERS=1 cargo test -p kaplumba-engine --test perf_probe_dispatch_census`.
    2. Sabotage test: force excessive buffer limit causing stall, record diagnostic, revert.
  - **Commit target:** `spec/pure-decode-20tps: T4 tune command buffer rolling cadence`

- [ ] **T5: End-to-End Benchmark & Verification**
  - **Files touched:** `spec/pure-decode-20tps/tasks.md`, `spec/pure-decode-20tps/implementation_notes.md`
  - **Intent:** Run full model generation benchmarks (`scripts/gpuq run -- ./target/release/kaplumba --model ~/.kaplumba/models/Gxma-repacked --prompt "Explain the theory of relativity." --max-new-tokens 64 --temperature 1.0 --top-k 64 --metrics`) and interactive chat sessions. Demonstrate $\ge 20.0\text{ tok/s}$.
  - **Commit target:** `spec/pure-decode-20tps: T5 demonstrate >= 20 tok/s pure decode`
