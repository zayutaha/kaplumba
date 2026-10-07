# Implementation Notes: Pure Autoregressive Decode $\ge 20\text{ tok/s}$

## 1. Initial State & Problem Diagnosis
- Hardware: Apple M4 Pro (20 GPU cores, 24 GiB unified RAM, 273 GB/s peak bandwidth).
- Target Model: Gemma 4 31B (`Gxma-repacked`, EXL3 ~3 bpw, 12.055 GB weights).
- Baseline: ~13.5–14.2 tok/s in chat sessions; ~16.0 tok/s in greedy CLI runs.
- Physical requirement for $\ge 20.0\text{ tok/s}$:
  $$\frac{12.055\text{ GB}}{0.050\text{ s}} = 241.1\text{ GB/s}$$
  (88.3% of the 273 GB/s hardware ceiling).

## 2. Empirical Grounding & Probe Evidence
- **Raw DRAM Bandwidth (`probe_bw_250gb.rs`)**: 253.6 GB/s sustained on 12 GB buffer (92.9% of wire peak). DRAM controller easily handles the throughput.
- **ALU vs DRAM Bottleneck (`crux3_decode_algebra.rs`)**: Removing integer latency chains in MCG dequantization lifted GEMV throughput from 203.0 GB/s to 239.5 GB/s on `down_proj` and 217.9 GB/s on `o_proj`.
- **4-Way Unrolling (`alu_boost.rs`)**: Demonstrated 1.114x inner loop speedup with 0 mismatches against reference math (`arm_decode_u4`).
- **Dispatch Census (`perf_probe_dispatch_census.rs`)**: 749 dispatches per step, including 121 single-threadgroup dispatches taking 4.6 µs each vs 0.7 µs for multi-threadgroup dispatches.
- **Critical Path Latency (`perf_probe_criticalpath_latency.rs`)**: Total fixed per-dispatch overhead accounts for ~2.2 ms per token.

## 3. Log of Execution Steps
### Task T1: Vectorize and Pipeline EXL3 Dequantization Inner Loop
- Implemented `decode_and_accum_4tile_2col` in `crates/kernels/src/linear_repacked.metal`.
- Replaced the scalar per-tile dequantization loop inside `linear_gemv_ishizuki_k3_colmajor_2col`, `linear_gemv_ishizuki_k3_fused2_2col`, and `linear_gemv_ishizuki_k3_fused3_2col` with 4-way vectorized parallel tile evaluations (`uint4` vector multiply, AND, XOR).
- **Sabotage Verification**: Deliberately introduced bug in `decode_and_accum_4tile_2col` (`w0_0_0` instead of `w0_0_1`). Running `cargo test -p kaplumba-engine --test live_kernel_bit_exact` immediately triggered 84 mismatches on `down_proj`.
- **Validation**:
  - `cargo test -p kaplumba-engine --test live_kernel_bit_exact`: 0 mismatches across down_proj, gate_proj, o_proj (16.13s).
  - `cargo test -p kaplumba-engine --test decode_matvec_golden`: 2/2 passed (30.59s).
  - `cargo test -p kaplumba-engine --test repacked_golden`: 2/2 passed (4.97s).
  - `scripts/gpuq try -- cargo run --release -p kaplumba-engine --example crux3_decode_algebra`: Confirmed 243.1 GB/s on `down_proj`.

