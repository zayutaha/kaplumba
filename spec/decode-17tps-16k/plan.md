# Technical Plan: Decode 17+ tok/s (up to 4k) and 15+ tok/s (up to 16k)

## 1. Architecture & Data Flow

At 17.0 tok/s, each autoregressive decode step has a hard time budget of **$\le 58.82\text{ ms}$**.
At 15.0 tok/s (at 16k context), the time budget is **$\le 66.67\text{ ms}$**.
Our baseline step time on `Gxma-repacked` is **67.4 ms** (14.85 tok/s). To bridge the 8.6 ms gap to 17 tok/s while preventing degradation at 16k context, optimizations are divided into:
1. **GEMV Grid Launch & Scheduling:** Eliminate idle threadgroup launches in `gate_up_gelu` (which currently launches 1,344 threadgroups for 672 tiles of work).
2. **ALU Optimization in Split-K QKV:** Replace 64-bit `ulong` shifts with 32-bit scalar MCG extraction in `fused2_sk2` and `fused3_sk2` (proven in `spec/alu-decode-tgs-audit` to yield +3.6% kernel throughput).
3. **Down Projection Tuning:** Optimize `down_proj` (currently 13.5 ms) grid partitioning and tile accumulation.
4. **Decode Attention Scaling at 4k–16k:** Profile and tune the 10 full-attention layers scanning up to 16,384 keys using online-softmax key splitting so that 16k attention remains under 3 ms across the entire model.

## 2. File & Module Touchpoints

- `crates/kernels/src/linear.rs`:
  - `encode_gemv_fused2_mlp`: Correct threadgroup count from `(tiles_a + tiles_b) / 2` (1344) to `tiles_a / 2` (672) for `fused2_2col` so that zero threadgroups are spawned into idle early exits.
- `crates/kernels/src/linear_repacked.metal`:
  - `linear_gemv_ishizuki_k3_fused2_sk2` & `linear_gemv_ishizuki_k3_fused3_sk2`: Convert 64-bit window assembly to 32-bit scalar MCG extraction (`extract_v_mcg_32`).
- `crates/kernels/src/attention.metal`:
  - Verify key scanning tiling in decode attention for full layers at 4k, 8k, and 16k keys.
- `crates/engine/src/model.rs`:
  - Validate sliding vs. full attention dispatch parameters and verify memory consumption at 16k context.

## 3. Requirement to Task Mapping

| Requirement | Tasks |
|---|---|
| R1 (Tier-1 $\ge 17\text{ tok/s}$ at $\le 4\text{k}$) | T1, T2, T3, T5 |
| R2 (Tier-2 $\ge 15\text{ tok/s}$ at 16k) | T4, T5 |
| R3 (GEMV Kernel Optimization) | T1, T2, T3 |
| R4 (Full-Attention Scalability) | T4 |
| R5 (Bit-Exact Verification) | T1, T2, T3, T4, T5 |
| R6 (Zero Swap Guarantee) | T4, T5 |

## 4. Risks & Fallbacks

- **DRAM Bandwidth Saturation:** Total weights per token are ~11.75 GB. 58.82 ms requires $\approx 200\text{ GB/s}$ effective bandwidth. If DRAM bandwidth peaks below 205 GB/s on mixed kernels, optimizations must target eliminating kernel launch overhead and maximizing memory coalescence.
- **Cache / Memory Pressure at 16k Context:** The 16k KV cache adds ~1.1 GiB. Total physical memory is 24 GiB. Pre-flight checks with `sysctl vm.swapusage` will strictly verify zero swap paging before and after runs.
