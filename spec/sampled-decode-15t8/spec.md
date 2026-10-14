# Restore 15.8 tok/s Decode Under Temperature=1.0 and Top-K=64

## Goal

Restore autoregressive decode throughput on Apple M4 Pro to at least **15.8 tok/s** for `Gxma-repacked` under production sampling settings (**`temperature = 1.0`**, **`top_k = 64`**, **`top_p = 0.95`**) across both interactive chat (`kaplumba --chat`) and CLI generation (`kaplumba --prompt ...`).

## Non-goals / Restrictions

- **No KV cache quantization.** (AGENTS.md §8)
- **No quality loss.** Temperature-0 greedy output must remain bit-exact.
- **No Python in the engine or runtime.** (AGENTS.md §8)
- **No unbounded command buffer retention or graph scratch.** Retain bounded per-step queue discipline.
- **Do not regress memory residency or mlock guarantees.** Keep MAP_SHARED mapping and zero-copy mlock bounds intact.
- **No silent fallback or clamping for unstagable top-k.** Requests with `top_k > 1024` must fail immediately with an explicit error.

## Requirements

1. **R1 — Lean GEMV Kernel Restoration:** Revert `linear_gemv_ishizuki_k3_colmajor_2col`, `linear_gemv_ishizuki_k3_colmajor_2col_sk2`, `linear_gemv_ishizuki_k3_fused2_2col`, and `linear_gemv_ishizuki_k3_fused3_2col` in `crates/kernels/src/linear_repacked.metal` from the 4-trip unroll / `decode_and_accum_4tile_2col` loop to the lean 2-trip burst loop with `extract_v_mcg_32` and single `float v[8]` reuse (~28 registers).
2. **R2 — Bit-Exact Kernel Verification:** All restored GEMV kernels must pass `live_kernel_bit_exact` and `repacked_golden` with zero fp16 differences against the reference.
3. **R3 — Bandwidth Recovery:** Restore memory bandwidth in `gemv_bandwidth` benchmarks to $\ge 195\text{ GB/s}$ across major shapes ($K=21504, N=5376$ and $K=5376, N=21504$).
4. **R4 — Unified Pipelined Sampling in Generation:** Wire `HeadPick::Sample(params)` into `crates/engine/src/generate.rs`'s pipelined decode loop, overlapping GPU execution for step $N+1$ with CPU token staging for step $N$ when `temperature > 0` and `top_k <= 1024`.
5. **R5 — Strict Top-K Bound Enforcement:** Reject any request with `top_k > 1024` with `SamplerError::DeviceTopKTooLarge` across both chat and CLI generation.
6. **R6 — Target Throughput Acceptance:** Reach $\ge 15.8\text{ tok/s}$ median decode throughput on the canonical benchmark at `temperature = 1.0, top_k = 64, top_p = 0.95`, and at `temperature = 0.0`.

## Acceptance Criteria

- Canonical benchmark (`Count from 1 to 50: 1, 2, 3,`, 32 generated tokens) achieves $\ge 15.8\text{ tok/s}$ median at `temperature = 1.0, top_k = 64, top_p = 0.95`.
- Canonical benchmark at `temperature = 0` achieves $\ge 15.8\text{ tok/s}$ median and bit-exact golden tokens.
- `live_kernel_bit_exact` passes with 0 fp16 differences.
- `top_k = 1025` is rejected with an explicit error.
- All kernel and engine tests pass with no regressions.
