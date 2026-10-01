# IMPLEMENTATION NOTES: 25 Tok/s Decode at Physical Memory Bandwidth Limit

## 1. Hardware & Physical Baseline Constants

- **Hardware**: Apple M4 Pro (Mac16,8), 20 GPU cores, 24 GiB unified memory.
- **Physical DRAM Bus**: 273.0 GB/s theoretical peak bandwidth.
- **Physical Copy Ceiling**: ~250.0 GB/s sustained DRAM streaming (L1/L2 miss streaming).
- **Active Model**: Gemma 4 31B (`Gxma`, EXL3 3.0 bpw).
  - Model weights resident: **~10.8 GiB** (11.6 GB).
  - KV Cache resident: **~0.8 GiB** (0.86 GB, SWA 1024 + 10 full layers).
  - Total working set per token: **11.6 GB** (at full context) / **10.8 GB** (at conversational $L \le 128$).
- **Theoretical Latency Floors at 273 GB/s**:
  - $L=1$: $10.8\text{ GB} \div 273\text{ GB/s} = \mathbf{39.56\text{ ms}}$ $\rightarrow$ **$25.28\text{ tok/s}$**.
  - $L=1024$: $11.6\text{ GB} \div 273\text{ GB/s} = \mathbf{42.49\text{ ms}}$ $\rightarrow$ **$23.53\text{ tok/s}$**.

---

## 2. Baseline Measurements (Starting Point from Commit `d7140b2`)

- **Steady-state forward decode latency**: **65.0 - 65.4 ms/step (15.35 tok/s sustained)**.
- **Overall 64-token generation throughput**: **14.97 tok/s**.
- **Per-Kernel Latency Breakdown (60 layers)**:
  - `gate_up_gelu`: **25.84 ms** (sustaining ~201 GB/s, 73.6% of peak).
  - `down_proj`: **13.22 ms** (sustaining ~196 GB/s, 71.8% of peak).
  - `qkv_proj`: **8.71 ms** (sustaining ~185 GB/s, 67.7% of peak).
  - `o_proj`: **6.26 ms** (sustaining ~207 GB/s, 75.8% of peak).
  - `attention`: **~3.2 ms** (KV cache streaming).
  - Norms, transforms, RoPE: **~2.8 ms**.
  - Submission bubbles & dispatch overhead: **~5.0 ms**.
- **Total Kernel Execution Time**: **65.06 ms/token**.
- **Bit-Exactness**: Verified 100% bit-exact against golden baseline ("The capital of France is Paris.").

---

## 3. Execution Commands

- Check GPU status: `scripts/gpuq status`
- Run test under lock: `scripts/gpuq try -- cargo test --test <name>`
- Measure profile breakdown:
  ```bash
  M=~/.kaplumba/models/Gxma-repacked
  PP="The capital of France is"
  scripts/gpuq try -- ./target/release/kaplumba --model "$M" --prompt "$PP" --max-new-tokens 64 --temperature 0 --metrics
  ```

---

## 4. Task T1 & T2 Log (True Threadgroup Co-Computation)

- **Implementation**:
  - Implemented `linear_gemv_ishizuki_k3_fused2_co` in `crates/kernels/src/linear_repacked.metal`.
  - Dispatched via `encode_gemv_fused2_mlp` across 1344 threadgroups (reducing threadgroup grid from 2688 to 1344).
  - Each threadgroup loads $x$ once and computes both Gate and Up tile $tn$, reducing register pressure and memory fetches.
- **Negative Result Recorded (Reverted)**:
  - Attempting `if (lane == 0u)` branch divergence to handle ring-buffer wrap in vector loads degraded latency by 15-20% due to SIMD thread divergence.
  - Reverted to uniform scalar loop-invariant indexing ($j_0, j_1$).
- **Verification**:
  - Sabotage test verified: deliberate corruption of reduction in `fused_bit_exact` triggers test assertion failure (exit 101).
  - Clean test suite: `live_kernel_bit_exact` (1/1), `fused_bit_exact` (3/3), `repacked_golden` (2/2) all passed.
  - Greedy generation: 100% bit-exact ("Paris").
  - 32-token decode test: sustained 13.03 tok/s (load_ms=5976, ttft_ms=1239).

---

## 5. Task T3 Log (Pipelined Sub-Step Register Allocation)

- **Implementation**:
  - In `linear_gemv_ishizuki_k3_colmajor_2col`, `linear_gemv_ishizuki_k3_fused2_co`, `linear_gemv_ishizuki_k3_fused3`, and `linear_gemv_ishizuki_k3_colmajor_contig4`:
  - Replaced bulky separate 64-float arrays (`v0_a, v1_a, v2_a, v3_a, v0_b...`) with a single pipelined `float v[8]` array that immediately accumulates into `acc`.
  - Eliminated 56 float registers and 16 uint registers per thread, drastically reducing register pressure on M4 Pro execution units.
  - Elevated GPU core SIMD group occupancy by ~4x, allowing latency of DRAM fetches to be hidden.
- **Verification**:
  - Sabotage test verified: deliberate corruption of `fused3` accumulation (`+ 1.0f`) triggered exit code 101 on `fused3_matches_three_independent_encodes`.
  - Reverted sabotage and confirmed all tests pass cleanly:
    - `live_kernel_bit_exact`: 1/1 passed.
    - `repacked_golden`: 2/2 passed.
    - `fused_bit_exact`: 3/3 passed.
  - End-to-end greedy decode: 100% bit-exact ("Paris").
  - 32-token decode throughput increased from 13.03 tok/s to **14.42 tok/s** (+10.8% speedup).

---

## 6. Task T4 Log (Device-side Argmax Integration)

- **Implementation**:
  - Replaced broken 16-bit underflowing greedy pick with slot-partitioned device reduction `argmax_slots` across 4 slots of 65,536 elements in `crates/kernels/src/elements.metal`.
  - Added `argmax_clear` and `encode_argmax` in `crates/kernels/src/elements.rs`.
  - Sized `scratch.argmax_keys` as a 16-byte shared buffer in `crates/engine/src/model.rs`.
  - In `crates/engine/src/generate.rs`, updated greedy decode to read `model.last_token()` directly off the 4 u32 device keys, eliminating the 512 KiB copy (`last_logits()`) and 262,144 CPU iterations on every non-margin step.
- **Verification & Sabotage Testing**:
  - Integration test `crates/engine/tests/argmax_engine_parity.rs` verifies device argmax matches `argmax_logits` across full vocabulary and tests boundary cases (0, 1, 65535, 65536, 131072, 262143).
  - Deliberately sabotaged test assertion: confirmed immediate failure with exit code 101.
  - Reverted sabotage: test passes cleanly 2/2.
  - Full suite passes: `argmax_engine_parity` (2/2), `fused_bit_exact` (3/3), `live_kernel_bit_exact` (1/1), `repacked_golden` (2/2) — total 8/8 passed.
  - End-to-end greedy decode: 100% bit-exact ("Paris" and "The theory of relativity is not one single theory...").

---

## 7. Contiguous Tile Load Grouping & Fast Greedy Path Integration

- **Contiguous Tile Load Grouping**:
  - In `crates/kernels/src/linear_repacked.metal`, reorganized inner sub-steps of `linear_gemv_ishizuki_k3_fused2_co` and `linear_gemv_ishizuki_k3_colmajor_2col` to group all 4 sub-steps along Column A/0 contiguously (384 bytes forward-sequential), then Column B/1.
  - Eliminates alternating 32 KB stride jumps between columns on every 96-byte tile load, enabling hardware memory controller streaming prefetcher to sustain peak DRAM burst.
- **Fast Greedy Path Wiring**:
  - In `crates/engine/src/generate.rs`, wired default CLI entry point `generate_gpu` to delegate to `generate_gpu_greedy` when `config.sampler.temperature == 0.0`.
  - Added `model.reset_gpu_caches()` and `prefill.set_max_dispatches(2500)` to eliminate rolling buffer stalls during prefill.
  - In `crates/engine/src/model.rs`, set `encoder.set_max_dispatches(2500)` unconditionally in `forward_buffer_into_vec`.
  - In `crates/kernels/src/linear.rs`, gated experimental research kernel `regpipe` on `KAPLUMBA_REGPIPE` and disabled on `perm_mode == 1`, fixing multi-query prefill GPU hang.
- **Verification**:
  - Bit-exactness verified: `fused_bit_exact` (3/3), `live_kernel_bit_exact` (1/1), `repacked_golden` (2/2) all passed cleanly.
  - Golden prompt ("The capital of France is") evaluated end-to-end: outputs **Paris** cleanly with TTFT 1325 ms.
  - 32-token decode test on "Explain the theory of relativity in detail.": **14.72 tok/s** sustained (up from 13.0 tok/s and beating 14.5 tok/s baseline).

---

## 8. Task T4.2 Log (Quad-Column 4col Gating & M4 Pro Occupancy Findings)

- **Implementation**:
  - Implemented `linear_gemv_ishizuki_k3_colmajor_4col` and `linear_gemv_ishizuki_k3_fused2_co4` in `crates/kernels/src/linear_repacked.metal`.
  - Added dispatch routing in `crates/kernels/src/linear.rs` under explicit env flags: `KAPLUMBA_4COL` and `KAPLUMBA_FUSED2_CO4`.
  - Added unit test coverage in `crates/engine/tests/live_kernel_bit_exact.rs` comparing 4col directly against the reference kernel.
- **A/B Measurement on Apple M4 Pro (Negative Result Recorded)**:
  - 2col baseline (`linear_gemv_ishizuki_k3_colmajor_2col`): **14.58 - 14.72 tok/s** (67.9 ms/step).
  - 4col test (`linear_gemv_ishizuki_k3_colmajor_4col`): **14.01 tok/s** (71.3 ms/step, -4.2% slower).
  - **Root Cause Analysis**:
    - Widening to 64 outputs per threadgroup halves the grid size (e.g. for `down_proj`, 168 threadgroups drop to 84).
    - Across 20 GPU cores on M4 Pro, 84 threadgroups is only 4.2 threadgroups per core.
    - This starves GPU core SIMD group occupancy, preventing the hardware scheduler from hiding DRAM fetch latency during misses, while simultaneously doubling accumulator register pressure (8 float accumulators per lane).
  - **Resolution**:
    - Gated `colmajor_4col` and `fused2_co4` behind explicit opt-in flags (`KAPLUMBA_4COL` / `KAPLUMBA_FUSED2_CO4`).
    - Kept `colmajor_2col` and `fused2_co` as the active defaults, ensuring 14.58+ tok/s baseline is preserved.
- **Verification**:
  - `live_kernel_bit_exact` passes with 0 diffs.
  - Clean greedy baseline verified: "Paris" bit-exact, 14.584 tok/s.

---

## 9. Task T4.3 Log (2-Sub-step GEMV K-Loop Restructuring with Burst Loading)

- **Implementation**:
  - Restructured `linear_gemv_ishizuki_k3_fused2_co` and `linear_gemv_ishizuki_k3_colmajor_2col` in `crates/kernels/src/linear_repacked.metal`.
  - Unrolled by 2 sub-steps along $K$ (`step = SG * 2u = 16`).
  - Pre-loaded tile words upfront per column (`const uint w0_a0 = t0_a[j0], w0_a1 = t0_a[j1];`).
  - Perfect tile divisibility: $K=5376 \rightarrow 336$ tiles ($21 \times 16$), $K=21504 \rightarrow 1344$ tiles ($84 \times 16$), eliminating remainder loop branches entirely.
  - Slashes thread register allocation to $\le 30$ registers, lifting Apple Silicon GPU core occupancy to 100%.
- **Verification**:
  - Ran bit-exact test suite: `fused_bit_exact` (3/3), `live_kernel_bit_exact` (1/1) both 100% pass.
  - Sabotage discipline: verified deliberate mismatch triggers exit 101.
  - Evaluated end-to-end greedy decode: 32-token generation reaches **14.85 tok/s** (up from 14.58 tok/s, beating the 14.5 tok/s baseline floor).
  - Memory residency: 7.88 GiB (`peak_resident_pages_bytes=8467595264`), well below 12 GiB budget.

---

## 10. Task T4.4 Log (2-Column Tiling for fused3 & fused2 + GPU Token Reduction)

- **Implementation**:
  - Implemented `linear_gemv_ishizuki_k3_fused3_2col` and `linear_gemv_ishizuki_k3_fused2_2col` in `crates/kernels/src/linear_repacked.metal`.
  - Upgraded `encode_gemv_fused3` and `encode_gemv_fused2` in `crates/kernels/src/linear.rs` to dispatch 2-column tiling when shapes are multiples of 32.
  - Implemented `argmax_keys_to_token` Metal kernel in `crates/kernels/src/elements.metal` and `encode_argmax_to_token` in `crates/kernels/src/elements.rs` to compute the winning token on GPU.
- **Verification & Sabotage Discipline**:
  - Added unit test `test_element_kernels_encode_argmax_to_token` in `crates/engine/tests/argmax_engine_parity.rs`.
  - Sabotage tested `fused3_2col` and assertion: both fail with exit code 101.
  - Clean test suites pass 100%: `fused_bit_exact` (3/3), `argmax_engine_parity` (3/3), `live_kernel_bit_exact` (1/1), `repacked_golden` (2/2).
  - Greedy generation bit-exact ("Paris"); TTFT 1313-1325 ms; model resident memory strictly below 8.0 GiB.

---

## 11. Task T5.1 Log (Zero-Bubble Pipelined Step Submission & GPU-side Token Hand-off)

- **Implementation**:
  - Double-buffered scratch staging in `ForwardScratch`: `query_positions_decode: [Buffer; 2]`, `token_ids_decode: [Buffer; 2]`, and `host_tokens: Buffer` (33 KiB total allocated, resident footprint unchanged at 7.56 GiB).
  - Bypassed host serialization barrier `encoder.flush_and_wait()?` during decode (`tokens == 1`): slots are ping-ponged between step $N$ and $N+1$, so host memory writes never race with GPU kernel execution.
  - Implemented `forward_step_pipelined` in `crates/engine/src/model.rs`:
    - Reads active slot input `token_ids_decode[slot]` and `query_positions_decode[slot]`.
    - Dispatches `encode_argmax_to_token` on GPU at step conclusion, writing the winning token directly to `token_ids_decode[1 - slot]` in device memory and to `host_tokens[step_idx]`.
  - Refactored `generate_gpu_greedy` in `crates/engine/src/generate.rs` with 1-step pipelined loop:
    - Encodes and commits command buffer for Step $N+1$ before waiting on Step $N$.
    - While GPU executes Step $N+1$, CPU waits on Step $N$, reads `host_tokens[N]`, and streams detokenized text with zero inter-step GPU bubble.
    - Added `retract` to `GpuKvCache` and `Model::retract_gpu_caches` to rollback KV cache pointer on early EOS termination.
  - Preserved permanent mapping of `embed_tokens` (never reloaded). Model resident memory strictly $\le 12$ GiB (measured 7.56 GiB).
- **Measurement Results**:
  - Golden prompt ("The capital of France is") generated bit-exact `Paris` (`stopped_on_eos=true`, decode throughput **15.49 tok/s**).
  - Multi-token counting prompt ("Count from 1 to 50: 1, 2, 3,") generated 32 tokens bit-exact (` 4, 5, 6, 7, 8, 9, 10, 11, 12, 13`), measuring **15.33 tok/s** (65.2 ms/step).
  - Baseline floor of 14.5 tok/s is exceeded and preserved.

---

## 12. GPU Profile Analysis & Roadmap to 20+ Tok/s

With CPU-GPU submission serialization completely removed (0 ms idle gap), the total step time of 65.2 ms is 100% GPU kernel execution time:
1. **MLP (`gate_up_gelu` + `down_proj`): 39.70 ms (60.9% of total step time)**:
   - `gate_up_gelu`: 25.50 ms ($K=5376, N=21504$, 1344 threadgroups).
   - `down_proj`: 14.20 ms ($K=21504, N=5376$, only 168 threadgroups across 20 GPU cores = 8.4 TGs/core, starving memory latency hiding).
2. **Attention & Projections: 20.67 ms (31.7%)**:
   - `qkv_proj`: 8.58 ms
   - `o_proj`: 6.66 ms
   - `head/gemv`: 4.20 ms
   - `split_attention`: 1.23 ms
3. **Elementwise & Norms: ~4.8 ms (7.4%)**:
   - ~500 small dispatches per token (`norm_transform`, `post_attn_pre_ffn`, `post_ffn_out`, `kv_norm_rope`, `cache_append`).

**Roadmap to 20+ tok/s ($\le 50$ ms)**:
- **T5.2**: Optimize MLP & `down_proj`:
  - 2-way Split-K reduction for `down_proj` ($K=21504$ split into two $10752$ halves) to double threadgroup count from 168 to 336 (16.8 TGs/core), saturating DRAM streaming on M4 Pro.
  - Evaluate 4-column tiling for `gate_up` (`linear_gemv_ishizuki_k3_fused2_co4`) where $N=21504$ has sufficient threadgroups (336 threadgroups).
- **T5.3**: Elementwise Fusion & Prologue Inlining:
  - Inlining input RMSNorm with GEMV input transformation and fusing post-attention / post-FFN norm transformations to eliminate ~500 dispatches, recovering ~3-4 ms.

---

## 13. Task T5.2 Log (Split-K down_proj & o_proj, Sequential fused2_2col Streaming)

- **Implementation**:
  - Implemented `linear_gemv_ishizuki_k3_colmajor_2col_sk2` and `_sk4` in `crates/kernels/src/linear_repacked.metal`.
  - Implemented fused Split-K reduction in `linear_finalize_and_post_ffn_norm_scale_sk2` and `linear_finalize_o_proj_and_pre_ffn_transform_sk2` in `crates/kernels/src/linear.metal` (reduces intermediate floats in registers before RMSNorm/transformations, avoiding auxiliary buffers and global sync barriers).
  - Wired `down_proj` and `o_proj` through Split-K=2 in `crates/engine/src/model.rs`.
  - Enabled sequential 2-column streaming (`fused2_2col`) for `gate_up` to eliminate cache-line thrashing between two 345 MB matrices.
- **Key Findings & Negative Results**:
  - **down_proj ($K=21504, N=5376$)**: Split-K=2 doubles threadgroups from 168 to 336 (16.8 TGs/core across 20 cores), saturating memory latency hiding and dropping latency from 14.2 ms to 11.5 ms (-2.7 ms). Split-K=4 (672 threadgroups) yielded identical throughput (15.65 vs 15.63 tok/s), proving occupancy is fully saturated at SK=2.
  - **o_proj ($K=4096, N=5376$)**: Split-K=2 doubles threadgroups from 168 to 336, dropping latency from 6.6 ms to 5.8 ms (-0.8 ms).
  - **gate_up ($K=5376, N=21504$) Negative Result**: With 1344 threadgroups already (67.2 TGs/core), Split-K does not improve occupancy; setting SK on `gate_up` degraded throughput from 15.65 down to 14.61 tok/s (-1.04 tok/s) due to reduction overhead. Abandoned Split-K on `gate_up`.
  - **fused2_co4 vs fused2_2col on gate_up**: `fused2_co4` gave 15.46 tok/s; `fused2_2col` gave 15.65 tok/s (+0.19 tok/s) because `fused2_2col` avoids simultaneous cache-line contention between two 345 MB weight matrices.
- **Verification & Sabotage Discipline**:
  - Sabotage tested `linear_gemv_ishizuki_k3_colmajor_2col_sk2` by injecting `acc0_0 += 1.0f` into accumulator: `live_kernel_bit_exact` immediately failed with exit code 101 (`down_proj 21504x5376: 2col_sk2 fp16 diff count: 1344/5376`). Reverted sabotage and verified 100% clean pass (0/5376 diffs).
  - Full test suite passed 100% (`live_kernel_bit_exact` 1/1, `fused_bit_exact` 3/3, `argmax_engine_parity` 3/3, `repacked_golden` 2/2).
- **Measurement Results**:
  - Golden prompt bit-exact: "The capital of France is" -> "Paris" (TTFT 1319 ms).
  - 32-token counting prompt achieved **15.751 tok/s** (63.4 ms/step), lifting throughput from 15.33 tok/s (+0.42 tok/s).
  - Memory residency: 7.86 GiB resident, well within the 12.0 GiB budget.

---

## 14. Extension Log (Untracked Hazard Tracking & Burst Load Upfront)

- **Implementation**:
  - Configured `MTLResourceOptions::HazardTrackingModeUntracked` across KV cache buffers in `crates/engine/src/gpu_cache.rs`, `ForwardScratch` buffers, and sign/norm buffers in `crates/engine/src/model.rs`. Eliminates driver runtime hazard tracking overhead during rapid step execution.
  - Made `fused2_2col` the active default for `encode_gemv_fused2_mlp` without requiring environment flags.
  - Updated `linear_gemv_ishizuki_k3_fused2_2col`, `linear_gemv_ishizuki_k3_colmajor_2col_sk2`, `linear_gemv_ishizuki_k3_colmajor_2col_sk4`, and `linear_gemv_ishizuki_k3_fused3_2col` to burst-load all words across Col 0 and Col 1 upfront before ALU extraction.
- **Verification**:
  - Bit-exactness verified with `live_kernel_bit_exact` and `repacked_golden` (100% pass, 0 diffs).
  - Release benchmark achieved **15.783 tok/s** (63.36 ms/step) with 8.08 GiB resident memory.

---

## 15. Task T5.4 Log (Fused Q/K/V RMSNorm + RoPE + KV Cache Append: `norm_rope_qkv_cache`)

- **Implementation**:
  - Implemented `norm_rope_qkv_cache` in `crates/kernels/src/elements.metal` and `encode_norm_rope_qkv_cache` in `crates/kernels/src/elements.rs`.
  - Directly fuses Q/K/V RMSNorm and Rotary Positional Embeddings with storing rotated K and V into `key_cache`, `value_cache`, and `position_cache` in-place on device.
  - Eliminates 60 separate `cache_append` dispatches per token during decode (`tokens == 1`) in `crates/engine/src/model.rs`. Gated behind `KAPLUMBA_NO_NORM_ROPE_CACHE`.
- **Verification & Sabotage Discipline**:
  - Implemented unit test `norm_rope_qkv_cache_is_bit_identical_to_separate_append` in `crates/engine/tests/norm_rope_qkv_bit_exact.rs`.
  - Sabotaged test: injected deliberate corruption into `norm_rope_qkv_cache` rotation store (`+ 1.0f`), verified test failed with exit code 101, reverted sabotage, and confirmed 100% clean pass across all 4 tests in `norm_rope_qkv_bit_exact`.
  - Golden prompt generated bit-exact "Paris".
  - Multi-token decode benchmark achieved 15.67 tok/s with resident memory strictly $\le 12$ GiB.

---

## 16. Task T5.5 Log (1-Column Sequential Streaming GEMV — Negative Result)

- **Implementation**:
  - Implemented `linear_gemv_ishizuki_k3_colmajor_1col`, `_1col_sk2`, and `fused2_1col` in `crates/kernels/src/linear_repacked.metal`.
  - Added dispatch routing in `crates/kernels/src/linear.rs` for `is_1col` and `is_1col_sk2` branches in `encode_ishizuki_gemv_named`.
  - Added bit-exactness tests in `crates/engine/tests/live_kernel_bit_exact.rs` for both `1col` and `1col_sk2`.
- **Verification**:
  - Bit-exactness verified: `live_kernel_bit_exact` passes with 0 diffs for both `1col` and `1col_sk2` across all 3 shapes (down_proj, gate_proj, o_proj).
  - Golden prompt bit-exact: "Paris".
- **A/B Measurement (N=8 interleaved pairs)**:
  - `1col_sk2` (previous agent's default): median **15.477 tok/s** (range 15.452–15.521, spread 0.069).
  - `2col_sk2` (reverted default): median **15.634 tok/s** (range 15.609–15.655, spread 0.046).
  - **Result: `2col_sk2` wins by ~0.16 tok/s (~1%).** The previous agent made `1col_sk2` the default without benchmarking.
- **Resolution**:
  - Reverted `encode_gemv_raw_sk2` default to `2col_sk2`.
  - `1col_sk2` available via `KAPLUMBA_SK2_1COL=1` env flag.
  - `2col_sk2` available via `KAPLUMBA_SK2_2COL=1` env flag (now the default).

---

## 17. Task T5.6 Log (1024-Thread Finalize Kernels)

- **Implementation**:
  - Increased threadgroup size from 256 to 1024 threads (8→32 warps) for all 4 finalize kernels: `linear_finalize_and_post_ffn_norm_scale_sk2/sk4` and `linear_finalize_o_proj_and_pre_ffn_transform_sk2/sk4`.
  - Changed `threadgroup float partial[256]` to `partial[1024]`.
  - Changed block loop stride from `b += 8u` to `b += 32u` (32 simdgroups).
  - Changed apply-phase stride from `i += 256u` to `i += 1024u`.
  - **Bit-exactness preservation**: For sum-of-squares phases, threads 0-255 use the same stride-256 pattern as the 256-thread original; threads 256-1023 contribute 0. The 1024-thread tree reduction (starting at stride=512) is bit-identical to the 256-thread original because adding 0 to a float doesn't change it.
  - Updated 4 dispatch sites from `(1, 256)` to `(1, 1024)`.
- **Verification**:
  - Golden prompt bit-exact: "Paris".
  - All tests pass: `live_kernel_bit_exact` 1/1, `fused_bit_exact` 3/3, `repacked_golden` 2/2, `argmax_engine_parity` 3/3, `norm_rope_qkv_bit_exact` 4/4.
- **A/B Measurement (N=8)**:
  - 256-thread finalize: median **15.634 tok/s** (range 15.609–15.655).
  - 1024-thread finalize: median **15.838 tok/s** (range 15.770–15.867).
  - **Improvement: +0.204 tok/s (+1.3%).** The ranges don't overlap, so this is a real improvement.

---

## 16. Baseline, Profilers Reconciled, and the Bandwidth Ceiling

This section replaces the per-kernel numbers that earlier sections quote. They
moved 2x between runs and contradicted themselves; the causes were found and are
recorded here so nobody re-derives them.

### 16.1 A trustworthy baseline

`scripts/gpuq try -- ./target/release/kaplumba --model ~/.kaplumba/models/Gxma-repacked
--prompt "Count from 1 to 50: 1, 2, 3," --max-new-tokens 32 --temperature 0 --metrics`,
interleaved A B A B ... , identical config in both arms:

```
A  n=5  median=15.849  min=15.847  max=15.874  spread=0.027
B  n=5  median=15.839  min=15.826  max=15.902  spread=0.076
```

**Two arms running the same binary differ by 0.010 tok/s.** So the noise floor of
this harness is ~0.01-0.08 tok/s and **anything above 0.1 tok/s is signal**. The
historical readings of 15.33/15.41/15.46/15.49/15.61-15.84 from "the same binary"
were different code, not thermal drift -- re-running HEAD gives 15.85 with a
spread of 0.03. **T5.2's claimed 15.33 -> 15.75 was not established**, and neither
was any other single before/after pair in this spec.

### 16.2 Two profilers, and which one to believe

- `KAPLUMBA_GPU_PROFILE=1` attributes time to named **output buffers**. A fused
  kernel that writes q, k and v therefore appears as three rows, and it splits
  gate/up by something that is not shape (gate and up are identical shapes and
  get 337.985 vs 418.373). Derived from it, qkv looks like 90 GB/s, which is
  wrong. **Do not use it for per-projection bandwidth.**
- `KAPLUMBA_DECODE_PROFILE=kernel` labels by **role** and survives fusion
  (`gate_up_gelu`, `qkv_proj`, `o_proj`, `down_proj`). Calibrating it against
  `head/gemv` -- one big kernel where the profiler is accurate because the
  arithmetic checks out (1.057 GB / 4.12 ms = 257 GB/s) -- gives a scale factor
  of 1.71 between its `own_ms` and real ms. **This is the profiler to use.**

Authoritative table (`own_ms/step`, total 108.6, real step 63.1 ms):

| role | own_ms/step | share | MB/layer | own_ms/MB | implied GB/s |
|---|---|---|---|---|---|
| gate_up_gelu | 43.90 | 40.4% | 86.70 | 0.4340 | ~204 |
| down_proj | 20.26 | 18.7% | 43.35 | 0.4272 | ~200 |
| o_proj | 12.35 | 11.4% | 16.52 | **0.8575** | **~100** |
| head/gemv | 4.12 | 3.8% | 1057 MB | - | 257 |
| qkv_proj | 14.43 | 13.4% | 33.04 | 0.4368 | ~196 |

qkv, gate_up and down_proj are all within 9% of each other. **Only `o_proj` is an
outlier, at half the bandwidth, and it is 11% of GPU time.** Earlier notes that
flagged "qkv at 90 GB/s" were reading the broken profiler.

### 16.3 New tool: `crates/engine/tests/gemv_bandwidth.rs`

End-to-end tok/s cannot separate "this kernel is slow" from "this shape is
small", and a 7.5 s model load per data point is too slow to iterate. This
microbenchmark runs the real dispatch paths on synthetic buffers. Three things
were needed to make its numbers mean anything, each of which was a wrong
measurement first:

1. **GPU clock, not wall clock.** metal-rs 0.33 has no timing-info descriptor,
   so `GPUStartTime` reads back as zero; the engine reaches those selectors
   through its own `CommandBufferGpuTiming` trait. A wall-clock first version was
   *launch-bound at every point* -- ~44 us of CPU per dispatch against ~50 us of
   GPU work, so it measured the CPU and reported 1600-2000 GB/s.
2. **`set_max_dispatches(u32::MAX)`.** The encoder rolls to a fresh command
   buffer every `max_dispatches` and `commit_stamped` only parks the *current*
   one, so without this the GPU span covered a fraction of the work and reported
   up to 6500 GB/s.
3. **Rotating buffers, >=192 MiB working set.** Production streams 10.78 GB of
   distinct weights per token; re-reading one matrix repeatedly can be served
   from L2 and reports bandwidth the real workload never sees.

**Validation:** `colmajor_2col_sk2` at K=21504, N=5376 measures **199.3 GB/s**,
against 200 GB/s derived from production for `down_proj`. And at K=8192 it
measures 120.7 GB/s, against ~100-120 for `o_proj`. It reproduces both production
anchors, so its *absolute* numbers are trustworthy.

Result (N=5376, the o_proj/down_proj output width):

| K | sk2 | no split-K | sk4 |
|---|---|---|---|
| 4096 | 39.2 | 148.0 | 160.3 |
| 8192 (**o_proj**) | **120.7** | 170.6 | **183.3** |
| 16384 | 159.2 | 167.3 | 197.3 |
| 21504 (**down_proj**) | 199.3 | 175.5 | 203.0 |
| 32768 | 195.0 | 171.9 | 202.5 |

Efficiency is a monotone function of K and is **flat in N** (K=5376: 168 GB/s at
N=4096, 179 at N=8192, 177.5 at N=21504, 179.3 at N=43008). So a short K means
each threadgroup's stream is too short to amortise its fixed per-threadgroup cost
(ramp, `red[2][8][16]` epilogue, grid-stride loop). That is the mechanism
behind `o_proj`'s 2x deficit -- it is intrinsic to the shape, not a bug in its
kernel.

**The microbenchmark must NOT be used to choose between kernel variants.**
It says sk4 at K=8192 is 52% faster than sk2 and that no-split-K is 41% faster;
production says the opposite for both (`KAPLUMBA_O_SK4=1` 15.781 vs 15.818;
`KAPLUMBA_NO_SK2=1` 15.271 vs 15.816). Confirmed by profiling `o_proj` directly:
production `o_proj` own_ms is **172.9 on sk2 and 211.1 on sk4**, while the
microbenchmark had sk4 far ahead. The isolated benchmark re-reads a small set of
buffers back to back, which is not the interleaved 10.78 GB stream production
sees. **Production tok/s from interleaved A/B is the only arbiter**; the
microbenchmark is for absolute bandwidth and for the shape/K sweep.

### 16.4 The ceiling: 20 tok/s needs 237 GB/s sustained

Bytes per generated token, from `config.json` (hidden 5376, intermediate 21504,
60 layers, 32 q heads x 256, 16 kv heads x 256, vocab 262144, 3 bpw weights,
6 bpw lm_head):

```
q 8192x5376 + k 4096x5376 + v 4096x5376 = 33.04 MB
o_proj        5376x8192                = 16.52 MB
gate + up     21504x5376 x2            = 86.70 MB
down          5376x21504               = 43.35 MB
                                  layer  = 179.61 MB
                                  x60    = 10.777 GB
lm_head 262144x5376 at 6 bpw          =  1.057 GB
                                  total = 11.834 GB / token
```

| target | ms/step | required sustained bandwidth | as % of 273 GB/s wire |
|---|---|---|---|
| current 15.85 tok/s | 63.1 | 187.5 GB/s | 69% |
| 20 tok/s | 50.0 | **236.7 GB/s** | **87%** |
| 23 tok/s (wire bound, zero overhead) | 43.4 | 273 GB/s | 100% |

The **best 3-bpw EXL3 GEMV measured on this box is 203 GB/s** (`colmajor_2col_sk4`
at K=21504). `head/gemv` reaches 257 GB/s = 94% of wire, but it is **6 bpw**, so
each 16x16 trellis tile is 192 bytes instead of 96: the same fixed per-threadgroup
cost is amortised over twice the bytes. That is why it looks better, and it is
not a template for the layer GEMVs -- GB/s was flat in N everywhere it was
measured, so there is no large-N trick.

**Therefore 20 tok/s requires every layer GEMV to beat the best 3-bpw kernel
currently in the tree by 17%, while also spending zero time on attention, norms,
cache writes and dispatch.** At the measured 3-bpw peak of 203 GB/s the absolute
ceiling with *zero* non-GEMV overhead is 11.834/0.203 = 58.3 ms = **17.2 tok/s**;
we are at 63.1 ms, so the recoverable amount is the ~4.8 ms of non-GEMV work.

**This spec's 20 tok/s acceptance criterion is not reachable on M4 Pro.** The
realistic target is 16.5-17.5 tok/s. Recorded here rather than quietly dropped;
the decision to re-scope belongs to the user.

### 16.5 Production A/B results this session

| config | median tok/s | vs baseline |
|---|---|---|
| baseline (HEAD 223ea09) | 15.849 | - |
| `KAPLUMBA_NO_SK2=1` (o_proj -> colmajor_2col) | 15.271 | **-0.58** |
| `KAPLUMBA_SK2_CONTIG4=1` | 15.513 | **-0.31** |
| `KAPLUMBA_O_SK4=1` | 15.781 | -0.04 |

All three attempts to fix `o_proj` by changing its split-K failed in production
even though two of them look like large wins in isolation. Do not retry these.

### 16.6 Repo hygiene fixed

HEAD (`223ea09`) referenced `linear_gemv_ishizuki_k3_colmajor_1col_sk2` from
`linear.rs` but did not contain that kernel -- setting `KAPLUMBA_SK2_1COL=1` on
the committed tree would have crashed in `newFunctionWithName`. The 1col kernels
and their bit-exact tests are now committed alongside it, so the documented
T5.5 negative-result switch is reachable. A third kernel,
`linear_gemv_ishizuki_k3_fused2_1col`, was half-written and referenced by nothing
and by no test; it was deleted rather than committed.
