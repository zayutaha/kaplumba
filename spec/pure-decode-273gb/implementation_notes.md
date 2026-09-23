# Implementation Notes: Pure Decode 273 GB/s Bandwidth Saturation

## T1 Completion Audit (2026-09-23, session resumption)

State found in the working tree: T1 kernels, bindings, and engine rewire were
already written but uncommitted and untested (session died mid-T1). Full audit
results:

- **MSL kernels present and structurally complete** (`exl3.metal`):
  - `exl3_gemv_post` (L1806): single-linear GEMV, in-threadgroup SRAM butterfly,
    direct f16 out. Body is a faithful copy of `exl3_gemv`'s dot-product
    (dq4 64-bit window path for k≠7 incl. the aligned+tw≤32 `simd_shuffle`
    sub-branch, pair-table funnel for k=7) + `exl3_had_out`'s epilogue.
  - `exl3_gemv_post_fused2` (L2107) / `exl3_gemv_post_fused3` (L2316): same,
    with per-tile trellis pointer select (`is_b ? trellis_b : trellis_a`,
    3-way `sub` select) and `xh` row offset per sibling.
  - `exl3_gemv_reduce_post` (L1115): sums K-split partials `[nsplit, ot, 16]`
    in split order → f16 round → butterfly → svh → f16 out (128 threads/TG,
    8 out-tiles per TG).
- **Bindings present** (`exl3.rs`): pipelines `gemv_post`, `gemv_post_fused2`,
  `gemv_post_fused3`, `gemv_reduce_post` built in `Exl3Gemv::new`; public
  `encode_gemv_post` / `encode_gemv_post_fused2` / `encode_gemv_post_fused3` /
  `encode_gemv_reduce_post`; `encode_linear_with_scratch` rows=1 branch rewired
  to had_in → (`gemv_post` | `gemv` + `gemv_reduce_post`); fused2/fused3
  `*_with_scratch` entries now end in the post-epilogue kernels (no y, no
  had_out dispatch).
- **Param byte-layouts verified against `set_bytes` sizes**: `Exl3Params` 7×u32
  = 28 B, `Exl3Fused2Params` 8×u32 = 32 B, `Exl3Fused3Params` 9×u32 = 36 B.
  `exl3_gemv_post` never reads `p.nsplit` (harmless trailing member, keeps the
  struct identical to `exl3_gemv`'s).
- **Dispatch geometry**: post kernels run 8 out-tiles per 256-thread TG
  (8 warps × 32 lanes, warp w = out-tile, lanes = columns); butterfly runs on
  tid<128 after all 8 warps fill `sh[128]`. Grid = out_tiles.div_ceil(8).
  `exl3_gemv` stays WARPS=1/32 threads (new rows=1 K-split dispatch in the
  rows=1 branch matches `encode_gemv`'s geometry).
- **Rounding-point equivalence argument (why bit-exactness is preserved)**:
  old path rounds the *full* f32 column sum to f16 inside `had_out`
  (`sh[t] = float((half)y[...])`) before the butterfly; new paths round the
  same full sum (nsplit=1: in-kernel `float((half)accA/accB)` after the
  simd_shuffle_down collapse; nsplit>1: `gemv_reduce_post` sums the same
  partials in the same split order as `gemv_reduce` before rounding). Same
  products, same per-split accumulation order, same reduce order, same single
  rounding point → bit-identical f16 outputs.
- **Split-policy consequence**: with the P1-T2 argmax table (gate 2, q 2,
  down/o 8, k/v 8, lm 1) every decode linear *except lm_head* takes the
  `gemv + gemv_reduce_post` route; lm_head (16384 out-tiles → nsplit=1) is the
  only `exl3_gemv_post` caller in production. Both routes must therefore be
  unit-covered — see T1 test below.
- **Arena sizing**: `chunk_scratch_needs` + `gemv_partial_bytes` already
  single-source the `partial` requirement (`metal_linear.rs` maxes it in), so
  the decoder arena covers the new rows=1 K-split partial use without change.
- **Gap found by the audit**: T1's unit test (`gpu_gemv_post_matches_unfused`)
  did not exist yet — written in this session, see T1 entry in tasks.md.
  Note `gpu_encode_fused2_and_fused3_matches_separate_linears` (T5) now
  implicitly exercises the post epilogue too, because its separate/fused
  comparisons both route through the rewired `*_with_scratch` entries.

## Architecture Log & Live Measurements

- **Hardware**: Apple M4 Pro (20 GPU cores, 273 GB/s memory bandwidth).
- **Target Model**: Gemma 4 31B EXL3 (~3.0 bpw). Total weights per token: 11.833 GB.
- **Baseline Measured State**: 144.5 ms/token = 6.92 t/s (81.89 GB/s = 29.99% bus utilization).
- **Primary Bottlenecks Documented**:
  1. Grid under-occupancy on narrow projections (`k`, `v` at 256 tiles yielding only 18.7 GB/s).
  2. 2,165 dispatches per token incurring ~26 ms of hardware bubble latency.
  3. Intermediate fp32 `y` DRAM round-trips for standalone `had_out` (420 dispatches).
- **Key Invariants & Proven Dead-Ends from History**:
  - Generic arbitrary-stride `simd_shuffle` tile weight ladders (T5.3r) are a net loss on M4 Pro due to 160-ALU barrel shuffle cost on 24-word tiles. Direct L1-resident loads are faster.
  - Weights must stay in their original memory-mapped positions without copying (resident memory limit ≤ 14.6 GB). Fused GEMVs use pointer dispatch.

## Tasks Completed (T2, T3, T4, T5)
- **T4 (`exl3_had_in_batched`)**: Multi-row pre-Hadamard operating on row-stacked `suh_stack`. Replaces 2 separate dispatches for Gate/Up and 3 separate dispatches for Q/K/V with a single multi-row launch (`grid.y = 2` or `grid.y = 3`). Verified bit-for-bit identical via `gpu_had_in_batched_matches_separate_passes`.
- **T2 (`exl3_gemv_fused2`)**: Sibling 2-way GEMV dispatching 2,688 out-tiles (Gate 1344 + Up 1344) over 20 cores (134.4 threadgroups/core). Reaches 107 Gw/s in release mode. Verified exact match against separate GEMVs via `gpu_gemv_fused2_matches_separate_passes`.
- **T3 (`exl3_gemv_fused3`)**: Sibling 3-way GEMV dispatching 1,024 out-tiles (Q 512 + K 256 + V 256). Expands K and V grid from 256 to 1,024 tiles (51.2 threadgroups/core), eliminating the 18.7 GB/s under-occupancy collapse. Measured latency dropped from 1.489 ms (separate Q+K+V) to 0.953 ms. Verified exact match via `gpu_gemv_fused3_matches_separate_passes`.
- **T5 (`Fused2Handles` & `Fused3Handles`)**: Zero-copy device handles created on model load with pre-stacked `suh_stack` and `svh_cat` buffers (~4.4 MB total model overhead). Implemented `encode_fused2_with_scratch` and `encode_fused3_with_scratch`. Verified exact match via `gpu_encode_fused2_and_fused3_matches_separate_linears`.
- **T6 (`gpu_decode.rs` integration & MSL loop fix)**:
  - Fixed critical loop shadowing bug in `exl3_gemv_fused2` and `exl3_gemv_fused3` (`uint it = it0` redeclared in inner unroll and tail loops, causing double loop execution and out-of-bounds weight tile reads).
  - Integrated `qkv_fused` (sliding layers) and `qk_fused` (full layers) and `gate_up_fused` into `GpuDecoder::encode_token`.
  - Allocated `qkv_raw` and `gu_fused` scratch arenas in `GpuDecoder`.
  - All 143 engine and kernel tests passed.
  - Verified 20-step chat prompt in `trace_chat_prompt_steps_slow` bit-for-bit with reference, predicting token 818 ("The") on step 19.
  - Measured decode throughput: 140.9 ms/token (7.10 t/s, 79.5 GB/s bandwidth).
  - Profile analysis: 1.28 ms/layer (76.8 ms/token — 54.5% of total decode time) is spent purely in 8 standalone `had_in` and `had_out` dispatches per layer (each taking ~0.16 ms). Fusing these into GEMV epilogue and producer prologues will recover ~76.8 ms and double throughput toward 20 t/s.
