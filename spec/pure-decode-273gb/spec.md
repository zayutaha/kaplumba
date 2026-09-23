# Spec: Pure Decode 273 GB/s Bandwidth Saturation (≥ 20 t/s)

## Goal

Achieve **≥ 20.0 t/s** pure single-token autoregressive decode on Gemma 4 31B EXL3 (~3.0 bpw) on Apple M4 Pro (20 GPU cores, 273 GB/s memory bandwidth) by eliminating the dispatch-bubble avalanche, under-occupied grid starvation, and intermediate DRAM round-trips via sibling linear fusion, batched Hadamard prep, and fused post-Hadamard epilogue.

## Non-goals / restrictions

- **NO speculative decoding, NO n-gram, NO MTP, NO draft models, NO cheats** (standing user directive). 100% pure single-token autoregressive forward pass only.
- **No quality loss** (standing, AGENTS.md §8): bit-for-bit greedy-identical text generation to reference (`"The capital of France is Paris."`).
- **No KV cache quantization** (standing, AGENTS.md §8).
- **No Python in product/engine code** (standing, AGENTS.md §8): 100% pure Rust + raw Metal MSL.
- **No weight duplication or offline re-quantization**: Resident GPU weights must remain ≤ 12.1 GB (well within the 14.6 GB limit). Fused kernels read directly from existing memory-mapped buffer spans.
- **No generic arbitrary-stride `simd_shuffle` tile weight ladders** (proven poison on M4 Pro in T5.3r due to 160-ALU barrel shuffle cost).
- The `≥ 19.5 t/s` gate in `decode_throughput_slow` is flipped **only** after measured proof.

## Requirements

1. **R1 (Sibling Linear Fusion - Gate+Up)**: Implement `exl3_gemv_fused2` and engine bindings to execute `gate_proj` (1344 tiles) and `up_proj` (1344 tiles) in a single monolithic 2,688-tile GEMV dispatch reading their respective `trellis` spans by tile index.
2. **R2 (Sibling Linear Fusion - Q+K+V)**: Implement `exl3_gemv_fused3` and engine bindings to execute `q_proj` (512 tiles), `k_proj` (256 tiles), and `v_proj` (256 tiles) in a single monolithic 1,024-tile GEMV dispatch, expanding `k`/`v` grid occupancy by 4x.
3. **R3 (Batched Pre-Hadamard `had_in`)**: Implement batched multi-row `had_in` dispatch over stacked `suh` buffers (`grid.y = 3` for QKV, `grid.y = 2` for Gate/Up), cutting separate `had_in` launches.
4. **R4 (Fused Post-Hadamard Epilogue `gemv_post`)**: Port the 128-column threadgroup SRAM butterfly and `svh` scaling directly into the GEMV epilogue, eliminating 420 separate `exl3_had_out` dispatches and all intermediate `y` (fp32) DRAM buffer traffic.
5. **R5 (Pipeline Integration in `gpu_decode.rs`)**: Integrate fused sibling and epilogue dispatches into the 60-layer token execution pipeline in `gpu_decode.rs`, collapsing per-layer dispatches from ~36 down to ~14 (~940 dispatches/token).
6. **R6 (End-to-End Proof & Acceptance)**: Verify 100% token output coherence on `trace_chat_prompt_steps_slow` and assert forward decode throughput ≥ 19.5 t/s (target ≥ 20.0 t/s, ≥ 236 GB/s memory bandwidth) in `decode_throughput_slow`.

## Acceptance criteria

1. **Greedy Text Coherence**: `trace_chat_prompt_steps_slow` predicts `"The capital of France is Paris."` with 100% identical token IDs.
2. **Unit & Kernel Test Suite**: All existing kernel tests (21) and engine tests (118) pass with zero regressions.
3. **Throughput Gate**: `decode_throughput_slow` achieves **≥ 19.5 t/s** (target **≥ 20.0 t/s**) with stable RSS and zero swap file growth.
4. **Memory Bandwidth Proof**: Effective memory bandwidth during decode sustains **≥ 236.7 GB/s** (≥ 86.7% of physical bus limit).

## Open questions

*(None. All clarified and confirmed in the planning artifact.)*
