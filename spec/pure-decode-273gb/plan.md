# Plan: Pure Decode 273 GB/s Bandwidth Saturation (≥ 20 t/s)

## Architecture and Data Flow

The engine targets 100% pure autoregressive single-token decode. To saturate the physical 273 GB/s memory bandwidth on the 20-core Apple M4 Pro GPU, we eliminate grid starvation, dispatch bubbles, and DRAM round-trips.

```text
Input Activation x (5376, f16)
   │
   ▼
RMSNorm(x) ──► xn (5376, f16)
   │
   ├─► Batched had_in (Q, K, V) [grid.y = 3, single dispatch]
   │     ├──► xh_q (5376, f16)
   │     ├──► xh_k (5376, f16)
   │     └──► xh_v (5376, f16)
   │
   ▼
exl3_gemv_post_fused3 (1,024 out-tiles: Q 512, K 256, V 256)
   ├── Tile loop: direct device words from trellis_q/k/v
   ├── In-threadgroup 128-column Hadamard butterfly (SRAM)
   └── Multiply svh_cat and write f16 out directly
         ├──► q_out (8192, f16)
         ├──► k_out (4096, f16)
         └──► v_out (4096, f16)
```

In the MLP block:
```text
x_mid (5376, f16)
   │
   ▼
RMSNorm(x_mid) ──► xn_mlp (5376, f16)
   │
   ├─► Batched had_in (Gate, Up) [grid.y = 2, single dispatch]
   │     ├──► xh_gate (5376, f16)
   │     └──► xh_up (5376, f16)
   │
   ▼
exl3_gemv_post_fused2 (2,688 out-tiles: Gate 1344, Up 1344)
   ├── Tile loop: direct device words from trellis_gate/up
   ├── In-threadgroup 128-column Hadamard butterfly (SRAM)
   └── Multiply svh_cat and write f16 out directly
         ├──► gate_out (21504, f16)
         └──► up_out (21504, f16)
   │
   ▼
gelu_mul(gate_out, up_out) ──► gu (21504, f16)
   │
   ▼
had_in(gu) ──► xh_down (21504, f16)
   │
   ▼
exl3_gemv_post(down) [336 out-tiles + HadOut + SVH] ──► down_out (5376, f16)
```

## File and Module Touchpoints

1. `crates/kernels/src/exl3.metal`:
   - Implement `exl3_had_in_batched`: Multi-row `had_in` running `grid.y` transforms in parallel over stacked `suh` buffers.
   - Implement `exl3_gemv_post_fused2`: 2-way sibling GEMV (Gate + Up, 2,688 tiles) with inline 128-column Hadamard butterfly + `svh` epilogue, outputting final `f16` directly.
   - Implement `exl3_gemv_post_fused3`: 3-way sibling GEMV (Q + K + V, 1,024 tiles) with inline 128-column Hadamard butterfly + `svh` epilogue, outputting final `f16` directly.
   - Implement `exl3_gemv_post`: Single linear GEMV with inline Hadamard epilogue (for `o_proj`, `down_proj`, and `lm_head`).
2. `crates/kernels/src/exl3.rs`:
   - Bindings, pipeline builders, and encoding methods for the new kernels.
   - Host unit tests verifying CPU-decoded reference equivalence against the fused kernels.
3. `crates/engine/src/metal_linear.rs`:
   - Add `Fused2Handles` (gate + up) and `Fused3Handles` (q + k + v) structures.
   - Add `encode_fused2_post` and `encode_fused3_post` methods on `MetalLinear`.
4. `crates/engine/src/gpu_decode.rs`:
   - Replace sequential Q/K/V and Gate/Up passes with monolithic fused calls.
   - Remove intermediate `y` buffer allocations and standalone `had_out` calls.
5. `crates/engine/src/gemma4.rs`:
   - Verify `trace_chat_prompt_steps_slow` and flip `decode_throughput_slow` to assert ≥ 19.5 t/s.

## Task Mapping

- R1 (Gate+Up Fusion) $\to$ T1, T3
- R2 (Q+K+V Fusion) $\to$ T2, T3
- R3 (Batched Pre-Hadamard) $\to$ T4
- R4 (Fused Post-Hadamard Epilogue) $\to$ T1, T2, T5
- R5 (Pipeline Integration in `gpu_decode.rs`) $\to$ T6
- R6 (End-to-End Proof & Acceptance) $\to$ T7
