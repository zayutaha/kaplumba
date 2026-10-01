# SPECIFICATION: 25 Tok/s Decode at Physical Memory Bandwidth Limit

## 1. Goal

Achieve pure autoregressive single-token decode throughput of **$\ge 25.0$ tok/s** ($\le 40.0$ ms/step) for Gemma 4 31B (`Gxma`, EXL3 3.0 bpw) on Apple M4 Pro (20 GPU cores, 273 GB/s peak unified memory bandwidth, 24 GiB RAM), operating at $\ge 91.5\%$ of theoretical physical DRAM bandwidth ceiling ($10.8\text{ GB active weights} \div 273\text{ GB/s} = 39.56\text{ ms/token}$). The implementation must maintain 100% bit-exact greedy generation against the reference model without quality degradation or speculative approximations.

---

## 2. Non-Goals & Restrictions

1. **Strictly Pure Single-Token Autoregressive Decode Only:**
   No speculative decoding, draft models, medusa/eagle heads, lookahead decoding, or multi-token speculation. Every token must execute through the full 60-layer forward network.
2. **100% Quality Invariant (No Quality Loss):**
   Zero degradation in output quality. Greedy generation must match the reference model output ("The capital of France is Paris." with identical token IDs).
3. **No KV Cache Quantization:**
   The KV cache remains FP16 (or BF16). No FP8/INT4/INT8 quantization of KV states.
4. **Preserve Source Checkpoint:**
   The original model weights in `~/.kaplumba/models/Gxma/Gxma` must remain completely untouched. Optimized repacked weights will be written to `~/.kaplumba/models/Gxma-repacked-v2` or generated in-place via clean offline scripts.
5. **No Python on the Product/Engine Hot Path:**
   Engine code is pure Rust + raw Metal shaders.
6. **Single-User Target & Memory Footprint:**
   Single-user chat target only. Resident GPU memory footprint must remain under the 16.5 GiB budget on the 24 GiB physical machine.
7. **GPU Access Discipline:**
   Strict adherence to `scripts/gpuq try -- <cmd>` before every model run. Never sit in blocking waits or run concurrent model instances.

---

## 3. Requirements

1. **R1: Sustained $\ge 250$ GB/s GEMV Bandwidth Across All Projections**
   - Every linear projection (`qkv_proj`, `o_proj`, `gate_up`, `down_proj`) must achieve $\ge 245\text{--}255$ GB/s of sustained DRAM read bandwidth on Apple M4 Pro (approaching the ~250 GB/s physical copy ceiling).
2. **R2: 128-Bit Aligned Vector Memory Transactions (`uint4` / 16-byte Loads)**
   - Repacked weight storage and GPU dequantization kernels must issue 128-bit aligned vector memory transactions (`uint4`) so that every DRAM request transfers full 32-byte cache line bursts without bank conflicts or lane serialization.
3. **R3: True Co-Packing and Concurrent Threadgroup Execution for Coupled Projections**
   - `gate_proj` and `up_proj` (both $5376 \times 21504$) must be computed within the same threadgroup so that the input vector $x$ is loaded once and arithmetic intensity is doubled.
   - `q_proj`, `k_proj`, and `v_proj` must share input vector streaming and unroll across threadgroups without redundant $x$ memory sweeps.
4. **R4: Multi-Queue Concurrent Overlap of Attention and GEMV**
   - The ~3.2 ms of Attention KV cache memory streaming must be scheduled concurrently across separate Metal command queues alongside GEMV dequantization, hiding Attention latency behind matrix multiplication.
5. **R5: Layer-Level Megakernel / Command Pipeline Streamlining**
   - Per-token dispatches must be reduced toward minimal boundaries per layer, keeping intermediate activations in on-chip SRAM/registers and eliminating ~5.0 ms of CPU-GPU submission bubbles and launch overheads.
6. **R6: Bit-Exact Verification and Sabotage Evidence**
   - All optimizations must be proven bit-exact against the reference (`live_kernel_bit_exact`, `fused_bit_exact`, `repacked_golden`).
   - Every newly implemented kernel path must have documented sabotage evidence confirming that intentional modification produces immediate, detectable failure.

---

## 4. Acceptance Criteria

1. **AC1: Sustained Decode Throughput $\ge 25.0$ Tok/s**
   - Steady-state single-token forward decode latency $\le 40.0$ ms/step on Apple M4 Pro, verified across reference prompts using `scripts/gpuq try -- ./target/release/kaplumba`.
2. **AC2: 100% Bit-Exact Greedy Text Generation**
   - Generates exact text: `"The capital of France is Paris."` with identical token IDs and 0 FP16 divergence on every layer.
3. **AC3: Zero Regressions in Test Suite**
   - `cargo test --test live_kernel_bit_exact`, `cargo test --test repacked_golden`, and `cargo test --test fused_bit_exact` pass with 0 failures.
4. **AC4: Validated Sabotage Tests**
   - Every kernel modification passes the sabotage requirement (§3.4 of AGENTS.md), documented in `implementation_notes.md`.
5. **AC5: Stable Memory Residency**
   - Physical memory footprint remains $\le 16.5$ GiB resident with 0 MiB swap usage on the 24 GiB machine.

---

## 5. Open Questions

*(None. All clarification questions resolved and confirmed by user on 2026-10-01.)*
