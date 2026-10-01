# TECHNICAL PLAN: 17-18 tok/s Decode

## 1. Architecture: What Changes

```
BEFORE (63.1 ms/step):
  Layer L: qkv(8.58ms) → attn(1.23ms) → [IDLE BUS] → transform(0.1ms) → o_proj(6.66ms@100GB/s)
           → finalize → post_ffn_norm → residual_add
  Layer L+1: input_norm → transform → qkv → ...
             ↑ 3 dispatches, 3 buffer round-trips

AFTER (~58 ms/step):
  Layer L: qkv → [attn + o_proj_prefetch concurrent] → o_proj(~4ms@180GB/s)
           → fused_finalize_and_next_input(1 dispatch)
  Layer L+1: [norm already done] → qkv → ...
```

Two structural changes:
1. o_proj weight prefetching overlapped with attention (R1)
2. Cross-layer norm fusion (R2)

## 2. File & Module Touchpoints

| File | Current | Intended Change |
|---|---|---|
| `crates/kernels/src/linear_repacked.metal` | `colmajor_2col_sk2` starts cold | Add `_warmstart` variant with atomic-flag spin-wait before first FMA |
| `crates/kernels/src/linear.rs` | `encode_gemv_raw_sk2` dispatches after attention | Add `encode_gemv_raw_sk2_warmstart` that dispatches concurrent with attention |
| `crates/kernels/src/linear.metal` | `finalize_and_post_ffn_norm_scale_sk2` handles one layer | Add `finalize_post_ffn_and_next_input_norm_transform_sk2` that fuses across layer boundary |
| `crates/engine/src/model.rs` | Sequential: attention → transform → o_proj → finalize → norm → residual → next_norm → transform | Concurrent: attention + o_proj_warmstart; fused finalize → next layer |
| `crates/engine/src/model.rs` (`ForwardScratch`) | No atomic flag | Add `attn_done: Buffer` (4 bytes, `device atomic_uint`) |
| `crates/kernels/src/encoder.rs` | Single command queue | (Fallback only) Add secondary `CommandQueue` + `MTLEvent` support |

## 3. The o_proj Warmstart Kernel (R1a)

```metal
kernel void linear_gemv_ishizuki_k3_colmajor_2col_sk2_warmstart(
    // ... same args as _sk2 ...
    device const atomic_uint& attn_done [[buffer(N)]],  // NEW
    // ...
) {
    // Phase 1: PREFETCH (runs concurrently with attention)
    // Begin streaming weight tiles into registers
    for (uint step = ...; step < K; step += SG * 2u) {
        // Load weight words (this warms the DRAM prefetcher)
        const uint w0_a0 = t0_a[j0]; // DRAM read ← this is the point
        const uint w0_a1 = t0_a[j1]; // DRAM read
        // ... accumulate normally BUT guard the input vector read:
        
        if (step == 0u) {
            // Wait for attention to complete before reading x[]
            while (atomic_load_explicit(&attn_done, memory_order_relaxed) == 0u) {}
            threadgroup_barrier(mem_flags::mem_device);
        }
        
        // Now x[] is valid, do normal FMA
        float v[8];
        extract_v_mcg_32(w0_a0, w0_a1, sh0, rsh0, v);
        // ... acc += v * x ...
    }
}
```

**Key insight:** The first K-loop iteration loads weight tiles from DRAM
(warming the prefetcher) but the `x[]` (input vector from attention) isn't
read until after the flag check. The weight loads are side-effect-free reads,
so they can proceed before attention completes.

**If Metal doesn't schedule them concurrently** (T1 verifies this), fall back
to R1b (dual-queue with MTLEvent), which explicitly schedules them on
separate queues.

## 4. Cross-Layer Fusion Kernel (R2)

```metal
kernel void finalize_post_ffn_and_next_input_norm_transform_sk2(
    // Layer L finalize inputs:
    device const float* raw          [[buffer(0)]],  // SK2 partials from down_proj
    device const half*  out_signs    [[buffer(1)]],  // down_proj output signs
    device const half*  post_ffn_w   [[buffer(2)]],  // post_ffn_norm weights
    device const half*  residual_in  [[buffer(3)]],  // hidden state (input to this layer)
    // Layer L+1 norm inputs:
    device const half*  next_input_w [[buffer(4)]],  // next layer's input_layernorm weights
    device const half*  next_signs   [[buffer(5)]],  // next layer's q_proj input signs
    // Outputs:
    device half*        hidden_out   [[buffer(6)]],  // updated residual for layer L+1
    device half*        transformed  [[buffer(7)]],  // input-transformed for L+1's qkv_proj
    // Params:
    constant uint&      N            [[buffer(8)]],
    constant float&     eps          [[buffer(9)]]
) {
    // Step 1: SK2 reduction + output signs (existing finalize logic)
    // Step 2: Post-FFN RMSNorm on the reduced output
    // Step 3: Residual add with hidden state
    // Step 4: Input layernorm on the updated hidden state (for next layer)
    // Step 5: Input transform (Hadamard sign flip for next layer's qkv_proj)
    // All in one threadgroup, one dispatch, no intermediate buffer
}
```

## 5. Sequencing & Dependencies

```
T1: Feasibility — does Metal schedule concurrent TGs in one CB? (no code change)
    ├── YES → T2a (warmstart kernel)
    └── NO  → T2b (dual-queue + MTLEvent)
T2: o_proj warmstart (R1a or R1b)
T3: Cross-layer norm fusion (R2) — independent of T1/T2
T4: A/B measurement + acceptance
```

T2 and T3 are independent. T4 depends on both.

## 6. Risks & Fallbacks

| # | Risk | Mitigation |
|---|---|---|
| 1 | Metal serializes TGs within a command buffer | Fall back to R1b (dual-queue). T1 verifies. |
| 2 | Spin-wait burns GPU cycles, throttles | Measure power/thermal. Worst case: remove spin-wait, dispatch o_proj normally. |
| 3 | MTLEvent overhead exceeds attention overlap | Measure. If overhead > 0.5 ms, abandon multi-queue. Record negative. |
| 4 | Cross-layer fusion breaks bit-exactness | The 256-lane strided reduction order is frozen. Use identical FP16 intermediate stores. |
| 5 | The ~3 ms gain doesn't materialise | Record the numbers. Each task commits independently, each can be reverted. |

## 7. Requirement → Task Mapping

| Requirement | Tasks |
|---|---|
| R1 (o_proj warm-start) | T1, T2 |
| R2 (cross-layer fusion) | T3 |
| R3 (bit-exact + sabotage) | T2, T3, T4 |
