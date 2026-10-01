# TECHNICAL PLAN: 25 Tok/s Decode at Physical Memory Bandwidth Limit

## 1. Architecture & Data Flow

To reach $\ge 25.0$ tok/s ($\le 40.0$ ms/step) at 273 GB/s peak bandwidth:

```
[DRAM: 10.8 GB Active Weights + 0.8 GB KV Cache]
       │ (128-bit aligned uint4 vector loads, 250+ GB/s streaming)
       ▼
┌────────────────────────────────────────────────────────┐
│ DECODER LAYER MEGAKERNEL / CONCURRENT QUEUES           │
│                                                        │
│  Queue 0: GEMV Projections                             │
│   ├── QKV GEMV (Co-streamed, single input x load)      │
│   ├── Out-Proj (2-column tiled, 32-channel reduction)  │
│   ├── Gate-Up Fused2-Co (Co-computed, shared input x)  │
│   └── Down-Proj (2-column tiled, 32-channel reduction) │
│                                                        │
│  Queue 1: Concurrent Attention Stream                  │
│   └── SWA/Full KV cache streaming overlapped with      │
│       GEMV dequantization (zero added latency)         │
│                                                        │
│  On-Chip Data Flow (SRAM / Threadgroup Registers):     │
│   ├── Norms & RoPE fused directly at projection edges  │
│   └── Zero intermediate DRAM buffer roundtrips         │
└────────────────────────────────────────────────────────┘
       │
       ▼
[Logits & Fast Zero-Copy Argmax] -> Next Token
```

### Key Technical Pillars:
1. **True Co-Computed `gate_up` (`linear_gemv_ishizuki_k3_fused2_co`):**
   Currently, `gate_up` splits threadgroups between gate and up, loading $x$ twice ($28.9$ MB loads). Co-computing gate and up within each threadgroup reduces $x$ loads by 50% ($14.4$ MB eliminated per token), doubles arithmetic intensity per byte of $x$, and reduces `gate_up` from $25.8$ ms toward $\le 16.5$ ms.
2. **128-Bit Aligned Vector Memory Transactions (`uint4`):**
   Currently, weights are loaded via scalar 32-bit words `t0[j0], t0[j1]`. Realigning memory transactions into 128-bit bursts (`uint4`) eliminates DRAM controller bank penalties and allows streaming at $\ge 250$ GB/s.
3. **Concurrent Multi-Queue Overlap (Attention + GEMV):**
   Metal compute command encoders on separate queues allow the GPU to overlap the pure memory bandwidth of Attention KV cache reading with the ALU-bound dequantization of linear projections, absorbing the 3.2 ms attention latency.
4. **Command Buffer & Dispatch Streamlining:**
   Coalesce all layer operations into single-encoder command streams to eliminate host dispatch bubbles (~5 ms).

---

## 2. File & Module Touchpoints

| File Path | Current Behavior | Intended Change |
|---|---|---|
| `crates/kernels/src/linear_repacked.metal` | `fused2` concatenates threadgroup grids (`tgid >= tiles_a`); `contig4` loads scalar `uint`s. | Add `linear_gemv_ishizuki_k3_fused2_co` (true threadgroup co-computation of gate and up); implement 128-bit `uint4` memory loads. |
| `crates/kernels/src/linear.rs` | Dispatches `fused2` across `tiles_a + tiles_b` threadgroups; handles GEMV dispatches individually. | Wire `fused2_co` into `encode_gemv_fused2_mlp` using `tiles_a` (1344) threadgroups; add 128-bit aligned dispatch variants. |
| `crates/engine/src/model.rs` | Dispatches separate encoders and serializes attention and GEMV within each layer. | Implement concurrent queue submission for Attention and GEMV; eliminate redundant barriers. |
| `crates/engine/src/exl3.rs` | Repacks tensors to column-major order. | Add 128-bit vector alignment and co-packed tensor ordering support. |
| `crates/engine/tests/live_kernel_bit_exact.rs` | Tests single-column and 2-column kernels. | Add bit-exactness verification and sabotage tests for `fused2_co` and 128-bit kernels. |

---

## 3. Sequencing & Dependencies

- **Step 1 (T1):** Implement `linear_gemv_ishizuki_k3_fused2_co` (true gate+up threadgroup co-computation). Verify bit-exactness and sabotage.
- **Step 2 (T2):** Wire `fused2_co` into `encode_gemv_fused2_mlp` and measure decode step speedup (targeting 25.8 ms -> 16.5 ms).
- **Step 3 (T3):** Implement 128-bit aligned vector loads (`uint4`) for repacked GEMV to push memory bus efficiency from ~73% to $\ge 92\%$ (250+ GB/s).
- **Step 4 (T4):** Implement concurrent multi-queue execution for Attention and GEMV in `model.rs` to hide 3.2 ms of attention latency.
- **Step 5 (T5):** Full layer megakernel / command pipeline streamlining to eliminate remaining CPU-GPU bubbles (~5 ms).
- **Step 6 (T6):** End-to-end benchmark validation across 16 interleaved pairs, verifying $\ge 25.0$ tok/s and 100% bit-exact greedy generation.

---

## 4. Risks & Fallbacks

- **Risk 1: Register pressure in `fused2_co`:**
  - *Mitigation:* `fused2_co` only needs 4 accumulators (`acc_a0, acc_a1, acc_b0, acc_b1`), which easily fit in the M4 Pro register file (32 KB per SIMD group).
- **Risk 2: Multi-queue race condition:**
  - *Mitigation:* Attention only depends on Q, K, V activations and outputs to `scratch.attended`. Queue synchronization is enforced via Metal event primitives (`MTLEvent` / `MTLSharedEvent`) without CPU blocking.
- **Risk 3: Drift / divergence in greedy text:**
  - *Mitigation:* Continuous validation against golden baseline ("The capital of France is Paris.") after every single task.

---

## 5. Requirement to Task Mapping

| Requirement | Tasks |
|---|---|
| R1 (Sustained $\ge 250$ GB/s GEMV) | T1, T2, T3 |
| R2 (128-Bit Aligned Vector Memory Transactions) | T3 |
| R3 (True Co-Packed Coupled Projections) | T1, T2 |
| R4 (Concurrent Attention + GEMV Overlap) | T4 |
| R5 (Megakernel & Dispatch Streamlining) | T5 |
| R6 (Bit-Exact Verification & Sabotage) | T1, T2, T3, T4, T5, T6 |
