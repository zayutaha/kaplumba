# Specification: Small-Batch High-Bandwidth Prefill (Multi-RHS GEMV)

## 1. Goal
In Gemma 4 31B, achieve sub-200ms Time-To-First-Token (TTFT) on typical interactive chat turns ($M \le 32$ delta tokens) by eliminating the 1.3-second small-batch bottleneck in [`linear_gemm_mma`](file:///Users/zayaantaha/coding/kaplumba/crates/kernels/src/linear.metal#L10188). Process small token batches via a high-bandwidth Multi-RHS GEMV kernel ([`linear_gemv_mrhs`](file:///Users/zayaantaha/coding/kaplumba/crates/kernels/src/linear.metal)) that streams model weights at full memory bandwidth (~185 GB/s) with zero threadgroup barriers in the inner loop, reducing TTFT on short prompts from ~1315 ms down to $\le 200\text{ ms}$ with zero quality loss.

---

## 2. Non-Goals & Restrictions
- **No KV cache quantization** (AGENTS.md §8 standing constraint).
- **No quality loss** (greedy-identical output; AC1 parity verification).
- **No regressions in large-batch prefill throughput** (prompts with $M \ge 128$ tokens must continue using the compute-optimized MMA tensor-core pipeline).
- **No Python in product code** (AGENTS.md §8 standing constraint; pure Rust + Metal).
- **No O(L²) scratch memory allocation** (fixed, context-independent scratch).

---

## 3. Requirements

### R1 — Multi-RHS Metal Kernel (`linear_gemv_mrhs`)
In [`crates/kernels/src/linear.metal`](file:///Users/zayaantaha/coding/kaplumba/crates/kernels/src/linear.metal), implement `linear_gemv_mrhs_exl3_k3_mcg`:
- Dispatches $M \in [1, 32]$ query vectors concurrently across $N=16$ output columns per threadgroup.
- Maintains unrolled register accumulators `float acc[32]` per thread.
- Fetches each 32-bit quantized weight word once into registers, dequantizes once, and accumulates across all $M$ query rows.
- Contains zero `threadgroup_barrier` calls in the inner $K$-loop, achieving contiguous DRAM streaming at ~185 GB/s.

### R2 — Kernel Host Wrapper (`encode_gemv_mrhs`)
In [`crates/kernels/src/linear.rs`](file:///Users/zayaantaha/coding/kaplumba/crates/kernels/src/linear.rs):
- Register the `linear_gemv_mrhs_exl3_k3_mcg` pipeline.
- Implement `encode_gemv_mrhs` with grid configuration `groups = (out_features / 16) * 2`, `threads = 256`.
- Support multi-row inputs and outputs of dimension `[M, in_features]` and `[M, out_features]`.

### R3 — Dynamic Shape Routing in Engine
In [`crates/engine/src/model.rs`](file:///Users/zayaantaha/coding/kaplumba/crates/engine/src/model.rs#L2542):
- When `tokens == 1`: dispatch `encode_gemv` (single-token decode).
- When `1 < tokens <= 32`: dispatch `encode_gemv_mrhs` (fast-path small-batch prefill).
- When `tokens > 32`: dispatch `encode_gemm` (128-query MMA tensor-core prefill).

### R4 — Multi-Turn `--chat` Delta Integration
In [`crates/engine/src/chat.rs`](file:///Users/zayaantaha/coding/kaplumba/crates/engine/src/chat.rs), interactive turns with delta tokens $\le 32$ must automatically route through the fast path while preserving standing RAM KV cache across turns without re-prefill.

### R5 — Latency and Overhead Bounds
Total TTFT for prompts with $M \le 32$ tokens must be $\le 200\text{ ms}$, with non-prefill host overhead strictly bounded by $\le 100\text{ ms}$.

---

## 4. Acceptance Criteria
1. **Mathematical Parity (AC1):** Output of `encode_gemv_mrhs` matches reference linear projection within fp16 tolerance.
2. **Sabotage Verification (AC2):** Sabotaging multi-RHS accumulation (e.g. dropping row $m > 0$) causes tests to fail with an explicit diagnostic message.
3. **Workspace Safety (AC3):** `cargo test --workspace --lib` and `multi_turn_chat.rs` pass with zero regressions.
4. **End-to-End Latency (AC4):** Measured TTFT on Gemma 4 31B for a 14-token turn drops from ~1315 ms to $\le 200\text{ ms}$.

---

## 5. Open Questions
*(None remaining; resolved during clarification interview.)*
