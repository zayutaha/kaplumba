# Specification: 14+ tok/s Decode Throughput at 17.5k Context

## 1. Goal
Achieve sustained $\ge 14.0\text{ tok/s}$ pure autoregressive decode throughput (step delta $\le 71.4\text{ ms}$) in interactive multi-turn chat (`kaplumba --chat`) using model `~/.kaplumba/models/Gxma-repacked` at **17,500+ tokens context length**, without speculative decoding / drafter (`--no-drafter`).

---

## 2. Non-goals & Restrictions
- **No KV Cache Quantization:** KV cache remains full fp16 as per AGENTS.md §8 standing constraints.
- **No Quality Degradation:** Exact match against reference logits; greedy output must remain bit-exact identical.
- **No Multi-Tenant / Dynamic Batching:** Single-user chat only (AGENTS.md §8).
- **Zero Python on hot path:** Pure Rust runtime + raw Metal compute.
- **Memory Footprint Bound:** Peak memory footprint must stay under 14 GiB on 24 GiB unified memory machine.

---

## 3. Current Baseline & Hardware Context
- **Target Hardware:** Apple M4 Pro (20 GPU cores, 273 GB/s unified memory bandwidth, 24 GiB unified RAM).
- **Model:** Gemma 4 31B dense (60 layers: 50 sliding attention window 1024, 10 full attention global head_dim 512). EXL3 ~3.0 bpw repacked layout (~15.5 GB resident weights).
- **Current Shipped Performance:**
  - Short context (1 turn, 18 ctx): **14.9 tok/s** (67.0 ms/step GPU own time).
  - 17.5k context (Turn 1: 17,514 prefill + 60 gen, 17,574 ctx):
    - Turn 1 generation average: **9.34 tok/s** (slowed by token 2 post-prefill pipeline barrier).
    - Turn 2 generation average: **12.00 tok/s**.
    - Steady-state decode step delta: **77.4 ms** ($\approx 12.92\text{ tok/s}$).
  - **Deficit to reach 14.0 tok/s:** $\mathbf{6.0\text{ ms}}$ per step (from 77.4 ms down to $\le 71.4$ ms).

---

## 4. Empirical Timing Breakdown at 17.5k Context

Measured using `KAPLUMBA_GPU_PROFILE=1 KAPLUMBA_GPU_PROFILE_DECODE=1` across 60 layers:

| Component | Kernel Name | Count | Total Time (ms) | Mean per Layer ($\mu$s) | % of Step |
|---|---|---|---|---|---|
| **MLP Gate + Up** | `linear_gemv_ishizuki_k3_fused2_co` | 60 | **26.22 ms** | 437.0 $\mu$s | 33.9% |
| **MLP Down Proj** | `linear_gemv_ishizuki_k3_colmajor_2col_sk2` | 60 | **14.05 ms** | 234.1 $\mu$s | 18.1% |
| **QKV Proj (Sliding)** | `linear_gemv_ishizuki_k3_fused3_2col` | 50 | **10.39 ms** | 207.8 $\mu$s | 13.4% |
| **Attention (All 60L)** | `attention_decode_flash_dim_256/512` | 60 | **3.20 ms** | 53.3 $\mu$s | 4.1% |
| *(Full Attention 10L)* | `attention_decode_flash_dim_512` | 10 | *2.22 ms* | *222.0 $\mu$s* | *2.9%* |
| *(Sliding Attention 50L)*| `attention_decode_flash_dim_256` | 50 | *0.98 ms* | *19.6 $\mu$s* | *1.3%* |
| **Output Proj** | `linear_gemv_ishizuki_k3_colmajor_2col_sk2` | 60 | **6.84 ms** | 114.0 $\mu$s | 8.8% |
| **LM Head GEMV** | `linear_gemv_ishizuki_k<6>` | 1 | **4.05 ms** | 4,050.0 $\mu$s | 5.2% |
| **QK Proj (Full 10L)** | `linear_gemv_ishizuki_k3_fused2_2col` | 10 | **2.31 ms** | 231.2 $\mu$s | 3.0% |
| **Elementwise / Norms** | RMSNorm, rope, post-attn/pre-ffn fuses | 360 | **2.88 ms** | 48.0 $\mu$s | 3.7% |
| **CPU / GPU Queue Gap** | Command buffer flushes & driver overhead | - | **~7.6 ms** | - | 9.8% |
| **Total Step Wall-Span**| - | - | **~77.5 ms** | - | 100.0% |

---

## 5. Key Findings & Levers for Claude Review

### Finding 1: Attention is NOT the Dominant Bottleneck at 17.5k Context
Although context grew from 512 to 17,500 tokens (34x growth), attention only grew from **0.97 ms** to **3.20 ms** (+2.23 ms total).
- 50 sliding layers are strictly bounded by their 1024-token window and execute in **0.98 ms** total across all 50 layers.
- 10 full attention layers take **2.22 ms** across all 10 layers (at 64 splits = 256 workgroups).
- Therefore, optimizing attention alone cannot close the 6.0 ms deficit, because total attention time is only 3.2 ms.

### Finding 2: `gate_up_gelu` Dominates the Step (26.22 ms)
- In `crates/kernels/src/linear_repacked.metal:linear_gemv_ishizuki_k3_fused2_co`, 60 layers evaluate gate and up projections fused with GELU.
- **Completed Optimization (Commit `32830cb`):** Hoisted the 4 loads of Col B (`w0_b0`, `w1_b0`) before dequantizing Col A. This brought all 8 column words into flight concurrently.
- **Potential 4-col Fusion (`linear_gemv_ishizuki_k3_fused2_co4`):**
  - Evaluated via `KAPLUMBA_FUSED2_CO4=1`.
  - While it cuts workgroups from `tiles_a` to `tiles_a/2`, register pressure and threadgroup reduction increased `gate_up_gelu` from 26.2 ms to 29.8 ms (+3.6 ms regression).
  - Verdict: `fused2_co` (2-col) remains faster than `fused2_co4` on this architecture.

### Finding 3: Inter-Boundary Queue / CPU Submission Overhead (~7.6–10.5 ms)
- Under the default decode configuration:
  - `decode_max_dispatches_per_buffer() = 50`.
  - With ~9 dispatches per layer across 60 layers ($\approx 540$ dispatches per decode step), this creates **11 distinct command buffers** committed and submitted to Metal per token.
  - Each buffer commit incurs driver dispatch overhead and scheduling bubbles.
- Increasing the buffer limit to encode the full step (or 270 dispatches / 2 buffers) can collapse the inter-boundary gap from ~7.6 ms down to ~2.0 ms, saving **~5.5 ms directly**.

### Finding 4: Token 2 First-Decode Stall
- Token 1 (prefill argmax): 0.24 ms.
- Token 2 (first decode step after prefill): **1759 ms** (Turn 1) / **331 ms** (Turn 2).
- Token 3..60 (steady state): **77.4 ms**.
- Root cause: The transition between prefill command buffer completion and decode pipeline initialization pipeline flush. In multi-turn chat, turn average is dragged down from 12.9 tok/s to 9.34 tok/s due to Token 2.

---

## 6. Acceptance Criteria
1. Steady-state token-to-token delta $\le 71.4\text{ ms}$ ($\ge 14.0\text{ tok/s}$) measured in `kaplumba --chat` at 17,500+ context tokens.
2. Bit-exact match on golden tests (`live_kernel_bit_exact`, `repacked_golden`, `multi_turn_chat`).
3. Single model instance GPU queue safety (`scripts/gpuq`) preserved.
