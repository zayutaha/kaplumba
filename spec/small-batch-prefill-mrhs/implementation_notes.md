# Implementation Notes: Small-Batch High-Bandwidth Prefill (Multi-RHS GEMV)

Append as work proceeds. Factual and dense.

## 0. Hardware & Environment Facts
- **Machine:** Apple M4 Pro (20 GPU cores, 24 GiB unified RAM, 273 GB/s peak bandwidth).
- **OS:** macOS 15+ (Darwin kernel 27.0).
- **Model:** Gemma 4 31B dense (`~/.kaplumba/models/Gxma-repacked`), EXL3 ~3.0 bpw.
- **Weights Residency:** 12.04 GiB wired via `LlamaMlock` + `MTLResidencySet`.
- **Target:** Time-To-First-Token $\le 200\text{ ms}$ on typical interactive chat inputs ($M \le 32$ delta tokens).

## 1. Measured Baseline & Root Causes
- Baseline TTFT measured at $M=14$ tokens: **1312–1315 ms** (`spec/chat-ttft-300ms/` Task 5).
- Breakdown from `KAPLUMBA_GPU_PROFILE=1`:
  - `down_proj`: 336.8 ms
  - `gate_proj`: 275.1 ms
  - `up_proj`: 273.4 ms
  - `o_proj`: 153.4 ms
  - `q_proj`: 141.9 ms
  - `k_proj`: 70.3 ms
  - `v_proj`: 61.5 ms
  - Sum of linear projections across 60 layers: **1312 ms**.
- Root cause: `linear_gemm_mma` uses a fixed $128 \times 64$ tile with threadgroup barriers and low occupancy (1.6 threadgroups/core at $M=14$), dropping memory bandwidth to 9.1 GB/s.
- Target bandwidth with Multi-RHS GEMV: ~185 GB/s (decode GEMV rate), projecting ~65–85 ms across all 60 layers.
