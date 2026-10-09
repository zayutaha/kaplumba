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

## 2. T3: Sabotage verification results (crates/kernels/tests/linear_mrhs.rs)

Test `mrhs_parity_against_single_row_gemv` covers M in {1, 2, 4, 7, 14, 16} × perm_mode {0, 1}.
Four sabotages were run against the compiled pipelines; each failure mode observed:

1. **Row-major epilogue zeroed rows m > 0** (`linear.metal` MRHS): caught —
   `MRHS divergence on perm_mode=0, M=2, row=1: max_diff=30.875 exceeds 1e-2`.
2. **Column-major epilogue zeroed rows m > 0** (`linear_repacked.metal` MRHS): caught —
   `MRHS divergence on perm_mode=1, M=2, row=1: max_diff=38.625 exceeds 1e-2`.
3. **Off-by-one overrun** (`m <= valid_m` in colmajor epilogue, writes one row past the
   last valid query row): initially **NOT caught**. The original test only checked a canary
   in the *finalized* `output` buffer, but the MRHS kernel writes the intermediate `raw`
   buffer — `linear_finalize_rows` never reads row `queries`, so the output canary is
   structurally blind to a raw overrun. The test already seeded a `0xDEAD_BEEF` canary in
   `raw` at row `queries` but never asserted on it. **This is a false-coverage claim in
   waiting** (same class as `spec/engine-performance` Q11). Fixed by asserting the raw
   canary after the run; the same sabotage now fails with
   `MRHS canary overrun in raw buffer at offset 0 for queries=1`.
   Lesson: a canary must live in the buffer the kernel under test actually writes.

Environment note: `LinearKernels::new()` fails if the Metal source fails to compile. An
earlier revision of the test silently `return`ed on that error, so the whole parity test
had been skipping on a box where the MRHS kernels did not compile (threadgroup memory is
not allowed inside a `static inline` template in MSL — it must be passed as a parameter
from the kernel entry point). The test now panics with `LinearKernels::new failed: {e}`
instead of skipping, which is what surfaced the compile error. Never `return`-skip a GPU
parity test.
