# Implementation Notes: Verify-Gated Speculative Decoding to ≥ 20.0 t/s

Running log. Append as work proceeds. See AGENTS.md §5.

## 2026-09-23 — Spec Initialized

- Baseline plain decode: 144.5 ms/token fwd-only (6.92 t/s) sits at the Apple M4 Pro int/fp32 ALU roof (~200G ops/token vs ~1.4–2.0T ops/s GPU ALU capacity).
- Mechanism for $\ge 20.0$ t/s: Verify-gated speculative decoding with shallow-layer target draft and row-batched target verification via `exl3_gemm_sg`.
- Memory budget: $\le 14.6$ GB resident. Using the target model's own resident weights for drafting adds 0 bytes of weight memory.

## 2026-09-23 — T1: Row-Batched Target Verify Latency Probe

- Measured with `bench_verify_batch_latency_slow`:
  - 1-row target forward (plain decode): 141.68 ms (7.06 t/s).
  - Initial `encode_span` for 2-row and 4-row was ~720 ms because `exl3_gemm_sg` has a fixed block size of 64 rows (`SG_BM = 64`), which computed 62 dummy padded rows for small batches.
  - Gated `exl3_gemm_sg` on `rows >= 32` so small row counts ($M \le 8$) take `exl3_gemm` (RGT=8) where decode ALU is spent once per weight tile and shared across $M$ rows with 0 dummy padding.
  - New measured latencies:
    - 2-row batched verify: 248.98 ms total (124.49 ms/token, 8.03 t/s, 1.14x speedup).
    - 4-row batched verify: 374.86 ms total (93.71 ms/token, 10.67 t/s, 1.51x speedup).
  - Confirmed decode ALU amortization without tensor core padding overhead.

## 2026-09-23 — T2: KV Cache Rewind/Truncate Support

- Added `KvCache::truncate(&mut self, target_len: usize) -> Result<(), GemmaError>`.
- Truncates sequence length to `target_len` without reallocations or memory leaks.
- Handles sliding window layers: accurately computes valid rows within `l.start .. target_len`, resets row pointers, and allows immediate continuation of `reserve_row` / `reserve_rows` at the rollback position.
- Unit test `truncate_rolls_back_speculative_tokens` passed; all 118 engine tests pass.

