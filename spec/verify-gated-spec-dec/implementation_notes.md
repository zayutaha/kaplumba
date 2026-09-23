# Implementation Notes: Verify-Gated Speculative Decoding to ≥ 20.0 t/s

Running log. Append as work proceeds. See AGENTS.md §5.

## 2026-09-23 — Spec Initialized

- Baseline plain decode: 144.5 ms/token fwd-only (6.92 t/s) sits at the Apple M4 Pro int/fp32 ALU roof (~200G ops/token vs ~1.4–2.0T ops/s GPU ALU capacity).
- Mechanism for $\ge 20.0$ t/s: Verify-gated speculative decoding with shallow-layer target draft and row-batched target verification via `exl3_gemm_sg`.
- Memory budget: $\le 14.6$ GB resident. Using the target model's own resident weights for drafting adds 0 bytes of weight memory.
