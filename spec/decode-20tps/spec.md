# SPECIFICATION: 20 tok/s Decode — Route Analysis & Clarification

**Status: AWAITING CLARIFICATION — see decode-20tps-analysis.md artifact**

## 1. Goal

Raise sustained single-token decode throughput from the measured **15.85 tok/s
(63.1 ms/step)** toward **20.0 tok/s** for Gemma 4 31B (`Gxma`, EXL3 3.0 bpw)
on Apple M4 Pro (20 GPU cores, 273 GB/s peak, 24 GiB RAM).

## 2. The Ceiling Problem

**20 tok/s is physically unreachable at 3 bpw pure autoregressive decode.**

- 11.834 GB of weights must stream per generated token
- Best measured 3 bpw GEMV bandwidth: 203 GB/s
- Ceiling with zero overhead: 11.834 / 0.203 = 58.3 ms = **17.2 tok/s**
- 20 tok/s requires 237 GB/s sustained (87% of 273 GB/s wire peak)
- 3 bpw amplification factor 2.667 is structurally unfixable (37-bit decode window)

**Route menu:**

| Route | Max potential | Status |
|---|---|---|
| A: ICB dispatch elimination | +0-1 tok/s → ~16.5 tok/s | Unverified, may be moot post-T5.1 |
| B: Remaining fusions + overlap | +0.5-2 tok/s → ~17.5 tok/s | 4 tasks from decode-3bpw-amplification |
| C: Verify-gated speculative decoding | **+5-20 tok/s → 20-35 tok/s** | Not started, currently barred by spec |

## 3. Open Questions (blocking)

- Q1: Accept ~17 tok/s, or lift spec-decode ban for 20+?
- Q2: Supersede the three existing specs?
- Q3: Draft model constraints (if spec-decode)?
- Q4: macOS / Metal version?
- Q5: Priority order?

## 4. Non-Goals & Restrictions

*(Inherited from standing constraints in AGENTS.md §8 until amended)*

1. No quality loss — greedy-identical output
2. No KV cache quantization
3. No Python in engine code
4. Single-user target
5. Resident ≤ 16.5 GiB, 0 swap
6. Original checkpoint untouched

## 5. Requirements

*To be populated after clarification.*

## 6. Acceptance Criteria

*To be populated after clarification.*
