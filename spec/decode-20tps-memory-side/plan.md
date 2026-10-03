# spec/decode-20tps-memory-side — plan

## Architecture and data flow (unchanged; this spec moves bytes, not structure)

Decode step = 60 layers × {q,k,v,o,gate,up,down} GEMV + attention + head, at
M=1. Weight bytes come from `Gxma-repacked` trellis through
`linear_gemv_ishizuki_k3_colmajor_2col` (`linear_repacked.metal:2072-2206`),
reached from `LinearKernels::encode_gemv_raw` (`linear.rs:2133-2157`).

Per lane per 16×16 tile (8 codes): 2–3 u32 ring-word loads, `extract_v_mcg_32`
(2 variable shifts + 1 OR + constant shifts/masks), 8 × `decode_state_mcg`
(mul, and, xor, fp16 add, f16→f32), 8 fp32 FMA, 1 x load.

## The budget, and where the GB/s must come from

```
production                    506–511 Gcodes/s   190–192 GB/s
+ remove weight-load cost     ~568 Gcodes/s     213 GB/s     (arm C, measured)
+ remove loop/address overhead ~634 Gcodes/s    237.6 GB/s   (arm G, measured)
20 tok/s needs                631 Gcodes/s      236.7 GB/s
copy ceiling                  —                 250.5 GB/s
```

So the route is: **claim the arm (C) 11% first** (memory-side, no instruction
change), then attack the remaining ~13% of loop/address overhead. Arm (G) bounds
what the second step can give.

## Sequencing

**Phase 1 — attribute, don't guess.** (R1, R2, R3)
1. Per-role bandwidth table on real weights, all five live shapes. The step
   average of 192 GB/s hides a distribution; the synthetic sweep suggests a
   2.3× spread, which would mean one or two roles carry the whole loss.
2. Latency vs pattern. The two need opposite fixes and nothing existing
   distinguishes them. Discriminator: hold the access pattern identical and vary
   only the number of independent loads in flight (arithmetic already done by arm
   (C)/(G) style ablations), or vary the grid with a fixed shape.

**Phase 2 — claim the memory-side term.** (R4)
Candidates, in order of expected value, none yet measured:
- **More loads in flight.** If Phase 1 says latency, the fix is unrolling the K
  loop to issue more independent loads before the first use. Note
  `dispatch-e4-0.9%` records an earlier `kloop_loads_in_flight` attempt at
  −0.55/+0.62%, so this may already be closed — re-check before spending GPU time.
- **The lane read amplification (2.667×).** A lane reads 2 u32 = 64 bits to
  recover 8 codes = 24 useful bits. At a 3-bit stride a 64-bit window fits **17**
  codes (16×3−3+16 = 61 ≤ 64), not 8. Serving 17 codes per lane per 2 words would
  halve the per-code read traffic and amortise the extraction setup over twice
  the codes. Requires re-laying out the lane→code map and the reduction, so it is
  a kernel rewrite, not a tweak. The tree's own `simd_shuffle` attempt at sharing
  the overlap measured *worse* (180.5 → 146.5 GB/s), which is evidence against
  sharing but not against a wider per-lane window — those are different changes.
- **The 10% shape/tensor spread (Q1).** Free if it is placement.

**Phase 3 — the residual ~13%.** Address arithmetic and loop overhead. Only
after Phase 2, and only with arm (G)'s caveat in mind: much of what arm (G) does
not pay for is addressing that a real kernel must pay.

## File/module touchpoints

| path | change |
|---|---|
| `crates/engine/tests/perf_probe_code_rate_p01.rs` | role sweep (read-only w.r.t. product code) |
| `crates/kernels/src/linear_repacked.metal` | the live kernel, if Phase 2 finds a win |
| `crates/kernels/src/linear.rs` | dispatch selection, only if a variant is added |
| `spec/decode-20tps-memory-side/findings/*` | one file per hypothesis, written before the run |

## Risks and fallbacks

- **Arm (G) may not transfer.** It is an upper bound by construction. Fallback
  target is arm (C)'s 213 GB/s = 17.7 tok/s, which is still +10% on today.
- **20 tok/s may not be reachable without bytes/token.** Then it becomes a
  quality decision for the user, and this spec says so rather than quietly
  trading quality (AGENTS.md §8).
- **Measurement noise.** 35% same-config spread is on record. Every claim needs
  interleaved A/B with a control arm; a single reading is not a result.

## Requirement → task map

| req | task |
|---|---|
| R1 | T1 |
| R2 | T1, T2 |
| R3 | T2 |
| R4 | T3, T4 |
| R5 | every task |
