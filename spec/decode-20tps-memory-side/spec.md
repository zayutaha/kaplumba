# spec/decode-20tps-memory-side — get decode to 20 tok/s by closing the memory-side gap

## Goal

Raise the sustained weight-streaming bandwidth of the live 3 bpw decode GEMV from
its measured **174–192 GB/s** to **236.7 GB/s**, which is 20 tok/s at the
corrected 12.039 GB/step. The bytes/token figure is fixed (no precision change),
so this is entirely a question of moving the existing bytes faster.

## Why this spec exists at all

`spec/decode-ceiling` concluded 20 tok/s was **unreachable**, on this chain:

> 9.4 instructions/code caps this loop at ~590 Gcodes/s = 221 GB/s at 3 bpw, so
> 20 tok/s needs 108% of the machine's issue capability.

That chain is **refuted by its own pre-registered experiment**, which had never
run. See `findings/F1-arm-g-runs.md`. Measured, three runs, spread <0.6%:

| | Gcodes/s | GB/s at 3 bpw |
|---|---|---|
| arm (G) register-resident, zero memory | **633.4 / 637.2 / 634.0** | **237.6** |
| what 20 tok/s needs | 631.2 | 236.7 |
| production, same runs | 463.6–469.3, 511.1–512.4 | 174–192 |
| arm (C) no weight loads (L1-resident) | 567.9 | 213.0 |

The pre-registered falsification criterion was "G < 631 Gcodes/s ⇒ 20 tok/s is
impossible, unconditionally". **It did not fire.** Therefore the instruction count
is not the wall, and the barrier the last 30 agents accepted does not exist.

## Non-goals / restrictions (AGENTS.md §8, non-negotiable)

- **No quality loss.** No precision change, no narrower 16-bit window, no
  changed dequant. The 16-bit window at a 3-bit stride is the format.
- **No KV cache quantization.**
- Single-user chat; no batching for its own sake.
- No graph/tree prefill scratch; prefill stays online-softmax with a fixed
  working set.
- Do not touch total bytes/token. This spec moves the same 12.039 GB faster.
- `--temperature 0` must stay bit-identical; `--temperature > 0` must stay
  distribution-identical to the host sampler.

## Requirements

1. **R1** — Reproduce the production bandwidth on real weights, per tensor role,
   with the repaired probe. The step average (192 GB/s) hides the distribution.
2. **R2** — Decompose the gap (production → arm G) into named components with
   measured attribution, not arithmetic. Arm (C) says ~11% is memory-side; the
   remainder is instruction/loop overhead.
3. **R3** — Identify whether the memory-side term is **latency** (too few loads
   in flight) or **pattern** (inefficient DRAM access). These need opposite
   fixes, and no existing measurement distinguishes them.
4. **R4** — Land at least one change that raises the step-level bandwidth, with
   interleaved A/B, medians and spread, and a control arm.
5. **R5** — Every claim carries a sabotage-verified gate (AGENTS.md §3.4).

## Acceptance criteria

- Step-level decode bandwidth measured **≥ 236.7 GB/s** on `Gxma-repacked`, or
  a measured, reproduced attribution of every remaining GB/s to a named wall.
- `--temperature 0` byte-identical over ≥3 runs, generated text only.
- No regression in peak footprint; the 18 GB arena limit still holds.

## Open questions

- **Q1** — Is the two-`gate_proj` 10% spread (464 vs 511 Gcodes/s, identical
  shape) a memory-placement effect that applies to the whole model? If so it is
  worth ~10% for free. Unexplained.
- **Q2** — Arm (G) is an *upper* bound (its accumulate is 7 ops per 8 codes vs
  production's 8 FMAs, and it has no address arithmetic). How much of the 634 is
  reachable in a kernel that must address real memory? Arm (C)'s 568 is the
  better near-term target.
- **Q3** — The synthetic shape sweep shows a 2.3× spread in code rate by shape
  (241 → 547 Gcodes/s). If real shapes behave the same way, the step average is
  dominated by the worst shapes, not the best.

## Decision that is the user's, not mine

Nothing here needs one yet. Precision changes remain barred; if the memory-side
route closes at, say, 19 tok/s, the only route past it is bytes/token, which is
a quality trade and a user call. Not reached.
