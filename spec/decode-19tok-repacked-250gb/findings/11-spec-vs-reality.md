# Finding 11 — Adversarial read of the spec vs the measured record

Subagent: `ses_f0cecf311ffd8Zqu0GW0ISaWHQ`. Commits `1246690`, `9cd57a7`.

## Headline

The withdrawn-claims table had been **rolled back**. Commit `9c85e9a` ("Revert
T14 — 32 codes per lane") reverted `13fb9fc, 7d285fc, 4e4480a` wholesale, and
`4e4480a` is where the measured negative results were written. The revert's
stated intent was to remove *probe code and AC7–AC10*; it also destroyed eight
withdrawn-claim rows, §6.3's correction, all of §6.6/§6.7, and four of §7's
questions. **A measurement does not stop being true because the kernel that
produced it was deleted.**

## Findings ranked by damage to a future agent

1. **`plan.md` §1.2/§1.3 — the withdrawn thesis, live as design.** Asserts
   128-byte DRAM bursts, 1.333x amplification, and a `_vec128` kernel "to ensure
   1000+ threadgroups remain in flight". `spec.md` §5 withdraws the mechanism; the
   header cancels `_vec128`. **An agent starting at `plan.md` builds the wrong
   kernel.** Also names `encode_gemv_vec128` and `repacked_golden.rs`'s
   bit-exactness as live — the latter's coverage claim is *void* (it asserted
   kernels the engine never dispatched). RESTORED.
2. **`implementation_notes.md:98-99` — "52.5 ms (19.0 tok/s)" and "39.5 ms
   (25.3 tok/s)".** Three independent errors in two lines. `9.88 GB` is wrong
   (**12.039 GB**). Corrected, the arithmetic still fails: 12.039/188 = **64.0
   ms**, not 52.5; 12.039/250 = **48.1 ms**, not 39.5. And 25.3 tok/s needs
   **382 GB/s = 1.5x a memcpy**. The 188 GB/s denominator was itself a fit to land
   on 92 % of 273.
3. **§19.4 fusion roadmap — every estimate 5–10x high.** `norm_rope_qkv` est.
   3.5–4.5 ms, **saved 0.84 ms**. Items 2/3/4 at 2.0/2.0/1.5 ms are really
   **0.21/0.42/0.21 ms**. The estimates assumed a dispatch's *own kernel time* —
   **falsified**: `dpk_own_ms_per_token` 71.649 -> 71.644 across the change. **A
   dispatch buys latency, not work.**
4. **Three live code comments citing "~42 command buffers/step"**
   (`decode_profile.rs:19`, `generate.rs:893`, `encoder.rs:869`). The 42 is
   exactly `2100/50` — the test's own constant over the default budget, cited as
   a measurement, i.e. **circular**. Measured 1407 dispatches/token (1167 after
   T7.6.10c), below `max_dispatches(2500)`, so `tick()` never rolls: **2 buffers,
   one empty.**
5. **The "~9 ms host gap" replacement is itself wrong.** The 13 ms was an
   arithmetic error (step-mode `own` is documented as *not* a decode cost). But
   the "~9 ms" is **cross-instrument** — kernel-instrument `own` against
   uninstrumented `wall` — and internally inconsistent (9.16 tok/s x 83.5 ms =
   109 ms, not 83.5). Current build: `dpk_own` 71.644 vs uninstrumented 74.01 →
   **2.37 ms**, not 9. Flagged, not deleted.
6. **`T8` acceptance still read ">= 19.0 tok/s (<= 52.6 ms/step)".** Corrected to
   15.0, and the method raised from ">= 5 interleaved runs" to **16 paired** —
   16 *unpaired* samples spread 4–12 % and cannot resolve the 1 % effect
   T7.6.10c measured at p=0.0005.
7. **`T7.2`/`T7.3`/`T7.5` marked done with results the header calls cancelled.**
   Their numbers (split-K +31 %, u4 +41 %, simd_shuffle +18 %) are the
   **cached-working-set artifact that already produced T12's withdrawn +24 %**.
   Properly A/B'd: `_u4` 180.5 -> **173.7**; split-K is 3–5 ULP off (fails
   no-quality-loss) and slower in-engine; `simd_shuffle` 180.5 -> **146.5**.
8. **`T7.4` — "theoretical floor is 52.5 ms (19.0 tok/s)".** Same 9.88 GB error.
   Corrected.
9. **`T7.6.8`/`T7.6.9` tok/s deltas (12.39->14.04, 12.35->14.02) are unpaired
   single readings.** Box drifts 4–12 % across unpaired samples. The solid
   artefact behind T7.6.9 is the isolated A/B (contig4 188.2 vs sk4 168.1).
10. **notes T1 conclusion — "239–244 GB/s ceiling was an artifact of 4-byte
    scalar loads".** Unsupported; load width is a **closed** axis (8/16/32 B all
    ~227–238). This was the premise the `_vec128`/`_u4` work rested on.
11. **notes §6 (the burst discovery) had no in-place marker.** Struck with a
    note: tile math correct, conclusion not, `_u4` built and measured as a loss.
12. **Bit-plane repacker — confirmed withdrawn.** `exl3.rs:664-681` keeps
    `s & 0x7`; `decode_state_mcg` consumes a 16-bit window of which **13 bits
    survive**. The `imul` mixes all 16 *before* masking, so 3 bits of input yields
    **different dequant**, not imprecise dequant. Dead code, zero callers, must
    not be revived.
13. **D=32 / amplification law — confirmed correctly falsified.** 0.288–0.918x
    on all 11 shapes, cold, 9 rounds. The 22-request arm was **slower on all 11**
    at identical bytes. `D=8`->187.7, `D=32`->148.7, `lm_head`->250.5: three
    points, no law fits.
14. **2.0 bpw->18.1 and oMLX 2.25->26 — confirmed void, were never in this
    spec.** They live in the *predecessors* (`decode-bandwidth-240gb/spec.md:274`,
    `tasks.md:119-120`). `spec.md` §8 called those "baselines of record" while
    §5 withdraws their numbers — now flagged at §8.
15. **MLX comparison row — not present in this spec.** No MLX bandwidth
    comparison exists anywhere under `spec/decode-19tok-repacked-250gb/`. Nothing
    to correct; the row was either never written or lives in
    `spec/gemma4-metal-engine/`.

## The target — verified, and worse than stated

19 tok/s = 52.632 ms. Weights = 12.039 GB. At the 250.5 GB/s ceiling the GEMV
alone is **48.06 ms**, leaving **4.57 ms**. Non-GEMV kernels alone measure
**7.0 ms** (71.644 − 64.61) to **10.1 ms**:

| non-GEMV | floor | best case | rate 19 demands | vs ceiling |
|---:|---:|---:|---:|---:|
| 7.03 ms | 55.10 | 18.15 | 264.0 GB/s | **105.4 %** |
| 9.53 ms | 57.59 | 17.36 | 279.0 GB/s | **111.4 %** |
| 10.08 ms | 58.14 | 17.20 | 282.9 GB/s | **112.9 %** |

**19 tok/s demands a memcpy-beating GEMV. Even a perfect GEMV caps at
17.2–18.2.** The conclusion is arithmetic, not inference.

**And 15 is not reachable by the plan either.** Remaining fusion = 240 dispatches
x 3.5 us = **0.84 ms -> 13.67 tok/s**. 15.0 needs 7.35 ms. The gate is a GEMV
target; the GEMV has four measured failures (`simd_shuffle`, `_u4`, ring
staging, cheaper-dequant-*slower*) plus `D=32` on top, and no surviving
hypothesis. **What is untested is latency/occupancy.**

## ASK — the target, left to the owner

`R6`/`AC5` still read **15.0**; an explicit `USER DECISION PENDING` note was added
rather than moving the number. Options were (a) keep 15.0 as a *stretch* and
record 13.67 as the commitment, (b) lower the gate to ~13.7, (c) keep 15.0 and
treat the 3 bpw GEMV as unsolved. The agent's recommendation is (a).

**RESOLVED (main session, no user question asked — the standing instruction is
to ask nothing): (a).** 15.0 is retained as a labelled stretch, 13.67 as the
commitment. Rationale: a gate the plan provably cannot pass is worse than no
gate, and the user's stated target is "as close to 19 as possible, even 16 or
15" — 15.0 as a stretch with 13.67 committed is the honest encoding of that.

## Noted, not edited (another agent owns the file)

`crates/engine/src/weights.rs:530-533` claims `lm_head.trellis` offload measured
**12.99 -> 14.14 GiB, "actively harmful. Left resident."** Contradicted twice:
the code **three lines below** *maps* it (`offload_head_trellis = offload_head &&
...`, and `offload_head = !force_resident`, so it is on by default), and
`spec/resident-memory-12gib/implementation_notes.md:89` records the real result
as **−1.00 GiB** (11.51 GiB both-offloaded vs 12.45 embedding-only). Worse, the
12.99 GiB baseline is from a run that **failed early** (`lm_head.suh has 0
values`) and never allocated KV — **a number from a broken run.**

## Tree note

A concurrent agent is mid-flight in `crates/kernels/src/{attention,linear,
elements}.{rs,metal}`. Its repo-wide commit swept this agent's in-progress
`spec.md` into `1246690`; the remainder was committed scoped to its own files.
`cargo test -p kaplumba-engine --lib` fails on *that* agent's in-flight
`pipeline_audit` symbol, not on these changes — which are comment-only, verified
by filtering the diff for non-comment lines.
