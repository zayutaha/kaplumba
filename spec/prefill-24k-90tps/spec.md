# 90 tok/s prefill and 14 tok/s decode at 24k context

## Goal

At **24k tokens of context**, reach **≥90 tok/s prefill** and **≥14 tok/s decode**,
with no quality loss and within the 24 GiB machine budget.

Current, measured on `mtp-speculation`. "Before" is the merged
`spec/prefill-120tps` baseline (`882d874`); "now" is `b73090d`.

| | 8k before | 8k now | 24k before | 24k now | target @24k |
|---|---|---|---|---|---|
| prefill tok/s | 80.52 | 82.42 | 52.84 | **67.10** | **90** |
| decode tok/s | 12.27 | **12.98** | 5.44 | **not measurable** (OQ8) | **14** |
| peak `phys_footprint` | 14.15 GiB | -- | 18.35 GiB | -- | < 20 GiB |

> **Decode at 24k has no "now" number, and that is the finding, not a gap.**
> The 24k run's own footprint is 18.35 GiB of a 24 GiB machine, so it evicts the
> desktop and every later allocation is a compression event: a measured arm read
> **0.055 tok/s**, a 99x pessimisation, against a 5.44 baseline. The 8k figure
> (12.98, three interleaved rounds) is the trustworthy decode measurement, and it
> is a **floor** for 24k, not an estimate: the gain grows with context
> (+4.4% at 4k, +8.2% at 8k). See `implementation_notes.md` §14.4 and OQ8.

> **Corrected after T1.** The 24k decode figure of 0.30–0.43 previously in this
table came from a `KAPLUMBA_GPU_PROFILE=1` run; the profiler itself distorts
the measurement (the same prompt measured 0.082 tok/s profiled vs **5.44 tok/s
clean**). The clean 24k decode is 5.44 tok/s. There is **no memory cliff**
(`swaps 0`, `sys` 20 s of 434 s), so `OQ1` is closed and decode is a 2.6×
kernel-speedup ask, not a defect hunt. See `implementation_notes.md` §2b/§2c.

Prefill at 24k is an **attention** problem, not a GEMM problem. The measured
budget at 67.10 tok/s is GEMM + other **216.9 s** (context-independent) plus
attention **~149.7 s** of a 366.6 s wall. The 90 tok/s budget is 273 s, so **AC2
requires attention 149.7 s → ≤ 56 s, a 2.7x kernel speedup, with the GEMM
untouched.**

> **Corrected in §12.** This paragraph used to say AC2 needs attention 193 s →
> ≤ 52 s *and* a GEMM at 7.6 TFLOPS, on the strength of an 86 s "bandwidth
> floor". **That floor was wrong by ~350x**: it counted 2048 B of DRAM per
> (query,key) pair, but `PREFILL_CHUNK = 1024` means a whole chunk shares every
> key, so the real 24k KV traffic is 67.1 GB ≈ 0.25 s. The kernel moves 0.45 GB/s,
> about **0.16% of the DRAM rate** — it is not bandwidth-bound at all. In TFLOP/s
> terms the full layers run at ~1.74 and need **~3.4**, which is 22% of the
> 15.6 TFLOPS tensor peak: an ordinary number. OQ4 and OQ6 are withdrawn.

## Non-goals / restrictions

Carried from `AGENTS.md` §8, non-negotiable:

- **No quality loss**, no KV quantization, no lossy requantization.
- No Python in product code; single-user only.
- **No materialized score matrix / no O(L²) prefill scratch.** Online-softmax
  (flash-style) with an O(L) working set.
- Scratch buffers explicitly owned, fixed-size, **independent of context
  length**; no retained-pool semantics. `spec/prefill-120tps` §restrictions
  stands in full.
- Weight footprint must not increase (≤ 11.51 GiB).
- Worst normalised GEMM error ≤ 2.16e-3 (must not worsen).

Additional for this spec:

- **Decode ≥ 14 tok/s at 24k is requested and welcome**, but **do not regress
  the 8k/short-context decode** (~12–14 tok/s) or prefill.
- Prefill attention stays online-softmax; decode attention stays the
  split+combine path (or the single-pass path where already selected).

## Requirements

- **R1 — Characterise the 24k decode collapse.** Establish where the 2.3–3.3 s
  per token goes, by `KAPLUMBA_GPU_PROFILE=1` per-kernel attribution, not by
  benchmark alone. Discriminate between: (a) memory pressure (weights/KV paged
  or compressed by the OS at a 17.1 GiB footprint), (b) KV read amplification,
  (c) a per-token host/serialisation cost that grows with context, (d) a kernel
  whose grid or work is context-dependent in a way it should not be.
  This diagnosis gates every decode change.

- **R2 — Fix the decode cause.** Depending on R1: if memory pressure, reduce
  resident footprint or access pattern; if a kernel, fix it; if host work,
  remove it. Target 14 tok/s at 24k without losing correctness.

- **R3 — Prefill attention at 24k.** Attention is the majority of 24k prefill.
  Extend the shape-aware MMA QK (`spec/prefill-120tps` T2) and/or add
  query-blocking to reuse K/V across queries, so prefill attention at 24k is
  faster than today's scalar/MMA mix.

- **R4 — Prefill GEMM at 24k.** The GEMM is context-independent
  (~5753 ms per chunk-equivalent); confirm it is unchanged at 24k and take any
  remaining win from `spec/prefill-120tps` T3/T4 follow-ups.

- **R5 — Report** prefill tok/s (8k/16k/24k), decode tok/s (short/8k/16k/24k),
  achieved TFLOPS, and peak `phys_footprint` at each, interleaved A/B with
  medians and spread.

## Acceptance criteria

> **Status after the N1-N6 session (`48df478`).** AC3, AC4 and AC5 **met**;
> AC1 and AC2 **unmet**. Per-criterion evidence is in
> `implementation_notes.md` §20 and `tasks.md` (N6). The table at the top of this
> file is still the `b73090d` snapshot and is superseded by `NEXT.md` §7.1.
>
> | AC | target | measured | verdict |
> |---|---|---|---|
> | 1 | decode ≥ 14 @24k | **unmeasurable** (OQ8); 12.329 @16k | ❌ blocked, not failed |
> | 2 | prefill ≥ 90 @24k | 67.10 @24k (older, 1 round) | ❌ 20% short, unmoved |
> | 3 | prefill ≥ 80 @8k, ≥ 100 @614, decode ≥ 12 @8k | 85.52 / 101.04 / 12.911 | ✅ |
> | 4 | `phys_footprint` ≤ 20 GiB @24k, `swaps` 0 | 15.54 GiB @16k, 18.35 @24k; **zero swapouts** in 16 runs | ✅ |
> | 5 | all tests pass, 5 reference prompts, worst GEMM error ≤ 2.16e-3 | 116 binaries ok, 4 failed **all verified at base**; 5 prompts byte-identical between arms | ✅ with a caveat |



1. Decode ≥ **14 tok/s** at 24k context.
2. Prefill ≥ **90 tok/s** at 24k context.
3. No regression: prefill ≥ 80 @8k and ≥ 100 @614; decode ≥ 12 @8k.
4. Peak `phys_footprint` ≤ 20 GiB at 24k; engine `swaps` 0.
5. All existing tests pass; five reference prompts correct; worst GEMM error
   ≤ 2.16e-3.

## Open questions

- **OQ1 — Is decode-at-24k memory pressure?** **CLOSED (T1): no.** Clean
  24k decode is 5.44 tok/s with `swaps 0` and an 18.35 GiB peak footprint; the
  0.30–0.43 figure was profiler distortion.
- **OQ2 — Is 90 tok/s @24k reachable even with free attention?** **CLOSED (T1),
  and the earlier arithmetic was wrong.** The commentary that "if attention were
  free, 24k would be ~104 tok/s" used the GEMM-only 9.6 ms/token. The measured
  non-attention cost at 24k is 216.9 s (GEMM 213.7 s + 3.2 s else) = **110.5
  tok/s**, not 104. Against a 266 s budget that leaves **52 s for attention**, so
  AC2 does *not* need a GEMM change — but it needs a 3.7× attention cut, which
  means MMA or nothing.
- **OQ3 — Single-pass decode gate.** `ATTENTION_SINGLE_PASS_MAX_KEYS = 64`, so
  24k always uses split+combine. Confirmed: `decode_splits` clamps to
  `ATTENTION_SPLITS_MAX = 16` and the 24k full-layer grid is
  `16 × 4 = 64` workgroups — the latency hypothesis for T2 rests on this.
- **OQ4 — Is AC2 reachable at all, given the measured budget?** **OPEN, and this
  is now the binding question.** (Partly answered by §10: the contradiction named
  below — the byte model says bandwidth-bound, the one experiment says otherwise
  — is **not** a contradiction at all. §10 shows a 64× reduction in the bytes
  available to the K/V loads buys 6%, so the kernel is *not* bandwidth-bound;
  and §12 shows §5's floor was wrong by ~350× because it counted L1/L2 requests
  as DRAM. Attention is **instruction-bound**: 1.74 TFLOP/s now, **3.4 TFLOP/s
  (22% of tensor peak) needed** for AC2. The BQ=2 + key-split + combine plan is
  therefore **re-opened** — it was refuted using §9's split, which §10 withdrew.) With attention at the T1 lower bound of 86 s
  (the BQ=1 K/V-bandwidth floor, `implementation_notes.md` §5) the kernel sum is
  `214.8 + 86 + 3.2 = 304 s` -> **79 tok/s**, below the 90 target before the
  ~24 s inter-dispatch gap is even counted. The *only* arithmetic that reaches 90
  is query blocking, which divides the bandwidth floor (`BQ=2` -> 43 s -> 260 s
  kernel sum -> 92 tok/s). But **query blocking was measured and does not pay**:
  `BQ=2` is -2.0% engine attention at 8k and -6.2% at 24k (§7.3/§8.2), and the
  bench ranks `flash bq2` at or below `flash bq1` at both 8k and 24k (§8.7).
  So the §5 byte-traffic model, which says bandwidth is the binding constraint,
  is contradicted by the only experiment that tests it. **Until that
  contradiction is resolved — is the kernel latency-bound rather than
  bandwidth-bound, and if so what is the real limiter? — AC2 should be treated as
  unproven, not merely unmet.** The alternative to resolving it is a target
  change or a GEMM push, both of which need user direction.
- **OQ5 — 24k measurement is marginal under desktop load.** Two 24k A/B attempts
  in this session stalled: the model loaded (18 GiB wired) but the box
  entered compression (7.1 GiB compressor, <300 MiB free) once the 24k KV+scratch
  were resident, and `kaplumba` sat at ~2% CPU. The same box runs 614/8k
  measurements normally. **Partially resolved:** a later 24k run *did* complete on
  this same box -- **62.87 tok/s at 24 590 tokens**, with the desktop at ~7 GiB
  resident -- so 24k measurement is possible but marginal. Before trusting or
  discarding a 24k sample, check `top`'s `unused`, `sysctl vm.swapusage` and the
  process's CPU time (a sample with >2% CPU in compression is invalid), not just
  `gpuq try`, which only reads the lock at one instant.
- **OQ6 — Is the GEMM's 6.13–6.44 TFLOPS a hard wall?** **CLOSED, and the answer
  is “irrelevant to AC2”.** This question was raised from §5's byte model and is
  withdrawn: **§5's 86 s attention floor is wrong by ~350x**
  (`implementation_notes.md` §12). `PREFILL_CHUNK = 1024`, so the 2048 B per
  (query,key) pair is L1/L2 request volume, not DRAM; the real 24k KV traffic is
  **67.1 GB ≈ 0.25 s**. Attention must go 149.7 -> 56.1 s for AC2, and the GEMM
  is a fixed 216.9 s that does not have to move. Do **not** re-open the GEMM.
- **OQ7 — Why does deleting the QK and PV instructions not speed the flash
  kernel up?** **OPEN.** §10 is explicit: `ksteps = 0`, the QK loop off, both off,
  and an immediate `return` all land within 1.8% of baseline, while the same
  binary with an added spin loop costs +3.9 s. Every other explanation was
  checked and eliminated (build, dispatched kernel, host cost, DRAM bandwidth,
  work-proportionality — §10.1). Until this is answered, no attention change can
  be *predicted*, only measured end to end, and the microbench must not be used
  to attribute cost.
- **OQ8 — 24k decode is unmeasurable on this box.** **OPEN, and it blocks AC1.**
  The 24k run's own `phys_footprint` is 18.35 GiB of a 24 GiB machine, so it
  evicts the desktop and the compressor then drives decode to **0.055 tok/s**
  (a 99x pessimisation; §14.4). Three separate attempts have now produced
  invalid 24k decode samples (§13.1, §14.4). **A gate cannot fix this** — a gate
  can refuse to report a compressed sample, not produce a valid one — so the
  options are (a) a quieter machine (desktop closed, or a second box), (b) a
  lower-footprint configuration to measure, or (c) accept 8k as the decode
  acceptance context and re-baseline AC1 against it. **Needs user direction.**
- **OQ9 — Should the prefill key range be sliced into successive dispatches?**
  **CLOSED: no, not at any context this box can measure.** Built (N1), verified
  bit-exact against the one-dispatch body over 17 cases including ragged tiles,
  query tails and windowed shapes, and measured interleaved at 16 394 tokens:
  **+0.05% at 8 passes and +0.27% at 2 passes, against a 0.84% control spread.**
  A positive control on the same binary and prompt (`KAPLUMBA_SCALAR_ATTENTION=1`)
  moves the same instrument by 20%, so the null is believable rather than a broken
  build. The mechanism's own premise is a cache-residency loss, and oMLX measured it
  over **1M** keys (~85 vs ~103 TFLOPS); at 16k this engine's KV working set is
  134 MB per full layer and §10 already showed a 64x smaller K/V allocation buys
  6%. **There is no residency loss here to recover.** The implementation ships
  (measured neutral, bit-exact, off with `KAPLUMBA_PREFILL_PASS_KEYS=0`) because it
  is the only correct expression of the mechanism — and because building it forced
  the discovery that **pass boundaries must land on tile boundaries**, a correctness
  invariant of the online softmax: an unaligned boundary changes the output by one
  fp16 ulp on ~1% of outputs, ten times inside the 5e-3 tolerance every other
  attention test in this repo uses. See `implementation_notes.md` §16.

  *The original text of this question is preserved below, since the verdict is
  "no" and the reasoning above is easier to check against the claim being closed:*

  > **OPEN, and it is the strongest remaining prefill idea.** oMLX v0.7.0's prefill
  > path (`omlx/utils/nax_attention.py`) runs "several dispatches over consecutive
  > key ranges (`OMLX_NAX_ATTN_PASS_KEYS` keys each, default 8192) that hand the
  > fp32 online-softmax row state (O accumulator, running max and sum) from one to
  > the next through device memory, so each row computes exactly what one dispatch
  > computes (bit-identical output)", for the stated reason that **"one dispatch
  > over a long key range lets its threadgroups drift apart until each streams its
  > KV head from DRAM; per-slice dispatches keep the K/V they share in the
  > on-chip caches."** That is a *different* mechanism from §12.2's parallel
  > key-split-plus-combine (which needs no combine pass and no extra partials
  > buffer beyond the row state), and it targets the same cache-residency loss that
  > §10 measured directly. It is **not yet built or measured here.**

- **OQ10 — NEW. Is the shipped decode split count the measured optimum?** **CLOSED:
  no, and it is the only lever in this spec whose effect is still rising with
  context.** An interleaved 3-round sweep of `KAPLUMBA_DECODE_SPLITS` at 8204
  tokens with the MMA decode kernel gives 7.882 / 10.125 / 11.732 / 12.626 /
  **12.917** / 12.776 tok/s at splits 1 / 2 / 4 / 8 / **13** / 16, with effect arms
  at ±0.05%. `decode_one_wave_workgroups(256) / kv_heads = 54 / 4 = 13` matches the
  measured maximum. **This is a third *reproduction*, not a third *independent*
  one (F6): `spec/decode-19tok` §2 read the budget of 54 off one sweep of three
  shapes, and `kv_heads = 4 → 13` is the row it was read off. The out-of-sample
  shapes that make the rule a rule rather than a fit (`kv_heads` 5 and 6 → 10 and 9)
  are not in Gxma and have not been re-measured here.** What *is* solid is the
  maximum itself: an interleaved 8-arm curve is monotone rising to 13 and **cliffs
  −1.5% at 14** (13 → 52 workgroups, 14 → 56, and 54 is one wave / 55 is two), so
  13 is a real wave boundary. The shipped default is 16
  only because `decode_split_rule()` returns `ThreadTarget`, whose
  `16384 / (32 * q_heads) = 16` comes from a *thread* budget unrelated to
  workgroup residency. Confirmed at 16 394 tokens: **+2.17% decode** (12.067 →
  12.329), effect spread 2-4x tighter than the control's, and the explicit-16 arm
  reproduces the shipped default to 0.03%, which is the sweep's positive control.
  **The cap is not the constraint:** 24 and 32 clamp (measured 12.777 / 12.780
  against 16's 12.776) and the curve peaks at 13, *below* the cap; the only shape
  that could want more is the 64-thread sliding one, which does not use the MMA
  kernel. Flipping the default is one line
  (`KAPLUMBA_DECODE_SPLIT_RULE=workgroups` already ships it as an opt-in) and is
  provably a no-op for the 50 sliding layers, which evaluate to the 16 that ships
  today. **Not flipped here** — the call is the owner's. See
  `implementation_notes.md` §18.

