# 90 tok/s prefill and 14 tok/s decode at 24k context

## Goal

At **24k tokens of context**, reach **≥90 tok/s prefill** and **≥14 tok/s decode**,
with no quality loss and within the 24 GiB machine budget.

Current, measured on `mtp-speculation` (`882d874`, merged `spec/prefill-120tps`):

| | 8k | 24k | target @24k |
|---|---|---|---|
| prefill tok/s | 80.52 | **52.84** | **90** |
| decode tok/s | 12.27 | **5.44** | **14** |
| peak `phys_footprint` | 14.15 GiB | 18.35 GiB | < 20 GiB |

> **Corrected after T1.** The 24k decode figure of 0.30–0.43 previously in this
table came from a `KAPLUMBA_GPU_PROFILE=1` run; the profiler itself distorts
the measurement (the same prompt measured 0.082 tok/s profiled vs **5.44 tok/s
clean**). The clean 24k decode is 5.44 tok/s. There is **no memory cliff**
(`swaps 0`, `sys` 20 s of 434 s), so `OQ1` is closed and decode is a 2.6×
kernel-speedup ask, not a defect hunt. See `implementation_notes.md` §2b/§2c.

Prefill at 24k is a **compute** problem with a measured budget: GEMM 213.7 s +
attention 192.9 s of a 409.9 s kernel sum. The target 90 tok/s is 266 s total,
so AC2 requires **attention 193 s → ≤ 52 s with GEMM untouched**. Full-layer
attention currently runs at **1.06 TFLOPS** (12% of SIMT fp32 peak, 7% of the
15.6 TFLOPS tensor peak); at 6 TFLOPS the run would land at 91.8 tok/s. AC2 is
reachable only through an MMA flash-attention rewrite (T3).

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
  is now the binding question.** With attention at the T1 lower bound of 86 s
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
- **OQ5 — 24k measurement is not reproducible under desktop load.** Two 24k A/B
  attempts in this session stalled: the model loaded (18 GiB wired) but the box
  entered compression (7.1 GiB compressor, <300 MiB free) once the 24k KV+scratch
  were resident, and `kaplumba` sat at ~2% CPU. The same box runs 614/8k
  measurements normally. The 24k figures in this spec (`52.84`, `~63.2`) were
  taken with less concurrent load; a 24k run needs the desktop footprint
  minimised, and any 24k sample taken under contention is invalid.
