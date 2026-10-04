# 90 tok/s prefill and 14 tok/s decode at 24k context

## Goal

At **24k tokens of context**, reach **≥90 tok/s prefill** and **≥14 tok/s decode**,
with no quality loss and within the 24 GiB machine budget.

Current, measured on `mtp-speculation` (`882d874`, merged `spec/prefill-120tps`):

| | 8k | 24k | target @24k |
|---|---|---|---|
| prefill tok/s | 80.52 | **52.84** | **90** |
| decode tok/s | 12.27 | **0.30–0.43** | **14** |
| peak `phys_footprint` | 14.15 GiB | 17.12 GiB | < 20 GiB |

The two gaps are different problems. Prefill at 24k is a **compute/bandwidth**
problem (attention is ~60% of wall at 24k, GEMM the rest). Decode at 24k is a
**collapse** — 32× slower than 8k for 3× the context, ~11× worse than the O(L)
KV-read scaling predicts — and is therefore expected to be a defect or a
memory-pressure cliff rather than a slow kernel. Characterising it is R1.

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

- **OQ1 — Is decode-at-24k memory pressure?** 17.12 GiB measured footprint on a
  24 GiB box, with the OS growing swap. If decode is disk-bound, the fix is
  memory, not kernels, and 14 tok/s may require the footprint to come down
  materially.
- **OQ2 — Is 90 tok/s @24k reachable even with free attention?** At 24k the
  GEMM is ~9.6 ms/token-equivalent; if attention were free, 24k would be
  ~104 tok/s. So 90 is between the current 52.8 and the attention-free bound —
  it does **not** require a GEMM improvement, only a large attention one.
- **OQ3 — Single-pass decode gate.** `ATTENTION_SINGLE_PASS_MAX_KEYS = 64`, so
  24k always uses split+combine. Verify the split grid at 24k actually fills the
  machine (`splits × kv_heads` workgroups).
