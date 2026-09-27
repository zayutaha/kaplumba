# Prefill at 200 tok/s

## Goal

Sustained **200 tokens/s of prefill on an 8192-token prompt**, measured as the
prefill kernel rate (time to process the prompt tokens through the model,
excluding checkpoint load, weight mapping and tokenization). Today the same
work runs at **9.67 tok/s** (measured at 600 tokens), so this is roughly a 20x
improvement and is the highest-leverage item outstanding.

## Why prefill and not decode

Prefill is currently barely faster than decode, which is the actual finding:

| prompt tokens | prefill tok/s | decode tok/s |
|---|---|---|
| 24 | 4.28 | 8.00 |
| 120 | 8.31 | 7.75 |
| 600 | 9.67 | 6.86 |

At 600 tokens prefill is only 1.41x faster per token than decode, even though
prefill reads each weight **once for all 600 tokens** rather than once per token.
A memory-bound decode should be beaten by a large factor. The GEMM path is
delivering roughly decode-level throughput, so it is not exploiting arithmetic
intensity at all.

Measured in FLOPS: prefill is running at **0.56 TFLOPS**. 200 tok/s at 8192
tokens means 4.7e14 FLOPs in 41.0 s, i.e. **11.5 TFLOPS** — 21x current.

## Non-goals and restrictions

Carried from `AGENTS.md` §8, unchanged:

- No Python in engine or product code.
- Single-user chat only. No batching for its own sake, no multi-tenant scheduling.
- No KV cache quantization, no lossy requantization, no quality loss.
- No CPU fallback, no dense checkpoint materialization.
- **No materialized score matrix and no O(L^2) prefill scratch.** Prefill
  attention must be online-softmax (flash-style) with an O(L) working set.
- Scratch buffers explicitly owned, fixed-size, and **independent of context
  length**. No retained-buffer-pool semantics.
- No speculative decoding.
- Peak RSS is a measured acceptance number, not a prediction.

Agreed for this spec:

- **Decode work is paused.** The 12 tok/s decode target is deferred. The
  grid-stride matvec change is committed unswept (`63f94e5`) and untouched here.
- **KV cache may be allocated as required** for the target context length.
- **No new large scratch buffers** may be introduced for the GEMM. This is the
  binding constraint on the design and is called out as a risk (see OQ2).

## Requirements

- **R1 — Attribute the current cost.** Before optimising, measure what the 62 s
  at 600 tokens is actually spent on: the prefill GEMM, the prefill attention
  kernel, weight streaming, and per-dispatch overhead. Measured by ablation, not
  by microbenchmark, because the microbenchmarks in this repo were found to be
  contaminated by fixed `commit_and_wait` overhead.
- **R2 — Tensor-core path.** The prefill GEMM and prefill attention must use
  Metal `simdgroup` matrix multiply. Availability is confirmed (see plan); the
  data layout convention is not yet established.
- **R3 — Achieved prefill FLOPS is measured and reported** at 1k / 2k / 8k, so
  progress toward 11.5 TFLOPS is visible rather than inferred from tok/s.
- **R4 — Prefill attention working set is O(L),** independent of a materialised
  score matrix, and must not spill its per-thread accumulator to local memory.
  The current serial kernel declares `float accum[512]` indexed by a runtime
  loop bound, which cannot stay in registers.
- **R5 — Scratch is fixed-size and context-independent,** per the standing
  restriction. The same buffers must serve 1k and 8k.
- **R6 — Peak RSS recorded at 1k / 2k / 8k** via `/usr/bin/time -l`
  `phys_footprint`, and reported against the budget.
- **R7 — Numerical agreement with the current path** within a stated tolerance,
  and the existing test suite plus the five reference prompts must stay green.
- **R9 — Bit-exactness is a hard requirement, and it may gate the whole
  design.** Bit-exact accumulation against the current path means fp32
  accumulation in the same order, which rules out fp16/half2 accumulation and
  rules out exllamav3's fp16-MMA-with-periodic-fp32-fold optimisation (+14% at
  batch 1, `exl3_gemm_inner.cuh:11-20`). The measured SIMT fp32 FMA peak is
  4.38e12 FMA/s = 8.76 TFLOPS, and 4.71e14 FLOPs at that rate is 53.8 s, i.e.
  **152 tok/s — short of 200**. So:

      bit-exact + scalar fp32  ->  ceiling ~152 tok/s, target NOT reachable
      bit-exact + tensor core  ->  target reachable, but bit-exactness UNKNOWN

  Whether `simdgroup_multiply` can reproduce the current prefill accumulation
  order bit-for-bit is therefore the **gating question of this spec**, ahead of
  measuring peak tensor TFLOPS. If the MMA cannot, then bit-exactness and 200
  tok/s are mutually exclusive on this machine, and that is a decision for the
  user rather than an engineering problem. T0.3 settles it.
- **R8 — The macOS IOGPU interactivity preemption cliff is audited** for prefill.
  oMLX measured it above ~50 ms per dispatch on an M3 Max and targets 10 ms on
  every tier; M4 Pro is pre-NAX. Prefill is the regime where a single dispatch
  can run long, and it has never been checked here.

## Acceptance criteria

1. `prefill_tok_s >= 200` for an 8192-token prompt, measured as the prefill
   kernel rate.
2. Achieved prefill TFLOPS recorded at 1k / 2k / 8k, with the 11.5 TFLOPS
   requirement stated against it.
3. Peak `phys_footprint` recorded at 1k / 2k / 8k and within the agreed budget
   (see OQ1 for the budget itself).
4. Prefill attention working set demonstrated independent of context length.
5. Full workspace suite green; five reference prompts coherent; decode path
   unchanged from `63f94e5` (7.89 tok/s, no regression).
6. Preemption audit result recorded, per-dispatch wall clock reported.

## Open questions

- **OQ1 — RESOLVED (2026-09-27).** KV cache is pre-allocated to **exactly the
  context size, nothing extra** — no growth headroom, no doubling. Measured
  from the real config (10 `full_attention` layers at 4 kv heads x 512,
  50 `sliding_attention` layers at 16 kv heads x 256, `sliding_window` 1024):

  | context | full layers | sliding (capped at 1024) | KV total | + 11.51 GiB weights |
  |---|---|---|---|---|
  | 1024 | 0.08 GiB | 0.78 GiB | 0.86 GiB | 12.37 GiB |
  | 2048 | 0.16 GiB | 0.78 GiB | 0.94 GiB | 12.45 GiB |
  | 8192 | 0.62 GiB | 0.78 GiB | 1.41 GiB | **12.92 GiB** |

  Note the sliding window is 1024, so 50 of the 60 layers **stop growing past
  1024 context**; only the 10 full layers scale. KV at 8192 is 1.41 GiB, not
  the ~1.9 GiB an all-layers-scale estimate suggests.

  The consequence is that the 8192 case totals **12.92 GiB against the 12 GiB
  cap — over by 0.92 GiB**. The user has authorised the KV allocation, so this
  is recorded as the accepted budget for the 8192 case rather than an open
  question: 12 GiB remains the *weight residency* figure, and the total with an
  8192-token context is 12.92 GiB. T6 measures and confirms it.
- **OQ2 — "No extra buffers" versus a double-buffered GEMM.** A tiled GEMM
  normally wants two staging buffers for the weight tile. If the answer to OQ1
  is a hard zero-new-bytes rule, the design must reuse existing scratch and may
  lose meaningful throughput. Confirm whether "no extra buffers" means zero new
  allocations, or no new *large* allocations.
- **OQ3 — RESOLVED (2026-09-27): bit-exact.** The user requires bit-exactness
  against the current prefill path. This is a hard constraint and it has a large
  consequence that is now the central risk of the spec — see R9 below.
- **OQ4 (superseded by R9) — Is 11.5 TFLOPS comfortable or a wall?** Unknown until the tensor-core
  peak is measured (task T0). SIMT fp32 FMA peak is 8.8 TFLOPS, which is *below*
  the requirement, but the tensor-core path is a separate unit that the SIMT
  benchmark does not measure. If tensor peak is ~40 TFLOPS this is ordinary
  engineering; if it is ~12 it is unreachable without changing bits-per-weight.
