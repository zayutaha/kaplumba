# Spec: Decode to 20 t/s (PonyExl3 parity + verify-gated speculation)

## Goal

End-to-end Gemma 4 31B EXL3 decode on the M4 Pro (273 GB/s, 20 GPU cores) at
**≥ 20.0 t/s** (`decode_throughput_slow` gate ≥ 19.5), with:
1. Measured plain-decode parity with the reference (PonyExl3) kernel structure
   (P0–P2), then
2. Verify-gated speculative decoding (P3) as the only physically-sound route
   that crosses the bus ceiling.

This re-bases `spec/perf-and-memory-overhaul` T5.5r: same-algorithm decode
physically caps near ~40–45% of bus (~105–120 GB/s → ~9.5–11 t/s at 3 bpw for
our 11.2 GB/token stream; see plan §Evidence). 20 t/s = 224 GB/s = 82% of bus,
which no M-series EXL3 decode reaches; speculative decode multiplies against
the bus by emitting several tokens per weight-stream read.

## Non-goals / restrictions

- **No quality loss** (standing, AGENTS.md §8): speculative path must be
  verify-gated — draft tokens only count when validated greedy-identical
  against the target. No approximate/early-exit acceptance.
- **No KV cache quantization** (standing).
- **No Python** in product/engine code (standing).
- Prefill (stripe GEMM) is out of scope for THIS spec; it does not gate
  decode-to-20. `encode_span` prefill work stays in perf-and-memory-overhaul
  T6–T8.
- The `≥ 19.5 t/s` assert in `decode_throughput_slow` is flipped **only** after
  a measured gated run proves it. Never land red.
- Any kernel A/B that regresses `bench_passes_slow` on the A/B comparison must
  be reverted (proven by T5.1r/T5.3r).
- Do not reintroduce the proven-poison mechanisms: coalesced-word
  `simd_shuffle` ladders for k<5 (T5.3r), dynamic-index prefetch rings in local
  arrays (T5.1r).

## Requirements

1. R1 (P0, evidence): A stream-sum Metal probe reporting both (a) contiguous
   DRAM read bandwidth and (b) decode-geometry read bandwidth (strided as the
   gemv grid reads trellis) on this exact device. Printed in a gated bench.
2. R2 (P1-T1): `exl3_gemv` rewritten to the PonyExl3 v12 MT=1 production
   geometry — 128-thread TG (4 simdgroups, `NSG = 4`), 8 static register
   accumulators per lane, static funnel windows, direct (non-coalesced) word
   loads, x fetch via `simd_shuffle`, in-kernel cross-simdgroup reduction, no
   dynamic indexing. Keeps the T5.2r `nsplit_override` hook **and** in-tile
   split-K. Gate: all kernel/parity/one-hot tests green AND `bench_passes_slow`
   gate+down ≥ current (168 Gw/s) or revert.
3. R3 (P1-T2): nsplit re-sweep on the new geometry using the reference chooser
   (double splits while `out_tiles * n_splits < 8192` and `in_tiles //
   (splits*2) >= 128`; post-divisor of 8); bake argmax per shape.
4. R4 (P1-T3): fuse had_in into the M=1 gemv as an in-kernel prologue block
   (v12 `had_prologue`), killing the separate had_in dispatch + xh buffer
   round-trip per linear (411 → fewer dispatches/token). Optionally fuse
   had_out/svh as epilogue (v18) if it measures. Bit-exact vs CPU (≤1 ulp as
   today).
5. R5 (P2): end-to-end `decode_throughput_slow` strictly below the current
   142.3 ms/token with swap-usage protocol; record phase-split
   (host / gpu_wait / kernels). If the practical plain-decode ceiling lands
   ~9.5–11 t/s, amend `spec/final-same-algorithm-ceiling` with the measured
   40–45%-of-bus number and keep the gate unflipped.
6. R6 (P3): new spec `spec/verify-gated-spec-dec` with a user interview
   (draft choice, acceptance criterion, memory budget) and a verify-gated
   implementation that is greedy-identical only. Gate `≥ 19.5 t/s` flips on
   measured proof from this path.

## Acceptance criteria

- Gated `decode_throughput_slow` prints ≥ 19.5 t/s with the assert enabled
  and passing, stable across two consecutive runs with no swap-file growth.
- Everything P1 touches preserves bit-exactness as currently proven (parity,
  one-hot, trace fixture 236773 margin > 8, "The capital of France is Paris."
  CLI prompt exact).
- Each claimed speed step has its isolated A/B committed (path + numbers) so
  the history is revertable per step.

## Open questions

- P3 interview answers (draft architecture, acceptance sampling, memory
  budget) — deferred to the P3 spec. Expected now: verify-gated,
  greedy-identical; draft small enough to fit the ~14.6 GB budget.

## Confirmation

Plan reviewed with user 2026-09-23. Decisions: (1) do all P0–P3, do not stop
until 20 t/s; (2) 8-acc/128-thread register-pressure tradeoff acceptable.