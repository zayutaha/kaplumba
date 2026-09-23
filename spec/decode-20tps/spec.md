# Spec: Decode to 20 t/s (PonyExl3 parity + verify-gated speculation)

## Goal

End-to-end Gemma 4 31B EXL3 decode on the M4 Pro (273 GB/s, 20 GPU cores) at
**≥ 20.0 t/s** (`decode_throughput_slow` gate ≥ 19.5), by:
1. Landing the cheap plain-decode baseline wins available without quality loss
   (P1-T2 split chooser, P1-T3 fused had_in, P2 end-to-end re-verify), then
2. **Verify-gated speculative decoding (P3) as the primary route** — plain
   decode is ALU-roofed at ~7–10 t/s, so speculation with row-batched
   target-verify (the only mechanism that amortizes the decode ALU without
   quality loss) is required to cross to 20 t/s.

This re-bases `spec/perf-and-memory-overhaul` T5.5r: same-algorithm decode
reaches an **ALU roof, not a bus ceiling** — ~6–7 int/fp32 ops per decoded
weight × ~31G weights ≈ 200G ops/token vs ~1.4–2.0T ops/s on the M4 Pro GPU.
Measured 142 ms/token (7.03 t/s) sits on that roof. Speculative decode with
8-row `exl3_gemm_sg` verify amortizes the decode ALU across rows and multiplies
accepted tokens per step.

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
2. R2 (P1-T1): `exl3_gemv4`, a PonyExl3 v12 MT=1-geometry kernel built
   **alongside** `exl3_gemv` for A/B — 128-thread TG (4 simdgroups, `NSG = 4`),
   8 static register accumulators per lane, static funnel windows, direct
   (non-coalesced) word loads, x fetch via `simd_shuffle`, in-kernel
   cross-simdgroup reduction, no dynamic indexing. Keeps the T5.2r
   `nsplit_override` hook and in-tile split-K. **Outcome 2026-09-23: A/B
   negative at every nsplit — v1 stays; `exl3_gemv4` + `gemv_pass_v4` + parity
   test kept as regression bench (spec/decode-20tps/implementation_notes).**
   Gate: all kernel/parity/one-hot tests green AND `bench_passes_slow` + sweep
   never regress the v1 baseline (168 Gw/s gate).
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

Plan re-reviewed 2026-09-23 after P1-T1 (v12-parity A/B negative + ALU-roof
analysis): (3) **P3 verify-gated speculation is the primary path** to 20 t/s —
plain decode is ALU-roofed ~7–10 t/s and 20 t/s plain would need f16x2 packed
accumulate, blocked by no-quality-loss; (4) do the cheap plain-decode wins
(P1-T2 chooser, P1-T3 fused had_in, P2 re-verify) first, then P3;
(5) P3 draft architecture = **shallow-layer target draft** (target's own first
few layers + reduced head; zero extra weights; acceptance profiled on artifact).