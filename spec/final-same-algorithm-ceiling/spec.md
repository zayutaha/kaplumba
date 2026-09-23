# spec/final-same-algorithm-ceiling

## Goal
Close the `spec/rust-metal-engine` same-algorithm track by (a) measuring and closing the small remaining gaps to the funnel-ALU ceiling (9–12 t/s decode, 100–110 t/s prefill), (b) verifying long-context decode stability (ctx ≥ 1024), and (c) documenting the hard ceilings so 18/200 t/s is formally acknowledged as unreachable under the strict MTP gate.

## Non-goals / restrictions
- **No MTP / speculative decoding** (user gate 2026-09-23: "Hold the gate strictly").
- **No quality loss** — all token gates hold (fixture argmax 236773, span |argmax| < 0.55, batched-vs-gemv rel < 2e-3).
- **No Python / no MLX** in engine (DE19 offline converter only, DE20 `scripts/bench.py` retire at R7.4).
- **No KV cache quantization**.
- **No memory increase** — scratch transient ≤ ~128 KB OK.
- **No multi-user, no request batching**.
- Target: Apple M4 Pro, 20 GPU cores, 273 GB/s; Gemma 4 31B EXL3 ~3 bpw (16.028 GB weights + ~1 GB KV).

## Requirements (numbered, testable)

1. **T3.10 acceptance**: `prefill_throughput_slow` at 512 tokens ≥ +8% over pre-T3.10 baseline (86.8 t/s → **≥ 93.7 t/s**). Fixture tokens identical, span gate holds.
2. **T2.10 decode phase split**: `decode_throughput_slow` reports three measured durations per token:
   - `cpu_encode_ms` (encoder build + dispatch encoding)
   - `gpu_wait_ms` (`commit` + `wait_until_completed`)
   - `host_overhead_ms` (sample + detokenize + loop)
   Acceptance: all three non-zero, sum ≈ wall ms/token; `cpu_encode_ms` ≤ 5 ms (if >5 ms, pipeline opportunity quantified).
3. **T2.11 long-context decode**: New gated test `decode_long_context_slow` with prompt length ≥ 1200 tokens, 64 steps. Measures t/s vs short-context baseline. Acceptance: t/s drop ≤ 10% (if >10%, rotation fix required and implemented).
4. **T2.12 glue/churn trim**: 
   - Widen/fuse the 371 single-TG rmsnorm dispatches (grid ≥ rows div_ceil(64) or fuse into neighbor linear).
   - Revert k/v over-split: change TARGET_TGS policy for shapes with `out_tiles ≤ 256` to avoid `nsplit > 4` when it regresses isolated Gw/s (measured k/v 106→94 at nsplit=16).
   Acceptance: fast suite green, full gated suite 114/0/5, `decode_throughput_slow` ≥ current 8.39 t/s, k/v isolated bench ≥ 100 Gw/s.
5. **T6.1 final report**: `implementation_notes.md` section with measured ceilings table (decoded 9–12 t/s, prefill 100–110 t/s, BW 53–70%), llama.cpp/omlx comparison citations, 18/200 marked unreachable with gate references.

## Acceptance criteria
- All gated tests pass: `KAPLUMBA_RUN_SLOW=1 cargo test -p kaplumba-engine --release --lib -- --test-threads=1` → 114 passed / 0 failed / 5 ignored.
- `prefill_throughput_slow` ≥ 93.7 t/s at 512 tokens.
- `decode_throughput_slow` prints phase split + achieved GB/s; t/s ≥ 8.39.
- `decode_long_context_slow` (new) runs and reports ≤ 10% drop vs short ctx.
- k/v isolated bench ≥ 100 Gw/s (regression fixed).
- `implementation_notes.md` final ceiling section committed.
- All tasks committed separately with `spec/final-same-algorithm-ceiling: T<id> ...` messages.

## Open questions
- None — all gates and ceilings are measured; this spec only closes the remaining measurement/trim work.

## Ceiling correction 2026-09-23 (supersedes the "9–12 t/s" estimate)

Re-measured after `spec/decode-20tps` P0–P2: the same-algorithm decode ceiling
is an **ALU roof, not a bus ceiling**. The funnel+`exl3_decode_f`+MAC path
costs ~6–7 int/fp32 ops per decoded weight ⇒ ~200G ALU ops/token on the M4 Pro
(~1.4–2.0T ops/s) ⇒ **plain-decode ceiling ≈ 6.9–7.0 t/s measured (144.5 ms/
token forward-only, 2026-09-23), ~7–10 t/s extrapolated at optimistic ALU**. The
earlier "9–12 t/s / 40–45% of bus" prediction conflated bus headroom with ALU
availability; the bandwidth probe (P0) confirmed 214–381 GB/s is achievable for
the load pattern alone, i.e. memory is not the binder. Parity restructures
(v12-geometry `exl3_gemv4`) and nsplit argmax tuning (P1-T2) did not move the
measured number beyond run noise. `spec/decode-20tps` therefore moves the 20 t/s
deliverable to MTP/verify-gated speculation as the primary route.