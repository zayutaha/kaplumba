# Spec: Decode gap measurement — split the 117 ms/token

## Goal

Explain, with gated measurements rather than inference, where the ~76 ms/token
gap between our decode (**117.3 ms/token = 95.7 GB/s = 35% of the 273 GB/s
roofline**, `decode_throughput_slow`) and a bandwidth-resident decode
(**41.0 ms/token = 24.4 t/s ceiling** for the 11.2 GB/token stream) actually
goes, at **realistic context (1k–8k)** rather than the cache-cap-65 baseline
where the sliding-window rotation cost is invisible.

Two competing hypotheses are already on the table, each backed by published
numbers that the other contradicts:

- **H-host**: the GPU does ~46 ms of GEMV (11.833 GB ÷ 258 GB/s, the measured
  `bench_full_model_stream_slow` rate on our own weights) and **~70 ms/token is
  everything else** — `exl3-load-latency` §"the measurement was the problem",
  which reinstated T4/T6 fusions on exactly this reasoning. Supporting: MLX on
  same-class silicon sustains ~87% of peak at batch 1, i.e. a well-structured
  loop should approach the roofline; our loop is strictly
  `commit → wait_until_completed` per token with logits readback, softcap,
  greedy argmax and (at ctx ≥ 1024) a host `copy_within` KV rotation all on the
  critical path, none of which MLX pays.
- **H-gpu**: the token is **almost entirely GPU kernel time** — the isolated
  stage table in `bandwidth-saturate` (QKV 21.90 + RoPE 0.35 + attn 0.40 +
  O 11.16 + add 1.65 + gate+up 47.07 + down 23.75 + add 1.58 + head 4.69 =
  **112.55 ms**) sums to the measured **112.40 ms** total, with "no hidden
  cross-stage overhead", and `KAPLUMBA_PROBE_SKIP_EW`/`_ATTN` barely move the
  needle (113.7 → 110.1 ms, ≈0.015 ms/dispatch). If that table is right, the
  linears execute at **92–111 GB/s in-chain while the same kernels amortised
  measure 227–377 GB/s** (`exl3-load-latency` T7 re-sweep, 60 distinct tensors,
  2.60 GB, uncacheable) — a 2–3× in-chain-vs-amortised gap that no spec has
  measured directly, and the real target.

H-gpu and H-host cannot both be right, and no existing measurement separates
them: every breakdown is host wall-clock around `commit+wait_until_completed`,
which by construction cannot tell host time from GPU time. This spec produces
the first direct split, resolves the in-chain-vs-amortised contradiction, and
delivers a ranked fix list for a follow-up implementation spec.

## Interview decisions (2026-09-25, user-confirmed)

- Gate metric: **`fwd_ms` (GPU forward only)**, e2e reported alongside.
- Context target: **1k–8k realistic**; tests must exercise ctx ≥ 1024.
- Sampling: **greedy fast path is acceptable** for the fast path (fix spec).
- Scope: **two specs** — this one measures only; a second spec implements fixes.
- `spec/verify-gated-spec-dec`: **paused** until these measurements land (T3.5
  stays queued as a follow-up probe, not run in this spec).
- KV cache layout: **any fix allowed** in the future fix spec (ring/offset,
  GPU-side rotate, amortized shift) — this spec changes no layout.
- Ground-truth MLX run on our Gemma 4 config: **skipped** (the published
  ~475 GB/s bf16 datapoint is enough to frame the gap).

## Non-goals / restrictions

- **No performance changes.** No kernel edits, no fusion, no pipelining
  shipped, no sampling change, no KV layout change. Probes are timing-only and
  may produce garbage output when they omit stages.
- **No quality change** (standing): the decode path that fast tests assert on
  must stay bit-identical; probes run only behind `KAPLUMBA_RUN_SLOW`.
- **No Python** in engine code (standing).
- **No new model weights / no downloads** (standing); measurement only uses the
  existing artifact at `~/.kaplumba/models/Gxma/Gxma` (or `KAPLUMBA_MODEL_DIR`).
- **No gate flips.** The `≥ 19.5 t/s` assert in `decode_throughput_slow` and
  every other existing gate stays untouched and green.
- **Never land red**: each task ships with its own test run + the fast suite.
- **No per-linear commit+wait timing** (the `exl3-load-latency` measurement
  bug): every phase measurement amortizes over ≥ 8 tokens or ≥ 1 full model
  pass, and the submit overhead floor (~0.16 ms/round trip) is stated next to
  every number that approaches it.
- No MLX benchmark run, no reference/ modifications.

## Requirements

- **R1 (host vs GPU split).** A gated test reports, per decode token, the
  breakdown `host_encode | gpu (commit→completed) | readback+softcap+sample |
  inter-token gap`, at ctx ∈ {~0, ~512, ~2048, ~4096}, medians over ≥ 32
  tokens. Acceptance: sums reconcile with measured e2e within 10%, and the
  numbers identify whether the missing 76 ms is host-side or GPU-side.
- **R2 (GPU phase attribution + in-chain vs amortised).** A gated test splits
  the GPU time of one token into coarse phases (layers 0–19 / 20–39 / 40–59 +
  lm_head) using separate command buffers, reports each phase's ms and share,
  **and** reports the in-chain GB/s for the two dominant linears (gate+up, down)
  so it can be compared directly with the amortised T7 numbers (227–377 GB/s).
  Acceptance: the phases plus measured per-CB submit overhead reconcile with
  R1's GPU total within 10%, and the in-chain-vs-amortised ratio for gate+up
  and down is stated as a number (settling which hypothesis holds).
- **R3 (logits path cost).** A gated test measures, separately: f16→f32
  readback+collect, softcap, greedy argmax/`sample`, and `end_token`, per token.
  Acceptance: each phase reported in ms/token with its share of `fwd_ms` and of
  e2e; the 262 144-element allocations per token are counted.
- **R4 (KV rotation cost vs context).** A gated test measures `reserve_row`
  host time and bytes moved at pos ∈ {64, 512, 1024, 2048, 4096} on a real
  `GemmaCfg` (60 layers, window 1024) built from the artifact's `config.json` —
  **no weight load required**. Acceptance: ms/token and GB/s reported per pos,
  the rotation cliff at pos ≥ 1024 either proven or disproven, and the
  `kv_cache.rs` "8 KB memmove per layer per token" doc comment corrected to the
  measured truth.
- **R5 (nsplit contradiction).** A probe-only env override forces `nsplit` in
  the decode path so `o`/`down` (and gate/q/k/v/lm_head) can be A/B'd inside a
  full decode run, not in isolation. Acceptance: the `exl3-load-latency`
  "partial store is the 3× lever" vs "partial store is free" contradiction is
  settled with one full-decode number per nsplit; policy table untouched.
- **R6 (overlap prize + rotation blocker).** A gated test measures the
  achievable gain from overlapping host encode of token N+1 with GPU execution
  of token N, at a context where no sliding-window rotation fires (window not
  yet full), and demonstrates/quantifies why the same overlap is unsafe once
  rotation fires (host `copy_within` racing an in-flight command buffer).
  Acceptance: (a) a measured overlapped-vs-serial tokens/s pair, (b) an explicit
  statement of the blocker and the ctx threshold at which it bites.

## Acceptance criteria

1. All six gated tests exist, run green on this machine
   (`KAPLUMBA_RUN_SLOW=1 cargo test -p kaplumba-engine --release --lib -- --ignored --nocapture --test-threads=1`),
   and print their numbers.
2. All fast tests still pass (no regressions), `cargo check`/`clippy` clean.
3. `implementation_notes.md` contains: the R1–R6 numbers, a reconciliation
   table proving where the 76 ms lives, a ranked fix list with estimated
   ms/token savings and the metric each targets (TTFT / tg/s / pp/s / memory),
   and an updated t/s projection for the follow-up fix spec.
4. Every task in `tasks.md` checked off with its commit hash; every finding
   that contradicts an older spec documented with file:line so the older spec
   can be amended later.
5. No existing benchmark gate weakened or assertion removed.

## Open questions

None — interview above closed them. (If R1 shows the gap is GPU-side, the fix
spec's priority order changes; that is a decision for the fix spec, not an
open question here.)
