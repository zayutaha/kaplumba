# Finding 04 — The 8.9 ms host gap, attributed. The fix is one line.

Subagent: `ses_f0cf01f4cffegnIzxpdiAIKs1U`. Read-only; no production file edited.
Measuring stick + raw outputs: `/private/var/folders/.../T/opencode/kaplumba-attr/`.

**Caveat:** all runs used one consistent binary at HEAD `e9c2785` plus then-
uncommitted kernel work. A concurrent session has since replaced the binary
twice; the paired A/B on the new binary could not complete (runtime Metal compile
broken at `program_source:12049`). **Absolute tok/s may shift under their edits;
the structure of the decomposition will not.**

## Control (uninstrumented, default path `generate_gpu`)

| run | tok/s | ms/step |
|---|---|---|
| A1 | 12.337 | **81.06** |
| A2 | 11.371 | 87.94 <- outlier |
| A3 | 12.313 | **81.21** |
| A4 | 12.357 | **80.93** |

Control = **81.07 ms/step, 12.33 tok/s** (A1/A3/A4 spread 0.28 ms).

## The decomposition closes exactly

`dp_own` (GPU) = 72.74 ms. Reported mean step = 81.07 ms.

| component | ms | % of gap | status |
|---|---:|---:|---|
| **(a) host CPU encode, 100 % exposed** | **5.75** | **69 %** | MEASURED x3: `span−wall` 5.727/5.597 (B1/B2) + direct `encode=` timer 5.71 mean / 5.88 median (C1) |
| cold first decode step, amortised /63 | 1.77 | 21 % | MEASURED (C1 step 0 = 190.38 vs 79.18 steady) |
| **(b) explicit synchronisation** | **0.00** | 0 % | MEASURED: 1.00 buffers/step ⇒ no `wait_until_completed` inside the step, `MAX_IN_FLIGHT` never engaged |
| (b') `end_encoding` + `commit` + GPU-end->host-wake | 0.64 | 8 % | MEASURED (`dp_wall` 73.38 − `dp_own` 72.74) |
| **(c) host argmax** | **0.00** | 0 % | MEASURED: 0.262 ms, but **outside** `decode_tok_s` (`generate.rs:285` precedes `:319`) — not in this gap at all |
| (c') encoder create + 512 KiB logits copy + loop | 0.17 | 2 % | MEASURED (residual; C1's independent residual is 0.04 ms) |
| (d) measurement overhead | 0.02–0.03 | — | MEASURED by component. Observed B1−A1 = +1.0 ms is **inside the control spread** |

**Closure: 5.75 + 0.00 + 0.64 + 0.17 + 1.77 = 8.33 ms vs the 8.33 ms gap. Exact.**

**Steady state** (amortising the cold step away, which shrinks with generation
length): 79.30 ms = 72.74 GPU + 5.75 encode + 0.64 commit/wake + 0.17 loop.
**The steady-state gap is 6.56 ms, of which 5.75 ms (88 %) is the exposed host
encode.**

## THE STRUCTURAL CAUSE — and the fix

**`model.rs:3111` — `encoder.set_max_dispatches(2500)` — with 1617
dispatches/step, `MetalEncoder::tick` (`encoder.rs:462-469`) never reaches the
threshold and `roll()` never fires.** MEASURED at **1.00 command buffer per step**
(`KAPLUMBA_COUNT_BUFFERS=1` → `BUFFERS per_step=1.00`, `DISPATCHES
per_step=1617.0`).

The entire step is encoded into a single *uncommitted* buffer, so the GPU
receives nothing until the host has finished encoding, then runs 72.7 ms with the
host blocked in `commit_and_wait`. **The step is a strict concatenation with zero
overlap.** All the pipeline machinery already written (`roll`,
`MAX_IN_FLIGHT = 8`, back-pressure on the oldest buffer) is **switched off for
decode.**

This is not a fundamental limit. The real serialisation — next token = argmax of
this step's logits (`generate.rs:285`) — forces only **one barrier per step**. It
says nothing about whether the 1617 dispatches *within* a step can be submitted
incrementally.

**Dropping or lowering `set_max_dispatches(2500)` to 50 turns the step from
`5.75 + 72.74` into `max(GPU, encode) + ε ≈ 73.0 ms`:**

- **Expected: 79.30 -> ~73.1 ms/step, 12.33 -> ~13.4 tok/s (+8.6 %).**
- Cost: ~32 extra `commit()` x 0.002–0.005 ms (measured) ≈ 0.1 ms, plus ~0.13 ms
  of GPU buffer-boundary overhead (from the 3.8 µs/buffer figure). **Net expected
  +7.5 to +8 %.**
- `MAX_IN_FLIGHT = 8` at 50/buffer = 400 dispatches of lookahead; the host fills
  it in ~1.4 ms then correctly throttles in `roll()`. Intended behaviour, not a
  stall.
- **The shared host-staged scratch does NOT block this.** `model.rs:3040-3057`
  warns `forward_chunk` writes `scratch.token_ids[0]` / `scratch.query_positions[0]`
  from the host — but **once per forward, not per dispatch**. Intra-step rolling
  is safe; the between-step barrier the terminal `commit_and_wait` provides is
  still required and still correct.
- **Correctness caveat:** `KAPLUMBA_MAX_IN_FLIGHT=1` exists to serialise prefill
  chunks against that same shared scratch. For decode, intra-step rolling writes
  no host-staged data, so it must be **re-verified with a greedy golden-output
  test rather than assumed.**

## Launch floor — bounded at ≤ ~1.5 ms

`KAPLUMBA_DECODE_PROFILE=kernel` (E1) sums **76.201 ms/step over 905 stamped
buffers** — *more* than the 72.74 ms single buffer containing the same work, i.e.
~3.8 µs of per-buffer overhead x 905 = **3.46 ms of artefact**. **The sum of the
parts exceeds the whole, so a large positive intra-buffer bubble is excluded by
measurement.**

## Doc bugs found (stale, both now wrong)

- `decode_profile.rs:36-37` says step-mode `own` is "only the two boundary
  buffers, so it is small and not a decode cost". **`dp_own` here is 72.74 ms —
  the real GPU cost.** Same 72.6 ms the brief calls "kernel-own".
- `decode_profile.rs:16-22` and `generate.rs:893` still say "~42 rolled buffers
  per step". **Measured command buffers/step: 1.00.** (The 42 is exactly
  `2100/50` — the test's own constant over the default budget, cited as a
  measurement, i.e. circular.)
- **`model.rs:3192`'s empty commit is wrong twice.** (1) There is **no empty
  commit in the default path** — with `Mode::Off` the real buffer holding all
  1617 dispatches is committed there. The empty commit exists *only because of*
  `KAPLUMBA_DECODE_PROFILE=step`. (2) It does not cost 117–160 µs: MEASURED
  end_encoding+commit+wait on a 0-dispatch buffer = **0.0139/0.0143/0.0150 ms
  (14–15 µs)**, min 10.2 µs — **~10x less**. The 117–160 µs figure is almost
  certainly the *first commit after the GPU has gone idle* (wake-up latency).
- `generate.rs:867`'s unconditional `argmax_with_margin`: `MarginLog::record` is
  gated but the call is not, so `--legacy-path` always pays 2 extra passes.
  **0.133–0.164 ms per extra pass, ~0.30 ms/token wasted on `--legacy-path`
  only.** Default path correctly guarded at `generate.rs:293-299` — confirmed it
  touches no measured number (C1 and A1 step time agree within 0.3 ms).

## Per-field semantics (first ever `=step` and `KAPLUMBA_STEP_TRACE=1` runs)

- `dp_span` = `step/close.GPUEndTime − step/open.GPUStartTime` — whole step on the
  GPU timeline **including GPU idle**.
- `dp_wall` = host time *after* `forward_buffer_into` returned, deliberately
  **excluding the encode**, which is what makes the difference interpretable.
- `dp_own` = `open.own_ms + close.own_ms` = **the entire step's GPU occupancy**.
- `dp_span_over_wall = 1.077` -> **7.7 % of the step is GPU idle.**

**`dp_span_over_wall` CANNOT separate inter-kernel scheduling latency from
CPU-side encode.** It is a host-pipelining metric: because the step is one
uncommitted buffer, *all* host encode lands in the idle term. Turn rolling on and
the identical code reports `span ≈ wall` even though the same 5.7 ms of encode is
happening — it would just move inside the span.

`KAPLUMBA_STEP_TRACE=1` (legacy path, 60 steady steps): encode **5.71** mean /
5.88 median (p10 4.26, p90 7.80); wait 73.26; total 78.98. Step 0 = **190.38 ms**
(cold pipelines), +111.2 ms over steady = **+1.77 ms/token amortised over 63
steps**.

**`encode = 5.71 ms / 1617 = 3.5 µs per dispatch` of engine CPU. Raw Metal encode
is 0.10 µs/dispatch — so ~97 % of the encode is the engine's own host work, not
Metal API cost.**

## Irreducible, currently invisible

The 0.262 ms host argmax is a true cross-step RAW dependency, excluded from
`decode_tok_s` by construction. Removing it needs a device-side argmax;
`model.rs:3052-3057` and `:3378-3391` record why the existing
`encode_greedy_pick` is unusable at 262,144 vocab (`greedy_key` packs
`0xFFFF - index` into 16 bits, underflowing for every index >= 65,536).

## Where the remaining time actually is

The 8.9 ms is 11 % of the step. Inside the 72.74 ms of GPU own time, 12.04
GB/token at 250 GB/s would be 48.2 ms — so **~24.5 ms of the gap to the bandwidth
roof is inside the GPU number and needs kernel work, not host pipelining.** The
launch-bubble share of that is bounded at ≤ ~1.5 ms, so the remainder is kernel
efficiency, not dispatch count.

## Recommended next steps (not done — read-only brief)

1. Delete `encoder.set_max_dispatches(2500)` at `model.rs:3111` and
   `generate.rs:891`; A/B at 12.33 vs ~13.4 tok/s with
   `KAPLUMBA_DECODE_PROFILE=step` **and a greedy golden-output check**.
2. Fix the stale docs listed above.
3. If (1) is taken, re-measure whether the 5.7 ms encode still matters — it will
   be hidden, and the next target becomes the 3.5 µs/dispatch of engine-side
   parameter marshalling (0.10 µs is the Metal floor), a ~35x gap worth its own
   spec.
4. Gate `argmax_with_margin` behind `MarginLog::enabled` at `generate.rs:867`.

## Open question flagged by another agent, not resolved here

`implementation_notes.md` §20.7 records `norm_transform`'s two dispatches together
at 17.77 µs while `head/final_norm` — the same kernel at the same shape — costs
16.7 µs alone, leaving ~1.1 µs for the 42-group `linear_input_transform` beside
it. Finding 07 measures the same kernel elsewhere at 11.2 µs of "work". **11.2
vs 16.7 µs for one dispatch of the same kernel at the same shape is unexplained
and should be reconciled before either number is used to justify anything.**
