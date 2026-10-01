# Logit-distribution probe (KL / top-1 agreement between two checkpoints)

## Goal

Make the quality cost of a re-quantization **measurable**. Concretely: an offline
instrument that, given two checkpoint directories, captures the full fp16
next-token logit row (262,144 values) at every teacher-forced position of a fixed
prompt corpus from each checkpoint and reports, per position and in aggregate:

- top-1 agreement rate,
- KL(reference ‖ candidate) and KL(candidate ‖ reference) in nats,
- the 1-based rank of the reference argmax under the candidate,
- mean and max |Δlogit|,
- next-token NLL and perplexity for both, and their difference.

This unblocks the 3.0 bpw → 2.0 bpw decision (13.5 → ~19.8 tok/s arithmetic) and
every later re-quantization. It does not change decode at all.

## Non-goals / restrictions

- **No change to the decode path.** The probe is an example binary
  (`crates/engine/examples/logit_probe.rs`) plus a pure-computation module
  (`crates/engine/src/logit_probe.rs`). The engine's forward path gains no branch,
  no env check and no per-token work.
- **No Python**, in product code or harness (`AGENTS.md` §8).
- **No KV quantization**, no quality changes, no sampler changes.
- **No multi-user / batching**; one model process at a time (`AGENTS.md` §10).
- Not a perplexity benchmark in the textbook sense: prompts are *corpus-like,
  self-authored* rather than wikitext, because there is no dataset on this box.
  It is a *relative* instrument (A vs B on identical inputs), which is what a
  re-quantization decision needs.
- The probe does **not** attempt to load a bitrate the engine does not support.
  Reporting that a 4 bpw checkpoint is rejected by the loader is a finding, not a
  bug to paper over with a silent fallback.

## Requirements

- **R1 — Full row capture.** Every captured position records all 262,144 fp16
  logits, not a top-k summary. Readback uses the engine's existing
  `forward_buffer_into_vec` path, which already reads the whole `scratch.capped`
  row once per step; no new readback hook is required.
- **R2 — Two-argument comparison.** `capture` writes a self-describing file;
  `compare` consumes two such files. This lets one reference capture be
  compared against N candidates without re-running the reference, and keeps peak
  host memory at two rows (1 MiB) rather than N rows.
- **R3 — Metrics per row**, exactly as listed in the Goal.
- **R4 — Self-check is a property of the tool, not a claim.** `compare` always
  computes `metrics(ref, ref)` and prints `SELF-CHECK: PASS/FAIL`; it exits
  non-zero if the self-check fails or if the two captures disagree about
  vocab size, softcap or row count. An instrument that reports 0 always is
  caught by its own exit code.
- **R5 — Deterministic corpus.** The default prompt set is compiled into the
  binary, and `--prompts-from <file>` overrides it. Rows are tagged with
  `(prompt_index, position, fed_token, next_token)`, so two captures from
  differently-corpus'd runs are rejected rather than silently misaligned.
- **R6 — Numeric ground truth is unit-tested, not asserted.** Metric math is
  covered by synthetic rows with analytically known answers (constant logit shift
  ⇒ KL exactly 0; delta vs uniform ⇒ KL = ln V; hand-computed 4-token softmax),
  **and** each of those tests must be shown to fail when sabotaged.
- **R7 — End-to-end detection arm.** A real perturbed checkpoint must be built
  from the real one by patching bytes on disk, and the probe must report a
  non-zero KL and a measurable disagreement rate on it. The perturbation must be
  chosen so the correct answer is known in closed form (see `plan.md`), so the
  test is a comparison against a prediction and not merely "the number moved".
- **R8 — Baseline on the real corpus.** `Gxma-repacked` against itself, and
  against `Gxma` (same weights, different on-disk packing), are reported. The
  first is the zero control; the second measures the instrument's floor on a
  *real* pair that should be identical.
- **R9 — Second checkpoint assessed.** `~/Downloads` (EXL3, reported 4.0 bpw
  layers / 8 bpw head) is inspected for loadability and the blockers are
  enumerated precisely. A metadata shim is a legitimate deliverable; a silent
  fallback that pretends a 4 bpw checkpoint is a 3 bpw one is not.

## Acceptance criteria

1. `cargo test -p kaplumba-engine --test logit_probe` passes, including the
   synthetic-metric tests.
2. Each synthetic test has been **sabotaged and observed failing** (AGENTS.md
   §3.4), with the failure message recorded in `implementation_notes.md`.
3. `SELF-CHECK: PASS` with `kl = 0` and `agreement = 100%` when a capture is
   compared with itself.
4. `SELF-CHECK: PASS` and `kl = 0` for `Gxma-repacked` vs `Gxma-repacked` across
   two separate processes.
5. The perturbed checkpoint produces `kl > 0` by orders of magnitude, agreement
   well below 100%, and the measured per-row result matches the closed-form
   prediction for the perturbation that was applied.
6. A written acceptance band for a 2 bpw model exists in
   `implementation_notes.md`, grounded in measurements taken here. If the
   literature does not support a defensible threshold, that is stated rather than
   replaced with an invented number.

## Open questions

None requiring user input: the brief fixed the metric set, the two-argument
interface, the validation arms and the reporting targets. The one genuinely
undetermined item — what KL is "acceptable" — is answered by R6/criterion 6 with
whatever the data supports.
