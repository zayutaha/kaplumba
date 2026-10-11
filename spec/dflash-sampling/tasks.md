# spec/dflash-sampling — tasks

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done (commit hash appended).
This spec does not start until `spec/dflash-correctness` reaches **S1**.

## Pre-flight

- [ ] **P0** Confirm `spec/dflash-correctness` S1 is green (acceptance >= 2.4,
      greedy equivalence, footprint under budget). Read
      `/private/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/zlab/dflash/dflash/model.py:82-126`
      again and `crates/engine/src/sampler.rs` in full. **Tested by**: the
      correctness spec's T11 example runs and reports a number.

## Tasks

- [ ] **T1 — `Sampler::probability_of` and `sample_residual`.** Add to
      `crates/engine/src/sampler.rs` a helper that returns `P(token)` under a
      `SamplerConfig` (temperature, `top_k` on the scaled scores, then `top_p`
      with the "running sum before this token" rule, renormalised) and one that
      samples from `norm(max(0, p − q))` over the surviving candidate set.
      Do not alter `Sampler`'s existing behaviour. **Files**:
      `crates/engine/src/sampler.rs`. **Tested by**
      `crates/engine/tests/dflash_sampling_unit.rs`: hand-built 8-token
      distributions where the expected `probability_of` and the residual
      sample are computable in closed form (a single-token `p`, a uniform `q`,
      disjoint supports, overlapping supports). *Sabotage*: apply `top_p`
      before `top_k` and confirm the ordering test fails.
- [ ] **T2 — unit tests for the rejection rule.** `p`/`q` arrays with known
      ratios, a seeded RNG, asserting the accepted run length and the emitted
      token against a reference computed by a straightforward Python-free
      reference implementation inside the test file. Cover `accepted == 0`,
      `accepted == 15` with the bonus drawn from `p_15`, and a mid-window
      rejection. **Files**: `crates/engine/tests/dflash_sampling_unit.rs`.
      *Sabotage*: compare each position independently instead of cumulatively
      and confirm the test fails.
- [ ] **T3 — `draft_block_sampled`.** `dflash.rs` gains a sampled variant that
      uses `Sampler` for the 15 draft tokens and returns their probabilities
      (R1). **Files**: `crates/engine/src/dflash.rs`. **Tested by**: the 15
      returned tokens are reproducible under a fixed seed and their
      probabilities match a direct recomputation of
      `probability_of(logits_i, token_i)`.
- [ ] **T4 — the rejection loop.** `generate.rs`: at `temperature > 0`, accept
      while `u_i * q_i < p_i` (cumulative), then residual or bonus (R2, R3).
      The greedy branch stays byte-identical. **Files**:
      `crates/engine/src/generate.rs`. **Tested by**: T2's harness driven by
      the real loop with a stub drafter that produces a known `q`.
- [ ] **T5 — per-position masks and EOS.** Advance filters, repetition
      penalties and past-id sets after every accepted draft position (R7); a
      stop token inside the window truncates the accepted count (R6).
      **Files**: `crates/engine/src/generate.rs`. **Tested by**: a
      banned-string case that triggers mid-window and asserts the emitted
      text never contains it, plus a multi-turn run at `temperature 1.0` with
      no degenerate repetition.
- [ ] **T6 — two-sample distribution test (AC1).** `crates/engine/tests/dflash_distribution.rs`:
      N = 20 000 sampled tokens from non-speculative and speculative DFlash at
      `(temperature, top_p)` ∈ {(1.0, 1.0), (1.0, 0.95), (0.7, 1.0)}; report
      TV distance and a two-sample χ² p-value. **Files**: new test. **Tested
      by**: the test itself, plus AC4's degraded-drafter rerun.
- [ ] **T7 — determinism (AC2).** Same seed twice → identical output; and
      output invariant under a reduced drafter block size beyond the documented
      near-tie exemption. **Files**:
      `crates/engine/tests/dflash_distribution.rs`. *Sabotage*: draw the
      uniforms lazily (only when a position is reached) and confirm the
      determinism test fails.
- [ ] **T8 — degraded drafter (AC4).** Perturb the drafter's weights so
      acceptance is far below 1 and rerun T6. This is the only test that
      exercises R3's residual path at scale. **Files**: new test fixture.
      *Sabotage*: replace the residual with a plain argmax of `p` and confirm
      T6 fails on the degraded run.

## Stage-gate

- [ ] **S1** AC1 passes at all three `(temperature, top_p)` points, AC2 and
      AC4 pass, AC3 still passes, and AC5's tok/s regression is reported.
      If AC1 fails only at `top_p = 0.95`, that is Q1: fix the truncation to
      match, rerun, and record which definition was wrong.

## Notes

- Commit after every task: `spec/dflash-sampling: T<n> <one line>`.
- No "distribution-faithful" claim without the two-sample numbers written into
  `implementation_notes.md`.
