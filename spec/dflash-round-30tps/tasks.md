# spec/dflash-round-30tps — tasks

**Implementation authorized.** The user's latest 2026-10-13 instruction is to
implement, fully test, and continue toward 30 tok/s. Follow the ordered gates;
do not report the throughput target as met without end-to-end measurement.
Preserve existing task IDs so historical commits remain traceable.

Execution order:
T17/T18 → V0a → V0b → verifier decision. If adequate headroom exists, T0c/T5/T6/T7/T8
establish independent drafter evidence; T1c/T1b/T16/T3 establish the new round
contract; T12's round harness precedes T1/T2/T4 and its end-to-end checks follow
CLI wiring. T10/T15/T13/T14 demonstrate real gain. T9 and F1a/F1b are conditional
branches selected from those results, not mandatory simultaneous implementations.

## Proved blockers from the exllamav3 review

- [x] **T17 — honor packed offsets in buffer GEMM.**
  Files: `crates/kernels/src/linear.rs`, existing buffer-linear tests and head
  diagnostics. Bind packed tensors at `packed_offset` in fp16/fp32 buffer paths,
  keeping independent `packed_bytes` length validation. Test zero/nonzero offsets,
  irrelevant prefix canaries, row strides, 4/6-bpw and both output dtypes; compare
  identical nonzero inputs with production `encode_gemv`. Reintroduce zero binding
  and observe failure. Avoid a permanent copied head as the fix. The temporary
  real-head rebase control already isolated this bug. Passed and sabotage
  evidence: below; commit `fda508d`.
- [x] **T18 — isolate and repair target batch/decode divergence.**
  Files: `model.rs`, `crates/engine/tests/dflash_verify_head.rs`, and
  `crates/kernels/tests/linear_mrhs.rs`. QKV and gate/up fusion now require
  matching projection input-sign vectors; otherwise each projection applies
  its own transform. Cross-layer finalize→next-norm is opt-in pending repair.
  Forced-load test: three prompts × M=2/4/8/16, all-row greedy identity; max
  observed logit drift ≤1.646 at M=16 across reruns, below the 2.0 diagnostic
  bound. Per-layer M=2 cache/tap comparison: K max 0.0031, V max 0.0508, hidden
  max 0.375.
  Sign-predicate sabotage failed its named assertion; re-enabling cross-layer
  fusion failed at layer 1 (K 0.3288, V 9.8483). Sampled-distribution
  equivalence remains open. Commit recorded below.

## First gate: correct verifier and optimistic ceiling

- [ ] **V0a — sequential versus batched target verification.**
  Files: `crates/engine/tests/dflash_verify_head.rs` and `model.rs` only if a
  diagnostic interface is required. Compare identical prefixes, token IDs and
  positions at M=2/4/8/16. Reset/isolate state for each arm. Test per-row greedy
  identity, justified logit tolerance, cursor accounting and ring boundaries;
  sabotage row selection and positions and record diagnosable failures.
- [ ] **V0b — ideal-proposal verification cost.**
  Files: a Rust verifier timing harness under `crates/engine/tests/` and these
  notes. Use saved correct target continuations; measure body/head/full wall
  time at M=2/4/8/16, ≥3 prompts, short/1k/2k/8k context, ≥3 interleaved runs
  per shape. Test row counts and timing boundaries. Report both anchor-contract
  synthetic ceilings, medians/spread and remaining complete-cost budget for
  30 tok/s. These results do not count as product throughput. If all ceilings
  fail, record the existing verifier bottleneck before selecting a drafter.

## Audit and existing evidence

- [x] **A1 — reconcile specification with code and sources.** Files: this spec's
  four documents and supersession notices in the two older `spec.md` files.
  Validation: read-only source/history inspection, independent arithmetic,
  and a two-token distribution counterexample. Commit: recorded in Status.
- [x] **T0a — existing overlap rerun recorded.** Historical 2026-10-13 notes:
  mean .0000 default/config+1, .0565 colmajor, best .8479 at one off-diagonal
  pair. This is a probe observation, not production acceptance or coverage.
- [x] **T0b — existing stage scan recorded.** Historical notes report distinct
  rows after RoPE. This does not certify correct attention or head semantics.
- [x] **T0d — spec reconciliation.** A1 supersedes stale speed gates, head blame,
  and the old phase order. This is documentation completion only.
- [x] **A3 — exllamav3 reference review and authorized diagnostics.**
  Files: documentation and retained text evidence in this spec folder. Compare
  pinned reference revision, run temporary factorial/head/cache diagnostics and
  the existing target verify test. Record the proved offset and ring failures,
  layer-0 target discrepancy and invalidated layout inference. No engine/CLI
  edits. Commit recorded in Status after committing.

## Correctness before product wiring

- [ ] **T0c — repair the head-oracle control.**
  Files: `crates/engine/tests/dflash_head_oracle.rs` and, only if needed for a
  trustworthy identical-input control, a target diagnostic interface.
  Fix vocabulary row stride, establish transform/scaling/softcap conventions,
  compare the actual target head, and assert justified numerical tolerances.
  Test: multiple nonzero rows and column slices; deliberately corrupt indexing
  and scaling and observe named numerical failures. Shape-only success is invalid.
- [x] **T1a — sampled proposal correctness.**
  Files: `crates/engine/src/dflash.rs` and its unit tests.
  Draw candidates from q in sampled mode, retain exact q for verification,
  reject invalid/empty distributions and uniforms, keep greedy deterministic.
  Test: deterministic quantile coverage, anchor-row skip, top_k=0, zero-mass
  entries, invalid uniforms, and argmax sabotage with a named failure.
  Completed in `2f35201`; focused DFlash suite 12/12 passes.
- [ ] **T1c — complete supported sampler policy.**
  Files: `dflash.rs`, generation interfaces and sampling tests. Apply the same
  declared temperature/top-k/top-p to p and q; retain q used for the draw.
  Support CLI default (1,64,.95) and greedy, reject unsupported configurations.
  Test independent probability fixtures, inclusive nucleus/ties, top-k boundaries,
  bad logits/uniforms and residual support; sabotage truncation/temperature.
- [ ] **T1b — transaction and ring rollback.**
  Files: `crates/engine/src/model.rs`, `gpu_cache.rs`, `dflash.rs`, and
  cache/round tests.
- [ ] **T1b.1 — preserve the attention window during chunk writes.**
  Sliding physical capacity is `window + prefill_chunk()`; attention still masks
  to the configured window. This preserves keys needed by the earliest query
  during any bounded chunk append, including verify. Gxma's default reserve
  increases by 839,065,600 bytes (0.78 GiB), from 3,524,730,880 to
  4,363,796,480 bytes. Measure the actual process footprint with
  `/usr/bin/time -l`. Test the 1024 boundary against sequential decode and
  assert key position 0 remains present after the batched write. Sabotage the
  physical capacity and confirm that named assertion fails.
  Completed in `COMMIT_HASH_PENDING`.
- [ ] **T1b.2 — transactional rollback after rejection or error.**
  Rejected future rows must be excluded by absolute positions after cursor
  rollback; verify stale rows cannot leak into later outputs. Test zero/partial/
  full acceptance before/at/after ring wrap, repeated rejection, error rollback,
  and equality of retained K/V/positions with sequential decode. Sabotage position
  accounting and restoration independently.
- [ ] **T3 — context and taps.**
  Files: `crates/engine/src/dflash.rs`, `generate.rs`, and round/context tests.
  Save old-anchor and accepted verify taps before scratch reuse; append each
  verified feature once with explicit exclusive-end position; process next
  once per new round's verification. Never rewind a tail that was not appended.
  Test: prompt initialization and consecutive rounds for K=0, partial, full;
  context ends at new anchor−1; boundary and overwrite sabotages must fail.
- [ ] **T16 — unprocessed-anchor round API.**
  Files: `dflash.rs`, `model.rs` if needed, and round tests. Replace the planned
  loop's processed-anchor dependency with verify `[anchor, proposals]`; derive
  candidate distributions from verify rows and leave bonus unprocessed. Test
  K=0/partial/full, B=1 fallback, cache end/tap ownership and consecutive rounds;
  sabotage duplicate anchor feeds, row shifts and extra correction forwarding.
- [ ] **T5 — validate existing permutation propagation.**
  Plumbing already landed in `9a4cf60`; do not implement it again.
  Files: existing kernel/drafter permutation tests and implementation notes.
  Test: independent expected values through actual buffer calls, not merely
  “mode changes output”; hardcode the wrong mode and observe failure.
- [ ] **T6 — layout/default evidence.**
  Sweep work recorded in `6622031`; complete its default/reference validation.
  Its overlap-based colmajor conclusion is invalidated by the ignored head
  offset. Corrected-head control favors original row-major on the measured prompt.
  Files: drafter loader/tests and implementation notes if a mismatch is proven.
  Test: checkpoint reconstruction and distinct target-head/drafter modes; log
  actual default and justified layout. Do not choose layout solely by overlap.
- [ ] **T7 — head diagnosis after T0c.**
  Files: repaired head-oracle test and notes; production change only if proven.
  Test: independent host vs target head vs drafter head on identical inputs,
  declared tolerances, representative rows/slices, and path-specific sabotage.
- [ ] **T8 — aligned checkpoint semantics.**
  Files: `dflash_alpha_probe.rs`, diagnostic tests, and notes; fix a proven
  semantic mismatch as a separately committed task if necessary.
  Test: z-lab stage contracts for raw embeddings, taps/fc/norm, K/V, RoPE,
  masks and row selection. Log per-depth ranks/overlap on ≥3 prompts. Replace
  hardcoded .919 ship language with diagnostic-only reporting.
- [ ] **T12 — round equivalence gate.**
  Files: `crates/engine/tests/dflash_equivalence.rs` and relevant round tests.
  Test: actual round path greedy equality on ≥3 prompts × ≥200 tokens where
  EOS permits, including ties; sampled proposal/accept/residual identity with
  controlled distributions and end-to-end evidence. Different seed consumption
  need not give identical sampled streams. Sabotage each production branch
  claimed covered; record failures before reporting coverage.

## Experimental wiring after correctness gates

- [ ] **T1 — generation loop.**
  Files: `crates/engine/src/generate.rs` and generation tests. Reuse detokenizer
  and metrics; initialize prompt taps and implement the unprocessed-anchor
  contract without a separate correction forward.
  Test: consecutive rounds, no-proposal fallback, EOS in accepted/rejected tail,
  max tokens, context boundary, cancellation/error and transactional truncation.
- [ ] **T2 — opt-in CLI.**
  Files: `crates/cli/src/{args,main,chat}.rs` and CLI tests. Expose `--dflash`
  and supported block selection with coherent initialization/sampling settings.
  Test: help, invalid flags/path/settings, supported chat or explicit rejection,
  and plain default regression. Unsupported sampling must not silently change.
- [ ] **T4 — honest metrics.**
  Files: generation/CLI metrics and an end-to-end harness. Report actual emitted
  tokens, proposals, accepted prefix, round count, complete decode wall time,
  and disjoint component times. Test: counts reconcile and hidden correction/
  head work is included. Record observed acceptance rather than requiring zero.

## Performance work after measured correctness

- [ ] **T9 — define any two-block experiment.**
  Files: `dflash.rs`, proposal tests and notes. Establish conditioning and exact
  count first: two re-anchored B16 blocks give 30 proposals. Self-draft K/V is
  distinct from target-feature context. Test positions, q conditioning, anchor
  deduplication and rollback before claiming a 31-proposal path.
- [ ] **T10 — measure trained B16 and shorter proposal prefixes.**
  Files: benchmark harness and notes. Same prompts/settings with each variant's
  actual acceptance; complete disjoint timings, including full-block draft cost
  if prefixes are truncated. B32/two-block experiments depend on T9's semantic
  design and evidence; do not assume native B32 training.
  Test: interleaved repeated runs,
  totals, prefix survival and spread. No requirement for identical alpha.
- [ ] **T15 — choose length from complete cost and prefix survival.**
  Files: proposal/generation selection policy and notes. Select measured useful
  lengths; use plain continuation when speculation has no measured gain.
  Test zero/low/high acceptance, short tails and workload changes. Bound state
  and measure selection overhead; no added scheduling layer on the hot path.
- [ ] **T11 — guarded batched head.**
  Files: `model.rs`, existing kernel path if repair is justified, head tests.
  After T17/T18 resolve offset and whole-target discrepancies, compare the head
  independently; do not replace GEMV blindly with GEMM.
  Test identical-input numerical and greedy equality, correct
  6-bpw dispatch, sabotage, then interleaved speed. No new kernel family without
  scope amendment; keep sequential head if validation fails.
- [ ] **T13 — acceptance and throughput.**
  Files: bounded acceptance harness and notes. ≥200 rounds per mode across ≥3
  prompts, accepted-prefix survival, total tokens/time, median run rates and
  spread. Test: actual correct sampled/greedy generation. Compare against a
  contemporaneous plain baseline; no fixed overlap threshold is a ship gate.
- [ ] **T14 — performance/regression/memory acceptance.**
  Files: notes and acceptance status. Test ≥30 tok/s median, M=1 ≥15.0 tok/s,
  14-token TTFT ≤350 ms and interleaved regressions; `/usr/bin/time -l` at
  1k/2k/8k, <20 GiB at 2k, record swap. A failed 30 tok/s gate remains unmet.
- [ ] **T19 — measure and design GPU-resident drafter stages.**
  Files: plan/notes first, then `dflash.rs` and existing linear/element/attention
  encoder interfaces if selected after correctness. Attribute host copies,
  allocations and per-operation waits; design bounded owned GPU scratch and
  persistent draft KV. Test graph/precision/greedy equivalence and state ownership
  before any A/B performance claim. Use existing kernel bodies; no new family
  is authorized by this performance item.

## Conditional alternatives after the shared verifier gate

- [ ] **F1a — lookup evaluation design and optional experiment.**
  Files: this spec's design first; any future Rust lookup helper/harness named
  after match selection, capacity and eviction are resolved. Use deterministic
  history continuations with point-mass q and exact target verification. Test
  no/ambiguous match, token bounds, EOS, sampled rejection/residual and sabotage.
  Measure copying/repetition and general chat separately. Implementation remains
  conditional; zero model cost is not proof of an end-to-end speedup.
- [ ] **F1b — EAGLE-3 feasibility design.**
  Files: spec/plan/notes only until selected. Review the Gemma-specific checkpoint's
  loader, feature taps, embedding, draft-vocabulary mapping, recurrence/cache and
  exact sampled policy; calculate then measure footprint in a separately
  authorized experiment. External model-card acceptance is not local evidence.
  Test/design gate: no target p truncation, exact verifier contract, compatible
  target/tokenizer and <20 GiB at 2k. A loader implementation needs an amended
  concrete file/task plan; this item does not authorize downloading or coding.

## Status

| item | state | evidence |
|---|---|---|
| A1 / T0d | documentation complete | audit notes 2026-10-13; `6d53e4c` |
| A2 | documentation complete | verifier-first draft, new anchor contract and conditional fallbacks; `697dcc2` |
| A3 | review/tests complete | `exllama_review.md` and text evidence; `59a6f4b` |
| T0a / T0b | historical observations recorded | earlier rerun notes; not repeated during this audit |
| T0c / T7 | open | oracle indexing/scaling/control/assertion flaws identified |
| T5 plumbing | implemented | `9a4cf60`; independent correctness/coverage gate open |
| T6 sweep | prior work recorded | `6622031`; reference/default gate open |
| T1a | done | code/tests `2f35201`; argmax sabotage failure in implementation notes |
| V0a / V0b | drafted, held | verifier equivalence and ceiling gate before selecting a drafter |
| V0a current test | failing evidence | M=2 max logit error 19.247559; full gate remains incomplete |
| T17 | done | offset binding, fp32 sliced API, multirow 4/6-bpw tests; `fda508d` |
| T18 | complete | projection-sign routing repaired; cross-layer path disabled by default; greedy M=2/4/8/16 pass on 3 prompts; see implementation notes/commit |
| T19 | conditional design, held | GPU stage ownership/wait overhead needs measurement after correctness |
| T1c / T15 / T16 | drafted, held | sampling policy, measured length selection, new anchor contract |
| T1b / T3 / T8 / T12 | open | correctness work before product wiring |
| T1 / T2 / T4 | open | experimental product loop absent |
| T9–T11 / T13 / T14 | open | measurements and performance acceptance outstanding |
| F1a / F1b | conditional, held | alternative evaluation requires shared verifier evidence and design decisions |

Historical 2026-10-13 follow-up (before latest authorization): user had limited
work to drafting and exllamav3 diagnostics.
Re-ran T1a's focused no-model suite: 12 passed. `scripts/gpuq status` reported
1.3 GiB available, so no model or throughput benchmark was started. T1b/C1 is
still a hard precondition: `GpuKvCache::retract` only lowers the write cursor,
while sliding-window speculative writes can overwrite live ring entries.

Latest follow-up (2026-10-13): implementation and testing authorized again, with
30 tok/s as the requested outcome. T17 completed in `fda508d`: fp16/fp32 buffer
GEMM group binds now apply `packed_offset`; fp32 has a sliced-region API; the
validator checks exact tensor byte length and that offset+length fits the buffer.
`cargo test -p kaplumba-engine --release --test linear_metal` passed 8/8 before
the final boundary assertions; the focused test then passed 1/1 with 4 rows,
4/6-bpw, fp16/fp32, and too-short/out-of-range checks. A zero-offset-binding
sabotage failed with the named rate-4 canary comparison, exit 101. The existing
`buffer_perm_mode` suite passed 3/3. Workspace `cargo fmt --check` fails on
pre-existing formatting throughout unrelated files; no workspace reformat was
applied. T18 is next; 30 tok/s remains unmeasured.

Forced-load diagnostic rerun: `dflash_alpha_probe` passed (1/1, 38.67 s) using
`Gxma-repacked`. It is diagnostic evidence only; see implementation notes for
the measured overlap and limitations. It does not satisfy T13/T14.
