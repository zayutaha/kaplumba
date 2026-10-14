# spec/dflash-round-30tps — tasks

**Implementation authorized.** The user's latest 2026-10-13 instruction is to
implement, fully test, and continue toward 30 tok/s. Follow the ordered gates;
do not report the throughput target as met without end-to-end measurement.
Preserve existing task IDs so historical commits remain traceable.

Execution order:
T17/T18 → V0a → V0b → verifier decision. If adequate headroom exists, T0c/T5/T6/T7/T8
establish independent drafter evidence; T1c/T1b/T16/T3 establish the new round
contract; T12's round harness precedes T1/T2/T4 and its end-to-end checks follow
CLI wiring. T10/T15/T13/T14 demonstrate real gain. T9 and F1a are conditional
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

- [x] **V0a — sequential versus batched target verification.**
  Files: `crates/engine/tests/dflash_verify_head.rs` and `model.rs` only if a
  diagnostic interface is required. Compare identical prefixes, token IDs and
  positions at M=2/4/8/16. Reset/isolate state for each arm. Test per-row greedy
  identity, justified logit tolerance, cursor accounting and ring boundaries;
  sabotage row selection and positions and record diagnosable failures.
- [x] **V0b — ideal-proposal verification cost.**
  Files: `crates/engine/tests/dflash_verify_ceiling.rs` and these notes. Saved
  target-greedy continuations; full `forward_verify` wall time at M=2/4/8/16,
  three deterministic prefixes, contexts 32/1k/2k/8k, and three interleaved
  repeats (nine samples per shape). Every proposal row matched the saved next
  target token. GPU body/head attribution was measured at M=16/short context;
  profiling serializes dispatch boundaries, so only its GPU-own component
  split is diagnostic and its profiled wall time is excluded. Both synthetic
  anchor-contract ceilings, grouped medians/spread, and the 30 tok/s remaining
  cost budget are in implementation notes. No result here is product throughput.

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

- [x] **T0c — repair the head-oracle control.**
  Files: `crates/engine/tests/dflash_alpha_probe.rs` (the historical
  `dflash_head_oracle.rs` file does not exist). `forward_with_hidden` accepts
  one-token decode rows; prefill all but the final three prompt tokens, capture
  those three nonzero hidden rows and target logits with sequential one-token
  calls, then compare full vocabulary output against `draft_logits` using the
  same target head and softcap. Assert a predeclared max-logit tolerance and
  argmax equality. Sabotage the selected hidden row and a logit scale and
  observe named numerical failures. This is a diagnostic harness fix; do not
  change production head behavior unless this comparison proves it wrong. See
  2026-10-13 notes; commit hash recorded in Status.
- [x] **T1a — sampled proposal correctness.**
  Files: `crates/engine/src/dflash.rs` and its unit tests.
  Draw candidates from q in sampled mode, retain exact q for verification,
  reject invalid/empty distributions and uniforms, keep greedy deterministic.
  Test: deterministic quantile coverage, anchor-row skip, top_k=0, zero-mass
  entries, invalid uniforms, and argmax sabotage with a named failure.
  Completed in `2f35201`; focused DFlash suite 12/12 passes.
- [x] **T1c — complete supported sampler policy.**
  Files: `dflash.rs`, generation interfaces and sampling tests. Apply the same
  declared temperature/top-k/top-p to p and q; retain q used for the draw.
  Support CLI default (1,64,.95) and greedy, reject unsupported configurations.
  Test independent probability fixtures, inclusive nucleus/ties, top-k boundaries,
  bad logits/uniforms and residual support; sabotage truncation/temperature.
  Eleven DFlash sampler tests pass. Top-p sabotage failed the named support
  assertion; restored. Commit: `f95b040`.
- [ ] **T1b — transaction and ring rollback.**
  Files: `crates/engine/src/model.rs`, `gpu_cache.rs`, `dflash.rs`, and
  cache/round tests.
- [x] **T1b.1 — preserve the attention window during chunk writes.**
  Sliding physical capacity is `window + prefill_chunk()`; attention still masks
  to the configured window. This preserves keys needed by the earliest query
  during any bounded chunk append, including verify. Gxma's default reserve
  increases by 839,065,600 bytes (0.78 GiB), from 3,524,730,880 to
  4,363,796,480 bytes. Measure the actual process footprint with
  `/usr/bin/time -l`. Test the 1024 boundary against sequential decode and
  assert key position 0 remains present after the batched write. Sabotage the
  physical capacity and confirm that named assertion fails. Baseline and
  deliberate capacity sabotage both failed with key position 0 evicted; restored
  capacity passed the sequential boundary comparison. Implementation commit
  recorded in Status.
- [ ] **T1b.2 — transactional rollback after rejection or error.**
  Rejected future rows must be excluded by absolute positions after cursor
  rollback; verify stale rows cannot leak into later outputs. Test zero/partial/
  full acceptance before/at/after ring wrap, repeated rejection, error rollback,
  and equality of retained K/V/positions with sequential decode. Sabotage position
  accounting and restoration independently.
- [x] **T1b.2a — prove rejected rows do not leak into the next decode.**
  Files: `crates/engine/tests/dflash_verify_head.rs` and implementation notes.
  Forced-load a 4-row verify, retract for zero/partial/full accepted prefixes,
  then compare correction and one further decode against sequential references
  before/at/after the sliding-ring wrap. All 9 cases passed; max next-logit delta
  0.213806 and identical greedy correction token. The helper's off-by-one
  sabotage failed the named unit assertion (14 vs 15); restored. This closes
  stale-row leakage for these cases; T1b.2 remains open for repeated rejection,
  error rollback, and explicit retained K/V/position equality. Commit: `e5a9be8`.
- [x] **T1b.2b — repeat rejection across the window boundary and check preflight rollback.**
  Files: `crates/engine/tests/dflash_verify_head.rs`, implementation notes.
  Starting before position 1024, repeat full rejection and one-token correction
  for multiple rounds across the sliding-window boundary. Compare correction
  logits and retained live K/V/positions against an independently replayed
  sequential stream. Inject a non-contiguous verify-position error before a
  valid run and prove cursor/cache state is unchanged. Sabotage rollback or
  position validation and observe a named test failure. Six repeated full
  rejections crossed position 1024; live layer-0 K/V/positions matched the
  sequential replay exactly and correction logits matched (<2.0 max delta,
  same argmax). A +1 retract sabotage failed at the cursor assertion
  (1023 vs 1022); restored test passed. Preflight error left the full layer-0
  snapshot unchanged. Commit recorded in Status. T1b.2 remains open for device
  errors after dispatch and the broader accepted-prefix/ring matrix. Commit:
  `b8e6781`.
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
- [x] **T16a — evaluate the unprocessed-anchor contract; reject current path.**
  Forced-load T12 greedy stream failed exact equality at prompt 0, token index 77
  (DFlash 99382, sequential target 84750). T19i later found clean natural M=3
  rows match sequential, but replaying the altered accepted-prefix pattern
  reproduced the flip after accumulated batched-cache K/V rounding. This
  explains the rejected trajectory but does not validate the unprocessed-anchor
  contract for general prompts. Reverted the experiment. T16 stays open pending
  an exact row-zero method. Original experiment commit: `7cebcd5`; follow-up:
  `ad37199`; spec correction: `4eef4f4`.
- [x] **T16b — retry row-zero verification with the corrected head and small blocks; reject.**
  Added a separate unprocessed-anchor API and tested its row mapping/retraction
  fixtures. The one-prompt 200-token end-to-end test failed at the same output
  index 77 for both B=16 and B=4: emitted 99382, sequential target 84750. This
  path therefore fails greedy identity even without the rejected forced-target
  proposal override from T19h. Removed the API and its integration harness;
  T16 remains open. The failure is consistent with accumulated batched-cache
  drift but does not isolate the first differing cache tensor. No performance
  result is accepted for this path.
- [x] **T5 — validate existing permutation propagation.**
  Plumbing landed in `9a4cf60`; this task adds direct setter coverage to
  `dflash_golden`: mode 0 must match the independent host reference at `fc`,
  mode 1 must diverge from it, and restoring mode 0 must reproduce the original
  output. The existing all-linears host test remains the default-layout oracle.
  Setter no-op sabotage must fail by name. T6 separately chooses the checkpoint
  default from this evidence. Commit: `f934c0d`.
- [x] **T6 — layout/default evidence.**
  Sweep work recorded in `6622031`; complete its default/reference validation.
  Its overlap-based colmajor conclusion is invalidated by the ignored head
  offset. The corrected host oracle confirms the loader's mode-0 default for all
  36 linears; mode 1 fails both the host reference and setter canary. This is
  the justified row-major layout for this checkpoint. Do not choose layouts
  from overlap. Commit: `c71a335`.
- [x] **T7 — head diagnosis after T0c.**
  Files: `crates/engine/tests/dflash_alpha_probe.rs` and notes; production
  change only if proven. Reconstruct selected 128-column head bands directly
  from EXL3 tiles and sign vectors to avoid allocating the full 5.6 GB matrix.
  Test: independent host vs target head vs DFlash head on identical nonzero
  rows, slices at the vocabulary start/interior/end, declared 0.05 tolerance,
  and path-specific indexing/scale sabotage. Results and commit hash recorded
  in notes and Status.
- [x] **T8 — aligned checkpoint semantics.**
  Files: `crates/engine/tests/dflash_alpha_probe.rs` and notes; fix any proven
  semantic mismatch as its own separately specified task. Capture all 15 masked
  draft positions, report per-depth target-top-token rank and top-k overlap on
  three distinct tokenized prompts, and retain the z-lab checkpoint's actual
  tap offset/layout candidates as diagnostics. Replace the hardcoded alpha ship
  thresholds with an explicit note that overlap is diagnostic and full-round
  throughput is not measured by this test. Test all prompt lengths and output
  row counts; sabotage the one-position target-row mapping and observe its
  named alignment fixture fail. No semantic patch was demonstrated. Evidence
  and commit hash recorded in notes and Status.
- [ ] **T12 — round equivalence gate.**
  Files: `crates/engine/tests/dflash_equivalence.rs` and relevant round tests.
  Test: actual round path greedy equality on ≥3 prompts × ≥200 tokens where
  EOS permits, including ties; sampled proposal/accept/residual identity with
  controlled distributions and end-to-end evidence. Different seed consumption
  need not give identical sampled streams. Sabotage each production branch
  claimed covered; record failures before reporting coverage.
- [x] **T12a — forced-load greedy streaming and contemporaneous baseline.**
  Files: `crates/engine/tests/dflash_equivalence.rs` and implementation notes.
  Run 200 generated tokens on each of three real prompts, carrying anchor and
  accepted target taps into the DFlash context between rounds; compare every
  output token to sequential target greedy decode. All matched exactly. DFlash
  measured 3.85–4.34 tok/s versus 12.30 tok/s plain in the same process. This
  isolates a correctness-proved but severe performance regression; T12 remains
  open for sampled equivalence and branch sabotage. Commit: `20be6d7`.

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

- [ ] **T9 — probe the re-anchored two-block B16 proposal.**
  Files: `crates/engine/src/dflash.rs`, a focused DFlash integration test,
  `spec/dflash-round-30tps/{spec,plan,tasks,implementation_notes}.md`. Preserve
  target-feature K/V as the base context. Block 1 proposes `p+1..p+15`; block 2
  uses the sampled token at `p+15` as anchor and appends only block 1's own K/V
  rows at `p..p+14`, then proposes `p+16..p+30`. Retain q distributions from
  both forwards and verify all 30 proposals in one target pass. First measure
  the 30-row verifier alone; run full generation only if correctness passes and
  measured cost leaves a plausible ≥30 tok/s path. Do not wire this into the
  default or CLI in T9. Test exact positions and proposal count, q-distribution
  conditioning, anchor deduplication, sliding-window absolute positions, and
  target-cache rollback at rejection depths 0, 1, 14, 15, 16, 29, and 30.
  Sabotage the anchor-row slice and a q-row mapping and confirm named tests fail.
  Report actual end-to-end throughput; the 30 tok/s gate remains open otherwise.
- [ ] **T10 — measure trained B16 and shorter proposal prefixes.**
  Files: benchmark harness and notes. With the proposal-temperature diagnostic
  in T10a complete, measure trained B16 and shorter prefixes with actual
  acceptance and complete disjoint timings,
  including full-block draft cost when truncating outputs. B32/two-block tests
  depend on T9's semantic design; do not assume native B32 training.
  Test interleaved repeated runs, totals, prefix survival and spread. No
  requirement for identical alpha.
- [x] **T10a — measure proposal-temperature sensitivity.**
  Files: `dflash_alpha_probe.rs`, plan and implementation notes. Sweep proposal
  q temperature at 0.5/0.75/1/1.25/1.5/2 against fixed target p at 1.0/top-k
  64 for all 15 positions on the three existing real prompts. Diagnostic only;
  no product sampler changes. The fast fixture proves q changes with temperature
  while target p remains fixed. Sabotage the temperature divisor and observe
  the named assertion fail. Result: no stable cross-prompt temperature winner;
  see implementation notes. Commit: `70f9578`.
- [ ] **T15 — choose length from complete cost and prefix survival.**
  Files: proposal/generation selection policy and notes. Select measured useful
  lengths; use plain continuation when speculation has no measured gain.
  Test zero/low/high acceptance, short tails and workload changes. Bound state
  and measure selection overhead; no added scheduling layer on the hot path.
- [x] **T11 — guarded batched head.**
  Files: `crates/engine/src/model.rs`, `crates/engine/tests/dflash_head_ab.rs`,
  `crates/engine/tests/dflash_verify_head.rs`, and these notes. After T17 fixed
  packed offsets, route all-row verify through the existing `encode_gemm` path
  with contiguous fixed-size logits scratch. The M=1/2/4/8/16 test across three
  prompts matched each sequential greedy token; maximum logit difference was
  0.4375. A zero-offset sabotage failed at M=1, row 0 (greedy 14697 vs 236777),
  then the restored route passed. The every-row signal check passed. A short
  B-A-B M=16 timing run (three prompts × three repeats) measured medians
  368.240 / 376.376 / 367.591 ms for batched / rowwise / batched, a repeatable
  ~8.5 ms (~2.3%) verifier reduction. The complete V0b matrix also passed all
  144 rows and reduced the 8k M=16 median from historical 461.548 to 452.821 ms;
  do not treat this separate-run difference as the paired timing result. No new
  kernel family was added. 30 tok/s remains unmet. Commit recorded in Status.
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
- [x] **T19a — split the current processed-anchor round wall time.**
  Files: `dflash_equivalence.rs`, implementation notes. Record anchor target
  forward plus tap read, `draft_and_verify` (draft + target verify), DFlash
  context append, and total generation on the same 3 × 200 greedy prompts.
  Keep instrumentation opt-in/test-only. Compare sums against elapsed wall time;
  this localizes T19 before GPU-resident changes. Test output remains identical.
  Completed: 3 × 200 greedy tokens exact; commit `389dd88`.
- [x] **T19b — separate DFlash proposal, target verify, and acceptance time.**
  Files: `crates/engine/src/dflash.rs`, `crates/engine/tests/dflash_equivalence.rs`,
  implementation notes. Add temporary opt-in diagnostic timings around
  `draft_block`, proposal selection, `forward_verify`, and verify acceptance;
  report the aggregate across the same exact 3 × 200 run. Remove diagnostic
  production hooks after measurement unless T19 explicitly adopts a stable
  profiler interface. Test: all output IDs remain greedy-identical and the
  phase sum agrees with the existing draft-and-verify bucket within timer noise.
  Completed: 229 rounds averaged 170 ms draft + 374 ms verify; all 600 output
  tokens exact. Temporary timing hook removed; commit `845fb32`.
- [x] **T19c — measure shorter verified proposal blocks.**
  Files: `crates/engine/tests/dflash_equivalence.rs`, implementation notes.
  Add test-only block-size and prompt-count controls, then run exact 200-token
  greedy generation at B=2/4/8/16 on the same prompt(s). Record accepted tokens,
  total DFlash and plain time, and ensure every generated token matches the
  sequential target. Select no runtime default from a single noisy reading.
  Completed: B=4 verified on 3 × 200 tokens; B=2 repeated three times; B=4
  repeated three times; B=8 and B=16 compared. Result in notes; no runtime
  default changed. Implementation commit: `0d9c1a6`.
- [x] **T19d — add encoder APIs for existing f32 residual operations.**
  Files: `crates/kernels/src/elements.rs`, `crates/kernels/tests/` focused
  element tests. Add encoder-only `rms_norm_f32_in` and `residual_add_f32`
  wrappers using the current pipelines, with validation matching slice APIs.
  Compare encoder results to slice outputs at 1 and multiple rows, including
  large residual values above fp16 range. No new Metal shader. Passed focused
  encoder test and binding sabotage; commit `6e7f582`.
- [x] **T19e — add an encoder API for gated SiLU.**
  Files: `crates/kernels/src/elements.rs`, `crates/kernels/tests/dflash_encode.rs`.
  Encode the current gated-SiLU pipeline into caller-owned buffers; test mixed
  positive/negative inputs, parity against the slice method, and invalid count
  and buffer sizes. Passed parity and wrong-binding sabotage; commit `1715925`.
- [x] **T19e.1 — add an offset-aware encoder API for f32-output GEMM.**
  Files: `crates/kernels/src/encoder.rs`, `crates/kernels/src/linear.rs`,
  focused linear tests. Encode the existing fp32-output pipeline into a caller
  encoder while binding exact packed offset/length; test nonzero offsets, 4/6
  bpw, row counts 1 and 4, slice parity, and invalid sizing. Passed parity and
  offset-binding sabotage; commit `5a68534`.
- [x] **T19f — encode DFlash block attention into a caller encoder.**
  Files: `crates/kernels/src/attention.rs`, focused kernel tests. Preserve the
  current DFlash attention shader and mask/window semantics; add a buffer-based
  request and encode method. Compare against `dflash_block_attention` for
  bidirectional and sliding modes, empty context, and block/context boundaries.
  Exact parity and wrong-context-value-binding sabotage passed; commit
  `f1fd6e0`.
- [x] **T19g.1 — add owned fixed-shape DFlash GPU buffers and cached weight views.**
  Files: `crates/engine/src/dflash.rs`. Allocate reusable ping-pong activation,
  projection, attention, MLP and linear scratch from block dimensions; cache
  norm/sign Metal buffers beside the drafter. Test sizing/shape validation and
  ensure no scratch size depends on context length. Keep existing loading and
  mmap/mlock ownership unchanged. Real checkpoint allocated 5.37 MiB scratch
  and 1.04 MiB cached signs/norms; sizing sabotage failed as expected; commit
  `84c126c`.
- [x] **T19g.2 — encode `forward_block` on one bounded encoder.**
  Files: `crates/engine/src/dflash.rs`, DFlash correctness tests. Encode all five
  layers using the T19d–T19f APIs and existing f16 linear/RoPE paths. Preserve
  f32 residuals and exact block masks/positions. Compare hidden/K/V block output
  against the slice implementation on empty, normal, and sliding contexts
  (relative max error <0.5%); sabotage one buffer binding and require failure.
  All three contexts passed; the context-V binding sabotage failed on normal
  context (V relative error 0.584% vs the 0.5% bound). Fixed scratch is now
  5,980,160 bytes (5.70 MiB), including the third f32 residual buffer. Commit
  commit `529f1af`.
- [x] **T19g.3 — route draft generation through the encoded block and measure.**
  Files: `crates/engine/src/dflash.rs`, `dflash_equivalence.rs`, notes. Preserve
  the slice path as a test oracle; compare draft logits/argmax and exact target
  generation, then run 3 × 200 and interleaved A/B plus footprint/swap. Adopt
  the encoded path only if exactness, memory, and complete-round speed all pass.
  Early prompt-0 measurements: B=4 exact at 7.05 tok/s encoded vs 6.92 tok/s
  slice; B=16 exact at 4.17 tok/s encoded. Full 3 × 200 and footprint/swap
  measurements remain. Do not make a speedup claim until swap pressure is
  rechecked: `gpuq status` had been reporting 0 MiB due to a parser bug.
  Updated three-prompt runs: B=4 encoded 7.08/7.01/7.16 tok/s (median 7.08),
  slice 6.92/6.89/7.05 (median 6.92); exact on all 600 tokens. `time -l`
  measured 14.35 GiB footprint and zero swapouts for the encoded run. These
  grouped A/B runs alone were not enough to call a stable 2.3% gain. B=16
  encoded now passed all three prompts at 4.16/4.06/4.58
  tok/s (median 4.16; rounds 78/80/71; accepted 127/123/128), exact on all
  600 tokens. Prompt-0 interleaved A/B/A/B/A/B measured encoded
  7.07/7.05/7.08 and slice 6.96/6.94/6.95 tok/s; all six runs were exact.
  Encoded median gain is ~1.7%, within practical noise and not enough to change
  the default. Full B=4 footprint was 13.37 GiB with zero swapouts. The encoded
  route remains opt-in; commits `5b7981e`, `50c393e`.
- [x] **T19g.4 — test two chained blocks and reject if slower.**
  Files: `crates/engine/src/dflash.rs`, this spec's plan/tasks/notes.
  One prompt, B=4: exact greedy output, 74 rounds, 129 accepted, 5.43 tok/s
  versus 7.05 tok/s for one encoded block. Removed the second-block branch;
  cleanup committed as `5b7981e`.
  speculative K/V chaining adds enough draft work to erase the verify savings.
- [x] **T19g.5 — repair GPUQ swap telemetry.**
  Files: `scripts/gpuq`, `scripts/gpuq_mem.sh`, `scripts/test_gpuq_mem.sh`,
  this spec's notes. Parse the named `used` value and convert K/M/G units.
  Before the fix, `sysctl -n vm.swapusage` showed about 3.2 GiB used while
  `gpuq status` incorrectly showed 0 MiB. Parser tests pass for M/G and missing
  fields fail closed. Coverage sabotage read `total` and failed all five swap
  assertions; the restored parser passed the full script. Commit `161b9f2`.
- [x] **T19g.6 — measure GPU tied-head encoding and reject if slower.**
  Files: `crates/engine/src/dflash.rs`, this spec's plan/tasks/notes.
  One prompt, B=4, exact output: encoded block + encoded head 6.69 tok/s;
  encoded head with slice block 6.54 tok/s; single encoded block 7.05 tok/s;
  slice block 6.92 tok/s. Removed the head path; separate command submission,
  staging, and synchronization outweigh its compute savings. Cleanup commit
  `5b7981e`.
- [x] **T19g.7 — attribute the current B=4 encoded round by synchronized phase.**
  Files: `crates/engine/src/dflash.rs`, this spec's plan/tasks/notes.
  Temporarily measure encoded block body, tied head, target `forward_verify`,
  and acceptance/retraction separately under an opt-in diagnostic. Run exact
  greedy 3 × 200 prompts and compare aggregate phase totals to the existing
  round timing. Removed the runtime hook after measurement. The 3 × 200 test
  stayed exact; phase medians and footprint are in implementation notes.
  Commit `50c393e`.
- [x] **T19h — test reusing the known greedy target token as the first proposal; reject.**
  Files: `crates/engine/src/dflash.rs`, `crates/engine/tests/dflash_equivalence.rs`,
  this spec's plan/tasks/notes. Coverage assertion failed under sabotage at
  prompt 0, round 7. With the override enabled, exact greedy output failed at
  prompt 0, token 77 (emitted 99382 vs target 84750). Removed the override; do
  not retain it until the verifier discrepancy is fixed.
- [x] **T19i — isolate the output-77 verifier drift from the rejected override.**
  Files: `crates/engine/tests/dflash_verify_head.rs`, this spec's notes. On the
  natural sequential prefix, M=3 rows 75–77 match sequential greedy, including
  the close output-77 row (gap 0.015625; max logit delta 0.046875); retracted
  continuation and repeated M=3 one-accepted rounds match through output 79.
  Replaying the accepted-prefix pattern created only by T19h's rejected
  target-token override reproduces the output-77 flip and exposes accumulated
  batch-cache rounding drift. A +1 verify-position sabotage fails before
  comparison with `positions must begin at the current cache position`; the
  restored test passes. The cache drift explains the experimental failure but
  does not identify a defect in the unmodified DFlash stream, which remains
  greedy-exact on its measured prompts. Captured taps were not causal to this
  investigation; no target-model change was justified. Do not retain T19h's
  override or the diagnostic assertion that expects the altered path to drift.
  Implementation/test commit: `ad37199`.
- [x] **T20 — load DFlash packed weights through base mmap + mlock path.**
  Files: `crates/engine/src/dflash.rs`, `crates/engine/tests/dflash_load.rs`,
  this spec's plan/tasks/notes. Replace the DFlash loader's per-tensor file reads
  and packed copies with `TextWeights::load_dedicated`, use exact zero-copy
  tensor offsets/lengths, and retain the `TextWeights` owner in `DflashWeights`.
  Keep the target model loader unchanged. Test a DFlash checkpoint load, tensor
  byte/shape parity, pinned-byte reporting, and both default mmap and explicit
  `KAPLUMBA_RESIDENT_WEIGHTS` arena paths. Verified the unaligned offset
  fallback separately. `cargo check -p kaplumba-engine` and all three focused
  loader tests passed. The unaligned test failed with a temporary bypass of
  `align_packed_view` (offset 409 instead of aligned offset 0), then passed when
  restored. Implementation commit: `6a5d36d`.

- [x] **T20.1 — bind each mapped DFlash projection to its exact packed slice.**
  Files: `crates/engine/src/dflash.rs`, DFlash mapped-weight integration tests,
  and this spec's plan/tasks/notes. The mapped Metal buffer spans the shard, so
  fp16 and fp32 GEMM must pass `packed_len * sizeof(u16)` as the tensor slice
  length alongside `metal_offset`. Test every mapped drafter linear against
  independent host decode and the full block/context path; temporarily restore
  whole-buffer length inference and observe a diagnosable failure, then restore
  the exact-slice calls. Keep target loading and mmap ownership unchanged. All
  36 linears and full context block passed; sabotage failed with `packed buffer
  length 768973144 does not match 86704128`. Implementation commit recorded in
  the Status table.

- [x] **T20.2 — infer integer EXL3 rate from DFlash trellis shape.**
  Files: `crates/engine/src/dflash.rs`,
  `crates/engine/tests/dflash_load.rs`,
  `crates/engine/tests/dflash_golden.rs`, notes. Replace the hardcoded
  4-bpw/64-word assumption in the loader and independent host decoder with shape-based integer
  rate inference for supported EXL3 integer rates; retain exact byte-count,
  codebook and mapping checks. Generalize the real-shard checks for 4/6 bpw and
  add rate-table plus malformed/fractional rejection coverage. Sabotage the
  inferred rate back to 4 bpw and require the 6-bpw fixture/candidate loader to
  fail diagnostically. Keep the local drafter and runtime defaults unchanged.
  Loader infers integer rates 1–8; six focused unit tests pass and fractional/
  malformed shapes reject with tensor names. Fixed-4 inference sabotage failed
  the 6-bpw unit assertion (`left: 4, right: 6`). The aligned candidate device
  load mapped 1,151,926,272 bytes and mlocked 1,152,516,096. Candidate encoded
  block passed (normal K/V/hidden relative errors below 0.1%; encoded median
  70.41 ms vs 93.55 ms slice). Generalized `HostLinear` to infer rate from the
  trellis shape; its 6-bpw all-36-linears comparison passed (worst absolute
  difference 0.002741), as did the existing 4-bpw shard (0.003663). Fixed-4
  host-oracle sabotage failed on candidate `layers.0.self_attn.q_proj` with
  16,515,072 words found vs 11,010,048 expected. Candidate target-stream and
  throughput checks remain F1d, not T20.2.

## Conditional alternatives after the shared verifier gate

- [ ] **F1a — lookup evaluation design and optional experiment.**
  Files: this spec's design first; any future Rust lookup helper/harness named
  after match selection, capacity and eviction are resolved. Use deterministic
  history continuations with point-mass q and exact target verification. Test
  no/ambiguous match, token bounds, EOS, sampled rejection/residual and sabotage.
  Measure copying/repetition and general chat separately. Implementation remains
  conditional; zero model cost is not proof of an end-to-end speedup.
- [x] **F1b — complete EAGLE-3 feasibility design; hold implementation.**
  Files: this spec's spec/plan/notes only. Reviewed
  `thoughtworks/Gemma-4-31B-Eagle3` without downloading weights; fetched config
  and only the first 65,536 Safetensors bytes. Same 5376 hidden width, taps
  `[2,29,56]`, one dense BF16 decoder layer, `fc.weight [5376,16128]`, reduced
  32k `lm_head`, `d2t`/`t2d` maps, and 4096 position limit. The 1,296,995,832-byte
  shard implies an estimated <20 GB short-context footprint, but no load or 2k
  memory measurement exists. Exact sampled decoding can preserve full target p
  only by mapping q's 32k support back to the full 262144 target vocabulary and
  using the exact verifier/residual; no target truncation is allowed. Current
  engine has no dense BF16 GEMM for its draft matrices. The publisher's 1.05–1.72×
  H200 results are external; even 1.72× applied illustratively to local 12.3 tok/s
  is ~21.2 tok/s, not evidence for 30. A compatible loader, kernels, mapping and
  target-feature parity would be a substantial new implementation with no
  demonstrated path to AC5, so weights/code remain held. Details and source links
  are in implementation notes/plan; feasibility review commit `a926319`.
- [x] **F1c — review same-target DFlash checkpoint compatibility.**
  Files: this spec's spec/plan/tasks/notes. Compare the public RedHatAI
  Gemma-4-31B DFlash config and tensor header with the existing loader without
  downloading checkpoint weights. Record expected acceptance, file size, tensor
  dtype/layout, config-schema and attention-shape compatibility. Result:
  rejected for the current engine path: the public 5.08 GB checkpoint is dense
  BF16, while `DflashWeights::load` requires EXL3 `.trellis`/`.mcg`/scale tensors;
  its `aux_hidden_state_layer_ids` and nested transformer config also do not
  satisfy `DflashConfig::load`. Do not spend 5 GB on an artifact the loader
  cannot consume. Reopen only with a separately scoped BF16/quantized format
  design and memory estimate. Public acceptance (3.98–6.62 tokens on listed
  subsets) is external and does not show 30 tok/s here.

- [x] **F1d — evaluate a compatible same-target 6-bpw EXL3 DFlash candidate.**
  Files: `crates/engine/tests/dflash_golden.rs`, this spec's notes and
  acceptance status. Candidate:
  `eepos/gemma-4-31B-it-DFlash-exl3-6.0bpw`, repo commit
  `48be48d8035ec6c94d6b9b3f867c7dae1e4dc627`, 1,153,033,071-byte Safetensors,
  Apache-2.0 metadata. Header-only review found 166 recognized names, identical
  key sets and architecture/tap config; the third trellis dimension is 96
  (=6 bpw) instead of the local checkpoint's 64 (=4 bpw). The owner says the
  conversion was only briefly tested, so treat it as experimental. Its source
  header is 16,599 bytes; all packed starts are 3 mod 4 and currently trigger
  36 alignment copies. Normalize one JSON-padding byte to the canonical 8-byte
  header boundary in the isolated model copy, then verify the 1.15 GB payload is
  unchanged and packed offsets map without copies. After T20.2, verify all 36
  lines and exact greedy output on the existing three 200-token prompts, then
  compare repeated same-prompt timings, footprint and swap to the current
  drafter. Candidate mapped load, block implementation, and all-36-linears host
  checks pass under T20.2. Three 200-token B16 streams were greedy-exact at
  3.72–4.04 tok/s (median 3.92); B4 slice was exact at 5.86–6.21 (median 6.05),
  and B4 encoded was exact at 6.09–6.49 (median 6.31). B2 and B8 prompt-0
  controls were exact at 5.27 and 5.63 tok/s. Prompt-0 B4 encoded A/B/A/B
  measured 4-bpw 6.23/6.24 vs 6-bpw 6.50/6.51 tok/s, a repeatable 4.3% gain
  on that prompt, while remaining far below 30. Peak `phys_footprint` was
  14.44–14.53 GB on these short prompts; process swapouts were zero. With the
  target loaded, the target pinned 11.22 GiB, then DFlash mlock failed on its
  first 86.7 MB (4-bpw) or 130.1 MB (6-bpw) range; no drafter pinned-byte
  profile was reported. Do not promote the larger candidate from this modest,
  prompt-local gain. The false-green missing-path harness was fixed and tested;
  nonexistent explicit path failed and actual candidate reached all 36 checks;
  `eb3771c`. F1d evaluation is complete; DFlash mlock budgeting and the 30 tok/s
  gate remain open. Candidate B16/B4 stream evidence is in implementation notes;
  results recorded in `implementation_notes.md` at `2601faf`.

- [x] **F1e — evaluate the official same-target 5-bpw EXL3 DFlash quant.**
  Files: `crates/engine/tests/dflash_golden.rs`, this spec's
  implementation notes and acceptance status. Candidate:
  `turboderp/gemma4-31b-it-DFlash-exl3`, branch `5.00bpw`, shard size
  961,045,318 bytes, expected LFS SHA-256
  `5ae9ffe6f4c6250d7f08f0265b1aaff17bca6645b0879097e24aa447bd2f95ca`.
  Keep it under `~/.kaplumba/models/Drafter-turboderp-5bpw`; do not change the
  default. Check downloaded hash and config/taps/rates, mmap alignment, and
  mlock profile; run all 36 host linears; run exact 3 × 200 greedy streams at
  B=4/8/16 with `--test-threads=1`; compare B=4/8/16 to the existing drafter
  with repeated same-prompt A/B/A/B; record `phys_footprint` and swap. Drop it
  without promotion if exactness, stable gain or memory acceptance fail. All
  36 host linears passed (worst abs error 0.002120); exact B4/B8/B16 medians
  were 6.00/5.53/3.91 tok/s. Prompt-0 encoded B4 A/B/A/B was 4-bpw
  6.22/6.24 vs candidate 5-bpw 6.46/6.46; a small +3.5% prompt-local gain.
  Target pinned 11.22 GiB but drafter mlock failed; process footprint was
  14.53–14.55 GB with zero swapouts and block input operations. Do not promote;
  the candidate does not solve 30 tok/s. Full measurements are in notes at
  `bddf913`.

## Status

| item | state | evidence |
|---|---|---|
| A1 / T0d | documentation complete | audit notes 2026-10-13; `6d53e4c` |
| A2 | documentation complete | verifier-first draft, new anchor contract and conditional fallbacks; `697dcc2` |
| A3 | review/tests complete | `exllama_review.md` and text evidence; `59a6f4b` |
| T0a / T0b | historical observations recorded | earlier rerun notes; not repeated during this audit |
| T0c | done | 3 nonzero rows; full-vocab max error ≤0.0156; zero-row and ×2 scale sabotage fail; `00ae90b` |
| T7 | done | 45 host/target/DFlash bands; worst error 0.0088; tile-index and ×2 scale sabotage fail; `8863b83` |
| T5 | done | setter mode 1 diverges from host; mode 0 restores bit-exact; no-op setter sabotage fails; `f934c0d` |
| T6 | done | all 36 actual buffer linears match independent mode-0 host decode; wrong-mode controls fail; `c71a335` |
| T1a | done | code/tests `2f35201`; argmax sabotage failure in implementation notes |
| V0a | done | 3 prompts × M=1/2/4/8/16 greedy identity; tap/KV and ring boundary parity; row and query-position sabotage both fail; see implementation notes |
| V0b | done | 3 prefixes × 4 contexts × M=2/4/8/16 × 3 repeats; greedy identity; `00f8c38` |
| T17 | done | offset binding, fp32 sliced API, multirow 4/6-bpw tests; `fda508d` |
| T18 | complete | projection-sign routing repaired; cross-layer path disabled by default; greedy M=2/4/8/16 pass on 3 prompts; see implementation notes/commit |
| T10a | done | six q temperatures × 15 rows × 3 prompts; no stable winner; temperature sabotage fails; `70f9578` |
| T1c | done | Shared temp/top-k/top-p policy for p/q; 11 sampler tests and top-p sabotage; `f95b040` |
| T1b.2a | done | 9 cache rollback cases across ring boundary; off-by-one sabotage fails; no stale-row decode divergence; `e5a9be8` |
| T1b.2b | done | six repeated full rejections across position 1024; live layer-0 K/V/positions exact to sequential; preflight failure atomic; retract sabotage fails; `b8e6781` |
| T19h / T19i | override rejected / follow-up complete | forced proposal reached a sensitive altered trajectory; natural M=3 rows and repeated rollback match; position sabotage fails; `ad37199` |
| T12a | done | 3 × 200 greedy tokens match exactly; DFlash 3.85–4.34 tok/s vs plain 12.30; `20be6d7` |
| T16a | experiment rejected | altered acceptance path flips at token 77 from accumulated batch-cache drift; exact row-zero contract remains open; `7cebcd5`, `ad37199`, `4eef4f4` |
| T19 | conditional design, held | GPU stage ownership/wait overhead needs measurement after correctness |
| T1b.1 | done | 2048-slot sliding ring; boundary greedy identity, max logit delta 0.136719, 13.59 GiB phys_footprint, no swap; sabotage fails; `6e69bf7` |
| T20.1 | done | exact fp16/fp32 mapped slices; host and full-context tests plus sabotage; `880b7e4` |
| T20.2 | done | 1–8 bpw table/rejections, mapped candidate, encoded block, 4/6-bpw all-linears host comparisons and fixed-4 sabotage; `aaf4120` |
| T1c / T15 / T16 | drafted, held | sampling policy, measured length selection, new anchor contract |
| T8 | done | 3 prompts × 15 depths; row mapping fixture and shift sabotage; overlap is diagnostic only; `c2e63c9` |
| T1b / T3 / T12 | open | correctness work before product wiring |
| T1 / T2 / T4 | open | experimental product loop absent |
| T9–T11 / T13 / T14 | open | measurements and performance acceptance outstanding |
| F1a | conditional, held | lookup remains workload-dependent; not selected |
| F1b | feasibility complete; implementation held | Thoughtworks EAGLE3 needs dense BF16 kernels and 32k→262k mapping; no measured route to 30 tok/s |
| F1c | done, candidate rejected | public same-target model uses dense BF16 linears and incompatible config; header-only review, no weights downloaded |
| F1d | evaluated; not promoted | 3×200 exact, B4 prompt-0 A/B/A/B +4.3%, but only 6.31 tok/s median and drafter mlock fails with target resident; see notes |
| F1e | evaluated; not promoted | all 3×200 exact at B4/B8/B16; best median 6.00 tok/s, only +3.5% prompt-0 encoded B4 gain; target-resident drafter mlock fails; notes `bddf913` |
| T11 | done | all-row greedy identity, offset sabotage, and short-context B-A-B timing; `aaa010c` |
| T16b | experiment rejected | unprocessed anchor B=4 and B=16 both diverge at output 77 (99382 vs 84750); removed; `cb9de00` |

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
