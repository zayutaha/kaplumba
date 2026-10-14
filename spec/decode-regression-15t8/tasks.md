# Tasks

- [x] **T0 — create and reconcile the specification.** Files: this folder's
  four documents. Record scope, constraints, current evidence, technical
  sequencing, and acceptance. Validate document consistency and `git diff
  --check`. Completed `2026-10-14`; documentation commit hash will be recorded
  in this entry before finalizing the commit. Commit `38ba6c3`.
- [ ] **T1 — pin the matched decode baseline and causal changes.** Files: this
  folder's notes and, if needed, a temporary isolated build/worktree. Compare
  the canonical 32-token workload on the current tree and pre-T5g reference;
  interleave at least five pairs, preserve the user's dirty edits, record exact
  hashes, IDs, timings, lock/swap state, then isolate relevant T5 changes.
  Test: `scripts/gpuq status`, then `scripts/gpuq try --` for each model run;
  treat exit 75 as GPU busy. Commit evidence and note if the local machine cannot
  provide a clean matched comparison.
- [x] **T2 — add bounded deferred command-buffer submission.** Files:
  `crates/kernels/src/encoder.rs` and its tests. Add opt-in deferred rolling,
  ordered terminal submission, and safe abandonment/drop. Keep default encoder
  semantics unchanged. Test rollover, submission order, no early execution,
  error/drop, max bound, and terminal misuse; sabotage early commit and record
  the named failing assertion. `cargo test -p kaplumba-kernels --lib` passed
  36/36 before adding the bound test; `cargo test -p kaplumba-kernels deferred_
  --lib` passed 4/4 after it. An empty cache-growth queue barrier preserves
  deferred mode for subsequent rolls; terminal submit also rejects staged
  work until the caller explicitly authorizes it. Early-commit sabotage failed with
  `a rolled buffer was submitted before shared input staging` (`Committed` vs
  `NotEnqueued`). A 64-buffer cap test stalled in Metal command-buffer creation;
  reduced the explicit bound to 32, and the 32-buffer overflow case passed.
  Completed `2026-10-14`; implementation commits `87630f7`, `5c9bc02`, and
  `c4417c9` readiness guard.
- [x] **T3 — separate sparse row staging from forward encoding.** Files:
  `crates/engine/src/model.rs` and focused model tests. Remove unconditional
  steady-state drains, encode sparse pipelined work against bounded row scratch,
  and expose post-wait exact-row staging. Keep prefill/synchronous staging and
  KV growth barriers. Test exact BF16/FP16 bits, offset, duplicate/boundary IDs,
  slot identity, invalid IDs, and staging-before-submit ordering. Full engine
  library suite passed 145/145 (2 fixture-dependent ignored). Deliberately
  changed the selected row index; the exact BF16/FP16 equality test failed with
  named assertion `bf16=false`, then the sabotage was reverted. `cargo check
  -p kaplumba-engine` passed. Completed `2026-10-14`; commit hash will be
  recorded in this entry.
- [ ] **T4 — integrate stage-then-submit in chat and CLI generation.** Files:
  `crates/engine/src/chat.rs`, `crates/engine/src/generate.rs`, and relevant
  pipeline tests. Encode next step while prior GPU work runs; after prior wait,
  read token, handle EOS, stage row, then commit. Preserve exact greedy IDs and
  cache retract semantics. Test EOS at first/last positions, max-token stop,
  error cleanup, reset/multi-turn, sliding growth boundary, sampled fallback,
  and non-pipelined decoding. Run relevant integration tests.
- [ ] **T5 — prove no regressions in memory and command lifecycle.** Files:
  focused tests and this folder's notes. Demonstrate deferred buffers remain
  bounded to one step, only submit after row staging, and release on EOS/error.
  Run relevant engine/kernel suites and preserve actual T5g lock/residency
  accounting. Sabotage the lifecycle guard and record the observed failure.
- [ ] **T6 — benchmark, attribute, and reconcile claims.** Files: this folder
  and specific superseded historical notes. Run interleaved canonical decode
  acceptance (at least five pairs), plus 1k/2k/8k ordinary-chat comparisons
  when resources allow. Record throughput, TTFT, lock summary, pressure, swap,
  exact output IDs and environment. Reach >=15.8 tok/s canonical median or keep
  the task open with evidence and a next measured cause. Update old claims and
  close only demonstrated acceptance criteria.
