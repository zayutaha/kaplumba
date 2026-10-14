# spec/mlock-clean-mmap — tasks

Every implementation step requires its meaningful tests, a demonstrated failing
negative control before any coverage claim, notes, status update and commit.

- [x] **D0 — investigate the regression and draft the spec.**
  Files: this folder's `spec.md`, `plan.md`, `tasks.md`,
  `implementation_notes.md`. Compare current `e1032d4` with preserved `d6354c5`,
  inspect the selected checkpoint's headers, and reproduce private-versus-shared
  locking in an isolated scratch probe. Confirm user scope and preserve real
  `mlock`, warning-and-continue and the normal-chat memory/I/O goal. Validate
  documentation consistency and `git diff --check`. Documentation commit:
  `5abcb0c`. This closes the drafting task only.
- [x] **T1 — restore shared backing for the real loader.**
  Files: `crates/engine/src/weights.rs`, spec notes. Change only the shard mmap
  flag to read-only `MAP_SHARED`, retain actual `mlock` and the current hot lock
  set. Test a small shard through `TextWeights::load_dedicated`, force the real
  lock path and read through its Metal no-copy view. Check successful lock
  accounting, clean page state, unchanged bytes, mapping lifetime and release.
  Exercise sub-page/final-page shards and open errors. Sabotage the actual loader
  to `MAP_PRIVATE`; require a named dirty/private-page failure, restore and
  rerun. Preserve existing weight/alignment regression results. Completed
  `2026-10-14`; implementation commit `8143e6e`.
- [x] **T2 — account for unique locked pages and partial prefixes.**
  Files: `crates/engine/src/weights.rs` and focused lock-ledger tests, notes.
  Separate tensor coverage from unique merged page bytes and preserve successful
  prefixes on failure. Test disjoint/touching/overlapping windows, EOF rounding,
  empty ranges, no lock, copied weights and arithmetic boundaries. Sabotage an
  overlap count and a partial-prefix count; require diagnostic failures before
  restoration. Do not introduce a lock cap or cold-gap locking in this step.
  Completed `2026-10-14`; implementation commit `413e01e`.
- [x] **T3 — make incomplete locking visible per load.**
  Files: `crates/engine/src/weights.rs`, focused load/failure tests, notes.
  Summarize requested/locked/evictable bytes per model and distinguish disabled,
  excluded, copied and failed paths. Keep warning-and-continue. Inject first and
  later `ENOMEM`/permission failures, verify retained successful bytes, and
  continue with exact readable fixture bytes after a partial lock. Ensure a
  second load still reports its shortfall. Sabotage warning emission and
  demonstrate a named failure. Completed `2026-10-14`; implementation commit
  `6cecdd6`.
- [ ] **T4 — prove ordinary-chat memory and I/O behavior.**
  Files: a bounded acceptance harness if required and notes. Run single-model
  `Gxma-repacked` A/B/A/B at 1k/2k/8k with generation inside the existing limit,
  plus repeated turns and idle/resume. Before each model run check `gpuq status`
  and launch with `gpuq try`; exit 75 is scheduling. Verify greedy equality,
  successful hot-set locking, pressure/Activity Monitor green, process/shared
  memory, swap deltas and attributed warm-window layer/head I/O. Record load,
  TTFT, prefill and decode medians/spread and all negative findings. Initial load
  and sparse embedding reads must not be mislabeled as repeated weight paging.
- [ ] **T5 — resolve any remaining physical-memory demand.**
  Files: spec/plan first, then named loader/allocation files only if selected,
  notes. If T4 passes, record that shared mapping with the existing hot locks is
  sufficient and that further policy changes are unnecessary. If T4 fails,
  attribute remaining demand to copies, KV/scratch, background pressure or
  incomplete locks; amend design/tasks before implementing the smallest measured
  fix. Retain actual `mlock` for the hot set; an unlocked-weight workaround
  cannot close this task. Repeat only the failed/affected acceptance cases after
  a change, with appropriate regressions and failure controls.
- [ ] **T6 — reconcile stale claims and close acceptance.**
  Files: this folder and the specific historical loader documentation whose
  claims contradict the new evidence. Mark old `MAP_PRIVATE`-is-clean-under-lock
  and footprint-only shared-residency claims as superseded with source/result
  links. Preserve historical measurements and their environments. Map evidence
  to AC1–AC5, record exact commands/commits and unresolved gates. Mark the full
  spec complete only when its actual acceptance results pass.
