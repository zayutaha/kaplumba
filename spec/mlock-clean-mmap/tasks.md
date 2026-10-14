# spec/mlock-clean-mmap — tasks

Every implementation step requires its meaningful tests, a demonstrated failing
negative control before any coverage claim, notes, status update and commit.

Current result: **acceptance failed** on the forced 1,034-token candidate run.
This amendment changes documentation only. Completed T1–T3b are component work,
not a completed paging fix; all subsequent implementation remains unchecked.

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
- [x] **T3b — expose successful locks on every load.**
  Files: `crates/engine/src/weights.rs`, focused load-output tests and notes.
  Always print one compact lock/policy summary, including full success,
  without requiring `KAPLUMBA_LOAD_PROFILE`. Test complete, partial, disabled,
  excluded/copied and repeated loads. Suppress the successful summary as a
  negative control, require a named missing-summary failure, then restore.
  Independent OS wire verification remains T5/T4; do not label syscall counts
  alone as that verification. Completed `2026-10-14`; implementation commit
  `81df21a`.
- [ ] **T4 — prove ordinary-chat memory and I/O behavior.**
  Files: a bounded acceptance harness if required and notes. Run single-model
  `Gxma-repacked` A/B/A/B at 1k/2k/8k with generation inside the existing limit,
  plus repeated turns and idle/resume. Before each model run check `gpuq status`
  and launch with `gpuq try`; exit 75 is scheduling. Verify greedy equality,
  successful hot-set locking, pressure/Activity Monitor green, process/shared
  memory, swap deltas and attributed warm-window layer/head I/O. Record load,
  TTFT, prefill and decode medians/spread and all negative findings. Initial load
  and sparse embedding reads must not be mislabeled as repeated weight paging.
  **First candidate run failed AC3:** 1,034 prompt + 200 generated tokens;
  system swap 1,017.88 → sampled 2,930 MiB; model non-file regions ~724.9 MiB
  swapped. Exact full hot-set OS wiring was not captured. No valid A/B,
  2k/8k, repeated-turn or idle/resume acceptance result exists. Resume the matrix
  after T3b and T5 diagnosis/selected fixes.
- [ ] **T5 — resolve any remaining physical-memory demand.**
  Files: spec/plan first, then named loader/allocation files only if selected,
  notes. If T4 passes, record that shared mapping with the existing hot locks is
  sufficient and that further policy changes are unnecessary. If T4 fails,
  attribute remaining demand to copies, KV/scratch, background pressure or
  incomplete locks; amend design/tasks before implementing the smallest measured
  fix. Retain actual `mlock` for the hot set; an unlocked-weight workaround
  cannot close this task. Repeat only the failed/affected acceptance cases after
  a change, with appropriate regressions and failure controls.
  T4 has now failed, activating this task. Sequence small independent steps:
  first diagnostic allocation/region ownership with before/after phase snapshots;
  then full hot-set observed wire verification after GPU binding/read/idle;
  then one measured allocation reduction at a time (amend the plan with exact
  files/tests before each); finally capacity reporting and honest `gpuq` wording
  in `scripts/gpuq`. Explicitly account scratch retention, side-vector/arena
  duplicates, KV replacement peaks and whole-shard Metal residency. Preserve
  precision and all-hot real locking. Benchmark every selected reduction;
  untested candidate ideas do not become completed tasks.
- [ ] **T5a — attribute sliding KV and independently inspect hot-region wires.**
  Files: a bounded opt-in diagnostic if necessary, notes. Match live Metal cache
  addresses/lengths to VM regions and measure phase allocation; inspect every
  selected layer/head range's observed wiring. Use existing supported diagnostics
  first. Resident pages and successful syscall counts alone cannot close AC6.
  Record failure controls before claiming any new diagnostic coverage.
- [x] **T5b — allocate sliding KV on demand with safe ring migration.**
  Files: `crates/engine/src/gpu_cache.rs`, `crates/engine/src/model.rs`,
  `crates/engine/tests/kv_growth_policy.rs` and named attention/model regressions.
  Follow D2's plan: small warmup capacity, checked retained-window + chunk
  capacity, absolute-position remapping, empty tags, completed GPU references
  before migration, and bounded replacement overlap. Test unwrapped/wrapped
  growth, window/chunk boundaries, repeated wraps, reset/retract interactions,
  invalid inputs, no-op growth and queued GPU writes. Sabotage ring remapping,
  synchronization and chunk capacity separately; require diagnostic failures.
  Verify attention/greedy equality despite changed physical scan order. Completed
  `2026-10-14`; code/test commit `4e36728`.
- [x] **T5c — correct reclaimability and capacity reporting.**
  Files: `scripts/gpuq`, `scripts/gpuq_mem.sh`, `scripts/test_gpuq_mem.sh`.
  Status now labels `top`'s unused number and system-wide active pages; removed
  claims that clean weights can be evicted on demand and that a memory snapshot
  proves a model load safe. Preserved warning-and-continue, parsing threshold,
  and engine-owned lock. `bash scripts/test_gpuq_mem.sh` passed; sabotaging the
  status with “clean reclaimable pages are evicted on demand” failed at the new
  live-report assertion. Completed `2026-10-14`; implementation commit
  `d0f8cdc`.
- [x] **T5e — make sliding growth wait across pipelined encoders.**
  Files: `crates/kernels/src/encoder.rs`, `crates/engine/src/gpu_cache.rs`,
  `crates/engine/tests/kv_growth_policy.rs`. A fresh empty encoder commits and
  waits on its queue before CPU migration. Shared-event negative control failed
  in 208 ns under the old early return; both regressions pass with the fix. The
  release model answered `hi` (14 prompt + 9 generated) across multiple growth
  points with the user's GEMV kernel diff still present. Completed
  `2026-10-14`; implementation commit `30b7bb3`.
- [ ] **T5d — demonstrate selected reduction and resume T4.**
  Files: acceptance evidence and notes. A/B/A/B idle/short/1k/2k/8k workloads,
  repeated turns and idle/resume. Measure demand and replacement peaks, confirm
  hot locks persist, and verify output/throughput and paging/I/O acceptance.
  No measured allocation saving or full-spec success is claimed yet.
- [ ] **T6 — reconcile stale claims and close acceptance.**
  Files: this folder and the specific historical loader documentation whose
  claims contradict the new evidence. Mark old `MAP_PRIVATE`-is-clean-under-lock
  and footprint-only shared-residency claims as superseded with source/result
  links. Preserve historical measurements and their environments. Map evidence
  to AC1–AC6, record exact commands/commits and unresolved gates. Mark the full
  spec complete only when its actual acceptance results pass.
- [x] **D1 — investigate the forced-run failure and amend the spec.**
  Files: this folder's four documents only. Preserve failed metrics and sampled
  swap/model-region state; inspect actual lock call/lifetime and Metal ownership.
  Run a bounded native shared-mapping/Metal-read probe with observed wire counts,
  deliberate `munlock` failure and restored passing result. Record that it is
  mechanism evidence, not actual-loader/full-model coverage. Require visible
  successful summaries and allocation/capacity attribution. Validate document
  consistency and `git diff --check`. Completed `2026-10-14`; commit `2a1646c`.
- [x] **D2 — diagnose the live screenshot and amend documentation only.**
  Files: this folder's four documents. Inspect existing PID 39967 without a
  second load. Record the complete lock summary, screenshot/system accounting,
  live clean shared mappings and swapped graphics regions. Match cache-shape
  arithmetic to 100 observed 16 MiB regions, retaining pointer attribution and
  per-region wire checks as open gates. Specify sliding growth and truthful
  capacity reporting as future tasks; record numerical-order and GPU-lifetime
  risks. No implementation or code-test coverage claim. Verify diff and ensure
  this task changed only the four spec documents. Completed `2026-10-14`;
  documentation commit `b50f0d2`.
