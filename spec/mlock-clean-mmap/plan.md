# spec/mlock-clean-mmap — plan

## Architecture and data flow

Amendment status (2026-10-14): T1–T3 do not satisfy full acceptance. The forced
1,034-token candidate run grew swap and had swapped non-file regions in the
model process. Keep the shared mapping fix, but require T3b diagnostics and T5
allocation/capacity attribution before selecting further product changes.

Preserve the existing path:

```text
safetensors header/selection
  → read-only shared shard mmap
  → merge selected packed-tensor page ranges
  → lock chosen ranges and retain an exact success ledger
  → Metal no-copy shard buffer + tensor offsets
  → ordinary chat forward / KV / scratch
  → retire GPU buffers, then unmap/release backing storage
```

Small side tensors retain their current arena copies. The embedding remains a
sparse mapped lookup. Word-misaligned packed tensors retain the existing aligned
Metal copy; their source pages must not be reported as the live locked weights.
`Gxma-repacked` has no such packed tensors, so the primary fix is the mapping
flag rather than an alignment format change.

Apply the narrow mapping fix first while retaining current locking selection.
Retain locking on the current hot set, including the head. The historical
`MAP_SHARED` line is relevant; historical budgets assume a different context
policy and their comments conflate shared wiring with process footprint. They
must not be ported as current physical-memory facts.

## File/module touchpoints

| Path | Current behavior | Planned change |
|---|---|---|
| `crates/engine/src/weights.rs` (`ShardReader`) | Read-only `MAP_PRIVATE`; whole shard wrapped without copying | Read-only `MAP_SHARED`; preserve buffer offset/lifetime and error handling |
| `crates/engine/src/weights.rs` (`pin_trellis_pages`) | All aligned mapped trellis selected; `pinned_bytes` sums fully covered tensor windows | Record successful unique wired page bytes separately from source tensor bytes; include any successful prefix of a tensor |
| `crates/engine/src/weights.rs` (`LlamaMlock`) | Incremental locks; latched failure; D0 baseline warned once per process | Keep soft failure semantics; make caller summary accurate and warn per model; retain successful prefix |
| `crates/engine/src/model.rs` (`load_from_dir`, `finish_construction`) | Normal target loads through `load_dedicated`, then builds layers/caches, residency and warmup | Measure each phase's live allocation and external/shared demand; diagnose capacity without reducing the required hot lock set |
| `crates/engine/src/model.rs:6299` | Misaligned view copied into shared Metal buffer | Compatibility checks only; no new copy path or buffer-pointer lock |
| `crates/engine/src/dflash.rs:131` | Uses the same `TextWeights` loader | Loader regression only; no generation/performance changes |
| `crates/engine/src/weights.rs` test module | Existing loader tests do not prove actual locked shard pages stay clean | Small shard fixture using `TextWeights::load_dedicated` and a no-copy Metal view; OS page-state assertions and private-map sabotage. Keep the test beside private `ShardReader` state rather than widening production APIs for an integration test. |
| `crates/engine/tests/probe_llama_mlock.rs`, weight/alignment tests | Incremental locking and layout checks | Extend meaningful partial/failure/unique-page assertions; preserve existing behavior |
| This spec's notes and an acceptance harness if needed | No matched full-chat evidence for the proposed fix | Record workload, pressure, I/O, regions, memory and interleaved timing evidence |
| `crates/engine/src/weights.rs` (`pin_trellis_pages`) | Successful summary only appears with `KAPLUMBA_LOAD_PROFILE` | T3b prints one summary for every load, including complete success; keep details out of the generation loop |
| `crates/engine/src/weights.rs` (`ShardReader::metal_buffer`, `create_residency_set`) | Whole shards, including cold ranges, become Metal buffers and residency-set members; POSIX locks select only hot ranges | Measure actual physical/residency effects of the cold ranges. Treat narrowed buffer/residency scopes as a measured candidate, not an assumed fix |
| `crates/engine/src/model.rs:5466` (`ensure_scratch`) | Scratch grows for prefill and remains at its largest row capacity during decode/idle | Account retained and replacement-overlap bytes; evaluate smaller chunks or phase-specific lifetime only if measurement selects them |
| `crates/engine/src/model.rs:6351` (`load_quantized_linear`) and `gpu_cache.rs` | Signs exist in source arena, CPU vectors and uploaded buffers; KV growth holds old/new buffers transiently | Measure duplicate side allocations and KV growth peaks before selecting a reduction; preserve precision/layout/output |

## Interfaces and accounting

Keep public generation/CLI interfaces and checkpoint formats unchanged. Mapping
is read-only, so `MAP_SHARED` does not authorize writes to safetensors. Model
files must remain immutable during a session; backing mappings outlive all GPU
views. Do not release or invalidate mappings while command buffers use them.

Plan each shard's candidate tensor windows by outward rounding to the actual OS
page size, clamping the final mapping page correctly, sorting and merging
overlaps. Never span a cold gap to simplify locking. Accounting uses interval
unions; per-tensor status is a separate view and must not be summed into unique
physical bytes. A failed incremental grow leaves a successful prefix which must
appear in the ledger even if the last tensor is only partly covered.

The load summary records eligible, selected, attempted, locked, failed/evictable,
disabled and excluded bytes, plus mapped versus copied ownership. Existing
`KAPLUMBA_NO_PIN_TRELLIS` behavior remains explicit. Loading continues on lock
failure with the reason and a degraded-residency summary. Do not imply that an
excluded head is cold when every decode token reads it.

T3b makes the success summary unconditional. A successful syscall ledger is
distinct from observed `user_wired_count`: the latter must be checked against
all selected layer/head mappings after Metal construction and at later phases.
Use an opt-in supported native/OS diagnostic, never a guessed Rust struct offset.
The current small probe confirms that Metal no-copy wrapping and residency do
not automatically undo a user lock, but cannot substitute for the full hot set.

Real `mlock` remains required for the aligned mapped hot set, including the
head. Record actual physical demand, context reserve and desktop supply; clean
shared wired pages can be outside process footprint. Insufficient capacity warns
and continues per the user and is reported as degraded. It cannot pass healthy
acceptance. Do not add hot-path pressure polling or speculative headroom
constants from old checkpoints. If shared mapping alone still misses pressure
acceptance, T5 diagnoses the remaining demand and defines any further change
before implementing it; turning off locking is not a candidate fix.

## Measurement and validation sequence

1. Record current source/commit, model headers, OS, process list, `gpuq status`,
   pressure level and swap baseline. The old tree is comparison material only.
2. Test the actual loader on a small valid shard, force locking, observe region
   `user_wired_count`, `external_pager`, private pages and `mincore` copied/
   anonymous flags, and execute a real read-only Metal operation. Fixture bytes
   must be checked afterward. Deliberately use `MAP_PRIVATE` in the loader and
   require the clean-page assertion to fail; restore and rerun.
3. Add and sabotage unique-page and failure-summary checks. Force syscall errors
   through a test seam rather than relying on the desktop running out of RAM.
4. With one model process, interleave current/candidate A/B/A/B using identical
   checkpoint, prompts, contexts, sampler and output count. Use isolated target
   directories if separate worktrees build the binaries. Record both initial
   load/prefill and warmed decode windows, plus multi-turn and idle/resume cases.
   Use 1,024/2,048/7,992-token prefixes with 200 generated tokens; the last case
   reaches the agreed 8,192-token acceptance workload. The current engine default
   is 32,768 (`generate.rs:35`), not the 8,192 limit incorrectly stated in D0.
   No context-policy change is authorized here. Measure tokenized lengths rather
   than inferring them from repetitions; the first nominal 1k prompt was 1,034
   actual tokens. Report prefix and total separately.
5. Before each full model run, use `scripts/gpuq status`; launch through
   `scripts/gpuq try --`, treating exit 75 as scheduling. Pause a run plan when
   memory readings show an unsuitable baseline rather than mixing conditions.
6. Capture `/usr/bin/time -l`, `footprint`/`vmmap` region observations, verified
   pressure notifications or `kern.memorystatus_vm_pressure_level`, process I/O
   and host paging/swap deltas. Global counters are background-sensitive; pair
   them with per-process/region evidence. Pressure level 1 on the investigated
   OS is a baseline observation, not an assumed cross-version interface.
7. If process disk I/O is nonzero in warm decode, distinguish sparse embedding
   misses from repeated layer/head reads using file/region attribution. If the
   available instrumentation cannot distinguish them, leave the I/O gate unmet.
   Do not infer SSD traffic from ordinary minor page faults or stale swap.
8. If corrected all-hot locking misses pressure acceptance, identify remaining
   duplicate allocations, KV/scratch demand and incomplete locks. Record
   negative results and amend the design before adding fixes. Keep real locks
   as a requirement; do not pass the gate by making hot weights evictable.

### Required follow-up after the forced-run failure

1. Preserve the failed run as AC3 failure. Capture complete lock summaries on
   all subsequent diagnostic runs; implement T3b before declaring them normal
   acceptance. Profile flags are acceptable for diagnosis until T3b exists.
2. Record baseline, arena-copy phase, successful locks, Metal binding/residency,
   warmup, prefill, decode and idle snapshots in a single model session. Reconcile
   each owned buffer's logical/allocated bytes with actual region resident,
   dirty, compressed/swapped and wired state. Do not sum overlapping footprint
   and external lock counters into a fabricated total.
3. Attribute the observed ~725 MiB swapped inside the model to region owners;
   distinguish dormant tokenizer/CPU data from live GPU side weights, scratch
   or KV. The remaining system swap increase needs background-process/counter
   evidence. No packed-weight streaming diagnosis follows from global swap alone.
4. Evaluate the actual live allocation demand before selecting a fix: unused
   arena/host duplicates; retained prefill scratch and phase transition overlap;
   KV growth overlap; whole-shard Metal residency including sparse embedding.
   Change one measured contributor at a time, and A/B its peak/steady physical
   demand, greedy output and timing. Do not remove real hot locks or blindly
   lock more anonymous allocations, which can worsen host pressure.
5. Report a capacity deficit at load and continue with a visible degraded state.
   Do not describe an over-capacity forced run as healthy merely because weights
   locked successfully or the pressure sysctl happened to read its normal level.
   For the healthy matrix, document a representative desktop baseline and the
   measured reserve it requires; an arbitrary busy desktop cannot be guaranteed
   zero swap on fixed 24 GiB hardware.
6. If filesystem attribution requires root privileges unavailable to the harness,
   preserve AC3's I/O gate as unproven. `proc_pid_rusage` process disk-byte deltas
   can supplement region samples, but cannot by themselves distinguish layer/head
   reads from sparse embeddings. macOS `time -l`'s `0 swaps` does not establish
   zero paging: the failed run printed that while both host swap and swapped
   model-region bytes were observed.

The existing `gpuq status` prints a contradictory reassurance even when only
0.1 GiB is available, derived from one earlier success at 6.7 GiB. Its
`in use by processes` field is `vm_stat` active pages, not measured process
ownership. T5 must amend those misleading diagnostics without treating either
counter as a proven capacity bound. Do not purge caches, swap or user apps to
manufacture a passing result.

Current CPU probe evidence establishes COW on the investigated Darwin release;
it does not establish live Metal/no-copy attribution or full-chat throughput.
Use supported SDK layouts for any external native diagnostic; do not introduce
guessed Rust `TASK_VM_INFO` offsets. Shared physical pages must be reported
alongside process footprint; neither number alone is total unified-memory use.

## Sequencing, risks and fallbacks

T1 (mapping) precedes T2/T3 (accounting and failure reporting). T4's first forced
run failed. T3b then precedes T5 allocation/capacity diagnosis and its selected
small fixes; rerun the failed T4 gates after those changes. T6 reconciles docs and
records acceptance. None authorizes DFlash work.

| Risk | Handling |
|---|---|
| Shared locking still consumes too much physical RAM | Separate private-copy removal from wired demand; use T4/T5 results, not process footprint alone |
| Hot head re-streams from SSD | Verify its successful lock and actual backing; retain head protection |
| Partial lock looks complete | Record actual successful prefix/unique pages and warn per model; keep degraded test results separate |
| Copied/arena buffers remain pageable | Disclose them; do not reintroduce unsafe buffer-pointer locking or source-copy pinning |
| Normal desktop has existing swap/background I/O | Record before/after and background controls; attribute growth before blaming the engine |
| Metal binding changes page accounting | Require the actual loader + GPU-read regression and full-model region inspection |
| Historical budget contradicts current OS accounting | Retain the historical result with its assumptions; derive any new budget from current measurements |

## Requirement → task mapping

| Requirement | Tasks |
|---|---|
| R1 | T1, T4 |
| R2 | T4, T5 |
| R3 | T2, T3 |
| R4 | T2, T4, T5 |
| R5 | T1, T3, T4 |
| R6 | T4, T5, T6 |
| R7 | T3b, T4, T5 |
| R8 | T4, T5 |
| AC1 | T1 |
| AC2 | T2, T3, T3b |
| AC3, AC4 | T4, T5 |
| AC5 | T6 and per-task commits |
| AC6 | T3b, T4, T5 |
