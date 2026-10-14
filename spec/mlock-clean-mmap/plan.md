# spec/mlock-clean-mmap — plan

## Architecture and data flow

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
| `crates/engine/src/weights.rs:1427` (`ShardReader`) | Read-only `MAP_PRIVATE`; whole shard wrapped without copying | Read-only `MAP_SHARED`; preserve buffer offset/lifetime and error handling |
| `crates/engine/src/weights.rs:816` (`pin_trellis_pages`) | All aligned mapped trellis selected; sum of fully covered tensor windows | Unique-page ledger, partial-prefix accounting, explicit exclusions and policy selected from measurements |
| `crates/engine/src/weights.rs:1237` (`LlamaMlock`) | Incremental locks; latched failure; warning once per process | Keep soft failure semantics; make caller summary accurate per model; retain successful prefix |
| `crates/engine/src/model.rs:3811` | Normal target loads through `load_dedicated` | Supply actual configured KV/scratch reserve only if measurements require a byte cap |
| `crates/engine/src/model.rs:6299` | Misaligned view copied into shared Metal buffer | Compatibility checks only; no new copy path or buffer-pointer lock |
| `crates/engine/src/dflash.rs:131` | Uses the same `TextWeights` loader | Loader regression only; no generation/performance changes |
| `crates/engine/tests/clean_mmap_locking.rs` (new) | No test proving actual loader pages stay clean under lock + GPU use | Small shard fixture using real loader and no-copy read; OS page-state assertions and private-map sabotage |
| `crates/engine/tests/probe_llama_mlock.rs`, weight/alignment tests | Incremental locking and layout checks | Extend meaningful partial/failure/unique-page assertions; preserve existing behavior |
| This spec's notes and an acceptance harness if needed | No matched full-chat evidence for the proposed fix | Record workload, pressure, I/O, regions, memory and interleaved timing evidence |

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
   reaches the existing 8,192-token limit. Report prefix and total separately;
   never request generation after an already-full 8,192-token prompt.
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

Current CPU probe evidence establishes COW on the investigated Darwin release;
it does not establish live Metal/no-copy attribution or full-chat throughput.
Use supported SDK layouts for any external native diagnostic; do not introduce
guessed Rust `TASK_VM_INFO` offsets. Shared physical pages must be reported
alongside process footprint; neither number alone is total unified-memory use.

## Sequencing, risks and fallbacks

T1 (mapping) precedes T2/T3 (accounting and failure reporting), then T4 (chat
A/B). T5 (remaining-demand investigation) is conditional on T4. T6 reconciles docs and
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
| AC1 | T1 |
| AC2 | T2, T3 |
| AC3, AC4 | T4, T5 |
| AC5 | T6 and per-task commits |
