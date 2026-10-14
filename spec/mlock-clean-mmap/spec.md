# spec/mlock-clean-mmap

Status: approved for implementation; T1 and T2 are complete, T3 is in progress.
User confirmed scope after clarification: ordinary chat with `Gxma-repacked`,
green memory pressure and no repeated weight streaming after load, 1k/2k/8k
contexts, and warning-and-continue when locking fails. DFlash work remains paused.

## Goal

Restore the clean file-backed mmap loading behavior lost in the reset, so
ordinary chat on the 24 GiB M4 Pro does not create private copies of packed
weights while locking them. Keep the weights used by inference resident with
truthful lock accounting, reduce memory pressure and repeated SSD reads, and
preserve output quality, decode throughput, and load/TTFT behavior.

## Evidence and difference

Current baseline: `e1032d44984000f1f4e5e2946810089c63599fe1`.
Preserved pre-reset comparison: `d6354c54def4ca87a4abb08811e3848a4111473e`.

| Behavior | Preserved loader | Current loader |
|---|---|---|
| Read-only shard mapping | `MAP_SHARED` | `MAP_PRIVATE` |
| Lock set | Budgeted layer trellis, excluding output head | All aligned mapped trellis, including output head; no byte budget |
| Lock failure | Continue with warning | Continue with warning, printed once per process |
| Forced arena weights | Separate arena policy | Mapped-trellis pass skips arena allocations |
| Misaligned zero-copy weights | Copied to aligned buffers | Copied to aligned buffers |

An isolated 128 MiB probe on this machine reproduced private copies after a
successful `mlock`: 8,192/8,192 pages became copied, anonymous and modified.
The same successful lock on `MAP_SHARED` left all pages file-backed and clean;
both regions reported `user_wired_count=1`. This verifies the mapping/locking
mechanism, not a completed live-chat fix. Full baseline/candidate chat A/B is
still required.

All 411 packed tensors in the local `Gxma-repacked` checkpoint are four-byte
aligned. Their payload totals 12,039,487,488 bytes. The unique page-rounded
lock set is 12,046,221,312 bytes including the head and 10,989,240,320 excluding
it. Alignment copies are not the cause for this checkpoint. The historical head
exclusion explains a 1,056,980,992-byte lock-set difference. This spec retains
the head lock to protect its repeated reads.

## Non-goals / restrictions

- Do not resume DFlash generation, change its checkpoint, or pursue 30 tok/s here.
- No weight/KV quantization changes, numerical changes, model downloads or edits
  to model files. Preserve existing mmap/no-copy Metal ownership and alignment.
- Retain real `mlock` for hot packed weights, including the current output-head
  lock. Prefetch hints or merely resident pages cannot substitute for locking.
- No whole-shard locking or prefetching of cold tensors; no anonymous copies of
  aligned packed weights. Keep embedding accesses sparse and unpinned except
  unavoidable shared boundary pages, which must be accounted for.
- No `MADV_DONTNEED`, live GPU mapping invalidation, new residency-set tricks,
  automatic sysctl/ulimit changes, or POSIX locking of Metal-owned buffers.
- Do not claim that successful `mlock` guarantees green system pressure, that
  `madvise` proves wiring, or that process RSS/footprint counts all shared wired
  file pages. Do not port the old 13.6 GB budget or 4 GB unpinned allowance as a
  currently measured policy.

## Requirements

1. **R1 — clean mapping.** Use read-only `MAP_SHARED` for safetensors shards
   whose packed tensors bind directly to Metal. Retain a single live backing
   mapping, unchanged tensor bytes, and its lifetime until all GPU views retire.
   Verify that locking and GPU reads do not turn this mapping into private
   copied/anonymous pages. Keep aligned packed weights on the no-copy route.
2. **R2 — retain successful hot-weight locking.** Keep real `mlock` on the
   aligned mapped hot trellis, including the output head. Normal acceptance runs
   must have the complete selected hot set successfully locked, stay green, and
   avoid hot weight streaming. Measure actual maximum reachable context,
   KV/scratch allocations and physical demand. A smaller displayed footprint or
   an intentionally unlocked hot set cannot satisfy acceptance. Any future cap
   must use merged unique pages and distinguish a degraded warning-and-continue
   run; do not select unprotected weights as a silent normal default. If clean
   shared locking still misses acceptance, record the cause and amend the plan
   before changing allocations or the required lock set.
3. **R3 — honest partial failure.** Keep loading on lock failure with an explicit
   warning. Record unique bytes requested, successfully locked and left
   evictable, including a successful partial prefix. Separate policy exclusions,
   disabled locking, alignment copies and failed locks. Do not suppress a later
   model's incomplete-lock summary because a previous model already warned.
   Treat degraded runs as degraded, not as proof of zero paging.
4. **R4 — physical accounting.** Pair the per-model unique lock ledger with OS
   region/page-state evidence, process footprint and system pressure/paging
   deltas. A clean shared file mapping may be wired without appearing in process
   footprint. Do not double-count page overlaps or mistake virtual shard size,
   cumulative page-ins, minor faults or pre-existing swap for new physical demand.
5. **R5 — exactness and compatibility.** Preserve normal chat output and existing
   handling of valid/invalid shards, aligned/misaligned tensors, embedding/head
   views, load failures and release. Leave copied Metal buffers on their existing
   path; report their evictable bytes without locking their pointers. Shared
   `TextWeights` changes must keep DFlash loading compatible, without requiring
   DFlash performance acceptance in this spec.
6. **R6 — measured steady state.** Validate ordinary chat at 1k/2k/8k prompt
   contexts with generation fitting inside the existing context limit, plus
   repeated chat turns and idle/resume. Distinguish initial load/prefill I/O,
   sparse embedding faults and steady decode weight I/O. Compare matched
   workloads A/B/A/B and report medians and spread for load, TTFT, prefill and
   decode. No new scheduler or polling in the generation hot path.

## Acceptance criteria

- **AC1 (R1/R5):** Actual loader-created mappings, before and after successful
  lock and Metal read, retain external file backing with no private copied or
  anonymous packed-weight pages. A deliberate `MAP_PRIVATE` substitution makes
  the named regression test fail; restore it and rerun before claiming coverage.
- **AC2 (R2/R3/R4):** Page-ledger tests cover disjoint/overlapping ranges, partial
  locks, first failure, disabled locking, final partial page, empty ranges and
  copied tensors. Force a lock failure and corrupt a byte count separately; each
  must fail a diagnostic assertion before restoration. The warning path loads
  and generates correctly and states the unprotected bytes.
- **AC3 (R2/R4/R6):** In recorded normal desktop conditions with a single target
  loaded, corrected ordinary-chat runs remain at normal memory-pressure level,
  with Activity Monitor green, no new swap growth attributable to the run, and
  no sustained packed-weight/head SSD streaming after warmup. Existing system
  swap is recorded before/after; it need not already be zero. Sparse embedding
  reads are recorded separately. A warm-window byte attribution to the shard is
  required; total device reads or page-fault counts alone cannot establish this.
- **AC4 (R5/R6):** Matched greedy output is identical for the acceptance workloads.
  Load, TTFT, prefill and decode have no repeatable regression outside A/B spread.
  Report process footprint and shared wired bytes together at 1k/2k/8k; compare
  actual physical demand on the 24 GiB machine, not a footprint-only prediction.
- **AC5:** Every implementation task is individually tested, documented and
  committed. Failed measurements remain visible; the spec is not complete until
  AC1–AC4 pass.

## Open questions / evidence gates

No unresolved user requirements. The user additionally restated that real
`mlock` must remain while paging is removed. Implementation measurements remain:
the size
of live private copies in a full baseline run; whether restoring shared mapping
alone with the existing hot lock set meets normal-chat pressure/I/O acceptance;
and whether further physical-memory changes are needed. These are explicit
measurement gates, not defaults silently chosen by this draft.
