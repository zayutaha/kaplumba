# spec/mlock-clean-mmap

Status: T1–T3b are implemented; full-model acceptance FAILED on the forced run.
T5 allocation and capacity work remains open. D2 identifies eager sliding KV
allocation as the first reduction to evaluate; no allocation fix is implemented.
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

| Behavior | Preserved loader | D0 baseline loader (before T1–T3b) |
|---|---|---|
| Read-only shard mapping | `MAP_SHARED` | `MAP_PRIVATE` |
| Lock set | Budgeted layer trellis, excluding output head | All aligned mapped trellis, including output head; no byte budget |
| Lock failure | Continue with warning | Continue with warning, historically printed once per process (D0 baseline) |
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

### Forced-run failure and corrected diagnosis (2026-10-14)

Candidate `0bb4309`, built from the main working tree, generated 200 tokens
after a 1,034-token prompt. System swap rose from 1,017.88 MiB before launch
to a sampled 2,930 MiB during/at completion, then 2,850 MiB afterward. This
fails AC3. The run started with only 3.5 GiB reported available and roughly
8.2 GiB of active pages; neither figure is an exact desktop allocation total.
The user explicitly requested the forced load despite that baseline.

Live `vmmap` showed 11.2 GiB resident in the mapped-file class, zero dirty or
swapped bytes in that class, and approximately 724.9 MiB swapped in other
regions of the model process. System wired memory was about 13 GiB; the process
peak footprint was only 2,592,942,688 bytes (2.415 GiB). A lower Activity Monitor
process figure therefore does not establish that locking stopped. Clean shared
wired pages can sit outside that figure while using physical RAM.

The complete-lock summary was gated on `KAPLUMBA_LOAD_PROFILE`, which was not
set. Thus the run did NOT record its exact successful lock count or inspect
every hot region's wire count. No incomplete-lock warning appeared. Together
with the clean resident mapped pages and increased system wiring this supports
successful wiring, but it does not close the full-model locking evidence gate.
The source still calls real `mlock`, keeps locking enabled by default, and does
not unlock when its temporary accounting object drops.

A separate 16 MiB lock/32 MiB mapping probe retained `user_wired_count=1`
through Metal no-copy binding, residency registration, a GPU read and residency
release, with no copied/anonymous/modified pages. Deliberate `munlock` made the
wire assertion fail; restoration passed. This rules out automatic lock loss in
that small sequence; it is not full-model acceptance.

Diagnosis: shared backing removes the reproduced private-copy mechanism but
does not make the locked hot set free. The forced workload displaced pageable
memory, including allocations in the model process itself. The exact allocation
owners and the host share of the swap growth remain to be measured. Claims that
the entire model is not locked, or that the packed weights themselves were
swapped, are not established by this run. The current design fails the user's
no-paging outcome and requires allocation/capacity work before further acceptance.

### D2 — live Activity Monitor check and allocation diagnosis (2026-10-14)

The user requested a reload and then documentation only. The existing release
build (`a118f23` tree, T3b source `81df21a`) loaded `Gxma-repacked` as PID 39967
in 3,278 ms, then stayed idle at the chat prompt. Its unconditional summary
reported requested=attempted=locked=12,046,221,312 bytes, failed=evictable=0,
failure=none. That is successful real `mlock` accounting, not a sampled kernel
wire count for each region.

The user's 08:08:48 screenshot shows kaplumba 1.81 GB, system wired 14.67 GB,
green pressure, and 3.22 GB existing swap. Live `vmmap` at 08:09:55 reports
footprint 1.8 GiB (peak 2.2 GiB), mapped-file resident 11.2 GiB, dirty=0,
swapped=0. The two checkpoint mappings are read-only `SM=SHM`, with rounded
resident sizes 5.9 GiB and 5.3 GiB. Thus the small process figure does not mean
the large shared weight working set disappeared or is available for eviction.
The lock ledger and system wired total support successful wiring; direct
per-hot-region verification remains AC6. Neither a resident snapshot nor green
pressure alone proves that every required page remains locked.

At the same sample, `IOAccelerator (graphics)` reports 1.8 GiB virtual,
1.6 GiB resident, 5,888 KiB dirty and 1.6 GiB swapped. These columns must not
be summed as disjoint physical allocations: shared Metal accounting can overlap.
System swap was 2,713 MiB before this reload and 3,203.69 MiB at inspection
(about +491 MiB); the host share of that increase is unresolved. The screenshot
is a green idle snapshot, not a no-paging acceptance result.

The first identified excess allocation is sliding KV: 50 layers preallocate
2,048 positions (1,024 window + 1,024 maximum append chunk). Each layer owns
two 16 MiB FP16 buffers (16 KV heads × 256 head width), totaling
1,677,721,600 bytes / 1.5625 GiB, plus 409,600 bytes of position tags. Both
live inspections contain exactly 100 graphics regions of 16 MiB, mostly
swapped in the idle snapshots. This matches the cache shape and identifies a
strong allocation candidate; definitive pointer-to-region ownership remains
required before claiming every one of those swapped regions is KV. Warmup
uses only two prefill tokens, resets, then one-token decode steps. Allocating
the maximum cache at construction is unnecessary for that idle workload.

The actual remaining problem is excessive pageable runtime demand beside the
locked hot weights, compounded by desktop demand. Small side-weight GPU copies
are also outside the packed-weight lock set; do not claim all model allocations
are wired. Evaluate sliding KV allocation on demand while preserving the exact
window, chunk overlap, safe GPU retirement and output requirements. The intended
benefit is lower idle/short-chat allocation; long prefill can require the full 2,048
slots. No result yet proves this candidate fixes 1k/2k/8k paging.

`scripts/gpuq` also wrongly reassures that the locked model is reclaimable and
can be evicted on demand. Its active-page count is labeled as process ownership.
Those claims conflict with the intended real locking and must be corrected in
a future implementation step.

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
7. **R7 — visible successful locking.** Emit one compact per-load summary even
   when every requested lock succeeds. Include policy state, requested/attempted/
   successful/evictable unique page bytes, excluded/copy bytes and syscall error.
   Distinguish syscall success from independently observed OS wire state. Verify
   layer and head regions after binding/warmup, prefill, decode and idle/resume
   in an opt-in diagnostic, without generation hot-path polling. A missing
   profiling environment flag must not hide whether the hot set was locked.
8. **R8 — bound the whole live workload.** Measure mapped hot pages, anonymous
   side weights and duplicates, embedding working set, Metal residency scope,
   KV growth overlap, retained scratch, CPU/tokenizer/driver memory and desktop
   pressure. Attribute model-region swapped bytes separately from host swap.
   Preserve real locks and reduce unnecessary live allocations before changing
   capacity policy. Report an insufficient-capacity load as degraded and keep
   warning-and-continue. Raising wire limits or forcing a load cannot make an
   over-capacity desktop pass green/no-swap acceptance. Healthy acceptance needs
   a recorded desktop supply that covers the measured total demand; arbitrary
   background workloads cannot be promised zero swap.
9. **R9 — allocate sliding KV for actual demand.** Evaluate a small initial
   sliding cache with growth before append to hold the retained attention
   window plus the incoming chunk. Preserve all live absolute positions and
   FP16 K/V bytes when ring capacity changes; synchronize before reading or
   replacing buffers referenced by queued GPU work, including work submitted
   by a different pipelined encoder. A fresh empty encoder must still create a
   queue barrier; its no-op `flush_and_wait` does not retire prior encoders'
   commands. Keep the existing maximum
   chunk/window behavior and context policy. Demonstrate lower idle/short-chat
   allocation and replacement peaks without output or throughput regression.
   Do not mark this requirement met from predicted byte savings.
10. **R10 — explain lock and capacity state accurately.** Report packed hot-set
    locking separately from pageable side weights, KV/scratch and host demand.
    A low Activity Monitor process footprint must not trigger weight copying or
    a claim of failed locking. Capacity diagnostics must never describe locked
    pages as evictable cache or active pages as summed process ownership.

11. **R11 — bound Metal resource spans.** GPU bindings for packed tensors must
    cover only their outward page-rounded tensor range, with the correct local
    byte offset. Repeated identical ranges share one view; unrelated cold ranges
    must not become part of a hot resource. Retain the backing mapping and locks.
12. **R12 — sparse embedding residency for ordinary chat.** Stage only the raw
    embedding rows required by the current forward into bounded shared scratch.
    Preserve the existing GPU BF16/FP16 conversion, rounding and scaling exactly.
    When a pipelined input token is still on the GPU, wait for the preceding queue
    work before reading it or reusing staging. Measure the synchronization cost
    and wired-memory reduction. Keep a diagnostic full-table control for A/B.

## Acceptance criteria

- **AC1 (R1/R5):** Actual loader-created mappings, before and after successful
  lock and Metal read, retain external file backing with no private copied or
  anonymous packed-weight pages. A deliberate `MAP_PRIVATE` substitution makes
  the named regression test fail; restore it and rerun before claiming coverage.
- **AC2 (R2/R3/R4):** Page-ledger tests cover disjoint/overlapping ranges, partial
  locks, first failure, disabled locking, final partial page, empty ranges and
  copied tensors. Force a lock failure and corrupt a byte count separately; each
  must fail a diagnostic assertion before restoration. The warning path loads,
  preserves readable weight bytes and states the unprotected bytes. Greedy output
  equality is checked in AC4 on normal acceptance runs.
  A complete load must also print its successful summary without profiling flags;
  sabotage summary suppression and require a diagnostic failure (T3b).
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
  AC1–AC4 and AC6 pass.
- **AC6 (R7/R8):** Inspect all selected hot regions, including the head, for real
  wiring after load/Metal binding and at the acceptance phases. Build an allocation
  ledger that explains swapped model regions and the pressure peak. Reconcile
  external/shared pages without blindly adding footprint and locked bytes, which
  can overlap. Prove any selected allocation fix reduces its measured demand.
  Forced insufficient-capacity runs must visibly report degraded capacity and
  remain failed acceptance even if all weight locks succeed.
  For R9/R10, correlate KV buffers to regions, compare idle/short/long-chat
  allocation and growth peaks, and verify diagnostics do not promise eviction
  of locked weights. Ring-growth tests must fail under deliberately broken
  modulo remapping, missing GPU synchronization and insufficient chunk capacity;
  preserve exact output across wrap, growth, reset and repeated turns.

## Open questions / evidence gates

No unresolved user requirements. The user additionally restated that real
`mlock` must remain while paging is removed. Implementation measurements remain:
the size
of live private copies in a full baseline run; whether restoring shared mapping
alone with the existing hot lock set meets normal-chat pressure/I/O acceptance;
which runtime/desktop allocations account for the forced run's swap growth;
and the smallest allocation reduction that satisfies the recorded desktop
budget. D2 identifies sliding KV preallocation as a measured-shape candidate;
pointer attribution, actual savings and output/pressure acceptance remain open.
Further physical-memory work is required for the failing workload.
The exact full-model OS wire state remains unrecorded. These are explicit
measurement gates, not defaults silently chosen by this draft.
