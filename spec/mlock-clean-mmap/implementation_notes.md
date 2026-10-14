# spec/mlock-clean-mmap — implementation notes

## 2026-10-14 — D0 investigation and confirmed scope

User reports Activity Monitor memory pressure turning yellow with
`Gxma-repacked`; selected ordinary chat only, warning-and-continue on incomplete
locking, and confirmed spec-only work with 1k/2k/8k acceptance. Subsequently
restated: keep actual `mlock`, remove paging. DFlash remains paused. No product
source, kernel, test or model files changed by this drafting task.

Source baseline: `stepwise-clean-3724c01`,
`e1032d44984000f1f4e5e2946810089c63599fe1`. Preserved pre-reset comparison:
`mmap-before-reset-20261014`, `d6354c54def4ca87a4abb08811e3848a4111473e`.

Verified differences:

- Current `crates/engine/src/weights.rs:1470` maps `PROT_READ | MAP_PRIVATE`.
  Preserved `weights.rs:2315` used `MAP_SHARED`. Both keep mmap/no-copy buffers;
  the relevant regression is the mapping flag before actual locking.
- Current `weights.rs:816` locks aligned mapped `.trellis` tensors including
  `lm_head.trellis`. Preserved `weights.rs:1160` excluded the head and applied
  a budget in `pin_mapped_trellis` (`:1281`). Current has neither exclusion nor
  that budget. This adds pressure, but selecting a smaller unlocked hot set is
  not the requested fix. Restore shared backing with hot locks first.
- Preserved `weights.rs:1012` had arena pin handling; current mapped pass skips
  tensors with arena spans. This concerns the optional forced-arena path, not
  the selected aligned default checkpoint.
- Current `model.rs:6299` and preserved `model.rs:6410` both make aligned Metal
  copies for misaligned packed views. Neither tree has an actual
  `pin_realigned_copy` call despite comments referring to one. Do not describe
  that stale comment as an implemented historical guarantee.
- Current `weights.rs:1271` calls real `mlock`, preserves successful prefixes
  and latches a failure per lock object. `weights.rs:1297` prints the first
  warning once per process. Tensor-window sums do not prove unique locked bytes
  or expose every partially covered tensor.

### Checkpoint/header evidence

Read only the safetensors headers of
`~/.kaplumba/models/Gxma-repacked/model-0000{1,2}-of-00002.safetensors` using
system Ruby/JSON; no payload edits or checkpoint download. Both shards have
four-byte-aligned absolute starts for every `.trellis` tensor.

| Item | Bytes |
|---|---:|
| 411 trellis payloads | 12,039,487,488 |
| Head payload | 1,056,964,608 |
| Unique 16 KiB-rounded all-trellis windows | 12,046,221,312 |
| Unique layer-only windows | 10,989,240,320 |
| Unique head addition | 1,056,980,992 |
| Misaligned trellis payload | 0 |

Shard 1 has 212 trellises and 173 unique page runs; shard 2 has 199 trellises
and 164 runs (163 without head). Headers rule out the alignment-copy hypothesis
for this checkpoint. These byte totals are candidate lock-set arithmetic, not
live allocation measurements.

### Isolated mechanism measurement

Environment: macOS 27.0 build 26A5425a, Darwin 27.0.0, 16,384-byte pages.
No model process was running. Initial observed system swap was 1,218 MiB and
pressure level was `kern.memorystatus_vm_pressure_level=1`; existing swap cannot
be attributed to this test or the user's run.

Scratch C probe at `/private/tmp/kaplumba-mlock-review.z4YAIo/probe.c`, compiled
against the installed SDK headers, maps 128 MiB of the first target shard
read-only, touches one byte per page, applies `mlock`, and samples official
`task_vm_info`, `vm_region_submap_info_64`, `mincore`, and external `footprint`.
It runs each arm in a fresh process and releases the mapping before the next.
The source is reproduced below; it is diagnostic documentation, not engine code.

Private/shared no-lock controls both remained clean, copied/anonymous counts
zero, with footprint 1,376,712 bytes in the initial probe. The final probe ran
private/shared/private/shared with actual locks; every lock returned 0 and both
repeats of each arm agreed on the per-region and per-page result:

| After actual lock | Private | Shared |
|---|---:|---:|
| Region `user_wired_count` | 1 | 1 |
| Region `external_pager` | 0 | 1 |
| Copied / anonymous / modified pages | 8,192 each | 0 each |
| Resident pages | 8,192 | 8,192 |
| `footprint` mapped-file dirty | 128 MiB | 0 |
| `footprint` mapped-file clean | 0 | 128 MiB |
| Process footprint before lock | 1,393,096 B | 1,393,096 B |
| Process footprint after lock | 135,627,208 B | 1,409,480 B |

The additional 16 KiB in the shared process footprint belongs to probe scratch;
the 128 MiB mapping remains externally backed. Successful shared locking does
consume physical wired memory: the region's wire count changes from 0 to 1.
Early host samples also increased by 8,192 wired pages and returned to baseline
on release, but later immediate host samples were unchanged despite a wired
region. Host counter timing/background activity makes them unsuitable as the
sole assertion. Use the per-region state plus actual lock result.

Inference: current private locking can create a large dirty private population
and raise pressure even when `mlock` succeeds. Successfully wired pages are not
then swapped out; other pageable allocations can be pressured, and failed lock
ranges remain evictable. This reproduces the mechanism, not a full live-chat
diagnosis or proof that shared mapping alone satisfies all acceptance gates.
Do not claim product-test coverage from this standalone probe. T1 must exercise
the actual loader/Metal path and demonstrate failure when sabotaged.

The shared arm also shows why process footprint alone is insufficient for a
physical-memory budget. It reports roughly 1.4 MB while a 128 MiB region is
actually wired. Full Metal binding may change attribution; T4 must measure it.
Current `weights.rs:805` claims locked private pages remain clean; the probe
contradicts that on this machine. Historical shared-footprint budget assumptions
also require revision, rather than simply restoring all old prose.

Primary external source: [Apple XNU `mlock` implementation](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_mman.c),
lines 1596–1630 in the source inspected on 2026-10-14: `mlock` calls VM wiring,
returns an error when wiring fails, and `munlock` unwires. Local page-state
observations establish the private/shared difference on the installed OS.

### Reproduction commands and diagnostic source

```sh
git diff mmap-before-reset-20261014..e1032d4 -- crates/engine/src/weights.rs
cc -O2 -Wall -Wextra probe.c -o probe
./probe ~/.kaplumba/models/Gxma-repacked/model-00001-of-00002.safetensors private lock
./probe ~/.kaplumba/models/Gxma-repacked/model-00001-of-00002.safetensors shared lock
./probe ~/.kaplumba/models/Gxma-repacked/model-00001-of-00002.safetensors private lock
./probe ~/.kaplumba/models/Gxma-repacked/model-00001-of-00002.safetensors shared lock
# Baselines: use private none and shared none.
```

```c
#include <sys/mman.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
static void *mapping;
static size_t mapping_len;
static void stats(const char *stage) {
    task_vm_info_data_t info = {0};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t rc = task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count);
    printf("STATS stage=%s rc=%d footprint=%llu resident=%llu internal=%llu external=%llu compressed=%llu\n", stage, rc, (unsigned long long)info.phys_footprint, (unsigned long long)info.resident_size, (unsigned long long)info.internal, (unsigned long long)info.external, (unsigned long long)info.compressed);
    vm_statistics64_data_t vm={0}; mach_msg_type_number_t vm_count=HOST_VM_INFO64_COUNT;
    host_statistics64(mach_host_self(),HOST_VM_INFO64,(host_info64_t)&vm,&vm_count);
    printf("HOST stage=%s wired_pages=%u\n",stage,vm.wire_count);
    if(mapping) {
        mach_vm_address_t addr=(mach_vm_address_t)mapping;mach_vm_size_t size=0;natural_t depth=0;
        vm_region_submap_info_data_64_t region={0};mach_msg_type_number_t region_count=VM_REGION_SUBMAP_INFO_COUNT_64;
        kern_return_t rrc=mach_vm_region_recurse(mach_task_self(),&addr,&size,&depth,(vm_region_recurse_info_t)&region,&region_count);
        printf("REGION stage=%s rc=%d user_wired_count=%u private_pages=%u dirtied=%u share_mode=%u external_pager=%u\n",stage,rrc,region.user_wired_count,region.pages_shared_now_private,region.pages_dirtied,region.share_mode,region.external_pager);
        size_t pages=mapping_len/(size_t)sysconf(_SC_PAGESIZE);
        char *vec=calloc(pages,1);int rc=mincore(mapping,mapping_len,vec);
        size_t resident=0,copied=0,anon=0,dirty=0;
        for(size_t i=0;i<pages;i++){resident+=!!(vec[i]&MINCORE_INCORE);copied+=!!(vec[i]&MINCORE_COPIED);anon+=!!(vec[i]&MINCORE_ANONYMOUS);dirty+=!!(vec[i]&MINCORE_MODIFIED);}
        printf("PAGES stage=%s rc=%d resident=%zu copied=%zu anonymous=%zu modified=%zu total=%zu\n",stage,rc,resident,copied,anon,dirty,pages);free(vec);
    }
    fflush(stdout);
}
int main(int argc, char **argv) {
    if(argc != 4) return 2;
    int fd = open(argv[1], O_RDONLY); if(fd < 0) {perror("open");return 3;}
    size_t len = 128 * 1024 * 1024;
    int shared = strcmp(argv[2], "shared") == 0;
    int lock = strcmp(argv[3], "lock") == 0;
    void *p = mmap(NULL, len, PROT_READ, shared ? MAP_SHARED : MAP_PRIVATE, fd, 0);
    if(p == MAP_FAILED) {perror("mmap");return 4;}
    mapping=p;mapping_len=len;
    unsigned long sum=0; long page=sysconf(_SC_PAGESIZE);
    for(size_t i=0; i<len; i+=(size_t)page) sum+=*((volatile unsigned char *)p+i);
    printf("ARM mapping=%s lock=%d page=%ld bytes=%zu checksum=%lu pid=%d\n",argv[2],lock,page,len,sum,getpid());
    stats("before_lock");
    if(lock) {int rc=mlock(p,len); printf("LOCK rc=%d errno=%d message=%s\n",rc,rc?errno:0,rc?strerror(errno):"ok"); if(rc) return 5;}
    stats("after_lock");
    char pid[32]; snprintf(pid,sizeof(pid),"%d",getpid());
    pid_t child=fork(); if(child==0) {execl("/usr/bin/footprint","footprint",pid,(char *)NULL);_exit(127);}
    int status=0;waitpid(child,&status,0);printf("FOOTPRINT_TOOL status=%d\n",status);
    if(lock) munlock(p,len); munmap(p,len);mapping=NULL;close(fd);stats("after_unmap");return 0;
}
```

No full model was loaded for this investigation. No new green-pressure, paging,
throughput or zero-swap acceptance result is claimed. D0 validation is limited
to source comparison, header arithmetic, the standalone mechanism probe and
documentation review. All implementation tasks were still pending at that point.

## 2026-10-14 — T1 shared shard mapping

Changed `ShardReader::open` from read-only `MAP_PRIVATE` to read-only
`MAP_SHARED`. `LlamaMlock`, its enabled-by-default policy, the selected lock
set (including `lm_head.trellis`), buffer offsets and mapping lifetime are
unchanged. Updated the `ShardReader` thread-safety comment to state the actual
mapping mode.

Added `weights::tests::dedicated_loader_keeps_locked_packed_shard_clean_and_zero_copy`
beside the private loader internals. It loads a 2 MiB packed fixture through
`TextWeights::load_dedicated`, brute-forces an actually four-byte-aligned tensor
offset, checks the loader recorded a successful lock, verifies the Metal
no-copy buffer aliases the mmap and returns unchanged bytes, reads the shard
back from disk, and checks `mincore` reports zero modified pages for the locked
range. It repeats loading with a shard smaller than one VM page to cover the
final partial page. A companion test preserves missing-file and empty-shard
errors. Keeping this test inside `weights.rs` avoids exposing private mapping
state as a production API.

Negative control: temporarily changed the actual loader back to `MAP_PRIVATE`
and ran the focused test. It failed at the named page-state assertion with
`locked mmap pages became modified/private (129/129); expected clean shared file
backing`. Restored `MAP_SHARED` and reran successfully. The initial fixture
attempt also caught the existing four-byte alignment filter: a non-aligned
header offset had `pinned_tensors().len() == 0`. The test now brute-forces and
measures the aligned absolute offset, so it cannot pass without exercising
`mlock`.

Commands that passed with the restored source:

```sh
cargo test -p kaplumba-engine --lib weights::tests::dedicated_loader_keeps_locked_packed_shard_clean_and_zero_copy -- --exact --nocapture
cargo test -p kaplumba-engine --lib weights::tests::shard_open_preserves_missing_and_empty_file_errors -- --exact --nocapture
cargo test -p kaplumba-engine --lib weights::tests::retains_packed_exl3_bytes -- --exact
cargo test -p kaplumba-engine --lib weights::tests::pinning_follows_the_realigned_copy_split -- --exact
cargo test -p kaplumba-engine --lib weights::tests::trellis_stays_zero_copy_even_in_the_dedicated_loader -- --exact
git diff --check
```

All five focused tests passed. Existing unrelated compiler warnings remain.
These are fixture tests on the real Metal device, not a model run. Full
`Gxma-repacked` chat pressure, swap, SSD-I/O and 1k/2k/8k acceptance remain
unmeasured and are still T4 gates.

T1 implementation commit: `8143e6e` (`spec/mlock-clean-mmap: T1 shared mmap locking`).

## 2026-10-14 — T2 unique page and prefix accounting

Added checked page rounding for shard tensor intervals, clamped to the OS-rounded
mapping extent so EOF and sub-page shards include the final mapped page. Lock
selection still filters to aligned, mapped `.trellis` tensors and retains the
same per-tensor incremental `mlock` growth; it does not span cold gaps or add a
budget. After each lock run, `pinned_unique_bytes()` now sums the actual
successful `LlamaMlock::size()` prefixes, so shared/overlapping tensor windows
are counted once. `pinned_bytes()` and `pinned_tensors()` remain source-tensor
coverage views, and now include a tensor's covered prefix if a later grow fails.
The current public payload sum may double-count overlapping tensor bytes by
design; use the new unique-page accessor for physical lock totals.

Focused tests cover disjoint/touching/overlapping intervals, empty/reversed
ranges, final partial-page EOF, empty tensors, out-of-file bounds, page-rounding
overflow, address overflow, partial tensor prefixes, explicitly disabled
locking, and an arena-copied tensor. The real dedicated
loader test also checks the unique total against its actual lock run and checks
the sub-page fixture is counted as one page. Existing misalignment coverage
continues to prove copied/misaligned trellis ranges are not counted as mapped
locks. Lock failure injection and end-to-end per-model warning assertions are
T3.

Sabotage controls:

- Replaced union accounting with a naïve sum. The overlap test failed
  diagnostically: actual `32`, expected unique `24`.
- Replaced prefix clipping with full tensor length. The partial-prefix test
  failed diagnostically: actual `100`, expected `50`.

Restored both helpers. `cargo test -p kaplumba-engine --lib weights::tests`
passed: 19 passed, 1 ignored. The ignored case requires a real checkpoint.
`cargo test -p kaplumba-engine --test dflash_load --no-run` compiled the shared
loader consumer. The focused suite does not load a model or establish T4
memory-pressure and SSD read acceptance.

T2 implementation commit: `413e01e` (`spec/mlock-clean-mmap: T2 unique lock accounting`).

D0 draft was committed as `5abcb0c`; the scratch probe and its logs were removed
after preserving source, commands and observations above. No standalone probe
process or temporary project harness remains.
