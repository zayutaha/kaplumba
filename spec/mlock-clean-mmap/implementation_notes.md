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

## 2026-10-14 — T3 per-load lock summaries

Added a per-`TextWeights` `MlockSummary` with the model path, unique eligible,
requested, attempted, locked, failed and evictable page bytes, plus disabled,
excluded and arena-copied source bytes and the first syscall error. The loader
prints an incomplete/disabled/excluded summary on each affected load; it no
longer suppresses a later model's warning process-wide. Complete loads print
the summary only under `KAPLUMBA_LOAD_PROFILE`. Lock injection is private to
the loader implementation and leaves the normal system `mlock` call unchanged.

Failure-injection checks exercise a first `ENOMEM` and a later `EPERM`, verify
the successful prefix remains accounted, prove the fixture's no-copy packed
bytes remain readable, and load a second model to confirm its own warning and
summary. The test captures child-process stderr and checks exactly two warning
lines with their respective errors. The fixture validates loader continuation
and bytes; normal generation output equality remains the full-model AC4 gate.

Negative control: replaced the per-load `eprintln!` with an unused formatted
string. The two-load test failed diagnostically (`expected 2 warning lines,
observed 0`). Restored warning emission and reran successfully. This checks
warning visibility, not actual kernel lock success; injected syscall outcomes
are accounting tests.

Commands passed:

```sh
cargo test -q -p kaplumba-engine --lib weights::tests
cargo test -q -p kaplumba-engine --test probe_llama_mlock
cargo test -q -p kaplumba-engine --test dflash_load --no-run
git diff --check
```

Results: weights tests 22 passed, 1 ignored (checkpoint-only); mlock probe 3/3;
DFlash loader integration target compiled. Existing workspace warnings are
unrelated. No Gxma model was loaded for T3. Implementation commit:
`6cecdd6` (`spec/mlock-clean-mmap: T3 per-load lock summaries`).

D0 draft was committed as `5abcb0c`; the scratch probe and its logs were removed
after preserving source, commands and observations above. No standalone probe
process or temporary project harness remains.

## 2026-10-14 — D1 forced-run failure; specification amendment only

User requested implementation, then explicitly requested a forced load despite
the low reported supply. After that run the user reported severe swap / apparently
missing locking and requested investigation plus specification amendment. This
amendment changes only this spec folder. T1–T3 remain committed component work;
the paging fix and full-spec acceptance are NOT complete.

### Failed full candidate run

Source `0bb4309` on `stepwise-clean-3724c01`, release rebuilt with
`cargo build --release -p kaplumba-cli`. The first attempted package name
`kaplumba` did not exist; that command failed before building. No concurrent
model process was running. Full-model command:

```sh
scripts/gpuq status
scripts/gpuq try -- /usr/bin/time -l ./target/release/kaplumba \
  --model "$HOME/.kaplumba/models/Gxma-repacked" \
  --prompt-file /tmp/mlock-clean-mmap-prompts/1024.txt \
  --max-new-tokens 200 --temperature 0 --show-ids --metrics
```

The prompt was `"Explain the theory of relativity. "` repeated 170 times,
created with Ruby outside the project. Actual tokenized prompt length was 1,034;
it was not exactly 1,024. Run PID 17278, no second model instance, exit 0.

| Metric | Observed |
|---|---:|
| Prompt / generated tokens | 1,034 / 200 |
| Load | 7,201.167 ms |
| TTFT | 11,412.522 ms |
| Prefill | 90.624 tok/s |
| Decode | 11.175 tok/s |
| Peak resident pages (`time -l` / metrics) | 12,449,284,096 B |
| Peak physical footprint (`time -l`) | 2,592,942,688 B (2.415 GiB) |
| Host swap before launch | 1,017.88 MiB |
| Host swap sampled after load | 2,747.62 MiB |
| Later samples | 2,887.19 / 2,930 MiB |
| Host swap sampled after exit | 2,850 MiB |

One candidate run is not an A/B result or a stable throughput claim. The prepared
`/tmp/kaplumba-mmap-baseline` worktree was at `e1032d4`, but its build command
ran from the MAIN working directory with a different target directory. That
produced a second candidate binary, not a baseline binary. Neither it nor a
baseline model was run. Do not infer any A/B performance or swap difference.

Immediately before launch `gpuq` printed 3.5 GiB available, 8.2 GiB `in use by
processes`, and 1,018 MiB swap. Earlier `top` showed 19 GiB used, 3,213 MiB
wired and 888 MiB compressor. During loading one sample showed 23 GiB used,
about 13 GiB wired, 1,944 MiB compressor and 86 MiB unused. Firefox dominated
the process RSS list before launch. `gpuq`'s process field actually reads
`vm_stat` active pages, which can include other ownership; it is not a process
allocation ledger. Its reassurance that even 0.1 GiB is "usually ENOUGH" is
unsupported extrapolation from a historical 6.7 GiB load and must be corrected.

`vmmap -summary 17278` sampled during generation reported:

| Region/accounting class | Resident | Dirty | Swapped |
|---|---:|---:|---:|
| mapped file (16.0 GiB virtual; 9 regions) | 11.2 GiB | 0 | 0 |
| Whole process | 13.5 GiB | 1.6 GiB | 724.9 MiB |
| malloc zones total | 4,480 KiB | 4,464 KiB | 105.7 MiB |

The live process footprint rounded to 2.4 GiB. Shared/external mapped residency
explains why this process figure is much smaller than its actual resident
working set. It is not evidence that the mapping is unlocked. At least ~725 MiB
of this process's non-file regions were swapped at that sample; the complete
region-owner snapshot was not saved, so their exact buffer owners cannot now
be assigned. Remaining host swap growth cannot be allocated to specific other
processes from the collected global counters.

`kern.memorystatus_vm_pressure_level` read 1 at sampled instants, while
`memory_pressure -Q` reported 4–5% free during the run and the user reported
severe swap. Do not use an isolated normal-level sample or the free percentage
as proof of Activity Monitor green throughout. `time -l` printed `0 swaps`;
that field plainly did not establish zero paging in this run.

The run had no incomplete-lock warning, but no success summary either because
`KAPLUMBA_LOAD_PROFILE` was unset. Full hot-set lock counts and OS wire state
were NOT captured. Source inspection shows `system_mlock_range` still calls
`libc::mlock` (`weights.rs:1603`), `pin_trellis` is enabled absent the disabling
environment variable (`:1490`), and `LlamaMlock` has no `Drop` unlock (`:1513`).
The current shell had no disabling/forced-arena environment variables. Strong
inference: clean hot weights were wired and removed from the pageable pool,
displacing runtime/desktop memory. This does not prove every full-model region
was successfully wired and must not replace the missing OS verification.

AC3 fails on new swap growth. AC1 full-loader GPU/wire-state acceptance, AC4
output A/B, 2k/8k, repeated turns, idle/resume and warm-I/O attribution are unmet.
`fs_usage -w -f diskio -t 5 17278` could not run without root. Its initial
`-p` invocation was invalid; this OS uses positional PIDs. No privileged trace
was taken. Cumulative process bytes from `proc_pid_rusage` can be useful later,
but cannot by themselves distinguish layer/head traffic from sparse embeddings.

### Bounded Metal lock-persistence probe

An external Objective-C diagnostic mapped 32 MiB of the first actual shard
read-only `MAP_SHARED`, applied real `mlock` to 16 MiB, created a whole-mapping
no-copy Metal buffer with shared/untracked options, registered/requested a
Metal residency set, dispatched a GPU byte-read, and ended residency. It used
the installed SDK definitions for `mach_vm_region_recurse` and `mincore`.
No full model was loaded and no product/kernel/test source was changed.

For all normal stages after locking, `user_wired_count=1`, `external_pager=1`,
all 1,024 selected 16 KiB pages resident, and zero copied/anonymous/modified
pages. After residency release the user wire remained 1. The GPU output equaled
the original byte. Negative control called `munlock` after GPU completion;
observed wire count became 0 and the diagnostic exited 11 with
`ASSERTION FAILED: successful user lock disappeared after Metal GPU read`.
Restored invocation passed. This verifies that sequence on a small mapping;
it does not exercise `TextWeights` or establish full-model lock coverage.

Commands:

```sh
cc -O2 -fobjc-arc -framework Foundation -framework Metal probe.m -o probe
scripts/gpuq status
scripts/gpuq try -- ./probe "$HOME/.kaplumba/models/Gxma-repacked/model-00001-of-00002.safetensors"
scripts/gpuq try -- ./probe "$HOME/.kaplumba/models/Gxma-repacked/model-00001-of-00002.safetensors" sabotage
# exit 11; observed wire count 0 and named failure above
scripts/gpuq status
scripts/gpuq try -- ./probe "$HOME/.kaplumba/models/Gxma-repacked/model-00001-of-00002.safetensors"
# restored, exit 0
```

### Required next work

Added R7/R8 and AC6, T3b unconditional success reporting and a concrete T5
allocation/OS-state diagnosis before choosing more changes. Source touchpoints:
`weights.rs:1802` whole-shard Metal binding and `:1274` residency scope,
`model.rs:5466` scratch grows and remains allocated for decode/idle,
`model.rs:6351` source/CPU/device sign representations,
`model.rs:1151` initial/growing KV and `gpu_cache.rs` old/new growth overlap.
These are measurable candidates, not proven fixes or assigned swap owners.
Do not disable locks, drop the head lock, wire anonymous Metal pointers, raise
sysctls or unload user apps as an unexplained workaround.

Corrected the plan's context claim: `generate.rs:35` currently has 32,768 as
the default limit. The agreed acceptance matrix still ends around 8k; this
amendment authorizes no context-policy change. Future prompt sizes must be
measured after tokenization rather than labeled from repetition counts.

Validation: source/phase/accounting review and `git diff --check` passed. The
bounded probe passed, the deliberate unlock failed with exit 11, and the
restored invocation passed. No product test or source was changed in D1.
Removed the unused comparison worktree, its target directory, prompt files,
native diagnostic binaries/logs and original scratch run log after preserving
the diagnostic source and complete run log below. No model/probe process remains.

Diagnosis and spec amendment commit: `2a1646c`. No further full-model run was
started after the user's report. Probe source is preserved below for reproduction.

### D1 diagnostic source (external scratch, not product code)

```objective-c
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/mman.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
static unsigned state(const char *stage, void *p, size_t n) {
    mach_vm_address_t addr=(mach_vm_address_t)p; mach_vm_size_t sz=0; natural_t depth=0;
    vm_region_submap_info_data_64_t info={0}; mach_msg_type_number_t count=VM_REGION_SUBMAP_INFO_COUNT_64;
    kern_return_t rc=mach_vm_region_recurse(mach_task_self(),&addr,&sz,&depth,(vm_region_recurse_info_t)&info,&count);
    size_t pages=n/sysconf(_SC_PAGESIZE),dirty=0,copied=0,anon=0,resident=0;
    char *vec=calloc(pages,1); int mc=mincore(p,n,vec);
    for(size_t i=0;i<pages;i++){dirty+=!!(vec[i]&MINCORE_MODIFIED);copied+=!!(vec[i]&MINCORE_COPIED);anon+=!!(vec[i]&MINCORE_ANONYMOUS);resident+=!!(vec[i]&MINCORE_INCORE);}free(vec);
    printf("stage=%s rc=%d mincore=%d wired=%u external=%u resident=%zu dirty=%zu copied=%zu anonymous=%zu pages=%zu\n",stage,rc,mc,info.user_wired_count,info.external_pager,resident,dirty,copied,anon,pages);
    return rc==KERN_SUCCESS ? info.user_wired_count : 0;
}
int main(int argc,char **argv){@autoreleasepool{
    if(argc<2)return 2;size_t total=32*1024*1024,hot=16*1024*1024;
    int fd=open(argv[1],O_RDONLY);if(fd<0){perror("open");return 3;}
    void *p=mmap(NULL,total,PROT_READ,MAP_SHARED,fd,0);if(p==MAP_FAILED){perror("mmap");return 4;}
    state("before_lock",p,hot);if(mlock(p,hot)){perror("mlock");return 5;}state("after_lock",p,hot);
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
    id<MTLBuffer> weights=[dev newBufferWithBytesNoCopy:p length:total options:MTLResourceStorageModeShared|MTLResourceHazardTrackingModeUntracked deallocator:nil];
    if(!weights)return 6;state("after_metal_binding",p,hot);
    MTLResidencySetDescriptor *desc=[MTLResidencySetDescriptor new];desc.initialCapacity=1;NSError *err=nil;
    id<MTLResidencySet> set=[dev newResidencySetWithDescriptor:desc error:&err];if(!set){fprintf(stderr,"residency unavailable\n");return 7;}
    [set addAllocation:weights];[set commit];[set requestResidency];state("after_residency",p,hot);
    NSString *src=@"#include <metal_stdlib>\nusing namespace metal;\nkernel void readbyte(const device uchar *x [[buffer(0)]], device uint *out [[buffer(1)]]) { out[0]=x[0]; }";
    id<MTLLibrary> lib=[dev newLibraryWithSource:src options:nil error:&err];if(!lib)return 8;
    id<MTLComputePipelineState> pipe=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"readbyte"] error:&err];if(!pipe)return 9;
    id<MTLBuffer> out=[dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue=[dev newCommandQueue];id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];[enc setBuffer:weights offset:0 atIndex:0];[enc setBuffer:out offset:0 atIndex:1];[enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted || *(unsigned*)out.contents!=*(unsigned char*)p)return 10;
    if(argc>2)munlock(p,hot);
    unsigned wired=state("after_gpu_read",p,hot);
    if(!wired){fprintf(stderr,"ASSERTION FAILED: successful user lock disappeared after Metal GPU read\n");return 11;}
    [set endResidency];[set removeAllAllocations];[set commit];state("after_end_residency",p,hot);
    munlock(p,hot);munmap(p,total);close(fd);return 0;
}}
```

### D1 original candidate log

SHA-256: `5a35bea895721e1ccf2a881d53201acc463e1dc71a590c343deeaa7ddc618531`.

```text
loading /Users/zayaantaha/.kaplumba/models/Gxma-repacked ...
loaded in 7201 ms
Albert Einstein’s Theory of Relativity is actually composed of two separate theories: **Special Relativity** and **General Relativity**. Together, they fundamentally changed how we understand space, time, and gravity.

Here is a breakdown of both:

---

### 1. Special Relativity (1905)
Special Relativity deals with objects moving at constant speeds, particularly those moving close to the speed of light. It is based on two main pillars:
*   **The Speed of Light is Constant:** Light always travels at roughly 300,000 km per second, regardless of how fast the observer is moving.
*   **Physics is the Same for Everyone:** The laws of physics are identical for all observers moving at a constant velocity.

**The consequences of these pillars are mind-bending:**
*   **Time Dilation:** Time moves slower for someone moving at high speeds relative to someone standing still. If you traveled near the speed of light for a year
token_ids: 93480 44281 236858 236751 18188 529 135383 563 3643 16635 529 1156 7732 21530 236787 5213 23244 135383 1018 532 5213 14947 135383 84750 29743 236764 901 51935 6692 1217 692 3050 2557 236764 990 236764 532 18899 236761 108 8291 563 496 25890 529 1800 236787 108 7243 108 10354 236743 236770 236761 10624 135383 568 236770 236819 236771 236810 236768 107 23244 135383 15663 607 7751 6049 657 4512 23257 236764 6412 1724 6049 3107 531 506 4249 529 2214 236761 1030 563 2721 580 1156 1689 60039 236787 107 236829 139 1018 818 23902 529 10847 563 36284 53121 10847 2462 33036 657 20129 236743 236800 236771 236771 236764 236771 236771 236771 7251 810 1855 236764 17194 529 1217 4592 506 28910 563 6049 236761 107 236829 139 1018 64446 563 506 35303 573 26335 53121 669 9734 529 16505 659 15779 573 784 52737 6049 657 496 4512 9254 236761 108 1018 818 15169 529 1239 60039 659 3666 236772 143244 53121 107 236829 139 1018 2866 622 13619 53121 7578 12489 31878 573 5026 6049 657 1494 23257 6658 531 5026 8101 2036 236761 1637 611 33402 3541 506 4249 529 2214 573 496 1051
prompt_tokens=1034 generated_tokens=200 stopped_on_eos=false load_ms=7201.167 ttft_ms=11412.522 prefill_tok_s=90.624 decode_tok_s=11.175 peak_resident_pages_bytes=12449284096
       37.64 real         0.68 user         2.35 sys
         12449284096  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
               40682  page reclaims
              751512  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
                2248  voluntary context switches
               86222  involuntary context switches
         33698763974  instructions retired
         11380756850  cycles elapsed
          2592942688  peak memory footprint
```

## 2026-10-14 — T3b complete-load lock summaries

The normal complete-lock path now prints `format_mlock_summary` once on every
load without `KAPLUMBA_LOAD_PROFILE`. Incomplete, disabled and copied cases
retain their warnings. The existing per-load failure-injection test continues
to exercise two different failing models.

Extended the real-loader child fixture across three modes: complete mapped
locking, explicit disabled locking, and copied weights. The parent captures
stderr and requires exactly one summary, the correct page/source-byte fields,
and warning status in each mode. The complete case uses real `mlock` on a
16 KiB-rounded fixture and checks requested=locked with failure=none. A
deliberately suppressed complete summary made the parent fail at the named
assertion with `each complete load must emit one lock summary ... left: 0,
right: 1`. Restored the print and reran.

`cargo test -q -p kaplumba-engine --lib weights::tests` passed: 22 passed,
1 ignored (checkpoint-only). `git diff --check` passed. The successful syscall
ledger remains separate from full-model OS region wire verification. Source
commit `81df21a`.
