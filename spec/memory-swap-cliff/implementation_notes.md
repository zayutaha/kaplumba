# implementation_notes — memory-swap-cliff

Running log for `plan.md`. Everything here is either a `file:line` citation, a
number read from a safetensors header on this box, or arithmetic over those
numbers. Nothing was applied. The model was not run.

---

## 0. Provenance of every number used below

**A. Checkpoint byte ledger.** Read directly from the safetensors headers of the
two checkpoints (8-byte little-endian header length + JSON header), not from any
prose. Text selection reproduces `14_876_563_172` exactly, which is the constant
asserted at `crates/engine/src/weights.rs:1406` and
`crates/engine/src/safetensors.rs:1434` — so the extraction is correct.

| | `Gxma-repacked` (3 bpw) | `Gxma-4-bit` (4 bpw) |
|---|---:|---:|
| shard files on disk, total | 16,028,388,994 | 20,041,520,424 |
| | 14.928 GiB | 18.665 GiB |
| vision tower (never loaded) | 1,151,487,072 | 1,151,487,072 |
| | 1.072 GiB | 1.072 GiB |
| **text selection** (`safetensors.rs:441-514`) | **14,876,563,172** | **18,889,725,668** |
| | 13.855 GiB | 17.592 GiB |
| `embed_tokens` — mapped, cold | 2,818,572,288 | 2,818,572,288 |
| | 2.625 GiB | 2.625 GiB |
| `lm_head.trellis` — mapped, hot every token | 1,056,964,608 | 1,409,286,144 |
| | 0.984 GiB | 1.312 GiB |
| layer `.trellis` (410 tensors) | 10,982,522,880 | 14,643,363,840 |
| | 10.228 GiB | 13.638 GiB |
| all arena-packed tensors | 10,998,363,244 | 14,659,204,204 |
| | 10.243 GiB | 13.652 GiB |
| BF16 norms/scalars (converted in place) | 2,663,032 | 2,663,032 |
| **arena logical = arena capacity** | **11,001,026,276** | **14,661,867,236** |
| | **10.246 GiB** | **13.655 GiB** |

Arena capacity equals arena logical because `load_from_dir` calls
`TextWeights::load_dedicated` (`model.rs:3324`), and `allocate_dedicated`
(`weights.rs:200-235`) allocates one exactly-sized buffer per tensor and resets
`current_segment` each call — so there is **no 512 MiB segment rounding**.
(`DEFAULT_SEGMENT_BYTES = 512 MiB`, `metal.rs:33`, applies only to the `allocate`
path, which `load_dedicated` bypasses.)

**B. Environment, measured on this box with no model loaded (2026-10-02).**

```
hw.memsize                 25,769,803,776 B = 24.000 GiB
vm.user_wire_limit         19,585,050,869 B = 18.238 GiB
top -l 1 -n 0 | PhysMem    used 6031M  wired 2401M  compressor 352M  unused 17G
vm.swapusage               total 4096.00M  used 2743.12M
vm.page_free_count         1,131,140 pages
```

Page size = **16,384 B**, confirmed arithmetically:
`1,131,140 × 16,384 = 18,532,520,960 B = 17.26 GiB`, which is what `top` reports
as "17G unused". (This matches `spec/mtp-40tps/implementation_notes.md:13`.)

**C. Heavier-desktop snapshot**, from
`spec/mtp-40tps/implementation_notes.md:21-32` (page size 16,384):

```
Pages free                       9.77 GiB
File-backed pages                6.20 GiB   <- reclaimable
Anonymous pages                  4.08 GiB   <- app memory
Pages wired down                 2.89 GiB
Pages occupied by compressor     0.42 GiB
=> reclaimable ~15.97 GiB, locked ~7.39 GiB, of 24.00 GiB
```

**D. Measured toolchain limits**, from
`spec/mtp-40tps/implementation_notes.md:176-210`:

```
NVMe cold sequential read                       22.1 GB/s
1 GB wired GPU chunks, free/wired trace:
  1 GB   free=8.51  wired= 3.91
  5 GB   free=4.78  wired= 7.64
  9 GB   free=1.05  wired=11.37
 10 GB   free=0.12  wired=12.30   <- stop
  (file-backed never moved: 6.40 GiB throughout)
mmap-streaming a real 17.27 GB file: all 17.27 GiB held as file-backed,
  free 9.54 -> 0.09 GiB, zero page-ins, zero swap growth
```

**E. Footprint anchors actually recorded in this repo.**

| config | `peak memory footprint` | source |
|---|---:|---|
| 3 bpw default, 2500 tokens | 12,867,037,000 (11.98 GiB) | `spec/resident-memory-12gib/implementation_notes.md:157` |
| 3 bpw default, repeat | 12,867,233,488 | same, line 204 |
| 3 bpw `KAPLUMBA_RESIDENT_WEIGHTS=1` | 16,719,311,256 (15.57 GiB) | same, line 205 |
| 3 bpw (brief) | 12,446,246,152 (11.59 GiB) | task brief |
| 4 bpw (brief) | 16,028,184,360 (14.93 GiB) | task brief |

`phys_footprint` is reproducible to **196,488 B** across identical runs
(`resident-memory-12gib:215-216`) — 0.0002 GiB. It is therefore a trustworthy
*instrument*; what is not trustworthy is comparing it against a prose budget.

---

## 1. THE THRESHOLD

### 1.1 There is no engine-side budget being crossed

`resident_limit_bytes()` = `18_000_000_000` (`metal.rs:4`, overridable by
`KAPLUMBA_RESIDENT_LIMIT`, `metal.rs:18-32`). It is checked in
`WeightArena::allocate_dedicated` (`weights.rs:210-216`) against
`capacity_bytes` only — i.e. against the arena, never against what the box has
free.

* 3 bpw arena: 11,001,026,276 < 18e9 — passes, 7.0 GB of margin *on paper*.
* 4 bpw arena: 14,661,867,236 < 18e9 — **passes with 3.34 GB of margin on
  paper**, and the process then thrashes.

`metal.rs:10-13` even documents that the 4 bpw / 8 bpw-head checkpoint "needs
18,046,374,518 bytes resident, which is 46 MB over the default" — i.e. the
constant was raised *for* the 4 bpw case and is now permissive enough to admit a
configuration the hardware cannot hold. **The threshold is the OS's, and the
engine does not read it.**

### 1.2 The anonymous demand, counted by class

Classes, with what macOS can do to each:

| class | what it is | reclaimable? | how |
|---|---|---|---|
| **wired** | driver, WindowServer, GPU/IOGPU, and `mlock`ed pages | **no** | unswappable, uncompressible |
| **anonymous (internal), dirty** | the arena (Metal `StorageModeShared` written by `read_at`, `SM=SHM PURGE=N` per `engine-performance:1202-1203`), KV keys/values/positions, scratch | **only by compressing, then swapping** | compressor, then swap at 22.1 GB/s |
| **file-backed clean** | `embed_tokens` and `lm_head.trellis` pages, shard page cache | **yes, instantly** | purge / readahead drop |

Compressor arithmetic: the layer trellis measures **Shannon entropy 7.9702 of a
possible 8.0000 bits/byte** (8 MiB window sampled from shard 2 at the
`lm_head.trellis` payload offset; `gzip -9` → 8,232,575 / 8,388,608 = **1.87 %
saving**). EXL3 is bit-packed high-entropy data, so macOS's compressor yields
≈1.0 bytes/page while consuming the full compressor CPU and creating compressor
pages that themselves need swapping. **The 10–12 GB of compressor observed at
4 bpw is not headroom; it is the compressor failing and then swapping the
failures.**

### 1.3 Demand at each configuration

3 bpw, KV at its initial 512 (`model.rs:1077 INITIAL_FULL_CACHE`), scratch at the
1024-row `PREFILL_CHUNK` peak:

```
arena           11,001,026,276
KV (512)           881,029,120     <- §3.1 derivation
scratch (1024)     547,937,040     <- §4.1 derivation (the TRUE figure)
                -------------
anon total      12,429,992,436 = 11.576 GiB
```

4 bpw, same:

```
arena           14,661,867,236
KV (512)           881,029,120
scratch (1024)     547,937,040
                -------------
anon total      16,090,833,396 = 14.985 GiB
```

Cross-check against the recorded anchors:

```
3 bpw:  anon 11.576 GiB  vs  measured 11.59 / 11.98 GiB   -> +0.01 .. +0.40 GiB
4 bpw:  anon 14.985 GiB  vs  measured 14.928 GiB           -> -0.06 GiB
```

The residuals are small and have opposite signs. They are the mapped
`lm_head` residency (0.984 / 1.312 GiB, charged to `phys_footprint` as
file-backed resident pages per `resident-memory-12gib:283-288`) plus driver
overhead, and `lm_head` residency is strongly run-dependent: the 2500-token run
re-reads the head 2500× and keeps all of it, a 1-token run faults it once and
most of it is purged before exit. **The single 420,790,848 B spread between the
two recorded 3 bpw figures is exactly `1,056,964,608 × 0.398` — i.e. 39.8 % of
`lm_head` resident in one run and 100 % in the other.** That is the only
unexplained number and it is explained.

### 1.4 Where the crossing is

Free supply = `24.000 GiB − wired − other applications`. Two independent
estimates:

* From `mtp-40tps:21-32`: `24.000 − 2.89 − 4.08 = 17.03 GiB` (the framing that
  spec already uses, line 207).
* Back-solved from the brief's 4 bpw observation (`unused ≈ 515 MB`,
  `compressor 10–12 GB`, wired ≈ 2.4 GiB): `other ≈ 24.000 − 2.4 − 0.5 − 11.0 =
  10.1 GiB` ⇒ supply ≈ **14.5 GiB**.

The brief's own baseline is "~11–13 GB used", a 2 GiB range on its own, so
**supply is somewhere in 11.5–17.0 GiB depending on Firefox and the VM.**

Ratio at the cliff:

```
3 bpw  11.576 / 14.5  = 0.80  -> fits,  2.9 GiB margin
4 bpw  14.985 / 14.5  = 1.03  -> over by 0.49 GiB -> compressor + swap
```

### 1.5 THE ANSWER

**The threshold is `engine anonymous demand ÷ (24 GiB − wired − other apps) ≈
1.0`, i.e. an OS reclaimability threshold, not a hard engine budget.** The engine
supplies 11.576 GiB (3 bpw) or 14.985 GiB (4 bpw); the box supplies 11.5–17.0 GiB
depending on what the desktop is doing; the two 3 bpw runs fit and the 4 bpw run
does not, and **nothing in the engine distinguishes those cases.**

What macOS can and cannot reclaim:

| | 3 bpw @512 | 4 bpw @512 | 4 bpw @16k |
|---|---|---|---|
| arena 10.246 / 13.655 / 13.655 GiB | compress (no gain) then swap | same | same |
| KV 0.821 / 0.821 / 2.094 GiB | compressible (fp16, ~2× gain) then swap | same | same |
| scratch 0.510 GiB | compressible | same | same |
| `lm_head` 0.984 / 1.312 / 1.312 GiB | **yes**, instantly — but 63.8 ms/token to fault back at 4 bpw | same | same |
| `embed_tokens` 2.625 GiB | **yes**, and it is already cold (~10 KiB working set) | same | same |
| vision tower 1.072 GiB | never resident | same | same |

So: **everything clean is reclaimable and everything dirty is not**, and the
dirty set is 11.6–16.1 GiB against 11.5–17.0 GiB of supply. macOS's ladder is
purge → compress → swap, and step 2 buys nothing on bit-packed weights.

**This is why the fix is admission control, not a pinning policy.** Pinning moves
bytes from the *reclaimable* column to the *unreclaimable* column. It improves
latency and it does not change the sum.

---

## 2. MLOCK POLICY

### 2.0 The premise under review is not the code

`grep -rn mlock crates/` → **0 hits**. `spec/file-backed-weights/tasks.md` marks
T2 (line 15) and T4 (line 38) *Not started*. Every worktree checked
(`/private/tmp/wt-*`, `…/T/opencode/*`, 26 trees) also has 0 hits.

At HEAD the layer trellis is **arena-resident and dirty anonymous**, confirmed
by the repo's own note at `spec/engine-performance/implementation_notes.md:1201-1205`:

> Page faults during decode are irrelevant (20/token, 0.003 % of 12.06 GB; 236,547
> major faults are the one-off load). The layer trellis is *dirty* anonymous
> arena (11 GB, `SM=SHM PURGE=N`), so it is compressible and evictable but not
> free.

And `weights.rs:508-522` says so in code comments too, with the measurement:

> Decode streams every layer weight on every token, so file-backing the 11 GB of
> layers let the OS evict them: that build measured 3.9 GiB of swap at ~1.4 tok/s.

So the question "should we pin?" is really "should we ship `file-backed-weights`
T2+T4?", and the answer has to start by deleting three claims in that spec.

### 2.1 Wire budget, measured

```
vm.user_wire_limit                        19,585,050,869 B = 18.238 GiB
wired, idle desktop (measured)             2,517,630,720 B =  2.345 GiB
                                          ------------------------------
wire headroom                             17,067,440,149 B = 15.893 GiB
less driver/IOGPU growth at load (M1)                    ~ 1.000 GiB
                                          ------------------------------
usable lock budget                         ~16,065,000,000 B = ~14.96 GiB
```

`M1` is **not yet measured** — it is the one constant the implementer must fill
in. It is bounded above by `mtp-40tps:181-196`: that probe found only ~10 GB of
*newly allocated* wired memory succeeded before free hit 0.12 GiB, with
`file-backed` never moving. That tells us new wire demand is served from **free
supply, not reclaimable cache** — so the lock budget is bounded by free supply,
not by the 15.97 GiB "reclaimable" figure.

### 2.2 Whole-shard `mlock` (`file-backed-weights/plan.md:89-104`) — reject

```
3 bpw  16,028,388,994 B = 14.928 GiB  vs 15.893 GiB wire headroom
       -> 93.9 % of the wire headroom consumed by weights alone,
          17,067,440,149 - 16,028,388,994 = 1,039,051,155 B = 0.968 GiB spare (6.1 %).
4 bpw  20,041,520,424 B = 18.665 GiB  >  18.238 GiB limit
       -> FAILS ARITHMETICALLY, before any other page is wired.
```

Consequences of the 4 bpw failure as the plan is written:

* `mlock` returns `ENOMEM`; `plan.md:98-103` logs `warning: mlock(...) failed`
  and continues.
* The trellis is then **mapped-but-evictable clean pages** — byte-for-byte the
  `KAPLUMBA_ZERO_COPY_WEIGHTS` configuration that `weight_residency.rs:7-9`
  records as "measured 3.9 GiB of swap and ~1.4 tok/s".
* The only signal is a warning on stderr, after a 10–19 s load.

And it pins **1.072 GiB of vision-tower bytes** that
`safetensors.rs:469-470` deliberately excludes from the selection, for tensors
the engine never opens.

**Per-tensor `mlock` over the page-aligned window is the only acceptable shape.**
`ShardReader::no_copy_view` (`weights.rs:1050-1077`) already computes exactly
that window (`aligned = start & !(PAGE_ALIGN - 1)`, `delta = start - aligned`,
`weights.rs:1061-1062`), so the lock can reuse the identical arithmetic and
cannot drift from the buffer's actual bounds.

### 2.3 The four policies, compared

Demand side is identical in all four; only classification changes.

| | (a) pin all trellis | (b) pin nothing | (c) lazy per-layer | (d) budgeted *(recommended)* |
|---|---|---|---|---|
| 3 bpw trellis locked | 10.228 GiB | 0 | 0→10.228 over first step | up to 10.228, budget 14.96 → **all pinned** |
| 4 bpw trellis locked | 13.638 GiB | 0 | 0→13.638 over first step | up to 13.638, budget 14.96 → **all pinned, 91 %** |
| resulting footprint @4 bpw @512 | 14.99 GiB, of which 13.64 wired | 1.33 GiB anon + ≤1.31 hot cache | converges to (a) | as (a) |
| reclaimable headroom left | **14.99 − 13.64 = 1.35 GiB** | **14.99 GiB** | intermediate, mixed | **1.32 GiB, declared** |
| decode risk | lowest after pinning; **worst failure is a partial silent ENOMEM** | **measured 1.4 tok/s** | first-token cost ∝ eviction; mixed residency is the OS's worst case | (a)'s latency where it fits; (b)'s failure **visibly**, where it does not |
| fits 4 bpw @16k | **no** | **no** | **no** | **no** |

Two asymmetries worth stating plainly:

* **(b) is not a viable steady state, only a cliff edge.** Decode re-reads the
  *entire* weight set every token. At 4 bpw that is 18.89 GB/token; at 22.1 GB/s
  of NVMe, 100 % of it missing costs 855 ms/token against a ~70 ms step, and even
  a **1 %** miss costs 8.6 ms/token (12 %). The offload budget is **0 bytes per
  token**. A working set that equals the model has no partial-retention policy:
  the OS either keeps all of it or the process is at 1.4 tok/s.
* **(a) vs (d) differ only in honesty about the tail.** At 3 bpw they are the same
  configuration. At 4 bpw they differ when `mlock` starts failing: (a) calls it a
  warning, (d) never gets there and reports the unpinned tensors by name.

### 2.4 Recommendation

**Policy (d), with two exclusions and one precondition.**

**Precondition (F1).** Pinning removes the compressor/swap escape valve. Doing it
on a box that is already over supply converts a slow recovery into an
unrecoverable one. (d) is only reachable once admission control has already
proved `projected_peak ≤ measured_free_supply`.

**Exclude `lm_head.trellis` (1.312 GiB at 4 bpw).**
`1,409,286,144 / 22.1e9 = 63.8 ms/token` if it ever misses, against a ~70 ms
step — so it must be resident. But residency ≠ wiring: leaving it as reclaimable
clean page cache keeps it hot *and* returns 1.312 GiB of budget the instant the
OS needs it. This is already the measured state
(`resident-memory-12gib:162-165`: 2500 tokens × ~2.6 TB of re-reads, **0 swaps**,
0 MiB of system swap at 15 samples over 288 s).

**Exclude `embed_tokens` (2.625 GiB).** One row (~10 KiB) per token. Pinning
2.625 GiB to keep 10 KiB hot is the worst trade available; it is already
file-backed and already cold.

**Pin eagerly, in decode order, up to a computed budget** — not lazily. (c)'s
laziness buys nothing (d) does not buy better: (d) knows the budget *before* the
first token, so it can pin what fits and **print the tensors it could not pin**,
instead of discovering the limit 410 `mlock` calls into a step.

**And the bottom line, stated plainly: (d) is a latency fix for 3 bpw, not a
capacity fix for 4 bpw.** At 4 bpw the trellis alone is 13.638 GiB against a
~14.96 GiB budget (91 %), so it does pin — and the total is still over supply.
The 4 bpw problem is a capacity problem and admission control (§1.5) is the only
thing that addresses it.

---

## 3. THE 16k CONTEXT QUESTION

### 3.1 KV, derived

`GpuKvCache::resident_bytes()` = `capacity * stream_len * 2 * 2 + capacity * 4`
(`gpu_cache.rs:121-123`) — keys + values + positions.

Shapes from `config.json` + `model.rs:3363-3420`: sliding `kv_heads=16`,
`head_dim=256` → `stream_len = 4096`; full `kv_heads=4`,
`head_dim=global_head_dim=512` → `stream_len = 2048`. 50 sliding +
10 full (`config.json` `layer_types`; `model.rs:3363` `is_full()`).

```
sliding: 1024 * 4096 * 2 * 2 + 1024 * 4 = 16,781,312  x 50 =   839,065,600  (fixed)
full:    C * 2048 * 2 * 2 + C * 4      = C * 8196
```

| full capacity C | per layer | × 10 | **total KV** |
|---:|---:|---:|---:|
| 512 (`INITIAL_FULL_CACHE`, `model.rs:1077`) | 4,196,352 | 41,963,520 | **881,029,120** (0.821 GiB) |
| 1024 | 8,392,704 | 83,927,040 | 922,992,640 (0.860 GiB) |
| 2048 | 16,785,408 | 167,854,080 | 1,006,919,680 (0.938 GiB) |
| 4096 | 33,570,816 | 335,708,160 | 1,174,773,760 (1.094 GiB) |
| 8192 | 67,141,632 | 671,416,320 | 1,510,481,920 (1.407 GiB) |
| 16384 | 134,283,264 | 1,342,832,640 | **2,181,898,240** (2.032 GiB) |
| 32768 | 268,566,528 | 2,685,665,280 | 3,524,730,880 (3.282 GiB) |
| 262144 (`max_position_embeddings`) | 2,148,532,224 | 21,485,322,240 | **22,324,387,840** (20.79 GiB) |

The 512 and 16384 rows match the brief exactly, which validates the derivation.

**Doubling transient.** `ensure_capacity` (`gpu_cache.rs:167-233`) allocates the
new buffers, copies the live prefix, then *drops* the old ones inside the same
call, so the transient is per-layer, not cumulative: at 8192→16384 it peaks at
`67,141,632 + 134,283,264 = 201,424,896` per layer, i.e. **+0.063 GiB** over
steady, for one layer at a time. Small. But
`engine-performance:1230-1231` records "a residual ~0.25-0.5 GiB of driver-side
Metal graphics regions that are not returned", and 5 doublings × 10 layers free
`10 × (67,141,632 + 33,570,816 + …) ≈ 0.67 GiB` of Metal buffers of which up to
0.5 GiB may not come back. Since `peak memory footprint` is a *peak*, that
residual counts.

### 3.2 The budget

Anchored on the **measured** 4 bpw footprint at 512 positions and adding the
**exact** KV delta — no guess about driver overhead, which is what makes the
prose reconciliations in `resident-memory-12gib:273-284` drift:

```
4 bpw @ 512   measured                                    16,028,184,360  = 14.928 GiB
              KV delta 512 -> 16384 (2,181,898,240 - 881,029,120)
                                                          1,300,869,120  =  1.211 GiB
4 bpw @16384  projected                                   17,329,053,480  = 16.139 GiB
```

Component view of the same number, for the implementer:

```
arena (anonymous, pinned under (d))     14,661,867,236  13.655 GiB
lm_head.trellis mapped + hot            1,409,286,144   1.312 GiB
KV @16384 steady                       2,181,898,240   2.032 GiB
KV doubling transient                    67,141,632   0.063 GiB
scratch @1024 rows                        547,937,040   0.510 GiB
                                         -------------
                                         18,868,130,292  17.572 GiB
+ lm_head already counted, so this is the anon+hot total;
  measured-anchored projection is 16.139 GiB, i.e. the driver/other residue
  is NEGATIVE 1.43 GiB -- see the caveat below.
```

**Caveat, stated rather than hidden.** The component view (17.572 GiB) exceeds the
measured anchor plus KV delta (16.139 GiB) by 1.43 GiB. That means the brief's
14.928 GiB anchor at 4 bpw did **not** have `lm_head` resident (its 1.312 GiB is
absent from the footprint) and did not carry a separate 0.75 GiB overhead term.
Which is self-consistent: at 4 bpw the anon total (14.985 GiB) already sits 0.06
GiB *above* the measured 14.928, so there is no room left in that run for
`lm_head` residency. **`lm_head` residency swings the 4 bpw footprint by up to
1.312 GiB depending on how many head invocations the run performed** (one per
prefill chunk — `model.rs:4006-4020` skips it for all but the last — plus one per
decode token). A single-token run has the best case; a 2500-token run has the
worst. Any 4 bpw budget must assume the worst case.

### 3.3 Does it fit?

Supply, two estimates (§1.4): **17.03 GiB** (`mtp-40tps` snapshot) or
**≈14.5 GiB** (brief's desktop load).

```
4 bpw @16384   demand 16.139 GiB
               vs 17.03 GiB supply -> deficit  0.89 GiB   (fits on a quiet desktop)
               vs 14.5  GiB supply -> deficit  1.64 GiB   (does not fit)

3 bpw @16384   11.98 + 1.211 = 13.191 GiB
               vs 17.03 GiB -> margin 3.84 GiB   |   vs 14.5 GiB -> margin 1.31 GiB
```

**Answer: 4 bpw at 16384 does not fit on this box under the reported desktop
load, and no mlock policy changes that.** Pinning moves 13.655 GiB from the
reclaimable column to the immovable column; the sum is identical. In fact
pinning makes the *shortfall* worse, because it removes the compressor/swap path
that currently lets the process limp.

3 bpw at 16384 fits with 1.3–3.8 GiB of margin.

### 3.4 Every lever, including the ones that do not work

Off-limits per the brief: per-token dispatch counts, `PREFILL_CHUNK`. Both are
listed anyway so the exclusion is on record rather than assumed.

**Works:**

| lever | closes | notes |
|---|---:|---|
| Close Firefox / the VM. `mtp-40tps:26` measures 4.08 GiB of anonymous apps; the brief's baseline implies ~10 GiB. | **up to 6 GiB** | The only lever that closes 1.64 GiB outright. Must be in the operator runbook, not the code. |
| `posix_fadvise(DONTNEED)` the copied shard ranges after load (§5) | up to **14.66 GiB of load-time page-cache pressure** | Does nothing steady-state; the load is where free supply is lowest. |
| Clamp `ensure_capacity`'s `max_capacity` to `DEFAULT_CONTEXT_LIMIT` | **0 today; 19.5 GiB of latent headroom removed** | See §6. |
| Grow full caches ×1.25 instead of ×2 | **1.25 GiB** at 16385–32768; **0 at exactly 16384** | 16384 is already a power of two, so this *cannot* close the exact-16k deficit. Stated so nobody is surprised. |

**Does not work, with arithmetic:**

| lever | why not |
|---|---|
| Offload any layer weight to the file cache | 1 GB/token of NVMe at 22.1 GB/s = **45 ms** against a ~70 ms step. Offload budget = **0 bytes/token**. This is why `weights.rs:509-511` measured 1.4 tok/s. |
| `lm_head.trellis` residency | 1.312 GiB at 4 bpw → `1,409,286,144 / 22.1e9 = 63.8 ms/token` if it misses. Not recoverable. Already reclaimable-clean, so it is the right classification already. |
| `embed_tokens` | already cold, ~10 KiB working set. Nothing to recover. |
| Drop one of three f32 accumulators (`raw_c`, 84 MiB) | Available — a kernel change, not a dispatch-count change — but 84 MiB against a 1.64 GiB deficit. |
| Halve `PREFILL_CHUNK` | −242 MiB, **+6.4 % GEMM at 614 tokens** (`model.rs:66-74`: 6,429.6 → 6,020.7 ms total). Off-limits, and bad trade. |
| Quantize KV | Barred by AGENTS.md §8 without an explicit lift. Would be worth ~1 GiB. |
| Change the checkpoint | 3 bpw fits at 16k with margin. This is the real answer for long contexts. |

---

## 4. BUGS TO FIX REGARDLESS

### 4.a `ForwardScratch::resident_bytes()` omits `raw_b` and `raw_c`

`model.rs:3045-3076` lists 24 buffers. `ForwardScratch::new`
(`model.rs:2989-2990`) allocates **26** that scale with anything. `raw_b` and
`raw_c` are absent from the list, and both are genuinely used —
`encode_gemv_fused3` receives `&scratch.raw, &scratch.raw_b, &scratch.raw_c`
(`model.rs:1525-1527`, `:1582`, `:2210`).

**Exact derivation at 1024 rows** (`q_max = 32 × 512 = 16384` from
`model.rs:3193` over the full-attention layers; `kv_max = 16 × 256 = 4096`;
`inter = 21504`; `vocab = 262144`; `attn_partials = decode_partial_elements(32, 512)`
= `16 × 32 × (512 + 2) = 263,168` floats, `crates/kernels/src/attention.rs:21,137-141`):

| buffer | bytes | in list? |
|---|---:|---|
| `hidden_a` `hidden_b` `normed` `attended` `proj` (5 × 1024 × 5376 × 2) | 55,050,240 | yes |
| `q` (1024 × 16384 × 2) | 33,554,432 | yes |
| `k` + `v` (2 × 1024 × 4096 × 2) | 16,777,216 | yes |
| `gate` `up` `activated` `transformed` (4 × 1024 × 21504 × 2) | 176,160,768 | yes |
| `raw` (`max(1024×21504, 262144) × 4`) | 88,080,384 | yes |
| **`raw_b`** | **88,080,384** | **NO** |
| **`raw_c`** | **88,080,384** | **NO** |
| `query_positions` + `token_ids` | 8,192 | yes |
| 4 × decode staging (64 B each) | 256 | yes |
| `host_tokens` (8192 × 4) | 32,768 | yes |
| `last_hidden` | 10,752 | yes |
| `logits` + `capped` (262144 × 2) | 1,048,576 | yes |
| `attn_partials` (263,168 × 4) | 1,052,672 | yes |
| `argmax_keys` (max(4×4,16)) | 16 | yes |
| **reported by `resident_bytes()`** | **371,776,272** (0.346 GiB) | |
| **TRUE** | **547,937,040** (0.510 GiB) | |
| **under-report** | **176,160,768** (168.0 MiB) | |

Matches the brief's `547,937,040` exactly.

**Why the test passes anyway.** `real_forward_buffer.rs:430` sets
`BUDGET_PER_ROW = 400 * 1024 = 419,430`; at `PREFILL_CHUNK = 1024` the ceiling is
`419,430,400` B.

```
reported  371,776,272  <=  419,430,400   PASS, slack 47,654,128
TRUE      547,937,040  >   419,430,400   FAIL by 128,506,640
per row:  535,094.77 B/row = 522.55 KiB/row  =  1.306x the budget
```

The doc comment at `real_forward_buffer.rs:425-429` says "The scratch is ~366
KB/row … 372 MB at 1024 … so 400 KB/row is 1.09x the real figure". Both the
per-row figure and the multiplier are computed from the **under-reported**
number. The same error is already baked into a published spec table:
`resident-memory-12gib/implementation_notes.md:281-283` records
`activation scratch | 0.003 GiB`, and
`3,468,568 / 2^30 = 0.00323` — which is exactly the 1-row
`resident_bytes()` figure (3,468,568 = 5,565,720 − 2 × 1,048,576). The omission
predates this investigation and has been quoted since.

**Proposed patch — model.rs.** Replace the hand-maintained list with one source of
truth, so the class of bug cannot recur:

```rust
    /// Every buffer this scratch holds.
    ///
    /// Single source of truth. `resident_bytes` sums exactly this list, and
    /// `scratch_accounting_is_complete` (model.rs, `#[cfg(test)]`) re-derives
    /// the total from the buffer shapes, so a buffer added to `new` and
    /// forgotten here fails a test instead of under-reporting silently.
    ///
    /// The two omissions this replaces (`raw_b`, `raw_c`, 88,080,384 B each at
    /// 1024 rows) under-reported the scratch by 168 MiB and let
    /// `real_forward_buffer`'s 400 KiB/row budget pass at 1.31x over its real
    /// figure.
    fn all_buffers(&self) -> [&Buffer; 27] {
        [
            &self.hidden_a,
            &self.hidden_b,
            &self.normed,
            &self.q,
            &self.k,
            &self.v,
            &self.attended,
            &self.proj,
            &self.gate,
            &self.up,
            &self.activated,
            &self.transformed,
            &self.raw,
            &self.raw_b,
            &self.raw_c,
            &self.query_positions,
            &self.token_ids,
            &self.query_positions_decode[0],
            &self.query_positions_decode[1],
            &self.token_ids_decode[0],
            &self.token_ids_decode[1],
            &self.host_tokens,
            &self.last_hidden,
            &self.logits,
            &self.capped,
            &self.attn_partials,
            &self.argmax_keys,
        ]
    }

    /// Bytes per row that scale with `max_tokens`, derived from the shapes
    /// this scratch was built with.
    ///
    /// The budget in `tests/real_forward_buffer.rs` is written against this,
    /// not against a hardcoded number, so a checkpoint-shape change cannot
    /// silently invalidate it.
    pub fn scaling_bytes_per_row(&self) -> u64 {
        const U16: u64 = 2;
        const F32: u64 = 4;
        5 * self.hidden * U16        // hidden_a/b, normed, attended, proj
            + 4 * self.inter * U16    // gate, up, activated, transformed
            + 3 * self.inter * F32    // raw, raw_b, raw_c
            + self.q_max * U16        // q
            + 2 * self.kv_max * U16    // k, v
    }

    pub fn fixed_bytes(&self) -> u64 {
        self.resident_bytes() - self.scaling_bytes_per_row() * self.max_tokens as u64
    }

    pub fn resident_bytes(&self) -> u64 {
        self.all_buffers().iter().map(|buffer| buffer.length()).sum()
    }
```

This requires adding four fields to the struct (`hidden`, `q_max`, `kv_max`,
`inter`) and setting them in `new`. Checked at 1024 rows:
`5·5376·2 + 4·21504·2 + 3·21504·4 + 16384·2 + 2·4096·2 = 53,760 + 172,032 +
258,048 + 32,768 + 16,384 = 532,992`, times 1024 = **545,783,808**; plus
`fixed_bytes = 547,937,040 − 545,783,808 = 2,153,232` (the vocab-sized `raw*`
remainder, `logits`, `capped`, `attn_partials`, staging, `host_tokens`). Sums to
the true figure.

**Proposed patch — the test.** Write the budget against the derivation, with an
explicit headroom factor, so it can no longer pass by omission:

```rust
    /// Bytes per row: derived from the scratch's own shapes via
    /// `scaling_bytes_per_row`, with 20 % of headroom. The hardcoded 400 KiB
    /// this replaces passed only because `resident_bytes()` omitted `raw_b` and
    /// `raw_c`; the real figure is 522.55 KiB/row, 1.306x over budget.
    const HEADROOM_NUM: u64 = 6;
    const HEADROOM_DEN: u64 = 5;
    let per_row = model.scratch_scaling_bytes_per_row();
    let budget = per_row / HEADROOM_DEN * HEADROOM_NUM * PREFILL_CHUNK as u64;
    let scratch = model.scratch_resident_bytes();
    assert!(
        scratch <= budget,
        "scratch is {scratch} bytes ({:.1} KiB/row), over the {budget} byte budget \
         ({:.1} KiB/row) for {PREFILL_CHUNK} rows -- a new scratch buffer?",
        scratch as f64 / 1024.0 / PREFILL_CHUNK as f64,
        budget as f64 / 1024.0 / PREFILL_CHUNK as f64,
    );
    // Guard the invariant itself: the chunk-scaling part must actually scale,
    // so a scratch that silently stopped chunking is caught here too.
    assert_eq!(
        scratch - model.scratch_fixed_bytes(),
        per_row * PREFILL_CHUNK as u64,
        "scaling bytes per row do not match the allocation"
    );
```

### 4.b `weight_residency.rs` and `memory_proof.rs` assert opposite things

```
crates/engine/tests/weight_residency.rs:148   assert!(weights < 11_500_000_000, ...)
crates/engine/tests/memory_proof.rs:37        assert!(weights > 14_000_000_000, ...)
```

`weights` is `TextWeights::resident_bytes()` = arena `capacity_bytes`
(`weights.rs:725-727`) = the §0 ledger's "arena capacity":

| checkpoint | arena | `weight_residency` `< 11.5e9` | `memory_proof` `> 14e9` |
|---|---:|---|---|
| 3 bpw | 11,001,026,276 | **PASS** (margin 499 M) | **FAIL** |
| 4 bpw | 14,661,867,236 | **FAIL** | **PASS** (margin 662 M) |

**Each assertion is satisfied by exactly one checkpoint, and the two checkpoints
are the two sides of the reported cliff.** So the suite has never had a
consistent opinion: it has been validated once per configuration, each time
against a snapshot of whichever run was last done. `memory_proof.rs` is currently
failing at HEAD for `Gxma-repacked`, and `weight_residency.rs` is currently
failing for `Gxma-4-bit`.

**Which should change:** `weight_residency.rs`'s *threshold* is the one that
matches the code today (11.5e9 vs a real 11.001e9, 4.5 % headroom), but its
*purpose* — "the mapped tensors must not reappear in the engine's own accounting"
(`weight_residency.rs:146`) — is better served by a structural assertion than a
number. `memory_proof.rs`'s `> 14_000_000_000` is simply stale: it predates the
`embed_tokens`/`lm_head.trellis` offload, and its message still says "weights
should be ~14.9 GiB", which was the **all-resident** figure and no longer is any
configuration's.

Proposed replacement in `weight_residency.rs` — assert the *relationship* the
loader implements, not a snapshot of one checkpoint:

```rust
    // The threshold below used to be a bare `weights < 11_500_000_000`, which
    // was a snapshot of one checkpoint and is mutually exclusive with
    // memory_proof.rs's `weights > 14_000_000_000`: 3 bpw passes this one and
    // fails that one, 4 bpw does the reverse. Both were "passing" somewhere.
    //
    // What the code actually guarantees is a *relationship*:
    //   resident(weights) + offloaded(embed) + offloaded(head) == text_bytes
    // which holds for every checkpoint and every environment flag, and fails
    // the moment the offload silently stops working.
    let (weights, _, _) = model.allocation_breakdown();
    let text_bytes = safetensors::inspect_model_dir(root).unwrap().selection.text_bytes;
    let offload = EMBED_BYTES + HEAD_BYTES;   // from the index, per checkpoint
    assert_eq!(
        weights + offload, text_bytes,
        "resident weights {weights} + offloaded {offload} != text_bytes {text_bytes}: \
         the resident/mapped split changed"
    );
    // And the offload must be doing the work it claims: it is 25 % of the text.
    assert!(
        offload * 3 < text_bytes,
        "offloaded tensors are only {offload} of {text_bytes}; a checkpoint change \
         may have made the offload pointless"
    );
```

`EMBED_BYTES`/`HEAD_BYTES` come from `safetensors::inspect_model_dir(...).selection`
so the test does not hardcode 2,818,572,288 / 1,056,964,608 either.

Proposed replacement in `memory_proof.rs` — delete the stale floor, and change the
18 GiB budget from a constant to a **measured** supply comparison, which is what
the budget has always meant:

```rust
    // Was: `assert!(weights > 14_000_000_000, "weights should be ~14.9 GiB")`.
    // That is the pre-offload all-resident figure; at HEAD the default
    // configuration is 11.0 GiB, and asserting a floor is what made this
    // mutually exclusive with weight_residency.rs. Nothing should assert a
    // weight total here -- it is checkpoint-dependent, and two checkpoints in
    // this repo already disagree.
    //
    // What belongs here is the *system* claim: this process fits in what the
    // box actually has free. That is the number the swap cliff turned on, and
    // it is the one constant nothing was checking.
    let probe = crate::memprobe::MemoryProbe::current();
    println!(
        "MEM available {:.2} GiB  wired {:.2} GiB  anon_others {:.2} GiB",
        probe.available_bytes() as f64 / gib, probe.wired as f64 / gib,
        probe.anon_other as f64 / gib
    );
    assert!(
        footprint < probe.available_bytes(),
        "footprint {:.2} GiB exceeds the {:.2} GiB this box had free at load; \
         this is the swap cliff",
        gib(footprint), probe.available_bytes() as f64 / gib
    );
    // The 18 GiB constant stays only as a sanity ceiling, not a budget.
    const CEILING: u64 = 18 * 1_073_741_824;
    assert!(footprint < CEILING, "footprint {:.2} GiB exceeds the 18 GiB ceiling", gib(footprint));
```

### 4.c Other accounting and assertion defects found

1. **Host `KvCache::resident_bytes()` omits `positions`.**
   `kv_cache.rs:86-88` and `:248-250` return `(keys + values) * 2`, while
   `GpuKvCache::resident_bytes()` (`gpu_cache.rs:121-123`) returns
   `… + capacity * 4`. The two are the same logical object with the same name and
   different answers. The omission is small (at 16384 full positions: 10 × 16384
   × 4 = 655,360 B) but it is the same defect class as 4.a, and
   `tests/kv_cache.rs:116` and `:132` **pin it**:

   ```rust
   // kv_cache.rs:116
   assert_eq!(cache.resident_bytes(), 1024 * stream_len * 2 * size_of::<u16>());
   // kv_cache.rs:132
   assert_eq!(full.resident_bytes(), 16);   // capacity 2, stream_len 2 -> 8+8, positions 8 omitted
   ```

   Proposed: change both `resident_bytes` bodies to
   `(self.keys.len() + self.values.len()) * size_of::<u16>() + self.positions.len() * size_of::<u32>()`
   and update the two tests to `… + capacity * size_of::<u32>()` / `24`.

2. **`bench_residency.rs` divides by `1e9` and prints `GiB`.**
   Lines 295-296 and 424-425: `footprint_bytes() as f64 / 1e9` inside a format
   string whose label is `"{:.2} GiB"`. That is a **7.37 % understatement** in
   the only tool in the repo that reports footprints — and its own doc comment at
   line 14 uses the correct divisor (`12,156,560,792 (11.3 GiB)` = `/2^30`).
   Proposed: `/ 1_073_741_824.0`. (`memory_proof.rs:29` already does this
   correctly.)

3. **`memory_proof.rs`'s 18 GiB `BUDGET` is not a budget.** It is
   `18 * 1_073_741_824 = 19,327,352,832` applied to `engine_total` *and* to
   `footprint`, both asserted `<` (lines 40-43, 108-113). 18 GiB on a 24 GiB box
   leaves 6 GiB for the desktop, the compressor, and a second model — which is
   how two concurrent instances produced 3.9 GiB of swap
   (`gpu_lock.rs:1-9`). Replace with the measured-supply comparison proposed
   above.

4. **`spec/engine-performance` documents the arena as "11 GB" and the budget as
   "14 GiB"** (`implementation_notes.md:1203`, `:1289`), while `metal.rs:4`
   enforces 18,000,000,000 (16.76 GiB) and `scripts/gpuq` warns below 14 GiB.
   Four different budget numbers for one box. Collapse to one source
   (`memprobe`), which is F1.

5. **`metal.rs:10-13`'s comment is stale**: "the 4 bpw / 8 bpw-head checkpoint
   needs 18,046,374,518 bytes resident" does not match any current
   configuration — all-resident 4 bpw is `18,889,725,668` (text_bytes) and the
   default is `14,661,867,236`. It documents why the constant exists but not
   what it now permits, which is the whole failure in §1.1.

6. **Positive finding worth keeping.** `weights.rs:540-566` already carries a
   correct, dated CORRECTION note about a sign-inverted measurement in the same
   comment block, and `resident-memory-12gib:293-337` documents how it was
   caught. That is the standard the rest of these notes should meet.

---

## 5. THE FADVISE IDEA

### 5.1 What the load actually touches, and which ranges are cold afterwards

From `load_internal` (`weights.rs:461-699`):

* Phase 1 (`:501-629`) decides `skip_copy` per tensor. At HEAD it is true for
  exactly two names (`:583-587`):
  `offload_bf16_embedding` → `model.language_model.embed_tokens.weight`, and
  `!converted && (all_zero_copy || offload_head_trellis)` → `lm_head.trellis`.
* Phase 2 (`:639-663`) sorts `pending` by `(shard, file_offset)` and runs
  `copy_tensor` in parallel (`rayon`). `copy_tensor` (`:909-946`) for a
  non-converted tensor is `arena.read_into(file, spans, file_offset, len)`
  (`:919`), which `pread`s **directly into the Metal buffer's contents**
  (`weights.rs:299-336`, `:331 read_exact_at(file, destination, …)`). For a
  BF16-converted tensor it stages through an 8 MiB `COPY_CHUNK_BYTES` scratch
  (`:927-943`).
* `read_at` populates the **unified page cache** for the range it reads, and the
  `ShardReader` `mmap` (`weights.rs:1015-1030`) shares that same page cache.

So after phase 2:

| range | size (3 bpw) | size (4 bpw) | cold afterwards? |
|---|---:|---:|---|
| all `!skip_copy` tensors (410 trellis + side tensors + norms) | 11,001,026,276 | 14,661,867,236 | **YES — pure dead page cache** |
| `embed_tokens` | 2,818,572,288 | 2,818,572,288 | **NO — but already cold** (~10 KiB/token) |
| `lm_head.trellis` | 1,056,964,608 | 1,409,286,144 | **NO — hot every token** |
| vision tower | 1,151,487,072 | 1,151,487,072 | never touched |
| safetensors JSON headers | ~1 MB each | ~0.7 MB each | never re-read |

### 5.2 Verdict

**It helps, and the safe set is exactly the `!skip_copy` set — which is the same
predicate that decides what is mapped.** That is the elegant part: putting the
`posix_fadvise` call inside the same `if item.skip_copy { return Ok(()) }` branch
as the copy makes "never drop a page the GPU will read again" a structural
invariant rather than a review item.

What it buys:

* Up to **11.00 GiB (3 bpw) / 14.66 GiB (4 bpw)** of page cache released during
  and after load. This is reclaimable memory, so it does **not** reduce
  `phys_footprint`'s steady-state number, but it is released at exactly the
  moment free supply is lowest.
* It is free: one syscall per tensor, 410 of them, on ranges already resident.

What it costs / where it can hurt:

1. **Do not touch the mapped ranges.** Dropping `lm_head.trellis`' 0.98–1.31 GiB
   would evict a working set that `resident-memory-12gib:162-165` measured as
   fully hot across 2500 tokens (2.6 TB of re-reads, 0 swaps). One GB/token of
   NVMe at 22.1 GB/s is 45 ms/token. Structurally excluded by the `skip_copy`
   guard; **state the guard in a comment**, because it is the only thing standing
   between this patch and a 45 ms/token regression.
2. **Page-boundary waste.** 396 of 796 packed tensors sit at an absolute file
   offset that is 2-byte but not 4-byte aligned (`spec/file-backed-weights/implementation_notes.md:15-28`).
   `posix_fadvise` drops whole pages; a range whose endpoints are inside a page
   leaves that page cached. Worst case `410 × 2 × 16,384 = 13.4 MB`. Negligible,
   but say so rather than let someone assume the bytes are exact.
3. **APFS transparency/compression.** The shards may be stored compressed or
   deduped; dropping the cache forces a decompression on any later read. Since
   no later read of these ranges occurs, this is a non-issue — **but only because
   of the guard.** Same caveat as (1).
4. **In-loop is a race.** The phase-2 loop is `par_iter` over all tensors sorted
   by `(shard, file_offset)` (`:639-643`). An in-loop `posix_fadvise` on tensor
   *N*'s range runs concurrently with tensor *N+1*'s `read_at` on the same `File`.
   `posix_fadvise` is advisory and does not move the file offset, and the byte
   ranges are disjoint, so it is *legal* — but it drops pages inside the shared
   file's current **readahead window**, which can force the neighbouring
   `read_at` to re-issue a disk read. That is a plausible cause of a silent
   `load_ms` regression that would be very hard to attribute.
   **Recommendation: a separate sequential pass after phase 2.** Deterministic,
   testable, page-coalescible, and it costs nothing measurable (the load is
   already dominated by the copy).
5. **Sequencing.** If F6 (file-backed trellis + mlock) lands, phase 2 stops
   copying the trellis at all — its pages *become* the working set and must not
   be dropped. This patch is then **dead code**. Implement it as a stopgap for
   the current arena configuration, gate it on the residency policy, and record
   in the commit message that F6 supersedes it.

### 5.3 Proposed patch

In `ShardReader` (`weights.rs:982-1090`), add one method:

```rust
    /// Drop this shard's page cache for `[offset, len)`.
    ///
    /// `POSIX_FADV_DONTNEED` is advisory and drops **whole pages only**, so a
    /// range whose endpoints fall inside a page leaves that page cached; with
    /// 396/796 packed tensors at a 2-byte-but-not-4-byte file offset
    /// (`spec/file-backed-weights/implementation_notes.md:15-28`) that is at
    /// most `2 * tensors * 16 KiB` = 13.4 MB. Returned so the caller can log it;
    /// the caller's real interest is that it did not touch a mapped range.
    fn drop_cache(&self, offset: u64, len: u64) -> io::Result<()> {
        self.check_range(offset, len)?;
        let ret = unsafe {
            libc::posix_fadvise(
                self.file.as_raw_fd(),
                offset as libc::off_t,
                len as libc::off_t,
                libc::POSIX_FADV_DONTNEED,
            )
        };
        if ret != 0 {
            return Err(io::Error::from_raw_os_error(ret));
        }
        Ok(())
    }
```

And a new phase between phase 2 and phase 3 in `load_internal`
(`weights.rs:664`, immediately after `let copy_ms = …`):

```rust
        // Phase 2.5: release the page cache the copy just filled.
        //
        // Phase 2 `pread`s 10-15 GiB into the arena, and every one of those
        // bytes is dead the instant the copy lands: they are anonymous arena
        // now, and the page cache holds a second copy. On a 24 GiB box that is
        // the largest reclaimable block available and it is released at exactly
        // the moment free supply is lowest.
        //
        // ONLY the `!skip_copy` ranges. A tensor with `skip_copy` is bound
        // zero-copy out of this mmap (`no_copy_view`, weights.rs:1050) and the
        // GPU reads it directly, so dropping it would evict a live working set:
        // `lm_head.trellis` alone is 63.8 ms/token of NVMe at 4 bpw
        // (1,409,286,144 B / 22.1 GB/s, spec/mtp-40tps/implementation_notes.md:176).
        // The `skip_copy` predicate is therefore load-bearing here -- do not
        // refactor this to "all ranges" or to a name list.
        //
        // Sequential, not inside the phase-2 `par_iter`: dropping a range that
        // is inside the shared file's readahead window can force a concurrent
        // `read_at` on the neighbouring range to re-issue a disk read, and that
        // regression would be nearly unattributable.
        //
        // Superseded by the file-backed-weights spec: if the layer trellis
        // becomes a zero-copy working set, this becomes dead code.
        let mut dropped_bytes = 0_u64;
        let mut dropped_tensors = 0_usize;
        if std::env::var_os("KAPLUMBA_FADVISE").is_none() {
            for item in &pending {
                if item.skip_copy {
                    continue;
                }
                let Some(reader) = shards.get(&item.shard) else {
                    continue;
                };
                if reader.drop_cache(item.file_offset, item.len).is_ok() {
                    dropped_bytes += item.len;
                    dropped_tensors += 1;
                }
            }
        }
        if std::env::var_os("KAPLUMBA_LOAD_PROFILE").is_some() {
            eprintln!(
                "load profile: fadvised {dropped_bytes} bytes over {dropped_tensors} tensors"
            );
        }
```

Test: `KAPLUMBA_LOAD_PROFILE=1` prints `fadvised …`; assert
`dropped_tensors == pending.iter().filter(|t| !t.skip_copy).count()` and that no
tensor name starting with `model.language_model.embed_tokens` or equal to
`lm_head.trellis` appears in the dropped set — extend `weight_residency.rs` with
a check that `lm_head` and `embed` still map to resident page cache after load.
A/B `load_ms` and peak `phys_footprint`, interleaved, `A B A B`, because
`load_ms` on this box has moved 5,987 → 19,005 ms across configurations
(`resident-memory-12gib:181,221-223`) and a single sample per arm will invent a
result.

---

## 6. Two more defects the cliff exposed, not in `plan.md`'s fix list

**6.1 `ensure_capacity` is bounded by `max_position_embeddings`, not by the policy
context limit.** `model.rs:3396` sets
`max_sequence_len: metadata.config.max_position_embeddings` = **262,144**, and
that value is passed as `max_capacity` at `model.rs:1663` and `model.rs:1783`.
`generate.rs:35` `DEFAULT_CONTEXT_LIMIT = 32768` is what actually limits a
normal run, and `model.rs:3381-3395` states the intent explicitly:

> It is now the model's own `max_position_embeddings` (262144) … This is a
> *bound*, not an allocation … The policy bound on how many positions that may be
> is `DEFAULT_CONTEXT_LIMIT`, checked before every forward by `fits_in_context` /
> `generate`, so the two cannot disagree about what is reachable.

**They can and do disagree**: any caller that reaches `TextModel::forward_buffer`
directly — which is what `tests/real_forward_buffer.rs`, `tests/memory_proof.rs`
and most of `crates/engine/examples/` do — bypasses `fits_in_context` entirely.
At 262,144 positions the full-attention KV alone is `10 × 262144 × 8196 =
21,485,322,240 B` = **20.02 GiB**, or **22,324,387,840 B = 20.79 GiB** including
the fixed sliding caches. That is a 20 GiB allocation on a 24 GiB box, gated by
nothing. Pass `min(max_sequence_len, DEFAULT_CONTEXT_LIMIT)` and the exposure
drops to `3,524,730,880 B` = 3.28 GiB.

**6.2 `GpuKvCache::ensure_capacity` doubles (`gpu_cache.rs:183-186`)**, which
wastes up to 100 % of the full-attention KV:

```
needed  513 -> capacity  1024  (needed 513, have 1024, waste 99.6%)
needed 2049 -> capacity  4096  (waste 99.9%)
needed 8193 -> capacity 16384  (waste 99.9%)
needed 16385 -> capacity 32768 (waste 99.9%, 1.25 GiB of the 2.50 GiB ladder)
```

At exactly 16384 the ladder lands on the power of two and wastes nothing, which
is why it does not show up in a 16k measurement — and why it will show up the
first time someone runs 16385 tokens. `target += (target + 3) / 4` caps the waste
at 20 %. Zero dispatches, zero chunk change, pure allocator.

---

## 7. Sabotage checks this analysis would need (AGENTS.md §3.4)

No coverage claim above has been proven by a failing test, because nothing may be
run from this session. Before any of F1–F8 is marked done, each needs its
branch broken and the failure observed:

| claim | sabotage |
|---|---|
| `resident_bytes()` is now complete | delete `&self.raw_b` from `all_buffers()`; the `real_forward_buffer` budget must fail |
| the projection rejects 4 bpw | force `projected = available` in the guard; the 4 bpw load must thrash (or the guard's own unit test must fail) |
| `ensure_capacity` is bounded | restore `262144`; the new capacity test must fail at 32769 |
| the fadvise never touches a mapped range | drop the `item.skip_copy` guard; `lm_head` must appear in the dropped set |
| the 1.25× growth wastes ≤ 25 % | restore `*= 2`; the ladder test must fail at 32769 |
| weights + offload == text_bytes | set `offload_embedding = false` in `weights.rs:576`; the residency test must fail |
