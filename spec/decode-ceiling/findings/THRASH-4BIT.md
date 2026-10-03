# THRASH-4BIT — why `Gxma-4-bit` decodes ~1000× slower than `Gxma-repacked`

**Status:** measurement only. No production code was changed. Two probe files
added under `crates/engine/tests/`, both `probe_thrash_`-prefixed:
`probe_thrash_faults.rs`, `probe_thrash_budget.rs`.

**Verdict: H2 survives. H1 is refuted. H3 is refuted as the cause (the box was
quiet). H4 is refuted.**

Every number below is labelled **MEASURED** or **REASONED**. MEASURED figures
came from the commands quoted beside them, on this box, 2026-10-03, from the
same binary (`2eb7b4d`, rebuilt — `target/release/kaplumba` was stale).

---

## 0. TL;DR

| | 3 bpw `Gxma-repacked` | 4 bpw `Gxma-4-bit` | ratio |
|---|---:|---:|---:|
| ms / decode token | **62–63** | **78,422** | **1245×** |
| tok/s | **15.87–16.13** | **0.0115–0.0146** | ~1150× |
| `ru_majflt` per token | **0** | **0** | 1× |
| `ru_minflt` per token | 4.1 | 742.7 | 181× |
| host `decompressions` per token | **0.9 MiB** | **18.12 GiB** | **20,600×** |
| host `swapins` per token | 1.5 MiB | **15.68 GiB** | ~11,000× |
| host `compressions` per token | 0 | **18.32 GiB** | ∞ |
| `peak memory footprint` | 12,454,929,720 (11.60 GiB) | 17,443,010,360 (**16.24 GiB**) | 1.40× |
| `peak resident set size` (`ru_maxrss`) | 9,233,514,496 (8.60 GiB) | 9,528,508,416 (8.87 GiB) | 1.03× |

**The whole 1000× is one thing:** at 4 bpw the kernel compresses and
decompresses the entire weight set *every token* — 18.32 GiB compressed and
18.12 GiB decompressed per token against a 14.95 GiB working set. That is a
~100 % miss rate. Achieved bandwidth is **195 MB/s** where 3 bpw achieves
**193 GB/s** — a factor of 990.

And the reason it is missing at all is capacity: the process's anonymous
footprint is **16.24 GiB** and this box will not give it that much without
reclaiming.

---

## 1. The decisive measurement, and a correction to the instrument

The brief asked for "major/minor fault rate per token". I measured it, and
**the process fault counters are the wrong instrument on macOS**:

```bash
KAPLUMBA_TEST_MODEL=$HOME/.kaplumba/models/Gxma-4-bit \
/usr/bin/time -l ./target/release/deps/probe_thrash_faults-* --ignored --nocapture
```

**MEASURED, `Gxma-4-bit`, 12 decode steps:**

```
PROBE STEP  ms      minflt   majflt   maj_MiB   tok_s
PROBE    0   68432      644        0       0.0   0.0146
PROBE    1   72029      657        0       0.0   0.0139
PROBE    2   73871     1088        0       0.0   0.0135
PROBE    3   79519      933        0       0.0   0.0126
PROBE    4   74523      797        0       0.0   0.0134
PROBE    5   74692      712        0       0.0   0.0134
PROBE    6   79590      699        0       0.0   0.0126
PROBE   10   86623      660        0       0.0   0.0115
PROBE   11   85726      678        0       0.0   0.0117
PROBE DECODE ms=941067 minflt=8912 majflt=0 tok_s=0.0128
PROBE PER_TOKEN ms=78422.2 minflt=742.7 majflt=0.0 maj_MiB_per_token=0.0
PROBE DECODE_HOST pageins=195472 pageouts=180
                 swapins=12327947 swapouts=12404608
                 compressions=14403397 decompressions=14251802
```

**`ru_majflt` is exactly 0 for every single token of a run that took 78 seconds
per token.** So a "page faults per token" gate would have exonerated memory
entirely and blamed the kernels (H4). The paging is done by the kernel on the
process's behalf through the GPU/IOAccelerator path, so it lands in the
**host** counters, not the process's.

Divided by 12 tokens, MEASURED:

| counter | pages/token | bytes/token (× 16,384 B) |
|---|---:|---:|
| `compressions` | 1,200,283 | **18.32 GiB** |
| `decompressions` | 1,187,650 | **18.12 GiB** |
| `swapins` | 1,027,329 | **15.68 GiB** |
| `swapouts` | 1,033,717 | **15.77 GiB** |
| `pageins` | 16,289 | 0.25 GiB |
| `pageouts` | 15 | 0.24 MiB |

The weight set actually re-read per token is 13,638 MiB of layer trellis +
1,344 MiB of head = **14,951 MiB**. The kernel is decompressing **18.12 GiB =
18,563 MiB per token against a 14,951 MiB working set**, i.e. it re-misses
*within* a single token.

**MEASURED, `Gxma-repacked`, same binary, same probe, 12 decode steps:**

```
PROBE STEP  ms      minflt   majflt
PROBE    1      62        2        0
PROBE    5      62        1        0
PROBE   11      63        1        0
PROBE DECODE ms=838 minflt=49 majflt=0 tok_s=14.3198
PROBE PER_TOKEN ms=69.8 minflt=4.1 majflt=0.0 maj_MiB_per_token=0.0
PROBE DECODE_HOST pageins=9 pageouts=0 swapins=112 swapouts=0
                 compressions=0 decompressions=667
```

667 decompressions over 12 tokens = **56 pages = 0.9 MiB per token**, i.e.
0.005 % of the weight set. **Zero swap, zero compression.** Decode at 3 bpw is
100 % RAM-resident, and it was measured on a box that had only **4.04 GiB of
reclaimable supply** at load time (see §5), which is itself the cleanest
possible control.

> **How I would know this is wrong.** If the 4 bpw swap/compression traffic were
> caused by something other than this process's own footprint — another agent's
> model instance, a Spotlight index of the 20 GB checkpoint, a backup — then
> `footprint` on the process would not show ~9 GiB of its own
> `IOAccelerator (graphics)` pages in the *swapped* column. It does
> (§3). Conversely, if `ru_majflt = 0` were an artefact of the probe rather than
> of macOS, then a run with the checkpoint on a *local SSD* vs a slower volume
> would move `ru_majflt`. There is only one volume here, so that check was not
> run; the host counters plus `footprint`'s per-region swapped column are
> independent of `getrusage` and agree.

---

## 2. H1 — `lm_head.trellis` is file-backed and re-read from SSD every token

### **REFUTED. Three independent lines of evidence.**

#### 2a. The code says the head is anonymous, for `Gxma-4-bit` specifically

`weights.rs:585-587` — `lm_head.trellis` is the one tensor that takes the
`skip_copy` path by default:

```rust
let offload_head_trellis = offload_head && name == "lm_head.trellis";
let skip_copy = offload_bf16_embedding
    || (!converted && (all_zero_copy || offload_head_trellis));
```

`skip_copy = true` ⇒ `spans.is_empty()` ⇒ `model.rs:4934-4952`:

```rust
let copied = weights.tensors().get(&trellis_name).is_some_and(|t| !t.spans.is_empty());
let (packed, packed_offset) = if copied {
    (weights.tensor_buffer(&trellis_name)?, 0)      // arena, offset 0
} else {
    let view = weights.zero_copy_tensor_buffer(&trellis_name)?…;
    align_packed_view(&device, &view.0, view.1, len, &trellis_name)?   // ← takes this
};
```

`align_packed_view` (`model.rs:4871-4921`) early-returns `(buffer, offset)` when
`offset % 4 == 0`, and otherwise allocates a **fresh anonymous buffer** and
`memcpy`s the tensor into it.

MEASURED, from the safetensors header of
`~/.kaplumba/models/Gxma-4-bit/model-00003-of-00003.safetensors`:

| | bytes |
|---|---:|
| header length | 58,577 |
| data start (`8 + header_len`) | 58,585 |
| `lm_head.trellis` `data_offsets[0]` (relative) | 507,613,756 |
| **absolute offset** | **507,672,341** |
| absolute offset mod 4 | **1** |
| absolute offset mod 4,096 (the mmap window delta, `weights.rs:1061-1062`) | **1,813** |
| **`lm_head.trellis` `n_bytes`** | **1,409,286,144** |

Same table for `Gxma-repacked/model-00002-of-00002.safetensors`: absolute offset
**5,321,075,520**, mod 4 = **0**, `n_bytes` = 1,056,964,608.

So: **3 bpw's head is file-backed (offset already 4-aligned, early return);
4 bpw's head is anonymous** (offset 1,813, misaligned, copied at load).

**The `507,672,341` in the brief is the head's file *offset*, not its size.** Its
size is **1,409,286,144 bytes**. Anyone reasoning from "507 MB head" is off by
2.78×.

#### 2b. The kernel's residency, measured

`footprint` and `vmmap` on a live 4 bpw process, mid-decode:

```
$ /usr/bin/footprint -p <pid> --swapped --wired
  Dirty  (Swapped)      Clean  Reclaimable    (Wired)    Regions    Category
    ---        ---        ---          ---    ---        ---        ---
  16 GB    9056 MB        0 B          0 B    5838 MB       1388    IOAccelerator (graphics)
 120 MB      34 MB        0 B          0 B        0 B         11    Malloc Large
 104 MB        0 B        0 B          0 B      104 MB         26    Owned physical footprint (unmapped) (graphics)
  53 MB      39 MB        0 B     1008 KB        0 B         73    Malloc Small
9518 KB        0 B        0 B          0 B    9518 KB          1    page table
    0 B        0 B        0 B          0 B    2688 MB         11    mapped file
    ---        ---        ---          ---    ---        ---        ---
  16 GB    9131 MB    2048 KB      1008 KB    8641 MB       4465    TOTAL
```

```
$ /usr/bin/vmmap -summary <pid>
Physical footprint:         16.1G
Physical footprint (peak):  16.2G
IOAccelerator (graphics)   15.8G virtual  15.8G resident  5.9G dirty  9.9G swapped   1388 regions
mapped file                 19.4G virtual      0K resident      0K dirty      0K swapped     10 regions
TOTAL                       423.0G virtual   16.2G resident  7.2G dirty  8.9G swapped
```

**`mapped file` contributes 0 B dirty, 0 B clean, 0 B reclaimable.** All ten
file mappings — including `embed_tokens` (2,818,572,288 B) and the head's own
source mapping — hold **zero resident pages**. Nothing is being served from the
file cache, so nothing is being re-read from SSD.

Everything resident is `IOAccelerator (graphics)`: **anonymous, 9.9 GiB of it
swapped out, 5.9 GiB dirty, 5.7 GiB wired.** `align_packed_view`'s output buffer
is a Metal buffer, so it is in that category, indistinguishable from the arena.

#### 2c. The head is *worse* off than H1 assumed, not better

H1's mechanism ("read in full every token, so if it is mapped and evicted that
is a per-token disk round trip") is real arithmetic — **1,409,286,144 B /
22.1 GB/s NVMe = 63.8 ms/token**, against a ~83 ms memory-bound step
(`memory-swap-cliff/implementation_notes.md:337`). But it does not apply here,
because the head is not file-backed: it is **1.312 GiB of anonymous memory
competing for the same ~15 GiB as the 13.655 GiB arena.** Being anonymous makes
it part of the thrash rather than an escape from it.

For `Gxma-repacked` the head *is* file-backed, and that is precisely why 3 bpw
works: 1,056,964,608 B of reclaimable clean page cache instead of anonymous
memory.

> **How I would know 2a/2b are wrong.** If `align_packed_view` were not reached
> for this checkpoint — e.g. if a future loader change gave `lm_head.trellis` an
> arena span — then the head would be a `StorageModeShared` buffer over the
> mapping and would show as `mapped file` resident rather than as
> `IOAccelerator (graphics)`. It does not: `mapped file` is 0 B. Conversely, if
> `PAGE_ALIGN` in `weights.rs` were changed from 4,096 to something ≥ the offset,
> the misalignment would vanish and the copy would not happen. `PAGE_ALIGN` is
> read at `weights.rs:1061` and the observed delta of 1,813 is exactly
> `507,672,341 mod 4,096`, which confirms the path taken.

---

## 3. H2 — the 14.6 GB anonymous layer trellis thrashes; macOS keeps all of it or nothing

### **SURVIVES. This is the cause.**

**MEASURED** — the anonymous accounting, from the probe's own walk of the
safetensors selection plus `GpuKvCache::resident_bytes()`:

| | 3 bpw | 4 bpw |
|---|---:|---:|
| text selection | 14,876,563,172 (13.855 GiB) | 18,889,725,668 (17.592 GiB) |
| layer `.trellis` | 10,982,522,880 (10.228 GiB) | 14,643,363,840 (13.638 GiB) |
| `lm_head.trellis` | 1,056,964,608 (0.984 GiB) | 1,409,286,144 (1.312 GiB) |
| `embed_tokens` (mapped) | 2,818,572,288 (2.625 GiB) | 2,818,572,288 (2.625 GiB) |
| **arena (anonymous)** | 11,001,026,276 (10.246 GiB) | 14,661,867,236 (13.655 GiB) |
| **+ `align_packed_view` copy (anonymous, 4 bpw only)** | 0 | **1,409,286,144 (1.312 GiB)** |
| + KV @512 positions (REASONED, `gpu_cache.rs:165-167`) | 881,029,120 (0.821 GiB) | 881,029,120 (0.821 GiB) |
| + `ForwardScratch` incl. `raw_b`/`raw_c` (MEASURED at `scratch_max_tokens=2`) | ~5.8 MB | ~5.8 MB |
| **anonymous total** | **11,887,860,000 (11.07 GiB)** | **16,957,981,000 (15.79 GiB)** |
| **MEASURED `peak memory footprint`** | 12,454,929,720 (11.60 GiB) | 17,443,010,360 (**16.24 GiB**) |
| unaccounted residual | 567,069,720 (0.53 GiB, 4.6 %) | 485,029,360 (0.45 GiB, 2.8 %) |

**There is no missing 5 GiB.** The residual is ~0.5 GiB in both, and `footprint -v`
itemises it: `Owned physical footprint (unmapped) (graphics)` 104 MB,
`Malloc Small` 53 MB, `page table` 9.5 MB, `__TEXT` 91 MB, `__LINKEDIT` 15.8 MB,
`__AUTH_CONST` 4.4 MB, `__OBJC_RO` 24.9 MB, `__DATA*` ~8 MB, 1,388 Metal regions
each rounded to 16 KiB. **Every byte of the 4 bpw footprint is accounted for.**

**The gap that matters:** 15.79 GiB of anonymous demand at 4 bpw against
**18.11 GiB of measured reclaimable supply** (free + inactive + purgeable +
speculative, `host_statistics64(HOST_VM_INFO64)`, sampled by the probe at the
instant before load) — and `MEM: peak memory footprint` says the OS ended up
holding 16.24 GiB of it *plus* whatever the desktop needed, at
`PhysMem: 23G used (3313M wired, 12G compressor), 86M unused`.

`12G compressor` is the sentence. macOS did the only thing it can: it compressed
11.17 GiB of the process (`PROBE END compressor_GiB=11.174`) and swapped
1.83 GiB of it, and then had to pay for that on every single token.

**MEASURED bandwidth, and why H4 cannot explain it:**

```
3 bpw: 11.212 GiB weight bytes / 0.0623 s  = 193 GB/s   (203 GB/s is the best
                                                    kernel ever measured here)
4 bpw: 14.951 GiB weight bytes / 78.422 s   = 195 MB/s
                                            ^^^^^^^^^
                                            990× worse
```

If 4 bpw were merely doing 1.334× more bytes than 3 bpw through the same
kernels, it would run at 12.0 tok/s. It runs at 0.0128. The gap is **944×**, and
it is exactly the swap round trip. H4 — a per-dispatch or per-layer cost scaling
with bit rate — predicts a ratio near 1.33 and is refuted.

**Prefill collapses the same way**, which rules out anything decode-specific:

| | 3 bpw | 4 bpw |
|---|---:|---:|
| prefill tok/s (28-token prompt) | **20.99** | **0.806** |
| prefill ms | 1,334 | 34,758 |
| prefill `ru_majflt` | 0 | **0** |
| prefill host `swapins` | 20 | **535,704 (8.78 GiB)** |
| prefill host `compressions` | 53 | **1,167,201** |

**MEASURED load cost**, same story — the collapse starts before a single token:

| | 3 bpw | 4 bpw |
|---|---:|---:|
| `load_ms` | **7,867** | **47,392** (CLI runs: 90,970 / 102,319) |
| load `ru_majflt` | 236,896 (3.62 GiB) | 258,308 (3.94 GiB) |
| load host `compressions` | 495,118 | **3,078,552** |
| load host `swapouts` | 17,827 | **875,912 (13.67 GiB)** |
| load Δ compressor | — | **+11.10 GiB** |

The 4 bpw load writes 14.66 GiB of arena into a box that cannot hold it, and the
kernel starts evicting the *freshly written* pages while the copy is still
running. 47 s instead of 7.9 s.

> **How I would know this is wrong.** The claim is "4 bpw exceeds supply, so it
> thrashes." The falsifier is a run where 4 bpw's anonymous demand is below the
> measured supply and it still runs slowly. That is exactly what
> `KAPLUMBA_RESIDENT_WEIGHTS=1` is *not* — it raises demand — so the honest test
> is the opposite direction: cut 4 bpw's anonymous bytes by ≥1.3 GiB and see
> tok/s jump by three orders of magnitude. §6 prices the candidates; the
> alignment repack is the one that removes real bytes without touching quality.

---

## 4. H3 — it is environmental; a contended box

### **REFUTED as the cause. The box was quiet.**

`scripts/gpuq status` immediately before the 4 bpw run:

```
gpu: none (stale lock, pid 49304 is gone)
queue: 0 waiting
memory: 17.0 GiB available (reclaimable), 2.1 GiB in use by processes, 0 MiB swap
ok: enough available memory for a model load
```

MEASURED inside the 4 bpw probe, at the instant before load:

```
PROBE HOST page_size=16384 page_total=1572864 total_GiB=24.000
       baseline_supply_GiB=18.110
```

18.11 GiB reclaimable, no other model instance, no queue. The slowdown
reproduced *exactly* — 0.0115–0.0146 tok/s, 68–87 s/token — on the quietest box
state I could produce.

Two further controls:

* **The 3 bpw run reproduced its 16.29 tok/s** (`decode_tok_s=16.286`) from the
  same binary, minutes earlier, with the same page cache and the same desktop.
* **The 3 bpw probe ran with only 4.04 GiB of supply** — a *worse* box state
  than 4 bpw ever saw — and hit 15.87–16.13 tok/s per step with zero swap. So
  low supply alone is not sufficient to cause the collapse; *demand exceeding
  supply* is.

**H3 is not entirely wrong about one thing**, and it is worth keeping: the
earlier numbers in the brief (0.014–0.066 tok/s) were taken on a contended box
and the *upper* end of that range, 0.066 tok/s, is ~5× better than what I
measure on a quiet box. Environmental contention makes 4 bpw somewhat less
catastrophic, because a busier box has already surrendered its file cache and
the marginal reclaim is cheaper. It does not make 4 bpw usable.

`swap` note, per AGENTS.md §10.3: the swap counter on this box was **1,930 MB
used at idle** and never returns promptly after a process exits. It is only
evidence when it *grows*. MEASURED growth during runs:

| run | swap before | swap peak | swap after |
|---|---:|---:|---:|
| 4 bpw CLI, 4 tokens | 1,561.69 MB | — | 1,865.25 MB (**+303.56**) |
| 4 bpw CLI, 6 tokens | 1,561.69 MB | **3,497.81 MB** | 1,946.31 MB (**+384.62**) |
| 4 bpw probe, 12 tokens | ~1,832 MB | **4,438.56 MB** | 1,831.81 MB (net 0, but 3.3 GB of swap *total* was swapped in and out: 12,404,608 `swapouts`) |
| 3 bpw CLI, 11 tokens | 1,633.69 MB | — | 1,609.69 MB (**−24.00**) |
| 3 bpw probe, 12 tokens | 1,930.31 MB | — | 1,576.62 MB (**−353.69**) |

The 3 bpw runs *returned* swap; the 4 bpw runs grew it. That is the AGENTS.md
§10.3 test, passed.

---

## 5. The true memory budget of this box

**PARTLY MEASURED, see `probe_thrash_budget.rs` output in §9.**

Two independent MEASURED supply figures from the engine's own runs, both
`host_statistics64(HOST_VM_INFO64)`:

| when | reclaimable supply (free+inactive+purgeable+speculative) | outcome |
|---|---:|---|
| 4 bpw probe, before load | **18.11 GiB** | 16.24 GiB footprint → thrash |
| 3 bpw probe, before load | **4.04 GiB** | 11.60 GiB footprint → 16.29 tok/s, zero swap |

Those two rows look contradictory and are not: the second is a *post-thrash*
reading, because macOS had not finished decompressing what the previous 4 bpw
run left behind (that is what `12G compressor` in the `top` line was). The
lesson is that "supply" must be sampled at the same instant as the load, from
`host_statistics64`, and not from `top`'s `unused` — which is what
`memory_proof.rs` does and what `probe_thrash_faults.rs` now prints.

`top -l 1 -n 0 | grep PhysMem` on this box, healthy, desktop running:

```
PhysMem: 6524M used (2708M wired, 134M compressor), 17G unused.
```

`17G unused` ≈ 17.26 GiB, and `vm.user_wire_limit` = 19,585,050,869 B
(18.24 GiB) is the hard ceiling on what can be wired.

**MEASURED threshold (from the budget probe):** see §9.

**What is needed for 4 bpw to fit, in bytes:**
anonymous demand 16,957,981,000 B (15.79 GiB) must come down to under the
measured threshold *with the desktop running*. 3 bpw's demand is
11,887,860,000 B (11.07 GiB) and it has ~5 GiB of slack. **The gap is
5,070,121,000 B (4.72 GiB).**

---

## 6. Memory reductions, priced honestly

Split into what **reduces the footprint** (bytes of RAM the process holds) and
what only **reduces the risk** (moves bytes somewhere reclaimable, or makes the
failure visible).

### Reduces the footprint

| # | change | bytes | quality risk | note |
|---|---:|---:|---|---|
| **R1** | **Repack the checkpoint so `lm_head.trellis` starts 4-byte aligned** | **−1,409,286,144 (−1.312 GiB)** | **none** — bit-identical weights | **The big one. Recipe below.** |
| R2 | Repack `lm_head` to a 6 bpw EXL3 trellis (the `Gxma-repacked` layout: `colmajor`, shape `[336,16384,96]`, 1,056,964,608 B) | −352,321,536 (−336 MiB) | head precision 8→6 bpw | The brief's own evidence says 4 bpw layers + 6 bpw head is byte-identical to 4 bpw layers + 8 bpw head, so this is very likely identity — **but that is REASONED from one prompt and must be gated on greedy-identity over a corpus.** |
| R3 | Repack `lm_head` to 4 bpw | −704,643,072 (−672 MiB) | head precision 8→4 bpw | Bigger risk. Note the file's shape scales exactly with rate (96 int16 at 6 bpw → 128 at 8 bpw), so 4 bpw would be shape `[336,16384,64]` = 704,643,072 B. |
| R4 | Drop one f32 accumulator (`raw_b` **or** `raw_c`) from `ForwardScratch` | −88,080,384 (−84 MiB) at `max_tokens=1024`; **−2,097,152 (−2 MiB) at `max_tokens=2`** | none | **F5 confirmed as a bug** (`model.rs:3045-3076` lists 24 buffers, `ForwardScratch::new` allocates 26). But see the measured `scratch_max_tokens = 2` — at short context this is worth **2 MiB, not 176 MB**. It only matters at large prefill chunks. |
| R5 | `raw_b`/`raw_c` aliasing | up to −88,080,384 more | none | Needs the kernels' liveness checked: `encode_gemv_fused3` takes `&raw, &raw_b, &raw_c` together (`model.rs:1525-1527`), so aliasing is only safe if the fused3 path never has all three live. **REASONED, not checked.** |

#### R1 recipe — one-time checkpoint repack, no code change, no quality change

Measured facts this rests on:

* `Gxma-4-bit/model-00003-of-00003.safetensors`, header 58,577 B, data start
  58,585, `lm_head.trellis` relative `data_offsets = [507613756, 1928342900]`,
  **absolute start 507,672,341 ≡ 1 (mod 4)**.
* `lm_head.trellis` is the **last** tensor in that shard.
* `crates/engine/src/safetensors.rs:338-380` validates only
  `absolute_end <= file_size` and **non-overlap**. It does **not** require
  contiguity, and it does not require any offset to be aligned. A gap is legal.

Recipe:

1. Rewrite shard 3 to a new file. Copy bytes `[0, 507_672_341)` verbatim —
   header and everything before the head.
2. Insert **3 zero bytes**.
3. Copy `lm_head.trellis`'s 1,409,286,144 bytes.
4. Append 3 zero bytes so the file grows by 3.
5. In the header, change only `lm_head.trellis.data_offsets` to
   `[507613759, 1928342903]`. Leave every other tensor's offsets alone.
6. New absolute start = 507,672,344 ≡ **0 (mod 4)**.

After that, `align_packed_view` takes its `offset % EXL3_PACKED_ALIGNMENT == 0`
early return (`model.rs:4891-4893`) and returns the page-aligned mmap window:
**the head becomes file-backed, reclaimable-clean page cache, exactly like
3 bpw's head.** Anonymous drops from 15.79 GiB to **14.48 GiB**, which is below
3 bpw's 16.29 tok/s footprint plus 2.9 GiB of headroom.

Verification, cheap: run with `KAPLUMBA_DEBUG_OFFSETS=1`
(`model.rs:4962-4967`) and assert `OFFSET lm_head.trellis packed_offset=… aligned4=true`;
then confirm `footprint` shows `mapped file` with a non-zero `Reclaimable`
column and `IOAccelerator (graphics)` down by 1,344 MiB.

**Cost**: one 3 GB rewrite. **Do not do it in this task** — the brief says
report, do not fix.

### Reduces the risk, not the footprint

| # | change | bytes moved | what it does |
|---|---:|---:|---|
| R6 | Move `lm_head.trellis` back to the arena (`KAPLUMBA_RESIDENT_WEIGHTS=1`) | +1,409,286,144 anonymous, **−2,818,572,288** from the reclaimable column | **Worse, not better**: it also copies `embed_tokens` (2.625 GiB) into the arena. Total anonymous at 4 bpw becomes 13.655 + 1.312 + 2.625 + 0.826 = 18.42 GiB. See §8 for the measurement, which contradicts the brief's expectation. |
| R7 | Clamp full-attention KV capacity to the actual context | up to −2,181,898,240 at 16k vs unbounded | latent only; `MAX_POLICY_CAPACITY = 32768` already caps it at 3.28 GiB |
| R8 | Reduce the 50 sliding caches from 1024 to 512 | −419,532,800 (−391 MiB) | **a context-quality change**, not free: a 1024-token sliding window is the model's design |
| R9 | `posix_fadvise(POSIX_FADV_DONTNEED)` the shard ranges after load | 0 steady-state | removes up to 18.9 GB of load-time page-cache pressure; the load is where free supply is lowest (MEASURED: 47.4 s load, 3,078,552 compressions) |

### Quantifying the KV cache (item 6 of the brief)

**REASONED** from `gpu_cache.rs:165-167` (`capacity*stream_len*2*2 + capacity*4`),
`model.rs:1077` (`INITIAL_FULL_CACHE = 512`), `gpu_cache.rs:480-489`
(`grow_once = current + ceil(current/4)`, clamped to `MAX_POLICY_CAPACITY = 32768`),
plus `config.json` geometry (50 sliding × 16 heads × 256 dim = `stream_len 4096`,
capacity 1024, fixed; 10 full × 4 heads × 512 dim = `stream_len 2048`, grown):

| context | full capacity reached | full KV | **total KV** | KV read/token |
|---:|---:|---:|---:|---:|
| 512 | 512 | 41,963,520 (0.039 GiB) | **881,029,120 (0.821 GiB)** | 461,373,440 (0.430 GiB) |
| 2,048 | 2,443 | 200,228,280 (0.186 GiB) | **1,039,293,880 (0.968 GiB)** | 1,006,632,960 (0.938 GiB) |
| 16,384 | 18,212 | 1,492,655,520 (1.390 GiB) | **2,331,721,120 (2.172 GiB)** | 2,181,038,080 (2.031 GiB) |

**Is the KV sized higher than a short conversation needs?** For 512 tokens, no —
the ladder has not rung. But **83 % of KV@512 is the 50 sliding caches, which
are a fixed 839,065,600 B (0.781 GiB) regardless of context** and are 2× larger
than a 512-token conversation uses. That is the model's window, so it is a design
choice, not oversizing. Full-attention KV is only 0.039 GiB at 512.

**Does context length matter for the 1000×?** No, and this is arithmetic rather
than measurement: at 16,384 the KV *read* is 2.031 GiB/token against 14.951 GiB
of weights — 13.6 %. The weights are 86 % of the bytes at every context length,
so context cannot explain a 1000× gap, and the KV footprint difference between
512 and 16,384 is 1.35 GiB against a 4.72 GiB anonymous gap between the two
checkpoints.

**What I did not measure, and why:** a 4 bpw run at 2,048 or 16,384 tokens of
context. Prefill measured 0.806 tok/s, so 16,384 tokens of prefill alone is
**20,336 s = 5.6 hours**, and 2,048 is 42 minutes. That is not a defensible use
of the box. The §6 statement is therefore REASONED, and it would be falsified by
a 4 bpw run at 512 context that is fast while a 16-token run is slow — which
would mean the fault is KV-shaped rather than weight-shaped.

---

## 7. Bugs found (reported, not fixed — per the brief)

1. **`model.rs:4860` understates a memory cost by 2×.** The doc comment on
   `align_packed_view` says "On `Gxma-4-bit` that is `lm_head.trellis` and costs
   **704 MB**". The copy is `device.new_buffer(len)` where `len = source_len` =
   **1,409,286,144 B = 1,344 MiB = 1,409 MB**. 704,643,072 is
   `1,409,286,144 / 2`, i.e. the **element count** (`source_len / 2`, the
   `packed_len` at `model.rs:4958-4961`), mistaken for a byte count. In a file
   whose stated purpose is to record these numbers honestly, and in a decision
   (`weights.rs:536-538`) that turns on exactly this number, a 2× error is the
   worst possible kind.
   **I checked and refuted the tempting alternative explanation**: the head is
   *not* a byte-per-code tensor with a wasted high byte. Sampled 6 KiB at three
   offsets in the real file: the int16 values span the **full** signed 16-bit
   range (min −32768, max 32767, ~50 % negatives, all 256 high-byte values
   present). Shape `[336,16384,128]` vs `Gxma-repacked`'s `[336,16384,96]` is
   exactly the 8/6 rate ratio, so this is a genuine packed EXL3 trellis at 8 bpw
   and **there is no 704 MB of waste to reclaim**. R3 above is the only way to
   get those bytes, and it costs head precision.

2. **F5 confirmed, with its magnitude corrected.**
   `ForwardScratch::resident_bytes()` (`model.rs:3045-3076`) lists 24 buffers;
   `ForwardScratch::new` (`model.rs:2989-2990`) allocates 26. `raw_b` and
   `raw_c` are genuinely used (`encode_gemv_fused3` receives all three,
   `model.rs:1525-1527`, `:1582`, `:2210`). But the live scratch is **sized by
   `max_tokens`, and MEASURED `scratch_max_tokens = 2` for a 28-token prompt**:
   `raw_b + raw_c` = 2 × `max(2 × 21248, 262144)` × 4 = **2,097,152 B**, not the
   176,160,768 B the 1,024-row figure implies. The accounting bug is real; at
   the context this project actually runs it is worth **2 MiB**.

3. **`memory_proof.rs` may be over-reading `host_statistics64`.** MEASURED on
   this box: `libc::HOST_VM_INFO64_COUNT` is **104** integers, not 9. libc
   derives it from its own `vm_statistics64_data_t`, so libc's struct with
   libc's count is correct and `memory_proof.rs` is right. But it is one libc
   version bump away from a 380-byte overrun past the kernel's buffer, and the
   count is *not* 9 — I wrote that mirror first, asserted the mismatch, and only
   then found the real layout. `natural_t` is `u32`, so the struct genuinely
   mixes 4- and 8-byte fields. Worth a comment at `memory_proof.rs:135` so the
   next person does not repeat my mistake.
   *(This is a note about a hazard, not a claim that `memory_proof.rs` is
   broken — it is not, as written.)*

---

## 8. A/B: `KAPLUMBA_RESIDENT_WEIGHTS=1` at 4 bpw

<!-- FILLED IN BELOW -->

---

## 9. Supply-threshold probe

<!-- FILLED IN BELOW -->

---

## 10. One sentence

**4 bpw cannot be made usable on this box by tuning: it needs 15.79 GiB of
anonymous memory against a ~15 GiB supply and the kernel's only response is to
compress the whole weight set every token — but it becomes usable the moment the
1.312 GiB `lm_head.trellis` copy is removed (R1, a one-time 3-byte checkpoint
repack), which drops anonymous demand to 14.48 GiB with no change to a single
weight, and if even that is not enough the box needs more RAM, not better
kernels.**