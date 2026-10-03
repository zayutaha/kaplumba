# Memory / Swap Cliff — Recommended Design and Ordered Fix List

Status: **read-only investigation, nothing applied.** Every byte figure below was
read from the safetensors headers of the two checkpoints or measured with
`sysctl`/`top`/`/usr/bin/time -l`; see `implementation_notes.md` for the
arithmetic and `file:line` evidence.

Scope note that changes the whole question: **there is no `mlock` in this
repository.** `grep -rn mlock crates/` returns zero hits in any `.rs`, and
`spec/file-backed-weights/tasks.md` marks T2 (expand `skip_copy`) and T4 (add
`mlock`) as *Not started*. At HEAD the 10.2 GiB (3 bpw) / 13.6 GiB (4 bpw) of layer
trellis is **dirty anonymous arena**, confirmed by the repo's own measurement at
`spec/engine-performance/implementation_notes.md:1202-1203`
(`SM=SHM PURGE=N`). The mlock question is therefore a question about a *proposed*
configuration (`spec/file-backed-weights` T2+T4), and this document answers it on
that basis. Section 2.0 states what must be deleted from that proposal.

---

## 1. The design

### 1.1 One sentence

**Replace the fixed byte budget with a measured one: at load, read the free page
supply the box actually has, compute the projected peak of this checkpoint at the
requested context, and refuse the load (exit 75) when the projection exceeds
supply — then, only once that check has passed, file-back and `mlock` the layer
trellis up to the *measured* `vm.user_wire_limit` headroom, and never pin
`lm_head` or `embed_tokens`.**

### 1.2 What the cliff actually is

It is **not** a hard engine budget and **not** a property of the checkpoint.
`resident_limit_bytes()` is `18_000_000_000` (`crates/engine/src/metal.rs:4`) and
the 4 bpw arena needs `14_661_867_236` — the engine's own guard passes and hands
macOS a configuration the box cannot hold.

It is an **OS reclaimability threshold**, and it is crossed at
`engine anonymous demand ÷ (24 GiB − wired − other apps) ≈ 0.93–1.0`. Two measured
anchors, same binary, same box:

| config | anonymous demand | `peak memory footprint` | outcome |
|---|---:|---:|---|
| `Gxma-repacked` 3 bpw | 11.576 GiB | 11.59 – 11.98 GiB | fits, no new swap |
| `Gxma-4-bit` 4 bpw | 14.985 GiB | 14.93 GiB | compressor 10–12 GiB, ~515 MB unused, thrash |

The gap between the two is 3.41 GiB. The run-to-run drift of *other* applications
on this desktop is ~2 GiB (the brief records "~11–13 GB used" as the baseline),
and `mtp-40tps`'s snapshot records 4.08 GiB of anonymous apps against the
heavier load's ~10 GiB. **The margin is the same size as the noise, and the
engine never reads the noise.** That is the whole defect.

Weights make it unforgiving: the layer trellis measures **Shannon entropy 7.9702
bits/byte of 8** (8 MiB sampled from `lm_head.trellis`' neighbourhood in shard 2;
`gzip -9` gets 1.87%). Compression therefore buys ~nothing while consuming the
full compressor CPU and creating compressor pages that must themselves be
swapped. The 10–12 GB of compressor observed at 4 bpw is *not* headroom.

### 1.3 The three components that must be separated in the accounting

Today `allocation_breakdown()` returns one number that mixes all three:

1. **Unreclaimable** — arena (anonymous dirty), KV cache, scratch. Only swap
   relieves these, and swap is 22.1 GB/s against a 70 GB/s step.
2. **Reclaimable-clean** — `embed_tokens` (2.625 GiB, ~10 KiB working set) and
   `lm_head.trellis` (0.984 / 1.312 GiB, read in full every token). The OS drops
   these first and re-faults them at NVMe speed.
3. **Absent** — the 1.072 GiB of vision-tower bytes the engine never loads, and
   the ~11–14.7 GiB of shard page cache that load populates and then never needs.

A budget that cannot tell these apart cannot be enforced. `allocated_bytes()`
today reports (1) plus nothing, and `phys_footprint` charges (1) + (2) +
driver overhead — a 0.32–0.41 GiB unexplained gap at 3 bpw that has been
papered over in prose since `resident-memory-12gib`.

---

## 2. MLOCK POLICY — the decision

### 2.0 Three things the current proposal gets wrong, and must be deleted

1. **"mlock + file-backed = green pressure" is false and must not ship as a
   premise.** `spec/file-backed-weights/plan.md:44-56` and
   `implementation_notes.md:39-51` both assert it. `mlock` makes pages *wired*.
   Wired has no reclaim path. It cannot increase available pages, so it cannot
   improve a pressure metric. The claim is also unfalsified by anything in this
   repo — no A/B exists. Delete the "Key Insight" section; keep the measurement.
2. **Whole-shard `mlock` is arithmetically impossible at 4 bpw and is 94 % of the
   limit at 3 bpw.** `plan.md:89-104` locks `reader.length` for every shard:
   * 3 bpw: 16,028,388,994 B (14.928 GiB) against the **15.893 GiB** of wire
     headroom that is left after the measured 2.345 GiB of idle wired →
     **0.968 GiB spare, 6.1 %, for the weights alone.**
   * 4 bpw: 20,041,520,424 B (18.665 GiB) **exceeds `vm.user_wire_limit`
     (18.238 GiB) before a single other page is wired.** The call fails with
     `ENOMEM`, `plan.md:98-103` logs a warning and continues, and the process
     silently lands in the 3.9 GiB-swap configuration T4 was written to prevent.
   * It also pins the **1.072 GiB of vision-tower bytes** that
     `load_text_selection` (`safetensors.rs:469-470`) deliberately excludes —
     1.31 GiB of pinned memory for tensors the engine never reads.
3. **The `mlock` budget must be a byte budget, not "all of it".**

### 2.1 The four policies, with arithmetic

Wire headroom, measured on this box with no model loaded:

```
vm.user_wire_limit              19,585,050,869 B  = 18.238 GiB
wired, idle desktop (top)        2,517,630,720 B  =  2.345 GiB
                                  ----------------------------
wire headroom                   17,067,440,149 B  = 15.893 GiB
less driver/IOGPU growth at load (M1, to be measured)     ~1.0 GiB
                                  ----------------------------
usable lock budget               ~16,065,000,000 B  = ~14.96 GiB
```

Cross-check from a completely independent measurement —
`spec/mtp-40tps/implementation_notes.md:181-196` allocated 1 GiB GPU-wired
chunks and found only **~10 GB of newly-allocated wired memory succeeded** before
free hit 0.12 GiB, and that `file-backed` **never moved**. So new wire demand is
served out of *free supply*, not out of reclaimable cache. 14.96 GiB of usable
lock budget is consistent with that (which saw ~9.5 GiB of free at the time).

| policy | resulting footprint (4 bpw) | reclaimable headroom | decode risk | fits 16k? |
|---|---|---|---|---|
| **(a) pin all layer trellis** | 13.638 GiB becomes **wired, immovable**; total unchanged ≈ 17.32 GiB at 16k | **−13.638 GiB.** Nothing in the weight set can be dropped again. Free supply for KV+scratch collapses from 14.96 GiB to 1.32 GiB | lowest *once pinned*, but a silent partial failure: if `mlock` ENOMEMs partway, those tensors stay evictable and the measured 3.9 GiB-swap collapse returns, with a warning nobody reads | **no** |
| **(b) pin nothing** (today's `KAPLUMBA_ZERO_COPY_WEIGHTS`) | best case ≈ 1.33 GiB anon (KV 0.821 + scratch 0.510) + up to 1.31 GiB of lm_head | maximum — 14.6 GiB of weights droppable at any instant | **catastrophic, measured**: 3.9 GiB swap, 1.4 tok/s (`weights.rs:509-511`, `weight_residency.rs:7-9`). Decode streams the *entire* weight set every token, so the working set **is** the model and there is no partial-retention policy | **no** |
| **(c) pin per-layer lazily on first touch** | converges to (a) after one step, but every unpinned layer is (b) until then | intermediate, and *worse than (a)* in the tail: some layers immovable, some not, so the OS keeps thrashing the unpinned ones | first-token penalty proportional to eviction (410 `mlock` calls, each populating its window); a mixed residency state is the one the OS handles worst | **no** |
| **(d) pin up to a byte budget derived from the measured wire limit** | as (a) when the budget is sufficient | **the budget is the floor, stated not assumed.** Unpinned tensors are named in the log | (a)'s latency where it fits; (b)'s failure — *visibly* — only where it does not | **no** |

### 2.2 Recommendation

**Policy (d), with two hard exclusions and one precondition.**

**Precondition — admission control first (F1 below).** Pinning does not reduce
total demand; it only reclassifies 13.6 GiB from *evictable* to *immovable*.
Doing (d) before the box is known to have room converts a slow recovery into an
unrecoverable one. So (d) may only be reached once the load has already proved
`projected_peak ≤ measured_free_supply`.

**Exclusion 1 — never pin `lm_head.trellis`.** 1.312 GiB at 4 bpw. It is read in
full every token, so if it is evicted it costs `1,409,286,144 / 22.1e9 =
63.8 ms/token` against a ~70 ms step. It *must* be resident — but residency does
not require wiring. Leaving it in the reclaimable file cache (which is where it is
today, and where `resident-memory-12gib` measured 2500 tokens × 2.6 TB of re-reads
at **0 swaps**) keeps the pages hot *and* returns 1.312 GiB of budget the moment
the OS needs it. Pinning it is the worst available trade.

**Exclusion 2 — never pin `embed_tokens`.** 2.625 GiB to keep ~10 KiB hot (one
row per token). It is already file-backed and already cold.

**Pre-size, do not lazily pin.** (c)'s laziness buys nothing that (d) does not
buy better: (d) knows the budget *before* the first token, so it can pin in decode
order up to the budget and print what it could not pin, instead of discovering
the limit 410 `mlock` calls into a step.

**And the honest bottom line: (d) does not make 4 bpw work.**
At 4 bpw, layer trellis alone is 13.638 GiB against a ~14.96 GiB budget — 91 %.
It pins, but the *total* is unchanged and the total is over supply. Pinning makes
4 bpw at 16k **worse**, not better, because it removes the compressor/swap escape
valve that currently keeps the process limping. Pinning is a **latency** fix for
3 bpw, not a **capacity** fix for 4 bpw.

---

## 3. The 16k question

**4 bpw at 16384 context does not fit on this box, at any pinning policy.**
Calibrated on the measured 4 bpw anchor (14.928 GiB at 512 positions) plus the
exact KV delta — no overhead guessing:

| 4 bpw, 16k | bytes | GiB |
|---|---:|---:|
| measured footprint @ 512 positions (anchor) | 16,028,184,360 | 14.928 |
| KV delta, 512 → 16384 (2,181,898,240 − 881,029,120) | 1,300,869,120 | 1.211 |
| **projected footprint @ 16384** | **17,329,053,480** | **16.139** |

Free supply on this box, measured, with the brief's desktop load: **≈ 14.5 GiB**
(24.000 − 2.4 wired − ≈ 10.1 other apps, back-solved from the observed "unused
515 MB + compressor 10–12 GB"). With the quieter `mtp-40tps` snapshot:
**17.03 GiB**.

* vs 17.03 GiB supply → **deficit 0.89 GiB**, i.e. it fits on a quiet desktop.
* vs 14.5 GiB supply → **deficit 1.64 GiB**, i.e. it does not.

3 bpw at 16384: 11.98 (measured @ 512) + 1.211 = **13.19 GiB** → fits either way,
margin 1.3–3.8 GiB.

### Levers that close the deficit without touching dispatch counts or the chunk

Off-limits per instruction: per-token dispatch counts, `PREFILL_CHUNK`. Assessed
anyway so the exclusion is visible rather than assumed.

| lever | closes | verdict |
|---|---:|---|
| **Free ~1 GiB elsewhere** (close Firefox / the Virtualization.framework VM — 4.08 GiB measured, ~10 GiB under load) | up to 6 GiB | **the only lever that closes 1.64 GiB outright.** Not the engine's call; must be in the operator runbook. |
| **Clamp `ensure_capacity`'s `max_capacity` to the policy limit.** `model.rs:3396` passes `max_position_embeddings` = **262 144** while `generate.rs:35` caps context at 32 768. A caller that bypasses `fits_in_context` grows KV to `10 × 262144 × 8196` = **20.02 GiB**. | 0 today, **19 GiB of latent headroom removed** | **fix regardless.** This is a footgun, not a current cost. |
| **Grow full caches ×1.25 instead of ×2** (`gpu_cache.rs:183-186`) | 0 at exactly 16384 (already a power of two); **1.25 GiB** at 16385–32768 | fix; it does *not* close the exact-16k deficit and the note must say so |
| **`posix_fadvise(DONTNEED)` the copied shard ranges after load** (§5) | 0 steady-state; up to **14.66 GiB of load-time page-cache pressure** | fix — the load is where free supply is lowest |
| **Offload any layer weight to the file cache** | negative | **0 GB/token budget.** 1 GB/token of NVMe at 22.1 GB/s = 45 ms against a 70 ms step. Confirms why `KAPLUMBA_ZERO_COPY_WEIGHTS` measured 1.4 tok/s. |
| **`lm_head.trellis`** | 0 | 63.8 ms/token if evicted. Must stay resident; 1.312 GiB is not recoverable. |
| **`embed_tokens`** | 0 | already cold and reclaimable. |
| **Drop `raw_c` (one of three f32 accumulators)** | 84 MiB | available — a kernel change, not a dispatch-count change — but not planned here |
| **Halve `PREFILL_CHUNK`** | 242 MiB | **off-limits**, and costs 6.4 % of GEMM at 614 tokens (`model.rs:66-74`). Listed so the trade is on record. |
| **Quantize KV** | up to 1 GiB | **barred by AGENTS.md §8** without an explicit lift. |

---

## 4. Ordered fix list

Each step is independently testable, and each is committed separately per
AGENTS.md §4.2. **F1 is the fix for the reported problem; F2–F4 are correctness;
F5–F6 are hygiene; F7–F8 are optional and may be rejected on measurement.**

### F1 — Pre-flight memory admission control (the fix for the cliff)
*Files:* new `crates/engine/src/memprobe.rs`; `crates/engine/src/metrics.rs`
(+`available_bytes()`); `crates/engine/src/model.rs::load_from_dir`;
`crates/cli/src/main.rs`.
*Do:* read `host_statistics64` `free_count`/`active_count`/`inactive_count` and
`sysctl vm.user_wire_limit` **before** allocating the arena. Project
`arena_bytes + kv_bytes_at(min(context_limit, max_seq_len)) + scratch_bytes +
mapped_hot_bytes`. If `projected > available_bytes * SAFETY`, return
`ModelError::InsufficientMemory { projected, available }`, mapped to
`gpu_lock::EXIT_BUSY` (75) by the CLI — a scheduling outcome, not a failure.
`SAFETY = 0.92`, which is where the two measured anchors put the cliff.
*Also:* print `projected` / `available` / `wired` / `anon_others` on **every**
successful load, so a future "why did it swap" has an answer in the log.
*Test:* (i) unit test of the projection against the two known checkpoints;
(ii) **sabotage**: force the projection to always pass, run against `Gxma-4-bit`,
and confirm the guard is what would have fired — then revert; (iii) run at
`Gxma-repacked` and confirm it loads. *Not* a claim until (ii) has been observed
failing.

### F2 — Bound `ensure_capacity` by the policy context limit, not `max_position_embeddings`
*Files:* `crates/engine/src/model.rs:3396` and `:1663`, `:1783`;
`crates/engine/src/gpu_cache.rs:167-187`.
*Do:* pass `min(max_sequence_len, DEFAULT_CONTEXT_LIMIT)` as `max_capacity`.
Today the bound is 262 144 and the *policy* bound is 32 768 — see
`model.rs:3381-3395`, which says the two "cannot disagree" and then lets them.
*Test:* a unit test that drives `GpuKvCache::ensure_capacity` to 262 144 and
asserts it errors at 32 768 + 1; **sabotage** by restoring 262 144 and confirm
the test fails.

### F3 — Grow full caches ×1.25
*Files:* `crates/engine/src/gpu_cache.rs:183-186`.
*Do:* `target += (target + 3) / 4` instead of `target *= 2`. Caps over-allocation
at 25 % instead of 100 %.
*Test:* table-driven capacity ladder asserting `capacity >= needed` and
`capacity - needed < needed/4` for `needed` in 513…40000. **Sabotage** the
predicate back to 2× and confirm it fails at `needed = 32769`.

### F4 — `posix_fadvise(POSIX_FADV_DONTNEED)` after load
*Files:* `crates/engine/src/weights.rs`, new phase 2.5 in `load_internal`.
See §5 in `implementation_notes.md` for the exact patch, the ranges that are safe
and the ones that must not be touched, and why a sequential post-pass beats an
in-loop call.
*Sequencing:* **only while the layer trellis is arena-resident.** If F6 lands,
this becomes dead code — say so in the commit message and gate it behind
`!skip_copy_layer_trellis`.
*Test:* `KAPLUMBA_LOAD_PROFILE` prints bytes advised; assert the advised ranges
are exactly the `!skip_copy` set and that `embed_tokens`/`lm_head.trellis` are
absent. A/B `load_ms` and peak `phys_footprint`.

### F5 — Fix `ForwardScratch::resident_bytes()` (accounting bug, 168 MiB)
*Files:* `crates/engine/src/model.rs:3045-3076`;
`crates/engine/tests/real_forward_buffer.rs:402-439`.
`resident_bytes()` omits `raw_b` and `raw_c` (both present in `new`,
`model.rs:2989-2990`, both used by `encode_gemv_fused3`,
`model.rs:1525-1526`/`1582`/`2210`). At 1024 rows that is **176,160,768 B** of
under-report, and the test's `400 KiB/row` budget passes *only* because of it —
the true figure is `535,095 B/row`, **1.31× over budget**. Proposed patch in
`implementation_notes.md` §4a: replace the hand-maintained 24-element list with a
single `all_buffers()` source of truth plus a derivation test, so the class of
bug cannot recur.

### F6 — Decide and document the layer-trellis residency + mlock policy
*Files:* `crates/engine/src/weights.rs:508-594`;
`crates/engine/src/weights.rs::ShardReader` (+`lock_window`);
`spec/file-backed-weights/{spec,plan,tasks}.md` (amend first).
*Do:* implement policy (d) — expand `skip_copy` to the layer trellis, then
`mlock` each trellis tensor's page-aligned window in decode order up to
`vm.user_wire_limit − wired_now − DRIVER_WIRE_RESERVE`, logging any tensor left
unpinned. **Never** `mlock` a whole shard mapping; **never** pin `lm_head` or
`embed_tokens`.
*Gated on:* F1 passing, and a **new** interleaved A/B/A/B over ≥16 paired samples
of `decode_tok_s`, because the existing evidence for the offload is a 35 %
same-config spread (`resident-memory-12gib/implementation_notes.md:199-213`) and
a single sample at a 24 % apparent effect.
*Test:* `weight_residency.rs` inverted to assert the *new* policy (layer trellis
mapped **and** pinned, head/embed mapped **and not** pinned); bit-exactness at
`--temperature 0`; a new test asserting `mlock`ed bytes ≤ computed budget.

### F7 — Fix the contradictory weight assertions
*Files:* `crates/engine/tests/weight_residency.rs:148`;
`crates/engine/tests/memory_proof.rs:35-43,108-113`.
`weights < 11_500_000_000` and `weights > 14_000_000_000` are mutually exclusive,
and **each is currently satisfied by exactly one of the two checkpoints** — which
is why both have been "passing" somewhere. Replace both with assertions on the
*relationship* the code actually implements, not on a snapshot of one checkpoint.
Proposed patch in `implementation_notes.md` §4b.

### F8 — Fix `KvCache::resident_bytes()` and the `bench_residency` GiB/GB units
*Files:* `crates/engine/src/kv_cache.rs:86-88,248-250`;
`crates/engine/tests/kv_cache.rs:116,132`;
`crates/engine/examples/bench_residency.rs:295-296,424-425`.
The host `KvCache::resident_bytes()` omits the `positions` array that
`GpuKvCache::resident_bytes()` (`gpu_cache.rs:121-123`) includes, and
`tests/kv_cache.rs:116,132` *pins the omission*. `bench_residency.rs` divides
bytes by `1e9` and prints the label `GiB` — a 7.4 % understatement in the one
tool that reports footprints.

---

## 5. Requirement → task map

| requirement | task |
|---|---|
| stop swapping at 4 bpw / turn a hang into a diagnosis | **F1** |
| KV cannot outgrow the policy context limit | **F2** |
| KV over-allocation ≤ 25 % | **F3** |
| release the 14.7 GiB of load-time page cache | **F4** |
| scratch accounting is truthful and structurally so | **F5** |
| layer weights non-evictable without exhausting the wire limit | **F6** |
| the test suite cannot assert two opposite budgets | **F7** |
| no `resident_bytes()` silently omits a buffer | **F5**, **F8** |
| no GiB label on a GB divisor | **F8** |
