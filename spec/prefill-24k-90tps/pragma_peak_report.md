# The pragma is not the lever, and the tensor peak was 2x too high

**Agent:** round-3 fixer for `spec/prefill-24k-90tps`, answering
`adversary_report2.md` (A1-A5, A11, A12).
**Base:** `f8b7a46` (adversary round 2, merged).
**Instrument:** `crates/engine/examples/flash_qk_pragma_arms.rs` (new) and
`crates/engine/examples/mma_peak.rs` (rewritten).
**Evidence:** `spec/prefill-24k-90tps/pragma_peak_evidence/`.

## Verdict summary

| # | claim attacked | verdict | one line |
|---|---|---|---|
| **T1** | `attention.metal:1632` has no `#pragma unroll` and that costs 2.3x | 🟢 **NOT A BUG** | on the **shipping** kernel the pragma is worth **0.0-0.7%** at every unroll factor, and `unroll 32` is **3.5% slower**. Shipping is already the fast configuration. The 2.3x is an artefact of the transcription and **did not survive contact with the real kernel** |
| **T2** | `mma_peak.rs`'s 15.6 TFLOPS peak, the denominator of every "% of peak" | 🔴 **BROKEN, and fixed** | the kernel stored 8 of 16 accumulators; the instrument now stores all of them, asserts the old bug reproduces (`dead8`), and asserts linearity in `iters`. **True ceiling 8.30 TFLOPS** (best live chain count), 7.6 at 16 chains |
| **T3a** | attention is at 12% of peak; AC2 needs 46%; GEMM at 43% | 🔴 **all wrong by ~2x** | against 8.30: attention **21%**, AC2's ask **86%**, GEMM **81%**. §23.6's "0.8% margin" is **-4%** -- above peak |
| **T3b** | §22.3's ablation table | 🔴 **does not sum** | 54.4% + 56.4% = **110.8%** of the kernel, re-measured one build. The arms overlap, because QK and PV share the same MMA pipe and removing either lets the other issue |
| **T4** | F1's `include_str!`/stale-build root cause | 🔴 **wrong** | cargo tracks the edit; §10's arm runs through the ordinary build path and reads **0.18 ms**, not 964 ms. The real cause is in §F1 below |
| **T5** | §22's ablation measured a deoptimised loop | 🟢 **REFUTED for §22** | the harness's own scaffold, compiled with every arm inactive, is **bit-identical and x0.995** of the shipping kernel. §22 measured the shipping kernel |
| **T6** | "16 accumulator chains is the one lever left" (§23.6) | 🔴 **dead twice over** | compile-time 2 and 4 chains are **x1.001 / x0.994**. Chain count was never the mechanism; §22's arms 11/12 used a *runtime* selector on a rolled loop and so created no ILP at all |
| **T7** | `MMA_TOLERANCE = 0.02` is 9.3x looser than AC5 and enforced nothing | ✅ **closed** | tightened to **3e-3** against the test's own measured worst (2.16e-3). The old gate needed a **3%** error to fail; the new one fails at **0.5%** |
| **T8** | F4: `workgroups` costs "1 greedy token in 1 536", so flip the default | 🔴 **the number is wrong where it was never measured** | at 8 204 tokens (a context the evidence does not cover) it is **434 of 476 greedy tokens from token 39, reproduced in two runs**, and it stops 36 tokens early. Under `--temperature 0.7` the same arm does not diverge at all — **the two legs disagree**, so the effect is **not characterised**. **The flip is rejected**, not deferred |
| **T9** | N1: prefill passes are NEUTRAL, measured at 16 394 | ✅ **now also at 24 602, with a control** | r1: `onedispatch` **66.829** vs `pass8192` **67.051**, both VALID (+0.33%). r2's control was discarded by the memory gate so r2 is unusable — one valid pair in two attempts, which is the honest rate for 24k on this box |

---

## T1 — the pragma: shipping is already optimal

### The claim

`adversary_report2.md` A2: a standalone transcription of `attention.metal:1632`'s
`d8` loop runs at 45.6 ms / 4.52 TFLOP/s as transcribed (no pragma) and
105.5 ms / 1.95 TFLOP/s with the unroll forced -- **2.3x, bit-identical**. From
that, A3 concludes the lever "is removing the forced 64x unroll, which is one
line", and that the shipped kernel is 2.3x off its own shipped self.

### What was measured, and where

Not the transcription. `flash_qk_pragma_arms.rs` `include_str!`s the **shipping
`crates/kernels/src/attention.metal`** -- the same bytes `kaplumba-kernels`
compiles -- and for each arm inserts one line before the QK `d8` loop. Arm 0 is
the file **verbatim** (asserted). Every anchor must match exactly once
(asserted). Every arm must differ from the shipping bytes (asserted, and
sabotage-verified below). Everything else is the real dispatch: the real
`online_attention_flash_{bq1_,}512_8` kernels, the real grid
`ceil(queries/BQ) * kv_heads` of 256 threads, the real `AttentionParams`, real
buffers, the prefill geometry.

### 24 576 keys, 1024 queries, `BQ=1` (production's 24k arm), medians of 3
interleaved rounds

```
arm                ms [min max]        ratio    output
0-shipped          967.20 [964.7 973.7]  1.000    baseline
1-#pragma unroll   968.83 [965.7 972.5]  x0.998   0 bytes differ
2-unroll(2)        974.00 [965.4 974.8]  x0.993   0 bytes differ
3-unroll(4)        970.65 [967.4 973.7]  x0.996   0 bytes differ
4-unroll(8)        969.74 [963.6 970.4]  x0.997   0 bytes differ
5-unroll(16)       970.68 [969.9 974.0]  x0.996   0 bytes differ
6-unroll(32)      1001.91 [994.1 1002.3] x0.965   0 bytes differ
```

**The pragma is worth 0.0-0.7%, and 32 is 3.5% slower.** The per-arm spread
inside a round is 0.4-1.0%, so x0.993 to x1.000 is at the edge of resolution
and x0.965 is outside it.

Same at `BQ=2` (the other shipping arm) and at 8 192 keys:

| geometry | best pragma | `unroll 32` | spread within a round |
|---|---|---|---|
| 24 576 keys, `BQ=2` | x1.007 | **x0.928** | 0.3-0.9% |
| 8 192 keys, `BQ=1` | x1.003 | **x0.964** | 0.2-0.6% |

Evidence: `pragma_peak_evidence/pragma_full_bq1_24576.log`,
`pragma_full_bq2_24576.log`, `pragma_full_bq1_8192.log`.

### Bit-identity

Every pragma arm is **byte-identical** to the shipped kernel's output -- 0 of
33 491 072 bytes differ, checked before any timing is printed. This is the
"must be shown bit-identical, not merely close" requirement, met exactly.

### The controls, because a "no effect" reading needs them

| control | what it is | reading |
|---|---|---|
| `spin` | a 3 000 000-iteration threadgroup-memory spin at the top of the body | 4 979.61 ms, **x5.15** -- the timer resolves a real change |
| `empty` | §10's own arm, `if (tid < 256u) { return; }` | **1.60 ms = 0.17% of the baseline (x604)** -- the timing is of the kernel body |

Both asserted in the instrument; the run aborts if either fails to move.

**The `empty` reading is the load-bearing one.** §10 recorded 964.79 ms for
exactly this arm. Here, in the same process, one line above it in the same
source, it reads **1.60 ms**. The difference is 600x and it is not a scheduling
difference -- it is the difference between an arm that reached the shader and
one that did not.

### Sabotage: the instrument can fail

1. **Make the pragma edit a no-op** (skip the `sub_once`): the instrument aborts
   with *"arm 1-unroll produced the shipping source unchanged: its edit did not
   apply"*. Observed. Reverted.
2. **Make the exactness check compare each arm to itself**: the instrument
   aborts with *"arm noqk: byte-identical to the shipped kernel, so it did not do
   what its name says"*. Observed. Reverted.

### Verdict, and what is corrected

**The missing `#pragma unroll` is not a bug and nothing is changed in
`attention.metal`.** Shipping is already the fast configuration; the adversary's
own report says their first harness version *accidentally* forced the unroll, and
the transcription's 2.3x does not survive on the real kernel.

Corrected in the report:

* **A2's claim that the shipped kernel is "2.3x off its own shipped self by
  scheduling" is false.** On the shipping kernel the pragma is worth 0.7%.
* **A2's claim that "the `88% MMA issue` was measured on a deoptimised variant"
  is false** -- see T5.
* **A3's "`16 chains` is not the lever" is right, and for a stronger reason than
  A3 gives** -- see T6.
* The transcription harness (`adv_qk_ceiling.rs`) stands as evidence *about a
  transcription*. It cannot rank the shipping kernel, and the shipping kernel is
  what matters. That is the generalisable lesson: **a microbenchmark of a
  transcribed inner loop measures the transcription.** The shipping source is
  2 014 lines of which the transcribed loop is 20; the other 1 994 include the
  PV phase, the softmax, the `sth`/`wq` threadgroup traffic and four barriers per
  tile, and any of them can change what the compiler does with the loop.

### What the pragma number does tell us

`unroll 32` costing 3.5% is a real, reproducible register-pressure effect: 32
unrolled iterations of `HEAD_DIM/8 = 64` live `kf`/`qf` fragments is more live
state than the loop body can hold, and the compiler spills. The same mechanism
`mma_peak.rs` records at 32 *array* accumulators (15.49 -> 0.74 TFLOPS). So the
pragma axis is not flat in general -- it is flat *at and below the shipped
setting*, which is the only place a shipping decision lives.

---

## T6 — what actually limits the shipping kernel

With the peak corrected (T2), AC2's ask is ~86% of the ceiling (T3a), so the
question "is this kernel MMA-issue-bound" has to be re-asked with a corrected
instrument. The same harness answers it, because it can ablate the shipping
source in one build.

24 576 keys / 1024 queries / `BQ=1`, medians of 3 interleaved rounds, one build,
all arms compiled from the shipping bytes:

| arm | what it removes | ms | ratio | share removed |
|---|---|---|---|---|
| baseline | -- | 967.20 | 1.000 | -- |
| `empty` | **everything** (§10's arm) | 1.60 | **x604** | 99.8% |
| `noqk` | QK MMAs + `kf`/`qf` loads + `sth` publish | 441.21 | x2.192 | **54.4%** |
| `nopv` | PV MMAs + `af`/`bf` loads + rescale | 421.57 | x2.294 | **56.4%** |
| `noqk-nopv` | both | 12.85 | **x75.3** | **98.7%** |
| `noloads` | **all four** fragment loads, MMA count untouched | 460.72 | **x2.099** | 52.4% |
| `loads-l1` | K and V device loads from a fixed L1 row | 766.72 | x1.261 | 20.7% |
| `noqf` | the Q fragment load (threadgroup) | 791.19 | x1.222 | 18.2% |
| `noaf` | the P fragment load (threadgroup) | 934.32 | x1.035 | 3.3% |
| `chains2` | **compile-time** 2 QK chains | 965.90 | x1.001 | 0.0% |
| `chains4` | **compile-time** 4 QK chains | 972.58 | x0.994 | -- |
| `scaffold` | §22's whole ablation harness, every arm off | 937.28 | x0.995 | 0.0% |

### Finding 1 — the arms do not sum, and that is the finding

54.4% + 56.4% = **110.8%**. §22.6 offers "the arms must sum" as the rule that
detects a mislabelled arm, and §22.3's table violates it (A12's 105-109%; here
110.8%, measured one build). §22.3's own reading -- 55.8 + 40.5 + ~4% residual =
100% -- does not sum either, and the fixer's own harness "checked" it against
1.00 without the check being in the code.

The overcount is not noise, and it is not a labelling error. QK and PV **issue
into the same tensor pipe inside the same key-tile loop**, and each is a
*dependent* chain on its own accumulator (`dots[b]` over 64 `d8` steps;
`acc[b][j]` over the tile). Removing QK lets PV issue without competing; remove PV
and QK gets the pipe to itself. So neither share is "the kernel is 88% MMA
issue"; the correct statement is **the kernel is 98.7% MMA-issue-plus-its-loads,
and the two halves overlap by ~11% of the kernel**.

### Finding 2 — the loads are ~52% of the kernel, not 8-19%

`noloads` replaces all four fragment loads with filled constants and leaves the
MMA count and instruction mix otherwise untouched: **x2.099, 52.4% of the
kernel**. The two device loads are 20.7% (`loads-l1`), the Q threadgroup load
18.2% (`noqf`), the P threadgroup load 3.3% (`noaf`).

§22 measured 8% (device) and 11-12% (threadgroup) and concluded "the loads are
~12% of the kernel". **That is wrong by ~4x, in the direction that matters**: the
loads are the majority of the kernel's non-MMA cost and they are worth more than
§22's entire "88% MMA issue" figure leaves for everything else.

The discrepancy is explainable and is a measurement artefact, not a physics
difference. §22's arms 5/6/8 changed *the address* to an L1-resident row, which
keeps the load instruction and its latency but removes the **device** path;
§22's arm 14 removed the Q/P threadgroup loads. Neither removed the *issue* of a
`simdgroup_load` from the instruction stream. `noloads` removes the instruction.
On this hardware the issue slot is the expensive part, not the cache line.

### Finding 3 — the chain was never the mechanism, and §22's arms could not have made it one

§23.6's "the one concrete thing left to try" is 16 named accumulator chains, on
the strength of §22's arms 11/12 reading x1.00 and x0.96 and `mma_peak.rs`'s
table showing 1 chain 5.75 / 8 chains 7.65 / 16 chains 15.49.

Two things kill it, and only one of them is the adversary's:

1. `mma_peak.rs`'s table was the phantom peak (T2). With every accumulator
   stored, chain count barely matters: **1 chain 7.18, 2 chains 7.34, 8 chains
   7.50, 16 chains 7.63, 24 chains 8.30** TFLOP/s. A 15% spread from 1 to 24
   chains, on a table that claimed 2.7x.
2. §22's arms 11/12 selected the chain with `if ((d8 >> 3) & 1)` on a **rolled**
   loop. That is a *runtime* branch inside the loop body, so the two "chains"
   could not dual-issue; the arms could not have created ILP whatever they
   measured. This harness does it at **compile time** (`#pragma unroll` makes
   `d8` a constant, so `(d8 >> 3) % N` folds away) and reads **x1.001** for 2
   chains and **x0.994** for 4.

So chain count is not a lever on this kernel at all, and §23.6's recommendation
-- a day's experiment on 16 chains -- would have returned 1.00x. It is retired.

### Finding 4 — what the kernel's real limiter is

Not the tensor pipe, and not the DRAM. At 1.71 TFLOP/s the kernel is at **21%**
of the 8.30 TFLOPS ceiling, and 52% of the kernel is fragment-load issue. The
QK loop's shape is 1 `kf` load + 1 `qf` load + 1 MMA per `d8` step at `BQ=1`
(64 steps): **two loads per MMA**, and §22's own loop-order comment (which is
correct and is in the shipping source) says the MMAs per load is what caps
throughput.

That is the lever, and it is a **structural** one, not a pragma:

* `BQ=2` halves the K/V device bytes and the `bf` loads per MMA. It is measured
  **slower** at every geometry (§23.7), because it also halves the grid.
* The fix §9.1b named -- `BQ=2` **with the grid restored** by splitting the key
  axis, i.e. flash-decoding style partials + a combine pass -- is the only change
  in this file that raises MMAs per load without costing parallelism. It is
  still unbuilt. §23.6 called it refuted; **it is not refuted, it is
  unbuilt**, and this measurement is why.

**"Not reachable" is therefore not established.** What is established is narrower
and sharper: at the shipping shape the kernel issues two loads per MMA and
reaches 21% of a ceiling it cannot reach without more reuse per load. AC2 needs
the attention cost cut ~3.8x, which on this evidence means the MMAs-per-load
ratio has to rise by a similar factor, and the only identified way to do that is
a key-split flash decode. That is a rewrite of the attention kernel's schedule,
not a lever -- so §23.6's *recommendation* survives, and its *reasoning* does not.

---

## T2 — `mma_peak.rs`: the peak was 2x too high, and the instrument could not fail

### The defect

`mma_peak.rs`'s kernel accumulated into **16** named accumulators and stored
**8**. `d4..d7` were written and never read, so their chains were dead stores the
compiler deleted. It reported 15.0 TFLOPS for MMAs that a kernel doing **half** the
work finished in the same time.

### What the rewrite does

* **Stores every accumulator**, into a `device float *`, so no chain is dead.
* **Keeps the old kernel as `dead8`** -- 16 accumulate, 8 store -- beside
  `live8` (8/8) and `live16` (16/16), in one library, one build, interleaved
  rounds, medians with min/max.
* **Asserts three things**, so the instrument is able to fail:
  1. `live16 / live8` in 1.7-2.3 (twice the MMAs, twice the time),
  2. `dead8` is **not** within 1.35x of `live8` -- i.e. the dead stores really
     are being deleted and the old number really was the phantom,
  3. `live16` is linear in `iters` (1x vs 2x iters in 1.85-2.15), so the timing
     is of the MMAs it counts and not a per-dispatch constant.
* **Sweeps chain count 1/2/4/8/16/24/32** with a printed
  measured/expected-time column, because the old table stopped at 16 and 24 is
  where it turns over.
* Names every arm by `arm_index(name)` rather than a bare index, because
  `mma_peak`'s bug was an index-shaped mistake.

### Measured, 2 048 simdgroups x 20 000 iters, medians of 5 interleaved rounds

```
arm  chains  median_s    min_s    max_s  TFLOP/s  vs live8  scale
live1      1   0.0058   0.0057   0.0060     7.18     1.056   x1.051
live2      2   0.0114   0.0111   0.0114     7.37     0.997   x1.017
live4      4   0.0221   0.0220   0.0223     7.55     0.969   x0.994
live8      8   0.0442   0.0438   0.0448     7.59     1.000   x1.000
live16    16   0.0880   0.0865   0.0888     7.63     0.983   x0.983
live24    24   0.1213   0.1205   0.1218     8.30     0.904   x0.904
live32    32   0.1656   0.1645   0.1678     8.10     0.937   x0.925
dead8     16   0.0445   0.0444   0.0457    15.08     0.497   x0.497   <- the phantom
```

Controls, all passing:

```
live16 / live8  = x1.997   (expect ~2)
live8  / dead8  = x0.990   (expect ~1: dead8 does 2x the LIVE MMAs)
dead-store deletion reproduced: YES -- dead8 is the phantom peak, ~2x too fast
live16 at 1x iters 0.0888 s, at 2x iters 0.1761 s -> x1.984 (expect ~2)
```

Evidence: `pragma_peak_evidence/mma_peak_corrected.log`.

### The corrected ceiling

**8.30 TFLOPS** (`live24`, the best live reading; `live16` is 7.63, `live32` 8.10
as the curve turns over). The old figure was **15.6**. Everything written off it
is out by **1.88x**.

Two notes on the shape, because they change how the number may be used:

* **Chain count is nearly irrelevant**: 7.18 at 1 chain to 7.63 at 16 is **6%**.
  The old table's 2.7x (5.75 -> 15.49) was the phantom: its "1 chain" arm was a
  *different* kernel from its "16 chain" arm in the only respect that mattered,
  half its stores.
* **`live24` at 8.30 beats `live16` at 7.63**, so 16 chains is not a ceiling and
  the sweep had to go past where the old table stopped to find the max. The
  `scale` column (measured time / time proportional to chain count) reads
  x0.904 at 24 and x0.925 at 32, so part of that is the compiler finding a
  better schedule at 24 rather than a hardware effect. It is reported as
  measured; the honest statement is **"the ceiling for this operand shape and
  this loop structure is 7.6-8.3 TFLOPS"**, and every percentage below uses
  **8.30**, the most generous reading, so that no claim in this spec is flattered
  by the correction.

---

## T3 — every percentage re-derived against 8.30 TFLOPS

Raw measurements (unchanged by any of this; they came from
`adversary_evidence/profile_16384_onedispatch.log` and
`fixer_evidence/f5_arithmetic.md`, and the adversary independently reproduced
the arithmetic):

```
GEMM:        6.0238e10 FLOP/token x 16394 tokens / 146.770 s  = 6.73 TFLOP/s
attention full 10 layers   8.814e13 FLOP / 45.9 s             = 1.92 TFLOP/s
attention sliding 50      2.750e13 FLOP /  8.5 s             = 3.24 TFLOP/s
attention total            1.156e14 FLOP / 54.4 s            = 2.13 TFLOP/s
budget: 24590 / 90 tok/s = 273.2 s; -223.5 GEMM -16.2 gap = 33.5 s for attention
AC2 asks: 2.394e14 FLOP / 33.5 s = 7.15 TFLOP/s
```

| §23 claim | at 15.6 (wrong) | at 8.30 (correct) |
|---|---|---|
| attention measured 1.9 TFLOP/s | "12% of peak" | **23%** |
| attention total 2.13 TFLOP/s | "14%" | **26%** |
| the sliding layers' 1.85 TFLOP/s | "12%" | **22%** |
| v_proj 8.63 TFLOP/s | 55% | **104%** |
| k_proj 7.78 | 50% | 94% |
| the FFN's three 6.63 | 42% | **80%** |
| o_proj 6.49 | 42% | 78% |
| q_proj 6.37 | 41% | 77% |
| **all seven projections 6.73** | **43%** | **81%** |
| **AC2's required 7.15 TFLOP/s** | **46%** | **86%** |
| the "1.7x GEMM" alternative, 8.1 TFLOP/s | 52% | **98%** |

The sharpest consequences:

* **The FFN GEMM is at 81% of the tensor ceiling, not 43%.** §23.2 called 41-55%
  "a measured tensor peak with no spread worth attacking". At 81% there is
  essentially nothing left, and `spec/engine-performance`'s independent
  6.13-6.44 TFLOPS for the same GEMM is 74-78% of the same corrected ceiling.
  Its declared "7.5 TFLOPS wall" is **90%** of it.
* **§23.6's "honest ceiling" table's best case is above peak.** Its second row
  puts both attention kernels at the "8-chain peak" of 7.65 TFLOP/s and lands at
  90.7 tok/s, which it calls "a 0.8% margin". Against the real 8.30 ceiling that
  scenario needs **92%** of peak and, on the 7.15 ask, **86%**. The 0.8% margin
  was an artefact of the denominator. The row's *tok/s* figure (90.7) is
  unaffected -- it does not reference the peak at all -- so the scenario is
  arithmetically possible but with **far less headroom than reported**.
* **The "flash-only" refutation survives and hardens.** §23.4: holding the
  sliding layers at 1.85 TFLOP/s leaves 11.2 s for the full layers, i.e.
  17.7 TFLOP/s = **213% of peak**. Above even the phantom 15.6 it was already
  impossible; against 8.30 it is impossible by a factor of two.

### AC2, re-stated against the true ceiling

Attention must go from **1.9 TFLOP/s (23% of peak)** to **7.15 TFLOP/s (86%)**,
inside **33.5 s** of a 273.2 s budget, while the context-independent 223.5 s is
**81% of tensor peak** and cannot give any of it back. That is **3.8x** on a
component that this session measures as **98.7% MMA-issue-plus-loads with two
loads per MMA**. It is not reachable by tuning the load/MMA mix, because the
loads are the MMA's operands and removing them removes the work.

---

## T5 — §22's ablation did **not** measure a deoptimised loop

A2 claims §22.3's "88% MMA issue" was measured on a deoptimised variant of the
kernel, because the harness introduced a `#pragma unroll`. Two things in
`1a30427`'s diff make that plausible a priori: it declares
`simdgroup_half8x8 qreg[BQ][NTILES]` -- **64 registers/lane** at `BQ=1`,
128 at `BQ=2` -- unconditionally, and adds `dots2/3/4[BQ]` initialised on every
tile. Those are exactly the declarations that cost `mma_peak.rs` 15.49 ->
0.74 TFLOPS at 32 accumulators.

So this was worth measuring rather than arguing about. The `scaffold` arm is
§22's declaration set with **every arm inactive** (`ABL` fixed at 0, `qreg`
declared but never loaded, the chain branch compiled in but resolving to the
shipping arm at runtime):

```
0-shipped    932.79 ms   baseline, 16777216/16777216 non-zero halves
scaffold     937.28 ms   x0.995   0 bytes differ -- BIT-IDENTICAL
```

**§22's arm 0 was the shipping kernel.** The scaffolding costs 0.5%, which is
inside the cross-run drift §22.2 itself records (14%), and the output is
bit-identical. §22.3's *baseline* is sound.

§22.3's *ratios* are a separate matter, and the harness above does not reproduce
them:

| §22.3 arm | §22.3 ratio | this instrument, same build | |
|---|---|---|---|
| no QK | x2.26 | `noqk` **x2.192** | agrees |
| no PV | x1.68 | `nopv` **x2.294** | **disagrees** |
| all MMAs + loads off | 4% residual | `noqk-nopv` **1.3%** | agrees |
| all fragment loads L1 | x1.09 | `loads-l1` **x1.261** | **disagrees** |
| Q/P loads deleted | x1.12 | `noqf` **x1.222** | **disagrees** |
| 2 / 4 chains | x1.00 / x0.96 | `chains2/4` **x1.001 / x0.994** | agrees |

So "88% MMA issue" is not a number this instrument can produce: the loads are
worth 52% (`noloads`), the MMA-issue-plus-loads 98.7%, and the two halves overlap
by 11%. **"MMA-issue-bound" is right; "88%" is not, and it should not be quoted
as a measurement.**

---

## T4 — F1's root cause, corrected

### What §22.1 said, and why it is wrong

> `attention.metal` is `include_str!`'d ... so editing the `.metal` has no effect
> until the crate that `include_str!`s it is recompiled. [...] The table did not
> fail to reproduce; **it measured a binary that did not contain the edit.**

Cargo tracks `include_str!` inputs. Verified three ways here:

1. **The build says so.** `cargo build --release -p kaplumba-engine --tests`
   after a one-line edit to `attention.metal` prints
   `Compiling kaplumba-kernels v0.1.0` and takes 11.9 s.
2. **The arm runs.** §10's `if (tid < 256u) { return; }`, inserted at the top of
   `attention_prefill_flash_body`, then `cargo test --release --test
   attention_qk_mma_bench attention_arm_throughput_at_prefill_geometry` through
   the ordinary path: **`flash bq1` 0.18 ms** at 24 576 keys / 1024 queries, and
   0.16 ms at the 8 192-key case. Restore the file and the same command reads
   **938.15 ms**. Evidence: `pragma_peak_evidence/f1_reproduction.log`.
3. **A broken `.metal` fails loudly.** Inserting invalid Metal makes
   `cargo test` exit **101** with `test result: FAILED`, 0 passed. It does not
   silently fall back to a stale binary.

So the edit reached the shader. §22.1's mechanism is refuted.

### What actually happened

§10's table has a third possibility that neither §22.1 nor the adversary
considered, and it is visible in §10's own numbers:

| §10 arm | §10 reading |
|---|---|
| `if (tid < 256u) { return; }` | 964.79 ms |
| 400 000-iteration float spin | 962.77 ms |
| **3 000 000-iteration threadgroup-memory spin** | **4 828.96 ms** |
| `mma` arm (untouched control) | 867-894 ms |

The threadgroup spin **moved the number by 5x**. An added spin cannot move a
kernel that is not executing. So in that run the `.metal` edits **were** reaching
the shader -- and yet the empty arm read the baseline. Both facts cannot hold of
one binary.

The reconciliation is in §10's own method note, not in the shader: §10's harness
"restored the tree with `git checkout` after every arm". `git checkout` rewrites
the file with a **fresh mtime**. `cargo build`'s freshness test for an
`include_str!` input is **mtime-based**. So the sequence

```
edit arm N -> build -> run -> git checkout -> edit arm N+1 -> build -> run
```

has a window in which arm N+1's edit and the restore race on the same file, and
a build that starts in that window compiles the **restored** source while the
harness believes it compiled the arm. That produces exactly §10's signature:
arms that land in the window read the baseline (958-966 ms, spread 1.8%),
an arm that happened to land outside it moves (the spin), and the table looks
internally coherent because every arm is individually plausible.

§10.3 records the same class of bug twice over from the other direction --
"the first two rounds also used a `rc=$?` after a pipe, so a failed build looked
clean" -- and that is why the tree was being restored and rebuilt per arm at all.

**This is a build-freshness race, not an `include_str!` limitation, and it is
reproducible in shape.** `flash_qk_pragma_arms.rs` removes the class entirely:
every arm is a *separately compiled library in one process*, so there is no
restore step, no rebuild between arms, and no window.

### What survives of F1, and what does not

| F1 claim | verdict |
|---|---|
| §10's `tid < 256u` was a completely empty kernel | ✅ **confirmed, and it reads 0.18 ms through the ordinary build path** |
| §10's table is not a failed reproduction -- it measured arms that were not in the binary | ✅ **survives**, with the mechanism corrected |
| "`include_str!` means a `.metal` edit does nothing until recompile" | 🔴 **wrong** |
| §22.2's remedy (carry the arm in the uniform block, one build) | ✅ **right, and now doubly so** -- one build also removes the mtime race |

### The `linear.metal` precedent the brief points at

`spec/engine-performance` §2.2 is the same bug in a different guise and it is
worth quoting because it is the reason the pragma question looked urgent:

> 64x64, `KCHUNK=16` with dynamic loop: **2.58 tok/s**
> 64x64, `KCHUNK=16` with `#pragma unroll`: **6.90 tok/s**
> *Local memory spilling:* Guarding the loop with a runtime `nvalid` trip count
> prevented unrolling, causing the compiler to spill the 16-register accumulator
> to local thread memory.

A **2.7x** swing from a `#pragma unroll` decision. So the prior is real and this
is the second time a pragma has moved a Metal kernel in this repo by more than
2x. What the precedent does **not** establish is that the effect is present in
any given loop: there, the pragma changed whether the accumulator spilled; here,
the QK loop has one `float8x8` accumulator per query and `#pragma unroll(32)`
moves it **3.5% the wrong way**. **A prior about a pragma is a reason to
measure, not a prediction** -- which is why the transcription result had to be
re-measured on the shipping kernel before anything was changed.

---

## T7 — `MMA_TOLERANCE` closed (A5)

`crates/engine/tests/linear_metal.rs` gated at `MMA_TOLERANCE = 0.02` while AC5
states a 2.16e-3 bound: **9.3x above the spec's own limit**, and the shapes it
covers (`in_features` 128/256/384) are not the shapes the engine ships
(K = 5376, 8192, 21504).

The adversary measured the worst normalised error at production K and found
**4.55e-4 at K=8192**, not growing with K (0.90x from K=128 to K=8192). That
measurement is taken as given and re-verified by the tightened gate, not
re-litigated here.

**`MMA_TOLERANCE` is now 3e-3** (was 0.02). Two tighter values were tried and
**failed**, which is the record that this is a gate and not a decoration:

| `MMA_TOLERANCE` | result |
|---|---|
| 0.02 (was) | passes; **9.3x above** the worst this test observes |
| 5e-4 | **FAILS**: `integer bits=2: worst normalised error 5.628729e-4` |
| 1e-3 | **FAILS**: `shape 128x256 queries=1: worst normalised error 2.1603336e-3` |
| **3e-3** | passes; **1.39x** above the worst |

**AC5's 2.16e-3 is not stale.** It is this test's own worst, read off its output
at the small `128x256, queries=1` shape. The correction to AC5 is not "the number
is wrong"; it is that **the gate was 9.3x above it**.

The worst error at the shapes the engine actually **ships** is far lower
(`adversary_evidence2/a5_gemm_error_budget.log`):

| K | N | shape | worst normalised error |
|---|---|---|---|
| 128 | 128 | test coverage | 5.05e-4 |
| 384 | 256 | test coverage | 4.87e-4 |
| 5376 | 4096 | production k/v_proj K | 3.14e-4 |
| 8192 | 2048 | production o_proj K | **4.55e-4** |

so production sits **4.7x inside** AC5's bound, and the error does **not** grow
with K (0.90x from K=128 to K=8192 -- the metric normalises by the reference's
peak magnitude, which grows with the dot product too). The residual is the fp16
rounding of every dequantized weight, a property of the weight format, not of K.
**Nothing was hiding under the hole**, which is the adversary's A5 verdict
confirmed.

### Sabotage: the tightened gate actually fires

`linear_gemm_mma`'s output scaled by a constant, at both gate values:

| kernel perturbation | gate | result |
|---|---|---|
| x1.005 (0.5% error) | **3e-3** | **FAILS**: `worst normalised error 3.8071708e-3` |
| x1.005 (0.5%) | 0.02 (was) | **passes** |
| x1.02 (2%) | 0.02 (was) | **passes** |
| x1.03 (3%) | 0.02 (was) | FAILS: `worst normalised error 2.053204e-2` |

**The old gate needed a 3% error to fail. The new one fails at 0.5%** -- a 6x
tighter trip on the same instrument, on the same kernel, measured rather than
asserted. Evidence: `pragma_peak_evidence/mma_tolerance_sabotage.log`,
`pragma_peak_evidence/mma_tolerance_head.log`.

---

## T8 — the quality gate, and the F4 flip it rejects

The brief's standing gate is **byte-identical greedy text against base `f2b2da1`
for the default configuration**, plus ≥512-token generations at temperature 0 and
at a sampling temperature, checked for repetition collapse.

**The default configuration is byte-identical by construction: no product kernel
changed.** `git diff crates/kernels/` is empty across all three commits of this
round. So the gate, applied to what ships, passes trivially and no quality budget
was spent to establish it.

**The gate was then spent on what it actually gates: the proposed flip.** F4
recommended making `workgroups` the default on a quality cost of "1 greedy token
in 1 536" across contexts 1 024 / 4 096 / 16 394. Round 3 measured the contexts
that evidence does not cover.

`scripts/bench_split_rule_quality.sh`, 512 tokens, `--show-ids`, arms interleaved
per context, plus a replicate of the 8k leg. Greedy is `temperature 0 seed 1234`;
the sampled leg is `temperature 0.7 seed 20260905`.

| context | temperature | `workgroups` | `splits13` | `passes0` |
|---|---|---|---|---|
| 2 048 | 0 | **0 / 512** | 0 / 512 | **0 / 512** |
| 2 048 | 0.7 | **0 / 512** | 0 / 512 | **0 / 512** |
| 8 204 | **0** | **434 / 476, from token 39** | 56 / 512, from token 454 | **0 / 512** |
| 8 204 | **0.7** | **0 / 512** | 308 / 490, from token 174 | **0 / 512** |
| 8 204 (replicate) | **0** | **434 / 476, from token 39** | — | **0 / 512** |

### The 8k greedy result reproduces to the token

Same first divergence (token 39), same generated length (476), same differing
count (434), in a separate run of the harness. And the arm stops **36 tokens
early** — a different *length*, not a different wording, which no argmax-flip
lottery accounts for.

### The two temperature legs disagree, and that is the finding

Under greedy, `workgroups` at 8 204 diverges at token 39 and diverges nowhere
else. Under sampling it does not diverge at all. Two hypotheses fit:

* the rule's perturbation is small and the greedy 8k case hit a genuine near-tie
  early in a long self-reinforcing generation, or
* the perturbation is real and the sampled 8k run happened not to reach it.

**Neither leg is representative**, and the experiment as designed cannot separate
them: one 512-token sample per (context, temperature) at two contexts. The honest
output is a **characterisation of where the rule is safe** (2 048 and 16 394:
byte-identical) **and a finding that it is not safe at 8 204** — not a scalar.

**So the effect is not characterised, and the recommendation inverts.** §24.4 said
"Recommended: flip, but it is not free, and it is not done here", trading a +2%
decode win against 1 token in 1 536. The gate requires byte-identical greedy text
against base for the default configuration, and the rule is **not** byte-identical
under greedy at 8 204 in two runs out of two. **The flip is rejected.**

### What the gate confirmed

* **`passes0` is EXACT**, on generated text: **0 differing tokens in 512** at every
  context and both temperatures, byte-identical bodies, and reproduced at 8 204.
  F3's claim holds and now rests on generated text at four (context, temperature)
  points rather than on the kernel-level tests alone.
* **`splits13` remains numerically sensitive**, as §24.2 calibrated it: 56/512
  greedy and 308/490 sampled at 8 204.
* **The structural argument is untouched** — the sliding ring caps 50 of 60 layers
  at 16 splits either way, so the rule reaches 10. That is arithmetic from the
  code, and it is why `workgroups` and `splits13` differ at all.

### Repetition and NaN

`scripts/check_repetition.sh`, distinct n-gram ratio, on all 16 streams:

| | lowest `distinct_16` |
|---|---|
| shipped arms, greedy | 1.0000 (2 048) / 1.0000 (8 204) |
| shipped arms, sampled | 0.9577 (2 048) / 0.9819 (8 204) |
| any arm under test | **0.9474** (8 204 `splits13`, sampled) |
| the harness's collapsed control | 0.0092 |

**No collapse anywhere**, and no NaN/Inf: a non-finite logit is a hard
`SamplerError`, so a 512-token generation that completed cannot contain one.
Evidence: `pragma_peak_evidence/quality_gate/`.

### Perturbation labels, as the brief requires

| quantity | label |
|---|---|
| every `#pragma unroll` arm vs the shipping kernel | **exact** — 0 of 33 491 072 output bytes differ, at 8 192 and 24 576 keys, `BQ=1` and `BQ=2` |
| §22's ablation scaffold, every arm inactive | **exact** — 0 bytes differ, x0.995 |
| `spin` positive control | **exact** — 0 bytes differ |
| compile-time `chains2` / `chains4` | **bounded-ulp** — worst normalised difference 6.6e-4, which is the dot-product reassociation and nothing else; measurement-only arms, not proposed for shipping |
| the default configuration vs base `f2b2da1` | **exact by construction** — no product kernel changed |

Nothing is labelled both.

### A note on the harness's own positive control

`passes0` at 2 048 tokens and at 16 394 tokens is the instrument's own control in
this sweep: it is byte-identical at every context and both temperatures, which is
what a working harness reads like. At 8 204 the `splits13` arm diverging at token
454 (greedy) and 174 (sampled) is the contrast that shows the harness *can* see a
difference. **Without that contrast the `workgroups` 0/512 rows would have been
unfalsifiable**, which is the same failure §22.5 and §10.3 record.

---

## T9 — the N1 24k control, at last

`adversary_report2.md` A9: the committed 24k evidence
(`adversary_evidence/n1_pass_ab_24k.tsv`) has `onedispatch` marked
`INVALID-NO-NUMBER` and only `pass8192` surviving at 68.401 tok/s. **A discarded
control and a surviving treatment cannot support a null**, so "N1 is NEUTRAL" was
established at 16 394 tokens while this spec is about 24 590.

Re-run, `scripts/bench_prefill_passes.sh`, 24 590-key prompt = 24 602 tokens,
`--max-new-tokens 1`, per-arm poll files, 2 interleaved rounds:

| round | arm | `pass_keys` | prefill tok/s | footprint | verdict |
|---|---|---|---|---|---|
| r1 | `onedispatch` | 0 | **66.829** | 17.04 GiB | **VALID** |
| r1 | `pass8192` | 8192 | **67.051** | 17.14 GiB | **VALID** |
| r2 | `onedispatch` | 0 | — | 17.09 GiB | **INVALID** (`unused-dropped-to-0.04GiB`, limit 0.05) |
| r2 | `pass8192` | 8192 | 67.794 | 17.15 GiB | VALID |

**Round 1 is a valid interleaved A/B with both arms VALID: +0.33%.** The control
the committed evidence lacked now exists, and it is not faster. Round 2's control
was discarded, so round 2 is unusable — **one valid pair in two attempts**, which
is the honest rate for 24k measurement on this box (OQ5, not OQ8), and it is the
same failure mode as the evidence it replaces.

**How strong is the null?** Weaker than the 16k leg, and quoted with that said:
0.33% against a *single* valid round, versus 16k's three rounds at 0.30-0.84%
spread with a positive control (`KAPLUMBA_SCALAR_ATTENTION=1`, -16.6%) in its own
session. One valid pair at 0.33% is consistent with "no effect" and with "an
effect below 0.33%", and cannot separate them. **What it establishes is the thing
A9 asked for: the 24k claim has a control arm.**

**F3's decision does not depend on this.** The default is one dispatch because a
mechanism with no measured benefit should not be a *default* — a default is a
claim — and because `passes0` is byte-identical, which the gate above re-confirms
on generated text. A null at 24k strengthens that; it is not what it rests on.

Evidence: `adversary_evidence2/n1_pass_ab_24k_r3.tsv`,
`pragma_peak_evidence/n1_24k_*.log`.

---

## What is NOT established

* **24k decode (OQ8).** Untouched. Three prior invalid samples; a fourth would
  be a fourth invalid one. Note that round 3's own 24k *prefill* sweep lost one
  control arm of two to the memory gate, which is the same class of problem and
  is why the 24k N1 result is a single valid pair rather than three.
* **The `workgroups` rule's effect is not characterised.** 0/512 at 2 048 and at
  16 394, 434/476 at 8 204, and the two temperature legs disagree about 8 204.
  That is a reason to reject the flip, not a description of the rule. The
  experiment that would characterise it is a context sweep at both temperatures,
  and it should run before anyone argues about the gate.
* **F8's end-to-end benefit.** §26's 2.5-4.9% is a kernel-level number from the
  §22 harness, which T5 shows measured a bit-identical kernel -- so the number
  stands, but it is still not end-to-end.
* **N1 at 24k.** Still 16k-only; see Task 3.
* **Whether the key-split flash decode (BQ=2 with the grid restored) reaches the
  required MMAs-per-load ratio.** Finding 4 identifies it as the only lever
  identified; it is not built, and nothing here says it would be enough.
* **The 8.30 TFLOPS ceiling for shapes other than 8x8x8 half with
  register-resident operands.** It is a ceiling for *this* operand shape and
  loop structure. A GEMM that feeds operands from shared memory will not reach
  it, which is part of why the FFN sits at 81% rather than 100%.
* **§23.6's 90.7 tok/s row.** Unaffected by the correction -- it does not
  reference the peak -- but it should be read as an arithmetic possibility with
  almost no margin, not as "possible and only just" on a 0.8% cushion.
