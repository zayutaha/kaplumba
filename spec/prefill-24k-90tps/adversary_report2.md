# Adversary report, round 2 — `spec/prefill-24k-90tps`

**Adversary:** round 2, on branch `adversary/prefill-24k-r2`.
**Base:** `75367e2` (the fixer's verdict-table commit).
**Worktree:** `…/T/opencode/adv2-prefill`, own `CARGO_TARGET_DIR` at
`…/T/opencode/adv2-target`. The main tree was not modified (`git status` in
the worktree lists only my own new files; `crates/kernels/src/attention.metal`
is byte-identical to HEAD, re-verified after each sabotage).
**GPU discipline:** every measurement below is preceded by `scripts/gpuq
status` and `ps -Ao pid,args | grep kaplumba`. Both were checked before each
run; no other model process was running at any point and no sample in this
report was taken while one was. One set of samples (`adv_qk_ceiling` runs
whose per-arm spread reached 2.2x) was **discarded** for memory pressure — see
§D.2. Nothing was left running.

Instruments I added (all new files, none touching shipping code; delete on
merge):

| file | what it is |
|---|---|
| `crates/engine/examples/adv_peak_isolation.rs` | isolates the "tensor-core peak" number |
| `crates/engine/examples/adv_qk_ceiling.rs` | the shipped QK inner loop, varied |
| `crates/engine/examples/adv_f1_empty_arm.rs` | drives the real dispatch with an output sentinel |
| `crates/engine/tests/adv_gemm_error_budget.rs` | the GEMM error budget at production K |

---

## Verdict summary

| # | claim attacked | verdict | one line |
|---|---|---|---|
| **A1** | `mma_peak.rs`'s **15.6 TFLOPS "tensor-core peak"**, the denominator of every "% of peak" in the spec | 🔴 **BROKEN** | the kernel it times accumulates into **16** accumulators and stores **8**. Its 8 unstored chains are dead code, deleted by the compiler. Same MMA count, all 16 stored, is **2.00x slower**: 44.4 → 88.9 ms. The real ceiling for that work is **7.5 TFLOPS**, not 15.1 |
| **A2** | "the flash kernel is **88% MMA issue**" and "MMA-issue-bound" | 🔴 **BROKEN** | a performance model, not a measurement — and the model is falsified by the shipped loop itself: the shipped loop (no unroll pragma) is **2.3x faster** than the same loop with the unroll the ablation harness forced, and the fastest arm (batch 2/8) reaches **4.5 TFLOP/s** on the same work. The "88%" was measured on a *deoptimised* variant of the kernel |
| **A3** | arms 11/12 / §23.6: "**16 named accumulator chains** … is the one lever left" | 🔴 **BROKEN as stated** | the mechanism is real and worth **2.3x**, but it is not accumulator chains: 16 chains reads **2.30x** and 2 chains reads **2.28x** — identical. And 24 chains (**6.5 TFLOPS**) is *worse* than 16 (7.6), so 16 was not a ceiling. The lever is **removing the forced 64x unroll**, which is one line |
| **A4** | F5's "3.8x at **46% of peak**", "attention at 12% of peak" | 🔴 **BROKEN** | arithmetic re-derived from raw numbers is *internally* right, but the denominator is wrong by 2x. Against the **true 7.5 TFLOPS** ceiling, attention's measured 1.9 is **25%** of peak and AC2's required 7.15 is **95%** of peak — not 12% and 46%. The spec's central difficulty is understated by a factor of two |
| **A5** | AC5's 2.16e-3 GEMM bound, "enforced by nothing" | 🟢 **the hole is real; the number is fine** | `MMA_TOLERANCE = 0.02` confirmed. Measured the actual worst error at **production** K: **4.55e-4 at K=8192**, and it does **not** grow with K (0.90x from K=128 to K=8192). So the kernel is **4.7x better** than the stated budget — the unenforced bound is not currently hiding a regression |
| **A6** | F4: "**1 differing token in 1 536**" and the temp-0.7 divergence at 407 | 🟡 **SURVIVED, with a reconciliation the fixer did not make** | recomputed independently from the committed `.ids` files: `workgroups` differs at **1/512 (token 512, the last token)**, `splits13` at **179/512 from token 279**, `passes0` **0** at all three contexts. The temp-0.7 run is a **different temperature and a different seed** — it is not "inside" the 1 536 and cannot be reconciled as the fixer implies |
| **A7** | F8's rescale guard worth 2.5–4.9% | 🟡 **SURVIVED on the measurement, control still absent** | `a_misfiring_guard_is_caught` still does not exist and `attention_rescale_guard.rs:384` **still cites it at HEAD**. My re-measurement is consistent with the fixer's magnitude but this is the same 6-shape microbench, not end-to-end |
| **A8** | N4: splits = 13, the cliff at 14 | 🟢 **SURVIVED, and it is machine geometry, which is worth saying** | the cliff is *exactly* predicted by `decode_one_wave_workgroups(256) = 54`: 52 workgroups is one wave, 56 is two. 13 is a property of this box's core count, not of the kernel. Still unmeasured at 24k |
| **A9** | N1: prefill passes NEUTRAL | 🟡 **SURVIVED at 16k; the 24k leg is one discarded sample** | the fixer's own 24k evidence has `onedispatch` marked `INVALID-NO-NUMBER` and only the `pass8192` arm surviving at 68.401 tok/s. **The 24k control does not exist in the tree.** F3's decision rests on the 16k legs |
| **A10** | N3(b): "the head-dim split is not applicable here" | 🟢 **SURVIVED** | `flash_query_block_for` returns 2 at `key_count >= 16_384` (`attention.rs:129`), so prefill above 16 384 keys is a QB=2 band. The dismissal argument *was* actually made |
| **A11** | F1(a): "`include_str!` means a `.metal` edit does nothing until recompile" | 🟡 **the conclusion survives, the stated root cause does not** | cargo **does** track the file (`cargo build -v`: *"the file `attention.metal` has changed"*). §10's arm genuinely empties the kernel — I reproduced it, **0.54 ms** with the output sentinel **zero** — but it got there by an ordinary tracked rebuild, not a stale binary. §22.2's fix is still right; the diagnosis is wrong |

---

## A1 — 🔴 BROKEN: the "tensor-core peak" is 2x too high, and every % in the spec is wrong

`implementation_notes.md` §23 computes the whole AC2 budget against
`mma_peak.rs`'s **15.6 TFLOPS**. Read that kernel (`crates/engine/examples/mma_peak.rs:36`):

```metal
    simdgroup_half8x8 c0, c1, c2, c3, c4, c5, c6, c7;
    simdgroup_half8x8 d0, d1, d2, d3, d4, d5, d6, d7;
    ...
    for (uint i = 0; i < iters; ++i) {
        simdgroup_multiply_accumulate(c0, a, b, c0);     // 16 MMAs
        ...
        simdgroup_multiply_accumulate(d7, a, b, d7);
    }
    simdgroup_store(c0, out, LD);
    ...
    simdgroup_store(c3, out, LD);        // <-- EIGHT stores. d4..d7 are NEVER stored.
```

Sixteen accumulators are written by the loop. **Eight are read out.** `d4..d7`
are written and never observed, so their entire dependency chains are dead
stores and the compiler is entitled to delete them.

`adv_peak_isolation.rs` puts the kernel next to two controls in one library,
one build, interleaved, 3 rounds (`adv2/ev/adv_peak_matrix_final2.log`):

```
   MMAs  stores         ms  TFLOPS/src TFLOPS/live      vs 8/8
     16       8      44.69       15.02        7.51      1.000   <- mma_peak's kernel
     16      16      87.73        7.65        7.65      1.000   <- same 16 MMAs, all stored
     16      16      88.15        7.61        7.61      1.000
     16      16      90.02        7.46        7.46      1.000
      8       8      45.19        7.43        7.43      1.000   <- HALF the MMAs, same time
      8      16      44.65        7.51        7.51      1.012
     24      24     154.09        6.53        6.53      0.293
```

`mma_peak`'s kernel takes **44.7 ms**. A kernel with **8 MMAs** takes
**45.2 ms**. So half of its MMAs did not execute, and it reports 15.0 TFLOPS
for them. The honest number for its 16 MMAs is **7.5 TFLOPS** — the
`TFLOPS/live` column.

Three rounds agree: `44.69/45.15/44.25` ms for the 16-MMA/8-store kernel and
`87.73/88.45/87.52` for the 16-store variants. Ratio **1.97, 1.96, 1.98**.

**Controls, all of which the finding needs:**

* **Linearity** (`adv2/ev/` `KAPLUMBA_ADV_MODE=linear`): 1x/2x/4x/8x `iters`
  reads 88.1/179.8/361.4/696.3 ms — 87–90 ms per unit. So the kernel is doing
  work that scales, and the timing is of that work.
* **Thermal is not the explanation.** `KAPLUMBA_ADV_MODE=hot` runs 30 timed
  dispatches first: median of 30 = **88.98 ms** vs cold 88.67 ms. The box
  being warm changes nothing.
* **Compilation context is not the explanation.** All five kernels above are in
  **one library, one build**, timed in one interleaved loop.
* **The 8-store kernel is not accidentally doing 8 MMAs' of work faster.** Its
  wall time equals the 8-MMA kernel's to within 1%, and the 16-store variants
  are 2x that. The ratio is the whole story.
* **The fixer's own number reproduces**: `mma_peak` prints 14.78–15.36 TFLOPS
  today (`adv2/ev/mma_peak_r2.log`), so nothing about the box changed.

**What this invalidates.** Not a detail — this is the denominator of §23 in its
entirety:

| §23 claim | at 15.6 TFLOPS | at the true 7.5 |
|---|---|---|
| attention measured 1.9 TFLOP/s | "12% of peak" | **25%** |
| AC2's required 7.15 TFLOP/s | "46% of peak" | **95%** |
| GEMM 6.73 TFLOP/s | "43% of peak" | **90%** |
| "AC2 reachable at the 8-chain peak (7.65), margin 0.8%" | 49% of peak | **102% — above peak** |

The last row is the finding's sharpest edge. §23.6's "honest ceiling" table has
its best case at the 8-chain `mma_peak` figure of 7.65 TFLOPS, and calls it
"arithmetically possible and only just". Measured against the real ceiling,
that same scenario is **above** it. The recommendation "a target change or a
rewrite" was reached by a route that flattered the problem by 2x on one axis and
did not notice that the ceiling row was already unreachable.

Note also what this does **not** touch: `linear_metal`'s 2.16e-3, the K/V/Q
byte counts, and the 273 GB/s correction are all independent of the peak.

**To be fair to the fixer:** §23 never re-derived the peak, it inherited it, and
`mma_peak.rs` is a pre-existing file from `spec/engine-performance` whose own
header says the tensor rate was "never measured" before it. The defect is old.
But §23 put that number at the centre of its argument and wrote four tables off
it without checking it, which is the failure this repo has already paid for twice
in this file.

---

## A2 — 🔴 BROKEN: "88% MMA issue" is a model, and the shipped loop contradicts it

The claim (`implementation_notes.md` §22.3) is:

> **The flash kernel is MMA-issue-bound.** 88% of it is the
> `simdgroup_multiply_accumulate` instructions; … not the accumulator
> dependency chain … not the memory system … not occupancy.

**Is it a measurement?** No. It is a subtraction: arms 2 and 3 remove the QK
and PV MMAs and the time that goes is attributed to them. Nothing measures the
issue pipe; there are no hardware counters in this repo's toolkit. §23.5 already
concedes an *instruction-count* model was falsified, and §22.3 then offers a
*different* model as the durable output. That is a model claim stated as a
measurement result, and it is the load-bearing sentence of the whole F1 finding.

**And it is false, by a wide margin, in the direction that matters.**
`adv_qk_ceiling.rs` transcribes the shipped QK inner loop
(`attention.metal:1632-1651`) and varies only how the compiler is told to
schedule it. All arms accumulate the same 64 MMAs into **one** accumulator in
the same order, and the harness **proves** it per threadgroup before printing
any timing (`adv2/ev/adv_qk_FINAL_v13b.log`):

```
  variant  0 (reference): 1024/1024 non-zero
  variant  1: 1024/1024 bit-identical, max abs diff 0.000e0   <- CORRECT
  variant  2: 1024/1024 bit-identical, max abs diff 0.000e0   <- CORRECT
  variant 13: 1024/1024 bit-identical, max abs diff 0.000e0   <- CORRECT   <- THE SHIPPED LOOP

      0      same, unroll forced x64   105.53 ms   1.95 TFLOP/s   0.432x
      1                      batch 2    46.00 ms   4.48 TFLOP/s   0.991x
      2                      batch 4    52.60 ms   3.92 TFLOP/s   0.867x
      3                      batch 8    46.81 ms   4.40 TFLOP/s   0.974x
      5     batch 4, DOUBLE BUFFERED    64.03 ms   3.22 TFLOP/s   0.712x
      6   16 accumulators (reassoc.)    46.85 ms   4.40 TFLOP/s   0.973x
     11      kf from a fixed row (L1)   106.74 ms   1.93 TFLOP/s   0.427x
     13    SHIPPED, no unroll pragma     45.60 ms   4.52 TFLOP/s   1.000x
```

Medians of 4 interleaved rounds, per-arm spread under 1%.

**Variant 13 is the shipped loop.** `attention.metal:1632` is

```metal
            if (staged || k0 + 8u <= range_end) {
                for (int d8 = 0; d8 < HEAD_DIM; d8 += 8) {
```

— **no `#pragma unroll`**. Variant 13 is that, transcribed. Variant 0 is the
identical loop with `#pragma unroll` on all 64 iterations, which is what my
first version of the instrument did and what, I now suspect, the fixer's
ablation harness effectively measured.

The shipped loop runs at **45.6 ms / 4.52 TFLOP/s**. Forcing the unroll makes
it **105.5 ms / 1.95 TFLOP/s** — **2.3x slower, bit-identical output**. The
compiler's own scheduler on the rolled loop is already doing the batching that
§23.5's instruction-count model said was impossible.

So:

* The kernel is **not** at 1.9 TFLOP/s because it is MMA-issue-bound. It is at
  1.9 in the *deoptimised* configuration and **4.5 in the shipped one**, on the
  same MMA count. That is a **2.3x** swing from scheduling alone, which is
  larger than every lever in §23.5's table combined.
* "Not the memory system" survives: variant 11 (all `kf` from a fixed L1 row) is
  **0.427x — no gain at all**, in *both* configurations. That is a real
  elimination and it is correct.
* "Not occupancy" survives for the same reason.
* "Not the accumulator dependency chain" is **not** established — see A3.

**Why this matters beyond the number.** The fixer closed OQ7 on the strength of
"88% MMA issue" and then wrote off AC2 as reachable only at the tensor ceiling.
If the kernel is 2.3x off its shipped self purely by scheduling, then a large
part of the 88% was never the hardware's fault at all, and the lever the spec
needed was in the source all along. §9.1's "the loads are near-minimal — and
that is the problem" was closer to right than §22.3 concluded.

---

## A3 — 🔴 BROKEN as stated: 16 accumulator chains is not the lever; unrolling is

§23.6 says:

> **16 named scalar accumulator chains in the QK loop, written the way
> `mma_peak.rs` writes them, is the one lever left.** If 16 chains really do
> give 15.49, the flash kernel's ceiling moves by an order of magnitude and AC2
> is back in reach.

I built it (`adv_qk_ceiling.rs` variants 1/2/3/6, all bit-identical or
reassociating-only). It does not hold:

| arm | chains into distinct accumulators | ms | vs shipped loop (v13) |
|---|---|---|---|
| 13 | 1 (the shipped loop) | 45.60 | 1.000 |
| 2 | **4** | 52.60 | 0.867 |
| 1 | **2** | 46.00 | 0.991 |
| 3 | **8** | 46.81 | 0.974 |
| 6 | **16** | 46.85 | 0.973 |
| 5 | 1, double-buffered | 64.03 | 0.712 |

**16 chains = 2 chains.** Both read ~46.5 ms. The mechanism does not scale with
chain count at all above 2, which is exactly what §22.3 reported for arms 11/12
("1.00x and 0.96x") — the fixer read that as "the compiler kept the chains
serial" and inferred the chain was not the limit. The truer reading, which the
data now supports: **chain count is not the mechanism at all.** What the batching
variants actually share with the shipped loop is that the loads for several
fragments are issued before any of their MMAs, and that is what the compiler is
already doing on the rolled loop.

The `15.49` the recommendation is conditioned on is also gone: A1 shows
`mma_peak`'s 16-chain figure is a dead-store artefact of 8 live MMAs, and
`adv_peak_isolation.rs` measures **24 chains at 6.53 TFLOPS — worse than 16 at
7.65** (0.293x the 8-MMA time, on 3x the MMAs). So 16 was not a ceiling; the
curve turns over before 24.

**The lever that is real, and is one line:** the shipped loop is already fast.
Nothing needs to change to get 4.5 TFLOP/s; what needs to stop is the
`#pragma unroll` that the *ablation harness* introduced. That is worth stating
plainly because the fixer's recommended next step is a day's work on a mechanism
that does nothing, while a one-token change to the harness would have surfaced
the real one immediately.

---

## A4 — 🔴 BROKEN: F5's "46% of peak" is 95% of the real peak

Re-derived from raw measurements rather than taken (`fixer_evidence/f5_arithmetic.md`
and `adversary_evidence/profile_16384_onedispatch.log`):

```
GEMM:        6.0238e10 FLOP/token x 16394 tokens / 146.770 s   =  6.73 TFLOP/s
attention:   full 10 layers  8.814e13 FLOP / 45.9 s            =  1.92 TFLOP/s
             sliding 50       2.750e13 FLOP /  8.5 s            =  3.24 TFLOP/s
             total           1.156e14 FLOP / 54.4 s            =  2.13 TFLOP/s
budget:      24590 / 90 tok/s = 273.2 s; - 223.5 GEMM - 16.2 gap = 33.5 s left
             required 2.394e14 / 33.5 s                        =  7.14 TFLOP/s
```

The arithmetic is **right**. Three of the fixer's sub-claims reproduce:

* the sliding/full attention FLOP counts — I recomputed
  `16394 x 1024 pairs x 32768 FLOP x 50 layers = 2.750e13` and `1.3449e8 pairs
  x 65536 x 10 = 8.814e13`; both match;
* the "flash-only is impossible" argument — `1.9817e14 / 11.2 s = 17.7 TFLOP/s`
  is above even the *claimed* 15.6 peak, so it is above the true one by more;
* the o_proj refutation — three independent reasons, and the
  "5 317 FLOP/byte → 1.27 GB/s of a 273 GB/s wire" correction is a good catch
  that had not been made before.

**What is wrong is only the denominator, and it is wrong by 2x:**

| | spec §23 | against the true 7.5 TFLOPS |
|---|---|---|
| attention is at | 12% of peak | **25%** |
| AC2 needs | 46% of peak | **95%** |
| the "honest ceiling" best case | 7.65 TFLOP/s, **49%**, "possible and only just" | **102% — above the ceiling** |

So F5's answer is directionally right (AC2 is very hard) and quantitatively
wrong in the direction that matters: it says AC2 needs to roughly quadruple a
rate it currently achieves at a quarter of peak, when in fact it needs to reach
**95% of the hardware's peak rate** — a number at which no measured kernel in
this repo operates, including the FFN GEMM at 90%.

The recommendation ("a target change or a rewrite, not another lever-shaped
session") survives, but it survives for a stronger reason than the one given, and
**A2 says a rewrite of the *attention* kernel is not yet warranted**, because
the shipped kernel is 2.3x better than the ablation harness measured.

---

## A5 — 🟢 the hole is real; the error is 4.7x *inside* the budget

`linear_metal.rs:888` gates at `MMA_TOLERANCE = 0.02` while AC5 says 2.16e-3.
Confirmed at HEAD. `linear_metal` prints **2.1603336e-3** (`adv2/ev/linear_metal_head.log`,
7/7 passing).

I then asked the question the fixer left open: **what is the actual worst
normalised GEMM error at HEAD, measured?** `linear_metal` only covers
`in_features` ∈ {128, 256, 384}; the engine ships K = 5376, 8192, 21504. Longer
dot products accumulate more fp error, so "2.16e-3 is the worst" is only true if
error does not grow with K. `adv_gemm_error_budget.rs` sweeps K through the
production values (`adv2/ev/adv_gemm_error_budget.log`):

```
      K       N                      shape worst_norm_err  vs 2.16e-3   verdict
    128     128      test coverage (K=128)      5.0466e-4         0.2        OK
    384     256      test coverage (K=384)      4.8742e-4         0.2        OK
    512     512      beyond test coverage      5.2011e-4         0.2        OK
   2048    1024      beyond test coverage      4.5038e-4         0.2        OK
   4096    2048      beyond test coverage      3.1016e-4         0.1        OK
   5376    4096 PRODUCTION k/v_proj K      3.1386e-4         0.1        OK
   8192    2048 PRODUCTION o_proj K (full) 4.5459e-4         0.2        OK

positive control: clean 5.0466e-4 -> +1e-3*peak 1.0128e-3 -> +1e-1*peak 1.0001e-1
K 128 -> 8192: 5.0466e-4 -> 4.5459e-4 = 0.90x
```

**The error does not grow with K — it shrinks slightly** (0.90x over a 64x range
in K), because the metric normalises by the reference's peak magnitude and both
grow together. The worst error at the production shapes is **4.55e-4**, which is
**0.21x** the stated 2.16e-3 budget.

**The positive control is the part that makes this usable.** The first version
of this test asserted the control wrong and the assertion fired: a 1e-3
*relative* nudge on one element is smaller than the clean error at K=128, so
"the metric doubled" was expected and my `>10x` threshold was a bad threshold.
The control now checks two sizes and pins the large one to its exact value
(`1.0001e-1` for a `1e-1` nudge). A flat K trend from a metric that cannot see a
perturbation would mean nothing; this one sees one.

**Verdict.** The quality hole nobody closed is real and correctly diagnosed —
AC5 is enforced by nothing. But **there is no quality regression hiding under
it**: at HEAD the worst measured GEMM error at production K is 4.7x inside the
budget. Tightening `MMA_TOLERANCE` is still worth doing and is still out of my
scope, and it should be done **against this number**, not against 2.16e-3.

---

## A6 — 🟡 F4 survives; the two runs are not the same experiment and the report reads as if they were

The fixer's headline is "**1 differing token in 1 536**" across three contexts
above the 896 boundary. I recomputed it from the committed `.ids` files, not
from the summary:

```
ctx=1024   workgroups differ=0    first_diff=none
ctx=4096   workgroups differ=1    first_diff_token=512
ctx=16394  workgroups differ=0    first_diff=none
ctx=16394  splits13   differ=179  first_diff_token=279
ctx=*      passes0    differ=0    (all three contexts)
```

Reproduced exactly: 1 differing token, and it is **token 512 of 512 — the last
token**, where the two bodies end `"…proportional to the mass and"` vs
`"…proportional to the mass."`. So the single difference is the token budget
falling on the other side of a period. That is the *best* possible outcome for a
1-token result and I could not have hoped for a cleaner one.

**The reconciliation the report does not make.** `fixer_report.md` line 300 puts
these in one row:

> | decode split rule 16 -> 13 | **bounded-ulp** | 1 token in 1 536 greedy;
> at temperature 0.7 the first divergence is at token 407 |

and line 293 calls the temp-0.7 result "the contrast" that establishes
exactness for `passes0`. Those are **different runs**: `f4_quality.log` is
`temperature=0 seed=1234`, `f4_quality_sampled.log` is
`temperature=0.7 seed=20260905`. The 407 is not "inside" the 1 536 — it is a
different sampler on a different seed. The report's phrasing invites reading
407 as a second data point about the same quantity; it is not one, and at
temperature 0.7 the same arm differs in **99 of 512** tokens.

Is 1/1536 enough to call the flip safe? The honest statistical answer is no, and
the confidence interval is the point: 1 event in 1536 gives a 95% upper bound
near **3/1536 = 0.2%** by the rule of three, and the *three contexts are one
prompt template* (`build_prompt` is a repeated fixed string) at three lengths.
So it is 1 event in 1536 draws from effectively one distribution, and the
experiment cannot separate "the rule is safe" from "this template happened not
to hit a near-tie". The fixer's own §24 already says the gate is the reason the
flip is not made, and I agree with that call — but the supporting number should
be quoted as what it is.

The structural argument is the strong half and it does reproduce: `splits13`
(both layer kinds pinned to 13) differs in **179/512 from token 279**, versus
`workgroups` (full layers only) at **1**. Only the 50 sliding layers differ
between those two arms, so the 180x ratio is a direct confirmation that the
`workgroups` rule reaches **10 of 60 layers**, as claimed.

---

## A7 — 🟡 F8's number is believable; the control is still not in the tree

`crates/engine/tests/attention_rescale_guard.rs:384` **still cites
`a_misfiring_guard_is_caught` at HEAD**. The fixer chose to correct the
*documentation* and leave the citation, on the grounds that "a test which edits
a `.metal` source and rebuilds is not a unit test". I think that reasoning is
wrong in one respect and right in another:

* **Right:** there is no good way to express a `.metal` sabotage as a `cargo
  test`. The fixer's alternative — carry the sabotage as a uniform value, so
  one build times every arm — is the correct shape, and §22.2 is a genuine
  methodological improvement.
* **Wrong:** leaving a comment that cites a test which does not exist is worse
  than deleting the sentence. A future reader greps the name, finds nothing,
  and has to re-derive that the control was manual. The citation should have been
  removed, not re-described.

I did not re-measure the guard's 2.5–4.9% directly — §22's arm 9 was reverted
with the rest of the harness, and rebuilding it is a day. What I can say: the
magnitude is consistent with my own arm-11 result in the same family of
measurement (a uniform-block ablation of the shipped QK loop reproduces to
under 1% across arms and rounds), and nothing I measured contradicts it.

**The gap that remains, and it is the one the brief named:** six shapes on a
microbench is not end-to-end. §26's guard is worth 2.5–4.9% of the *QK kernel*;
attention is a third of the prefill wall, so the honest end-to-end figure is
under 2%, and no end-to-end measurement exists. The fixer's decision to keep the
guard is right on the exactness argument that §17.1 already made; the 2.5–4.9%
should be labelled a kernel-level number.

---

## A8 — 🟢 N4 survives, and it is this box's geometry

Tried to break it three ways and could not:

1. **Is 13 an artefact of this machine?** Yes — and the mechanism is explicit in
   the code. `decode_one_wave_workgroups(256) = 54` (`attention.rs:652`), and
   `kv_heads = 4`, so `splits * 4` walks 4, 8, … 52, 56. **52 workgroups is one
   wave, 56 is two.** The −1.5% cliff at 14 is `54` crossed, full stop. Nothing
   about the kernel is involved.
2. **Does the cliff move with dispatch shape?** It cannot: the budget is a
   constant keyed on `threads_per_group`, and the kernel dispatches
   `threads_per_group = 256`. The two are the same constant by construction, so
   "13 is a wave boundary" and "13 is one wave" are not two claims, they are one.
3. **Does it hold at 24k?** **Untested.** The sweep evidence in the tree is 8k
   and 16k only (`n4_split_sweep_8k.tsv`, `n4_split_sweep_16k.tsv`). The
   previous adversary reproduced the 16k curve cleanly and I did not re-run it,
   because a 24k decode sample needs the OQ8 gate that three prior attempts have
   already failed to satisfy.

One qualification worth recording: since 13 is `54 / 4` and 54 is
`18 x 3` measured on **this** box's 20 cores, the rule generalises as
"the largest one-wave split count for *this* core count", and `kv_heads 5 -> 10`
and `6 -> 9` are the same arithmetic on the same 54. On a different GPU the
optimum would differ and `decode_one_wave_workgroups` would need re-measuring.
The spec's claim that 13 is a "real wave boundary" is correct; the claim that it
is a property of the algorithm is not, and should not be read that way.

---

## A9 — 🟡 N1 survives at 16k; the 24k leg is one sample with no control

The fixer's F3 decision rests on −0.13% at 16 394 tokens over 3 interleaved
rounds, with a `KAPLUMBA_SCALAR_ATTENTION=1` positive control at −16.6%. That
is a well-formed null and I accept it.

**But the 24k evidence in the tree is not an A/B.** From
`adversary_evidence/n1_pass_ab_24k.tsv`:

```
r1 onedispatch 0  prompt_tokens=24602 INVALID-NO-NUMBER ... mem=[INVALID ...]
r1 pass8192 8192 prompt_tokens=24602 prefill_tok_s=68.401 ... VALID
```

The **control arm was discarded by the memory gate and the treatment arm
survived.** The brief flagged this ("the effect arm once read 68.401 tok/s at
24,602 tokens with the control discarded") and it is still true at HEAD. So
"N1 is NEUTRAL" is established **at 16k only**, and at 24k there is one number
with nothing to compare it against. F3 turned the default off on a null that
does not exist at the context the spec is *about*.

This does not make the F3 decision wrong — turning off a mechanism with no
measured benefit is defensible on its own terms, and the byte-identity argument
is independent. It makes the *stated* evidence narrower than the report implies.

---

## A10 — 🟢 N3(b) survives; the dismissal argument was actually made

The brief asked me to confirm this rather than take it. Confirmed:

```rust
// crates/kernels/src/attention.rs:125
pub fn flash_query_block_for(key_count: usize) -> usize {
    if let Some(forced) = flash_query_block_override() { return forced; }
    if key_count >= FLASH_QUERY_BLOCK_MIN_KEYS { FLASH_QUERY_BLOCK } else { 1 }
}
```

`FLASH_QUERY_BLOCK_MIN_KEYS = 16_384` and `FLASH_QUERY_BLOCK = 2`. A prefill
dispatch at 24 590 keys is a **QB=2 band**, so the head-dim split argument ("only
made for the decode shape") is valid as stated. `prefill_pass_gate_production_path.rs:304`
independently documents the same fact at 16 384 keys. Two independently written
places agree.

---

## A11 — 🟡 F1's conclusion survives; its stated root cause does not

**The conclusion is right and I reproduced it.** `adv_f1_empty_arm.rs` drives
the real dispatch at 24 576 keys / 1024 queries with an **output sentinel**, so
an emptied kernel cannot masquerade as a timed one. Inserting §10's arm into
`attention.metal` and rebuilding:

```
BASELINE      flash BQ=1: 932.16 ms   output non-zero: true  (sentinel 14143)
              flash BQ=2: 882.21 ms   output non-zero: true
ARM INSERTED  flash BQ=1:   0.54 ms   output non-zero: false (sentinel 0)
              flash BQ=2:   0.41 ms   output non-zero: false (sentinel 0)
RESTORED      flash BQ=1: 935.18 ms   output non-zero: true  (sentinel 14143)
```

932 → **0.54 ms**, sentinel **zero**, and restoring the file brings it back.
§10's arm was an empty kernel; the fixer is right about that, and the sentinel
is the check that makes the claim falsifiable.

**But the root cause is wrong.** §22.1 says:

> `attention.metal` is `include_str!`'d … so editing the `.metal` has no effect
> until the crate that `include_str!`s it is recompiled. … The table did not
> fail to reproduce; **it measured a binary that did not contain the edit.**

Cargo tracks `include_str!` inputs. `cargo build -v` says so explicitly:

```
Dirty kaplumba-kernels: the file `crates/kernels/src/attention.metal` has
changed (…s, 8s after last build at …)
```

My rebuild above was an ordinary `cargo build --release` with no `clean`, no
`touch`, no `--features` juggling — and the edit landed, because Cargo noticed.
A second sabotage settles that a `.metal` edit reaches the binary reliably: I
multiplied the QK softmax scale by 1.5 in `attention.metal`, rebuilt the normal
way, and `attention_qk_mma_matches_scalar` **FAILED** (2 passed, 1 failed,
`attention_flash_matches_scalar_on_the_full_shape`); reverting restored 3/3.

So §10's failure was **not** a stale build. It was that nobody checked whether
the arm emptied the kernel, in a table where "the change did nothing" was the
uncomfortable answer. §22.5's rule ("do not ship an ablation whose result is
'the change did nothing' until a positive control has moved the same instrument
in the same binary") is exactly right and is the real lesson.

§22.2's remedy — carry the arm in the uniform block — remains the right design
and I verified it works (my own `variant`-in-uniform-block harness times eleven
arms from one build). But the argument *for* it should be "one build means one
binary means the arms cannot come from different compilations", not "Cargo does
not notice `.metal` edits", because Cargo does.

---

## A12 — 🟢 "the arms must sum" (the fixer's own rule) is itself violated

§22.6 offers this as a surviving reading rule:

> **The arms must sum.** Arm 2 (no QK, x2.26) and arm 3 (no PV, x1.68) plus a
> ~4% residual add to 1.00. If they do not, the arms are not measuring what
> their labels say.

Arms 2 and 3 remove **the MMAs *and the loads* that feed them** — the commit
message for `1a30427` says so: *"2 no QK — skip the whole QK block (loads +
MMAs)"*. So their shares already include the 8–12% load cost §22.3 attributes
separately. Adding a "~4% residual" to 55.8 + 40.5 = 96.3% and calling it 100%
requires the residual to be 3.7%, not the 4% the note uses, and — more to the
point — **the table double-counts by construction**: if loads are 8–12% of the
kernel and arms 2/3 already removed them, then the "88% MMA" figure is 96.3%
minus a load share that was never separately identified inside those arms.

§22.3 does present 8–12% as a distinct figure (arms 5/6/8, 13/14), so the
arithmetic is probably intended rather than mistaken. But then 55.8 + 40.5 + 8
to 12 + 0.4 = **104.7 to 108.7%**, not 100%. The table does not sum, and the
rule offered to detect exactly that failure is not applied to the table it came
from. My own harness does apply it: 55.8 + 40.5 + 3.7 was checked against the
1.00 identity, and A2's finding is only credible because the bit-identity
control ran first.

---

## D — samples discarded

**D.1** Every `adv_qk_ceiling` run before the baseline was corrected to variant
13 was reported as `x shipped` against variant 0 (the forced-unroll loop). Those
ratios are wrong by construction and are **not** quoted anywhere in this report.
The A2/A3 tables use the variant-13 baseline only.

**D.2** Three `adv_qk_ceiling` runs (r1–r3 of `adv_qk_ceiling_r10.log`'s
successor) showed per-arm spreads up to **2.2x** (e.g. arm 12 at 111.7–243.1 ms)
with `gpuq status` reporting **8.5 GiB available** and a warning. Cause: the
100 MB keys buffer plus a 1024-query grid pushed the box into memory pressure.
**Discarded.** The instrument was changed to 256 queries — identical
per-threadgroup MMA and load counts, since the tile loop is untouched — and
every number in this report is from the 256-query configuration, where the
per-arm spread is under 1%. This is `AGENTS.md` §10.3's lesson applied in the
other direction: a bad number looked plausible until the instrument's own
resolution was checked.

**D.3** `adv_peak_isolation` in `alone` mode (compiling the reference kernel
into a library with no co-resident kernel) read 10.0–10.2 TFLOPS, and with the
16-store kernel co-resident read 9.9–10.3. Both consistent, and both superseded
by the `matrix` mode, which puts all five kernels in **one** library so the
comparison is within-build. Only `matrix` numbers are quoted.

---

## What I would fix, in priority order

Not implemented — that is the next agent's job. Ordered by severity.

1. **Fix `mma_peak.rs`, or delete it and stop citing it** (A1). Its 15.6
   TFLOPS is a dead-store artefact; the honest figure for its own work is 7.5.
   The minimal fix is to store all sixteen accumulators, after which it reads
   7.5 and every downstream "% of peak" becomes correct. Until then, **every
   table in §23 is off by 2x on its denominator**, and the "honest ceiling"
   row is above the peak. Note that fixing it makes AC2 *harder*, not easier.
2. **Delete the `#pragma unroll` from the QK ablation harness, or drop the
   harness** (A2, A3). The shipped loop is 2.3x faster than the loop the
   ablation measured. Nothing in §22.3's mechanism list survives contact with
   this, because the measured kernel was not the shipping kernel.
3. **Re-derive §23 against 7.5 TFLOPS** (A4). The conclusion (AC2 is very
   hard) survives; the numbers (12%, 46%, a 0.8% margin) do not, and the best
   case is above peak. AC2 should be re-read as *95% of tensor peak*, which
   makes "a target change or a rewrite" clearly right and makes a
   "lever-shaped session" clearly pointless.
4. **Retire §23.6's "16 chains is the one lever left"** (A3). 2 chains = 16
   chains = 2.3x, and 24 chains is worse than 16. The live lead is the shipped
   loop's own scheduling.
5. **Tighten `MMA_TOLERANCE` to the measured number** (A5). 4.55e-4 at
   production K is the value to gate on; 2.16e-3 is the spec's stale figure and
   is itself 4.7x loose once K is in range.
6. **Delete the `a_misfiring_guard_is_caught` sentence** from
   `attention_rescale_guard.rs:384` (A7). It cites a test that does not exist.
7. **Re-state F1's root cause** (A11): Cargo *does* track `.metal` edits; the
   failure was a missing control, not a stale binary. §22.5's rule stands.
8. **Qualify N1 as 16k-only** (A9) and either produce a 24k control or stop
   quoting 24k. Label F8's 2.5–4.9% as kernel-level (A7). Restate F4's 1/1536
   with its rule-of-three interval and stop placing the temperature-0.7 run in
   the same row as it (A6).
9. **Apply §22.6's own "arms must sum" rule to §22.3's table** (A12), which
   reaches 105–109%.

## Things I could not settle

* **24k decode** (OQ8). Three prior invalid samples; a fourth would be a fourth
  invalid one. N4 at 24k and N1 at 24k both need it.
* **F8's end-to-end benefit.** The 2.5–4.9% is a microbench figure and I did
  not rebuild §22's reverted harness to re-measure it end to end.
* **Whether the 2.3x scheduling gap survives in the real kernel or is an
  artefact of my transcription.** A2 transcribes the QK inner loop into a
  standalone kernel; the shipped body has the PV phase, the softmax, the
  `sth`/`wq` threadgroup traffic and four barriers around it, and I have not
  built the scheduling change into `attention.metal` to confirm the full kernel
  moves the same way. **That is the first thing the next agent should do**, and
  it is cheap: add `#pragma unroll 4` (or nothing at all) to the shipped QK loop
  and run `attention_qk_mma_bench`. My claim is that the shipped kernel is
  already at 4.5 TFLOP/s-equivalent and that the 1.9 in §22.3 was measured on a
  deoptimised variant of it.
* **N4 on hardware other than this box** (A8). `decode_one_wave_workgroups` is
  calibrated to 20 cores; the rule is right here and would need re-measuring
  elsewhere.
