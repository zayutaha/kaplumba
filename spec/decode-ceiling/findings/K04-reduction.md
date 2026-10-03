# K04 — the cross-lane reduction, and whether it can be cheaper

**Verdict: the reduction is not the lever, and the two 1-threadgroup finalizes
are not worth attacking either. Nothing in this angle is worth more than
+0.24 tok/s, and the largest single item is worth +0.10 tok/s.**

Everything below is derived from source arithmetic and CPU-only proofs. No GPU,
no model, no `gpuq`. Probe: `crates/engine/tests/probe_reduction_k04.rs`
(11 tests, all green; every claim sabotaged and observed to fail — §6).

---

## 0. Two corrections to the brief before the numbers

1. **The live kernel is `linear_gemv_ishizuki_k3_colmajor_2col_sk2`, at
   `crates/kernels/src/linear_repacked.metal:2407-2547`, not 2072-2206.**
   2072-2206 is `..._2col` (no split-K). They have the same reduction block, but
   `_2col_sk2` halves the per-threadgroup K loop and doubles the grid, so its
   reduction is *twice as expensive in relative terms*. Every percentage below is
   for the kernel that actually runs.
2. **The two finalizes execute 23 and 22 barriers, not "10 and 18".** The 10 and
   18 are the reduction-tree loops only; both kernels also carry standalone
   barriers that the brief's count omits (§5.1).

---

## 1. Exact count of the live GEMV's reduction

`linear_repacked.metal:2518-2545`. Runs **once per threadgroup**: the dispatch
grid is `pairs_n * SK` (= `N/16`), which equals the kernel's `total_groups`
(`:2438`), so the `for (tg = tgid; tg < total_groups; tg += ntg)` loop body
executes exactly one iteration. Proved for all 10 real shapes in
`reduction_block_runs_exactly_once_per_threadgroup`.

Counted *as a lockstep GPU pays*, i.e. a divergent arm charged to the whole warp:

| item | slots | source |
|---|---|---|
| `simd_shuffle_xor` | 8 | `:2518-2526` |
| the `fadd` paired with each shuffle | 8 | same 8 lines |
| `if (lane % 4u == 0u)` predicate | 1 | `:2528` |
| threadgroup stores (`red[0]`×2, `red[1]`×2) | 4 | `:2529-2532`, 4 of 32 lanes |
| `threadgroup_barrier` #1 | 1 | `:2534` |
| final sum, `lid < 32u`: 8×(tg load + fadd) + selects + device store | 21 | `:2535-2543` |
| `threadgroup_barrier` #2 | 1 | `:2545` |
| **per warp** | **44** (warp 0) / 25 (warps 1-7) | |
| **mean per warp** | **27.4** | |

Threadgroup memory: `red[2][SG][16]` = 2·8·16·4 = **1024 B** — 16× under the
~16 KB occupancy cliff, and 32× under the 32 KiB limit.

### Against the K loop

Structural count of one unrolled trip: 4 × `extract_v_mcg_32` (52 slots each =
20 field extractions + 8×4 decode ops; the 2 shifts and the `or` hoist out,
`sh0` is loop-invariant), 8 `uint` weight loads, 8 `half` x-loads + 8
conversions, 4 calls × 2 accumulators × 8 mul/add, 6 address IMADs, 3 loop ops
= **~300 slots per trip for 32 weights = 9.38 instr/weight**, which reconciles
with the established 9.375 figure (asserted, tolerance 1.0).

Trips per threadgroup at `SK = 2`, `SG = 8`, `step = SG*2 = 16`:

| K | tiles_k | per split | unrolled trips | remainder | K-loop slots |
|---|---|---|---|---|---|
| 5376 (q/k/v/o/gate/up) | 336 | 168 | 10 | 1 | **3070** |
| 21504 (down) | 1344 | 672 | 42 | 0 | **12600** |

| shape | reduction / K loop |
|---|---|
| K = 5376 | 27.4 / 3070 = **0.854%** |
| K = 21504 | 27.4 / 12600 = **0.214%** |

### Per token, all 60 layers (10 of them `Full`, `config.rs:241`)

420 GEMV dispatches. Weighted: **0.664% of GEMV issue slots.**

**Is reduction >5% of kernel time? No.** Two independent routes:

* *Issue-slot route*: 0.664% of GEMV time. GEMV is 12.039 GB / 199.4 GB/s =
  60.4 ms, so the reduction is **0.40 ms/token**, and **deleting it entirely**
  buys 62.3 → 61.9 ms = **16.154 tok/s (+0.104)**. That is an unreachable
  ceiling — you cannot delete the reduction, only move it.
* *Measured-barrier route*: the tree already records the whole linear family's
  barrier budget at **0.30% of a token** (`ROUND1-2-SUMMARY.md:207`), i.e.
  0.19 ms for ~840 barriers = **0.226 µs per barrier ≈ 543 cycles**. That is
  ~6× a barrier's issue-slot cost, which is the expected barrier-rendezvous
  latency for an 8-warp group — so the reduction's cost is latency-dominated,
  not issue-dominated. Charging the 2 barriers at the measured rate and the
  remaining 25.4 slots + ~120 cycles of shuffle/threadgroup-load latency:
  **≈ 0.51 µs per GEMV → 0.21 ms/token = 0.34% of a step → +0.05 tok/s if
  deleted.**

Bracket: **the reduction is 0.34–0.66% of a token, worth ≤ +0.10 tok/s.**
Both methods are far under 5%, and both are under the ±0.1 tok/s noise floor
recorded in `decode-25tok-peak-bandwidth/implementation_notes.md` §16.1
(two arms of the same binary differ by 0.010; range across arms 0.076).

---

## 2. Is the MLP worse off? No — and this is the cleanest part of the answer

The reduction cost **per threadgroup** is a constant (27.4 slots). Total
reduction slots per dispatch scale with N (`N/16` groups × 8 warps × 27.4). So:

* **Absolute**: gate/up (N = 21504) do exactly **4.0×** o_proj's (N = 5376)
  reduction work. Asserted.
* **Fraction**: identical, because they also do exactly 4× the K-loop work at
  the same K. `reduction_fraction_is_independent_of_n` asserts
  `frac(21504) == frac(5376)` to 1e-12.

The fraction is a function of **K alone**. The only place it improves is
`down_proj` (K = 21504, 0.214%) and full-attention `o_proj` (K = 16384, 0.285%)
— both because their K loop is longer, which *makes the reduction look smaller*.
**MLP projections are not a special case.** The premise of question 2 is that
N-scaling makes the MLP reduction-heavy; it does not.

---

## 3. The four alternatives — all strictly worse

**(a) More output columns per thread.** Already tried: `..._2col_4col` exists at
`:2691` and `4col` is on the provably-dead list (`WHERE-THIS-STANDS.md:22-25`,
measured). Arithmetic agrees: 4col halves the threadgroup count and doubles the
K loop per group, so the reduction *fraction* is unchanged. Already 32 outputs
per 256-thread group — 8 threads per output. Nothing to win.

**(b) Wider shuffle tree (8 or 16 lanes).** Currently `lane % 4 == 0` writes, so
each output has **8 partials** (one per simdgroup) and each simdgroup has 8
active lanes of 32. Widening to `lane % 16 == 0` gives 2 partials per simdgroup
but **16 per output**: +8 shuffle/add slots per group (`red` doubles to
16 slots per output, and the `lid < 32` final sum grows from 8 to 16 loads),
while the store arm goes from 4 active lanes to 2. Reduction 27.4 → ~43
slots/warp, **0.854% → ~1.34%. Worse.**

**(c) Eliminate threadgroup memory in the epilogue.** Replacing the 8 threadgroup
loads + 7 adds with an in-simdgroup finish needs 5 `simd_shuffle_xor` + 5 adds
per output × 32 outputs, i.e. 320 slots on one warp, against the current 21 × 8
= 168 across 8 warps. **Worse by ~2.4×.** The threadgroup round trip is the
cheap half of the reduction; the shuffles are the expensive half, which is
exactly why the code stops at `lane % 4`.

**(d) `simd_prefix_exclusive_sum` / matrix intrinsics.** A prefix sum emits all
32 partial sums to compute 1 total — 32× the adds. A matrix intrinsic
(`simd_sum` across a 4×8 matrix) does not exist in Metal's shuffle set and would
not be bit-exact anyway. **Strictly worse.**

**The current geometry is already the cheap one.** This is a clean negative, not
a gap in the search.

---

## 4. Is reassociating ever safe here? No — and the frozen order is now fully
## characterised

Not safe, and I can now say exactly *why* the "obvious" parallelisations fail.

The frozen order (`linear.metal:937-948`) is: 256 leaves, each a 21-long
sequential chain `sum += x[lid + 256k]`, then the **descending butterfly on
masks 512, 256, 128, …, 1** over `partial[1024]`. Masks 512 and 256 add exact
zeros from the 768 idle threads, so it is bit-identical to the 256-thread
original — the property the kernel header relies on. The effective tree is the
**butterfly on masks 128…1 over 256 leaves**.

In a butterfly, mask `2^j` maps `2^(j+1)` nodes to `2^j` nodes and node `r` then
covers leaves `{r + k·2^j}`. That single invariant decides everything:

* **Contiguous ranges are not subtrees.** `T(lo..hi)` over a contiguous leaf
  range never exists in this tree. My first attempt grouped leaves as
  `[g·per, (g+1)·per)` and was **off by 1 ULP** at G = 32, 64 and 128 (and by
  2 ULP at G = 2, 4, 8 on a second input).
* **Strided leaves with ascending recombination is not it either.** My second
  attempt gave group `g` the leaves `≡ g (mod G)` and recombined under masks
  `128/G, 128/2G, …` — the *ascending* order. Also **off by 1 ULP**.

### The correct, bit-exact multi-group split — derived, not guessed

The node-count invariant says masks 128/64/32 leave exactly **32** nodes, index
`0..31`, node `r` covering leaves `{r, r+32, …, r+224}`. Masks 16…1 then reduce
those 32 to 1, and all five are *stage-wide* (mask 16 pairs node `r` with
`r+16` for `r < 16`). So:

> **`G` threadgroups, `G | 32`, each owning `32/G` of those 32 whole subtrees,
> computing each with the local masks 128/64/32 folded in plus 16/8/4/2/1; then a
> second, trivial pass applies the five stage-wide masks to 32 values.**

Verified bit-exact for **G = 1, 2, 4, 8, 16, 32** in
`power_of_two_group_splits_are_bit_exact_and_42_blocks_is_not`.

**This is the answer to question 4, and it is "no" for the planned shape.**
`42 ∤ 32` (and `42 ∤ 256`). The 42-group plan in `tasks.md` T5.6 and
`ROUND1-2-SUMMARY.md:107` — one group per 128-element block — is precisely the
contiguous-range partition, which is a reassociation: it changes
**5376/5376** normed elements, re-derived here independently of the existing
`perf_probe_fusion_census.rs:1266` result.

**So a bit-exact parallel finalize is possible, but only at G ∈ {2,4,8,16,32},
and it has a cost the plan does not account for:** group `g` owns elements with
`i mod 32 ∈ [g·(32/G), (g+1)·(32/G))`. With a 128 B cache line = 32 f32, that is
**`G`× L1-line amplification** — at G = 32 each group fetches 21.5 KB of lines
to extract 672 B of useful squares. G = 8 is the sane operating point (8×, 8
cores). This is not in any spec and it is the real reason the T5.6 rewrite is
risky, not the bit-exactness.

### One bit-exact win that IS available: the tree is a butterfly, so 8 barriers
### are really 3

Masks 128/64/32 cross simdgroup boundaries (node `r` at depth 7 pairs with
`r + 128`, in simdgroups 0-3 and 4-7). Masks 16/8/4/2/1 are **entirely inside
simdgroup 0**, lane `= r`. So:

> **10 barriers → 3 `threadgroup_barrier` + 5 `simd_shuffle_xor`.**

Proved bit-identical in `the_norm_tree_is_a_butterfly_so_three_barriers_and_five_
shuffles_are_exact` (the descending mask order is load-bearing — the ascending
sabotage differs, §6). Applies to `linear.metal:945-948`, `:962-965` and
`:1142-1147`. Saves **14 barriers per token-pair of dispatches**, i.e. 7 × 120 =
840 barriers/token. At the measured 0.226 µs/barrier that is **0.19 ms** →
**+0.05 tok/s**.

---

## 5. The two 1-threadgroup finalizes

### 5.1 Exact inventory

`linear_finalize_o_proj_and_pre_ffn_transform_sk2` (`linear.metal:902-981`,
dispatched `crates/kernels/src/linear.rs:3065` with `dispatch_groups(..., 1, 1024)`):

| | count |
|---|---|
| threadgroup memory | `partial[1024]` 4096 + `shared_buf[5376]` 10752 = **14,848 B** |
| barrier call sites (static) | 5 (`:933`, `:944`, `:947`, `:961`, `:964`) |
| **runtime barrier executions** | 3 standalone + 2 × 10 tree = **23** |

`finalize_post_ffn_and_next_input_norm_transform` (`linear.metal:1097-1208`,
dispatched `linear.rs:3197`):

| | count |
|---|---|
| threadgroup memory | `shared_buf` 10752 + `input_shared` 10752 + `partial[1024]` 4096 = **25,600 B** — **over the ~16 KB occupancy cliff** |
| barrier call sites (static) | 6 (`:1129`, `:1141`, `:1146`, `:1161`, `:1174`, `:1179`) |
| **runtime barrier executions** | 4 standalone + 10 + 8 tree = **22** |

The brief's "10 + 18 serial barriers" = the tree loops only. Both numbers are
asserted in `finalize_threadgroup_memory_and_barrier_counts`.

### 5.2 A real defect found: the cross-layer kernel's last step is 3× serialised

`linear.metal:1187` — `for (uint b = sg; b < params.blocks; b += 8u)`. With 1024
threads there are **32** simdgroups, but the stride is 8, so **24 of 32 simdgroups
execute nothing**, and with `blocks = 5376/128 = 42` the loop runs **6 rounds**
where the sibling kernel's `b += 32u` (`:921`, `:969`) runs **2**.

This is a missed edit in T5.6: `implementation_notes.md` §17 records "Changed
block loop stride from `b += 8u` to `b += 32u`" for the four finalize kernels.
The cross-layer kernel's **first** loop was bumped (`:1117`) but its **last** one
(`:1187`) was not, because it was carried over verbatim from the 256-thread
`input_norm_and_qkv_transform` (`:1238`), where `8u` was correct. The 1024-thread
widen at `linear.rs:3197` therefore left it 3× under-parallelised.

Utilisation: 42 blocks / (8 groups × 6 rounds) = 87.5% of 8 groups = **21.9% of
the threadgroup**, versus 65.6% for the sibling. Asserted in
`cross_layer_last_step_under_parallelises_its_own_sibling`.

### 5.3 What is the prize? Smaller than the tree thinks.

These two kernels are the only ones pinned to 1 of 20 cores, and they are
**device-traffic bound on that one core**:

| kernel | reads | writes | total |
|---|---|---|---|
| `..._o_proj..._sk2` | `raw` 2N f32 = 43,008 B + 5 half buffers × 10,752 = 53,760 B | `hidden_b`, `transformed` = 21,504 B | **118,272 B** |
| `finalize_post_ffn...` | `raw` 43,008 + 5 × 10,752 = 53,760 | `output`, `normed`, `transformed` = 32,256 B | **129,024 B** |

Per token: 60 × 118,272 + 60 × 129,024 = **14.8 MB, all through one core.**
A single Apple GPU core sustains roughly 10–25% of a 20-core peak on dependent
loads — 20 to 50 GB/s — so **0.30 to 0.74 ms/token**, midpoint ~0.5 ms. That
brackets the tree's unmeasured "0.4–0.9 ms/step" estimate. Spread across 20
cores it would be 14.8 MB / 200 GB/s = **0.074 ms**.

**Upper bound, assuming free parallelisation and zero overhead:**

| | ms saved | tok/s |
|---|---|---|
| tree's optimistic 0.9 ms | 0.90 | 16.05 → **16.29 (+0.24)** |
| my midpoint 0.50 ms | 0.50 | 16.05 → **16.18 (+0.13)** |
| after paying 240 extra dispatches/token (0.7 µs each = 0.17 ms) and the G× L1 amplification | ~0.35 | 16.05 → **16.14 (+0.09)** |

**The whole T5.6 prize is ≤ +0.24 tok/s, realistically +0.09, and the work is a
two-pass rewrite of the two most bit-exactness-critical kernels in the engine.**
That is not a good trade, and the "only fusion item with a ms-scale number"
framing in `ROUND1-2-SUMMARY.md:108` is what makes it look like one.

**The measurement is already wired and has never been run.** `linear.rs:3016` and
`linear.rs:3181` gate these exact two dispatches behind
`crate::ablated("finalize_o_proj")` / `crate::ablated("finalize_post_ffn")`. One
interleaved A/B, `KAPLUMBA_ABLATE=finalize_o_proj,finalize_post_ffn` against
baseline, settles the ≤ +0.24 question in about ten minutes and costs nothing.
Do that before writing a single line of the two-pass kernel.

---

## 6. Sabotage log (AGENTS.md §3.4)

Every claim was broken on purpose and the failure observed. All reverted;
`git diff crates/kernels/src/` is empty.

| # | sabotage | observed failure |
|---|---|---|
| S1 | add a 9th `simd_shuffle_xor` to `..._2col_sk2`'s reduction tail | `reduction_is_under_five_percent_of_the_live_gemv`, `..._fraction_is_independent_of_n`, `..._under_one_percent_of_gemv_time_per_token` all FAILED: `expected 8 simd_shuffle_xor ... left: 9, right: 8` |
| S2 | `linear.metal:1187` `b += 8u` → `b += 32u` | `cross_layer_last_step_under_parallelises_its_own_sibling` FAILED: `first loop not bumped, left: 2, right: 1` |
| S3 | drop one tree level: `linear_finalize_o_proj_and_pre_ffn_transform_sk2`'s first `stride = 512u` → `256u` | `finalize_threadgroup_memory_and_barrier_counts` FAILED. **First attempt it PASSED** — I had computed `tree_loops` without asserting it. Tightened to assert the loop-start literals and the derived runtime count; re-sabotaged, now FAILED. |
| S4 | ascending shuffle mask order in the butterfly | `a_wrong_shuffle_mask_breaks_the_butterfly_proof` FAILED at 1 ULP difference — **only after §7**; see below |
| S5 | drop one of the 32 subtrees from the G = 8 split | `a_misaligned_group_boundary_breaks_the_split_proof` FAILED — confirms whole-subtree ownership is load-bearing |
| S6 | forward vs. reversed sum of the 5376 squares | `the_test_input_discriminates_summation_orders` guards the whole file: with the first input I built (f16 **denormals**, `0x300 | mant` instead of `0x3000 | mant`) S4 did **not** fire — the input could not discriminate tree shapes at all, and every "exact" assertion would have been vacuous. Fixed, guard added |

**Two real errors of mine, both caught by running the code rather than reasoning
about it:** the strided/ascending partition (S4) and the degenerate test input
(S6). Both would have been reported as findings had I not executed them.

---

## 7. Recommendation

**Do not touch the reduction.** 0.34–0.66% of a token, ≤ +0.10 tok/s if deleted
outright, with all four cheaper-alternatives measured worse in §3.

**The one concrete change worth making, and it is a defect not an optimisation:**

```diff
--- a/crates/kernels/src/linear.metal:1187
+++ b/crates/kernels/src/linear.metal:1187
     // ---- Layer L+1: input norm + q input-sign transform -------------------
-    for (uint b = sg; b < params.blocks; b += 8u) {
+    for (uint b = sg; b < params.blocks; b += 32u) {
         const uint base = b * 128u + lane;
```

`b += 32u` matches its two siblings (`:1117` in the same kernel, `:921` and
`:969` in `linear_finalize_o_proj_and_pre_ffn_transform_sk2`). **Bit-exact by
construction**: the loop body is a pure elementwise map over `blocks` of 128
elements, each block writing its own disjoint `normed[base..]` and
`transformed[base..]` slots, with no cross-block reduction and no accumulation
across iterations — so the block→simdgroup assignment cannot change a value.
3 rounds saved out of 6 on one of two hot per-layer dispatches.

**Expected: ≤ +0.01 tok/s. I am not claiming more, because 0.01 is at the noise
floor.** Its value is that it is a one-token change to a documented mistake in a
kernel the spec calls out as misfiled, and it costs one line to verify against
`fused_bit_exact.rs`.

**And the one action that actually matters, which is not code:** run
`KAPLUMBA_ABLATE=finalize_o_proj,finalize_post_ffn` against baseline, interleaved
A/B, before anyone budgets the two-pass rewrite. `linear.rs:3016` and `:3181`
already provide it. The arithmetic above says the answer is between +0.00 and
+0.24 tok/s, and no amount of further reading narrows that — only the ablation
does. If it comes back at +0.2, then and only then is the butterfly restructure
of §4 (14 barriers/token, bit-exact, proved) worth folding into the same change.

---

## 8. What would change my answer

* A measured barrier cost much above 0.226 µs (my number is *derived* from the
  0.30%-of-a-token figure in `ROUND1-2-SUMMARY.md:207`, not measured
  directly). At 1 µs/barrier the reduction would be ~1.5% and the butterfly
  restructure ~+0.2 tok/s.
* `extract_v_mcg_32` compiling to more than 52 slots. The 9.375 instr/weight
  cross-check in `reduction_is_under_five_percent_of_the_live_gemv` has a
  tolerance of 1.0; if a Metal compiler change pushes the structural count to
  70+, the reduction's fraction roughly doubles and the analysis needs redoing.
  The test will catch it.
* An occupancy regime where only 1 threadgroup fits per core, which would expose
  the reduction tail serially instead of overlapping it with other groups' K
  loops. At 256 threads and 1 KB of threadgroup memory this does not happen on
  M4 Pro, but it is the assumption the 0.34% lower bound rests on, and it is not
  measured here.