# K03 — Column tiling: `_2col` / `_4col` / `_co` / `_co4` / `colmajor_contig4`

**Verdict: REFUTED as a lever for the 20 tok/s goal, with one narrow unmeasured
configuration that is cheap to test and cannot plausibly matter.** Column tiling
is an addressing-and-load-amortisation lever. The measured limiter of this decode
loop is *code-decode throughput* (330–556 Gcodes/s, GB/s scaling inversely with
bit-width). Column tiling does not change either the number of codes decoded per
byte or the bytes per token, so it cannot move the number that matters.

Two secondary results are worth more than the verdict: the `_co` family is
**dead code** in the MLP dispatch tree as written, and the 4col regression that
was measured is a *different* kernel shape from the 4col variant that would
actually be worth trying.

---

## 1. The mechanism

Column-major EXL3 layout (this repo's own repack): weight element `W[k][n]` lives
in a trellis at tile `(tk = k/16, tn = n/16)`, one tile = 24 `uint32` words =
96 B = 256 codes at 3 bpw (`RING = 256*KA = 768` bits, `WORDS = RING/32 = 24`).
A *column* `tn` is `tiles_k` tiles laid out **contiguously**, so a threadgroup
that owns column `tn` walks a single contiguous `tiles_k * 96` byte run.

Per lane, per tile, in the hot loop (e.g. `linear_repacked.metal:2072`):

```
2 x uint32 weight loads  (j0, j1 -- the two-word ring window)   =  8 B
4 x half   x loads      (xr[0], xr[1], xr[8], xr[9])            =  8 B
1 x extract_v_mcg_32    -> 8 codes
8 FMA                   (4 into acc0, 4 into acc1)
```

Column tiling changes the ratio of the second line to everything else. With C
columns per threadgroup, the 4 `x` loads are issued **once and reused C times**:

| C | weight B | x B | loads/code | extracts/code | FMAs/code |
|---|---|---|---|---|---|
| 1 | 8 | 8 | 6/8 | 8/8 | 8/8 |
| 2 | 8 | 4 | 3/8 | 8/8 | 8/8 |
| 4 | 8 | 2 | 1.5/8 | 8/8 | 8/8 |

That is the whole mechanism, and the table is the entire ceiling on it:

* **The `extract_v_mcg_32` / `decode_state_mcg` count per code does not change.**
  `decode_state_mcg` (`linear.metal:121`) is 1 multiply + 1 xor/and + 1 `half2`
  add — pure ALU, no memory. Eight of those per lane per tile is ~40 ALU ops, and
  `x` loads are ~6 of ~50 instructions. So amortising `x` removes at most
  **~6% of the instruction stream** at C=4. That is the size of the effect the
  whole family can have, and it matches what was measured: 1col vs 2col ≈ 1%
  (T5.5), 2col vs 4col ≈ −4.2% (T4.2).
* **`x` is not DRAM traffic.** `x` is `K` halves = 10.75 KB at K=5376 and 43 KB
  at K=21504. It is read by every threadgroup and is L2-resident after first
  touch. Amortising it saves L1/L2 bandwidth and load-issue slots, **not a single
  byte of the 12.04 GB/token that goes to DRAM**. Bytes/token is fixed by the
  checkpoint, not by the kernel.
* **Column tiling reduces the threadgroup grid**, which is the axis that
  actually moved the numbers. Grid = `tiles_n / C`. Halving the grid halves the
  number of concurrently resident simdgroups per core.

Correction to the brief: the *3-word* `ulong` window (`w0,w1,w2`, 16+8 = 24
registers for `A0..A3`) belongs to `linear_gemv_ishizuki_k3_colmajor`
(`linear_repacked.metal:121-128`) and to the row-major `_nod`/`_w2` arms — **not**
to the colmajor column-tiled variants, which all use the two-word
`extract_v_mcg_32(wa, wb, sh0, rsh0, v)` form (`linear_repacked.metal:53`). Two
words per lane is legal at KA=3; the derivation with both failure conditions is
in the `_k4/_k6 _bench` header (`linear_repacked.metal:3290-3350`) and
`TWO_WORD_SAFE_RATES = (1,2,3,4)`.

---

## 2. Register-pressure arithmetic

Metal's public API does not expose a per-kernel register count, so an exact
occupancy number is not obtainable from a test. What is obtainable — and what the
2col/4col difference turns on — is the count of simultaneously-live 32-bit words
in the unrolled K-loop. `crates/engine/tests/probe_coltile_registers.rs`
**derives these by scanning the Metal source** (nothing transcribed), so the
table cannot rot:

```
kernel                                           unroll cols   acc   act   ptr  base  words
linear_gemv_ishizuki_k3_colmajor_contig4              4    1     2    16     0     2     30
linear_gemv_ishizuki_k3_colmajor_1col                 2    1     2     8     2     1     24
linear_gemv_ishizuki_k3_colmajor_2col                 2    2     4     8     4     2     32
linear_gemv_ishizuki_k3_colmajor_4col                 4    4     8    16    16     4     72
linear_gemv_ishizuki_k3_fused2                        4    1     2    16     0     2     22
linear_gemv_ishizuki_k3_fused2_co                     2    2     4     8     4     2     32
linear_gemv_ishizuki_k3_fused2_2col                   2    2     4     8     4     2     32
linear_gemv_ishizuki_k3_fused2_4col                   2    4     8     8     8     4     48
linear_gemv_ishizuki_k3_fused2_co4                    4    4     8    16    16     4     72
linear_gemv_ishizuki_k3_fused3                        4    1     2    16     0     2     30
linear_gemv_ishizuki_k3_fused3_2col                   2    2     4     8     4     2     32
```

`acc` = float accumulators (2 per column: the two output halves). `act` = `p00..pNN`
activation scalars live at once (4 per K sub-step). `ptr` = `const device uint *t*`
tile pointers in the loop body (1 per column × sub-step, 64-bit, counted as 2
words). `base` = loop-invariant column bases (`col*` / `column`). `v[8]` = the
single shared decode buffer (8 words, everywhere, since T3 replaced the
64-float arrays).

### Reading the table

* **`_2col` = 32 words, `_4col` = 72 words (2.25×).** Not the 2× the note guesses:
  `_4col` doubles *both* the column count (2→4) **and** the unroll
  (`step = SG*2u` → `SG*4u`, `linear_repacked.metal:2115` vs `:2742`), so
  `act` goes 8→16 **and** `ptr` goes 4→16, on top of `acc` 4→8 and `base` 2→4.
* **The accumulators are not the problem.** 8 floats is 8 words; the brief's
  "4 accumulators + 4 column pointers" framing misses that `ptr` is measured in
  *sub-steps*, and 4col unrolls twice as deep. `ptr` (32 words) is the single
  largest term in 4col, then `act` (16), then `acc` (8).
* **`_4col` is not a register-spill case at 72 words** — that fits in a 128-word
  budget comfortably. What 72 words costs is *occupancy*: on a fixed per-core
  register file, 72 words/thread admits fewer resident simdgroups than 32, and
  fewer resident simdgroups means fewer DRAM misses in flight. Note the prior
  fix: T3 already removed 72 words of `v` arrays (`spec/decode-25tok-peak-bandwidth/implementation_notes.md:70`,
  "Eliminated 56 float registers and 16 uint registers per thread"), and T4.3
  re-tuned `_2col` to ≤30 words. `_4col` was written after both fixes and is
  still 2.25× the cheapest live configuration the tree has.
* **The counter-example that isolates unroll from columns:**
  `colmajor_contig4` is **1 column at unroll 4** and costs 30 words — *less* than
  `_2col` — while `_4col` is 4 columns at unroll 4 and costs 72. So the unroll
  depth alone is nearly free (`contig4` 30 vs `_1col` 24 = 6 words for the
  doubling), and it is the *columns × unroll product* that costs. T4.2 tested the
  worst corner of that product and never tested the middle.

### The mechanism behind the measured −4.2%, restated

Two independent causes, both grid- or register-shaped, and they point the same
way:

1. **Grid.** `colmajor_4col` at N=5376 launches `5376/64 = 84` threadgroups =
   4.2/core, against `_2col_sk2`'s `5376/16 = 336` = 16.8/core. The repo's own
   resident bracket is 168–672 TGs (`perf_probe_occupancy_grid.rs:201-215`), so
   84 is **below the measured lower bound** — it is not "4.2 TGs/core is a bit
   thin", it is a grid the repo has already shown is too small three separate
   ways (SK1→SK2 worth −2.7 ms; `KAPLUMBA_GEMV_MAX_GROUPS=84` costs −0.21 tok/s;
   `_stream` at 42 TGs measured 110.8 vs 178 GB/s).
2. **Registers.** 72 vs 32 words, i.e. at 84 TGs the per-core concurrency is
   also lower.

The T4.2 note attributes it to grid alone. Both are real and they compound; the
grid is the larger term because 84 TGs is under the bracket.

### What was NOT measured

* `linear_gemv_ishizuki_k3_fused2_4col` — **4 columns at unroll 2**, 48 words,
  grid `(1344+1344)/4 = 672` TGs = 33.6/core. Already written
  (`linear_repacked.metal:580`), already reachable (`KAPLUMBA_FUSED2_4COL=1`),
  never benchmarked.
* `colmajor_4col` with `step = SG * 2u` instead of `SG * 4u` — 48 words instead
  of 72. Not written.
* A 4-column *split-K* variant for o_proj/down_proj. `encode_gemv_raw_sk2` only
  knows `_2col_sk2` and `_2col_sk4` (`linear.rs:2251-2255`, `:2301`); a
  `4col_sk2` would launch `5376/32 = 168` TGs, inside the bracket. Not written.
  **Without it, every 4-column option for o_proj/down_proj is forced onto the
  non-split-K path and therefore onto the starved 84-TG grid that T4.2 already
  measured as a loss.**

---

## 3. `_co` / `_co4`: a different axis, not a cheaper one

`_2col` and `_co` compute the same total work per threadgroup with **identical**
live state (probe assertion A1: `acc` 4/4, `act` 8/8, `ptr` 4/4, unroll 2/2).
`_co4` and `_4col` are likewise identical (assertion A2). The two families differ
only in **which** two (or four) output columns a threadgroup owns:

| | `_2col` | `_co` |
|---|---|---|
| owns | `A[tn]`, `A[tn+1]` (same matrix, adjacent tiles) | `A[tn]`, `B[tn]` (same index, two matrices) |
| buffers | 1 | 2 |
| memory per threadgroup | one **contiguous** `2*tiles_k*96` B run | two `tiles_k*96` B runs in two buffers |
| grid (gate+up) | `(1344+1344)/2 = 1344` | `tiles_a = 1344` |

So:

* **`_co` does not avoid the register cost that kills 4col — it never had a
  different one.** There is no `_co` "register-saving trick" to port into a
  single projection.
* **Could `_co`'s technique be applied within a single projection?** That is
  literally what `_2col` is: two adjacent tile-columns of one matrix fed from one
  shared `x` tile. The technique is already applied. `_co` is `_2col` with the
  second buffer pointer moved to a different tensor.
* **`_2col` is strictly better on locality for gate+up.** Columns `tn0`/`tn0+1`
  are adjacent in the packed buffer (column `tn` starts at `tn*tiles_k*WORDS`
  words, `linear_repacked.metal:2111`), so `_2col`'s two streams are one 64.5 KB
  contiguous run; `_co`'s are two 32.3 KB runs in two allocations. Same registers,
  same grid, worse DRAM behaviour. `_2col` being the default is correct.

### `_co` is unreachable today

`encode_gemv_fused2_mlp` (`crates/kernels/src/linear.rs:2582-2641`) tests the
branches in this order:

```rust
2582  if env_flag("KAPLUMBA_FUSED2_4COL") && gate%64==0 && up%64==0  -> fused2_4col
2590  else if !env_flag("KAPLUMBA_NO_FUSED2_2COL") && gate%32==0 && up%32==0 -> fused2_2col
2598  else if gate == up && !env_flag("KAPLUMBA_NO_FUSED2_CO")     -> fused2_co4 / fused2_co
2631  else                                                             -> fused2
```

21504 % 32 == 0, `KAPLUMBA_NO_FUSED2_2COL` is unset, and
`linear_gemv_ishizuki_k3_fused2_2col` exists — so line 2590 **always** wins for
gate/up. `fused2_co`, `fused2_co4`, `fused2_4col` and `fused2` are all dead for
this checkpoint unless an operator sets `KAPLUMBA_NO_FUSED2_2COL=1`. No engine
code sets it (`grep -rn "set_var" crates/` finds only two test files). This
overlaps K15; recording it here because it is a property of the column-tiling
tree specifically and because it means `_co` has effectively never been measured
in production configuration.

---

## 4. Which variant runs today, per projection

Dispatch entries (all env flags unset; `env_flag` is `is_some()`,
`linear.rs:45-58`, and nothing in the product calls `set_var`):

| projection | K × N | entry point | **kernel dispatched today** | grid (TGs) | TG/core | threadgroup mem |
|---|---|---|---|---|---|---|
| q, k, v | 5376 × 8192 / 4096 / 4096 | `encode_gemv_fused3` (`model.rs:1504`) | `linear_gemv_ishizuki_k3_fused3_2col` (`linear.rs:2781`) | 512 | 25.6 | 1 KB |
| o_proj | 8192 × 5376 | `encode_gemv_raw_sk2` (`model.rs:2281`) | `linear_gemv_ishizuki_k3_colmajor_2col_sk2` (`linear.rs:2257`) | 336 | 16.8 | 1 KB |
| gate, up | 5376 × 21504 | `encode_gemv_fused2_mlp` (`model.rs:2194`) | `linear_gemv_ishizuki_k3_fused2_2col` (`linear.rs:2592`) | 1344 | 67.2 | 1 KB |
| down_proj | 21504 × 5376 | `encode_gemv_raw_sk2` (`model.rs:2281`) | `linear_gemv_ishizuki_k3_colmajor_2col_sk2` | 336 | 16.8 | 1 KB |
| lm_head | 5376 × 262144 @ 6 bpw | `encode_gemv` → `encode_gemv_raw` (`model.rs:4420`) | `linear_gemv_ishizuki_k6` (row-major; `ishizuki_gemv_pipeline`, `linear.rs:1535-1555`) | 16384 | 819 | 512 B |

Notes on the table:

* **`lm_head` is not a column-tiling candidate at all.** It is 6 bpw and
  **row-major** (`perm_mode = 0`); only the seven per-layer linears are repacked
  to column-major (`linear_repacked.metal:3317-3320`, which also explains the head's
  253 GB/s as tile size, not tiling: 192 B/tile vs 96 B). Both `colmajor_2col`
  guards at `linear.rs:2104` and `:2133` require `perm_mode == 1`, so the head
  skips them and falls to `ishizuki_gemv_pipeline`, which skips its colmajor
  lookups for the same reason and lands on the sorted-first rate-6 kernel.
  `linear_gemv_ishizuki_k6_w2` **does not exist** (only `_k1_w2`.._k4_w2`,
  `linear.metal:8450-8813`), so it is the 3-word `_k6`, not the 2-word form.
  Column tiling cannot reach 44% of the head's bytes without a repack, and a
  repack is an R1 (bytes/token) play, not an R2 kernel play.
* **`linear_gemv_ishizuki_k3_colmajor_2col` (non-sk2) and
  `colmajor_contig4` are also unreachable** for this checkpoint: o_proj and
  down_proj take the `use_sk2` branch (which is on with *no* env flag,
  `model.rs:2242`), and the only `encode_gemv_raw` caller left is the head, which
  is `perm_mode = 0`. `colmajor_contig4` is still selected by
  `ishizuki_gemv_pipeline` if any future single colmajor projection appears.
* **The file header is stale.** `linear_repacked.metal:28-29` asserts
  "`linear_gemv_ishizuki_k3_colmajor_contig4` is the kernel the engine actually
  dispatches (`KAPLUMBA_TRACE_GEMV=1`)". It is not: no production dispatch site
  reaches it. Worth fixing, because a reader auditing the kernel family will
  start from that line and mis-rank everything else.
* Minor cross-reference for P17/K01: `perf_probe_occupancy_grid.rs:179-186`
  records the head's kernel as `linear_gemv_ishizuki_k6 |
  linear_gemv_exl3_k6_mcg`. `encode_gemv_raw` tries **ishizuki first**
  (`linear.rs:2155`) and EXL3 second (`:2178`), and `linear_gemv_ishizuki_k6`
  exists (`linear.metal:2158`), so the ishizuki arm wins. The grid (16384) is
  right either way, which is why the assertion passes.

---

## 5. Applicability by shape

Column tiling is **not** MLP-only. It applies to every colmajor 3 bpw projection
that has `out_features % 32 == 0` (and `% 64` for 4col):

| projection | tiles | 2col grid | 4col grid | note |
|---|---|---|---|---|
| q | 512 | — (in `fused3_2col`) | — | 4col has no fused3 arm at all |
| k, v | 256 each | — | — | same |
| o_proj | 336 | 336 | **84** | 84 is below the 168 lower bound |
| gate, up | 1344 each | 1344 (fused) | **672** (fused) | 672 is the upper bracket bound |
| down_proj | 336 | 336 | **84** | 84 is below the lower bound |
| lm_head | 16384 | n/a | n/a | row-major |

Every 4-column option for o_proj/down_proj lands on a grid the repo has already
measured as too small. Every 4-column option for gate+up lands on a grid that is
fine. So the only shape where a 4-column variant has a *live* chance is
**gate+up**, which is also 5.202 of the 10.98 GB/token layer budget (47%) — the
right target, for once, and still not enough to matter (see below).

Note also that down_proj is the **longest-K** projection (K=21504, 2.601 GB/token,
40% of gate+up's bytes) and the one whose T5.2 split-K work bought the largest
single win in that spec (−2.7 ms). More columns is the *opposite* axis to split-K;
on this shape they trade against each other for the same resident threadgroups.
For down_proj, split-K already won; column tiling is not the missing half.

---

## 6. Is there an untested shape where more columns would win?

Three candidates, all unmeasured:

1. **`fused2_4col` (gate+up), unroll 2, 48 words, grid 672.** Already written
   and reachable today with `KAPLUMBA_FUSED2_4COL=1`. Zero new kernel. This is
   the cheapest possible test of the family and it is a one-env-flip A/B.
2. **`fused2_co4` (gate+up), unroll 4, 72 words, grid 672.** Same grid as (1),
   the register corner T4.2 already found expensive, but with the grid fixed.
   Together (1) and (2) separate "4 columns" from "unroll 4" on the one shape
   where the grid is not the problem. Reachable with
   `KAPLUMBA_NO_FUSED2_2COL=1 KAPLUMBA_FUSED2_CO4=1`.
3. **`colmajor_4col` at unroll 2** (`step = SG * 2u`), 48 words, grid 84 for
   down/o_proj. **Predicted to lose** — the grid is the dominant term and it is
   unchanged — so this is listed only to close the shape off, not to try it.

And the answer to the brief's literal question — "2col using the `_co`
register-saving trick on a single wide-N projection" — is **there is no such
trick**. `_co` and `_2col` are the same loop; see §3.

---

## 7. Conclusion

* Column tiling amortises `x` loads and address arithmetic over more output
  columns. It removes at most ~6% of the instruction stream and **0 bytes** of the
  12.04 GB/token DRAM stream. The loop is code-throughput bound at 330–556
  Gcodes/s; `x` amortization cannot raise Gcodes/s.
* Every 4-column configuration for o_proj/down_proj is forced onto the 84-TG
  grid, below the repo's own 168-TG resident lower bound. The measured −4.2% is
  therefore expected, and the arithmetic says it should not be retried on those
  shapes without a split-K 4-column kernel that does not exist.
* `_2col` is the right default everywhere it runs: it is the cheapest
  2-column form (32 words), it launches the same grid as the alternatives
  (`_co`), and it has strictly better DRAM contiguity (`_co`).
* **2-column tiling is at its local optimum.** Going to 1 column costs 1%
  (T5.5, measured); going to 4 columns on a split-K-less path costs 4.2%
  (T4.2, measured). The only untested point on the curve is 4 columns at unroll
  2 on gate+up, and the whole plausible range of that measurement is inside the
  35% same-config noise floor on this box.

---

## 8. Recommended change

**One change, no new kernel: measure `linear_gemv_ishizuki_k3_fused2_4col` on
gate+up.** It is already written at `crates/kernels/src/linear_repacked.metal:580`
and already dispatched at `crates/kernels/src/linear.rs:2582-2589`; today it is
reachable only via `KAPLUMBA_FUSED2_4COL=1`, which is opt-in and off.

* **Expected effect: 0.0 tok/s ± 0.3, i.e. inside the noise floor.** gate+up is
  40.3 ms of the 62.3 ms step (5.202 GB/token at 129 GB/s,
  `spec/decode-25tok-peak-bandwidth/implementation_notes.md:~495`). A ±5% swing
  on that kernel is ±2.0 ms, i.e. 16.05 → 15.85 or 16.58 tok/s. Given
  AGENTS.md §10.5 records 3.7–6.4 tok/s spread from the *same binary* minutes
  apart, this experiment needs N≥8 interleaved A B A B… pairs to resolve, and it
  will probably not resolve.
* **It cannot move the goal.** 18 tok/s needs 217 GB/s against a 203 GB/s
  measured best, and 20 needs 236.7 GB/s. Column tiling changes neither
  bytes/token nor Gcodes/s, so it cannot reach either. If it *does* win, it wins
  by fractions of a percent and the ceiling is elsewhere (R1 bytes/token, or the
  EXL3 per-code instruction cost in P01).
* **What to record if it is run:** whether `fused2_4col` at 672 TGs / 48 words
  beats `fused2_2col` at 1344 TGs / 32 words. That single data point separates
  "more columns helps when the grid is over-subscribed" from "more columns never
  helps", which no current measurement distinguishes. If it loses, K03 is closed
  and no further column-tiling work is warranted.

Secondary, zero-risk, do alongside: correct the stale header at
`linear_repacked.metal:28-29` (it names `colmajor_contig4` as the live kernel),
and note at `linear.rs:2598` that the `_co`/`_co4` branch is unreachable while
`KAPLUMBA_NO_FUSED2_2COL` is unset — so the next reader does not re-derive it.

---

## 9. Confidence

| claim | confidence | why |
|---|---|---|
| Column tiling amortises `x` loads only, and `x` is not DRAM traffic | **high** | read directly from the kernels (`linear_repacked.metal:2072`, `:2691`); `x` is `K` halves, 10.75–43 KB, shared across all threadgroups |
| `_2col`=32 words, `_4col`=72 words, `_co`≡`_2col`, `_co4`≡`_4col` | **high** | asserted by a test that derives the counts from the Metal source, not transcribed; `cargo test --test probe_coltile_registers` passes 2/2 |
| `_co`/`_co4` unreachable for gate+up without `KAPLUMBA_NO_FUSED2_2COL` | **high** | branch order at `linear.rs:2578-2637` read directly; no `set_var` anywhere in the product |
| Dispatch table in §4 | **high** | each row traced from the `model.rs` call site to the `linear.rs` selection; grids cross-checked against the pinned assertions in `perf_probe_occupancy_grid.rs:250-256` |
| lm_head is row-major 6 bpw and therefore out of scope | **high** | stated in `linear_repacked.metal:3317-3320` and consistent with `_w2` kernels existing only for rates 1–4 |
| 4col's −4.2% is grid-dominated rather than register-dominated | **medium** | the grid (84 TGs) is below a bracket the repo measured three independent ways; the register term (72 vs 32 words) is real but secondary. Separating them needs the unroll-2 4col kernel, which does not exist |
| Expected effect of `fused2_4col` is inside the noise floor | **medium-high** | derived from the measured gate+up share of the step and the documented noise floor; not measured |
| The `linear_repacked.metal:28-29` header is stale | **high** | no dispatch site reaches `colmajor_contig4` for this checkpoint; see §4 |
| The `perf_probe_occupancy_grid` lm_head kernel attribution is wrong (ishizuki, not exl3) | **medium** | `encode_gemv_raw` tries ishizuki first (`linear.rs:2155`) and `linear_gemv_ishizuki_k6` exists; but the census only asserts the grid, which is identical either way, so this is a documentation nit, not a behaviour claim |

**Not established, and not claimed:** any absolute register count; whether 72
words causes spilling (it should not, on a 128-word budget); whether 4col would
win on a shape with a 4-column *and* split-K kernel, which does not exist.

---

## Appendix — files read

* `crates/kernels/src/linear_repacked.metal` — `colmajor` 79, `_simd` 197,
  `fused2` 320, `fused2_2col` 434, `fused2_4col` 580, `fused2_co` 769,
  `fused2_co4` 906, `fused3` 1283, `fused3_2col` 1410, `contig4` 1970,
  `colmajor_2col` 2072, `colmajor_1col` 2208, `colmajor_2col_sk2` 2407,
  `colmajor_2col_sk4` 2549, `colmajor_4col` 2691; helper `extract_v_mcg_32` 53
* `crates/kernels/src/linear.rs` — `env_flag` 45, `decode_nr0` 75,
  `gemv_max_groups` 94, `ishizuki_gemv_pipeline` 1504, `encode_ishizuki_gemv`
  1452, `encode_gemv` 1956, `encode_gemv_raw` 2081, `encode_gemv_raw_sk2` 2225,
  `encode_gemv_raw_sk4` 2283, `encode_gemv_fused2` 2326,
  `encode_gemv_fused2_mlp` 2488, `encode_gemv_fused3` 2685
* `crates/kernels/src/linear.metal` — `decode_state_mcg` 121, `exl3_decode8_mcg` 345,
  `_k6` 2158, `_w2` family 8450-8813
* `crates/engine/src/model.rs` — qkv fused3 1504, qk fused2 1566, o_proj raw 2016 /
  sk2 2281-2340, mlp fused2_mlp 2194, down sk2/raw 2281-2360, head 4420
* `crates/engine/tests/perf_probe_occupancy_grid.rs` — census 131-199,
  resident bracket 201-215, pinned grids 250-256
* `crates/engine/tests/probe_coltile_registers.rs` — **new**, this hypothesis
* `spec/decode-25tok-peak-bandwidth/{tasks,implementation_notes}.md` — T3 (:70),
  T4.2 (:28-31, notes :114-140), T4.3 (:33-36, notes :141-155), T5.2 (:55-58),
  T5.5 (:70-74); §16.3/16.4 grids and GB/s
* `spec/decode-20tps/parallel_track_a_notes.md` A.2 (quoted in the census header)
