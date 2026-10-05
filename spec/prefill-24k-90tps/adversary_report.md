# Adversary report — `spec/prefill-24k-90tps`, branch `mtp-speculation` @ `f2b2da1`

**Adversary:** a fresh session, worktree `/…/opencode/adv-prefill`, branch
`adversary/prefill-24k`, own `CARGO_TARGET_DIR`s (`adv-target` for the shipped
build, `adv-sab-target` for every sabotage). The main tree was not touched.
Base for the "pre-existing" checks: a second detached worktree at `b92234f` with
its own target dir.

**Verdict summary**

| # | claim | verdict |
|---|---|---|
| **F1** | OQ7: "deleting the QK and PV instructions does not speed the flash kernel up" | **BROKEN, and §10's ablation never reached the kernel.** An empty body is **0.19 ms** against **920 ms** for the real one, with `scalar`/`mma` in the same binary as controls |
| **F2** | The shipped `PREFILL_PASS_MIN_GROUPS = 512` grid gate | **BROKEN — dead code.** `prefill_pass_count()` has no production caller; the shipped path slices a **20-threadgroup** grid |
| **F3** | "N1 kept, **default unchanged**" | **BROKEN — the default changed.** Passes ship **ON** at `key_count > 8192`; the notes say both "the default must stay one dispatch" and "the gate ships on by default" |
| **F4** | The `workgroups` default flip is quality-neutral, with "five reference prompts byte-identical between arms" | **BROKEN as evidence.** At ~120 tokens of context both rules compute `splits = 2`; the check cannot fail. The shipped source itself says the rule "changes **43 of 128**" greedy tokens |
| **F5** | AC2 arithmetic: attention 149.7 s → ≤ 56 s, a 2.7× cut at 22 % of tensor peak | **BROKEN by ~1.7×.** Measured: 88 % of the 90 tok/s budget is spent before attention starts; the real ask is **3.8×** at **46 % of peak** (62 % if only the flash kernel moves) |
| **F6** | "A third independent reproduction" of the one-wave rule | **BROKEN as stated** — it re-measures the row the rule was *derived* from; two of the three "shapes" are not in the model |
| **F7** | "PRE-EXISTING crash, reproducible every time" | **NOT REPRODUCIBLE** in 19 attempts across 9 token counts |
| **F8** | N3(a)'s "sabotage check" | **PARTLY BROKEN.** The sabotage *does* reproduce and the suite *does* fail (7/8, 0.2136 = 43× exactly as the file says) — but the cited sibling test `a_misfiring_guard_is_caught` does not exist, so the control is not in the tree |
| **F9** | `f2b2da1`: "bench_decode_splits.sh now honours `LOG=`" | **HALF-DONE.** `bench_prefill_passes.sh` and `scorecard.sh` still hard-code `$OUTDIR/summary.tsv`. It bit me: `LOG=$OUT/pass_16k.tsv` produced an empty file and the data was in `summary.tsv` |
| **N1** | prefill passes: bit-exact, measured neutral at 16k | **SURVIVED** (independently re-measured), with F2/F3 attached |
| **N3(a)** | the `msv == 1` rescale guard, exact, with a test that can fail | **SURVIVED** — inverting the guard fails 7/8 at 0.2136 (43×); the only defect is that the control is not in the tree (F8) |
| **N3(b)** | `HS == 2` inapplicable | **SURVIVED** on the arithmetic; under-analysed for prefill `BQ = 2` |
| **N4** | split optimum is 13, worth +2 % | **SURVIVED** — my own interleaved curve is monotone to 13 and cliffs 1.5 % at 14 |
| **N4** | "provably a no-op for 50 of 60 layers" | **SURVIVED** (verified), but for a reason nobody wrote down |
| **N5** | the memory gate cannot discriminate | **SURVIVED and strengthened** — I have the same-context pair they lacked |
| **N6** | 4 failures pre-existing at base | **SURVIVED** — same names, same counts at `b92234f` |

---

## F1 — OQ7 is not open, and §10's ablation never ran

**Sabotage (1 line, `crates/kernels/src/attention.metal:1476`):**

```metal
    // ADV-OQ7-EMPTY: sabotage. Immediate return at the top of the body, i.e. a
    // genuinely empty kernel at the real grid.
    if (true) { return; }
```

Built into `adv-sab-target`, run as `attention_qk_mma_bench --nocapture`.
The `scalar` and `mma` arms in the **same binary** are the positive control, and
the clean build was run in the same session.

`attention_arm_throughput_at_prefill_geometry`, full 32q/4kv/d512, median ms:

| case | clean build | immediate-return build | ratio |
|---|---|---|---|
| flash bq1, 24 576 keys, 1024 q | **920.38** [902.29 922.20] | **0.19** [0.19 1.04] | 4 800× |
| flash bq2, 24 576 keys, 1024 q | 857.50 [857.11 862.80] | 0.17 [0.17 0.19] | 5 000× |
| scalar, 24 576 keys, 1024 q | 1 234.67 | 1 227.05 | 1.006× |
| mma, 24 576 keys, 1024 q | 889.84 | 878.53 | 1.013× |
| flash bq1, 24 576 keys, 4096 q | 3 609.20 | 0.19 | — |
| scalar, 24 576 keys, 4096 q | 4 977.61 | 4 936.80 | 0.992× |
| mma, 24 576 keys, 4096 q | 3 586.46 | 3 541.13 | 0.987× |

The clean build reproduces §10 (baseline 958 → my 920; §10.1's 256/1024/4096 →
237/947/3825 → my 231/920/3609). So the comparison is like-for-like on a build
that reproduces the numbers being attacked.

**Conclusion.** The flash kernel's ~920 ms at 24 576 keys is **real work in its
body**: an empty body at the same grid, same buffers, same binary costs 0.19 ms.
Therefore §10's rows

| variant | §10 |
|---|---|
| `ksteps = 0` (no PV) | 966.88 |
| QK loop disabled | 959.90 |
| both off | 941.88 |
| `if (tid < 256u) { return; }` | **964.79** |

did not measure the kernel they claim to have ablated. That is precisely the
hazard §10.1 lists ("`pipeline_audit::build(...).ok()` turns a Metal compile error
into `None`, and the dispatch then falls back … silently") and precisely the
failure class AGENTS.md §3.4 records twice as shipped.

Consequences:

1. **OQ7 is closed, not open.** "Removing the QK and PV instructions moves
   nothing" is an artefact. There is no hardware mystery to explain, and §10.3's
   "the control nobody ran — does an empty kernel take zero time?" now has an
   answer: **0.19 ms**, and it was the one control that was not run.
2. **§10.2 #3 is broken as written.** "The microbench is a valid throughput
   instrument and an **invalid attribution instrument**." It attributes with a
   4 800× signal. What it still cannot do is **rank arms end to end** (§8.7, §11)
   — that half survives, and it is the half that matters.
3. **§12.2's reassurance is void.** "'Removing the MMAs does not help' is much
   less strange next to 'the loads hit L2'" is built on the falsified ablation.
4. **What is now attributable, and where to look first.** Per threadgroup the
   QK loop issues `HEAD_DIM/8 = 64` `simdgroup_load(kf, keys + k0*kv_stride +
   head_base + d8, kv_stride)` per 64-key tile with `kv_stride = 4·512 halves =
   4 KiB`. That is 32 lanes × 16 B at 4 KiB stride = **32 separate sector
   requests per fragment load**, 384 tiles deep. That is the first thing to
   ablate now that ablation is trustworthy, and the microbench is now a valid
   instrument for it.

---

## F2 — the documented prefill-pass grid gate is dead code in production

`crates/kernels/src/attention.rs`:

* `prefill_pass_count()` (`:339`) is the only implementation of the
  `PREFILL_PASS_MIN_GROUPS = 512` rule.
* `encode_online_attention_passes()` (`:1667`) — the function `model.rs:1911`
  calls — does **not** call it. It forwards straight to
  `encode_online_attention_passes_explicit()`, whose own comment says the gate
  "belongs to `prefill_pass_count`, which the production caller applies through
  `encode_online_attention_passes`" (`:1728-1729`). It does not.
* `grep -rn prefill_pass_count --include='*.rs'` → `attention.rs` (definition
  and doc comments) and `attention_prefill_pass_bit_exact.rs` **only**.

Runtime proof, clean build, `crates/engine/tests/adversary_pass_gate.rs` (added
here as evidence; it passes):

```
ADV: grid = 20 threadgroups (queries=10, kv_heads=4, block=2);
     PREFILL_PASS_MIN_GROUPS = 512; prefill_pass_count (the documented gate) = 0
ADV: encode_online_attention_passes (production) dispatched = true
ADV: pass edges for this key_count at the 8192 default = [0, 8192, 16384, 16394]
ADV: CONFIRMED -- the shipped path slices a 20-threadgroup grid, which the
     documented gate (512 groups) forbids.
test result: ok. 1 passed
```

That shape is not synthetic: it is **the last chunk of the 16 394-token prompt**
(10 queries, 16 394 keys, `flash_query_block_for` → `BQ = 2`), which every long
prefill produces. `implementation_notes.md` §16.5 states the opposite — *"The
grid gate is `<= 512` threadgroups … 128 queries x 4 kv heads at BQ=1 = 512
groups → one dispatch; 129 queries → sliced."*

Two consequences:

* The gate test (`the_grid_gate_holds_small_grids_to_one_dispatch`, and 3 more)
  tests a **function production never calls** — the documented trap "a test that
  calls a different entry point than the one claimed".
* Production pays state traffic (67 MB per pass boundary at a 1024-query chunk)
  for slices on grids that oMLX's own rule says will not pay.

---

## F3 — "default unchanged" is false; the N1 default is **on**

`attention.rs:242`: `pub const PREFILL_PASS_KEYS: usize = 8192;`, and
`prefill_pass_keys_for()` returns that when `KAPLUMBA_PREFILL_PASS_KEYS` is
unset. So **every prefill dispatch with `key_count > 8192` takes the pass path by
default**. Three documents disagree with the code and one another:

| place | says |
|---|---|
| commit `c61dcba` | "implementation kept, **default unchanged**" |
| `implementation_notes.md` §16.4 ¶1 | "the **default must stay one dispatch**, because 'no measurable difference' plus 64 MiB of scratch plus 2x the dispatch count is a net loss on anything that is not bandwidth-starved" |
| `implementation_notes.md` §16.4 ¶3 | "The gate **ships on by default** because it is measured neutral and bit-exact" |
| `spec.md` OQ9 | "The implementation ships … off with `KAPLUMBA_PREFILL_PASS_KEYS=0`" |

Measured cost of the default-on state, `KAPLUMBA_GPU_PROFILE=1`, 16 401 tokens,
same session:

| | attention own_ms | inter-boundary gap | kernel sum | wall span |
|---|---|---|---|---|
| `PASS_KEYS=0` (one dispatch) | 59 348.3 | **11 014.3** | 208 305.8 | 224 587.5 |
| default (`8192`, 2 passes) | 58 657.5 | **19 478.4** | 208 054.4 | 232 178.9 |

Attention itself is neutral (−1.2 %, noise), but the **submission gap grows
+8.5 s, +77 %** — the extra dispatches. Profiler-distorted walls, so take the
ratio and not the absolute; the clean A/B below is the wall number.

Also: at `pass_keys = 2048` the notes call the arm "8 passes". The last chunk
(16 394 keys) is a **9**-pass chain: `prefill_pass_edges(16394, 2048)` and
`prefill_pass_edges(16394, 8192)` = `[0, 8192, 16384, 16394]`, i.e. 3 passes at
the default. Only the 16 full chunks are 8/2 passes. Cosmetic, but it means the
"slices the key range" story is one pass over 8 of 16 chunks, not all of them.

---

## F4 — the default flip's quality evidence cannot fail, and the shipped source says it is not bit-exact

`attention.rs:433-454`, unmodified by this spec, in the file that ships:

> **The one-wave rule is NOT the default, and the reason is bit-exactness rather
> than doubt about the speed.** Changing the split count changes the order in
> which the combine pass sums the keys, and floating-point addition is not
> associative, so 13 splits is a different — equally valid, not equally
> identical — answer to the same question. […] This change is smaller (1 ULP on
> 3-4 of 16384 attention outputs, on 10 of 60 layers) and changes **43 of 128**.
> […] The measured consequence is in `spec/decode-19tok`: 1.34x on the
> full-attention layers at 8k of context, **and a different token stream**.

`NEXT.md` §7.5 offers the flip as "+2.17 % @16k, provably a no-op for 83 % of
layers. Needs an owner", and N6's AC5 evidence is "Five reference prompts
byte-identical between arms". That evidence is structurally vacuous.
`decode_splits_with` ends in `wanted.min(affordable)` with
`affordable = key_count.div_ceil(ATTENTION_MIN_KEYS_PER_SPLIT=64)`, so:

`crates/engine/tests/adversary_split_rule_table.rs` (added here), run under each
rule:

| key_count | ThreadTarget | WorkgroupsPerCore |
|---|---|---|
| 120 (the reference prompts) | **2** | **2** |
| 626 (the 614 scorecard row) | **10** | **10** |
| 658 (626 + 32 generated) | **11** | **11** |
| 832 | 13 | 13 |
| 896 | 16 | **13** |
| 1 024 … 24 590 | 16 | **13** |
| sliding, every key_count | 16 | 16 |

So both reference arms are *the same computation*; the diff can only ever be
empty. The first key count at which the flip can change anything is **896**,
i.e. any conversation longer than ~900 tokens.

I did try to break it in the other direction: 64 greedy tokens at 8 204 tokens of
context, one run per arm, `--temperature 0`:

```
$ diff q_shipped.txt q_workgroups.txt
2c2
< loaded in 7292 ms
---
> loaded in 7343 ms
```

Byte-identical. So the risk is **unquantified**, not demonstrated — but the only
evidence offered for "quality unchanged" is a test that cannot fail, set against
a source comment that says the change is not bit-exact and moved 43 of 128 tokens
in a previous measurement. AGENTS.md §8 is "no quality loss". This needs an
explicit quality decision, not a performance nod. (If the owner does want it:
measure the token-stream delta over ~256 greedy tokens at 8k and 16k, which is
the measurement `decode-19tok` already ran and `decode-19tok` is the only place
its number comes from.)

---

## F5 — the AC2 arithmetic is wrong by ~1.7×, and it charges dispatch overhead to attention

Measured, `KAPLUMBA_GPU_PROFILE=1`, this build:

| ms | 630 tokens (1 chunk) | 16 401 tokens (17 chunks) | per token |
|---|---|---|---|
| q/k/v/o/gate/up/down | 5 759.85 | 146 770.2 | 9.14 → **8.95** |
| attention (all 60 layers) | 378.04 | 59 348.3 | 0.60 → 3.62 |
| everything else | 91.48 | 2 187.2 | 0.145 → 0.133 |
| kernel sum | 6 229.4 | 208 305.8 | |
| **inter-boundary gap** | 314.7 | **11 014.3** | 0.719 ms / boundary |
| wall span | 6 560.8 | 224 587.5 | |

**What survives.** The GEMM is context-independent: 9.14 → 8.95 ms/token across a
26× context change. That part of the framing is measured and correct.

**What does not.** §12.1's "measured attention ~149.7 s" is `366.6 − 216.9` — a
**subtraction from a wall clock**, and the residual silently contains the
inter-dispatch gap, which is not attention and cannot be removed by making
attention faster. Extrapolating attention as `a·T + b·T²` fitted on the two
points above (the fit predicts **16.8 s at 8 204 tokens**, against §7.3's
independent "attention is ~17 s of a ~102 s prefill"):

```
at 24 590 tokens:
  GEMM + other          9.08 ms/token x 24 590 = 223.3 s
  attention             a*T + b*T^2            = 127.5 s
  inter-boundary gap    0.719 ms x ~22 530     =  16.2 s
                                              --------
                                                367.0 s  ->  67.0 tok/s
```

which closes on the measured 67.10. Against the 90 tok/s budget of 273.2 s:

* **239.5 s — 88 % — is already spent before attention begins.**
* attention gets **33.7 s**, not 56.1 s: a **3.8×** cut, not 2.7×.
* The FLOPs are not 1.88e14. Full layers `(T²/2) × 65 536 = 1.98e14`; sliding
  layers `T × window × 32 768 × 50 = 4.13e13`; total **2.40e14**. In 33.7 s that
  is **7.1 TFLOP/s = 46 % of the 15.6 TFLOPS tensor peak**, not the 22 % §12.1
  quotes.
* And the 2.40e14 breaks down badly. Measured per-shape at 16 401 tokens, the
  10 full layers' attention is **50.2 s** for 8.82e13 FLOP = **1.76 TFLOP/s
  (11 % of peak)** while the 50 sliding layers' is **8.9 s** for 2.75e13 =
  3.09 TFLOP/s (20 %). The kernel AC2 is actually about — the flash one — is the
  *slower* of the two per FLOP. Holding the sliding layers at today's 13.3 s at
  24k, the full-layer budget is **20.4 s for 1.98e14 FLOP = 9.7 TFLOP/s = 62 %
  of peak**, a **4.1×** improvement on that one kernel.

And the two ways to close it:

| lever | ask |
|---|---|
| attention 3.8× | 2.40e14 FLOP at 46 % of peak, or 62 % of it if only the flash kernel moves — in a kernel whose body cost (920 ms at 24 576 keys) has just been shown to be real and *unlocalised*, because four ablations never ran |
| GEMM 1.7× instead | 9.0 → 5.3 ms/token = **10.9 TFLOP/s = 70 % of peak**, against §3's record of seven named rejections and a declared 7.5 TFLOPS wall |

Floor with attention free: 24 590 / (223.3 + 16.2) = **102.7 tok/s**. So AC2 is
arithmetically possible and only just; but neither route is incremental, and the
honest recommendation is a **target change or a rewrite**, not another
lever-shaped session. Four levers have now been tried against one target
(T3 +0 %, T4f +6.7 %, T2b +8.2 % decode, N1 +0 %) and the residual is 3.8×.

---

## F6 — "a third independent reproduction" is the same observation again

`spec/decode-19tok/implementation_notes.md` §2: the budget 54 comes from **one**
sweep — "Swept every split count 1..=16 for three shapes that all use a
256-thread threadgroup and differ only in `kv_heads` […] 9 interleaved rounds",
printed as a single table. Two of those three shapes (`40q/5kv`, `48q/6kv`)
**are not in Gxma**; they are constructed by varying `kv_heads` while holding
`q_heads/kv_heads = 8`. And `kv_heads = 4 → 13` is the row the budget was read
off. This session re-measures the **derivation case** with a different decode
kernel.

That is a real and useful confirmation — it shows the constant transfers across
kernels — but it is a re-confirmation, not an independent third reproduction, and
the genuinely out-of-sample shapes (`kv_heads` 5 and 6, which is what makes
`budget / kv_heads` a rule rather than a fit) have still never been re-tested by
anyone on this machine.

The arithmetic itself checks out: `decode_one_wave_workgroups(32·(32/4)) =
decode_one_wave_workgroups(256) = 54`; grid `13·4 = 52 ≤ 54` (one wave),
`14·4 = 56 > 54` (two).

---

## F7 — the filed crash does not reproduce

`A ball is thrown straight up. What colour does it turn at the top of its flight?`
on `f2b2da1`, this worktree, `gpuq try`:

| `--max-new-tokens` | 16 | 17 | 24 | 31 | 32 | 33 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| rc | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Every one answered "It does not change color." Plus 10 further attempts at 96
tokens (10/10 rc=0) and `scripts/refprompts.sh` twice: `p1 Paris / p2 366 /
p3 CCCXCI / p4 the additive primaries / p5 no such numbers`, rc=0 both times.

"Reproducible every time" is therefore false on this commit, and the claim that it
was reproduced "on a build of base commit `b92234f`" cannot be checked from what
is committed. The most plausible reading, given their own polls show `unused`
down to 0.08 GiB on *good* runs and the session had many concurrent agents, is a
transient on a wedged box — not a mainline bug. **I could not confirm a mainline
defect and I am not going to invent one.** If someone still wants it, the useful
next step is to capture the failing run's `phys_footprint` and poll trace rather
than the prompt, because the prompt reproduces fine.

---

## F8 — N3(a): the sabotage reproduces and the suite does fail; one citation is fictional

`crates/engine/tests/attention_rescale_guard.rs:384` cites a sibling test
`a_misfiring_guard_is_caught` "where it reads 0.2136 — 43x the tolerance".
**There is no such test in the file.** The sabotage it describes is a manual
`.metal` edit, so nothing in the commit can re-run it.

I ran it. `if (msv == 1.0f)` → `if (msv != 1.0f)`, rebuilt into a separate target
dir:

```
attention_rescale_guard   rc=101
  test result: FAILED. 1 passed; 7 failed
  the guarded flash body is 0.21362305 from the scalar reference
    (queries=1024 keys=1024 window=0 block=2, 12 712 960 of 16 777 216 values differ)
  the guarded flash body is 0.24389648 from the scalar reference
    (queries=1024 keys=65 window=0 block=2, 2 283 638 of 16 777 216 values differ)
```

**0.21362305 — 43× tolerance — reproduces the number in the test file's comment
exactly.** So the two accounts do not actually disagree: the notes' 0.38110352
(76×) is the `ValueTable::Scaled` case and 0.2136 is the flat-`V` case, and both
are real. My "1.8× disagreement" reading was wrong; I withdraw it. What stands is
the narrower defect: **the control is not in the tree**, so the claim "with a test
that can fail" is only reproducible by hand.

Two things the sabotage also settles, both in the last agent's favour:

* **The `attention_prefill_pass_bit_exact` suite passes 17/17 with the guard
  inverted** — both of its arms run the same body, so it structurally cannot see
  a uniform corruption. That is the self-comparison trap, and they found it.
* **Removing the guard entirely also passes 8/8**, which is correct: the tests
  assert exactness of the result, not the presence of the optimisation.

Also worth stating plainly, because nobody did: **the guard's benefit has never
been measured.** §17.1 argues exactness only, and NEXT.md §7.2 lists it as
"landed" work without a number. It saves `NPER·2 = 16` fp32 multiplies per
(query, head, tile) against ~128 MMAs in the same tile. The guard *is* live rather
than dead — a tile of 64 keys moves a running max with probability ≈ 64/(T+64), so
over T/64 tiles the expected number of moves is ≈ ln(T/64), about 4.9 of 256 tiles
at 16k, i.e. **~98 % of tiles take the skip path** — but it is not a lever, and
no scorecard line should imply otherwise.

---

## N1 — prefill key-range passes: SURVIVED

**Engaged?** Yes, and not behind a flag (F3). The extra dispatches are visible in
the profile's inter-boundary gap (11.0 s → 19.5 s at 16 401 tokens).

**Coverage.** The 17 cases are genuinely good: ragged key counts around the 64-key
tile (129/191/255/257/1000/1023/1025/4095/4097/8191), pass sizes that do not
divide the key count, both query blocks, query tails at `BQ = 2` (1/3/5/7/511/513),
windowed shapes (64/256/1024/4096), a first pass in which every key is masked out,
a 128-dispatch chain, tile-alignment pinning, and a `bit_exactness_can_fail`
positive control. I re-ran it: **17/17 pass**, and **8/8** for
`attention_rescale_guard`. Gaps are F2 (the gate) and the fact that
`encode_online_attention_passes` — the production entry point — is called by no
test at all.

**"Bit-exact"** is the right word for the shipped code: pass edges are snapped to
`PREFILL_PASS_TILE = 64`, the state round trip is fp32→device→fp32, and the state
layout does not overflow its buffer
(`(queries−1)·q_heads·S + (q_heads−1)·S + (HEAD_DIM+1) = queries·q_heads·S − 1`).
The one-ulp finding was about the bug they caught, and it is a genuinely good
catch — one fp16 ulp on 1 % of outputs is 10× inside the house tolerance, so it
would have shipped as "exact".

**Measurement, independently repeated.** `scripts/bench_prefill_passes.sh`,
16 394 tokens, `--max-new-tokens 1`, interleaved, 2 rounds, per-arm files:

| arm | r1 | r2 | mean |
|---|---|---|---|
| one dispatch (`PASS_KEYS=0`) | 74.067 | 75.643 | 74.855 |
| `PASS_KEYS=8192` (the default) | 76.043 | 73.866 | 74.955 |

**+0.13 %**, inside a per-arm spread of ±1.4 % (their quoted spread was 0.12 % and
0.84 %). Neutral reproduces; it is a *weaker* neutral than they reported, and the
correct reading is "below the resolution of this instrument at this context".

**Is 16k the wrong place?** Partly, but the direction of the residual argument is
right and is theirs: §10.1 already showed a **64× smaller K/V allocation buys 6 %**,
so the kernel is not limited by K/V traffic at all — at any context — and a
mechanism whose whole premise is cache-residency loss has nothing to recover.
That argument gets *stronger* at 24k, not weaker. What is genuinely missing is a
clean 24k sample; I attempted one and got half of it (`adv-s6/summary.tsv`):

```
r1 onedispatch 0 prompt_tokens=24602 INVALID-NO-NUMBER footprint=17.07GiB
      mem=[INVALID unused-dropped-to-0.03GiB (limit 0.05)] cpu=[sys-cpu 3.2%]
r1 pass8192  8192 prompt_tokens=24602 prefill_tok_s=68.401 footprint=17.01GiB VALID
      mem=[VALID polls=70 compressor_max=7.12GiB unused_min=0.08GiB]
```

So: the effect arm measured **68.401 tok/s at 24 602 tokens** — the first valid
24k prefill sample of this session, and +1.9 % on the spec's one-round 67.10 — and
the control arm was **discarded** because `unused` fell to 0.03 GiB, i.e. *below*
the gate's own 0.05 GiB threshold. N1's 24k A/B therefore remains unmeasured, and
the reason is OQ8 exactly as stated: at 24 602 tokens the footprint is 17.0 GiB
and the box has 30–80 MiB of slack left, against 0.08 GiB at 16k. The gate
correctly refused to certify the control arm; it could not have produced a valid
one.

---

## N3(b) — `HS == 2`: SURVIVED, under-analysed for prefill

The arithmetic is right: oMLX's `_head_split(rows) = 2 if rows == 1 else 1` with
`QB = (G·ROWS)/8`; our decode shape is `G = 32/4 = 8`, `ROWS = 1` → `QB = 1`, so
`HS = 2` engages 2 of 8 simdgroups, and `NTILES/NSG = 64/8 = 8 = NPER` with
`nt = sg·NPER + j` already splits PV eight ways over the value dims.

The gap: §19 argues the **decode** shape only. Prefill is the other kernel, and it
is not `BQ = 1` — `flash_query_block_for` selects `BQ = 2` at
`key_count ≥ FLASH_QUERY_BLOCK_MIN_KEYS = 16 384` (`attention.rs:96`), i.e. at
both 16k and 24k, the two contexts the spec cares about. There `QB = (8·2)/8 = 2`
bands, so `HS = 2` would engage 4 of 8 simdgroups. The conclusion is unchanged
(half the simdgroups idle, no spare band to put in them), but it was not the
argument that was made, and the "PV is already split 8 ways" premise is a
statement about the PV half only — §19 says so itself for the QK half.

---

## N4 — the split sweep: SURVIVED

My own interleaved sweep, `scripts/bench_decode_splits.sh`, 8 204-token prompt,
MMA decode kernel, 8 arms × 2 rounds, poll file per arm:

| splits | grid (×4 kv heads) | r1 | r2 | mean |
|---|---|---|---|---|
| 8 | 32 | 12.652 | 12.639 | 12.646 |
| 10 | 40 | 12.737 | 12.754 | 12.746 |
| 11 | 44 | 12.818 | 12.815 | 12.817 |
| 12 | 48 | 12.867 | 12.872 | 12.870 |
| **13** | **52** | 12.924 | 12.918 | **12.921** |
| 14 | 56 | 12.720 | 12.729 | 12.725 |
| 16 | 64 | 12.801 | 12.801 | 12.801 |
| shipped (no override) | 64 | 12.790 | 12.811 | 12.801 |

**13 is a genuine optimum, not a point fit**: monotone rise from 8, a **−1.5 %
cliff at 14**, and 16 recovering only to −0.9 %. Round-to-round repeatability is
0.03–0.15 %, so the 3.7–6.4 tok/s "same-binary spread" the brief warns about is
**not present on this box right now** for interleaved decode — which is worth
recording, because four sessions of spec work were argued around it.

At 16 394 tokens, 2 interleaved rounds:

| arm | r1 | r2 |
|---|---|---|
| shipped | *4.732* | 12.168 |
| splits 13 | 12.375 | 12.393 |
| splits 16 (explicit) | 12.186 | 12.198 |

**+1.55 % / +1.60 %**, arm spread ±0.1 %. Reproduces their +1.97 %/+2.17 %.

> **The round-1 control is the headline.** `shipped` read **4.732 tok/s** — a
> 2.6×-degraded sample — and the memory gate passed it
> (`VALID polls=43 compressor_max=7.47GiB unused_min=0.08GiB`). Had it been used
> as the control, the scorecard would have read "+161 %". This is the documented
> "a 'did nothing' reading taken against a baseline that came from a failed run"
> trap, live, in my own sweep, and it is the strongest argument in this report for
> F7/N5.

**Short context: no evidence exists.** The −0.01 % at 614 is an identity, not a
measurement (F4 table): at key_count 626 both rules give 10 splits. The notes'
explanation ("below 4096 keys the decode kernel is not the MMA one and the split
rule does not apply") is also wrong — the rule applies, it just agrees.

**"Provably a no-op for 50 of 60 layers": SURVIVED.** Gxma is 60 layers, 10 full
(32q/4kv/d512) and 50 sliding (32q/16kv/d256, `config.json`). A sliding layer's
key count is `gpu_cache.key_count(false) = next_position.min(capacity)` — the
1024-slot **ring**, not the context — so it is ≤ 1024 at every position and both
rules return 16 (and identical values below that). The notes' own reason
(`288/16 = 18` → clamped to 16) is also correct. Two people can stop worrying
about the sliding layers; the reason is the ring, which is nowhere written down.

---

## N5 — the memory gate: SURVIVED, and I have the pair they did not

Their calibration compares a **626-token** good run against a **24k** stall —
different KV and scratch footprints by construction. The brief asked for the
like-for-like pair. It fell out of the 16k sweep by accident, at **identical
context and identical prompt**:

| run | decode tok/s | `compressor_max` | `unused_min` | verdict |
|---|---|---|---|---|
| degraded | **4.732** | **9.55 GiB** | **0.09 GiB** | `VALID` |
| good | 12.375 | 8.42 GiB | 0.06 GiB | `VALID` |
| good | 12.393 | 7.76 GiB | 0.08 GiB | `VALID` |

A 2.6× pessimised sample sits **inside** the good envelope on both fields, and is
in fact *higher* on both. Their conclusion — "a memory-pressure gate on this box
cannot tell a good run from a stalled one, and a threshold tuned on either signal
rejects good runs" — is **correct**, and now demonstrated at equal context length
rather than across two different workloads. The `MEM_UNUSED_LIMIT_GIB = 0.05`
threshold they shipped is, as they say, a "wedged box" tripwire and nothing more;
that is the honest setting.

One correction to their own tooling: the `mem_cpu_compression_ok` check is
`sys/real > 2 %`, which is **not** CPU-in-compression. On this box `sys` is
dominated by the 11.5 GiB mmap'd weight page-in; every good arm I ran read
**3.1–7.8 % of wall**. They correctly leave it ungated; the brief's instruction to
discard any sample over 2 % would have discarded every run in this report,
including all the good ones.

**Induced pressure, same context.** I also ran the 16 394-token prompt twice
clean and twice alongside a page-touching hog (`adv-s5.log`):

| arm | extra resident | prefill tok/s | decode tok/s | `unused` (poll min) | swap used after |
|---|---|---|---|---|---|
| healthy | 0 | 75.976 | 12.154 | 98M | 2 220M |
| hog 4 GiB | 4 | 75.865 | 12.158 | 99M | 2 180M |
| healthy 2 | 0 | 74.943 | 12.029 | 98M | 2 501M |
| hog 7 GiB | 7 | 76.641 | 12.147 | 99M | 2 495M |

**7 GiB of induced pressure changed nothing measurable** (spread 74.9–76.6 prefill,
12.03–12.16 decode) and the `unused` envelope is 98–99 MiB in all four arms. So
the documented 0.055 tok/s stall needs more pressure than 7 GiB on top of a
15.5 GiB footprint — i.e. enough to push the *model's own* pages out — and I did
not push that far, because wedging the box for other agents is not a price worth
paying for this claim. The same-context pair from the sweep above is the decisive
evidence; this only establishes that the stall is not reachable by *adding*
pressure at a fixed context, which is itself worth knowing: it means the stall is
about the 24k run's own 18.35 GiB footprint evicting the desktop, exactly as OQ8
says, and not about background load.

---

## N6 — the scorecard: mostly SURVIVED

* **The 4 failures are pre-existing.** Rebuilt `b92234f` in a separate worktree
  with its own `CARGO_TARGET_DIR` and ran the four binaries:
  `perf_probe_concurrency_dependency_graph` 13 passed / 1 failed,
  `perf_probe_dispatchsel_columns` 16/1,
  `perf_probe_host_ops` 14 passed / 1 failed / 1 ignored,
  `probe_cache_hierarchy` 15/1 — same names, same counts as N6 reports.
* **The two new suites pass at HEAD**: `attention_prefill_pass_bit_exact` 17/17,
  `attention_rescale_guard` 8/8.
* **"116 binaries ok" is unverified by me.** I did not re-run the full suite at
  HEAD. The diff touches shared `.metal` and `model.rs` scratch, so I cannot rule
  out collateral damage from the 112 I skipped; I would not have claimed it either.

---

## F9 — the harness fix landed in one of the three harnesses

`f2b2da1` fixed "`bench_decode_splits.sh` silently ignored `LOG=`" by writing
`SUMMARY=${LOG:-$OUTDIR/summary.tsv}`. The same defect is still present in

* `scripts/bench_prefill_passes.sh` — `SUMMARY="$OUTDIR/summary.tsv"`
* `scripts/scorecard.sh` — `SUMMARY="$OUTDIR/scorecard.tsv"`

so `LOG=… scripts/bench_prefill_passes.sh …` writes nothing where asked, which is
the exact failure mode §21 item 1 describes. It cost me a run: my N1 A/B wrote to
`adv-s3/pass_16k.tsv` (empty) while the numbers sat in `adv-s3/summary.tsv`.
Small, but it is the same trap re-armed in two of the three places.

---

## What I could not break, and what I tried

* **N4's 13.** Four attempts: my own interleaved 2-round curve with six interior
  points; the 14 arm specifically to look for a point-fit; reading
  `decode_splits_with` for a clamp that might manufacture the peak (there is
  none at 8k — `affordable` is 129); and reading the `spec/decode-19tok` table
  the rule came from. It is a real optimum at a real wave boundary.
* **The 50/60 no-op.** Checked `gpu_cache.key_count(false)` directly rather than
  trusting the notes' argument; it is the ring capacity, and both rules agree at
  every key count.
* **The four pre-existing test failures.** Rebuilt the base rather than believing
  the commit message.
* **N1's bit-exactness.** Re-ran the suite, and checked the state-layout
  arithmetic by hand against `prefill_pass_state_elements`.

## Reproduction

Raw logs are committed under `spec/prefill-24k-90tps/adversary_evidence/`:

| file | what |
|---|---|
| `f1_oq7_clean_kernel_bench.log` / `f1_oq7_empty_kernel_bench.log` | F1, clean vs immediate-return build |
| `f2_dead_grid_gate.log` | F2, runtime proof the gate is not applied |
| `profile_614_tokens.log`, `profile_16384_onedispatch.log`, `profile_16384_defaultpasses.log` | F3 + F5, per-kernel and inter-boundary-gap budgets |
| `n1_pass_ab_16k.tsv`, `n1_pass_ab_24k.tsv`, `n1_pass_suite_head.log` | N1 |
| `n3a_rescale_guard_head.log`, `f8_rescale_guard_inverted.log`, `f8_rescale_guard_removed.log` | N3(a) + F8 |
| `n4_split_sweep_8k.tsv`, `n4_split_sweep_16k.tsv` | N4 |
| `n5_induced_pressure.log` | N5 |
| `f7_refprompts.log` | F7 |

Commands:

```bash
# F1: add `if (true) { return; }` as the first statement of
# attention_prefill_flash_body, build into a SEPARATE CARGO_TARGET_DIR, run
#   scripts/gpuq try -- $TARGET/release/deps/attention_qk_mma_bench-* --nocapture
# then `git checkout crates/kernels/src/attention.metal`
# F2: cargo test --release -p kaplumba-engine --test adversary_pass_gate -- --nocapture
# F4: run adversary_split_rule_table twice, once with KAPLUMBA_DECODE_SPLIT_RULE=workgroups
# N4: ROUNDS=2 LOG=... scripts/bench_decode_splits.sh 8192 8 10 11 12 13 14 16
#     ROUNDS=2 LOG=... scripts/bench_decode_splits.sh 16384 13 16
# N1: ROUNDS=2 scripts/bench_prefill_passes.sh 16394 8192     (and 24590 8192)
```

Two artefacts are added to the tree as evidence and are clearly marked
`ADVERSARY ARTIFACT`: `crates/engine/tests/adversary_pass_gate.rs` (F2) and
`crates/engine/tests/adversary_split_rule_table.rs` (F4). No product code was
changed; the two `.metal` sabotages (F1, F8) were reverted and
`git diff crates/kernels/src/attention.metal` is empty. The main tree was not
touched at any point.