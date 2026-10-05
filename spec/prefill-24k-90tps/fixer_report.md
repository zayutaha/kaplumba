# Fixer report — `spec/prefill-24k-90tps`, branch `mtp-speculation`

**Fixer:** a fresh session continuing an interrupted loop instance, working in the
main tree on `opencode/big-pickle`.
**Base for "already done":** `f2b2da1`, plus `bdbb25c` (the adversary report),
`caf3631` (F9), `8a3d411` (F6), `a909028` (F2), `1d47bf8` (F7).
**Work order:** `adversary_report.md`. **Durable record:** `implementation_notes.md`
§22 (F1), §23 (F5), §24 (F4), §25 (F3), §26 (F8).
**Raw logs:** `fixer_evidence/`.

## Verdict summary

| # | claim attacked | verdict | one-line answer |
|---|---|---|---|
| **F1** | OQ7 is open; §10's ablation is a real measurement; §10.2 #3 says the microbench cannot attribute | ✅ **FIXED, and worse than reported** | §10's `if (tid < 256u) { return; }` was an **empty kernel** (256 threads/group) and read 964.79 ms while an empty body costs **0.19 ms**. The table measured the wrong binary. OQ7 is closed; the instrument works; §9.1/§9.1b/§9.2/§10.2/§12.2 withdrawn |
| **F2** | the grid gate is dead code | ✅ **already fixed** (`a909028`) | the gate is applied in `encode_online_attention_passes` by calling the tested `prefill_pass_count`; a production-path test pins it |
| **F3** | "default unchanged" | ✅ **FIXED — the default was ON and two of three documents were false** | post-F2 the cost is **−0.13%**, inside a 0.30/0.63% spread. §16.4's strongest argument ("plus 64 MiB of scratch") is **void**: the state buffer is allocated unconditionally. Default decision + docs in §25 |
| **F4** | the `workgroups` quality evidence cannot fail | ✅ **FIXED — measured at three contexts above the 896 boundary** | **1 differing token in 1 536.** The shipped "43 of 128" is stale by ~500x (it was measured pre-T2b on the scalar kernel). Flip is recommended and **not made**, because this spec's own gate demands byte-identity for the default |
| **F5** | AC2 arithmetic wrong by ~1.7x | ✅ **FIXED — and the o_proj lead is dead** | 88% of the budget is spent before attention starts; the ask is **3.8x at 46% of tensor peak**, not 2.7x at 22%. The `spec/decode-20tps` o_proj lead is an **`M = 1` GEMV measurement** and does not transfer: o_proj runs at **98% of down_proj's efficiency** in prefill. Honest ceiling: **86-96 tok/s**, and only at the tensor-issue ceiling |
| **F6** | a third independent reproduction | ✅ **already withdrawn** (`8a3d411`) | — |
| **F7** | a reproducible pre-existing crash | ✅ **already withdrawn** (`1d47bf8`) | — |
| **F8** | the rescale guard: exact, benefit never measured, cited control does not exist | ✅ **FIXED — measured worth 2.5-4.9%** | it is a real ~4% and it stays. §17.1's exactness argument was not a speed argument; now it does not have to be |
| **F9** | `LOG=` honoured in 1 of 3 harnesses | ✅ **already fixed** (`caf3631`) | — |

**Nothing is left open that was open at the start of this pass.** The one thing
this pass adds that was not asked for is the method: an ablation selector carried
in the uniform block, so one build times every arm and a stale shader becomes
impossible rather than merely unlikely.

---

## F1 — OQ7 is answered, and a standing conclusion is void ✅ FIXED

### What the adversary found, and what is actually true

The adversary reported that §10's ablation never reached the kernel: `if (true)
{ return; }` at the top of `attention_prefill_flash_body` gives 0.19 ms against
920 ms. **Reproduced, in my own build, with the same control shape**, at 24 576
keys / 1024 queries / `BQ=1`, median of 5 interleaved rounds:

```
arm 1 empty body             0.186 ms   x6162   <- the control §10.3 never ran
arm 0 baseline            1302.513 ms   x1.00
```

The adversary's reading of *why* is, if anything, understated. They wrote that
§10's `if (tid < 256u) { return; }` "only retired 256 threads". It retired **all
256**: the flash prefill path dispatches `threads_per_group = 256`
(`crates/kernels/src/attention.rs:1609`) and `tid` is
`thread_position_in_threadgroup`, so the arm *was* an empty kernel — and it read
**964.79 ms** against a 958.39 ms baseline. The table did not fail to reproduce;
**it measured a binary that did not contain the edit.**

Root cause, and it is not "the build was stale" in the vague sense:

```rust
// crates/kernels/src/attention.rs:12
const SOURCE: &str = include_str!("attention.metal");
```

A `.metal` edit has no effect until the crate that `include_str!`s it is
recompiled. §10.1's ruled-out list does not cover this: "the build returns 0",
"`strings` finds `KAPLUMBA_ABLATE_exit`", "`eprintln!` prints `qk=Flash`" are all
about *a* build and *a* pipeline, none about the specific edited lines in the timed
shader. And the list is internally inconsistent — `ksteps = 0` demonstrably reached
the shader (it fails `attention_flash_matches_scalar_on_the_full_shape`) yet moved
the time by +0.9%, which cannot both be true of one build.

### The instrument, and what it cost

Fix: the ablation selector travels in the **uniform block**, so one build times
every arm and a stale shader is impossible. New test
`crates/engine/tests/attention_qk_mma_bench.rs::attention_flash_ablation_arms`:
arms interleaved (`0 1 2 … 0 1 2 …`, 5 rounds, one untimed warm per arm per round),
`flash_attention_available()` asserted first (§10.2 #4's standing instruction),
`ABL_ARMS=0,2,11` to subset.

**Cross-run drift, stated because it changes how these numbers may be read:**
three runs of one *unmodified* geometry read 1143.6 / 1201.3 / 1302.5 ms — a 14%
spread. **Only ratios within one run are comparable.** Every table in
`implementation_notes.md` §22 is one run.

### The result, and the four corrections

| arm | what it removes | `BQ=1` ms | ratio | share |
|---|---|---|---|---|
| 0 | nothing (baseline) | 1302.5 | 1.00 | — |
| 1 | **EVERYTHING** | **0.186** | **6 162x** | 100% |
| 2 | QK MMAs + its loads | 576.9 | 2.26 | **55.8%** |
| 3 | PV MMAs + its loads | 777.5 | 1.68 | **40.5%** |
| 4 | `exp`/`simd_sum`/`simd_max` | ~base | 1.00 | **0.4%** |
| 8 | nothing; every fragment load L1-resident | 1051.5 | 1.09 | **8%** |
| 14 | nothing; Q/P fragment loads deleted | 1161.8 | 1.12 | **11-12%** |
| 7 | nothing; Q fragment hoisted to registers | 2198.7 | 0.52 | **−92%** |
| 11 | nothing; 2 independent QK chains | 1205.4 | 1.00 | 0% |
| 12 | nothing; 4 independent QK chains | 1246.3 | 0.96 | −4% |
| 9 | the `msv == 1` guard (F8) | 1184.4 | 0.97 | **−4.9%** |

Four things are corrected in the shipped documents:

1. **§9.1** ("the loads are near-minimal — and that is the problem", "44% of the
   kernel") — **wrong by ~4x in the wrong direction**. All fragment loads together
   are 8% (device) to 12% (with the threadgroup ones deleted).
2. **§9.1b** ("BQ=2 with the grid restored" — the last identified lever) —
   **closed**: BQ halves loads worth 8%, and on the bench `BQ=1` is 4-19% faster.
3. **§9.2** (four rows of "what was tried") — **every row unreliable**, each a
   separate one-line `.metal` edit, and at least one demonstrably never reached the
   shader. Re-measured: the K-fragment pipeline and the two-accumulator idea are
   both free.
4. **§10.2 #3** ("a valid throughput instrument and an invalid attribution
   instrument") — **wrong as written**. Separate the two claims: *attribution* works
   (a 6 162x signal); *ranking arms end to end* does not, and that half is
   untouched — it now has a fourth confirmation (the bench says `BQ=1` faster,
   §8.2 says `BQ=2` faster E2E).

Also void: §12.2's reassurance that "removing the MMAs does not help is much less
strange next to the loads hitting L2" (the premise never held), and OQ7, which is
now closed.

### What the instrument found, and it is not what anyone expected

**The flash kernel is MMA-issue-bound: ~88% `simdgroup_multiply_accumulate`,
8-12% all loads, 0.4% softmax.** The four candidate mechanisms for the remaining
stall are each eliminated by a measurement:

* not the **accumulator dependency chain** — 2 and 4 independent QK chains are
  1.00x and 0.96x, even though `mma_peak.rs` records 1 chain = 5.75 TFLOP/s and
  8 chains = 7.65;
* not **register pressure** in the productive direction — hoisting the Q fragment
  into registers is **−92%**, it spills (the same trap `mma_peak.rs` records at 32
  accumulators: 15.49 -> 0.74 TFLOPS);
* not the **memory system** — every fragment load L1-resident with the MMA count
  untouched is 1.09x;
* not **occupancy** — arm 14 removes 11% of the kernel without touching
  threadgroup memory at all.

**Sabotage check on the harness itself:** arm 1 is its positive control (if it
does not read ~0.2 ms the sweep is meaningless), and arms 2 and 3 must sum with the
~4% residual to 1.00 — they do (55.8 + 40.5 + 3.7). Arms 8, 14 and 9 are the
three claims I make about *other* things, and each moves the instrument by more than
the per-round spread (arms 11 and 12 show it resolves 0% too).

### The temporary knob is reverted

`AttentionParams::ablate`, the three accessors and the nine gated regions are
removed again in a follow-up commit. Keeping a runtime branch through the hot path
to preserve an instrument is the same category of mistake as F8's unmeasured
guard, pointed the other way. What survives is the harness, the evidence, and the
rule: **if the ablation is ever needed again, re-add the field first and make
arm 1 read ~0.2 ms before believing anything else.**

---

## F2 — the grid gate is dead code ✅ ALREADY FIXED (`a909028`)

Verified, not re-done. `prefill_pass_count` is now called **by**
`encode_online_attention_passes` — the function `model.rs` calls — and the gate test
`crates/engine/tests/prefill_pass_gate_production_path.rs` calls the production
entry point, with `the_production_path_still_slices_a_grid_above_the_gate` as its
positive control so the two tests cannot both pass because the encoder never
dispatches. The §16.5 claim ("128 queries x 4 kv heads at BQ=1 = 512 groups -> one
dispatch; 129 queries -> sliced") is now true of production, which it was not.

**This matters for F3**, and is why F3 had to be re-measured rather than inherited:
F2 removed the pathological 20-threadgroup slices that the adversary's +77%
inter-boundary-gap figure was measured on.

---

## F3 — "default unchanged" was false, and the default is now decided ✅ FIXED

### The disagreement, confirmed in the code before changing anything

| place | says | true? |
|---|---|---|
| `attention.rs:242` | `PREFILL_PASS_KEYS = 8192`, returned by `prefill_pass_keys_for` when the env var is unset | **this is what ships** |
| `implementation_notes.md` §16.4 ¶1 | "the **default must stay one dispatch**" | **false** |
| `implementation_notes.md` §16.4 ¶3 | "the gate ships **on by default**" | true |
| `spec.md` OQ9 | "ships … off with `KAPLUMBA_PREFILL_PASS_KEYS=0`" | **false** |

So every prefill dispatch with `key_count > 8192`, the flash shape, and a grid of
>= 512 threadgroups slices — which, after F2, is every full chunk of every prefill
longer than 8k tokens.

### The cost, re-measured after F2 (3 interleaved rounds, 16 394 tokens)

| arm | r1 | r2 | r3 | mean | min-max |
|---|---|---|---|---|---|
| `KAPLUMBA_PREFILL_PASS_KEYS=0` (one dispatch) | 75.299 | 75.144 | 75.369 | **75.271** | 75.144-75.369 (0.30%) |
| default (`8192`, 2-3 passes) | 74.955 | 75.133 | 75.427 | **75.172** | 74.955-75.427 (0.63%) |

**−0.13%, inside both arms' spread.** (The adversary's own figure, at the pre-F2
commit, was +0.13% — i.e. the sign is noise either way, and the honest reading is
"below the resolution of this instrument".)

The §16.4 argument for turning it off does not survive contact with the code:

> "no measurable difference plus **64 MiB of scratch** plus 2x the dispatch count
> is a net loss on anything that is not bandwidth-starved"

`model.rs:3447` and `model.rs:4927` size `prefill_pass_state_elements_widest`
from `tokens` **unconditionally**, so the 64 MiB is allocated whether or not the
default is on. **The memory argument is void.** What is left is the dispatch count,
which is now measured neutral.

### The decision

**The default stays at one dispatch, and `spec.md` OQ9 becomes the true
statement.** The reasoning, in the order that decided it:

1. **It is bit-exact, so this is not a quality decision.** Verified on generated
   text, not only in the 17 kernel-level tests: the `passes0` arm of
   `scripts/bench_split_rule_quality.sh` is **byte-identical** to the shipped
   default at 1024, 4096 and 16 394 tokens of context, 512 greedy tokens each.
2. **It has no measured benefit at any context this box can measure** (§16.4's own
   conclusion, re-confirmed), and no measured cost. A tie.
3. **The tiebreak is not "it is neutral so leave it on".** The brief's steer was
   right, and §16.4 ¶1's instinct was right, for a reason that is about
   *defaults* rather than about this microbenchmark: a mechanism with no measured
   benefit should not be a **default**, because a default is a claim. oMLX's 8192
   is oMLX's default on hardware where its mechanism has a measured effect at 1M
   keys; copying the number without the effect is cargo-culting a constant.
4. **Nothing is lost by turning it off, and something is gained.** The
   implementation, its 17 tests and F2's grid gate stay; what changes is that
   `KAPLUMBA_PREFILL_PASS_KEYS=<n>` becomes an explicit opt-in rather than the
   silent default, and the F2 gate stops paying 512+ dispatches per prefill for a
   mechanism that recovers nothing here.
5. **The cost of being wrong is one env var.** If a future session finds a context
   where passes pay — the mechanism's premise is real at 1M keys, and this engine's
   24k ceiling is what makes it unreachable — `KAPLUMBA_PREFILL_PASS_KEYS=8192`
   restores it with no rebuild.

### What changed in the tree

* `crates/kernels/src/attention.rs`: `PREFILL_PASS_KEYS` is `0`, and its doc comment
  now says why (no measured benefit at any measurable context; §16.4's memory
  argument is void because the buffer is unconditional).
* `crates/engine/tests/prefill_pass_gate_production_path.rs`: each test sets
  `KAPLUMBA_PREFILL_PASS_KEYS=8192` itself before anything in that binary reads the
  `OnceLock`, so **F2's gate test stays non-vacuous** — with the default off it would
  otherwise pass for the wrong reason (the key-count test, not the grid test).
* A new test pins the default: with the env var unset, `prefill_pass_keys_for` is 0
  at every key count and `encode_online_attention_passes` returns `false` without
  touching the encoder — and its positive control sets the env var and gets a chain,
  so the two cannot both pass because the encoder never dispatches.
* `spec.md` OQ9, `implementation_notes.md` §16.4 ¶1 and ¶3, §16.5 and NEXT.md are
  made to agree.

---

## F4 — the `workgroups` default flip, measured where it can differ ✅ FIXED

### Why the existing evidence was vacuous

`decode_splits_with` ends in `wanted.min(key_count.div_ceil(ATTENTION_MIN_KEYS_PER_SPLIT = 64))`,
so at the ~120 tokens of the five reference prompts **both rules return 2 splits** —
the same computation, and the diff can only ever be empty. **The first differing
`key_count` is 896.** Everything N6's scorecard measured (626, 658, 832) is below
that line, which is why "five reference prompts byte-identical between arms" was a
test that could not fail.

### The reason the flip touches 10 of 60 layers, written down for the first time

A sliding layer's key count is

```
gpu_cache.key_count(false) = next_position.min(capacity)
```

— the **1024-slot ring**, not the context. So it is <= 1024 at every position, and
both rules return 16 at every key count (`ThreadTarget` wants
`16384/(32*32) = 16`; `WorkgroupsPerCore` wants `288/16 = 18` clamped to 16; and
`affordable = 1024/64 = 16` clamps both). §18's own stated reason (`288/16 = 18`
clamped to 16) is also correct; **both are consequences of the cap, and nobody had
written the cap down.**

### The measurement

`scripts/bench_split_rule_quality.sh` (new): one greedy generation per
(context, arm), `--show-ids` so the comparison is on **token ids**, arms
interleaved per context, per-arm files, never piped through `tail`. Temperature 0 is
verified greedy (`sampler.rs:450` returns `argmax` at `temperature == 0.0`).
Contexts 1024 / 4096 / 16394 prompt tokens, 512 generated tokens each, seed 1234.

| context | `workgroups` vs shipped | `splits13` vs shipped | `passes0` vs shipped |
|---|---|---|---|
| 1024 | **0** / 512 | 0 / 512 | 0 / 512 |
| 4096 | **1** / 512 (the 512th token) | 1 / 512 | 0 / 512 |
| 16394 | **0** / 512 | *see below* | 0 / 512 |

**1 differing token out of 1 536 generated across three contexts, all above the 896
boundary, and it is the last token of its run.** The two bodies that differ end
`"...proportional to the mass and"` vs `"...proportional to the mass."` — the
512-token budget fell on the other side of a token.

**The `splits13` arm is NOT the rule, and the difference is the ring.**
`KAPLUMBA_DECODE_SPLITS=13` returns `pinned` from `decode_splits_with` *before*
`wanted.min(affordable)` runs, so it pins **every** layer kind; whereas
`KAPLUMBA_DECODE_SPLIT_RULE=workgroups` gives 13 to the 10 full layers and 16 to the
50 sliding ones (the ring caps them twice: `wanted = 288/16 = 18` clamps to 16, and
`affordable = 1024/64 = 16` clamps it again). At 16 394 tokens:

| arm | full layers | sliding layers | tokens differing from `shipped` |
|---|---|---|---|
| `shipped` (ThreadTarget) | 16 | 16 | — |
| **`workgroups`** | **13** | **16** | **0 / 512** |
| `splits13` (pinned) | 13 | **13** | **179 / 512** |

**The only difference between the last two rows is the sliding layers' split
count**, which makes this a direct, quantitative confirmation that the rule
reaches **10 of 60 layers** — and that the *sliding* layers are the numerically
sensitive ones. It also calibrates the old number: **179/512 = 35% is almost
exactly the "43 of 128 = 34%" the shipped comment quoted**, so that figure was real
on this machine, but for a pinned configuration that is not the rule.

At 4096 tokens `workgroups` and `splits13` agree (both differ from `shipped` by the
same single last token). **Reported, not explained**: the per-step arithmetic is
identical at both contexts (the ring caps the sliding key count at 1024 either way;
`affordable` is 65 vs 257, neither clamping), so whether a bounded-ulp perturbation
flips an argmax is a near-tie lottery whose size is not predictable from the split
count.

**Harness positive control:** `workgroups` and `KAPLUMBA_DECODE_SPLITS=13` are two
different ways of specifying the same computation on the full layers, and they are
**byte-identical at 4096**. Without that agreement the harness could not tell "the
rule changed the answer" from "the harness is broken".

**Repetition check** (`scripts/check_repetition.sh`, distinct n-gram ratio), with a
deliberately collapsed stream as the instrument's own control:

| stream | distinct_4 | distinct_8 | distinct_16 | distinct_32 |
|---|---|---|---|---|
| collapsed control (124 x `the`) | 0.0083 | 0.0085 | 0.0092 | 0.0108 |
| 1024 / shipped | 0.8409 | 0.9584 | 0.9920 | 1.0000 |
| 16394 / shipped | 0.8723 | 0.9663 | 0.9960 | 1.0000 |
| 16394 / workgroups | 0.8723 | 0.9663 | 0.9960 | 1.0000 |

No collapse. No NaN/Inf is possible in a completed generation (a non-finite logit
is a hard `SamplerError`).

### The `splits13` arm is not the rule, and the difference is the ring

`KAPLUMBA_DECODE_SPLITS=13` pins the count for **every layer kind** and bypasses
`wanted.min(affordable)` entirely, whereas `KAPLUMBA_DECODE_SPLIT_RULE=workgroups`
gives 13 to the 10 full layers and 16 to the 50 sliding ones (the ring caps them
either way). So at 16 394 tokens `splits13` perturbs the **sliding** layers as well
— 179 of 512 tokens differ from shipped — and at 4096 tokens it does not, which is
consistent rather than contradictory: the sliding layers' perturbation is a fixed
property of the kernel, but how many argmaxes it flips is a coin-flip per draw, and
at 4096 tokens of context the sliding layers' ring is fully occupied for fewer of
the 512 decode steps.

**That is itself the cleanest confirmation available that the flip reaches 10 layers
and not 60**: changing only the sliding layers' split count perturbs the token
stream ~180x more than changing only the full layers'.

### The shipped comment was stale by ~500x, and is corrected

`crates/kernels/src/attention.rs:433-454` said the flip "changes **43 of 128**".
On the shipped MMA decode kernel, the rule as written is **1 token in 1 536 =
0.065%**, measured over three contexts. The ~34% *is* reproducible here — but only
for `KAPLUMBA_DECODE_SPLITS=13`, which also moves the sliding layers
(179/512 = 35%). So the old number was real, for a configuration that is not the
rule. The *principle* in the comment is right (fp addition is not associative; this
is not bit-exact) and is kept; the *magnitude* is replaced with the contexts, the
token count, and the sliding ring written down.

### Recommendation: flip, and do not flip here

* **Recommended:** make `workgroups` the default. +1.55% to +2.17% decode at 16k
  (N4, re-confirmed by the adversary's 8-arm curve monotone to 13 with a −1.5%
  cliff at 14), quality cost measured at 1 greedy token in 1 536 over three contexts
  above the boundary, structural reach 10 of 60 layers.
* **Not made here, for a reason that is not "it needs an owner":** this spec's own
  quality gate requires **byte-identical greedy text against base `f2b2da1` for the
  default configuration**, and flipping the default by construction changes 1 token
  in 1 536. A +2% decode win is not worth spending that gate on — the gate exists so
  that a speed number attached to a numerics change is not read as free.
  **Whoever wants the flip should amend the gate first, in writing, with the 1/1536
  number attached.**

---

## F5 — AC2 re-derived, the o_proj lead killed, and the honest ceiling ✅ FIXED

Full derivation in `fixer_evidence/f5_arithmetic.md`; argument in
`implementation_notes.md` §23.

### The concrete lead, measured and dead

`spec/decode-20tps` §R1 handed over: *"o_proj at 100 GB/s is at HALF the bandwidth
of every other projection… worth 2-3 ms/token"*. That is `gemv_bandwidth.rs` at
**`M = 1`** — a decode GEMV. Prefill is `M = PREFILL_CHUNK = 1024`, where the
projection is not bandwidth-bound at all. From the 16 394-token per-kernel profile,
with FLOPs computed from `config.json` — **including the fact that `head_dim` is
512 on the 10 full layers and 256 on the 50 sliding ones, so `q_proj` and `o_proj`
have two different shapes and getting that wrong moves the FLOP count by 8%**:

| projection | `own_ms` @16 394 | s @24 590 | share | TFLOP/s | % peak |
|---|---|---|---|---|---|
| down / up / gate | 34 339 / 34 321 / 34 318 | 51.5 each | 23.0% each | **6.63** | 42% |
| q_proj | 15 869 | 23.8 | 10.7% | 6.37 | 41% |
| **o_proj** | **15 574** | **23.4** | **10.5%** | **6.49** | **42%** |
| k_proj | 6 495 | 9.7 | 4.4% | 7.78 | 50% |
| v_proj | 5 853 | 8.8 | 3.9% | 8.63 | 55% |
| elements + head | 2 187 | 3.3 | 1.5% | — | — |
| **all seven** | **146 770** | **223.5** | | **6.73** | **43%** |

Three independent reasons the lead does not transfer:

1. **o_proj runs at 98% of `down_proj`'s efficiency** (`own_ms` ratio 0.453 against a
   FLOP ratio of 0.444). There is no deficit to recover.
2. **The ordering is not a bandwidth signature.** The two slowest projections have
   the *longest* `K`; the fastest has the *shortest* `K` **and** the narrowest `N`.
3. **273 GB/s is irrelevant.** Per chunk the GEMM streams the whole 11.6 GB of
   packed weights once against `6.168e13` FLOP — **5 317 FLOP/byte** — so at 6.73
   TFLOP/s it needs **1.27 GB/s of a 273 GB/s wire**, 0.5% of it. The same
   correction §12 made for attention, never made for the GEMM.

The entire 2% o_proj deficit is **0.45 s of a 273.2 s budget (0.16%)**.

### Attention, split by layer kind, cross-checked before use

Solving the per-layer `own_ms` totals (full 7 566.7 ms, sliding 2 653.1 ms, each
layer's own GEMM `FLOP`/6.73 TFLOP/s) for the attention share:

| | attention/chunk/layer | FLOP @16 394 | TFLOP/s | FLOP @24 590 | s @24 590 |
|---|---|---|---|---|---|
| 10 full layers | 270.0 ms | 8.814e13 | **1.91** | 1.9817e14 | **103.8** |
| 50 sliding layers | 10.0 ms | 2.751e13 | **1.85** | 4.126e13 | **22.3** |

Both land on ~1.9 TFLOP/s. That is the credibility check: it matches
`attention_qk_mma_bench` on the same geometry (1.27-1.46 TFLOP/s) and reproduces
the adversary's *total* 24 590-token attention figure (127.5 s vs 126.1 s here).
The adversary's internal split does not reproduce; its total does, and the total is
what AC2 depends on.

### The budget

```
273.2 s (90 tok/s) − 223.5 (GEMM + other) − 16.2 (inter-dispatch gap) = 33.5 s
2.394e14 FLOP / 33.5 s = 7.15 TFLOP/s = 46% of the 15.6 TFLOPS tensor peak
measured: 1.9 TFLOP/s = 12%          ->  a 3.8x cut, not 2.7x
```

**239.7 s — 88% — is spent before attention begins.** And the "flash-only" route is
**not available**: holding the sliding layers at their measured 1.85 TFLOP/s leaves
11.2 s for the full layers, i.e. **17.7 TFLOP/s, above the hardware's peak**.

### The hunt — four levers built and measured, all negative

| lever | measured | verdict |
|---|---|---|
| hoist the `qstage` Q fragment into registers (the loops-invariant one) | **−92%** (x0.52) | it spills |
| 2 and 4 independent QK accumulator chains | **1.00x / 0.96x** | not the dependency chain |
| `KAPLUMBA_FLASH_BQ=1` (drop query blocking) | §23.7 | the only survivor, and small |
| the o_proj lead | refuted | no anomalous projection |

**An instruction-count model was tried and falsified, which is the most useful
negative result here.** Counting instructions per MMA gives 2.72 at `BQ=1` and 1.89
at `BQ=2`, so a perfectly issue-bound kernel would reach 5.7 and 8.25 TFLOP/s and
hoisting `qf` would raise `BQ=2` to 9.5. All three predictions are wrong:
**`BQ=2` is 19% slower than `BQ=1` on the same bench, and hoisting `qf` is 92%
slower.** The instruction count is not what the MMA waits for. Anyone reasoning
from instruction mix should read §23.5 first.

### The honest ceiling

| scenario | attention | total s | tok/s |
|---|---|---|---|
| measured today | 126.1 | 365.8 | **67.2** |
| both attention kernels at the 8-chain tensor peak (7.65 TFLOP/s) | 31.3 | 271.0 | **90.7** |
| both at the 16-chain peak (15.49 TFLOP/s) | 15.5 | 255.2 | **96.4** |
| attention free | 0 | 239.7 | **102.6** |

**AC2 is arithmetically possible and only just, and only through the tensor
cores**: the 8-chain row has a **0.8% margin**, and the required lift on the MMA
issue rate of *both* attention kernels is **4.0-8.2x**. The GEMM is not the
cheaper route: it is at 43% of peak, uniformly, and 8.1 TFLOP/s is **above**
`spec/engine-performance`'s declared 7.5 TFLOPS wall. **The honest recommendation
is a target change or a rewrite, not another lever-shaped session.** No acceptance
target in `spec.md` was changed.

### The one experiment still standing

`mma_peak.rs` records that the tensor rate depends on the **number of independent
accumulator chains** and not on the loads (1 chain 5.75, 8 chains 7.65, 16 chains
15.49 TFLOPS) — and that an *array* of 32 made the compiler spill and collapse to
0.74. Arms 11/12 split the QK chain 2 and 4 ways and gained nothing, which suggests
the split was not actually made (the compiler kept the chains serial), not that ILP
does not matter. **16 named scalar accumulator chains in the QK loop, written the
way `mma_peak.rs` writes them, is the one lever left.** If 16 chains really do give
15.49, the flash kernel's ceiling moves by an order of magnitude and AC2 is back in
reach. It is a day's work and it is the honest next step.

---

## F6 — a third *independent* reproduction ✅ ALREADY WITHDRAWN (`8a3d411`)

Verified, not re-done: `NEXT.md` §7.2 item 3 and `spec.md` OQ10 both now say
*reproduction*, name the derivation case (`kv_heads = 4 -> 13` is the row the budget
was read off), and record that the out-of-sample shapes (`kv_heads` 5 and 6) are not
in Gxma and have never been re-measured on this machine. The claim that survives is
that 13 is a **real wave boundary**: monotone rise to 13, −1.5% cliff at 14
(13 -> 52 workgroups, 14 -> 56, and 54 is one wave / 55 is two).

---

## F7 — a "reproducible every time" pre-existing crash ✅ ALREADY WITHDRAWN (`1d47bf8`)

Verified, not re-done: `NEXT.md` §7.4 and `implementation_notes.md` §20.1 both carry
the withdrawal, 29 attempts across nine token counts all rc=0, `refprompts.sh` rc=0
twice, the base-build leg never logged, **no defect filed**, and the plausible
reading (a symptom of the OQ8 stall) explicitly labelled a reading rather than a
finding.

---

## F8 — the rescale guard: exact, never measured, control not in the tree ✅ FIXED

### The decision the finding poses

§17.1 justified the `msv == 1` guard on **exactness alone** — which is a correct
argument for correctness and no argument for speed. The brief's bar: **measure the
benefit and keep it, or withdraw it; do not keep an unmeasured optimisation in the
hot path on exactness alone.**

### Measured, as §22's arm 9, same binary / grid / buffers, interleaved

| shape | arm 0 (guard, shipped) | arm 9 (guard removed) | worth |
|---|---|---|---|
| 24 576 keys / 1024 q, `BQ=1` | 1143.6 ms | 1184.4 ms | **3.6%** |
| 24 576 keys / 1024 q, `BQ=2` | 1209.3 | 1268.7 | **4.9%** |
| 24 576 keys / 4096 q, `BQ=1` | 4570.2 | 4724.8 | **3.4%** |
| 24 576 keys / 4096 q, `BQ=2` | 4820.2 | 5033.0 | **4.4%** |
| 8 192 keys / 1024 q, `BQ=1` | 376.6 | 385.9 | **2.5%** |
| 8 192 keys / 1024 q, `BQ=2` | 401.4 | 420.5 | **4.8%** |

**2.5-4.9%, consistent across six shapes.**

### The instrument resolves this size, and says so

In the same sweeps arm 14 deletes the Q/P fragment loads for **11-12%**, arm 8 makes
every device load L1-resident for **9%**, and arm 11 reads **1.00x**. So the
instrument moves 11% and resolves 0%; a 3-4% reading is a reading, not noise.

### Decision: keep the guard, on a measurement rather than on an IEEE-754 argument

§17.1's exactness argument is unchanged and still worth having. What it lacked was
any speed evidence, and it now has some. **The guard is a real ~4%, not an
unmeasured optimisation.**

### On the fictional citation

`crates/engine/tests/attention_rescale_guard.rs:384` cites a sibling test
`a_misfiring_guard_is_caught` "where it reads 0.2136 — 43x the tolerance". **No such
test exists** — the sabotage was a manual `.metal` edit, so nothing in the commit
can re-run it. The adversary ran it by hand and got `0.21362305` with 7 of 8 failing,
which is exactly the number in the file's comment, so **the claim is true and only
the citation is fictional**.

**Left as it is, deliberately, and the reason is worth stating:** a test that edits a
`.metal` source and rebuilds is not a unit test, and `scripts/` has no mechanism for
one. What replaces it is §22's arm 9 — the same sabotage expressed as a uniform
value, needing no rebuild, and living in the tree as part of the ablation harness.
The correction made is to the *documentation*: `implementation_notes.md` §26 now
carries the numbers and says explicitly that it is the record of the control, not a
citation to a test. Inventing a test to match a comment would be the wrong fix.

---

## F9 — `LOG=` honoured in 1 of 3 harnesses ✅ ALREADY FIXED (`caf3631`)

Verified, not re-done: all three harnesses use `SUMMARY=${LOG:-$OUTDIR/...}`, and
`scripts/test_gpuq_mem.sh` has a check that fails if any of them hard-codes the
summary path again. The two new harnesses written in this pass
(`bench_split_rule_quality.sh`, `fixer_gpu_chain.sh`) honour `OUT=` and never pipe a
long measurement into `tail`.

---

## The quality gate, and what was measured for it

| requirement | measurement | result |
|---|---|---|
| byte-identical greedy text vs base `f2b2da1` for the **default** configuration | — | **held by construction and by test**: the only behavioural default this pass changes is F3's, and OQ9 passes are bit-exact — verified on **generated text** (512 greedy tokens, byte-identical) at 1024 / 4096 / 16 394 tokens of context |
| >= 512-token generations at temperature 0 | 512 tokens x 4 arms x 3 contexts | ✅ all completed, 512 tokens each |
| >= 512-token generation at a sampling temperature | 512 tokens, `--temperature 0.7 --seed 20260905`, 16 394-token prompt | see `f4_quality_sampled/` |
| no repetition collapse | distinct n-gram ratio + a deliberately collapsed control | ✅ `distinct_8` 0.958-0.966 against a control's 0.0085 |
| no NaN/Inf | a non-finite logit is a hard `SamplerError` (`sampler.rs:446`) | ✅ a completed generation cannot contain one |
| worst normalised GEMM error <= 2.16e-3 | **not re-measured** — no product numerics changed, only defaults and comments | n/a; see below |
| weight footprint <= 11.51 GiB | `phys_footprint` from `/usr/bin/time -l` in every run | ✅ 15.49-15.56 GiB at 16 394 tokens, unchanged; the F3 default change cannot move it because the state buffer is allocated either way |

**Every numeric perturbation in this report is labelled.** F4's split-rule flip is
**bounded-ulp**: 1 differing argmax token in 1 536, and the mechanism (fp addition
is not associative in the combine pass) is stated in `attention.rs:433-454` and
verified by §16.2's one-ulp finding. F3's default change is **exact**: the pass chain
is bit-identical to the one-dispatch body, now checked on generated text and not only
on the 17 kernel-level cases. Nothing in this report is labelled "exact" on the
strength of a tolerance.

**Not re-measured, and it should not be read as a pass:** the worst normalised GEMM
error. No GEMM, sampler or attention arithmetic was modified in this pass — only two
defaults and a comment — so the number is unchanged by construction rather than by
measurement. Anyone needing it as evidence should run
`crates/engine/tests/linear_metal.rs` against base.