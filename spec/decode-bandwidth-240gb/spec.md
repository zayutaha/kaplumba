# Decode: sustaining 240 GB/s and what actually costs it

Companion to `spec/decode-19tok/spec.md`, which set the roofline. This document
records what has been **measured** about the 240 GB/s ceiling, what has been
**disproved**, and the two mechanisms that are still live. Read it before
proposing any bandwidth work — the obvious ideas are all closed, and five of
them were closed by finding bugs in the instruments that produced them.

## The goal, stated precisely

Sustain **240 GB/s** of streaming reads in the decode path, on an M4 Pro whose
nominal figure is 273 GB/s.

**240 is 88% of nominal, and that is a real target, not a stretch.** Apple
silicon's measured DRAM efficiency is 82-86% across four generations, and
NVIDIA's GB10 — an *identical* 256-bit LPDDR5X-8533 subsystem — measures
229-230 GB/s. We measure 239-244, so we are already at the top of the published
band. Pushing past it means finding a mechanism, not tuning harder.

**20 tok/s is not on the table and should stop being a target.** It needs
13.145 GB / 50 ms = **263 GB/s**, above the measured ceiling. And **250 GB/s
needs 12.088 GB / 50 ms of matvec plus everything else**, which is likewise
above it. Both are physically unavailable on this part at this bytes-per-token.
The only lever that changes this is **moving fewer bytes**, which is a
quantisation decision, not an implementation one.

## ⚠️ Read this first: the kernel the numbers below are about

`LinearKernels::encode_gemv` dispatches in this order:

1. **`linear_gemv_ishizuki_kN`** — whenever the rate is an integer in 1..8 and
   `packed_offset % 4 == 0`;
2. the **`linear_gemv_exl3_kN{,_u}{,_mcg}`** family — otherwise.

For this checkpoint, `KAPLUMBA_DEBUG_OFFSETS` reports **410 of 411 tensors are
4-byte aligned**, so the ishizuki arm takes all 60 layers' projections.
`KAPLUMBA_TRACE_GEMV=1` on a real run prints exactly two lines:

```
GEMV-DISPATCH linear_gemv_ishizuki_k3
GEMV-DISPATCH linear_gemv_exl3_k6_u_mcg
```

So **the entire `_mcg` family — the codebook specialisation of `3439d41`, and
the NR0 axis of `bcc95b8` — runs for `lm_head` alone**, which is 8.9% of a
decode token's bytes. Everything else runs on `linear_gemv_ishizuki_k3`.

This does **not** make `3439d41` wrong: ishizuki's `k3` also carries the
codebook as a compile-time constant (`constexpr int CB = 1`), so the
specialisation idea was already in the live kernel when that commit landed, and
the 8.1 -> 9.2 tok/s step stands. But it does mean that any EXL3-family
bandwidth number in this repository is a number about `lm_head`, and that the
next lever has to be aimed at ishizuki.

`examples/which_gemv.rs` prints the dispatch decision for every (rate,
alignment) without loading the model.

## What is settled, with the numbers

### The ceiling: 239-244 GB/s

Measured with a gated harness (`crates/engine/examples/conc.rs`, self-testing,
checksum-verified against an independent CPU reference, 4110-config correctness
gate) on a quiet box:

```
median 239.8   best 246.8   worst 230.0   spread 1.07x   (88% of 273)
per-run medians: 238.8 239.0 239.7 239.8 240.7 240.9 243.8   (spread 1.02x)
```

Reproduced 239.6-240.6 for the `lm_head` shape. This is the number everything
else is measured against.

### Five earlier bandwidth figures were artifacts

This matters more than the current number, because each was believed and acted
on:

| figure | what it really was |
|---|---|
| **55 GB/s** | `bw.rs` allocated a multi-GB buffer and streamed it **without ever writing a byte**. Untouched zero pages are not a memory measurement. This number became the "achievable bandwidth" denominator for a whole round of experiments. |
| **248 GB/s** | `bw.rs` reported **min-of-passes**. On a shared box that selects the one pass that ran while a competitor was idle. The same binary swung 79.9 -> 230.5 GB/s in 10-second intervals while load *fell*. |
| **254 GB/s** | `layer_bench.rs:78` hardcodes `rate: Integer(3)`, so its "lm_head" was a **3 bpw / 528 MB stand-in**, not the real 262144-wide 6 bpw head. The real head measures 239.6. |
| **220 GB/s** | Real, but the *old probe shape* (4 B scalar, 20480 threads). The gated harness's shape reads 237-241 at the same sizes. The ~7% gap was kernel shape, not the memory system. |
| **131,512 GB/s** | `gemv_shape.rs` divided by `passes` and then formatted the result as milliseconds, so seconds were labelled as ms. Its own self-test's PLAUSIBLE check caught it on the first run. |

**Rule that follows:** an instrument must be able to fail before its number may
be used. Every harness in this project that reported bandwidth was wrong first.

### The wall is DRAM bytes, not instructions

The decisive experiment: read at step 1, 2 and 4 while the hardware fetches
every sector, consuming a fraction of each.

| step | useful GB/s | **sector GB/s** | share used |
|---|---|---|---|
| 1 | 225.1 | **225.1** | 100% |
| 2 | 113.0 | **226.0** | 50% |
| 4 | 57.1 | **228.5** | 25% |

Flat sector rate, halved useful rate. **The hardware moves ~226 GB/s of sectors
no matter how many of those bytes the program uses.** The limit is below the
instruction stream.

Instruction issue is not the constraint either. Bandwidth is flat to ~16
uop/byte and collapses beyond; two independent methods agree on the knee (a
bending curve, and 3.95e12 uop/s / 238e9 B/s = 16.6 uop/B). **EXL3 runs at
1.35 uop/byte, 8x under the knee.** The ALU is not the cost.

## What is closed — do not re-open

| lever | result | why |
|---|---|---|
| **Concurrency** | flat 0.08M-2.62M loads in flight (32x, 1.06x) | request-throughput wall, not latency. Little's law is the wrong model. |
| **Load width** | 8B / 16B / 32B all ~227-238 | width is not the mechanism. |
| **Independent address streams** | 1 -> 64: 216.3 vs 214.1 | no effect. So 23 arena segments are not a problem. |
| **Working-set size** | rises to 4 GB then plateaus; 1 GB is the *slowest* large size | the old "248 at 1 GiB" was not a size artifact. |
| **Contiguity / DRAM row locality** | 4 KiB -> 3 MiB all 243-253, spreads overlapping | **zero effect. Weight layout is not an optimization target.** |
| **Wave tail** | 336 groups (93.3% wave) 237.4 -> 16384 (99.9%) 242.2 | the 2% is inside the 1.02-1.08x spreads. `KAPLUMBA_GEMV_MAX_GROUPS` 64/128/336 makes every shape monotonically *worse*. |
| **Per-dispatch / encoder overhead** | **0.08%**. GPU busy 99.9% of span over 1260 dispatches/token. `MAX_DISPATCHES_PER_BUFFER` 50->1000 and `MAX_IN_FLIGHT` 1->16 all give 107.5-107.9 (0.4% spread) | the encoder is not a lever. A previous "1.42 s stall" was legitimate back-pressure, not idle time. |
| **Multi-tile per thread (`N_R0`)** | **T2, re-measured on the current kernel: dead.** gate 129.4/137.0/137.3/132.0, down 123.4/125.9/117.1/98.6, q 104.6/109.5/105.4/95.0, lm_head 252.2/251.5/247.4/248.2 for NR0 = 1/2/4/8 | the spec's +88% was on the pre-specialisation kernel. Best case now +6.8%, not paid on `down_proj`. Reproduced at +6.3/+6.8%. |
| **`tk` unroll** | x2 1.008x, x4 0.889x | register pressure. |
| **`uint4` for the three window loads** | 0.939x | the fast path fires for only 8/32 lanes because of the ring wrap, so it pays a branch and buys nothing. |
| **32-bit window rewrite** | +2.4% at KA=3, **-3.7% at KA=1** | not worth it. |
| **Page faults** | 20/token, 0.003% of bytes | irrelevant. |
| **Compression** | 18G wired, ~450 MiB compressor, weights pinned | not stealing bandwidth. The real risk is only ~100 MiB of headroom. |
| **Power / thermal** | sustained 200.5 **beat** burst 117.5; `pmset -g therm` clean before and after 180 s | not throttled. (`powermetrics` needs sudo; this rests on pmset/top/sysctl.) |
| **Batching** | `B != 1` throws in oMLX; a prior batching attempt regressed to 0.913x | not applicable; single-user batch-1 is a hard requirement. |
| **`MTLResidencySet` pinning** | unreachable — `newResidencySetWithDescriptor:` SIGSEGVs three times on this binding | untested, not disproved. |

## What is NOT closed, and is the actual work

### 1. ~~Per-matvec fixed latency~~ — *CLOSED, and it was never a lever*

The spec carried **79 us**, this session corrected it to **59 us**, and the
measured answer is **~155 us per command buffer, 42 buffers per decode step**.
All three were wrong, and wrong in three different ways:

| | claimed | actual |
|---|---|---|
| quantity | per matvec | **per command buffer** |
| magnitude | 79 us, then 59 us | **117-203 us** |
| multiplier | 420 matvecs | **42 buffers** |

The 59 us was a fitting artifact: a three-point fit of time against bytes with
the pass count scaled as `1/bytes`, so a per-buffer cost is amortised differently
at every size and lands in the **slope**. Refitting today's numbers the same way
gives **8.1 us** — a 7x disagreement from the same instrument hours earlier,
which is what exposed it. Measured properly (vary *pass count* at fixed bytes,
`gemv_shape --fixed`) the overhead is flat across all five shapes, which is the
signature of a real fixed cost: a page-walk term would scale with the buffer, a
launch term would not.

Priced: **42 × 155 us = 6.5 ms/token**, which is 29% of the 22.3 ms *host* wall
but only **7.3% of the 89.5 ms GPU span** — and it is host time, spent while the
GPU has 67 ms queued that the host is not waiting on. **The host runs 4.01x ahead
of the GPU.**

So the cost is real, it is most of the host's budget, and it is **completely
invisible in the token time**. Closed. `MAX_DISPATCHES_PER_BUFFER` was already
measured (50→1000, 0.4% spread) and would only move time that is off the
critical path.

### 2. Load-traffic-per-useful-byte — *PARTIALLY CLOSED, and the rest is T7*

**This was the mechanism, measured on the live kernel.** Holding weight bytes at
~43.4 MB and varying *only* the bit rate:

| rate | out_features | MB | GB/s (pre-T6) | lane-loads per useful byte |
|---|---|---|---|---|
| 3 bpw | 21504 | 43.4 | **132.5** | 12 B -> 3 B = **4.0x** |
| 4 bpw | 16384 | 44.0 | 179.8 | 3.0x |
| 6 bpw | 10752 | 43.4 | 164.5 | 2.0x |
| 8 bpw | 8192 | 44.0 | 176.3 | 1.5x |

> **These four numbers are not currently trustworthy, and the table is retained
> only to show what the question looks like.** Re-run after T6, the same sweep
> gives 3 bpw 146.9 (+10.9%), 4 bpw 153.7 (-14.5%), 6 bpw 182.0 (+10.6%), 8 bpw
> 162.3 (-7.9%) — and **6 and 8 bpw cannot have changed from T6**, since no `_w2`
> kernel is emitted above 4 bpw. They are controls, and they moved in opposite
> directions by ~10%. Within-sweep spreads are 2.2x-3.7x. The sweep measures one
> shape to completion at a time and therefore cannot see box drift, and the drift
> is the same size as the effect. **Fixing this instrument is T7's first step**
> (`spec/decode-bandwidth-handover/tasks.md` H1); the `--w2` and `--nr0` sweeps
> in the same binary already interleave correctly and are the pattern to copy.
>
> What *is* trustworthy is T6's own A/B, which is interleaved with the baseline
> re-measured every round: +6.8% to +11.2% on five shapes, and +7.0% end to end
> over 5 paired runs.

**The cause, and the fix.** The kernel read a fixed three ring words — 12 B —
per lane whatever `KA` is, while 8 codes at `KA` bits are only `KA` bytes. But
the extraction only ever touches A0 bits `[48-7*KA, 63]`, and the third word
contributes only to bits `[0, sh)`. At KA=3 the largest `sh` over 32 lanes is
**27** and the lowest used bit is `48-21 = 27`, so **the third load is never
read**. Deleting it is bit-exact by construction.

`TWO_WORD_SAFE_RATES` in `scripts/gen_exl3_gemv.py` *derives* the safe set rather
than listing it, so a rate cannot be added without the proof. Emitted 1-4 bpw;
excluded 5, 6 (used bits below max `sh`) and 7, 8 (the `B0` arm is live). **6 bpw
is `lm_head`, so the one shape where the lever does not apply is the one already
at the ceiling.**

**Result: +7.0% end to end**, bit-exact, all five reference prompts identical.

**And why that is only half the story.** 3→2 loads is a 33% cut in lane-loads and
bought 7%, because the window is **sector-perfect**: word 3 was an L1 *request*,
not a DRAM sector, and the sector-rate experiment says the wall is DRAM. So
request count is **not free, but it is not the wall either** — and the residual
3-vs-6 bpw gap is unexplained. That is **T7**, and it is the highest-value open
question here.

### 3. Warp-level distinct-byte yield — *CLOSED*

Per warp-iteration the three ring loads issued **96 lane-loads** against **24
distinct words** (4.0x). This is item 2 seen from the warp's side, and T6 closed
the actionable half of it. `N_R0 = 4` measured +88% pre-specialisation and
**+6.8%** on the current kernel (`T2`), which is now a dead lever.

### 4. The 3-vs-4 bpw residual — *CLOSED, and it is request-bound (T7)*

**Measured with a trustworthy instrument** (`gemv_shape --rate-interleaved`,
every arm interleaved, 6 bpw carried as a control with 13% drift):

| rate | window | GB/s |
|---|---|---|
| 3 bpw | 3-word | 150.9 |
| 3 bpw | **2-word (shipped)** | **165.7** |
| 4 bpw | 2-word | 216.4 |
| 6 bpw | control | 215.7 |
| 8 bpw | 3-word | 217.9 |

**4, 6 and 8 bpw are all at 215–218 GB/s** — flat across a 2x range of rate.
They are DRAM-saturated. Only 3 bpw is slow, by 1.306x.

Two mechanisms both predicted 1.333x and **a bandwidth table cannot separate
them**: lane-load amplification per useful byte is 2.67x at 3 bpw vs 2.00x at
4 bpw, and `decode_state` calls per useful byte is *also* 2.67 vs 2.00 (8 codes
per tile is rate-invariant). The `_nod` probe — the verbatim kernel with
`decode_state` replaced by a one-AND stub, one asserted substitution, identical
memory traffic, garbage output, never dispatched — separates them:

| rate | real | no-decode | effect of removing decode |
|---|---|---|---|
| 3 bpw | 167.3 | 122.7 | **−26.7%** |
| 4 bpw | 217.1 | 206.3 | −5.0% |
| 8 bpw | 222.5 | 219.2 | −1.5% |

**Removing the decode made every rate slower.** Decode arithmetic is not the
cost — a cheaper decode cannot be the fix, because the free decode is slower
than the real one. So the gap is **request/latency-bound**: T6's mechanism, and
the residual 2.67x lane-load amplification is **structural** to the EXL3 lane
layout, whose on-disk format is fixed.

This closes T7 and the handover's H2a/H2b/H2c. No kernel-level lever for this
gap remains identified.

### 5. What is left: bytes, not speed — *the arithmetic, stated once*

At the measured 165.7 GB/s for 3 bpw:

| | ms/token | tok/s |
|---|---|---|
| measured now | 93.1 | 10.74 |
| if 3 bpw reached 4 bpw's 217 GB/s | 62.4 | 16.0 |
| if 3 bpw reached pure-streaming 240 GB/s | 57.1 | 17.5 |

**19 tok/s needs 52.6 ms.** With 7.4 ms of non-matvec work, the matvec has
45.2 ms for 11.932 GB = **264 GB/s** — above this box's measured 239–244 GB/s
ceiling. Working back: **19 tok/s requires 1.888 bpw.**

- **1.9 bpw** — below anything known to work on this model (oMLX runs 2.25,
  ishizuki benchmarks 2.0). A different, smaller checkpoint; a real quality loss.
- **2.0 bpw** — ~18.1 tok/s by the model above, and a published operating point.
  Still a different checkpoint; still the user's quality call.
- **16–17.5 tok/s at 3.0 bpw** — requires the 3 bpw gap to close, which the probe
  says is not reachable by any kernel change identified so far.

**The next real decision is a quantisation one, and it belongs to the user.**

## Where this leaves the target

Per-token budget, from the first honest per-decode-step profile
(`KAPLUMBA_DECODE_PROFILE=kernel`, landed in `a58fa7b`) **updated with T6**:

```
                                      pre-T6    post-T6
7x60 layer matvecs (10.875 GB)         85 ms     ~78 ms
  gate/up/down (65% of bytes)       68.5 ms    ~62 ms   (113.9 -> ~126 GB/s)
lm_head (1.057 GB, 6 bpw)             5.3 ms     5.3 ms  (198 GB/s, unchanged)
attention, norms, rope, gelu, cache     7.4 ms    7.4 ms
command-buffer overhead (42 x 155 us)      -     6.5 ms  (host time, hidden)
                                    -------   -------
measured GPU span per token          96.0 ms   89.5 ms  = 11.2 tok/s
host wall per token                  23.8 ms   22.3 ms  (host runs 4.01x ahead)
```

The three big shapes are still the prize. At 240 GB/s they would take 34 ms
instead of 62, and the whole token would be **~53 ms = 19 tok/s**. For **16 tok/s**
(62.5 ms) the matvecs must reach ~58 ms, i.e. **1.5x** — exactly the 3-vs-6 bpw
gap, of which T6 recovered about half. For **20 tok/s** the requirement is
263 GB/s and it is not available.

## Requirements for any future bandwidth work

1. **No result without a fail-first instrument.** Every bandwidth harness in
   this project was wrong before it was verified: one never wrote its buffer,
   one reported a wrong CPU reference twice, one read a 1024-slot ring against
   an 8192-key byte count (claiming 20,340 GB/s), one read an L2-resident buffer
   (claiming 1,592 GB/s), one reported min-of-passes, and one divided by its
   batch count and labelled seconds as milliseconds (claiming 131,512 GB/s).
2. **Median, not best.** Report median AND best AND worst AND spread, n>=7,
   interleaved A/B. Headline the median. On a quiet box the worst sample of a
   clean run is routinely 1.6x the best, so the *median* is what has to be clean
   and the spread has to be *visible*.
3. **Warm the GPU to steady state, and check the clock afterwards.** Cold clocks
   are up to 7x slower. Re-measure the first shape at the end of a sweep; if the
   two disagree, every number between them was taken on a moving clock.
4. **One GPU-touching agent at a time.** Contention produced the two largest
   errors of the day. `gpuq status` gates; a model load works at ~6.7 GiB
   "available" with 0 swap, so do not wait for a 14 GiB headline.
5. **Free memory per configuration.** A probe holding 3.8 GB without releasing
   drove the box to 3.7 GiB free and 573 MB of swap; swapped measurements are
   worse than none.
6. **`peak_resident_pages_bytes` is not a footprint.** It is `ru_maxrss`, which
   reported 6.23 GiB where the kernel charges 11.69. Use `footprint -p <pid>`,
   which quantises to 3 significant digits.
7. **Prove each test has teeth**, and remember three specific ways a coverage
   claim here can be vacuous:
   - a bit-exactness generator with an LCG of period 8, where every index
     perturbation that was a multiple of 8 was invisible;
   - FMA contraction, which is not stable, so two variants can each match alone
     and differ combined (30818/32768 did) — and, worse, a change applied to
     *every* kernel moves them all together, so a test that compares variants
     **against each other** passes. Compare against a committed reference.
   - a test that compares a kernel's output against a baseline **rebuilt from the
     same edited source**, which is why `tests/decode_matvec_golden.rs` exists
     and holds a digest rather than recomputing one.
8. **Know which kernel you are measuring.** See the warning at the top.

## Open questions

- **Can a lane's window be made word-aligned at KA=3**, so 3 loads become 1 and
  the 4x request amplification drops? This is the only lever left that attacks
  the measured mechanism rather than adding streams beside it. The blocker is the
  `r0`/`i` index arithmetic, and whether that can be changed without changing
  which *weights* a lane reads — because if it changes which codes a lane owns,
  the accumulation order changes and bit-exactness has to be re-earned.
- Is the 59 us fixed term first-touch page walk, launch/prologue, or tail drain?
  The answer decides whether the fix is in `linear.metal` or in how weights are
  mapped.
- Can `MTLResidencySet` pinning be made to work at all on this binding? It
  would pin pages we already have, at no memory cost, and no measurement
  exists either way.
- Is there any mechanism at all for >244 GB/s, given the sector-rate result says
  the DRAM wall is ~226? If the answer is no, then the only route to 20 tok/s
  is fewer bytes.
