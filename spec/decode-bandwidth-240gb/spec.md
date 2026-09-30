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

### 1. Per-matvec fixed latency: **59 us**, worth ~25 ms/token

Measured as the *intercept* of a three-point affine fit, which is the honest way
to separate it from the streaming term — one matvec per command buffer with a
wait measures the CPU/GPU round trip, not this (`gemv_shape.rs --selftest`,
AFFINE check, 0.0% residual on three sizes at in=5376):

```
t = 58.9 us + 0.00608 ms/MB      -> 164.6 GB/s streaming, 58.9 us fixed
```

Reproduced at 164.6 GB/s / 58.9 us. The spec previously recorded **79 us**; the
corrected figure is **59 us**, and 420 matvecs x 59 us = **24.8 ms of a 96 ms
token**.

The affine fit is a *much* better instrument than the old one, and the reason is
worth recording: requiring "2x the bytes must take 2x the time" fails at 1.62x
on a real fixed cost, and that failure is how a correct instrument gets declared
broken. The right statement is that time is **affine** in bytes.

Still not decomposed: how much is first-touch page walk, threadgroup launch and
loop prologue, or the tail drain. The natural fixes — more tiles per thread,
deeper unroll — are all measured dead, so the decomposition has to come first.
**Note this is now a question about `linear_gemv_ishizuki_k*`, not the EXL3
family.**

### 2. Load-traffic-per-useful-byte: why 3 bpw is 40% slower than 6 bpw

**This is the mechanism, and it is measured on the live kernel.** Holding the
weight bytes constant at ~43.4 MB and varying *only* the bit rate
(`gemv_shape.rs --only "bit rate"`, medians of 9, `KAPLUMBA_TRACE_GEMV`
confirming the kernel):

| rate | out_features | MB | GB/s | lane-loads per useful byte |
|---|---|---|---|---|
| 3 bpw | 21504 | 43.4 | **132.5** | 12 B -> 3 B = **4.0x** |
| 4 bpw | 16384 | 44.0 | 179.8 | 12 B -> 4 B = 3.0x |
| 6 bpw | 10752 | 43.4 | 164.5 | 12 B -> 6 B = 2.0x |
| 8 bpw | 8192 | 44.0 | 176.3 | 12 B -> 8 B = 1.5x |

Identical bytes. The rate is the only thing that moved, and it moves bandwidth
by 1.36x.

**The cause is in `exl3_decode8_mcg` / ishizuki's window: it reads a fixed three
ring words per lane — 12 bytes — whatever `KA` is, and 8 codes at `KA` bits each
are only `KA` bytes of information.** So the request amplification against
distinct bytes is `12/KA`: **4x at 3 bpw, 2x at 6 bpw.** The 96 B window is
sector-perfect (32 disjoint windows sit at the ceiling, 246.9-248.4), so DRAM
traffic is correct and this is not wasted bandwidth — it is wasted *requests*.

**3 bpw is 410 of the 411 tensors in this checkpoint.** That is the whole
remaining gap, and it is why the affine asymptote at 3 bpw is 164.6 GB/s while
`lm_head` at 6 bpw reaches 252.

Why the obvious cures do not work, all measured: `uint4` (0.939x — the fast
path fires for only 8/32 lanes because of the ring wrap), a 32-bit window
rewrite (+2.4% at KA=3 but -3.7% at KA=1), and `N_R0` (+6.8% best case,
`T2`). All of them add *streams* or widen *loads*; none of them reduces the
requests-per-useful-byte, which is the thing that is actually limiting.

**The untried direction** is to make a lane's window *word-aligned*, so 3 words
can become 1: a lane needs 24 bits of a 32-bit word for its 8 codes at KA=3, so
one 32-bit load could serve 8 codes instead of three. The blocker is that lane
`l`'s codes start at bit `(l*8+1)*KA + ring - 16`, which is not 32-aligned — that
is *why* three words are read. The EXL3 on-disk layout is fixed (the spec already
records that no repacking reduces bytes), so this cannot be fixed by rewriting
the file; it would have to be fixed by changing which codes a lane owns, i.e. the
`r0`/`i` index arithmetic in the window setup. **That is the next task.**

### 3. Warp-level distinct-byte yield: 4.00x redundancy

Per warp-iteration, `exl3_decode8_mcg`'s three ring loads issue **96 lane-loads**
(32 lanes x 3 words) against **24 distinct words**. The window itself is
sector-perfect, so the cost is requests, not sectors — this is the same fact as
item 2, seen from the warp's side rather than the lane's.

`N_R0 = 4` rows per thread measured **+88%** (101 -> 190 GB/s) on the
*pre-specialisation* kernel, and **+6.8% at best** on the current one (`T2`).
That question is now closed; see the closed-lever table.

## Where this leaves the target

Per-token budget at 10.0 tok/s, from the first honest per-decode-step profile
(`KAPLUMBA_DECODE_PROFILE=kernel`, landed in `a58fa7b`; own-time sums to 97.1 ms
against a 96.0 ms step span, so the attribution is sound):

```
7x60 layer matvecs (10.875 GB)   85 ms   at 102.7 GB/s aggregate
  of which gate/up/down (65%)   68.5 ms at 113.9 GB/s
lm_head (1.057 GB)                5.3 ms at 198 GB/s
attention, norms, rope, gelu,
  residual, cache, sampling       7.4 ms
```

The three big shapes are the prize: at 240 GB/s they would take 34 ms instead of
68.5, and the whole token would be **~53 ms = 19 tok/s**. For **16 tok/s**
(62.5 ms) the matvecs must reach ~58 ms, i.e. **1.5x**, which is exactly the
3-vs-6 bpw gap measured in item 2. For **20 tok/s** the requirement is 263 GB/s
and it is not available.

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
