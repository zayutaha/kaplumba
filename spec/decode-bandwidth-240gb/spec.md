# Decode: sustaining 240 GB/s and what actually costs it

Companion to `spec/decode-19tok/spec.md`, which set the roofline. This document
records what has been **measured** about the 240 GB/s ceiling, what has been
**disproved**, and the two mechanisms that are still live. Read it before
proposing any bandwidth work — the obvious ideas are all closed, and three of
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

### Four earlier bandwidth figures were artifacts

This matters more than the current number, because each was believed and acted
on:

| figure | what it really was |
|---|---|
| **55 GB/s** | `bw.rs` allocated a multi-GB buffer and streamed it **without ever writing a byte**. Untouched zero pages are not a memory measurement. This number became the "achievable bandwidth" denominator for a whole round of experiments. |
| **248 GB/s** | `bw.rs` reported **min-of-passes**. On a shared box that selects the one pass that ran while a competitor was idle. The same binary swung 79.9 -> 230.5 GB/s in 10-second intervals while load *fell*. |
| **254 GB/s** | `layer_bench.rs:78` hardcodes `rate: Integer(3)`, so its "lm_head" was a **3 bpw / 528 MB stand-in**, not the real 262144-wide 6 bpw head. The real head measures 239.6. |
| **220 GB/s** | Real, but the *old probe shape* (4 B scalar, 20480 threads). The gated harness's shape reads 237-241 at the same sizes. The ~7% gap was kernel shape, not the memory system. |

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
| **Multi-tile per thread** | T=2 0.98x, T=4 0.87x, T=8 0.64x | register pressure in `v[T][8]` cancels the MLP gain. |
| **`tk` unroll** | x2 1.008x, x4 0.889x | same cause. |
| **`uint4` for the three window loads** | 0.939x | the fast path fires for only 8/32 lanes because of the ring wrap, so it pays a branch and buys nothing. |
| **32-bit window rewrite** | +2.4% at KA=3, **-3.7% at KA=1** | not worth it. |
| **Page faults** | 20/token, 0.003% of bytes | irrelevant. |
| **Compression** | 18G wired, ~450 MiB compressor, weights pinned | not stealing bandwidth. The real risk is only ~100 MiB of headroom. |
| **Power / thermal** | sustained 200.5 **beat** burst 117.5; `pmset -g therm` clean before and after 180 s | not throttled. (`powermetrics` needs sudo; this rests on pmset/top/sysctl.) |
| **Batching** | `B != 1` throws in oMLX; a prior batching attempt regressed to 0.913x | not applicable; single-user batch-1 is a hard requirement. |
| **`MTLResidencySet` pinning** | unreachable — `newResidencySetWithDescriptor:` SIGSEGVs three times on this binding | untested, not disproved. |

## What is NOT closed, and is the actual work

### 1. Per-matvec fixed latency: ~79 us, worth ~33 ms/token

Measured per shape (GPU clock, n=7, one matvec per command buffer, idle
verified). The signature is unmistakable:

| shape | bytes | ms | GB/s |
|---|---|---|---|
| v_proj | 4.13 MB | **0.107** | 39 |
| k_proj | 8.26 MB | **0.107** | 39 |
| q_proj | 10.84 MB | 0.148 | 73 |
| o_proj | 16.52 MB | 0.182 | 91 |
| down_proj | 43.35 MB | 0.344 | 126 |
| gate/up | 43.35 MB | 0.336 | 128 |

**k_proj reads twice the bytes of v_proj and takes the same time.** Subtracting
the streaming time leaves a fixed ~79 us per matvec, identical for in_features
5376 vs 21504 and for out_features 5376 vs 21504, so it does not scale with
work. It is 54-75% of the narrow shapes and **420 matvecs x 79 us = 33 ms of a
109 ms token.**

Not yet decomposed: how much is first-touch page walk (a warm second pass is
untested), threadgroup launch and loop prologue, or the tail drain. The natural
fixes — more tiles per thread, deeper unroll — are all measured dead, so the
decomposition has to come first.

### 2. Warp-level distinct-byte yield: 4.00x redundancy

Per warp-iteration, `exl3_decode8`'s three ring loads fetch **96 distinct bytes
(24 words)** from 96 lane-loads, because lane `l` reads words `l+31, l+32, l+33`
and neighbours overlap. The 96 B window itself is sector-perfect: 32 disjoint
windows sit *at* the ceiling (246.9-248.4). Only the real decoder read costs
anything, and only **-13%** (214.4, reproduced 214.0).

**`N_R0 = 4` rows per thread measured +88%** (101 -> 190 GB/s), peaking at 4
(NR0=8 loses 13% to register pressure), with rows 32 KiB apart; *adjacent* rows
lose 4-11%. This is llama.cpp's shape: 4 independent rows, hoisted row pointers
so all four issue before the first FMA (`ggml-metal-impl.h:32-33`,
`kernels/mul_mv.metal:250-255,285-287`).

**Caveat that decides whether it is still worth doing:** the 101 GB/s baseline
was the **pre-specialisation** kernel. The matvec is now considerably faster,
so the +88% may be partly spent. It must be re-measured on the current kernel
before being believed, and it is a different axis from the dead multi-tile
experiment: that multiplied tiles *within one row's k-loop* and lost to
`v[T][8]` register pressure; this multiplies *rows*, changing the accumulator
count and the reduction.

### 3. The codebook branch — closed, and the template for how to look

`decode_state` took `codebook` as a **runtime** argument and branched on it 8
times per lane inside the unrolled loop. The codebook is a per-tensor constant.

```
loads only                      212.2 GB/s   70.6% of base time removed
production                       62.4
codebook -> compile-time const  163.0        61.7%
branch hoisted, still runtime   128.5        51.4%
no decode at all                134.3        53.5%
```

**2.61x, bit-exact.** The subtlety worth carrying forward: removing the 6
decode ops *in addition to* removing the branch does **not** help (134.3 <
163.0). The ops are free; **the branch is what serializes them.** So the lesson
is not "the decode is expensive" — it is "look for a per-tensor constant being
tested per element."

## Where this leaves the target

Per-token budget at 10.0 tok/s (109 ms), matvecs ~93%:

```
7x60 layer matvecs (12.04 GB)   ~101 ms   at ~107 GB/s aggregate
lm_head                          3.1 ms   at 169-240 GB/s
input transform (240/tok)        now ~1 ms after the simdgroup rewrite
rms_norm, attention, rope,
 residual, sampling, gaps         ~4 ms
```

For **16 tok/s (62.5 ms)** the matvecs must reach ~58 ms, i.e. **1.8x**,
i.e. the big shapes 125 -> 200 GB/s *and* the small ones off the 79 us floor.
For **20 tok/s** the requirement is 263 GB/s and it is not available.

## Requirements for any future bandwidth work

1. **No result without a fail-first instrument.** Every bandwidth harness in
   this project was wrong before it was verified: one never wrote its buffer,
   one reported a wrong CPU reference twice, one read a 1024-slot ring against
   an 8192-key byte count (claiming 20,340 GB/s), one read an L2-resident buffer
   (claiming 1,592 GB/s), one reported min-of-passes.
2. **Median, not best.** Report median AND best AND worst AND spread, n>=7,
   interleaved A/B. Headline the median.
3. **Warm the GPU to steady state.** Cold clocks are up to 7x slower and have
   produced several false results in this project's history.
4. **One GPU-touching agent at a time.** Contention produced the two largest
   errors of the day. `gpuq status` gates; a model load works at ~6.7 GiB
   "available" with 0 swap, so do not wait for a 14 GiB headline.
5. **Free memory per configuration.** A probe holding 3.8 GB without releasing
   drove the box to 3.7 GiB free and 573 MB of swap; swapped measurements are
   worse than none.
6. **`peak_resident_pages_bytes` is not a footprint.** It is `ru_maxrss`, which
   reported 6.23 GiB where the kernel charges 11.69. Use `footprint -p <pid>`,
   which quantises to 3 significant digits.
7. **Prove each test has teeth**, and remember that a bit-exactness generator
   with an LCG of period 8 makes the check pass vacuously, and that FMA
   contraction is not stable, so two variants can each match alone and differ
   combined (30818/32768 did).

## Open questions

- Is the 79 us first-touch page walk, launch/prologue, or tail drain? Untested
  warm-second-pass, and the answer decides whether the fix is in `linear.metal`
  or in how weights are mapped.
- Does N_R0=4 still pay on the post-specialisation kernel?
- Can `MTLResidencySet` pinning be made to work at all on this binding? It
  would pin pages we already have, at no memory cost, and no measurement
  exists either way.
- Is there any mechanism at all for >244 GB/s, given the sector-rate result says
  the DRAM wall is ~226? If the answer is no, then the only route to 20 tok/s
  is fewer bytes.
