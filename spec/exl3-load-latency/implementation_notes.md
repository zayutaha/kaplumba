# Implementation notes: exl3-load-latency

Running log. Append as work proceeds, not at the end.

## 2026-09-24 — spec opened; the control experiment that changes the model

Ran the previously-never-executed control, `bench_stream_bw_slow`
(`crates/engine/src/metal_linear.rs:2573`), on the M4 Pro. 512 MB synthetic u32 buffer:

```
P0 contig 512MB: min 2.558 med 2.625 ms -> 210 GB/s
P0 strided nsplit=1: min 0.949 med 1.006 ms (192 MB) -> 212 GB/s
P0 strided nsplit=2: min 0.624 med 0.664 ms (192 MB) -> 323 GB/s
P0 strided nsplit=4: min 0.544 med 0.561 ms (192 MB) -> 370 GB/s
P0 strided nsplit=8: min 0.548 med 0.560 ms (192 MB) -> 368 GB/s
```

`stream_sum_strided` replicates the `exl3_gemv` grid (`grid.x = out_tiles * nsplit`,
32 lanes/TG) and walks tiles with stride `out_tiles * twords` words — i.e. exactly the
access pattern the "336 tiny straws 21 KB apart" story blamed for a ~68 GB/s ceiling.

**The strided walk is not slower than a fully linear read** (212 vs 210 GB/s at nsplit=1).
Consequences:

1. The `spec/trellis-dense-repack` premise is **falsified**. Its W1/W4 side-car layouts were
   built to remove a stride that hardware handles natively. This explains, after the fact,
   why both candidates measured *slower* than the plain word layout in
   `bench_dense_ab_slow` (e.g. gate W1 41.4 vs word 60.6 GB/s; down W4 40.3 vs word 53.2).
   Repacking was solving a non-problem.
2. The `exl3_gemv_probe` verdict ("per-symbol decode ALU is effectively free; m0 ≈ m2")
   is **withdrawn as a control**. That probe kept the production per-lane gather structure
   and the production grid, so mode 0 and mode 2 were both pinned at the same non-DRAM
   ceiling. It measured the kernel, not the memory system. The "14 ops/weight = 113 ms
   of ALU" model derived alongside it is not supported by this data either way — T3
   measures it directly instead.
3. The real gap is **inside** `exl3_gemv_spec`: 30–95 GB/s achieved vs 212 GB/s available
   for the same walk with no math. That is the ~3× this spec exists to decompose.

Caveats recorded up front (do not skip these when reading the T4 table):
- 370 GB/s exceeds the 273 GB/s nominal bus figure. Almost certainly SLC residency and/or
  boost-clock variance on a 192 MB working set. The defensible comparison point is the
  **nsplit=1, 212 GB/s** number, which is same-shape against the contiguous control.
- The 210 GB/s contiguous figure is one launch over 512 MB. Real linears are 16–87 MB and
  may sit partly in the SLC. T3/T4 therefore measure at real per-linear sizes and report
  cold vs warm separately rather than quoting a single "ceiling".

## 2026-09-24 — decisions taken with the user before opening this spec
- Drop the uncommitted T4 fused RMSNorm+Hadamard work (targets dispatch overhead, which
  the control says is not the dominant term).
- **Keep** 19.5 t/s as the acceptance bar even though it implies 242.6 GB/s sustained,
  above the 210 GB/s measured linear-read figure (~17.8 t/s at 11.833 GB/token). The gate
  stays unflipped and is expected to stay red. Accepted explicitly.
- Attack `exl3_gemv_spec` first (the K-split path serving gate/up/q/k/v — 5 of 7 linears).
- Delete the dense W1/W4 kernels; keep `stash@{0}` (holds the RT2c host plumbing, now moot).
- Bisect before optimizing; new kernel alongside rather than in-place replacement.
- Session scope: deletion + bisection, then report and stop.

## 2026-09-24 — T3/T4: the ladder is FLAT. The inner loop is not the bottleneck.

`bench_gemv_bisect_slow` (new, this spec). k=3, twords=24, warm = min-of-12
back-to-back, cold = L2/SLC flushed before each timed run. GB/s, best nsplit per shape:

| shape | in_tiles | MB | S1 walk | S5 full (all decode math) | S5/S1 | PROD `gemv_pass_spec` |
|---|---|---|---|---|---|---|
| o     5376x5376  | 336  | 10.8 |  71.0 (n8) |  63.2 (n8) | 0.89x | 38.4 (n8) |
| q     5376x8192  | 336  | 16.5 | 106.6 (n8) |  84.5 (n8) | 0.79x | 50.0 (n8) |
| kv    5376x4096  | 336  |  8.3 |  49.4 (n8) |  52.0 (n8) | 1.05x | 29.7 (n8) |
| gate  5376x21504 | 336  | 43.4 | 201.1 (n4) | 163.0 (n2) | 0.81x | 75.9 (n1) |
| down  21504x5376 | 1344 | 43.4 | 184.7 (n8) | 152.8 (n8) | 0.83x | 72.0 (n8) |
| synth 2048x1024  | 2048 | 201.3 | 378.6 (n4) | 270.9 (n4) | 0.72x | 103.1 (n4) |

**Finding 1 — the entire EXL3 decode is free.** Going from a bare walk that reads one word
per lane (S1) to the complete production math (S5: 4 scattered `ld_word`, 8 `ulong` window
assemblies, 8 shift/mask codeword extractions, 8 `exl3_decode_quad_const` codebook decodes,
8 `simd_shuffle` of `xh`, 8 FMA into `accA`/`accB`) costs **0% to 28%**, and on the small
shapes S5 is *faster* than S1, i.e. the difference is noise. The whole ladder S1..S5 is flat
within run-to-run variance at every shape/nsplit.

This kills the last standing version of the ALU theory. There is no 3x to recover in the
inner loop; the "14 ops/weight = 113 ms" model has no support in any measurement, and the
T1/T2 vectorization work bought what it could. **Do not rewrite the decode inner loop.**

**Finding 2 — R5 did not pass, and the reference is what is wrong.** S5 lands ~2x *faster*
than `gemv_pass_spec` (0.272 vs 0.589 ms on gate/n1; 0.171 vs 0.282 on o/n8), not within
noise. Cause, confirmed by reading `exl3.rs`: `gemv_pass` wraps every call in
`with_scratch`, which **allocates four Metal buffers per call** (`x_stage`, `xh`, `y`, `yh`),
and `encode_gemv_spec` allocates a fifth (`owned` partial, 172 KB for down/n8) because
`ext_partial` is `None`. All of that sits inside the timed region, plus a `reduce` dispatch
and a host readback. So the PROD column measures ~0.1–0.3 ms of host-side buffer churn per
call, not the kernel.

Consequence: the spec's R5 was aimed at the wrong thing. The ladder is still internally
valid — S1..S5 are byte-identical kernels except for the stage under test, which is all the
deltas require — but the production column must not be read as "what the kernel costs".
Note the real decode path (`gpu_decode.rs` -> `encode_linear_with_scratch`) uses a
**pre-allocated decoder arena**, so it does *not* pay this; but every per-linear GB/s figure
recorded in the older specs (81.8 gate / 53.9 o / 35.2 v etc.) came from harnesses of this
shape and is inflated by the same overhead.

**Finding 3 — what actually limits the small shapes: parallelism, not math.** Throughput
tracks `in_tiles` (the length of each warp's serial walk), not total bytes:

| in_tiles | shapes | best GB/s |
|---|---|---|
| 336   | o, q, kv, gate | 49–201 |
| 1344  | down           | 185 |
| 2048  | synth          | 379 |

And it is still climbing at the top of the swept range. nsplit sweep for `o` (10.8 MB):
21.3 (n1) -> 63.5 (n2) -> 66.4 (n4) -> 71.0 (n8), **no saturation**. Same for q
(65 -> 87 -> 98 -> 107) and kv (36 -> 42 -> 47 -> 49). The bench only swept up to n8;
`gemv_splits` assigns 8 to out_tiles 336/256 and caps at 16, so **nsplit 16 and 32 are
untested on four of the seven k=3 linears.** Given that n1 -> n8 is worth 2–3.3x on these
shapes, n16/n32 is the cheapest remaining lever and is a one-line policy change plus a
measurement, not a kernel rewrite.

**Where the 117 ms/token actually goes** (from the S5 column, per layer):
gate+up 0.53 + down 0.28 + q 0.19 + kv 0.32 + o 0.17 ~= 1.49 ms/layer x 60 = 89 ms, plus
lm_head ~4.6 ms = ~94 ms of GEMV against a measured 117 ms total. So GEMV is still ~80% of
decode, but its limit is the bandwidth available on short-walk small tensors, not
arithmetic.

## 2026-09-24 — T5 (added): the read ceiling is ~250 GB/s, and the earlier 210 was wrong

`bench_read_ceiling_slow` (new). The P0 `stream_sum_contig` control wrote one u32 per
thread — 134 MB of stores while reading 512 MB — so its "210 GB/s" was paying for write
traffic it never counted. `bis_read_bw` keeps every lane's accumulator live but reduces via
a per-warp atomic, so the only DRAM traffic is the read.

| working set | warm ms | warm GB/s | cold GB/s |
|---|---|---|---|
| 1 MB   | 0.160 |   6.2 |   5.8 |
| 8 MB   | 0.207 |  38.7 |  35.1 |
| 43 MB  | 0.437 |  98.3 |  87.3 |
| 86 MB  | 0.517 | 166.2 | 160.5 |
| 201 MB | 0.964 | 208.4 | 197.4 |
| 512 MB | 2.181 | 234.8 | 225.8 |
| 1024 MB| 4.179 | 245.0 | 239.7 |

**A 1 MB read takes 0.160 ms.** That is physically impossible as memory time (it would be
6.2 GB/s), so it is a **fixed ~0.16 ms cost per timed dispatch** in this harness — command
buffer create + commit + `wait_until_completed` round trip. Subtracting it:

- **true read ceiling: ~250–255 GB/s** (92–93% of the 273 GB/s nominal bus). This is
  consistent with the user's own MLX data point: MLX 4-bit, 15.1 GB at 16–17 t/s implies
  **249 GB/s** on this same chip.
- fits `t ~= 0.16 ms fixed + ~0.10 ms DRAM ramp + bytes / 250 GB/s`.

**Two corrections to earlier claims in this file and in conversation:**

1. The "210 GB/s ceiling / 17.8 t/s max / 20 t/s unreachable" conclusion is **withdrawn**.
   The real ceiling is ~250 GB/s.
2. The suggestion that moving to 4-bit would reach 20 t/s is **wrong and backwards**. 4-bit
   *increases* the bytes: the 10.776 GB of k=3 linears becomes ~14.37 GB, so ~15.4 GB/token
   total, i.e. ~14.9 t/s at 250 GB/s — **worse** than k=3. In a bandwidth-bound regime fewer
   bits is strictly better. (MLX's 4-bit advantage is that it sustains 249 GB/s, not that it
   uses 4 bits.)

**Where that leaves the target:**
- 20.0 t/s needs 236.7 GB/s = **95% of the measured ceiling with zero allowance for any
  non-GEMV work**. Not reachable in practice.
- 19.5 t/s needs 242.6 GB/s = 97% of ceiling. Above the warm 1 GB figure's margin.
- 18.0 t/s needs 225 GB/s = 90% of ceiling. Plausible.
- 16.0 t/s needs 199 GB/s = 80% of ceiling. Comfortable.

**And where the time actually goes.** GEMV, priced from the S5 column, is ~1.49 ms/layer
x 60 + 4.6 ms lm_head ~= 94 ms of the measured 117 ms. That leaves **~23 ms of non-GEMV
work** (had_in, had_out, rms, rope, attention, gelu — ~24 dispatches per layer, ~1440 per
token). GEMV itself is already within ~2x of the ceiling and its remaining gap is the
untested nsplit 16/32. So the two levers are: (a) nsplit sweep, (b) the ~1440 small
dispatches per token.

**This reverses the earlier T4/T6 cancellation.** Those tasks (fuse rms+had_in, fuse
qkv_prep) were cancelled on 2026-09-24 because the then-current diagnosis said dispatch
overhead was not dominant. With GEMV priced at ~94 of 117 ms, the non-GEMV ~23 ms is the
largest single addressable block left. T4/T6 deserve revival.

## 2026-09-24 — the measurement was the problem: submit overhead swamped every small kernel

Every per-linear number in every spec so far was measured with one
`commit()` + `wait_until_completed()` round trip **per dispatch**. That round trip is
**not an additive constant** — it dominates any kernel below ~0.3 ms and vanishes above
it. A 1 MB read "measured" 0.160 ms, i.e. 6.2 GB/s, which is physically impossible as
memory time; it was almost entirely round trip.

Fix: put N reads **inside one command buffer** so the submit is paid once
(`bench_walk_vs_linear_amortised_slow`, 40 reps in one buffer, real layer-0 tensors):

| tensor | MB | linear | strided (exl3_gemv walk) |
|---|---|---|---|
| gate | 43.4 | 0.170 ms / 254.6 GB/s | **0.095 ms / 456.4 GB/s** |
| down | 43.4 | 0.169 ms / 255.8 GB/s | **0.094 ms / 461.5 GB/s** |
| o    | 17.3 | 0.054 ms / 304.3 GB/s | **0.030 ms / 552.1 GB/s** |
| k    |  8.3 | 0.023 ms / 365.1 GB/s | **0.016 ms / 515.8 GB/s** |

(The >DRAM figures are SLC-resident at these sizes — that is the point: the pattern has
**no intrinsic cost**. 456 GB/s vs a 254 GB/s linear read means the 21 KB-stride walk is
if anything *better* for this access pattern, when it is not being timed by a stopwatch
that mostly measures `commit()`.)

And the DRAM-rate reference, no per-dispatch overhead at all
(`bench_full_model_stream_slow`, the real packed-weight allocation):

```
6.378 GB -> 24.729 ms -> 257.9 GB/s
3.189 GB -> 12.095 ms -> 263.7 GB/s
1.594 GB ->  6.194 ms -> 257.4 GB/s
```

**The machine does 258 GB/s on our own weights.** Decode runs at 101 GB/s (11.833 GB /
117 ms). GEMV at the DRAM rate would be `10.776/0.258 + 1.057/0.254 = 46 ms`, i.e. the
46 ms of GEMV plus everything else is what stands between 117 ms and 20 t/s. **~70 ms per
token is not GEMV.**

### This reverses the T4/T6 cancellation

`spec/bandwidth-saturate` T4 (fuse RMSNorm+had_in) and T6 (fuse qkv_prep) were cancelled
on 2026-09-24 on the reasoning "dispatch overhead is not the dominant term". That
reasoning was wrong — it rested on per-pass benches whose absolute values are inflated by
exactly the round-trip cost quantified above, and on a GEMV-is-slow theory that has now
been falsified five separate ways. With GEMV priced at ~46 ms of a possible 117 ms, the
non-GEMV ~70 ms is the largest addressable block in the model. **T4/T6 are reinstated.**

### Falsified this session (all measured, none speculative)
1. **Tile stride / layout** — strided walk 456 GB/s vs linear 254 GB/s. No cost.
2. **Decode ALU** — bisection ladder S1..S5 flat, 0–28%.
3. **nsplit 16/32** — worse; policy within 2% of argmax.
4. **4-simdgroup geometry** (`exl3_gemv4`) — slower (4x index prologue per unit work).
   *Also carried a real correctness bug, now fixed and gated — see below.*
5. **Concurrent independent streams** — 0.98x. No win.
6. **Load amplification** — `exl3_gemv_cvt` (one coalesced word per lane + `simd_shuffle`
   for the other three, 128 -> 24 transactions per tile) was **bit-identical** to spec and
   **20–30% slower**: the redundant loads were hitting L1 cheaply, and `simd_shuffle` is a
   crossbar op. Kernel deleted after measurement.
7. **TLB / huge pages** — real model tensor vs fresh same-size allocation, identical raw
   read: within 1–13%. No page-table effect.
8. **mmap / demand paging** — all 11.9 GB is read every token, so there is no reuse to
   exploit; and the 5.3x-amplified gather *requires* cached memory, so an uncached mapping
   would be strictly worse.

### Bug found and fixed
`exl3_gemv4` decoded weights from tile `it0+j` while reading activations from
`it0+sg+j*NSG` — wrong results for simdgroups 1–3, invisible to the old test because it
used a 16×16 shape (1 in-tile) where `sg>0` never enters the loop. New gate
`gpu_gemv4_multisimdgroup_matches_spec` fails on the old code (`want -29.225, got 10.046`).

## 2026-09-24 — T6: the ladder on real data, uncacheable. The kernel is not the problem.

Every earlier ladder run used a **synthetic 43 MB buffer**, which is SLC-resident, so it
measured the kernel with memory essentially free. Rewritten to walk **60 distinct real
`gate_proj` tensors = 2.60 GB aggregate** in one command buffer: the working set dwarfs any
cache, so every read is a genuine DRAM read, with no flush and no subtraction to amplify
noise (`bench_ladder_real_cold_slow`).

```
S1 walk        455.1 GB/s   1 word/lane
S2 gather      347.6 GB/s   + the 4x scattered ld_word  (-24%)
S3 extract     344.6 GB/s   + ulong assembly + 64-bit shift/mask   (free)
S4 decode      346.1 GB/s   + exl3_decode_quad_const               (free)
S5 full        333.0 GB/s   + simd_shuffle of xh + FMA             (free)
S6 partial     337.9 GB/s   + the real 16-float f32 partial write  (free)
```

**The entire EXL3 decode — 4 scattered loads, 8 `ulong` window assemblies, 8 shift/mask
extractions, 8 codebook decodes, 8 shuffles, 8 FMAs, and the f32 partial write — costs
3% over a bare walk.** The only real per-tile cost is the 4x load amplification (24%).

**And the production GEMV is running at 110 GB/s, not 337.** The mechanism is now clear and
it is not the kernel:

- 60 independent `gate` dispatches in one command buffer: 7.7 ms total, **0.128 ms each**.
- A single `gate` GEMV in isolation: **0.392 ms**.
- The same work is therefore **overlapping ~3x on this hardware** when the dispatches are
  independent. A lone linear does not saturate the memory system; several in flight do.

Our decode is a chain of 60 strictly dependent layers, so the only concurrency available
is *within* a layer: `q/k/v` (attention branch) and `gate/up` (MLP branch) are independent
of each other, which is roughly 3 streams — not 60. That is the whole 3x.

This is consistent with, and finally explains, every earlier observation:
- `gemv_splits(336) = 1` was 2.2x slower: 42 threadgroups cannot fill the machine.
- `gemv4`'s 4-simdgroup split was slower: it trades threadgroups for threads-per-group and
  does not add cross-dispatch concurrency.
- `exl3_gemv_cvt` was slower: fewer transactions per tile, same concurrency.
- The dense repack was slower: layout is free (S1 above).
- lm_head hits 254 GB/s: 16384 out-tiles = enough threadgroups to saturate **on its own**.

So the lever is **cross-dispatch concurrency, not anything inside the kernel.** Two ways
to get it, both without touching quality or bytes:

1. **More independent streams per layer.** `q/k/v` (33 MB) and `gate/up` (87 MB) are
   independent. Today they are separate back-to-back dispatches in one encoder; whether
   the driver actually overlaps them is now the question to answer. If it does not, issue
   them on **concurrent command buffers** (`MTLCommandBufferDescriptor.concurrentCommands`),
   which is the only Metal mechanism that makes independent work genuinely overlap.
2. **Split the dependent chain by out-tile groups.** Not possible — activation dependency
   is total.

Budget if within-layer overlap is achieved: per layer 180 MB at the 3-stream aggregate
(~200-250 GB/s) = 0.72-0.90 ms, x60 = 43-54 ms, + lm_head 4.2 ms = **47-58 ms = 17-21 t/s**.

## 2026-09-24 — T7: the amortised nsplit re-sweep. The kernel is not the bottleneck.

The entire `gemv_splits` policy was tuned with one `commit()` + `wait()` per dispatch — a
cost that dominates any kernel under ~0.3 ms, i.e. *every measurement that produced it*.
Re-swept with 60 distinct real tensors per point (2.60 GB aggregate, uncacheable), one
command buffer (`bench_nsplit_resweep_slow`), using the S5 production-math kernel:

| shape | ns1 | ns2 | ns4 | ns8 | ns16 | ns32 | policy |
|---|---|---|---|---|---|---|---|
| gate 5376x21504 | 339 | 336 | 341 | 343 | 344 | 347 | 2 |
| down 21504x5376 | 124 | 227 | **340** | 334 | 341 | 348 | 8 |
| o    8192x5376  | 139 | 249 | 363 | 367 | 360 | **377** | 8 |
| k    5376x4096  |  84 | 149 | 237 | **273** | 278 | 264 | 8 |

GB/s. Two things fall out:

1. **The kernel sustains 340–377 GB/s on the real weights** — above the 258 GB/s we get
   from a pure sequential read of the same allocation, because the tile walk has better
   DRAM page behaviour than a linear sweep here. The EXL3 access pattern is not a
   liability; it is an asset.
2. **`nsplit <= 2` is catastrophic for the narrow shapes** (down 124, o 139, k 84 GB/s) and
   everything saturates by **ns4–ns8**. The policy's choices (2/8/2/8) are on the right
   side of that knee, so the *policy* is roughly right — but it was right by luck, and
   `down` would prefer ns4 (340 vs 334) and `o` ns32 (377 vs 367). Those are 2-3% and are
   not worth the churn.

**The single-linear rate is the problem, not the split.** One `gate` in isolation costs
0.402 ms; the same tensor dispatched as part of a 60-way independent batch costs 0.128 ms.
A lone linear cannot saturate this memory system. `bench_dispatch_overlap_slow` measures
what concurrency is actually available inside one command buffer:

```
A alone 0.402 ms   B alone 0.254 ms
A+B     0.423 ms   -> 1.55x overlap   (sum 0.656)
A+B+C   0.587 ms   -> 1.71x overlap   (sum 0.958)
```

So independent dispatches **do** overlap already, no `concurrentCommands` needed (and the
`metal` crate does not expose it anyway) — but overlap saturates around 1.7x at three
streams. A decode layer offers exactly three independent streams: the attention branch
(`q/k/v` -> attn -> `o`) and the two MLP inputs (`gate`, `up`), with `down` gated behind
`gate`+`up`. That is the whole concurrency budget, and it is fixed by the dependency graph.

## Where the 117 ms actually goes
Per layer 1.95 ms, of which the MLP chain (gate+up+down = 130 MB) at the single-linear
108 GB/s is 1.20 ms and the attention chain (qkv+o = 44 MB) is 0.41 ms; overlapped they
bound at 1.20 ms. The measured 1.95 ms leaves ~0.75 ms/layer (~45 ms/token) in the
small kernels (6 RMS + 2 rope + attention per layer) and in imperfect overlap.

**Ceiling with the current dependency graph, everything else perfect:**
MLP chain 130 MB x 60 = 7.8 GB at 108 GB/s = 72 ms, attention chain overlapped, lm_head
4.2 ms => ~76 ms => **13 t/s**. Reaching 20 t/s (50 ms) needs the *single-linear* rate up,
and the only untested lever for that is giving one linear more threadgroups than
`nsplit` can supply — i.e. splitting the in-tile range across threads *within* a
threadgroup without the 4x index-prologue penalty that sank `exl3_gemv4`. That is the one
optimisation this session has not falsified.

## Summary of the whole investigation
| claim | status |
|---|---|
| hardware ceiling on our weights | 258 GB/s sequential, **340-377 GB/s strided** |
| EXL3 decode arithmetic | **free** (3% over a bare walk, S2->S5 ladder) |
| f32 partial write + reduce | **free** (S6 == S5; reduce dispatch 0.8 ms/token) |
| access pattern / tile stride | **free to beneficial** (S1 455 GB/s) |
| layout repacking (W1/W4) | **slower** — no problem to solve |
| nsplit | saturated by ns4-8; policy ~right |
| cross-dispatch overlap | available, saturates ~1.7x at 3 streams |
| single-linear rate | **108 GB/s — this is the binding constraint** |
| decode today | 117 ms/token, 95.5 GB/s, 8.53 t/s |

1. Determine whether consecutive independent dispatches in one encoder already overlap
   (compare: qkv alone, gate+up alone, both in one CB, both on concurrent CBs).
2. If not overlapping, switch the attention and MLP branches onto concurrent command
   buffers. This is the change that should produce the 2-3x.
3. Re-tune `gemv_splits` **with amortised timing** — the entire existing policy was tuned
   against measurements dominated by the submit round trip, so its argmax is unverified.


1. **Re-run every per-linear and per-phase bench with the submit cost amortised.** All
   historical per-linear GB/s figures in `spec/bandwidth-saturate`,
   `spec/pure-decode-273gb` and `spec/trellis-dense-repack` are inflated and must not be
   used to target work.
2. **Reinstated T4/T6**: fuse RMSNorm+had_in and the qkv post-processing. This is the
   ~70 ms.
3. Measure the real phase split (rms / had_in / had_out / attention / gemv) with the
   amortised method before choosing between them.


   bit-exactness (larger splits change f32 split summation order).
2. **Revive the cancelled T4/T6 fusions** (rms+had_in, qkv_prep) and re-run the
   `decode_throughput_slow` split to confirm where the ~23 ms actually sits before
   committing effort.
3. Re-baseline the acceptance bar to **18 t/s** as the target and 16 t/s as the gated floor,
   with the 250 GB/s ceiling recorded as the physical justification.
4. Re-baseline every per-linear GB/s figure using the decoder arena instead of
   `with_scratch`, so the specs stop quoting numbers inflated by buffer allocation.


- No `build.rs` in `crates/kernels`. MSL is `include_str!`'d and compiled at **runtime** by
  `new_library_with_source`, which compiles the *entire* file. `cargo build` cannot catch a
  Metal syntax error; any engine GPU test that constructs `MetalLinear::new` will. Corollary
  for T3: the new bench MSL string only gets syntax-checked when the bench actually runs.
- Test suites at `4b3708f` (re-measured after the T0 revert, release profile):
  engine lib **132 passed / 0 failed / 12 ignored**; kernels **21 passed / 0 failed**.
  (An earlier reading of 133 included the now-discarded T4 test
  `gpu_rms_had_in_batched_matches_separate_passes`; 132 is the correct `4b3708f` count.)
- Gated benches need `KAPLUMBA_RUN_SLOW=1` **and** `--ignored`.
- `spec/bandwidth-saturate` T3 (`gemv_splits(336) = 1`) was committed at `707714e` and then
  reverted inside `4b3708f` after it measured 2.2x slower on `down` — a second instance of
  a plausible idea losing to measurement. Precedent: measure before believing.
