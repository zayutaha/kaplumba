# Tasks: 1.18 -> 20+ tok/s

Ordered by certainty, not by payoff. Stages 1 and 2 are certain bugs worth ~3x
together; do them before spending anything on the speculative stages.

## Stage 1 — the certain wins

- [x] **T1** Time `online_attention` in `elem_bench` at the real decode shapes:
  - Sliding: `queries=1, keys=1024, q_heads=32, kv_heads=16, dim=256`
  - Full: `queries=1, keys=200, q_heads=32, kv_heads=4, dim=512`
  Sizes stage 2 and answers Q3. Currently inferred from code structure, never
  measured.
  - Files: `crates/engine/examples/elem_bench.rs`
- [ ] **T2** `KAPLUMBA_ATTENTION_ONLY=1` attribution switch: run a valid forward
  with norms and linears intact but attention replaced by a no-op, so the
  attention share is measured in situ rather than inferred.
  - Note: this env var does not exist yet; T2 creates it.
  - Files: `crates/engine/src/model.rs`
- [x] **T3** Parallel `rms_norm_weighted`: one 256-thread threadgroup per row,
  strided partial sums, barrier tree reduction. Select only when `rows < 32`.
  *Gate: under 20 µs at `rows=1, width=5376`.*
  - Files: `crates/kernels/src/elements.metal`, `crates/kernels/src/elements.rs`
- [x] **T4** Same for `rms_norm_unweighted` (used for v_norm, not pre_ffn_norm
  as previously thought; `pre_ffn_norm` uses the weighted variant).
  - Files: `crates/kernels/src/elements.metal`, `crates/kernels/src/elements.rs`
- [x] **T5** Correctness for T3/T4 vs the current kernel at `rows` in
  {1, 2, 7, 32, 64, 134} and widths {128, 256, 512, 5376}, float tolerance.
  Include both weighted and unweighted variants.
  - Files: `crates/engine/tests/elements_metal.rs`
- [ ] **T6** Re-measure the layer stack. *Gate: 783 → under 650 ms.* (Adjusted
  from 600 ms; the corrected norm count saves ~143 ms, not ~179 ms.)
- [x] **T7** Two-pass split-K `online_attention`: per-split `(max, l, acc)`,
  then combine with `exp(m_i - m_global)`. No fp32 atomics. Keep the serial
  kernel when `queries * q_heads >= 1024` so prefill is untouched. The decode
  dispatch is 32 threads (1 query × 32 q_heads), not 8.
  - Files: `crates/kernels/src/attention.metal`, `crates/kernels/src/attention.rs`
- [x] **T8** Remove the per-thread `float accum[512]` for small grids; stage in
  threadgroup memory. At decode, 32 threads × 2 KB = 64 KB of registers just
  for the accumulators.
  - Files: `crates/kernels/src/attention.metal`
- [x] **T9** Attention correctness at `queries` {1, 8}, `keys` {1, 200, 1024},
  GQA ratios {1, 2, 8}, full and sliding windows. Test both the serial kernel
  (prefill shapes) and the split-K kernel (decode shapes).
  - Files: `crates/engine/tests/attention_metal.rs`
- [x] **T10** Re-measure. *Gate: layer stack under 300 ms, decode ≥ 3.0 tok/s.*
  (Adjusted from 250 ms / 3.4 tok/s to account for corrected norm savings.)
- [x] **T11** Prefill unchanged (8.4 tok/s at 134 tokens). Full suite green.
  Create an output comparison test (the spec references `output_identity` but
  no such test exists; build it as a generation comparison against a recorded
  baseline at temperature 0).
  - Files: `crates/engine/tests/generation.rs` or new test file.

## Stage 2 — the matvec

- [ ] **T12** Fix the hanging `cb_bench` A/B and get the LUT number. It stalled
  before producing a result; suspect the CPU-side fill of a 1 GB buffer, same
  failure mode as the streaming probe that had to be deleted. **Q1 gates R4.**
  - Files: `crates/engine/examples/cb_bench.rs`
- [ ] **T13** `half2` codebook packing: two codes per lane from adjacent lanes,
  targeting the measured 5.10e12 ops/s instead of half that. *Q5.*
  - Files: `crates/kernels/src/linear.metal`
- [ ] **T14** Cooperative ring load in `linear_gemv_exl3_k*`: the 32 lanes of a
  simdgroup collectively need one `256 * ka`-bit ring, so load it once per
  group (ideally one 128-bit load per thread) rather than three overlapping
  32-bit loads per lane. Targets the 55% load share. *Gate: `q_proj` ≥ 100 GB/s.*
  - Files: `crates/kernels/src/linear.metal`, `scripts/gen_exl3_gemv.py`
- [ ] **T15** Re-measure decode. *Gate: ≥ 5 tok/s.*

## Stage 3 — past the codebook wall

- [ ] **T16** Codebook LUT if T12 says it pays: 256 entries built per threadgroup
  from the existing hash, indexed by `code >> (16 - ka)`, reproducing the hash
  bit-for-bit for every rate and codebook. If T12 says it does not pay, try a
  `constant`-address-space table in a small buffer instead, and if that also
  fails, drop the idea and record why.
  - Files: `crates/kernels/src/linear.metal`, `scripts/gen_exl3_gemv.py`
- [ ] **T17** Re-measure. *Gate: ≥ 11 tok/s.*
- [ ] **T18** Decide honestly against Q2. If we are short of 20, say so and
  record the measured ceiling rather than continuing to chase it.
- [ ] **T19** Only if T18 leaves real headroom: design note on a repacked trellis
  layout so a simdgroup reads contiguous bytes (Q7). Requires rewriting the
  checkpoint at load time; too large to start without evidence.

## Throughout

- [ ] **T20** Keep `elem_bench` covering every decode-path kernel (R7). A
  serialisation must show up as a microsecond number, not as an end-to-end
  regression discovered hours later.
  - Files: `crates/engine/examples/elem_bench.rs`
- [ ] **T21** Update `implementation_notes.md` after every stage with the measured
  before/after, so the next session does not re-derive any of this.

## Progress (2026-09-27)

- **T1 done.** Attention measured: 132.4 us sliding (keys=1024, 32q/16kv, dim 256),
  128.3 us full (keys=200, 32q/4kv, dim 512), so **7.9 ms per token**, not the
  ~380 ms both specs inferred. **Stage 2 (T7-T9) is therefore worthless and is
  struck**: 8 ms of an 850 ms token does not justify split-K attention, dropping
  `accum[512]`, or the register-pressure work.
- **T3-T5 done.** `rms_norm` at `rows == 1`: **595.7 us -> 11.9 us**, a 50x
  improvement, by giving each row a 256-thread group with a strided partial sum
  and a barrier tree. Dispatch switches on `rows < 32` so prefill keeps the
  per-row kernel. New test compares both paths at widths {128, 256, 512, 5376} x
  rows {1, 2, 7, 31, 32, 64, 134}: exact for most shapes, worst relative
  difference 6.64e-4 at width 5376, which is the reordered sum.
- **T11 done.** Suite green, prefill unchanged at 8.4 tok/s, and all five
  reference prompts produce byte-identical text.

Result: layer stack 766 -> 665 ms, decode **1.18 -> 1.51 tok/s**.

### A bug worth recording

The first attempt at this produced fluent garbage from every prompt, and the
threadgroup norm was *not* the cause: an earlier attribution refactor had
replaced the four hidden-width norm call sites with a helper that hardcoded
`rows = 1`. Decode has one row so it looked fine, but prefill passes
`rows = tokens` and was silently normalising only the first row. Forcing the
original per-row kernels did not restore the output either, which is what proved
the refactor rather than the new kernel was at fault. The helper now takes `rows`.

### What is left

665 ms of layer time is now almost entirely the linear matvec:

| | ms | note |
|---|---:|---|
| norms | 2.9 | was 156 |
| attention | 7.9 | measured, not worth optimising |
| matvec + element ops | ~654 | 11.00 GB at **16.8 GB/s** |

10 tok/s needs the matvec at ~90 ms, i.e. **122 GB/s, a 7.3x improvement**. The
same kernel measures 40-64 GB/s in isolation, so the gap is not arithmetic. It is
either non-coalesced access (Q7) or the loss of sustained streaming when seven
different projections are interleaved with ~16 element kernels, 60 times a token.
That is the next investigation and it is the whole remaining problem.

## Split-key attention (2026-09-27)

**I was wrong to strike stage 2, and the reason matters.** The 7.9 ms figure came
from `elem_bench` with an 8.4 MB KV buffer, which is L2-resident. In situ the KV
cache is 1.5 GB and every read is a DRAM miss, and the kernel runs on 32 threads
(`queries * q_heads` with 32 query heads) each holding a 512-float accumulator.
Ablation with `KAPLUMBA_SKIP_ATTENTION=1` put the real cost at **153 ms**, a
19x gap against the isolated measurement. So stage 2 was the second-largest win
available all along, and I nearly deleted it on the strength of a benchmark that
was measuring cache.

This also explains the 2.3x matvec discrepancy that was left over from the
previous round. The old attention kernel held 2 KB of registers per thread and
ran 2.5 ms per layer; it was starving the matvecs queued behind it. With
attention fixed, the matvecs run at 229 ms against the 206 ms that
`layer_bench` predicts for 60 layers -- within 11%, where before they were 2.3x
off.

Implemented as three flash-decoding passes, chosen so no pass needs a per-thread
array proportional to `head_dim`:

  attention_scores    one thread per (head, key)   -- q_heads * key_count
  attention_row_reduce one thread per head
  attention_combine   one thread per (head, dim)   -- q_heads * head_dim

KV data is read once instead of once per thread. Selected only when
`queries == 1`; prefill keeps the serial kernel and is unchanged at 8.4 tok/s.

| | before | after |
|---|---:|---:|
| layer stack | 630 ms | **242 ms** |
| decode | 1.51 tok/s | **3.84 tok/s** |
| prefill | 8.4 tok/s | 8.4 tok/s |

Cumulative so far: 1.18 -> 3.84 tok/s, a 3.25x improvement, output byte-identical
on all five reference prompts.

## Where the remaining 242 ms is

| | ms | GB/s |
|---|---:|---:|
| attention | ~13 | |
| matvec (11.00 GB) | ~229 | **48** |

48 GB/s against `layer_bench`'s 50.7 GB/s means the matvec is now running at its
measured speed, so the isolation gap is closed and what remains is genuine matvec
efficiency. 10 tok/s needs 100 ms, i.e. ~130 GB/s, which is 2.7x from here. That
is stages 3-5 of this spec: cooperative ring load (the loads are 55% of the
kernel and do not coalesce, Q7), `half2` codebook packing, and the codebook LUT.

## ishizuki's matvec, ported verbatim (2026-09-27)

You were right that the format was not the ceiling. ishizuki's `exl3_matvec`
(EXL3.swift:286) is now ported line for line into `linear_gemv_ishizuki_k*` and
is the **default** for integer rates. `KAPLUMBA_OWN_GEMV=1` selects the old kernel.

| | GB/s |
|---|---:|
| our previous kernel | 50.9 |
| ishizuki, verbatim | **100.8** |

End to end: layer stack 242 -> 128 ms, decode **3.84 -> 6.77 tok/s**, prefill
unchanged at 8.4, all five reference prompts byte-identical, suite green.
Cumulative: 1.18 -> 6.77 tok/s, 5.7x.

### Three bugs I made while porting, all of which faked a result

Worth recording, because two of them produced *convincing* numbers:

1. **Unbound K and N.** My kernel read `K` and `N` from buffers 3 and 4, but the
   dispatch bound only three buffers, so `tiles_n` and `tiles_k` were garbage and
   the loop ran zero iterations. That measured **304 GB/s -- 6x "faster" than
   ours** while emitting pure noise. I reported that 6x before checking
   correctness. It was my bug, and it also exceeded DRAM peak, which was the tell.
2. **Swapped bindings.** The kernel signature was `(x, trellis, y)` while the
   dispatch provides `(packed, transformed, raw)`.
3. **The main loop was missing.** My template-extraction slice came out empty, so
   the kernel had no `for` loop and `acc0`/`acc1` were always zero.

Every one of these produced fast-and-wrong. The rule I should have applied from
the start: **an A/B that gets faster must be validated on output before the
number is believed.** Three of this project's benchmarks have now misled me --
clock ramp, L2 residency, and this -- and in all three the check that would have
caught it was cheap.

### What is actually different, now that both run correctly

Ours recomputes `tile_index * ring_words` and `input[xbase + r0 + N]` inside the
loop; theirs hoists `column` and `xr` as pointers so the addresses are affine in
`tk` and the compiler can strength-reduce them. Everything else -- one window per
eight codes, four `x` values for eight codes, 8-simdgroup striding, the
`simd_shuffle_xor` tree -- was already identical, and those changes account for
2.0x. Pointer hoisting and address strength reduction, nothing exotic.

### Where the remaining gap to 10 tok/s is

At 128 ms/token we need 100 ms. The matvec is ~115 ms of the 128. Their reported
28 tok/s is on a **2 bpw** model streaming 6.75 GB/token; ours is **3 bpw** at
12.06 GB/token. 3 bpw is ~1.5x more codes per byte, hence more ALU per byte, so
some of the residual gap is the rate and not the kernel. Scaling their 189 GB/s
(M1 Max) to our bandwidth predicts ~129 GB/s against the 100.8 we measure, so
roughly 1.3x remains and is plausibly reachable.

## Pipelined decode (2026-09-27)

Greedy decode now runs the pick on the GPU. `greedy_pick` reduces the capped
logits with a single unsigned max over packed (value, index) keys -- so the fp16
ordering is preserved by a sign flip, ties resolve to the lowest index, and the
reduction needs no second pass and no comparison branching. The token lands in
device memory, so the next step's embedding kernel reads it without the host.

The loop queues one command buffer per step, `PIPELINE_DEPTH` = 8, submits them
all, then waits and reads that many tokens together. Each step writes its own log
slot and mirrors it into slot 0 for the next step; a single slot would have every
step overwrite the last.

Result: **6.77 -> 6.99 tok/s**. Correct on all reference prompts, suite green.

**That is a much smaller win than the 14.35 -> 22.48 they measure, and the reason
is arithmetic rather than a shortfall in the work.** Their serial loop stalled
proportionally harder on the pick; ours stalls 12 ms out of a 148 ms token
(136 ms of it GPU). Removing the host round trip entirely can therefore only
recover that 12 ms, and it recovers most of it. The remaining 136 ms is GPU
matvec, and that is now the whole problem.

Also fixed along the way: `forward_buffer` created its Metal encoder before
validating, so every error return dropped a command encoder that was never ended
and Metal asserted at dealloc. The wrapper now validates first and abandons the
encoder explicitly on the error path.

## Correction: the roofline, recomputed properly (2026-09-27)

I previously claimed 14 tok/s needed ~234 GB/s, above this machine's streaming
ceiling, and was therefore unreachable. That was wrong -- I undercounted the
per-layer ideal by 2x. The real arithmetic, from the actual config
(hidden 5376, intermediate 21504, 60 layers, 3.04 bpw measured):

    layer stack   10.93 GB
    lm_head        0.54 GB
    TOTAL/token   11.47 GB

    absolute ceiling @ 222 GB/s   19.4 tok/s
    14 tok/s needs  160.6 GB/s  =  72% of streaming

So 14 tok/s is demanding but reachable. Correcting the record here because the
wrong number would have justified giving up on a target that is in range.

### Where the time actually goes (136.35 ms/token)

    matvec (sum of parts)   109.0 ms   80%
    attention                13.0 ms   10%
    lm_head                   8.5 ms    6%
    norms                    ~3.0 ms    2%

### The matvec is ALU-bound, not bandwidth-bound

Bandwidth per projection tracks threadgroup count, not matrix size -- the same
kernel at the same K:

    proj        N     tgroups  tg/core   GB/s   %stream
    k_proj   4096       256      12.8    34.0     15%
    v_proj   4096       256      12.8    33.9     15%
    o_proj   5376       336      16.8   105.9     48%
    q_proj   8192       512      25.6    69.0     31%
    gate    21504      1344      67.2   119.3     54%
    up      21504      1344      67.2   120.1     54%
    down    21504      1344      67.2   113.9     51%

k/v at 15% of streaming with 44 MB matrices is not a bandwidth problem. It is
too few threadgroups to hide memory latency. Separately, per weight the kernel
spends ~6.5 ALU ops (cmp+select+shift, mask, codebook LUT, FMA) and measured
`alu_peak` gives 2.84e12 logic ops/s, so:

     7 tok/s = 46% of logic ALU peak     (we are here)
    10 tok/s = 66%
    12 tok/s = 79%
    14 tok/s = 92%   <- requires near-perfect ALU efficiency
    18 tok/s = 118%  <- provably impossible at 3.04 bpw

Best projection sits at 54% of streaming and 45% of logic peak simultaneously.
Both saturating together is the signature of an ALU-bound kernel, and it means
**bandwidth work alone cannot reach 14**: the decode arithmetic is the wall.
That is also exactly why ishizuki's load-time transcode is worth 1.25x -- it
hoists the LUT/shift work to load time, leaving a FMA-only inner loop.

### Next, in order

1. Fuse q/k/v into one dispatch (N=16384, 1024 tgroups instead of 256/256/512).
   One pass over x instead of three. ~0.12 ms/layer -> ~7 ms/token.
2. Give o_proj more parallelism (336 tgroups, 48% of streaming). ~1 ms/token.
3. Attack the ALU inner loop directly: the `(r <= 64u) ? a >> (64u-r) : c >> (96u-r)`
   select is resolved per weight per iteration and should be hoisted, since for a
   fixed 8-weight group the branch is uniform across the unrolled body.
4. Then the transcode, which is the only lever that removes ALU work rather
   than relocating it. Budget check required: 10-40% more weight memory against
   0.49 GiB of headroom.

## I was wrong about the ALU: measured, not counted (2026-09-27)

I claimed the matvec was ALU-bound and 14 tok/s needed 92% of logic ALU peak.
That came from counting ~6.5 ops per weight in the source and comparing to
`alu_peak`. Counting is not measuring, and it was wrong. Two probe kernels
settle it (same loads, same body, one thing changed):

    gate_proj, 44 MB      time      GB/s
    baseline (load+decode) 0.363   119.5
    loads, NO decode      0.377   115.1   <- removing all the ALU changes nothing
    decode, NO loads      0.194   223.4   <- removing the loads is 1.87x faster

The decode arithmetic is free; it hides completely under memory latency. The
wall is the load path. 54% of streaming is not "saturated" either -- I labelled
it that way and it was meaningless. With the loads gone the kernel runs at 223
GB/s, ~95% of the 235 GB/s this machine actually streams, so 20 tok/s is
genuinely on the table.

### What actually fixed it

Bandwidth tracked *loads in flight per lane*, not occupancy or ALU:

    variant                full layer GB/s
    baseline                  102
    staged ring in tg mem     102   (no change -- not instruction issue)
    unroll 2, hoisted idx     129   <- the win
    unroll 3 / 4 / 6 / 8      128-130 (plateau)

Hoisting `j0/j1/j2/sh` out of the tile loop (they depend only on the lane, not
on the tile index) and unrolling by two so each lane keeps six weight words in
flight is worth **+27% on the matvec**. Applied to all eight bit widths.

End to end: **6.99 -> 7.43 tok/s**, suite green, all reference outputs
byte-identical, deterministic across runs.

Two things learned that cost time:
- `cargo build --workspace` does NOT rebuild examples. `layer_bench` was stale
  and kept reporting the old kernel until I built `--examples`, which briefly
  looked like the rewrite had not taken effect.
- Bash `%`-formatting in the codegen choked on `%%`, silently leaving the old
  kernel in place. Use concatenation, not `%`, when the payload has `%` in it.

### Still unexplained, and it is now the main suspect

    sum of parts, unroll2 : 1.411 ms/layer x 60 =  84.7 ms
    + attention/norm/head                        =  98 ms   predicted
    model in situ                                 = 134.6 ms  measured

A 36 ms gap that was only 18.6 ms before the rewrite -- it grew as the kernel
got faster. `layer_bench` measures isolated kernels on a boosted GPU; the model
runs sustained load. Next thing to test is whether the GPU downclocks under
sustained streaming, because if it does, isolated per-kernel GB/s overstates
the achievable rate and that changes what any further kernel work is worth.

## What the four probes actually prove (2026-09-27, second pass)

The first probe sweep said unroll2 was worth +27%. **That was wrong and did not
reproduce.** A controlled A/B of the rolled-out default against a byte-identical
probe kernel gave 108.9 vs 107.3 GB/s -- i.e. the real matvec gain is **+6%**
(101.7 -> 107.2), which matches the end-to-end 6.99 -> 7.43 (+6.3%) exactly.
The 129 reading came from a build that had seven extra kernels appended to the
library; it never came back. Do not trust a probe number without re-running it as
a byte-identical A/B in the same binary.

The real findings, all reproduced:

    gate_proj 44 MB        time      GB/s
    baseline                 0.347    124.7
    no decode (nod)          0.376    115.1   ALU work: free
    no x loads (nox)         0.348    124.6   activation loads: free
    no weight loads (nold)   0.194    223.4   <- the entire cost, 1.87x

So the decode arithmetic *and* the activation loads are both entirely hidden.
The one and only cost is reading the 44 MB of trellis weights, and it runs at
53% of the 235 GB/s this machine actually streams. Everything else in the
kernel is free.

Why the weight loads only reach 53%: the three words each lane needs (j0, j0+1,
j0+2) are a sliding window over a 24-word ring, so 32 lanes issue 96 lane-loads
to touch only 24 distinct words -- **4x redundant L1 traffic**. Staging the ring
through threadgroup memory was the obvious fix and it did nothing (118.8 vs
119.5) on its own; it is still untested *combined* with the unroll, and that is
the next thing to try.

### The arithmetic ceiling, now that compute is free

    all weights 11.47 GB at 235 GB/s (measured peak)  = 48.8 ms  = 20.5 tok/s

That is the absolute floor for one token, and it is 20.5 tok/s *only if every
byte streams at measured peak and attention, norms and the head cost nothing*.
The real budget has 24.5 ms of non-matvec work in it:

    attention 13.0 ms   (960 KB of KV -- this is latency, not bandwidth)
    lm_head    8.5 ms   (536 MB at 63 GB/s, vs 125 GB/s for a normal matvec)
    norms     ~3.0 ms

    14 tok/s = 71 ms  ->  matvec gets ~47 ms  ->  11.47 GB / 0.047 = 244 GB/s
                                              -> above measured peak. NOT possible.
    20 tok/s = 50 ms  ->  matvec gets ~25 ms  ->  458 GB/s. Not remotely possible.

So 20 tok/s is not reachable at 3.04 bpw by any amount of kernel work: it needs
458 GB/s where the machine streams 235. **The binding constraint is bpw, not the
engine** -- and the arithmetic says ~14-15 tok/s is the real ceiling, which
requires 100% of peak streaming on the matvec alone.

The highest-value remaining work is therefore *not* the matvec. It is the 24.5 ms
of non-matvec time, where lm_head at 63 GB/s and attention at 13 ms for 960 KB
are both far off what the hardware should do. Cutting that to ~5 ms is worth more
than every matvec idea above combined, because it comes straight off the budget.
