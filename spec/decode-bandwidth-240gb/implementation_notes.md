# Implementation notes: 240 GB/s

Running log for `spec/decode-bandwidth-240gb/`. Appended as work proceeds
(AGENTS.md §5). Dated 2026-09-30; this session landed the decode instrument,
answered T2, and found that the kernel under measurement was not the kernel
under execution.

## The finding that outranks everything else here

**`encode_gemv` dispatches `linear_gemv_ishizuki_kN` first, and the EXL3 family
only when `packed_offset % 4 != 0`.** For this checkpoint, 410 of 411 tensors are
4-byte aligned, so all 60 layers' projections run on ishizuki.

Proven, not inferred. `KAPLUMBA_DEBUG_OFFSETS=1` reports `aligned4=true` for 410
tensors and `false` for one; and `KAPLUMBA_TRACE_GEMV=1` (added in `bcc95b8`)
prints, for a real 3-token decode:

```
GEMV-DISPATCH linear_gemv_ishizuki_k3
GEMV-DISPATCH linear_gemv_exl3_k6_u_mcg
```

Two names. The second is `lm_head`, whose trellis starts at `% 4 == 2` — 8.9% of a
decode token's bytes.

Consequences, in order of how much they matter:

1. **The codebook specialisation of `3439d41` is not what made decode fast, but
   it is not wrong either.** ishizuki's `k3` already has
   `constexpr int CB = 1`, so the compile-time-codebook idea was already in the
   live kernel when that commit landed. The 8.1 -> 9.2 tok/s step stands. What
   `3439d41` added to the *live* path was the `encode_gemv` restructure and the
   dispatch table, not the `_mcg` kernels themselves.
2. **Every EXL3-family bandwidth number in this repository's history is a number
   about `lm_head`.** Including the 3 bpw rate-dependence I measured this
   session — though, as it happens, `gemv_shape.rs` calls `encode_gemv`, so
   *that* sweep is already ishizuki. The `N_R0` sweep calls
   `encode_gemv_named` with explicit `_mcg` names, so *that* one is `_mcg`. The
   distinction is easy to lose and I lost it once mid-session before noticing
   that `which_gemv` disagreed with what I assumed.
3. **Any future lever has to be aimed at `linear_gemv_ishizuki_k3`.**

`examples/which_gemv.rs` prints the decision per (rate, alignment) without
loading the model, for anyone who does not want to trust this paragraph.

## T2 is a clean negative, and why the +88% evaporated

Medians of 7 **interleaved** rounds, baseline re-measured every round:

| shape | NR0=1 | NR0=2 | NR0=4 | NR0=8 |
|---|---|---|---|---|
| gate_proj 5376x21504 | 129.4 | 137.0 | 137.3 | 132.0 |
| down_proj 21504x5376 | 123.4 | 125.9 | 117.1 | 98.6 |
| q_proj 5376x8192 | 104.6 | 109.5 | 105.4 | 95.0 |
| lm_head 6bpw | 252.2 | 251.5 | 247.4 | 248.2 |

Reproduced across two full sweeps (gate +6.3/+6.8%). The spec's +88% was
measured on the pre-specialisation kernel; the specialisation and the input
transform rewrite between then and now took most of what NR0 was going to buy.

**NR0=8 loses 20% on `down_proj`**, which is the shape with the most k per
thread: `acc[NR0][2]` + `v[NR0][8]` is 10*NR0 floats of accumulator and 8*NR0 of
decoded values per thread, and 8 rows is 80 registers of accumulator alone.

## The mechanism, which is the actual finding

Holding weight bytes at 43.4 MB and varying **only** the rate, on the live
kernel:

| rate | out_features | GB/s | lane-loads per useful byte |
|---|---|---|---|
| 3 bpw | 21504 | **132.5** | 12 B -> 3 B = 4.0x |
| 4 bpw | 16384 | 179.8 | 3.0x |
| 6 bpw | 10752 | 164.5 | 2.0x |
| 8 bpw | 8192 | 44.0 → 176.3 | 1.5x |

**`exl3_decode8_mcg` reads a fixed three ring words — 12 bytes — per lane
whatever `KA` is.** 8 codes at `KA` bits are `KA` bytes of information, so the
amplification against distinct bytes is `12/KA`: 4x at 3 bpw, 2x at 6 bpw. The
window is sector-perfect, so DRAM traffic is correct and the 96 B window itself
sits at the ceiling (246.9-248.4) — the cost is **requests**, not sectors.

**3 bpw is 410 of 411 tensors.** That is the whole remaining gap. It is also why
the affine asymptote at 3 bpw is 164.6 GB/s while `lm_head` at 6 bpw reaches 252.

Every cure on the closed list adds *streams* or widens *loads*; none reduces
requests per useful byte. `uint4` cannot help because the ring wrap means the
fast path only fires for 8 of 32 lanes. The one idea that attacks the mechanism
is to make a lane's window word-aligned — at KA=3 a lane needs 24 bits of a
32-bit word, so **one load could serve 8 codes** — and the blocker is that lane
`l`'s codes start at bit `(l*8+1)*KA + ring - 16`, which is not 32-aligned. That
is in the `r0`/`i` index arithmetic, and changing it changes which codes a lane
owns, hence the accumulation order, hence bit-exactness.

## T6 landed: the third ring word is dead at 3 bpw

`linear_gemv_ishizuki_k3` reads **three** ring words per lane per tile and builds

```metal
const ulong A0 = ((w0a << 32) | w0b) << sh0 | (sh0 ? (w0c >> (32 - sh0)) : 0ul);
```

then extracts, for each code `i` in `0..8`, `r = i*KA + 16` and a 16-bit field
from `A0 >> (64 - r)`. So the A0 bits any code can touch are exactly
`[48 - 7*KA, 63]` — and `w0c` contributes **only** to A0 bits `[0, sh)`, the low
bits shifted down from its top.

At KA=3, `sh = lo0 % 32` with `lo0 = (lane*8+1)*KA + RING - 16`, so `sh` takes
values `{19, 11, 3, 27}` over the 32 lanes. **max sh = 27, and the lowest used
bit is 48 - 21 = 27.** The third load is never read. Two words suffice.

`TWO_WORD_SAFE_RATES` in `scripts/gen_exl3_gemv.py` *derives* which rates are
safe rather than listing them — it checks `48 - 7*KA >= max(sh)` and
`7*KA + 16 <= 64`, and enumerates `sh` over all 32 lanes rather than bounding it.
Emitted: 1, 2, 3, 4 bpw. Not emitted: 5 (`used [13,63] < max sh 29`), 6
(`[6,63] < 22`), 7 and 8 (the `B0` arm is live and genuinely reads word 3).
**6 bpw is `lm_head`, so the one shape where the lever does not apply is the one
already at the ceiling** — which is a tidy result rather than a lucky one.

Microbenchmark, medians of 7 interleaved rounds, two full sweeps
(`gemv_shape --w2`):

| shape | 3-word | 2-word | ratio |
|---|---|---|---|
| gate_proj 5376x21504 | 151.6 / 152.3 | 166.6 / 167.1 | 1.099x / 1.097x |
| up_proj 5376x21504 | 148.0 / 152.7 | 164.5 / 166.7 | 1.112x / 1.092x |
| down_proj 21504x5376 | 148.2 / 148.6 | 163.3 / 161.8 | 1.102x / 1.089x |
| o_proj 5376x8192 | 145.7 / 147.0 | 157.1 / 157.1 | 1.078x / 1.068x |
| k_proj 5376x4096 | 135.5 / 140.4 | 146.6 / 154.3 | 1.082x / 1.099x |

End to end, 96 tokens, 5 interleaved A/B pairs:

```
3-word  9.034 10.115 10.073 10.095  9.054   median 10.07 tok/s
2-word 10.782 10.688 10.962 10.860  3.544   median 10.78 tok/s
```

**+7.0% on the median, 9 of 10 paired comparisons favour the 2-word arm**, and the
2-word arm won every microbenchmark shape, so the effect is the kernel and not
the box. The 3.544 sample is unexplained and is counted as neither evidence nor
counter-evidence: a single 3.5x-slow run in a set of five is the signature of the
paging failure already measured at `weights.rs:1217` (3.9 GiB of swap, 1.4
tok/s), and `gpuq` reported 0 swap at the time, so the cause is not established.

**The win is smaller than 3->2 loads suggests, and that is the informative part.**
The window is sector-perfect, so word 3 was an L1 *request*, not a DRAM sector —
and the wall is DRAM. What the result establishes is that request count is not
free, which is the first positive result on the axis every previous lever
(concurrency, load width, `N_R0`, `uint4`, unroll) missed.

## T1: the 79 us, the 59 us, and the 155 us were all the wrong quantity

The spec carried a "per-matvec fixed latency" of 79 us, corrected to 59 us in
this session, and **both were artifacts.** The sequence matters because each
correction looked like an improvement.

**Where 59 us came from.** A three-point affine fit of time against bytes, with
the pass count scaled as `1/bytes` so every size streamed ~300 MB. A *per-buffer*
cost is therefore amortised differently at every size, and lands in the **slope**,
not the intercept. Reading the intercept as a "per-matvec" fixed term is reading
a per-byte term and naming it a fixed one.

**The tell.** Refitting today's numbers the same way gives **8.1 us** — a 7x
disagreement with the 59 us recorded hours earlier from the same instrument. Two
readings of one method that differ by 7x mean the method is not measuring the
quantity, whatever it is called.

**The right instrument** (`gemv_shape --fixed`) varies the **pass count** at fixed
bytes, where the slope is per-dispatch work and the intercept is genuinely
per-dispatch fixed:

| shape | per-dispatch | streaming | command-buffer overhead |
|---|---|---|---|
| 1024x1024 (0.4 MB) | 9.3 us | 42.1 GB/s | 126.3 us |
| 5376x4096 (8.3 MB) | 56.1 us | 147.3 GB/s | 116.8 us |
| 5376x8192 (16.5 MB) | 104.4 us | 158.2 GB/s | 119.5 us |
| 5376x21504 (43.4 MB) | 261.2 us | 166.0 GB/s | 156.8 us |
| 21504x5376 (43.4 MB) | 269.1 us | 161.1 GB/s | 159.5 us |

The overhead is **flat across shapes** (117-203 us over two runs), which is the
signature of a real fixed cost: a page-walk term would scale with the buffer and
a launch term would not. Fit residuals 2.6-15%.

**The multiplier was wrong too, by 10x.** The spec multiplied a per-*buffer*
figure by the *matvec* count (420). Measured with `KAPLUMBA_COUNT_BUFFERS` on a
real 96-token run:

```
BUFFERS total=3990 steps=95 per_step=42.00
```

**42 command buffers per decode step**, from `MAX_DISPATCHES_PER_BUFFER = 50`
against ~1260 dispatches. This confirms a claim the code comment already made
("the ~42 rolled buffers in between") and had never measured.

**So the real budget:**

```
42 buffers x 155 us   =  6.5 ms per token
  = 29% of the 22.3 ms HOST wall per token
  =  7.3% of the 89.5 ms GPU span per token
```

and it is **host** time, spent while the GPU has 67 ms of queued work it is not
waiting on — the host runs **4.01x ahead** of the GPU (89.5 ms span / 22.3 ms
wall). The cost is real, it is 29% of the host's budget, and it is **completely
invisible** in the token time.

**T1's answer: the fixed cost is not a lever, and the spec's framing of it was
wrong in three ways** — the quantity (per-buffer, not per-matvec), the magnitude
(155 us, not 59), and the multiplier (42, not 420). What is left is 147-166 GB/s
of streaming, which is item 2 below.


## Three bugs the new tests found, all of which had teeth

1. **FMA contraction depends on register pressure, and NR0 changes register
   pressure.** With the accumulator written as
   `acc += x0*v0 + x1*v1 + x2*v2 + x3*v3`, `k8_mcg_r8` at 5376x21504 differed from
   NR0=1 in **2 of 21504 outputs**, one fp16 ulp, while every other
   (rate, NR0) pair across 192 comparisons was identical. The generator now emits
   an explicit `fma` chain. Cost at the rates the checkpoint dispatches: **3 bpw
   and 6 bpw are bit-identical to the pre-NR0 kernel** (verified by dumping both
   and diffing). 8 bpw still moves 1 ulp in 2 of 21504 and is unreachable here.

2. **Comparing kernel variants against each other cannot see a change that moves
   them all together.** Verified, not hypothesised: re-ordering the four-lane
   butterfly (`shuffle_xor 2` before `shuffle_xor 1` instead of after) is a
   *double* accumulation and changes **79325 outputs** across the harness's
   cases — and `decode_nr0.rs` **passed it**, because its baseline was rebuilt
   from the same edited source. Hence `tests/decode_matvec_golden.rs`, which
   holds the committed kernel's output as a reference digest. This is the single
   most transferable lesson from the session: *a bit-exactness test that
   recomputes its own baseline is not a bit-exactness test.*

3. **`Mutex` is not reentrant and my failure message deadlocked the test.**
   `assert_eq!(M.lock().len(), 0, "... {} ...", M.lock().len())` holds the first
   guard while evaluating the argument. The test hung for 60 s with a stack
   ending in `poison::lock`, which reads exactly like a GPU hang. It was not one.

## A coverage reduction I made that would have hidden a real bug

The first version of `decode_nr0.rs` swept 3 bpw over all eight shapes and only
four small shapes for the other seven rates, reasoning that building 115 M codes
eight times was too slow. Then:

- the 8 bpw / NR0=8 / 5376x21504 mismatch — the one combination that is *not*
  bit-exact — was in the part I had dropped;
- `examples/nr0_discriminate.rs` then showed the dropped shape builds in
  **0.21 s**, and the full sweep is a few seconds of host work.

A test left running over the whole cross product would have caught a real
re-association. The version written to save four seconds would not have, and
would have reported "bit-exact" while the claim was false. The suite now does the
full 8 x 8 x 3 and asserts the count is **exactly 192**, so a future shape that
does not divide evenly by some NR0 is a failure rather than a silent skip.

Generalisation, because this is the second time: *a coverage cut justified by a
runtime measurement needs the measurement to be about the thing that would be
missed, not about the thing that is expensive to run.*

## The first honest per-decode-step profile

`KAPLUMBA_DECODE_PROFILE=kernel` on Gxma, 39 recorded steps, own-time sum
**97.1 ms** against a **96.0 ms** step span from `=step` — so the attribution is
sound, where every prior per-kernel decode table was a prefill profile.

```
kernel          MB/tok   ms/tok    GB/s   %token  %bytes
gate            2601.1  22.526   115.5   18.7    21.8
up              2601.1  22.687   114.7   18.8    21.8
down            2601.1  23.294   111.7   19.3    21.8
o               1057.0  12.545    84.3   10.4     8.9
q               1057.0  12.064    87.6   10.0     8.9
k                478.9   6.843    70.0    5.7     4.0
v                478.9   5.947    80.5    4.9     4.0
lm_head         1057.0   5.337   198.0    4.4     8.9
other (attn, norms, rope, gelu, cache, sampling)  7.4
```

11.932 GB/token, 99.0 GB/s aggregate. The three big shapes are 65% of the bytes
at 113.9 GB/s. At 240 GB/s they would take 34 ms instead of 68.5 and the token
would be **~53 ms = 19 tok/s** — which is 1.5x on the matvecs, i.e. exactly the
3-vs-6 bpw gap. So T6 is the whole remaining prize at 3 bpw.

Also worth knowing: the host wall time for a step is 23.8 ms against a 96.0 ms
GPU span, so the GPU is ~4x oversubscribed by the host and **the host is not the
bottleneck**. Encoding is 25% of the step.

## The equal-bytes rate sweep cannot answer its own question

The last thing measured, and the most useful negative of the session.

`gemv_shape --only "bit rate"` is the table that motivates this entire spec — it
is where "3 bpw is 40% slower than 6 bpw at equal bytes" comes from. Run before
and after T6:

```
rate  has _w2?    pre-T6    post-T6    delta
3 bpw   YES        132.5      146.9     +10.9%
4 bpw   YES        179.8      153.7     -14.5%
6 bpw   no         164.5      182.0     +10.6%
8 bpw   no         176.3      162.3      -7.9%
```

**6 and 8 bpw cannot have changed from T6** — `TWO_WORD_SAFE_RATES` emits no `_w2`
kernel above 4 bpw, so those rows are **controls**, and they moved +10.6% and
-7.9%. A control that moves means the measurement is not measuring the kernel.
Within-sweep spreads are **2.2x to 3.7x** worst-to-best.

So the box drifts by >=10% between sweeps, and the drift is the same size as the
effect being looked for. **The instrument cannot resolve a 10% effect in either
direction**, and any conclusion drawn from it — in either direction — is
unsupported. I had drafted the conclusion "T6 halved the gap" from this table
before checking the controls. It does not support that.

The cause is structural, not a bug: the rate sweep measures **one shape to
completion at a time**, so any drift over the sweep is folded into the shape
number. The `--w2` and `--nr0` sweeps in the same binary do it correctly — every
round measures every arm once, with the baseline re-measured each round, so drift
appears as a moving *ratio*. That is why T6's own result (+7% end to end, +6.8%
to +11.2% microbenchmark) is trustworthy while the table it would extend is not.

**Lesson, and it is the same shape as every other one in this file:** the
instrument that produced the motivating observation is the one that had never been
stress-tested. The fix is to copy the interleaved structure and carry a control
arm that cannot change — here a 6 bpw shape in the same round as the 3 bpw shape,
which gives the drift for free.

Also worth recording: **the controls were only obvious because T6's safe-rate set
is derived rather than configured.** If I had hand-written "1, 2, 3, 4" into the
generator, there would have been no way to tell from the names alone which rows
were supposed to be unmovable, and the control argument would not have been
available. Deriving the set gave the experiment its own control group for free.

## The measurement traps, for whoever comes next
- **Seconds labelled as milliseconds.** `gemv_shape.rs` divided the batch time by
  `passes` and formatted the result as ms; it reported 131,512 GB/s. Its own
  PLAUSIBLE check caught it on the very first run. A self-test earns its keep
  before it is trusted, not after.
- The worst sample of an otherwise clean 9-sample run is routinely **1.6x** the
  best. Requiring the whole spread to be tight means refusing to report at all;
  requiring the *median* within 3% of the *best* is the check that works, and the
  spread is printed for information.
- **`encode_gemv` prefers ishizuki to EXL3.** See the top of this file. Three
  separate times this session I had to ask "which kernel is this?" and once the
  answer changed what a measurement meant.
- `assert_eq!(M.lock().len(), 0, "{}", M.lock().len())` deadlocks. One guard.
- A 16x-too-small working set in a bandwidth probe reports L2 bandwidth, not
  memory bandwidth; `gemv_shape.rs` rotates over a 1.6 GB working set for this
  reason and drops each case's buffers before building the next.
- `Mutex`-guarded shared state across `#[test]`s in one binary races on the
  one-shot mode caches; the accumulator is `#[doc(hidden)]` accessors for that
  reason.

## Open, and honestly uncertain

- **Nondeterminism persists**: 8 runs of a 64-token prompt give 3 distinct
  outputs. `rope` is fixed; the remaining source is unidentified. Until it is, no
  bit-exactness claim is fully verifiable. **A correctness bug, and the most
  important open item.**
- **The 59 us fixed per-matvec latency** is 24.8 ms/token and still
  undecomposed. Now a question about ishizuki, not EXL3.
- **Whether a lane's window can be word-aligned at KA=3** (T6) — the only lever
  that attacks the measured 4x request amplification rather than adding streams
  beside it. If it cannot, the only route to 19 tok/s is fewer bytes, which is
  T4 and a user decision.
- **`MTLResidencySet`** pinning is untested, not disproved — the API SIGSEGVs on
  this binding.
- The `gemv_shape` byte-axis sweep (4.1 MB -> 528 MB at 3 bpw, 57.6 -> 167.9
  GB/s) is the fixed term showing through, not a size effect: it fits the same
  affine line.
