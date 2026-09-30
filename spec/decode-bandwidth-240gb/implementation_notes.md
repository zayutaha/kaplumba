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

## The 79 us was 59 us, and the old method was measuring the wrong thing

`gemv_shape.rs --selftest` AFFINE, three sizes, least squares:

```
t = 58.9 us + 0.00608 ms/MB   ->  164.6 GB/s streaming,  58.9 us fixed
max residual 0.0001 ms = 0.0% of span
```

Reproduced. So the fixed per-matvec term is **59 us, not 79**, and 420 matvecs
x 59 us = 24.8 ms of a 96 ms token.

The old figure came from "one matvec per command buffer, idle verified", which
includes the CPU/GPU round trip. Two lessons:

- **"2x the bytes must take 2x the time" is the wrong check** when a real fixed
  cost exists. It failed here at 1.62x on a real effect, and I nearly "fixed" the
  instrument to satisfy it. The right statement is that time is **affine** in
  bytes, and the check is on the fit.
- **An affine intercept is the honest way to separate fixed from streaming**, and
  it hands you the streaming rate as a by-product.

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
