# Decode: 8.1 to 19 tok/s

Supersedes `spec/decode-20tok` and `spec/decode-tokened-parallelism`, both of
which are written against a 1.18 tok/s baseline and a 273 GB/s roofline. Both of
those numbers are now known to be wrong; see *Corrections* at the end.

## Goal

Decode runs at **8.1 tok/s** (123 ms/token) and moves **13.145 GB** of weights
per token. This spec exists to reach **19 tok/s** and to establish, with a
measurement that can be trusted, **where the 70 ms of unaccounted time goes**.

The target is not a number chosen for roundness. At the measured streaming rate
of 220 GB/s, 13.145 GB/token is 60 ms/token, which is 16.7 tok/s. 19 tok/s
requires either a higher streaming rate than we have measured, or fewer bytes per
token. **The spec is deliberately written to discover which, rather than to
assume it.**

## The one number everything hangs on

| | ms/token | tok/s |
|---|---:|---:|
| Now | 123 | 8.1 |
| At 220 GB/s, zero overhead | 60 | 16.7 |
| At 250 GB/s (disputed) | 53 | 19.0 |

Two independent questions, and conflating them has already cost this project
several days:

1. **Can the machine stream faster than 220 GB/s?** If 220 is a request-rate
   wall rather than a bandwidth wall, the achievable ceiling rises and 19
   becomes reachable. Evidence that it *might* be a wall: bandwidth is flat from
   8 to 32 bytes in flight per thread, which is the signature of a limit on
   requests issued rather than bytes moved. That sweep only went to 20,480
   threads and the device allows 81,920.
2. **Must we move 13.145 GB/token?** 13.145 is a property of the current
   design, not a law. At 220 GB/s every GB/token deleted is worth 4.5 ms, or
   about 0.6 tok/s. This is the lever that does not depend on question 1.

**Answering question 1 is cheap and comes first.** It bounds everything else.

## Requirements

**R1 — A decode timing instrument that can fail.** Decode's own_ms is flat at
~1270 ms across N=16/32/64/128, so it is not per-token decode time at all: the
load and prefill phases are being attributed to decode kernels. Every per-kernel
decode table in this project's history is therefore a prefill profile. A new
instrument must scale linearly with token count over at least N=16..128, must
exclude load and prefill, must resolve an injected cost, and must reproduce
~123 ms/token on a clean box. A number that cannot fail is not a measurement.

**R2 — Attribution of the 123 ms.** Split it into weights, KV attention,
everything else, and gaps, with milliseconds and percentages that sum to 123.
State the achieved GB/s per component against 220 GB/s.

**R3 — The streaming ceiling, settled.** Sweep concurrency well past 20,480
threads to the device maximum, sweep independent address streams from 1 to 64,
and vary bytes-in-flight and thread count as *separate* axes. If they are not
interchangeable, that is direct evidence of a request-rate limit. Report one best
number with median, best, worst and spread over n>=7.

**R4 — Bytes per token, costed.** Identify every mechanism that moves fewer
bytes per token and estimate the saving in ms/token. At minimum consider:
- `lm_head` is 1.057 GB at 6 bpw, 8% of all bytes, for one projection, and at
  262144 wide it is a different access shape from the other seven (5376-ish).
- Whether all 262144 vocabulary rows are needed. This is a vocabulary
  projection; rows that can never win an argmax are pure cost.
- Whether any weight is read more than once per token. An agent reported every
  weight is read exactly once, so re-reads are likely not a lever, but the claim
  was made on a contended box and is cheap to re-confirm.
- Fusing projections so intermediates do not round-trip, in case a fusion
  currently causes a re-read.

Note what is **not** a lever, each proven and each worth not revisiting: the
EXL3 rate is already exactly 3.000 bits/value on 410 of 411 tensors (lm_head is
6.000), so no repacking reduces bytes; page faults are 20/token, 0.003% of
12.06 GB; the weights already coalesce into 23 x 512 MB arena segments rather
than 411 separate buffers; compression is not stealing bandwidth (18G wired,
~450 MiB compressor, weights pinned); the box is not power- or thermally-limited
(sustained beat burst, no throttle warnings); and `maxTotalThreadsPerThreadgroup`
is 1024 even with 64 live float4, so occupancy is not register-limited.

**R5 — Bit-exactness preserved.** Decode is the one path where we currently have
exactness. No change may trade it. `cargo test --release -p kaplumba-engine`
20/20 suites. The five reference prompts stay exact: Paris / 1, 2, 3, 4, 5 /
red, blue, and yellow / Hello! How can I help you today? / 17 times 23 is
**391.** Tests must have teeth: a new test is not a claim until the code has
been sabotaged and the test has been seen to fail with a diagnosable message.

**R6 — `lm_head` stays file-backed and mapped.** The user has explicitly
rejected making it resident. Full weight residency is also rejected on
measurement: decode 8.70/8.93/8.99 default vs 7.59/8.31/8.80 resident, prefill
2.5-4.3x slower, +3.55 GiB.

## Acceptance criteria

1. A valid decode instrument exists, demonstrated to scale linearly with token
   count, to exclude prefill, to catch an injected cost, and to agree with
   ~123 ms/token on a clean box. Its noise floor and minimum detectable effect
   are stated.
2. The 123 ms is attributed into components summing to 123, each with its
   achieved GB/s.
3. The streaming ceiling is a single number with spread, and the spec states
   plainly whether it is a wall and what the real ceiling is.
4. If bytes-per-token is the only remaining lever, that is stated as a finding
   and the specific mechanisms are costed in ms/token, ranked.
5. A per-token cost of 19 tok/s (or a stated reason it is not physically
   available, with the number that shows why) is demonstrated with the valid
   instrument, with the five reference prompts exact.

## Restrictions

- No quality loss. Bit-exact decode (R5).
- No increase in memory footprint. The budget is 14 GiB on a 24 GiB box; one
  model instance already takes 11.7-13.0 GiB, so **only one instance fits at a
  time** and any second one will swap.
- No KV cache quantisation.
- `lm_head` file-backed (R6).
- Report median, not best, and never report a number taken below ~14 GiB
  free — swapping produces numbers that look real and are wrong.

## Open questions

- Is 220 GB/s a wall? Being answered (R3). Everything else is scoped by it.
- Where are the 70 ms? Being answered (R2). No candidate has been eliminated;
  the ones under consideration are dispatch/command-buffer gaps, per-thread
  bytes-in-flight at batch 1, dequant arithmetic not overlapped with loads, and
  `lm_head`'s access shape.
- Does the split count matter? An isolated harness measured full-attention
  layers at 8k running 1.91x faster at 12 splits than 16 (6975->5743 us and
  7301->5739 us, reproduced twice at 3 significant figures, zero byte cost).
  `decode_splits` returns 16 for both layer shapes, which gives sliding layers
  256 workgroups but full-attention only 64, on 20 cores. **The change is not
  bit-exact** (the combine sums partials in split order) so it needs the five
  prompts re-checked before landing, and it was measured in isolation, not
  end-to-end. `KAPLUMBA_DECODE_SPLITS` exists as a sweep knob.
- The 16 B/thread figure from the load-width sweep is **retracted**: that sweep
  found 8 B/thread already saturates, and the 30 B/thread figure that motivated
  a `uint4` change was based on a wrong threshold. The `uint4` change is
  therefore low-value and should not be pursued on those grounds.

## Corrections to prior specs

Recorded so they are not re-derived:

- **The roofline is 16.7 tok/s, not 22.6.** 22.6 came from dividing by the 273
  GB/s datasheet figure. 273 GB/s is arithmetic (256-bit x LPDDR5X-8533), not a
  measurement. Measured on a quiet box with a repaired probe: 220.8 / 218.1 /
  219.7 GB/s median at 6 GB, 228.6 at 2 GB, spread 0.8-0.9x, i.e. 80-84% of
  nominal. This matches independent evidence: NVIDIA's GB10 has an *identical*
  memory subsystem and measures 229-230 GB/s, and measured Apple silicon DRAM
  efficiency is 82-86%.
- **The 248.5 and 250.7 GB/s figures were artifacts.** The bandwidth probe
  reported min-of-passes, which on a shared box selects the one pass that ran
  while a competitor was idle. An agent watched the same binary swing
  79.9 -> 230.5 GB/s in 10-second intervals while load *fell*. Selecting the
  median fixed this.
- **`bw.rs` never wrote its buffer.** It allocated multi-GB and streamed
  untouched zero pages, which is not a memory-system measurement. Its 55 GB/s
  became the "achievable bandwidth" denominator for a full round of experiments.
  Repaired in `9906610`; the median/min fix in the same lineage.
- **Byte counts were wrong in places by ~1000x** when KB was used where the
  checkpoint is MB-scale. Authoritative per-token figures, read from the
  safetensors headers: down 2.872 GB, gate 2.872, up 2.872, o 1.230, q 1.230,
  k 0.527, v 0.485, head 1.057 = **13.145 GB/token** over 60 layers.
- **Decode's per-kernel `own_ms` is a prefill profile** (R1). Any prior
  statement of the form "component X is N% of a decode token" derived from it
  is void, including "KV attention is 9% at 614 and 50% at 8k" and "the 7
  matvecs are 98.6% of a decode token". The latter is also arithmetically
  impossible: it implies 265% of peak bandwidth.
- **Startup amortises over ~64 tokens.** Measured 5.2 tok/s at N=16, 6.9 at
  N=32, 8.1 at N=64, flat to N=256. Short generations read ~40% slower than
  steady state, which has skewed some comparisons.
