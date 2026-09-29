# Plan: decode 8.1 to 19 tok/s

Technical design for `spec/decode-19tok/spec.md`. Read the spec first; it
carries the corrected roofline and the corrections to prior specs.

## Sequencing: the order is forced by information, not preference

```
R3  streaming ceiling settled        <- bounds everything; cheap; do first
     |
     +--> if 220 is a WALL:
     |      R2 attribution -> target becomes 16.7, and R4 becomes the
     |      only route past it
     |
     +--> if 220 is PUSHABLE:
            R2 attribution -> recast the 70 ms against the new ceiling
```

R1 (the instrument) is a prerequisite for R2, and both are prerequisites for any
implementation. So the real order is **R1 -> R3 -> R2 -> R4 -> implement**.

R3 is cheap and runs alongside R1 because it needs no instrument: a standalone
probe cannot tell a bandwidth wall from a request-rate wall by itself, and its
answer determines whether the rest of the project is aimed at the kernels or at
the byte count.

## R1: the instrument

**Why the existing one fails.** `KAPLUMBA_GPU_PROFILE=1` uses `GPUStartTime` /
`GPUEndTime` selectors. Decode emits a very large number of small command
buffers, and the rolling encoder (`crates/kernels/src/encoder.rs`) reuses and
overwrites buffer state. The most likely mechanism is that decode's timestamp
queries resolve to command-buffer boundaries that all fall inside the
load+prefill window, so prefill's work is attributed to decode kernels. This
must be confirmed with `path:line` before any fix is attempted — the mechanism
determines whether the fix is in the profiler or in the encoder.

**The instrument's contract.** It must be shown to satisfy all four, not assumed
to:

| property | test |
|---|---|
| scales linearly with token count | linear fit over N=16/32/64/128 |
| excludes load and prefill | prefill time must not move when decode length changes |
| resolves an injected cost | add a deliberately slow dispatch, confirm it is seen, remove it |
| agrees with ground truth | reproduce ~123 ms/token on a clean box within a few percent |

Ground truth is the clean-box n=5 measurement: 7.841 / 8.176 / 8.062 / 8.050 /
7.947 tok/s at N=64, flat from N=64 to N=256.

**Reported alongside it:** noise floor and minimum detectable effect, so later
work knows what difference is real. Given the spread seen on this box across a
session (decode has been observed at 3.6, 5.2, 6.9, 7.5, 8.0, 8.1, 9.6 tok/s),
the minimum detectable effect is likely a few percent and should be stated, not
assumed.

## R3: the streaming ceiling

**The hypothesis under test.** Bandwidth is flat from 8 to 32 bytes in flight
per thread, which suggests a limit on *requests issued* rather than bytes moved.
If so, the flat region should be a property of request rate, and a different
concurrency shape should move it.

**Sweep, in this order:**

1. **Thread count, well past the old range.** The prior sweep stopped at
   20,480 threads (20 x 1024). The device allows `maxTotalThreadsPerThreadgroup`
   = 1024, so with 256-thread threadgroups the cap is 80 threadgroups = 20,480
   threads, and going higher requires blocks-per-thread. Sweep
   threads/threadgroup x threadgroups x blocks/thread and find the knee
   explicitly.
2. **Independent address streams, 1 to 64.** This is the untested axis and the
   one most likely to matter: weights live in 23 x 512 MB arena segments and a
   decode step reads many at once. The prior "multiple queues do not help"
   negative is a *different* question (queues vs addresses) and was taken on a
   contended box, so re-test it.
3. **Bytes-in-flight and thread count as separate axes** at fixed values of the
   other. If they are not interchangeable, that is direct evidence for a
   request-rate limit, and it says which one binds.
4. **Push, if anything exceeds 235 GB/s.** That is above NVIDIA's GB10
   (identical memory subsystem, 229-230 GB/s), so it would be a real finding.
   Try larger threadgroups, more blocks/thread, streams combined with high thread
   count, `cache::random` vs default, and 2 MB pages — a latency probe found
   4 KB-page TLB misses raise latency from ~98 ns at 16 MB to 423 ns at 512 MB,
   which at these working-set sizes is a plausible limiter.

**The trap history is mandatory reading for whoever runs this.** Every bandwidth
harness in this project had a bug before it was verified:

| harness | defect | false result |
|---|---|---|
| `bw.rs` | never wrote its buffer; streamed untouched zero pages | 55 GB/s, became the "achievable" denominator for a whole round |
| `bw.rs` (second) | reported min-of-passes | 248.5 / 250.7 GB/s |
| load-width probe | wrong CPU reference, twice (`uchar` wrap, then `uint2`/`uint4` output stride) | invalid |
| KV attention harness | scanned 7814 keys against an 7814-key count on a 1024-slot ring | 20,340 GB/s |
| KV attention harness | read a single L2-resident buffer (48 MB L2) | 1,592 GB/s |

So: build an independent reference, and **make it fail first**. Use
`crates/engine/examples/bw.rs` as repaired (`9906610`) and verify it writes
before trusting it.

## R2: attribution

**Method.** Wall-clock differential is the ground truth, because the instrument
must not be the thing under test. Vary one quantity, measure the delta, and
require linearity in that quantity:

| isolate | how |
|---|---|
| weights vs KV | vary context, 614 vs 8192; KV scales with L, weights do not |
| per-layer cost | synthetic 2-layer model; cost must be linear in layer count |
| `lm_head` | its own shape, 262144 wide at 6 bpw, 1.057 GB |
| dispatch gaps | compare a rolled run against a single-command-buffer run |

**Expected shape of the answer.** 13.145 GB at 220 GB/s is 60 ms; we take 123.
The 63 ms gap is the project. Candidates, all currently uneliminated:

- dispatch / command-buffer gaps. Note the encoder is already async
  (`MAX_IN_FLIGHT=8`, `MAX_DISPATCHES_PER_BUFFER=50`) and raising the window to
  16 changed nothing for prefill, but decode is far more latency-sensitive and
  that has not been tested for decode.
- per-thread bytes-in-flight at batch 1. A decode layer is 5376 wide, giving
  86,016 threads. Whether that is enough at 220 GB/s is unmeasured.
- dequant arithmetic not overlapped with loads. In prefill the dequant was
  measured free and removing it made the kernel 4.4% *slower*, but prefill is
  compute-bound and decode is not, so this does not transfer.
- `lm_head`'s access shape, which differs from the other seven and is 8% of bytes.

**The split-count question, separately.** `decode_splits` returns 16 for both
layer shapes, giving sliding layers 256 workgroups and full-attention layers
only 64, on 20 cores. An isolated harness measured full-attention layers at 8k
running 1.91x faster at 12 splits (6975->5743 and 7301->5739 us, reproduced twice
at 3 s.f.), at zero byte cost since the partials buffer is already sized for 16.
Requirements before this lands: it is **not bit-exact** (the combine sums in
split order), so measure the actual float divergence — if ~1e-7, that is four
orders of magnitude below the 2.16e-3 the project already accepts from prefill's
fp16 operand rounding — and re-check the five reference prompts. Use
`KAPLUMBA_DECODE_SPLITS` to sweep without rebuilding. A wave-quantisation
argument is the right justification, not the single 12-vs-16 data point.

## R4: bytes per token

**The arithmetic that makes this worth doing.** At 220 GB/s, 13.145 GB/token is
60 ms. Deleting 1 GB/token is worth 4.5 ms, or ~0.6 tok/s. Deleting the whole
`lm_head` would be 4.8 ms; halving it, 2.4 ms.

| mechanism | bytes at stake | notes |
|---|---|---|
| `lm_head` row pruning | up to 1.057 GB | a vocabulary projection; rows that cannot win an argmax are pure cost. Needs care: pruning by frequency is a quality decision, not a free win. |
| `lm_head` dtype | 1.057 GB at 6 bpw | double the density of the 3.000 bpw projections. Nothing to shrink without a quality cost. |
| weight re-reads | 0 | already verified read-exactly-once; re-confirm on a quiet box |
| projection fusion | 0 bytes | only worth it if it eliminates a re-read; measured no re-reads, so likely nothing |
| **speculative decoding** | **structural** | weights do not change between tokens. Amortising one weight read across several emitted tokens is the only lever that beats the byte-count floor rather than trimming it. Verify-gated, greedy-identical, per AGENTS.md §8. |

That last row is the one worth serious thought if R3 comes back "wall" and R4's
rows above are exhausted. It is a different project from the others and is named
here so it is not lost.

## Risks

- **Contention is the dominant risk to this spec.** Every number in it was taken
  on a box shared with other agents at some point, and the two largest errors
  found today (248.5 GB/s, the prefill-shaped decode profile) were both caused
  by it rather than by code. Rule: one GPU-touching agent at a time; `gpuq
  status` >= 14 GiB before every model run; never report a median from a run that
  swapped; record `top`'s "unused" beside every sample.
- **Probe memory.** A concurrency probe was observed holding 3.8 GB of RSS
  without freeing between configurations, driving the box to 3.7 GiB and into
  573 MB of swap. Probes must free per configuration, and a leaked probe is
  grounds to kill it immediately — swapped measurements are worse than none.
- **The ceiling may be lower than 220 under sustained in-model conditions**,
  where the working set is far larger than a 2-6 GB probe buffer. R3's number is
  a probe number; the acceptance test in the spec is an in-model one.
- **Byte-count targets may be quality-affecting.** Vocabulary pruning and
  anything touching `lm_head`'s precision are quality decisions. R5 is not
  negotiable; if a mechanism only works by degrading output, it is out.

## Requirement to task mapping

| requirement | satisfied by |
|---|---|
| R1 valid instrument | T1 |
| R2 attribution of 123 ms | T3 |
| R3 streaming ceiling | T2 |
| R4 bytes per token costed | T4 |
| R5 bit-exactness | every task; enforced in T5 |
| R6 `lm_head` mapped | respected throughout; verified in T5 |
| acceptance 1 | T1 |
| acceptance 2 | T3 |
| acceptance 3 | T2 |
| acceptance 4 | T4 |
| acceptance 5 | T5 |
