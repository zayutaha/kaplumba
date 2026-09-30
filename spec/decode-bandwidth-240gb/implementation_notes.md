# Implementation notes: 240 GB/s

Running log for `spec/decode-bandwidth-240gb/`. Appended as work proceeds
(AGENTS.md §5). Dated 2026-09-29; decode moved 8.1 -> 10.0 tok/s this session.

## Where decode stands

10.0 tok/s, 109 ms/token, from 8.1 at session start. Landed, all bit-exact,
23/23 engine suites, five reference prompts exact:

| commit | change | effect |
|---|---|---|
| `9bd62a9` | `rope` cross-warp race — one thread per rotation PAIR, not per column | correctness; the race made greedy decode nondeterministic in ~2/8 runs of a 64-token prompt |
| `3439d41` | codebook specialised into the pipeline index `[aligned][bits][codebook]` | **8.1 -> 9.2**; 2.61x synthetic, `lm_head` 121.6 -> 231.0 GB/s |
| `a15882d` | simdgroup-per-block input transform (4 elements/lane, no thread-local memory) | **9.2 -> 10.0**; was 42 threads x `float values[128]` spilling, 240 dispatches/token |

## The single most useful measurement of the day

Reading at step 1/2/4 while the hardware fetches every sector:

```
step 1 -> 225.1 GB/s useful, 225.1 GB/s sectors, 100% used
step 2 -> 113.0 useful,        226.0 sectors,  50% used
step 4 ->  57.1 useful,        228.5 sectors,  25% used
```

Flat sector rate, halved useful rate. **The DRAM wall is ~226-228 GB/s of
sectors and is below the instruction stream.** Instructions are flat to ~16
uop/byte; EXL3 runs at 1.35 uop/byte, 8x under the knee.

This is what makes 250 GB/s and 20 tok/s unavailable rather than merely hard,
and it is why every "make the kernel faster" idea has been a dead end.

## Four numbers that were believed and were wrong

| figure | what it was |
|---|---|
| 55 GB/s | `bw.rs` streamed a buffer it never wrote. Became the "achievable" denominator for a whole round. |
| 248 GB/s | min-of-passes on a shared box. The same binary swung 79.9 -> 230.5 GB/s in 10 s while load fell. |
| 254 GB/s | `layer_bench.rs:78` hardcodes `rate: Integer(3)`, so its "lm_head" was a 3 bpw / 528 MB stand-in. The real head is 239.6. |
| 14 ms attention | **6x too high.** Measured 2.2 ms @614, 3.8 ms @8k. |

The last two were mine: I briefed two agents on a "KV attention is 50% of a
decode token" figure that the profiler produced, and on a 254 GB/s ceiling that
was a mislabelled benchmark. Both were corrected only when measured directly.

## The profiler never measured decode at all

`KAPLUMBA_GPU_PROFILE=1` reported a flat ~1270 ms at N=16/32/64/128, so it
cannot be per-token decode time. Cause: `GpuProfile` is constructed inside
`forward_buffer_into` (`model.rs:2590`) once per call, a decode step is a
one-token call, and the gate at `model.rs:2588-2589` requires
`KAPLUMBA_GPU_PROFILE_DECODE` for that case. Nothing was stamped; the table
printed came from the **prefill's** single call. Every per-kernel decode number
in this project's history is a prefill profile, including ones quoted in earlier
notes here.

A validated instrument exists in `/tmp/wt-tm` (`KAPLUMBA_DECODE_PROFILE=step`,
scales 10.3x for 8x work, excludes prefill to 0.2%, resolves an injected cost to
0.7%, three sabotage modes fail diagnosably). **Unlanded.**

## Two bugs found in a single three-line kernel

`greedy_pick` packed the token index into **16 bits** while the vocabulary is
**262,144**, so `0xFFFFu - index` underflowed past 65535 and set the high bits —
any logit above index 65535 won the argmax regardless of value. And `~bits` on a
`uint` left `0xFFFF` above bit 15, which the old `sortable << 16` discarded, so
**all negative logits compared equal**. And `*(const device uint*)best` read the
atomic *object* rather than its value, returning the same index for every input.
Fixed as a two-pass reduction (a 64-bit key does not compile — Metal declares
neither `simd_shuffle_xor` nor `atomic_fetch_max_explicit` for `ulong`).
Currently **latent**: the live path is a host `argmax_logits`.

## The measurement traps, for whoever comes next

- A `uint code[8]` local array written in one `#pragma unroll` loop and read in
  another **miscompiled** — 32762/32768 wrong while the harness reported success.
- FMA contraction is not stable: `acc0 += x0*v0 + x1*v1` re-associates, so two
  variants each matching baseline alone differed when combined (30818/32768).
- A bit-exactness generator using an LCG of period 8 made every index
  perturbation that was a multiple of 8 invisible.
- A KV harness scanned 7814 keys against a 1024-slot ring's byte count and
  reported **20,340 GB/s**; another read a 48 MB L2-resident buffer and
  reported **1,592 GB/s**.
- `conc.rs`'s min-of-passes selection was how 248 got believed; a drift-only
  gate once accepted a whole sweep at 7.9 GB/s.
- **`ru_maxrss` under-reports by 5.5 GiB** (6.23 reported vs 11.69 real): Metal
  buffer pages are not counted as resident. Renamed to
  `peak_resident_pages_bytes`; use `footprint -p <pid>`.
- A model load works at ~6.7 GiB "available" with 0 swap, because the model sits
  in clean reclaimable cache. An over-strict 14 GiB gate kept two agents from
  ever validating anything.
- 85 stale worktrees in `/tmp` were worth 573 MB of swap and 11 GiB of
  apparent free memory.

## Open, and honestly uncertain

- **Nondeterminism persists**: 8 runs of a 64-token prompt give 3 distinct
  outputs. `rope` is fixed; the remaining source is unidentified. Until it is,
  no bit-exactness claim is verifiable, which is the most important open item
  and it is a correctness bug, not a performance one.
- **The 79 us fixed per-matvec latency** is 33 ms/token and undecomposed. Its
  natural fixes are all measured dead, so the decomposition must come first.
- **N_R0=4 rows** measured +88% on the pre-specialisation kernel; whether it
  still pays is unknown.
- **`MTLResidencySet`** pinning is untested, not disproved — the API SIGSEGVs on
  this binding.
