# Decode: the 785 ms that is not the matvecs

## Problem

Decode runs at **1.18 tok/s** (≈ 850 ms/token). Per-step breakdown, measured
with `KAPLUMBA_PROFILE_PHASES=1`:

```
embed 0.17 ms | layers 783 ms | lm_head 9.3 ms | total 793 ms
  of which cpu_encode = 0.00 ms, gpu_wait = 783 ms
```

The linear GEMVs account for roughly **220 ms** of that (11.0 GB at the
measured 40–56 GB/s). So **~560 ms per token is not the matvecs**, and it is not
CPU submission cost — the CPU encodes all 60 layers in under a millisecond.

Two kernels are responsible. Both are written "one thread per row", which is
fine when `rows` is large and catastrophic when it is 1. **Decode is exactly the
case where `rows == 1`.**

### Finding 1: `rms_norm` uses one thread for the whole row

`crates/kernels/src/elements.metal:92`, `rms_norm_weighted`:

```metal
if (gid >= params.rows) return;      // one thread per ROW
uint base = gid * count;
for (uint i = 0; i < count; ++i) sum += value * value;   // serial over 5376
for (uint i = 0; i < count; ++i) output[...] = ...;       // serial again
```

`encode_rms_norm` dispatches `rows` threads. For a decode step `rows == 1`, so a
**single GPU thread** walks 5376 elements twice, serially, converting fp16 to
fp32 each time.

Measured against its siblings at identical shapes (`elem_bench`):

| Op | µs |
|---|---:|
| **`rms_norm` 1×5376** | **595.7** |
| `gated_gelu` 21504 | 3.0 |
| `rope` 8×512 | 3.2 |
| `copy_last_row` 1×5376 | 2.7 |
| `residual_add` 1×5376 | 2.3 |

It is **200× slower than a 9× larger op next to it.** There are 5 norms per layer
(`input_norm`, `q_norm`, `k_norm`, `pre_ffn_norm`, `post_attention_norm`) across
60 layers, plus the final norm: **301 per token × 596 µs ≈ 179 ms per token.**

`rms_norm_unweighted` (`pre_ffn_norm`) has the identical structure.

### Finding 2: `online_attention` uses 8 threads with a 512-float register array

`crates/kernels/src/attention.metal:14`:

```metal
if (gid >= params.queries * params.q_heads) return;   // 1 * 8 = 8 threads
float accum[512];                                      // 512 floats per thread
```

For decode this is **8 threads** on a 20-core GPU, each holding a 512-float
accumulator and scanning every key position serially, twice (once for the score
pass, once for the weighted sum). Sliding layers scan their full 1024-slot ring.
This is the second large slice of the unexplained 560 ms, and it gets worse with
context length, which matches the observation that decode slows as a conversation
grows.

### Finding 3: the roofline, measured rather than assumed

`273 GB/s / 12.06 GB = 22.6 tok/s` is the **bandwidth** roofline. It is not the
binding limit, because at 3 bpw a token must decode 32.2e9 weights and each
weight costs real integer and fp16 work.

The first version of this analysis guessed a 2.4 GHz clock and concluded 19
tok/s. That was a guess. `examples/alu_peak.rs` now measures this machine
directly, with four independent accumulator chains so the loop is issue-bound
rather than latency-bound, at 80k threads:

| Workload | Measured peak |
|---|---:|
| int32 chain, 8 ops/iter | 2.36e12 ops/s |
| EXL3 codebook shape (mul, mask, xor, 2 fp16 extracts, 2 adds) | 3.63e12 ops/s |

The MCG codebook is ~6 ops and dominates: one multiply, one mask, one xor, one
shift, two fp16 bit-pattern extractions and an add, producing one weight. That
caps the machine at roughly `3.63e12 / 6 = 605e9 weights/s`, and:

| Limit | tok/s |
|---|---:|
| Bandwidth roofline (273 GB/s) | 22.6 |
| **ALU roofline (measured, 100% efficiency)** | **~18.8** |
| Practical (70-80% ALU efficiency) | 13-15 |

So the binding constraint is the ALU, at **~19 tok/s absolute and 13-15
realistically**. 20 tok/s is above the ALU roofline: it would need more integer
throughput than this GPU has. This agrees with ishizuki, whose `BENCHMARKS.md`
reports **~11 tok/s on M4 Pro** and ~20 on M1 Max, tracking memory bandwidth but
sitting well under the naive roofline on both.

We are at 1.18 tok/s, so the gap is ~11x to the ALU roofline and ~9x to a
realistic target. Two things follow:

- The two serial kernels above are worth ~560 ms of the 850 ms and are the whole
  of the cheap part of this gap. Fix them first.
- After that the codebook itself becomes the wall. There is a known way out: at
  3 bpw only the top `ka` bits of each 16-bit code carry information, so the
  codebook is a function of `2^ka` distinct inputs and can be a small lookup
  table instead of a hash. That is a separate spec, and it is the only route I
  can see to exceeding ~19 tok/s.

## Goal

Fix the token-1 parallelism collapse. Gate: **≥ 5 tok/s**, which needs the layer
stack under ~200 ms. Stretch: 10 tok/s, ishizuki-class. The 20 tok/s target is
above this machine's measured ALU roofline and is not reachable by any amount of
kernel work; see Finding 3.

## Non-goals

- No quality loss. A parallel RMS reduction reorders the sum, so accept float
  tolerance, not bit equality — the same trade the reference makes.
- No change to the EXL3 matvec, which was just rewritten and measured at
  1.9–2.3× on the layer projections.
- No prefill regression. These kernels are shared, so prefill (`rows = tokens`)
  must not get slower; the fix must be a fast path chosen on `rows`.
- No CPU fallback, no Python, no speculative decoding.

## Requirements

- **R1** `rms_norm_weighted` and `rms_norm_unweighted` must reduce in parallel
  over the row. When `rows` is small, assign one **threadgroup** per row
  (e.g. 256 threads) and do a `threadgroup_barrier` tree reduction. When `rows`
  is large (prefill), keep one thread per row so prefill does not regress.
- **R2** Attention must parallelise over keys for decode. Split the key range
  across threadgroups (flash-decoding style), each producing a partial
  `(max, l, acc)` triple, then a second small pass combines the partials with the
  correct `exp(m - m_global)` rescaling. Do **not** use atomics on fp32.
- **R3** The `accum[512]` register array must not be per-thread when the grid is
  small; it is 2 KB of registers and is a large part of why occupancy collapses.
- **R4** Both fixes must be selected at dispatch on `rows` / `queries`, not
  applied unconditionally.
- **R5** Every element and attention kernel gets a cost measurement in
  `elem_bench`, so a future regression shows up as a microsecond number rather
  than as end-to-end slowness.
- **R6** Output must match the current build within float tolerance, verified by
  `output_identity`.

## Acceptance criteria

1. `rms_norm 1×5376` under 20 µs, down from 596.
2. `online_attention` for `queries=1, keys=200, 8 q-heads, 4 kv-heads, dim=512`
   under 200 µs, and scaling sub-linearly in key count.
3. Layer stack (`KAPLUMBA_PROFILE_PHASES`) under 250 ms.
4. End-to-end decode **≥ 5 tok/s**, stretch 10.
5. Prefill unchanged (currently ~8.4 tok/s at 134 tokens).
6. `output_identity` passes within tolerance; full suite green.

## Risks

- **Register pressure.** Tree reductions need a `threadgroup` array; 256 floats
  is fine, 5376 is not, so the reduction must be chunked.
- **Two code paths per kernel** doubles the test matrix. Mitigate by testing both
  paths explicitly at `rows = 1` and `rows = 64`.
- **Attention split-K changes the reduction order**, so logits shift in the last
  bits. `output_identity` may need a tolerance form; that is expected, not a
  regression.
- After these two fixes the GEMVs become the dominant cost again at ~220 ms, and
  the 20 tok/s target still needs a further ~4× on the matvec, which the ALU
  analysis says is not available. Treat 10–13 tok/s as the destination.

## Open questions

- How much of the remaining 560 ms is attention versus the norms? The norms are
  measured at 179 ms; attention is inferred from its structure, not yet timed.
  Add it to `elem_bench` as the first task.
- Does the sliding-window ring (1024 slots scanned for every sliding layer) need
  a maintained running bound, or is a split-K pass enough?
