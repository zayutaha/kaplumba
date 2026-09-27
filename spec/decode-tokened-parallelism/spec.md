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

### Finding 3: the roofline we have been using is the wrong one

`273 GB/s ÷ 12.06 GB = 22.6 tok/s` assumes decode is bandwidth-bound. It is not.
At 3 bpw, one token requires decoding **32.2e9 weights**, and each weight costs
roughly 10 integer ops (two shifts for the sliding window, a multiply, an xor,
two fp16 adds, one fma). M4 Pro's integer throughput is about 6.1e12 int32
ops/s (20 cores × 4 SIMD × 32 lanes × ~2.4 GHz), so the ALU roofline is
**~19 tok/s at 10 ops/weight and ~13.6 at 14** — and 22.6 tok/s would need
1.2× the machine's entire integer throughput.

The reference implementation agrees: ishizuki's `BENCHMARKS.md` reports
**~11 tok/s on M4 Pro** and ~20 on M1 Max, i.e. it tracks memory bandwidth but
sits well under the naive roofline on both. So:

- 22.6 tok/s is **not** physically available for a 3 bpw model on this machine.
- ~11–13 tok/s is the credible target, and matching ishizuki is a reasonable
  definition of success.
- We are at 1.18, so the gap to close is roughly **9×**, and the two kernels
  above are worth ~560 ms of the 850 ms.

## Goal

Fix the token-1 parallelism collapse. Gate: **≥ 5 tok/s**, which needs the layer
stack under ~200 ms. Stretch: 10 tok/s, ishizuki-class.

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
