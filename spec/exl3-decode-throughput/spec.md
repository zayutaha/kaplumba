# EXL3 decode throughput: 1.4 tok/s to 20 tok/s

## Problem

Decode runs at **1.4 tok/s** on an M4 Pro. The weight-streaming roofline for
this model is **22.6 tok/s** (273 GB/s ÷ 12.06 GB/token). We are at **6% of
peak**, so roughly 16× is being lost inside the GEMV kernel.

Prefill is comparatively healthy (10 tok/s at 418 tokens), which is the tell:
prefill amortises one code extraction over many query rows, decode does not.

## Goal

Make decode memory-bound, the way the reference implementation is, and reach
**20 tok/s**. Intermediate gate: 10 tok/s.

## Research

### Reference: ishizuki (`github.com/struffl/ishizuki`)

`Sources/IshizukiKit/Core/EXL3.swift` contains the whole answer. Its file
comment is the design statement: *"EXL3 trellis-coded projections, decoded
inside the matvec rather than before it."*

Its `exl3_decode8` (EXL3.swift:190) decodes **eight codes from one window
setup**:

```metal
inline void exl3_decode8(const device uint* tile, uint lane, thread float* v) {
  uint lo = (lane * 8 + 1) * KA + RING - 16;   // integer case
  ... 3 word loads, build 64-bit windows a and c ...
  for (uint i = 0; i < 8; i++) {
    uint r = i * KA + 16;
    uint s = r <= 64 ? uint(a >> (64 - r)) : uint(c >> (96 - r));
    v[i] = exl3_decode(s & 0xffffu, CB);
  }
}
```

The window arithmetic is **identical to ours** — same `lo`, same `a`/`c`, same
`r = i*KA + 16`, same `& 0xffff`. Our extraction math is correct and is
bit-exact against their fixture. The difference is entirely in how often that
work is repeated and how the threads are shaped.

Its `exl3_matvec` (EXL3.swift:286):

```metal
constexpr int SG = 8;                       // 8 simdgroups per threadgroup
uint lane = thread_index_in_simdgroup;      // 0..31  -> one lane per thread
uint sg   = simdgroup_index_in_threadgroup; // 0..7   -> splits the K axis
uint tn   = threadgroup_position_in_grid.x; // output column-tile
const device uint* column = (const device uint*)trellis + tn * words;
for (uint tk = sg; tk < tiles_k; tk += SG) {          // strided over tile-rows
  float v[8];
  exl3_decode8(column + tk * tiles_n * words, lane, v);   // ONE window setup
  const device IT* xr = x + tk * 16 + r0;
  acc0 += x0*v[0] + x1*v[1] + x2*v[2] + x3*v[3];
  acc1 += x0*v[4] + x1*v[5] + x2*v[6] + x3*v[7];
}
// simd_shuffle_xor reduction over 4 lanes, then across the 8 simdgroups
```

Launch: `grid = (N/16) * 256`, `threadGroup = 256`.

Three structural properties we do not have:

1. **One window setup per 8 codes.** Ours: one per code.
2. **Thread count `(N/16)*256`.** For `q_proj` (N=5376) that is 86,016
   threads. Ours is one thread per output element: 5,376.
3. **The thread's `lane` matches the trellis's own 32-lane structure**, so the
   8 codes a lane owns scatter to rows `{r0, r0+1, r0+8, r0+9}` and columns
   `{c, c+8}` — and the 4 `x` values it needs are read once and reused for both
   column groups. Ours reads 16 `x` values per tile and uses each once.

The reduction is `simd_shuffle_xor` (xor 1, then 2) summing 4 lanes, then a
`threadgroup float red[8][vecs][16]` reduction across simdgroups. **No
atomics.**

### What ishizuki achieves, and what it implies

From their `BENCHMARKS.md`:

> Decode 17.5–20.1 tok/s … Decode roofline at 400 GB/s is ~46 tok/s.
> Decode scales with memory bandwidth: ~5 tok/s on M4, ~11 on M4 Pro,
> ~15–22 on M1/M4 Max, ~30 on M3 Ultra.

Their M4 Pro figure is ~11 tok/s. Their model streams ~24.8 GB/token at that
rate, i.e. `11 × 24.8 = 273 GB/s` — **100% of the M4 Pro's peak bandwidth.**
They are fully memory-bound.

Our model is lighter: 11.0 GB of layers + 1.06 GB `lm_head` = **12.06
GB/token**. Saturating the same 273 GB/s therefore gives **22.6 tok/s**. So
**20 tok/s is reachable, and the user's arithmetic is right** — provided we
reach the memory-bound regime they are in.

### Our measured achieved bandwidth

`crates/engine/examples/bench_gemv.rs`, one layer per shape, rate 3, MCG,
20 iterations after warmup:

| Shape | ms | GB/s | % of 273 | e9 weights/s |
|---|---:|---:|---:|---:|
| `q_proj` 5376×5376 | 0.65 | 16.6 | 6.1% | 44.2 |
| `up_proj` 5376×21504 | 1.78 | 24.4 | 8.9% | 65.1 |
| `down` 21504×5376 | 1.89 | 22.9 | 8.4% | 61.1 |
| `lm_head` 5376×262144 | 10.11 | 52.3 | 19.1% | 139.4 |

Two signals:

- **Everything is far below peak**, so the kernel is not bandwidth-bound. It is
  bound by the per-weight extraction work.
- **`lm_head` is 3× faster than `q_proj` with identical kernel structure.** The
  only difference is thread count (262,144 vs 5,376). That isolates occupancy as
  a second, independent bottleneck.

### Counting the redundant work in our inner loop

`linear_gemv_fast8` (linear.metal:689) runs, per 16×16 tile:

```metal
for t in 0..tiles:
  a0 += input[base+0]  * decode_state(exl3_code_at(packed, tile, 0,  col, ...));
  a1 += input[base+1]  * decode_state(exl3_code_at(packed, tile, 1,  col, ...));
  ... 16 calls, local_row = 0..15, local_column fixed ...
```

Each `exl3_code_at` does 3 word loads (`trellis_word` ×3) and 2 64-bit
shift/or pairs. So per 16 codes we issue **48 word loads and 32 64-bit shift
sequences**, where the reference issues **6 word loads and 4 64-bit shift
sequences** for the same 16 codes.

Secondary point: for a fixed `local_column`, `local_row` 0..15 maps to only
**4 distinct lanes** (`lane = (n%8)*4 + (k&6)/2`), so a perfect compiler could
in principle CSE to 4 window setups. It cannot, because the lane is buried
inside per-`local_row` index arithmetic it cannot prove. The reference sidesteps
this by making the lane the thread identity.

## Root cause

Two compounding defects, both in the decode GEMV, neither in the math:

1. **Window setup is not amortised.** One 64-bit window per code instead of per
   eight codes — roughly **8× the loads and 8× the 64-bit shifts**.
2. **Thread count is tied to output width, not tile structure.** `q_proj` gets
   5,376 threads for a 20-core GPU; the reference gets 86,016. Latency cannot be
   hidden.

The extraction arithmetic is correct and stays.

## Non-goals / restrictions

- **No quality loss.** Same logits. The reduction order changes
  (`shuffle_xor` tree vs sequential), so accept float tolerance, not bit
  equality — ishizuki notes a reordered sum is "neither consistently nearer",
  and their own kernel changed this way.
- **No CPU fallback, no Python, no requantization, no speculative decoding.**
- Prefill is not the target; its kernels are left alone unless a shared helper
  must change.
- `lm_head` already reaches 19% of peak and is the least urgent; it uses the
  same kernel so it comes along for the ride.

## Requirements

- **R1** One window setup per 8 codes. Add `exl3_decode8` to `linear.metal`,
  returning 8 decoded values for a given `(tile, lane)`.
- **R2** Thread mapping follows the reference: one thread per trellis lane,
  one threadgroup (256 threads = 8 simdgroups) per output column-tile, simdgroups
  striding the tile-row axis. Grid `(out_features/16) * 256`.
- **R3** Reuse the 4 `x` values (`r0`, `r0+1`, `r0+8`, `r0+9`) for both column
  groups of a lane, as the reference does.
- **R4** Reduction via `simd_shuffle_xor` within a simdgroup, then a
  `threadgroup` array across simdgroups. No atomics on the hot path.
- **R5** The existing specialised kernels (`linear_gemv_fast1..8`) are replaced
  by one rate-generic kernel templated on `KA`, or kept only if they measure
  faster. Duplicated per-rate copies of a 16-call inner loop are the problem, not
  the solution.
- **R6** Rate-generic over integer 1..8 and the MCG codebook (the real model's
  rate); fractional/mul1 may fall back to the old kernel until proven.
- **R7** Output must match the current implementation within float tolerance
  across all integer rates and both codebooks, on non-square shapes.
- **R8** `bench_gemv` reports achieved GB/s so the win is measured, not asserted.

## Acceptance criteria

1. `bench_gemv` shows `q_proj` and `up_proj` at **≥ 40% of 273 GB/s** (from 6–9%).
2. End-to-end decode **≥ 10 tok/s** on a 512-token prompt, greedy.
3. Stretch: **≥ 20 tok/s** (requires ≥ 88% of peak).
4. `linear_metal` and `exl3_metal` suites green; new tests cover rates 1..8 ×
   {Plain, Mcg}, square and non-square, several `out_features/16` values.
5. `output_identity` still passes — the five-prompt text must not change beyond
   float tolerance.
6. Prefill does not regress (currently 10 tok/s at 418 tokens).

## Risks and fallbacks

- **Occupancy vs register pressure.** 8 decoded floats + 8 accumulators per
  thread is heavier than today's 4. If occupancy collapses, stage `v[8]` in
  `threadgroup` memory or split into two `decode4` calls.
- **Small `out_features`.** Grid is `(N/16)*256`; for N=5376 that is 86,016
  threads, fine. For a tiny matmul it underfills the GPU — irrelevant here,
  where every projection is ≥ 5376 wide.
- **Reduction order changes results.** Covered by R7's tolerance, not equality.
- **The 8× extraction win may not be 8× end to end.** If decode lands at
  6–8 tok/s rather than 20, the remaining bottleneck is memory access pattern
  (each lane strided by 16 rows) rather than arithmetic, and the next lever is a
  tile-swizzled trellis layout so a simdgroup reads contiguous bytes. That is a
  separate spec; it requires repacking the checkpoint.

## Open questions

- Does the M4 Pro's 273 GB/s hold for a pure streaming read of mapped,
  page-aligned weight bytes, or is the practical figure lower? `bench_gemv`
  will answer this: if the new kernel plateaus near 150 GB/s, 20 tok/s is out of
  reach on this machine and the honest target is ishizuki's ~11 tok/s class.
  **This should be measured first**, since it decides whether the remaining work
  is worth doing.
