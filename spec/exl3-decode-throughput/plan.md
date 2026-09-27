# Plan: EXL3 decode throughput

## Target architecture

Adopt ishizuki's `exl3_matvec` structure. Our extraction arithmetic is already
correct and bit-exact; only the loop shape and thread mapping change.

### Thread mapping

```
grid      = (out_features / 16) * 256
threadgroup = 256                       // 8 simdgroups

tn  = threadgroup_position_in_grid.x     // output column-tile, 0 .. out/16-1
sg  = simdgroup_index_in_threadgroup     // 0..7, strides the tile-row axis
lane = thread_index_in_simdgroup         // 0..31, one trellis lane per thread
r0  = (lane % 4) * 2
```

For `q_proj` (out=5376): 336 column-tiles × 256 = **86,016 threads**, against
5,376 today.

### Inner loop

```
column = trellis + tn * words            // this column-tile's ring
for (tk = sg; tk < tiles_k; tk += 8):
    exl3_decode8(column + tk * tiles_n * words, lane, v)   // ONE window setup
    xr = x + tk * 16 + r0
    x0 = xr[0]; x1 = xr[1]; x2 = xr[8]; x3 = xr[9]
    acc0 += x0*v[0] + x1*v[1] + x2*v[2] + x3*v[3]
    acc1 += x0*v[4] + x1*v[5] + x2*v[6] + x3*v[7]
```

Note the `x` reads match the scatter exactly: code `i` lands on row
`rows[i%4]` and column `+ (i/4)*8`, so a lane needs 4 `x` values and uses each
twice.

### Reduction

```
for m in 0..vecs:
    a = acc0[m]; b = acc1[m]
    a += simd_shuffle_xor(a, 1); a += simd_shuffle_xor(a, 2)
    b += simd_shuffle_xor(b, 1); b += simd_shuffle_xor(b, 2)
    if (lane % 4 == 0):
        red[sg][m][lane/4]     = a
        red[sg][m][lane/4 + 8] = b
threadgroup_barrier(mem_flags::mem_threadgroup)
if (t < vecs*16): y[m*N + tn*16 + c] = sum over sg of red[sg][m][c]
```

Lane 0 of each group of 4 covers columns `lane/4` (0..7) and `lane/4+8` (8..15),
which is exactly the reference's column mapping.

### `exl3_decode8`

Same body as `exl3_code_at` but returning 8 values, so the window arithmetic
runs once:

```metal
static inline void exl3_decode8(
    const device ushort *packed, uint tile_index, uint lane,
    uint word_count, uint rate, uint codebook, thread float *v)
{
    const uint ka = rate, ring = 256u * ka;
    uint lo = (lane * 8u + 1u) * ka + ring - 16u;
    uint i0 = lo / 32u, sh = lo % 32u;
    uint j0 = wrap(i0, word_count), j1 = wrap(i0+1, word_count), j2 = wrap(i0+2, word_count);
    uint base = tile_index * word_count;
    ulong w0 = trellis_word(packed, base + j0);
    ulong w1 = trellis_word(packed, base + j1);
    ulong w2 = trellis_word(packed, base + j2);
    ulong a = ((w0 << 32) | w1) << sh | (sh ? (w2 >> (32 - sh)) : 0);
    ulong c = ((w1 << 32) | w2) << sh;
    for (uint i = 0; i < 8; ++i) {
        uint r = i * ka + 16u;
        uint s = (r <= 64u) ? uint(a >> (64u - r)) : uint(c >> (96u - r));
        v[i] = decode_state(s & 0xffffu, codebook);
    }
}
```

`trellis_word` (two 16-bit loads) is kept from the alignment fix; the reference
casts to `uint*` because MLX tensors are always aligned, which we cannot assume
for mapped checkpoint bytes.

## Touchpoints

| File | Change |
|---|---|
| `crates/kernels/src/linear.metal` | add `exl3_decode8`; add `linear_gemv_exl3`; keep `exl3_code_at` for the prefill kernels |
| `crates/kernels/src/linear.rs` | register the new pipeline; route integer+MCG to it |
| `crates/kernels/src/linear.metal` | retire `linear_gemv_fast1..8` once the generic kernel wins |
| `crates/engine/tests/linear_metal.rs` | new tests: rates 1..8 × {Plain, Mcg}, non-square, several tile widths |
| `crates/engine/examples/bench_gemv.rs` | already reports GB/s; add a decode tok/s projection |
| `crates/engine/tests/output_identity.rs` | must still pass (tolerance, not equality) |

`linear_gemv_tiled` (prefill) and `decode_state` are untouched.

## Sequencing

1. **Measure the practical bandwidth ceiling** (answers the open question first).
   Time a plain streaming reduction over the 11 GB of mapped layer weights with no
   EXL3 work at all. If that plateaus near 273 GB/s, 20 tok/s is live; if it
   plateaus near 150, the target is ~11 tok/s and the spec should say so.
2. Add `exl3_decode8` and unit-test it against the existing CPU decoder.
3. Add `linear_gemv_exl3` with the reference thread mapping and reduction.
4. Correctness: rates 1..8, both codebooks, non-square shapes, multiple tile
   widths, against the CPU reference.
5. `bench_gemv` comparison, old vs new, same shapes.
6. End-to-end decode tok/s at 512-token prompt.
7. Only then remove `linear_gemv_fast1..8`.
8. Update notes, commit per step.

## Risks

- Register pressure: 8 `v` + 8 accumulators per thread. Watch occupancy; stage
  in `threadgroup` memory or split to `decode4` if threads-per-simdgroup drops.
- The `red[8][vecs][16]` array is `vecs * 128` floats; for decode `vecs == 1`,
  so 128 floats = 512 bytes per threadgroup. Cheap.
- Reduction order differs from today, so results differ in the last float bits.
  Tests use tolerance; `output_identity` may need a tolerance-based variant.

## Requirement mapping

| Requirement | Task |
|---|---|
| R1 `exl3_decode8` | step 2 |
| R2 thread mapping | step 3 |
| R3 `x` reuse | step 3 |
| R4 shuffle reduction | step 3 |
| R5 retire per-rate copies | step 7 |
| R6 rate-generic, MCG first | steps 3–4 |
| R7 tolerance tests | step 4 |
| R8 measured GB/s | step 5 |
