#include <metal_stdlib>
using namespace metal;

struct LinearParams {
    uint in_features;
    uint out_features;
    uint tile_columns;
    uint packed_words;
    uint rate;
    uint codebook;
    uint queries;
    uint perm_mode;
};

struct PrefillRowParams {
    uint in_features;
    uint out_features;
    uint queries;
    uint perm_mode;
};

static uint read_word(const device ushort *packed, uint index) {
    return uint(packed[index * 2]) | (uint(packed[index * 2 + 1]) << 16);
}

static uint integer_window(const device ushort *packed, uint index, uint bits) {
    uint total = 256u * bits;
    uint b0 = index * bits + bits + total - 16u;
    uint b1 = b0 + 16u;
    uint i0 = b0 / 32u;
    uint i1 = (b1 - 1u) / 32u;
    uint shift = (i1 + 1u) * 32u - b1;
    uint word_count = total / 32u;
    ulong merged = (ulong(read_word(packed, i0 % word_count)) << 32) | ulong(read_word(packed, i1 % word_count));
    return uint((merged >> shift) & 0xffffu);
}

static uint fractional_width(uint index, uint bits) {
    return bits + ((0xAAAAu >> (index & 15u)) & 1u);
}

static uint fractional_end(uint index, uint bits) {
    uint depth = 16u * bits + 8u;
    uint value = (index >> 4u) * depth;
    for (uint i = 0; i <= (index & 15u); ++i) {
        value += fractional_width(i, bits);
    }
    return value;
}

static int normalize_ring(int value, uint length) {
    int result = value % int(length);
    if (result < 0) {
        result += int(length);
    }
    return result;
}

static uint fractional_window(const device ushort *packed, uint index, uint bits) {
    uint depth = 16u * bits + 8u;
    uint word_count = depth * 16u / 32u;
    int start = normalize_ring(int(fractional_end(index, bits)) - 16, depth * 16u);
    uint word_index = uint(start) / 32u;
    uint offset = uint(start) & 31u;
    ulong merged = (ulong(read_word(packed, word_index)) << 32) | ulong(read_word(packed, (word_index + 1u) % word_count));
    return uint((merged >> (48u - offset)) & 0xffffu);
}

// Threadgroup-resident twin of `fractional_window`, for the tiled prefill
// kernel, which stages a tile's 32-bit words in shared memory. `word_count` is
// in 32-bit words (== rate.packed_words() / 2, i.e. 8*bits + 4 for a fractional
// rate) and `index` is the plain row-major tile position `row * 16 + column`,
// matching `decode_tile`'s `tile[row * 16 + column]`. Bit-exact with
// `unpack_fractional`: same ring, same window start, same 16 bits, same order.
static uint fractional_window_tg(const threadgroup uint *tile, uint word_count, uint index, uint bits) {
    uint depth = 16u * bits + 8u;
    int start = normalize_ring(int(fractional_end(index, bits)) - 16, depth * 16u);
    uint word_index = uint(start) / 32u;
    uint offset = uint(start) & 31u;
    ulong merged = (ulong(tile[word_index]) << 32) | ulong(tile[(word_index + 1u) % word_count]);
    return uint((merged >> (48u - offset)) & 0xffffu);
}

static float decode_state(uint state, uint codebook) {
    uint value = state;
    if (codebook == 0u) {
        value = value * 89226354u + 64248484u;
        value = 0x3b603b60u ^ (value & 0x8fff8fffu);
        return float(as_type<half>((ushort)value) + as_type<half>((ushort)(value >> 16)));
    }
    if (codebook == 1u) {
        value = value * 0xCBAC1FEDu;
        value = 0x3b603b60u ^ (value & 0x8fff8fffu);
        return float(as_type<half>((ushort)value) + as_type<half>((ushort)(value >> 16)));
    }
    value = value * 0x83DCD12Du;
    uint sum = 0x6400u + (value & 0xffu) + ((value >> 8) & 0xffu) + ((value >> 16) & 0xffu) + ((value >> 24) & 0xffu);
    half byte_value = as_type<half>((ushort)sum);
    half inverse = as_type<half>((ushort)0x1eeeu);
    half bias = as_type<half>((ushort)0xc931u);
    return float(half(float(byte_value) * float(inverse) + float(bias)));
}

// Map a logical (row, column) of the 16x16 tile to its state index in the
// trellis. Mode 0 is plain row-major, which the reference packer implies; the
// remaining modes are the plausible fragment shuffles, selectable so the
// coherence oracle can identify the real one.
static uint permutation_index(uint linear, uint mode) {
    uint row = linear / 16u;
    uint col = linear % 16u;
    if (mode == 0u) {
        return linear;
    }
    if (mode == 1u) {
        return col * 16u + row;
    }
    if (mode == 2u) {
        // CUDA reconstruct shuffle: r0=(lane%4)*2, c0=lane/8, c1=c0+4.
        uint p = (row % 8u) / 2u;
        uint sub = (row % 8u) % 2u + 2u * (row / 8u);
        uint q = col % 4u;
        uint csel = col / 4u;
        return 64u * q + 8u * p + sub + 4u * csel;
    }
    if (mode == 3u) {
        // Original in-tree shuffle, kept so it can be re-tested.
        uint t = linear / 8u;
        uint j = linear % 8u;
        uint r = ((t & 3u) * 2u) + (j & 1u) + (((j >> 1) & 1u) * 8u);
        uint c = (t >> 2u) + (((j >> 2) & 1u) * 8u);
        return r * 16u + c;
    }
    if (mode == 4u) {
        return row * 16u + ((col + 4u) & 15u);
    }
    if (mode == 5u) {
        return row * 16u + (col % 4u) * 4u + (col / 4u);
    }
    if (mode == 6u) {
        return row * 16u + (col % 8u) * 2u + (col / 8u);
    }
    if (mode == 7u) {
        // bit-reverse the 8-bit index
        uint v = linear;
        uint r = 0u;
        for (uint i = 0u; i < 8u; ++i) { r = (r << 1u) | (v & 1u); v >>= 1u; }
        return r;
    }
    if (mode == 8u) {
        return ((row * 16u + col) & 0xF0u) | (((row * 16u + col) >> 4u) & 0x0Fu);
    }
    if (mode == 9u) {
        // column blocks of 8 swapped
        return row * 16u + ((col + 8u) & 15u);
    }
    return linear;
}

// One big-endian 32-bit trellis word, read as two 16-bit loads so it works at any
// 2-byte-aligned address. See the alignment note in `exl3_code_at`.
static inline uint trellis_word(const device ushort *packed, uint index) {
    return uint(packed[index * 2u]) | (uint(packed[index * 2u + 1u]) << 16);
}

// Read one 16-bit EXL3 code for logical (local_row, local_column) of a 16x16 tile.
//
// The reference producer lays the 256 codes out as eight-per-lane out of a
// 64-bit sliding window that overlaps by (16 - bits) per step, then scatters lane
// L's code i to tile row ((L%4)*2 + {0,1,8,9})[i%4] and column L/4 + (i/4)*8.
// Reading `bits`-wide codes at `index * bits` instead decodes to noise.
static uint exl3_code_at(
    const device ushort *packed,
    uint tile_index,
    uint local_row,
    uint local_column,
    uint word_count,
    uint rate) {
    const uint ka = rate;
    const uint ring = 256u * ka;
    uint p = (local_row % 8u) / 2u;
    uint i_mod4 = (local_row % 2u) + 2u * (local_row / 8u);
    uint q = local_column % 8u;
    uint i_div4 = local_column / 8u;
    uint i = i_mod4 + 4u * i_div4;
    uint lane = p + 4u * q;
    uint lo = (lane * 8u + 1u) * ka + ring - 16u;
    uint i0 = lo / 32u;
    uint sh = lo % 32u;
    uint j0 = i0 >= word_count ? i0 - word_count : i0;
    uint j1 = j0 + 1u >= word_count ? j0 + 1u - word_count : j0 + 1u;
    uint j2 = j1 + 1u >= word_count ? j1 + 1u - word_count : j1 + 1u;
    // Combine two 16-bit loads rather than casting to `uint *`. Trellis data is
    // not 4-byte aligned in the checkpoint (lm_head.trellis starts at offset
    // % 4 == 2), so a `uint *` cast reads every word at the wrong address. The
    // arena copy hid this by re-aligning the data as a side effect; mapping the
    // file does not, which is why every zero-copy build decoded to noise.
    uint base = tile_index * word_count;
    ulong w0 = ulong(trellis_word(packed, base + j0));
    ulong w1 = ulong(trellis_word(packed, base + j1));
    ulong w2 = ulong(trellis_word(packed, base + j2));
    ulong a = ((w0 << 32) | w1) << sh | (sh ? (w2 >> (32 - sh)) : 0);
    ulong c = ((w1 << 32) | w2) << sh;
    uint r = i * ka + 16u;
    return (r <= 64u ? uint(a >> (64u - r)) : uint(c >> (96u - r))) & 0xffffu;
}

// Decode all eight codes a single trellis lane owns, from ONE window setup.
//
// This is the whole performance story of EXL3 decode. The reader's window setup
// (three word loads plus two 64-bit shift/or pairs) is identical for all eight
// codes of a lane, because the window advances by one whole rate per code and
// the codes are read from overlapping 16-bit slices of it. Re-deriving the window
// per code -- as `exl3_code_at` forces a caller to do -- issues eight times the
// loads and eight times the 64-bit shifts for identical output.
//
// `lane` is 0..31; the eight values are scattered by the caller to tile rows
// {r0, r0+1, r0+8, r0+9} and columns {c, c+8}.
// `ka` is passed as a literal by the rate-specialised callers below. Metal has no
// templates, so `static inline` plus a literal argument is how the shift amounts
// and ring size get folded to constants.
template <bool ALIGNED>
static inline uint ring_word(const device ushort *packed, uint index) {
    if (ALIGNED) {
        // One 32-bit load. Valid only when `packed` is 4-byte aligned, which holds
        // for every arena-resident tensor (offset 0 in a page-aligned buffer).
        return ((const device uint *)packed)[index];
    }
    // Two 16-bit loads. Required for mapped checkpoint bytes, which are not
    // 4-byte aligned: `lm_head.trellis` begins at a file offset of % 4 == 2.
    return trellis_word(packed, index);
}

template <bool ALIGNED>
static inline void exl3_decode8(
    const device ushort *packed,
    uint tile_index,
    uint lane,
    uint ka,
    uint codebook,
    thread float *v) {
    const uint ring_words = ka * 8u;
    const uint ring = ring_words * 32u;
    const uint lo = (lane * 8u + 1u) * ka + ring - 16u;
    const uint i0 = lo / 32u;
    const uint sh = lo % 32u;
    const uint j0 = i0 >= ring_words ? i0 - ring_words : i0;
    const uint j1 = j0 + 1u >= ring_words ? j0 + 1u - ring_words : j0 + 1u;
    const uint j2 = j1 + 1u >= ring_words ? j1 + 1u - ring_words : j1 + 1u;
    const uint base = tile_index * ring_words;
    const ulong w0 = ulong(ring_word<ALIGNED>(packed, base + j0));
    const ulong w1 = ulong(ring_word<ALIGNED>(packed, base + j1));
    const ulong w2 = ulong(ring_word<ALIGNED>(packed, base + j2));
    const ulong a = ((w0 << 32) | w1) << sh | (sh ? (w2 >> (32 - sh)) : 0ul);
    const ulong c = ((w1 << 32) | w2) << sh;
    // Without this the 8-iteration loop cannot be unrolled, so `v[8]` spills and
    // the shifts serialise. ishizuki's exl3_decode8 has it.
    #pragma unroll
    for (uint i = 0u; i < 8u; ++i) {
        const uint r = i * ka + 16u;
        const uint s = (r <= 64u) ? uint(a >> (64u - r)) : uint(c >> (96u - r));
        v[i] = decode_state(s & 0xffffu, codebook);
    }
}

static uint packed_state(const device ushort *packed, uint tile_index, uint local_row, uint local_column, uint words, uint rate, uint mode) {
    if (rate < 16u) {
        return exl3_code_at(packed, tile_index, local_row, local_column, 256u * rate / 32u, rate);
    }
    uint lane = permutation_index(local_row * 16u + local_column, mode);
    const device ushort *tile = packed + tile_index * words;
    return fractional_window(tile, lane, rate - 16u);
}

static void transform_vector(thread float *values) {
    for (uint width = 1u; width < 128u; width <<= 1u) {
        for (uint start = 0u; start < 128u; start += width << 1u) {
            for (uint offset = 0u; offset < width; ++offset) {
                float left = values[start + offset];
                float right = values[start + offset + width];
                values[start + offset] = left + right;
                values[start + offset + width] = left - right;
            }
        }
    }
    for (uint i = 0; i < 128u; ++i) {
        values[i] *= 0.08838834764831845f;
    }
}

kernel void linear_input_transform(
    const device half *input [[buffer(0)]],
    const device half *input_signs [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &blocks [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= blocks) {
        return;
    }
    float values[128];
    uint base = gid * 128u;
    for (uint i = 0; i < 128u; ++i) {
        values[i] = float(input[base + i]) * float(input_signs[base + i]);
    }
    transform_vector(values);
    for (uint i = 0; i < 128u; ++i) {
        output[base + i] = half(values[i]);
    }
}

// EXL3 decode matvec, shaped like the reference implementation.
//
// Thread mapping follows the trellis rather than the output:
//   * one thread per trellis lane (32 lanes == one simdgroup), so the eight codes
//     a lane owns are decoded from a single window setup;
//   * one 256-thread group (8 simdgroups) per output column-tile;
//   * simdgroups stride the tile-row axis, splitting the K reduction.
//
// The previous kernel used one thread per output element -- 5,376 threads for a
// 5376-wide projection on a 20-core GPU -- and re-derived the 64-bit window once
// per weight instead of once per eight weights.
//
// A lane's eight codes land on rows {r0, r0+1, r0+8, r0+9} and columns {c, c+8},
// so four x values feed all eight, each used twice. The reduction is
// simd_shuffle_xor over four lanes then a threadgroup sum over the eight
// simdgroups. No atomics.
//
// Two axes are specialised, because both are worth real throughput and Metal has
// no templates for either:
//   * KA (1..8), so shift amounts and the ring size fold to constants.
//   * ALIGNED, so a 4-byte-aligned buffer is read with one 32-bit load per word
//     instead of two 16-bit loads. Arena-resident weights are always aligned;
//     only mapped checkpoint bytes are not. The two-16-bit-load form doubles the
//     load instruction count on the hottest path in the engine, and loads -- not
//     the codebook -- dominate: measured 0.336 ms of 0.612 ms for q_proj.
//     Variants are generated by `scripts/gen_exl3_gemv.py`; do not hand-edit.

// ---------------------------------------------------------------------------
// Verbatim port of ishizuki's `exl3_matvec`, for A/B only.
//
// Sources/IshizukiKit/Core/EXL3.swift:286. Their EXL3 kernel reports 28 tok/s on
// a 2-bit Qwen 27B on M1 Max, which has comparable *real* bandwidth to an M4
// Pro, so the format is not the ceiling and our kernel must be. This is a
// line-for-line transcription of theirs -- same `column` pointer hoisted out of
// the loop, same `xr` hoisted, same 8-simdgroup striding, same shuffle tree -- so
// that any difference in throughput is attributable to the code and not to a
// paraphrase. Do not "improve" it; that is the whole point.
//
// Differences from ours that are deliberate, and listed so they can be checked:
//   * `trellis` is `const device uint *` and indexed directly, so each ring word
//     is one 32-bit load. Every layer weight in our model is 4-byte aligned
//     (verified: all packed_offset == 0), so this is the same path our ALIGNED
//     specialisation takes.
//   * `column` is hoisted, so the address is affine in `tk` and the compiler can
//     strength-reduce it across the loop.
kernel void linear_gemv_ishizuki_k1(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    const device uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]]) {
    constexpr uint KA = 1u;
    constexpr int CB = 1;
    constexpr uint RING = 256u * KA;
    constexpr uint WORDS = RING / 32u;
    const uint K = shape[0];
    const uint N = shape[1];
    const uint tiles_n = N / 16u;
    const uint tiles_k = K / 16u;
    constexpr int SG = 8;
    threadgroup float red[SG][16];
    const uint r0 = (lane % 4u) * 2u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // j0/j1/j2 and sh depend only on the lane, not on the tile index, so they
    // hoist out of the tile loop. Unrolling by two then keeps six weight words
    // in flight per lane, which is worth ~27% on the full layer.
    for (uint tn = tgid; tn < tiles_n; tn += ntg) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint lo0 = (lane * 8u + 1u) * 1u + RING - 16u;
    const uint sh0 = lo0 % 32u;
    const uint hi0 = lo0 / 32u;
    const uint j0 = hi0 >= WORDS ? hi0 - WORDS : hi0;
    const uint j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u;
    const uint j2 = j1 + 1u >= WORDS ? j1 + 1u - WORDS : j1 + 1u;
    const device uint *column = (const device uint *)packed + tn * WORDS;
    const uint stride = tiles_n * WORDS;
    uint tk = sg;
    for (; tk + SG < tiles_k; tk += 2u * SG) {
        float v[8];
        const device uint *t0 = column + (tk + 0 * SG) * stride;
        const ulong w0a = t0[j0], w0b = t0[j1], w0c = t0[j2];
        const ulong A0 = ((w0a << 32) | w0b) << sh0 | (sh0 ? (w0c >> (32 - sh0)) : 0ul);
        const ulong B0 = ((w0b << 32) | w0c) << sh0;
        const device half *xr0 = x + (tk + 0 * SG) * 16u + r0;
        const float p00 = float(xr0[0]), p01 = float(xr0[1]);
        const float p02 = float(xr0[8]), p03 = float(xr0[9]);
        const device uint *t1 = column + (tk + 1 * SG) * stride;
        const ulong w1a = t1[j0], w1b = t1[j1], w1c = t1[j2];
        const ulong A1 = ((w1a << 32) | w1b) << sh0 | (sh0 ? (w1c >> (32 - sh0)) : 0ul);
        const ulong B1 = ((w1b << 32) | w1c) << sh0;
        const device half *xr1 = x + (tk + 1 * SG) * 16u + r0;
        const float p10 = float(xr1[0]), p11 = float(xr1[1]);
        const float p12 = float(xr1[8]), p13 = float(xr1[9]);
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            const uint r = i * KA + 16u;
            const uint s = (r <= 64u) ? uint(A0 >> (64u - r)) : uint(B0 >> (96u - r));
            v[i] = decode_state(s & 0xffffu, CB);
        }
        acc0 += p00 * v[0] + p01 * v[1] + p02 * v[2] + p03 * v[3];
        acc1 += p00 * v[4] + p01 * v[5] + p02 * v[6] + p03 * v[7];
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            const uint r = i * KA + 16u;
            const uint s = (r <= 64u) ? uint(A1 >> (64u - r)) : uint(B1 >> (96u - r));
            v[i] = decode_state(s & 0xffffu, CB);
        }
        acc0 += p10 * v[0] + p11 * v[1] + p12 * v[2] + p13 * v[3];
        acc1 += p10 * v[4] + p11 * v[5] + p12 * v[6] + p13 * v[7];
    }

    for (; tk < tiles_k; tk += SG) {
        const device uint *t0 = column + tk * stride;
        const ulong a0 = t0[j0], a1 = t0[j1], a2 = t0[j2];
        const ulong A0 = ((a0 << 32) | a1) << sh0 | (sh0 ? (a2 >> (32 - sh0)) : 0ul);
        const ulong B0 = ((a1 << 32) | a2) << sh0;
        const device half *xr0 = x + tk * 16u + r0;
        const float p0 = float(xr0[0]), p1 = float(xr0[1]), p2 = float(xr0[8]), p3 = float(xr0[9]);
        float v[8];
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            const uint r = i * KA + 16u;
            const uint s = (r <= 64u) ? uint(A0 >> (64u - r)) : uint(B0 >> (96u - r));
            v[i] = decode_state(s & 0xffffu, CB);
        }
        acc0 += p0 * v[0] + p1 * v[1] + p2 * v[2] + p3 * v[3];
        acc1 += p0 * v[4] + p1 * v[5] + p2 * v[6] + p3 * v[7];
    }
    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if (lane % 4u == 0u) {
        red[sg][lane / 4u] = acc0;
        red[sg][lane / 4u + 8u] = acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u) {
        float sum = 0.0f;
        for (int g = 0; g < SG; ++g) {
            sum += red[g][lid];
        }
        y[tn * 16u + lid] = sum;
    }
        // `red` is reused by the next output tile, so it must be drained first.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void linear_gemv_ishizuki_k2(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    const device uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]]) {
    constexpr uint KA = 2u;
    constexpr int CB = 1;
    constexpr uint RING = 256u * KA;
    constexpr uint WORDS = RING / 32u;
    const uint K = shape[0];
    const uint N = shape[1];
    const uint tiles_n = N / 16u;
    const uint tiles_k = K / 16u;
    constexpr int SG = 8;
    threadgroup float red[SG][16];
    const uint r0 = (lane % 4u) * 2u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // j0/j1/j2 and sh depend only on the lane, not on the tile index, so they
    // hoist out of the tile loop. Unrolling by two then keeps six weight words
    // in flight per lane, which is worth ~27% on the full layer.
    for (uint tn = tgid; tn < tiles_n; tn += ntg) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint lo0 = (lane * 8u + 1u) * 2u + RING - 16u;
    const uint sh0 = lo0 % 32u;
    const uint hi0 = lo0 / 32u;
    const uint j0 = hi0 >= WORDS ? hi0 - WORDS : hi0;
    const uint j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u;
    const uint j2 = j1 + 1u >= WORDS ? j1 + 1u - WORDS : j1 + 1u;
    const device uint *column = (const device uint *)packed + tn * WORDS;
    const uint stride = tiles_n * WORDS;
    uint tk = sg;
    for (; tk + SG < tiles_k; tk += 2u * SG) {
        float v[8];
        const device uint *t0 = column + (tk + 0 * SG) * stride;
        const ulong w0a = t0[j0], w0b = t0[j1], w0c = t0[j2];
        const ulong A0 = ((w0a << 32) | w0b) << sh0 | (sh0 ? (w0c >> (32 - sh0)) : 0ul);
        const ulong B0 = ((w0b << 32) | w0c) << sh0;
        const device half *xr0 = x + (tk + 0 * SG) * 16u + r0;
        const float p00 = float(xr0[0]), p01 = float(xr0[1]);
        const float p02 = float(xr0[8]), p03 = float(xr0[9]);
        const device uint *t1 = column + (tk + 1 * SG) * stride;
        const ulong w1a = t1[j0], w1b = t1[j1], w1c = t1[j2];
        const ulong A1 = ((w1a << 32) | w1b) << sh0 | (sh0 ? (w1c >> (32 - sh0)) : 0ul);
        const ulong B1 = ((w1b << 32) | w1c) << sh0;
        const device half *xr1 = x + (tk + 1 * SG) * 16u + r0;
        const float p10 = float(xr1[0]), p11 = float(xr1[1]);
        const float p12 = float(xr1[8]), p13 = float(xr1[9]);
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            const uint r = i * KA + 16u;
            const uint s = (r <= 64u) ? uint(A0 >> (64u - r)) : uint(B0 >> (96u - r));
            v[i] = decode_state(s & 0xffffu, CB);
        }
        acc0 += p00 * v[0] + p01 * v[1] + p02 * v[2] + p03 * v[3];
        acc1 += p00 * v[4] + p01 * v[5] + p02 * v[6] + p03 * v[7];
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            const uint r = i * KA + 16u;
            const uint s = (r <= 64u) ? uint(A1 >> (64u - r)) : uint(B1 >> (96u - r));
            v[i] = decode_state(s & 0xffffu, CB);
        }
        acc0 += p10 * v[0] + p11 * v[1] + p12 * v[2] + p13 * v[3];
        acc1 += p10 * v[4] + p11 * v[5] + p12 * v[6] + p13 * v[7];
    }

    for (; tk < tiles_k; tk += SG) {
        const device uint *t0 = column + tk * stride;
        const ulong a0 = t0[j0], a1 = t0[j1], a2 = t0[j2];
        const ulong A0 = ((a0 << 32) | a1) << sh0 | (sh0 ? (a2 >> (32 - sh0)) : 0ul);
        const ulong B0 = ((a1 << 32) | a2) << sh0;
        const device half *xr0 = x + tk * 16u + r0;
        const float p0 = float(xr0[0]), p1 = float(xr0[1]), p2 = float(xr0[8]), p3 = float(xr0[9]);
        float v[8];
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            const uint r = i * KA + 16u;
            const uint s = (r <= 64u) ? uint(A0 >> (64u - r)) : uint(B0 >> (96u - r));
            v[i] = decode_state(s & 0xffffu, CB);
        }
        acc0 += p0 * v[0] + p1 * v[1] + p2 * v[2] + p3 * v[3];
        acc1 += p0 * v[4] + p1 * v[5] + p2 * v[6] + p3 * v[7];
    }
    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if (lane % 4u == 0u) {
        red[sg][lane / 4u] = acc0;
        red[sg][lane / 4u + 8u] = acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u) {
        float sum = 0.0f;
        for (int g = 0; g < SG; ++g) {
            sum += red[g][lid];
        }
        y[tn * 16u + lid] = sum;
    }
        // `red` is reused by the next output tile, so it must be drained first.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void linear_gemv_ishizuki_k3(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    const device uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]]) {
    constexpr uint KA = 3u;
    constexpr int CB = 1;
    constexpr uint RING = 256u * KA;
    constexpr uint WORDS = RING / 32u;
    const uint K = shape[0];
    const uint N = shape[1];
    const uint tiles_n = N / 16u;
    const uint tiles_k = K / 16u;
    constexpr int SG = 8;
    threadgroup float red[SG][16];
    const uint r0 = (lane % 4u) * 2u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // j0/j1/j2 and sh depend only on the lane, not on the tile index, so they
    // hoist out of the tile loop. Unrolling by two then keeps six weight words
    // in flight per lane, which is worth ~27% on the full layer.
    for (uint tn = tgid; tn < tiles_n; tn += ntg) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint lo0 = (lane * 8u + 1u) * 3u + RING - 16u;
    const uint sh0 = lo0 % 32u;
    const uint hi0 = lo0 / 32u;
    const uint j0 = hi0 >= WORDS ? hi0 - WORDS : hi0;
    const uint j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u;
    const uint j2 = j1 + 1u >= WORDS ? j1 + 1u - WORDS : j1 + 1u;
    const device uint *column = (const device uint *)packed + tn * WORDS;
    const uint stride = tiles_n * WORDS;
    uint tk = sg;
    for (; tk + SG < tiles_k; tk += 2u * SG) {
        float v[8];
        const device uint *t0 = column + (tk + 0 * SG) * stride;
        const ulong w0a = t0[j0], w0b = t0[j1], w0c = t0[j2];
        const ulong A0 = ((w0a << 32) | w0b) << sh0 | (sh0 ? (w0c >> (32 - sh0)) : 0ul);
        const ulong B0 = ((w0b << 32) | w0c) << sh0;
        const device half *xr0 = x + (tk + 0 * SG) * 16u + r0;
        const float p00 = float(xr0[0]), p01 = float(xr0[1]);
        const float p02 = float(xr0[8]), p03 = float(xr0[9]);
        const device uint *t1 = column + (tk + 1 * SG) * stride;
        const ulong w1a = t1[j0], w1b = t1[j1], w1c = t1[j2];
        const ulong A1 = ((w1a << 32) | w1b) << sh0 | (sh0 ? (w1c >> (32 - sh0)) : 0ul);
        const ulong B1 = ((w1b << 32) | w1c) << sh0;
        const device half *xr1 = x + (tk + 1 * SG) * 16u + r0;
        const float p10 = float(xr1[0]), p11 = float(xr1[1]);
        const float p12 = float(xr1[8]), p13 = float(xr1[9]);
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            const uint r = i * KA + 16u;
            const uint s = (r <= 64u) ? uint(A0 >> (64u - r)) : uint(B0 >> (96u - r));
            v[i] = decode_state(s & 0xffffu, CB);
        }
        acc0 += p00 * v[0] + p01 * v[1] + p02 * v[2] + p03 * v[3];
        acc1 += p00 * v[4] + p01 * v[5] + p02 * v[6] + p03 * v[7];
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            const uint r = i * KA + 16u;
            const uint s = (r <= 64u) ? uint(A1 >> (64u - r)) : uint(B1 >> (96u - r));
            v[i] = decode_state(s & 0xffffu, CB);
        }
        acc0 += p10 * v[0] + p11 * v[1] + p12 * v[2] + p13 * v[3];
        acc1 += p10 * v[4] + p11 * v[5] + p12 * v[6] + p13 * v[7];
    }

    for (; tk < tiles_k; tk += SG) {
        const device uint *t0 = column + tk * stride;
        const ulong a0 = t0[j0], a1 = t0[j1], a2 = t0[j2];
        const ulong A0 = ((a0 << 32) | a1) << sh0 | (sh0 ? (a2 >> (32 - sh0)) : 0ul);
        const ulong B0 = ((a1 << 32) | a2) << sh0;
        const device half *xr0 = x + tk * 16u + r0;
        const float p0 = float(xr0[0]), p1 = float(xr0[1]), p2 = float(xr0[8]), p3 = float(xr0[9]);
        float v[8];
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            const uint r = i * KA + 16u;
            const uint s = (r <= 64u) ? uint(A0 >> (64u - r)) : uint(B0 >> (96u - r));
            v[i] = decode_state(s & 0xffffu, CB);
        }
        acc0 += p0 * v[0] + p1 * v[1] + p2 * v[2] + p3 * v[3];
        acc1 += p0 * v[4] + p1 * v[5] + p2 * v[6] + p3 * v[7];
    }
    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if (lane % 4u == 0u) {
        red[sg][lane / 4u] = acc0;
        red[sg][lane / 4u + 8u] = acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u) {
        float sum = 0.0f;
        for (int g = 0; g < SG; ++g) {
            sum += red[g][lid];
        }
        y[tn * 16u + lid] = sum;
    }
        // `red` is reused by the next output tile, so it must be drained first.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void linear_gemv_ishizuki_k4(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    const device uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]]) {
    constexpr uint KA = 4u;
    constexpr int CB = 1;
    constexpr uint RING = 256u * KA;
    constexpr uint WORDS = RING / 32u;
    const uint K = shape[0];
    const uint N = shape[1];
    const uint tiles_n = N / 16u;
    const uint tiles_k = K / 16u;
    constexpr int SG = 8;
    threadgroup float red[SG][16];
    const uint r0 = (lane % 4u) * 2u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // j0/j1/j2 and sh depend only on the lane, not on the tile index, so they
    // hoist out of the tile loop. Unrolling by two then keeps six weight words
    // in flight per lane, which is worth ~27% on the full layer.
    for (uint tn = tgid; tn < tiles_n; tn += ntg) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint lo0 = (lane * 8u + 1u) * 4u + RING - 16u;
    const uint sh0 = lo0 % 32u;
    const uint hi0 = lo0 / 32u;
    const uint j0 = hi0 >= WORDS ? hi0 - WORDS : hi0;
    const uint j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u;
    const uint j2 = j1 + 1u >= WORDS ? j1 + 1u - WORDS : j1 + 1u;
    const device uint *column = (const device uint *)packed + tn * WORDS;
    const uint stride = tiles_n * WORDS;
    uint tk = sg;
    for (; tk + SG < tiles_k; tk += 2u * SG) {
        float v[8];
        const device uint *t0 = column + (tk + 0 * SG) * stride;
        const ulong w0a = t0[j0], w0b = t0[j1], w0c = t0[j2];
        const ulong A0 = ((w0a << 32) | w0b) << sh0 | (sh0 ? (w0c >> (32 - sh0)) : 0ul);
        const ulong B0 = ((w0b << 32) | w0c) << sh0;
        const device half *xr0 = x + (tk + 0 * SG) * 16u + r0;
        const float p00 = float(xr0[0]), p01 = float(xr0[1]);
        const float p02 = float(xr0[8]), p03 = float(xr0[9]);
        const device uint *t1 = column + (tk + 1 * SG) * stride;
        const ulong w1a = t1[j0], w1b = t1[j1], w1c = t1[j2];
        const ulong A1 = ((w1a << 32) | w1b) << sh0 | (sh0 ? (w1c >> (32 - sh0)) : 0ul);
        const ulong B1 = ((w1b << 32) | w1c) << sh0;
        const device half *xr1 = x + (tk + 1 * SG) * 16u + r0;
        const float p10 = float(xr1[0]), p11 = float(xr1[1]);
        const float p12 = float(xr1[8]), p13 = float(xr1[9]);
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            const uint r = i * KA + 16u;
            const uint s = (r <= 64u) ? uint(A0 >> (64u - r)) : uint(B0 >> (96u - r));
            v[i] = decode_state(s & 0xffffu, CB);
        }
        acc0 += p00 * v[0] + p01 * v[1] + p02 * v[2] + p03 * v[3];
        acc1 += p00 * v[4] + p01 * v[5] + p02 * v[6] + p03 * v[7];
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            const uint r = i * KA + 16u;
            const uint s = (r <= 64u) ? uint(A1 >> (64u - r)) : uint(B1 >> (96u - r));
            v[i] = decode_state(s & 0xffffu, CB);
        }
        acc0 += p10 * v[0] + p11 * v[1] + p12 * v[2] + p13 * v[3];
        acc1 += p10 * v[4] + p11 * v[5] + p12 * v[6] + p13 * v[7];
    }

    for (; tk < tiles_k; tk += SG) {
        const device uint *t0 = column + tk * stride;
        const ulong a0 = t0[j0], a1 = t0[j1], a2 = t0[j2];
        const ulong A0 = ((a0 << 32) | a1) << sh0 | (sh0 ? (a2 >> (32 - sh0)) : 0ul);
        const ulong B0 = ((a1 << 32) | a2) << sh0;
        const device half *xr0 = x + tk * 16u + r0;
        const float p0 = float(xr0[0]), p1 = float(xr0[1]), p2 = float(xr0[8]), p3 = float(xr0[9]);
        float v[8];
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            const uint r = i * KA + 16u;
            const uint s = (r <= 64u) ? uint(A0 >> (64u - r)) : uint(B0 >> (96u - r));
            v[i] = decode_state(s & 0xffffu, CB);
        }
        acc0 += p0 * v[0] + p1 * v[1] + p2 * v[2] + p3 * v[3];
        acc1 += p0 * v[4] + p1 * v[5] + p2 * v[6] + p3 * v[7];
    }
    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if (lane % 4u == 0u) {
        red[sg][lane / 4u] = acc0;
        red[sg][lane / 4u + 8u] = acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u) {
        float sum = 0.0f;
        for (int g = 0; g < SG; ++g) {
            sum += red[g][lid];
        }
        y[tn * 16u + lid] = sum;
    }
        // `red` is reused by the next output tile, so it must be drained first.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void linear_gemv_ishizuki_k5(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    const device uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]]) {
    constexpr uint KA = 5u;
    constexpr int CB = 1;
    constexpr uint RING = 256u * KA;
    constexpr uint WORDS = RING / 32u;
    const uint K = shape[0];
    const uint N = shape[1];
    const uint tiles_n = N / 16u;
    const uint tiles_k = K / 16u;
    constexpr int SG = 8;
    threadgroup float red[SG][16];
    const uint r0 = (lane % 4u) * 2u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // j0/j1/j2 and sh depend only on the lane, not on the tile index, so they
    // hoist out of the tile loop. Unrolling by two then keeps six weight words
    // in flight per lane, which is worth ~27% on the full layer.
    for (uint tn = tgid; tn < tiles_n; tn += ntg) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint lo0 = (lane * 8u + 1u) * 5u + RING - 16u;
    const uint sh0 = lo0 % 32u;
    const uint hi0 = lo0 / 32u;
    const uint j0 = hi0 >= WORDS ? hi0 - WORDS : hi0;
    const uint j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u;
    const uint j2 = j1 + 1u >= WORDS ? j1 + 1u - WORDS : j1 + 1u;
    const device uint *column = (const device uint *)packed + tn * WORDS;
    const uint stride = tiles_n * WORDS;
    uint tk = sg;
    for (; tk + SG < tiles_k; tk += 2u * SG) {
        float v[8];
        const device uint *t0 = column + (tk + 0 * SG) * stride;
        const ulong w0a = t0[j0], w0b = t0[j1], w0c = t0[j2];
        const ulong A0 = ((w0a << 32) | w0b) << sh0 | (sh0 ? (w0c >> (32 - sh0)) : 0ul);
        const ulong B0 = ((w0b << 32) | w0c) << sh0;
        const device half *xr0 = x + (tk + 0 * SG) * 16u + r0;
        const float p00 = float(xr0[0]), p01 = float(xr0[1]);
        const float p02 = float(xr0[8]), p03 = float(xr0[9]);
        const device uint *t1 = column + (tk + 1 * SG) * stride;
        const ulong w1a = t1[j0], w1b = t1[j1], w1c = t1[j2];
        const ulong A1 = ((w1a << 32) | w1b) << sh0 | (sh0 ? (w1c >> (32 - sh0)) : 0ul);
        const ulong B1 = ((w1b << 32) | w1c) << sh0;
        const device half *xr1 = x + (tk + 1 * SG) * 16u + r0;
        const float p10 = float(xr1[0]), p11 = float(xr1[1]);
        const float p12 = float(xr1[8]), p13 = float(xr1[9]);
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            const uint r = i * KA + 16u;
            const uint s = (r <= 64u) ? uint(A0 >> (64u - r)) : uint(B0 >> (96u - r));
            v[i] = decode_state(s & 0xffffu, CB);
        }
        acc0 += p00 * v[0] + p01 * v[1] + p02 * v[2] + p03 * v[3];
        acc1 += p00 * v[4] + p01 * v[5] + p02 * v[6] + p03 * v[7];
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            const uint r = i * KA + 16u;
            const uint s = (r <= 64u) ? uint(A1 >> (64u - r)) : uint(B1 >> (96u - r));
            v[i] = decode_state(s & 0xffffu, CB);
        }
        acc0 += p10 * v[0] + p11 * v[1] + p12 * v[2] + p13 * v[3];
        acc1 += p10 * v[4] + p11 * v[5] + p12 * v[6] + p13 * v[7];
    }

    for (; tk < tiles_k; tk += SG) {
        const device uint *t0 = column + tk * stride;
        const ulong a0 = t0[j0], a1 = t0[j1], a2 = t0[j2];
        const ulong A0 = ((a0 << 32) | a1) << sh0 | (sh0 ? (a2 >> (32 - sh0)) : 0ul);
        const ulong B0 = ((a1 << 32) | a2) << sh0;
        const device half *xr0 = x + tk * 16u + r0;
        const float p0 = float(xr0[0]), p1 = float(xr0[1]), p2 = float(xr0[8]), p3 = float(xr0[9]);
        float v[8];
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            const uint r = i * KA + 16u;
            const uint s = (r <= 64u) ? uint(A0 >> (64u - r)) : uint(B0 >> (96u - r));
            v[i] = decode_state(s & 0xffffu, CB);
        }
        acc0 += p0 * v[0] + p1 * v[1] + p2 * v[2] + p3 * v[3];
        acc1 += p0 * v[4] + p1 * v[5] + p2 * v[6] + p3 * v[7];
    }
    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if (lane % 4u == 0u) {
        red[sg][lane / 4u] = acc0;
        red[sg][lane / 4u + 8u] = acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u) {
        float sum = 0.0f;
        for (int g = 0; g < SG; ++g) {
            sum += red[g][lid];
        }
        y[tn * 16u + lid] = sum;
    }
        // `red` is reused by the next output tile, so it must be drained first.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void linear_gemv_ishizuki_k6(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    const device uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]]) {
    constexpr uint KA = 6u;
    constexpr int CB = 1;
    constexpr uint RING = 256u * KA;
    constexpr uint WORDS = RING / 32u;
    const uint K = shape[0];
    const uint N = shape[1];
    const uint tiles_n = N / 16u;
    const uint tiles_k = K / 16u;
    constexpr int SG = 8;
    threadgroup float red[SG][16];
    const uint r0 = (lane % 4u) * 2u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // j0/j1/j2 and sh depend only on the lane, not on the tile index, so they
    // hoist out of the tile loop. Unrolling by two then keeps six weight words
    // in flight per lane, which is worth ~27% on the full layer.
    for (uint tn = tgid; tn < tiles_n; tn += ntg) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint lo0 = (lane * 8u + 1u) * 6u + RING - 16u;
    const uint sh0 = lo0 % 32u;
    const uint hi0 = lo0 / 32u;
    const uint j0 = hi0 >= WORDS ? hi0 - WORDS : hi0;
    const uint j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u;
    const uint j2 = j1 + 1u >= WORDS ? j1 + 1u - WORDS : j1 + 1u;
    const device uint *column = (const device uint *)packed + tn * WORDS;
    const uint stride = tiles_n * WORDS;
    uint tk = sg;
    for (; tk + SG < tiles_k; tk += 2u * SG) {
        float v[8];
        const device uint *t0 = column + (tk + 0 * SG) * stride;
        const ulong w0a = t0[j0], w0b = t0[j1], w0c = t0[j2];
        const ulong A0 = ((w0a << 32) | w0b) << sh0 | (sh0 ? (w0c >> (32 - sh0)) : 0ul);
        const ulong B0 = ((w0b << 32) | w0c) << sh0;
        const device half *xr0 = x + (tk + 0 * SG) * 16u + r0;
        const float p00 = float(xr0[0]), p01 = float(xr0[1]);
        const float p02 = float(xr0[8]), p03 = float(xr0[9]);
        const device uint *t1 = column + (tk + 1 * SG) * stride;
        const ulong w1a = t1[j0], w1b = t1[j1], w1c = t1[j2];
        const ulong A1 = ((w1a << 32) | w1b) << sh0 | (sh0 ? (w1c >> (32 - sh0)) : 0ul);
        const ulong B1 = ((w1b << 32) | w1c) << sh0;
        const device half *xr1 = x + (tk + 1 * SG) * 16u + r0;
        const float p10 = float(xr1[0]), p11 = float(xr1[1]);
        const float p12 = float(xr1[8]), p13 = float(xr1[9]);
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            const uint r = i * KA + 16u;
            const uint s = (r <= 64u) ? uint(A0 >> (64u - r)) : uint(B0 >> (96u - r));
            v[i] = decode_state(s & 0xffffu, CB);
        }
        acc0 += p00 * v[0] + p01 * v[1] + p02 * v[2] + p03 * v[3];
        acc1 += p00 * v[4] + p01 * v[5] + p02 * v[6] + p03 * v[7];
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            const uint r = i * KA + 16u;
            const uint s = (r <= 64u) ? uint(A1 >> (64u - r)) : uint(B1 >> (96u - r));
            v[i] = decode_state(s & 0xffffu, CB);
        }
        acc0 += p10 * v[0] + p11 * v[1] + p12 * v[2] + p13 * v[3];
        acc1 += p10 * v[4] + p11 * v[5] + p12 * v[6] + p13 * v[7];
    }

    for (; tk < tiles_k; tk += SG) {
        const device uint *t0 = column + tk * stride;
        const ulong a0 = t0[j0], a1 = t0[j1], a2 = t0[j2];
        const ulong A0 = ((a0 << 32) | a1) << sh0 | (sh0 ? (a2 >> (32 - sh0)) : 0ul);
        const ulong B0 = ((a1 << 32) | a2) << sh0;
        const device half *xr0 = x + tk * 16u + r0;
        const float p0 = float(xr0[0]), p1 = float(xr0[1]), p2 = float(xr0[8]), p3 = float(xr0[9]);
        float v[8];
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            const uint r = i * KA + 16u;
            const uint s = (r <= 64u) ? uint(A0 >> (64u - r)) : uint(B0 >> (96u - r));
            v[i] = decode_state(s & 0xffffu, CB);
        }
        acc0 += p0 * v[0] + p1 * v[1] + p2 * v[2] + p3 * v[3];
        acc1 += p0 * v[4] + p1 * v[5] + p2 * v[6] + p3 * v[7];
    }
    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if (lane % 4u == 0u) {
        red[sg][lane / 4u] = acc0;
        red[sg][lane / 4u + 8u] = acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u) {
        float sum = 0.0f;
        for (int g = 0; g < SG; ++g) {
            sum += red[g][lid];
        }
        y[tn * 16u + lid] = sum;
    }
        // `red` is reused by the next output tile, so it must be drained first.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void linear_gemv_ishizuki_k7(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    const device uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]]) {
    constexpr uint KA = 7u;
    constexpr int CB = 1;
    constexpr uint RING = 256u * KA;
    constexpr uint WORDS = RING / 32u;
    const uint K = shape[0];
    const uint N = shape[1];
    const uint tiles_n = N / 16u;
    const uint tiles_k = K / 16u;
    constexpr int SG = 8;
    threadgroup float red[SG][16];
    const uint r0 = (lane % 4u) * 2u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // j0/j1/j2 and sh depend only on the lane, not on the tile index, so they
    // hoist out of the tile loop. Unrolling by two then keeps six weight words
    // in flight per lane, which is worth ~27% on the full layer.
    for (uint tn = tgid; tn < tiles_n; tn += ntg) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint lo0 = (lane * 8u + 1u) * 7u + RING - 16u;
    const uint sh0 = lo0 % 32u;
    const uint hi0 = lo0 / 32u;
    const uint j0 = hi0 >= WORDS ? hi0 - WORDS : hi0;
    const uint j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u;
    const uint j2 = j1 + 1u >= WORDS ? j1 + 1u - WORDS : j1 + 1u;
    const device uint *column = (const device uint *)packed + tn * WORDS;
    const uint stride = tiles_n * WORDS;
    uint tk = sg;
    for (; tk + SG < tiles_k; tk += 2u * SG) {
        float v[8];
        const device uint *t0 = column + (tk + 0 * SG) * stride;
        const ulong w0a = t0[j0], w0b = t0[j1], w0c = t0[j2];
        const ulong A0 = ((w0a << 32) | w0b) << sh0 | (sh0 ? (w0c >> (32 - sh0)) : 0ul);
        const ulong B0 = ((w0b << 32) | w0c) << sh0;
        const device half *xr0 = x + (tk + 0 * SG) * 16u + r0;
        const float p00 = float(xr0[0]), p01 = float(xr0[1]);
        const float p02 = float(xr0[8]), p03 = float(xr0[9]);
        const device uint *t1 = column + (tk + 1 * SG) * stride;
        const ulong w1a = t1[j0], w1b = t1[j1], w1c = t1[j2];
        const ulong A1 = ((w1a << 32) | w1b) << sh0 | (sh0 ? (w1c >> (32 - sh0)) : 0ul);
        const ulong B1 = ((w1b << 32) | w1c) << sh0;
        const device half *xr1 = x + (tk + 1 * SG) * 16u + r0;
        const float p10 = float(xr1[0]), p11 = float(xr1[1]);
        const float p12 = float(xr1[8]), p13 = float(xr1[9]);
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            const uint r = i * KA + 16u;
            const uint s = (r <= 64u) ? uint(A0 >> (64u - r)) : uint(B0 >> (96u - r));
            v[i] = decode_state(s & 0xffffu, CB);
        }
        acc0 += p00 * v[0] + p01 * v[1] + p02 * v[2] + p03 * v[3];
        acc1 += p00 * v[4] + p01 * v[5] + p02 * v[6] + p03 * v[7];
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            const uint r = i * KA + 16u;
            const uint s = (r <= 64u) ? uint(A1 >> (64u - r)) : uint(B1 >> (96u - r));
            v[i] = decode_state(s & 0xffffu, CB);
        }
        acc0 += p10 * v[0] + p11 * v[1] + p12 * v[2] + p13 * v[3];
        acc1 += p10 * v[4] + p11 * v[5] + p12 * v[6] + p13 * v[7];
    }

    for (; tk < tiles_k; tk += SG) {
        const device uint *t0 = column + tk * stride;
        const ulong a0 = t0[j0], a1 = t0[j1], a2 = t0[j2];
        const ulong A0 = ((a0 << 32) | a1) << sh0 | (sh0 ? (a2 >> (32 - sh0)) : 0ul);
        const ulong B0 = ((a1 << 32) | a2) << sh0;
        const device half *xr0 = x + tk * 16u + r0;
        const float p0 = float(xr0[0]), p1 = float(xr0[1]), p2 = float(xr0[8]), p3 = float(xr0[9]);
        float v[8];
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            const uint r = i * KA + 16u;
            const uint s = (r <= 64u) ? uint(A0 >> (64u - r)) : uint(B0 >> (96u - r));
            v[i] = decode_state(s & 0xffffu, CB);
        }
        acc0 += p0 * v[0] + p1 * v[1] + p2 * v[2] + p3 * v[3];
        acc1 += p0 * v[4] + p1 * v[5] + p2 * v[6] + p3 * v[7];
    }
    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if (lane % 4u == 0u) {
        red[sg][lane / 4u] = acc0;
        red[sg][lane / 4u + 8u] = acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u) {
        float sum = 0.0f;
        for (int g = 0; g < SG; ++g) {
            sum += red[g][lid];
        }
        y[tn * 16u + lid] = sum;
    }
        // `red` is reused by the next output tile, so it must be drained first.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void linear_gemv_ishizuki_k8(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    const device uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]]) {
    constexpr uint KA = 8u;
    constexpr int CB = 1;
    constexpr uint RING = 256u * KA;
    constexpr uint WORDS = RING / 32u;
    const uint K = shape[0];
    const uint N = shape[1];
    const uint tiles_n = N / 16u;
    const uint tiles_k = K / 16u;
    constexpr int SG = 8;
    threadgroup float red[SG][16];
    const uint r0 = (lane % 4u) * 2u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // j0/j1/j2 and sh depend only on the lane, not on the tile index, so they
    // hoist out of the tile loop. Unrolling by two then keeps six weight words
    // in flight per lane, which is worth ~27% on the full layer.
    for (uint tn = tgid; tn < tiles_n; tn += ntg) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint lo0 = (lane * 8u + 1u) * 8u + RING - 16u;
    const uint sh0 = lo0 % 32u;
    const uint hi0 = lo0 / 32u;
    const uint j0 = hi0 >= WORDS ? hi0 - WORDS : hi0;
    const uint j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u;
    const uint j2 = j1 + 1u >= WORDS ? j1 + 1u - WORDS : j1 + 1u;
    const device uint *column = (const device uint *)packed + tn * WORDS;
    const uint stride = tiles_n * WORDS;
    uint tk = sg;
    for (; tk + SG < tiles_k; tk += 2u * SG) {
        float v[8];
        const device uint *t0 = column + (tk + 0 * SG) * stride;
        const ulong w0a = t0[j0], w0b = t0[j1], w0c = t0[j2];
        const ulong A0 = ((w0a << 32) | w0b) << sh0 | (sh0 ? (w0c >> (32 - sh0)) : 0ul);
        const ulong B0 = ((w0b << 32) | w0c) << sh0;
        const device half *xr0 = x + (tk + 0 * SG) * 16u + r0;
        const float p00 = float(xr0[0]), p01 = float(xr0[1]);
        const float p02 = float(xr0[8]), p03 = float(xr0[9]);
        const device uint *t1 = column + (tk + 1 * SG) * stride;
        const ulong w1a = t1[j0], w1b = t1[j1], w1c = t1[j2];
        const ulong A1 = ((w1a << 32) | w1b) << sh0 | (sh0 ? (w1c >> (32 - sh0)) : 0ul);
        const ulong B1 = ((w1b << 32) | w1c) << sh0;
        const device half *xr1 = x + (tk + 1 * SG) * 16u + r0;
        const float p10 = float(xr1[0]), p11 = float(xr1[1]);
        const float p12 = float(xr1[8]), p13 = float(xr1[9]);
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            const uint r = i * KA + 16u;
            const uint s = (r <= 64u) ? uint(A0 >> (64u - r)) : uint(B0 >> (96u - r));
            v[i] = decode_state(s & 0xffffu, CB);
        }
        acc0 += p00 * v[0] + p01 * v[1] + p02 * v[2] + p03 * v[3];
        acc1 += p00 * v[4] + p01 * v[5] + p02 * v[6] + p03 * v[7];
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            const uint r = i * KA + 16u;
            const uint s = (r <= 64u) ? uint(A1 >> (64u - r)) : uint(B1 >> (96u - r));
            v[i] = decode_state(s & 0xffffu, CB);
        }
        acc0 += p10 * v[0] + p11 * v[1] + p12 * v[2] + p13 * v[3];
        acc1 += p10 * v[4] + p11 * v[5] + p12 * v[6] + p13 * v[7];
    }

    for (; tk < tiles_k; tk += SG) {
        const device uint *t0 = column + tk * stride;
        const ulong a0 = t0[j0], a1 = t0[j1], a2 = t0[j2];
        const ulong A0 = ((a0 << 32) | a1) << sh0 | (sh0 ? (a2 >> (32 - sh0)) : 0ul);
        const ulong B0 = ((a1 << 32) | a2) << sh0;
        const device half *xr0 = x + tk * 16u + r0;
        const float p0 = float(xr0[0]), p1 = float(xr0[1]), p2 = float(xr0[8]), p3 = float(xr0[9]);
        float v[8];
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            const uint r = i * KA + 16u;
            const uint s = (r <= 64u) ? uint(A0 >> (64u - r)) : uint(B0 >> (96u - r));
            v[i] = decode_state(s & 0xffffu, CB);
        }
        acc0 += p0 * v[0] + p1 * v[1] + p2 * v[2] + p3 * v[3];
        acc1 += p0 * v[4] + p1 * v[5] + p2 * v[6] + p3 * v[7];
    }
    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if (lane % 4u == 0u) {
        red[sg][lane / 4u] = acc0;
        red[sg][lane / 4u + 8u] = acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u) {
        float sum = 0.0f;
        for (int g = 0; g < SG; ++g) {
            sum += red[g][lid];
        }
        y[tn * 16u + lid] = sum;
    }
        // `red` is reused by the next output tile, so it must be drained first.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void linear_gemv_exl3_k1(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tn [[threadgroup_position_in_grid]],
    uint tgdim [[threads_per_threadgroup]]) {
    const uint KA = 1;
    threadgroup float red[8][16];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint out_base = tn * 16u;

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        const uint tile_index = tk * tiles_n + tn;
        float v[8];
        exl3_decode8<true>(packed, tile_index, lane, KA, params.codebook, v);
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        acc0 += x0 * v[0] + x1 * v[1] + x2 * v[2] + x3 * v[3];
        acc1 += x0 * v[4] + x1 * v[5] + x2 * v[6] + x3 * v[7];
    }
    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if (lane % 4u == 0u) {
        red[sg][lane / 4u] = acc0;
        red[sg][lane / 4u + 8u] = acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}



kernel void linear_gemv_exl3_k1_u(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tn [[threadgroup_position_in_grid]],
    uint tgdim [[threads_per_threadgroup]]) {
    const uint KA = 1;
    threadgroup float red[8][16];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint out_base = tn * 16u;

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        const uint tile_index = tk * tiles_n + tn;
        float v[8];
        exl3_decode8<false>(packed, tile_index, lane, KA, params.codebook, v);
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        acc0 += x0 * v[0] + x1 * v[1] + x2 * v[2] + x3 * v[3];
        acc1 += x0 * v[4] + x1 * v[5] + x2 * v[6] + x3 * v[7];
    }
    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if (lane % 4u == 0u) {
        red[sg][lane / 4u] = acc0;
        red[sg][lane / 4u + 8u] = acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}



kernel void linear_gemv_exl3_k2(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tn [[threadgroup_position_in_grid]],
    uint tgdim [[threads_per_threadgroup]]) {
    const uint KA = 2;
    threadgroup float red[8][16];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint out_base = tn * 16u;

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        const uint tile_index = tk * tiles_n + tn;
        float v[8];
        exl3_decode8<true>(packed, tile_index, lane, KA, params.codebook, v);
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        acc0 += x0 * v[0] + x1 * v[1] + x2 * v[2] + x3 * v[3];
        acc1 += x0 * v[4] + x1 * v[5] + x2 * v[6] + x3 * v[7];
    }
    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if (lane % 4u == 0u) {
        red[sg][lane / 4u] = acc0;
        red[sg][lane / 4u + 8u] = acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}



kernel void linear_gemv_exl3_k2_u(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tn [[threadgroup_position_in_grid]],
    uint tgdim [[threads_per_threadgroup]]) {
    const uint KA = 2;
    threadgroup float red[8][16];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint out_base = tn * 16u;

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        const uint tile_index = tk * tiles_n + tn;
        float v[8];
        exl3_decode8<false>(packed, tile_index, lane, KA, params.codebook, v);
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        acc0 += x0 * v[0] + x1 * v[1] + x2 * v[2] + x3 * v[3];
        acc1 += x0 * v[4] + x1 * v[5] + x2 * v[6] + x3 * v[7];
    }
    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if (lane % 4u == 0u) {
        red[sg][lane / 4u] = acc0;
        red[sg][lane / 4u + 8u] = acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}



kernel void linear_gemv_exl3_k3(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tn [[threadgroup_position_in_grid]],
    uint tgdim [[threads_per_threadgroup]]) {
    const uint KA = 3;
    threadgroup float red[8][16];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint out_base = tn * 16u;

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        const uint tile_index = tk * tiles_n + tn;
        float v[8];
        exl3_decode8<true>(packed, tile_index, lane, KA, params.codebook, v);
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        acc0 += x0 * v[0] + x1 * v[1] + x2 * v[2] + x3 * v[3];
        acc1 += x0 * v[4] + x1 * v[5] + x2 * v[6] + x3 * v[7];
    }
    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if (lane % 4u == 0u) {
        red[sg][lane / 4u] = acc0;
        red[sg][lane / 4u + 8u] = acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}



kernel void linear_gemv_exl3_k3_u(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tn [[threadgroup_position_in_grid]],
    uint tgdim [[threads_per_threadgroup]]) {
    const uint KA = 3;
    threadgroup float red[8][16];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint out_base = tn * 16u;

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        const uint tile_index = tk * tiles_n + tn;
        float v[8];
        exl3_decode8<false>(packed, tile_index, lane, KA, params.codebook, v);
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        acc0 += x0 * v[0] + x1 * v[1] + x2 * v[2] + x3 * v[3];
        acc1 += x0 * v[4] + x1 * v[5] + x2 * v[6] + x3 * v[7];
    }
    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if (lane % 4u == 0u) {
        red[sg][lane / 4u] = acc0;
        red[sg][lane / 4u + 8u] = acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}



kernel void linear_gemv_exl3_k4(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tn [[threadgroup_position_in_grid]],
    uint tgdim [[threads_per_threadgroup]]) {
    const uint KA = 4;
    threadgroup float red[8][16];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint out_base = tn * 16u;

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        const uint tile_index = tk * tiles_n + tn;
        float v[8];
        exl3_decode8<true>(packed, tile_index, lane, KA, params.codebook, v);
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        acc0 += x0 * v[0] + x1 * v[1] + x2 * v[2] + x3 * v[3];
        acc1 += x0 * v[4] + x1 * v[5] + x2 * v[6] + x3 * v[7];
    }
    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if (lane % 4u == 0u) {
        red[sg][lane / 4u] = acc0;
        red[sg][lane / 4u + 8u] = acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}



kernel void linear_gemv_exl3_k4_u(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tn [[threadgroup_position_in_grid]],
    uint tgdim [[threads_per_threadgroup]]) {
    const uint KA = 4;
    threadgroup float red[8][16];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint out_base = tn * 16u;

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        const uint tile_index = tk * tiles_n + tn;
        float v[8];
        exl3_decode8<false>(packed, tile_index, lane, KA, params.codebook, v);
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        acc0 += x0 * v[0] + x1 * v[1] + x2 * v[2] + x3 * v[3];
        acc1 += x0 * v[4] + x1 * v[5] + x2 * v[6] + x3 * v[7];
    }
    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if (lane % 4u == 0u) {
        red[sg][lane / 4u] = acc0;
        red[sg][lane / 4u + 8u] = acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}



kernel void linear_gemv_exl3_k5(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tn [[threadgroup_position_in_grid]],
    uint tgdim [[threads_per_threadgroup]]) {
    const uint KA = 5;
    threadgroup float red[8][16];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint out_base = tn * 16u;

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        const uint tile_index = tk * tiles_n + tn;
        float v[8];
        exl3_decode8<true>(packed, tile_index, lane, KA, params.codebook, v);
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        acc0 += x0 * v[0] + x1 * v[1] + x2 * v[2] + x3 * v[3];
        acc1 += x0 * v[4] + x1 * v[5] + x2 * v[6] + x3 * v[7];
    }
    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if (lane % 4u == 0u) {
        red[sg][lane / 4u] = acc0;
        red[sg][lane / 4u + 8u] = acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}



kernel void linear_gemv_exl3_k5_u(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tn [[threadgroup_position_in_grid]],
    uint tgdim [[threads_per_threadgroup]]) {
    const uint KA = 5;
    threadgroup float red[8][16];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint out_base = tn * 16u;

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        const uint tile_index = tk * tiles_n + tn;
        float v[8];
        exl3_decode8<false>(packed, tile_index, lane, KA, params.codebook, v);
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        acc0 += x0 * v[0] + x1 * v[1] + x2 * v[2] + x3 * v[3];
        acc1 += x0 * v[4] + x1 * v[5] + x2 * v[6] + x3 * v[7];
    }
    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if (lane % 4u == 0u) {
        red[sg][lane / 4u] = acc0;
        red[sg][lane / 4u + 8u] = acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}



kernel void linear_gemv_exl3_k6(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tn [[threadgroup_position_in_grid]],
    uint tgdim [[threads_per_threadgroup]]) {
    const uint KA = 6;
    threadgroup float red[8][16];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint out_base = tn * 16u;

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        const uint tile_index = tk * tiles_n + tn;
        float v[8];
        exl3_decode8<true>(packed, tile_index, lane, KA, params.codebook, v);
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        acc0 += x0 * v[0] + x1 * v[1] + x2 * v[2] + x3 * v[3];
        acc1 += x0 * v[4] + x1 * v[5] + x2 * v[6] + x3 * v[7];
    }
    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if (lane % 4u == 0u) {
        red[sg][lane / 4u] = acc0;
        red[sg][lane / 4u + 8u] = acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}



kernel void linear_gemv_exl3_k6_u(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tn [[threadgroup_position_in_grid]],
    uint tgdim [[threads_per_threadgroup]]) {
    const uint KA = 6;
    threadgroup float red[8][16];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint out_base = tn * 16u;

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        const uint tile_index = tk * tiles_n + tn;
        float v[8];
        exl3_decode8<false>(packed, tile_index, lane, KA, params.codebook, v);
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        acc0 += x0 * v[0] + x1 * v[1] + x2 * v[2] + x3 * v[3];
        acc1 += x0 * v[4] + x1 * v[5] + x2 * v[6] + x3 * v[7];
    }
    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if (lane % 4u == 0u) {
        red[sg][lane / 4u] = acc0;
        red[sg][lane / 4u + 8u] = acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}



kernel void linear_gemv_exl3_k7(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tn [[threadgroup_position_in_grid]],
    uint tgdim [[threads_per_threadgroup]]) {
    const uint KA = 7;
    threadgroup float red[8][16];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint out_base = tn * 16u;

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        const uint tile_index = tk * tiles_n + tn;
        float v[8];
        exl3_decode8<true>(packed, tile_index, lane, KA, params.codebook, v);
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        acc0 += x0 * v[0] + x1 * v[1] + x2 * v[2] + x3 * v[3];
        acc1 += x0 * v[4] + x1 * v[5] + x2 * v[6] + x3 * v[7];
    }
    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if (lane % 4u == 0u) {
        red[sg][lane / 4u] = acc0;
        red[sg][lane / 4u + 8u] = acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}



kernel void linear_gemv_exl3_k7_u(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tn [[threadgroup_position_in_grid]],
    uint tgdim [[threads_per_threadgroup]]) {
    const uint KA = 7;
    threadgroup float red[8][16];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint out_base = tn * 16u;

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        const uint tile_index = tk * tiles_n + tn;
        float v[8];
        exl3_decode8<false>(packed, tile_index, lane, KA, params.codebook, v);
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        acc0 += x0 * v[0] + x1 * v[1] + x2 * v[2] + x3 * v[3];
        acc1 += x0 * v[4] + x1 * v[5] + x2 * v[6] + x3 * v[7];
    }
    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if (lane % 4u == 0u) {
        red[sg][lane / 4u] = acc0;
        red[sg][lane / 4u + 8u] = acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}



kernel void linear_gemv_exl3_k8(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tn [[threadgroup_position_in_grid]],
    uint tgdim [[threads_per_threadgroup]]) {
    const uint KA = 8;
    threadgroup float red[8][16];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint out_base = tn * 16u;

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        const uint tile_index = tk * tiles_n + tn;
        float v[8];
        exl3_decode8<true>(packed, tile_index, lane, KA, params.codebook, v);
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        acc0 += x0 * v[0] + x1 * v[1] + x2 * v[2] + x3 * v[3];
        acc1 += x0 * v[4] + x1 * v[5] + x2 * v[6] + x3 * v[7];
    }
    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if (lane % 4u == 0u) {
        red[sg][lane / 4u] = acc0;
        red[sg][lane / 4u + 8u] = acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}



kernel void linear_gemv_exl3_k8_u(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tn [[threadgroup_position_in_grid]],
    uint tgdim [[threads_per_threadgroup]]) {
    const uint KA = 8;
    threadgroup float red[8][16];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint out_base = tn * 16u;

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        const uint tile_index = tk * tiles_n + tn;
        float v[8];
        exl3_decode8<false>(packed, tile_index, lane, KA, params.codebook, v);
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        acc0 += x0 * v[0] + x1 * v[1] + x2 * v[2] + x3 * v[3];
        acc1 += x0 * v[4] + x1 * v[5] + x2 * v[6] + x3 * v[7];
    }
    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if (lane % 4u == 0u) {
        red[sg][lane / 4u] = acc0;
        red[sg][lane / 4u + 8u] = acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.out_features) {
        return;
    }
    float sum = 0.0f;
    uint tile_column = gid / 16u;
    for (uint i = 0; i < params.in_features; ++i) {
        uint tile_row = i / 16u;
        uint local_row = i & 15u;
        uint local_column = gid & 15u;
        uint tile_index = tile_row * params.tile_columns + tile_column;
        uint state = packed_state(packed, tile_index, local_row, local_column, params.packed_words, params.rate, params.perm_mode);
        sum += float(input[i]) * decode_state(state, params.codebook);
    }
    output[gid] = sum;
}

kernel void linear_finalize(
    const device float *input [[buffer(0)]],
    const device half *output_signs [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &blocks [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= blocks) {
        return;
    }
    float values[128];
    uint base = gid * 128u;
    for (uint i = 0; i < 128u; ++i) {
        values[i] = input[base + i];
    }
    transform_vector(values);
    for (uint i = 0; i < 128u; ++i) {
        output[base + i] = half(values[i] * float(output_signs[base + i]));
    }
}

kernel void linear_input_rows(
    const device half *input [[buffer(0)]],
    const device half *input_signs [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant PrefillRowParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint vectors_per_row = params.in_features / 128u;
    uint total = params.queries * vectors_per_row;
    if (gid >= total) {
        return;
    }
    float values[128];
    uint base = gid * 128u;
    uint sign_base = (gid % vectors_per_row) * 128u;
    for (uint i = 0; i < 128u; ++i) {
        values[i] = float(input[base + i]) * float(input_signs[sign_base + i]);
    }
    transform_vector(values);
    for (uint i = 0; i < 128u; ++i) {
        output[base + i] = half(values[i]);
    }
}

// exl3_code_at's index arithmetic, reading the 16x16 tile out of threadgroup
// memory instead of global. Kept adjacent so the two stay in step.
static inline uint exl3_code_at_tg(
    const threadgroup uint *tile,
    uint word_count,
    uint local_row,
    uint local_column,
    uint rate) {
    const uint ka = rate;
    const uint ring = 256u * ka;
    uint p = (local_row % 8u) / 2u;
    uint i_mod4 = (local_row % 2u) + 2u * (local_row / 8u);
    uint q = local_column % 8u;
    uint i_div4 = local_column / 8u;
    uint i = i_mod4 + 4u * i_div4;
    uint lane = p + 4u * q;
    uint lo = (lane * 8u + 1u) * ka + ring - 16u;
    uint i0 = lo / 32u;
    uint sh = lo % 32u;
    uint j0 = i0 >= word_count ? i0 - word_count : i0;
    uint j1 = j0 + 1u >= word_count ? j0 + 1u - word_count : j0 + 1u;
    uint j2 = j1 + 1u >= word_count ? j1 + 1u - word_count : j1 + 1u;
    const ulong w0 = tile[j0];
    const ulong w1 = tile[j1];
    const ulong w2 = tile[j2];
    const ulong a = ((w0 << 32) | w1) << sh | (sh ? (w2 >> (32 - sh)) : 0ul);
    const ulong c = ((w1 << 32) | w2) << sh;
    const uint r = i * ka + 16u;
    return (r <= 64u ? uint(a >> (64u - r)) : uint(c >> (96u - r))) & 0xffffu;
}

kernel void linear_gemv_tiled(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint3 local_position [[thread_position_in_threadgroup]],
    uint3 grid_position [[thread_position_in_grid]]) {
    // Changes from the original, all bit-exact (k = tile_row*16 + local_input still
    // runs 0,1,...,K-1 in order and each output still takes one fp32 add of one
    // product per k):
    //   - the accumulator is a register, written once, not a global read-modify-
    //     write per multiply;
    //   - the trellis words for a chunk are pulled in with one coalesced pass
    //     instead of six scattered 2-byte loads per code;
    //   - the staged k-range is 64 rather than 16, so the same two barriers
    //     amortise over four times as many FMAs.
    constexpr uint KCHUNK = 64u;
    constexpr uint TILES = KCHUNK / 16u;
    // `params.rate` encodes a fractional rate as 16 + bits (`Exl3Rate::encoded`).
    // A fractional tile holds `16*bits + 8` 16-bit words == `8*bits + 4` 32-bit
    // words, NOT `256 * rate / 32`. Before this branch the tiled prefill kernel
    // decoded fractional tiles with the integer ring arithmetic and produced
    // wrong numbers (the model is integer-only, so nothing caught it).
    const bool fractional = params.rate >= 16u;
    const uint fractional_bits = params.rate - 16u;
    const uint word_count = fractional ? (8u * fractional_bits + 4u) : (256u * params.rate / 32u);
    threadgroup uint tw[TILES * 64];
    threadgroup half wstage[TILES * 256u];
    uint local_query = (grid_position.y / 16u) * 16u + local_position.y;
    uint local_output_in_tile = local_position.x;
    uint tile_column = grid_position.x / 16u;
    uint output_index = tile_column * 16u + local_output_in_tile;
    uint tid = local_position.x + local_position.y * 16u;

    const bool active = local_query < params.queries;
    const uint query_base = local_query * params.in_features;
    const uint output_base = local_query * params.out_features;
    float accumulator = 0.0f;

    for (uint kbase = 0; kbase < params.in_features; kbase += KCHUNK) {
        for (uint i = tid; i < TILES * word_count; i += 256u) {
            uint t = i / word_count;
            uint w = i - t * word_count;
            uint tile_index = (kbase / 16u + t) * params.tile_columns + tile_column;
            tw[i] = trellis_word(packed, tile_index * word_count + w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid; i < TILES * 256u; i += 256u) {
            uint t = i / 256u;
            uint lr = (i & 255u) / 16u;
            uint lc = i & 15u;
            uint state = fractional
                ? fractional_window_tg(tw + t * word_count, word_count, lr * 16u + lc, fractional_bits)
                : exl3_code_at_tg(tw + t * word_count, word_count, lr, lc, params.rate);
            wstage[i] = half(decode_state(state, params.codebook));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            for (uint k_local = 0; k_local < KCHUNK; ++k_local) {
                accumulator += float(input[query_base + kbase + k_local]) *
                               float(wstage[k_local * 16u + local_output_in_tile]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        output[output_base + output_index] = accumulator;
    }
}

// Tensor-core prefill GEMM. Motivation (session notes SS8.4):
// linear_gemv_tiled spends its inner loop in one serial fp32 FMA chain per
// thread and re-decodes the same weight tile in every query-block threadgroup;
// measured in situ it runs a 256-token chunk at ~228 GFLOPS, ~3% of the 8.76
// TFLOPS SIMT roofline and ~1.5% of the tensor roofline measured by
// examples/mma_peak.rs.
//
// Geometry: one 256-thread threadgroup (8 simdgroups) computes a 128-query x
// 64-output block. Simdgroup s owns query rows [s*16, s*16+16) and keeps SIXTEEN
// NAMED 8x8 f32 accumulators: two row blocks x eight 8-column chunks. Named
// locals are not a style choice: mma_peak.rs measured an ARRAY of chains
// spilling to threadgroup memory and collapsing from 15.49 to 0.74 TFLOPS.
//
// Six things set the throughput, and the first three are what the previous
// 64-query x 32-output / 128-thread shape (32.2 -> 67.8 tok/s in situ, below)
// got wrong. The binding constraint was never tensor peak -- it was the
// instruction count spent on STAGING rather than on multiplies, and on the
// fragment loads that feed them.
//
//  1. The weight dequant is lane-parallel. `exl3_code_at_tg` derives the trellis
//     window per weight, so 16 weights per thread cost three threadgroup loads
//     and a 64-bit window setup EACH: ~400 instructions against 64 multiplies,
//     and the kernel ran at 13% of the tensor roofline. Here one thread decodes
//     the EIGHT codes a trellis lane owns from ONE window setup, exactly as
//     `exl3_decode8` does in the decode path: 16 weights per thread cost ~40
//     instructions. The codes are bit-identical (same window, same two
//     overlapping 64-bit windows, same 16-bit slices), so this is a pure
//     instruction-count win with no numerical change.
//
//  2. Activations are NOT staged. Each simdgroup owns a DISJOINT set of query
//     rows, so a staged astage was read exactly once by exactly one simdgroup:
//     32 global loads plus 32 threadgroup stores per thread per chunk bought
//     coalescing and nothing else. The A fragments now load straight from the
//     transformed activation buffer, which frees 8 KB of threadgroup memory.
//     Measured: staging them anyway (astage, 14.8 KB) was 5% SLOWER.
//
//  3. The tile is 128x64 rather than 64x32, so sixteen accumulators share ten
//     fragment loads per step -- 16 multiplies per fragment load, against 8 per
//     6. Weights are also streamed ceil(queries/128) times per output block
//     instead of ceil(queries/64), halving both the weight DMA and the
//     activation re-reads. A 32-query row block per simdgroup (32
//     accumulators, 2.67 multiplies per load) was measured and is 10% slower:
//     64 accumulator registers per thread spill or throttle occupancy.
//
//  4. The trellis words are read from global memory straight into registers,
//     not staged through threadgroup memory first. A staged tw block needs a
//     barrier whose only job is to publish a 1.5 KB copy, and it measured 15%
//     -- more than the dequant arithmetic it fed. Software-pipelined one chunk
//     ahead (below), so the load latency hides behind the multiplies.
//
//  5. BSTRIDE pads each 64-column weight row to 72 halfs. The lane-parallel
//     dequant scatters one 16-byte piece into each of eight rows; at a 128-byte
//     stride all eight land on threadgroup bank 0, an eight-way conflict on
//     every store. At 144 bytes the row bases fall on banks 0,4,...,28 and the
//     eight pieces are conflict-free.
//
//  6. Outputs leave the accumulators directly. A 128x64 f32 staging tile is
//     32 KB, far past the 16 KB ceiling, and `simdgroup_store` has no per-row
//     mask, so each lane recomputes its own (row, column) from its lane id and
//     does a row-guarded 4-byte store. The old shape needed an 8 KB tgmix bounce
//     for exactly this reason.
//
// Threadgroup budget: 64 x 72 halfs of dequantized weights = 9216 B, and
// nothing else. That is below the 14.3 KB of the shape it replaces and well
// under the 16 KB that costs occupancy (18 KB measured 21.1 tok/s; 27 KB -18%
// occupancy, dsa_indexer.metal:48-50).
//
// Per 64-wide K chunk the group (1) issues the next chunk's trellis-word loads,
// (2) dequantizes the current chunk's 16 tiles into 64x64 halfs (4 tile columns
// x 4 tile rows -- the same decode helpers, the same bit-exact codes as
// linear_gemv_tiled), (3) runs 8 steps x 16 MMAs, and repeats.
//
// Products are half x half accumulated in f32; the summation order differs from
// linear_gemv_tiled so outputs are close but not bit-identical (Q10: tensor
// path accepted by user decision 2026-09-27). The tile shape does not change
// any element's accumulation order, so this kernel and the 64x32 one agree to
// the last bit (2.1603336e-3 normalised error against the CPU reference, all
// of it from the fp16 weights).
kernel void linear_gemm_mma(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    constexpr uint KCHUNK = 64u;
    constexpr uint KTILES = KCHUNK / 16u; // 4 k-tiles per chunk
    constexpr uint NTILES = 4u;           // 4 output tile columns
    constexpr uint QTILE = 128u;          // query rows per threadgroup
    constexpr uint OUTS = 64u;            // output columns per threadgroup
    constexpr uint BSTRIDE = OUTS + 8u;   // bank-padded weight row stride
    constexpr uint SGROUPS = 8u;
    constexpr uint THREADS = SGROUPS * 32u;
    constexpr uint ROWS_SG = QTILE / SGROUPS;  // 16 query rows per simdgroup
    const uint sgid = tid / 32u;

    // `params.rate` encodes a fractional rate as 16 + bits; see
    // linear_gemv_tiled for the word-count derivation.
    const bool fractional = params.rate >= 16u;
    const uint fractional_bits = params.rate - 16u;
    const uint word_count = fractional ? (8u * fractional_bits + 4u) : (256u * params.rate / 32u);
    const uint ka = params.rate;

    threadgroup half bstage[KCHUNK * BSTRIDE];

    // Rows past the query count would read past the activation buffer, so clamp
    // the two A-fragment row pointers; their accumulators are never stored.
    const uint rbase = group.y * QTILE + sgid * ROWS_SG;
    const device half *ab0 = input + min(rbase, params.queries - 1u) * params.in_features;
    const device half *ab1 = input + min(rbase + 8u, params.queries - 1u) * params.in_features;
    // Lane -> (row, column) inside an 8x8 simdgroup_matrix, the inverse of the
    // load layout: element 0 of the lane's pair sits at (fm, fn).
    const uint lane = tid & 31u;
    const uint lane_fm = (lane >> 2 & 4u) + (lane >> 1 & 3u);
    const uint lane_fn = (lane >> 2 & 2u) << 1 | (lane & 1u) << 1;

    // Sixteen named accumulators. An array of chains spills to threadgroup
    // memory and collapses 15.49 -> 0.74 TFLOPS (mma_peak.rs).
    simdgroup_float8x8 acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    simdgroup_float8x8 acc8, acc9, acc10, acc11, acc12, acc13, acc14, acc15;
    acc0.thread_elements()[0] = 0.0f; acc0.thread_elements()[1] = 0.0f;
    acc1.thread_elements()[0] = 0.0f; acc1.thread_elements()[1] = 0.0f;
    acc2.thread_elements()[0] = 0.0f; acc2.thread_elements()[1] = 0.0f;
    acc3.thread_elements()[0] = 0.0f; acc3.thread_elements()[1] = 0.0f;
    acc4.thread_elements()[0] = 0.0f; acc4.thread_elements()[1] = 0.0f;
    acc5.thread_elements()[0] = 0.0f; acc5.thread_elements()[1] = 0.0f;
    acc6.thread_elements()[0] = 0.0f; acc6.thread_elements()[1] = 0.0f;
    acc7.thread_elements()[0] = 0.0f; acc7.thread_elements()[1] = 0.0f;
    acc8.thread_elements()[0] = 0.0f; acc8.thread_elements()[1] = 0.0f;
    acc9.thread_elements()[0] = 0.0f; acc9.thread_elements()[1] = 0.0f;
    acc10.thread_elements()[0] = 0.0f; acc10.thread_elements()[1] = 0.0f;
    acc11.thread_elements()[0] = 0.0f; acc11.thread_elements()[1] = 0.0f;
    acc12.thread_elements()[0] = 0.0f; acc12.thread_elements()[1] = 0.0f;
    acc13.thread_elements()[0] = 0.0f; acc13.thread_elements()[1] = 0.0f;
    acc14.thread_elements()[0] = 0.0f; acc14.thread_elements()[1] = 0.0f;
    acc15.thread_elements()[0] = 0.0f; acc15.thread_elements()[1] = 0.0f;

    // The window a trellis lane reads: three consecutive 32-bit words starting
    // at `lo / 32` in the tile, wrapped into the tile's ring. Same arithmetic as
    // exl3_decode8, in variables shared with the dequant below. `gbase` is a
    // USHORT offset because `packed` is a `const device ushort *`, and the word
    // index is doubled accordingly.
    thread uint pre[6];
    {
        #pragma unroll
        for (uint jw = 0u; jw < 2u; ++jw) {
            const uint w = tid + jw * THREADS;
            const uint t = w / 32u;
            const uint l = w & 31u;
            const uint lo = (l * 8u + 1u) * ka + 256u * ka - 16u;
            const uint i0 = lo / 32u;
            const uint j0 = i0 >= word_count ? i0 - word_count : i0;
            const uint j1 = j0 + 1u >= word_count ? j0 + 1u - word_count : j0 + 1u;
            const uint j2 = j1 + 1u >= word_count ? j1 + 1u - word_count : j1 + 1u;
            const uint gbase = 2u * ((t / NTILES) * params.tile_columns +
                                     (group.x * NTILES + (t % NTILES))) * word_count;
            pre[3u * jw + 0u] = trellis_word(packed + gbase, j0);
            pre[3u * jw + 1u] = trellis_word(packed + gbase, j1);
            pre[3u * jw + 2u] = trellis_word(packed + gbase, j2);
        }
    }
    for (uint kbase = 0; kbase < params.in_features; kbase += KCHUNK) {
        if (fractional) {
            // The fractional packer has no lane grouping to exploit (its window
            // ring is indexed by the flat tile position), so keep the
            // one-window-per-weight form, reading the device tile directly --
            // `fractional_window` is the same ring arithmetic as its threadgroup
            // twin `fractional_window_tg`. The model is integer-only; this path
            // exists so the kernel stays correct for every rate the loader can
            // produce.
            for (uint i = tid; i < KCHUNK * OUTS; i += THREADS) {
                uint kk = i / OUTS;
                uint n = i % OUTS;
                uint t = (kk / 16u) * NTILES + (n / 16u);
                uint gbase = 2u * ((kbase / 16u + t / NTILES) * params.tile_columns +
                                   group.x * NTILES + (t % NTILES)) * word_count;
                uint state = fractional_window(packed + gbase, (kk & 15u) * 16u + (n & 15u), fractional_bits);
                bstage[kk * BSTRIDE + n] = half(decode_state(state, params.codebook));
            }
        } else {
            // One thread per (tile, trellis lane). Lane l owns tile rows
            // {r0, r0+1, r0+8, r0+9} with r0 = 2*(l%4) and tile columns
            // {c, c+8} with c = l/4, indexed i = j + 4*dc; the 32 lanes tile
            // the 16x16 tile exactly once. 16 tiles x 32 lanes = 512 work items,
            // two per thread at 256 threads. `pre` holds this chunk's three
            // window words, fetched one chunk ahead.
            #pragma unroll
            for (uint jw = 0u; jw < 2u; ++jw) {
                const uint w = tid + jw * THREADS;
                const uint t = w / 32u;
                const uint l = w & 31u;
                const uint r0 = (l & 3u) * 2u;
                const uint c0 = l >> 2;
                const uint lo = (l * 8u + 1u) * ka + 256u * ka - 16u;
                const uint i0 = lo / 32u;
                const uint sh = lo % 32u;
                const ulong g0 = pre[3u * jw + 0u];
                const ulong g1 = pre[3u * jw + 1u];
                const ulong g2 = pre[3u * jw + 2u];
                const ulong wa = ((g0 << 32) | g1) << sh | (sh ? (g2 >> (32 - sh)) : 0ul);
                const ulong wc = ((g1 << 32) | g2) << sh;
                #pragma unroll
                for (uint j = 0u; j < 4u; ++j) {
                    const uint rr = (t / NTILES) * 16u + r0 + (j & 1u) + ((j >> 1u) << 3u);
                    for (uint dc = 0u; dc < 2u; ++dc) {
                        const uint r = (j + 4u * dc) * ka + 16u;
                        const uint s = (r <= 64u) ? uint(wa >> (64u - r)) : uint(wc >> (96u - r));
                        bstage[rr * BSTRIDE + (t % NTILES) * 16u + c0 + dc * 8u] =
                            half(decode_state(s & 0xffffu, params.codebook));
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Next chunk's trellis words go in flight here, so their latency is
        // covered by this chunk's multiplies. They are only consumed after the
        // closing barrier of this iteration.
        if (kbase + KCHUNK < params.in_features) {
            #pragma unroll
            for (uint jw = 0u; jw < 2u; ++jw) {
                const uint w = tid + jw * THREADS;
                const uint t = w / 32u;
                const uint l = w & 31u;
                const uint lo = (l * 8u + 1u) * ka + 256u * ka - 16u;
                const uint i0 = lo / 32u;
                const uint j0 = i0 >= word_count ? i0 - word_count : i0;
                const uint j1 = j0 + 1u >= word_count ? j0 + 1u - word_count : j0 + 1u;
                const uint j2 = j1 + 1u >= word_count ? j1 + 1u - word_count : j1 + 1u;
                const uint gbase = 2u * (((kbase + KCHUNK) / 16u + t / NTILES) * params.tile_columns +
                                     (group.x * NTILES + (t % NTILES))) * word_count;
                pre[3u * jw + 0u] = trellis_word(packed + gbase, j0);
                pre[3u * jw + 1u] = trellis_word(packed + gbase, j1);
                pre[3u * jw + 2u] = trellis_word(packed + gbase, j2);
            }
        }

        // Eight MMA steps over a 64k slice: 2 A fragments x 8 B fragments feed
        // 16 multiplies per step, against 2 x 4 feeding 8 in the 64x32 shape.
        for (uint step = 0; step < KCHUNK / 8u; ++step) {
            const uint k8 = step * 8u;
            simdgroup_half8x8 x0, x1;
            simdgroup_load(x0, ab0 + kbase + k8, params.in_features);
            simdgroup_load(x1, ab1 + kbase + k8, params.in_features);
            simdgroup_half8x8 y0, y1, y2, y3, y4, y5, y6, y7;
            const threadgroup half *bs = bstage + k8 * BSTRIDE;
            simdgroup_load(y0, bs + 0u, BSTRIDE);
            simdgroup_load(y1, bs + 8u, BSTRIDE);
            simdgroup_load(y2, bs + 16u, BSTRIDE);
            simdgroup_load(y3, bs + 24u, BSTRIDE);
            simdgroup_load(y4, bs + 32u, BSTRIDE);
            simdgroup_load(y5, bs + 40u, BSTRIDE);
            simdgroup_load(y6, bs + 48u, BSTRIDE);
            simdgroup_load(y7, bs + 56u, BSTRIDE);
            simdgroup_multiply_accumulate(acc0, x0, y0, acc0);
            simdgroup_multiply_accumulate(acc1, x0, y1, acc1);
            simdgroup_multiply_accumulate(acc2, x0, y2, acc2);
            simdgroup_multiply_accumulate(acc3, x0, y3, acc3);
            simdgroup_multiply_accumulate(acc4, x0, y4, acc4);
            simdgroup_multiply_accumulate(acc5, x0, y5, acc5);
            simdgroup_multiply_accumulate(acc6, x0, y6, acc6);
            simdgroup_multiply_accumulate(acc7, x0, y7, acc7);
            simdgroup_multiply_accumulate(acc8, x1, y0, acc8);
            simdgroup_multiply_accumulate(acc9, x1, y1, acc9);
            simdgroup_multiply_accumulate(acc10, x1, y2, acc10);
            simdgroup_multiply_accumulate(acc11, x1, y3, acc11);
            simdgroup_multiply_accumulate(acc12, x1, y4, acc12);
            simdgroup_multiply_accumulate(acc13, x1, y5, acc13);
            simdgroup_multiply_accumulate(acc14, x1, y6, acc14);
            simdgroup_multiply_accumulate(acc15, x1, y7, acc15);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Row = rbase + 8*rowblock + fm, column = group.x*OUTS + 8*chunk + fn +
    // element, so every store is row-guarded and a partial query block never
    // touches another row's output.
    const uint colbase = group.x * OUTS;
#define KAPLUMBA_MMA_STORE(acc, rowoffset, chunk)                                  \
    {                                                                             \
        const uint r = rbase + (rowoffset) + lane_fm;                              \
        if (r < params.queries) {                                                 \
            const uint c = colbase + 8u * (chunk) + lane_fn;                      \
            output[r * params.out_features + c] = (acc).thread_elements()[0];     \
            output[r * params.out_features + c + 1u] = (acc).thread_elements()[1]; \
        }                                                                         \
    }
    KAPLUMBA_MMA_STORE(acc0, 0u, 0u);
    KAPLUMBA_MMA_STORE(acc1, 0u, 1u);
    KAPLUMBA_MMA_STORE(acc2, 0u, 2u);
    KAPLUMBA_MMA_STORE(acc3, 0u, 3u);
    KAPLUMBA_MMA_STORE(acc4, 0u, 4u);
    KAPLUMBA_MMA_STORE(acc5, 0u, 5u);
    KAPLUMBA_MMA_STORE(acc6, 0u, 6u);
    KAPLUMBA_MMA_STORE(acc7, 0u, 7u);
    KAPLUMBA_MMA_STORE(acc8, 8u, 0u);
    KAPLUMBA_MMA_STORE(acc9, 8u, 1u);
    KAPLUMBA_MMA_STORE(acc10, 8u, 2u);
    KAPLUMBA_MMA_STORE(acc11, 8u, 3u);
    KAPLUMBA_MMA_STORE(acc12, 8u, 4u);
    KAPLUMBA_MMA_STORE(acc13, 8u, 5u);
    KAPLUMBA_MMA_STORE(acc14, 8u, 6u);
    KAPLUMBA_MMA_STORE(acc15, 8u, 7u);
#undef KAPLUMBA_MMA_STORE
}

kernel void linear_finalize_rows(
    const device float *input [[buffer(0)]],
    const device half *output_signs [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant PrefillRowParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint vectors_per_row = params.out_features / 128u;
    uint total = params.queries * vectors_per_row;
    if (gid >= total) {
        return;
    }
    uint query = gid / vectors_per_row;
    uint block = gid % vectors_per_row;
    uint base = query * params.out_features + block * 128u;
    uint sign_base = block * 128u;
    float values[128];
    for (uint i = 0; i < 128u; ++i) {
        values[i] = input[base + i];
    }
    transform_vector(values);
    for (uint i = 0; i < 128u; ++i) {
        output[base + i] = half(values[i] * float(output_signs[sign_base + i]));
    }
}

// Zeroes the f32 accumulator before a tiled prefill pass. The tiled kernel sums
// across query tiles, so each dispatch must start from a clean buffer; doing it
// on device keeps the whole prefill inside one command buffer.
kernel void linear_zero_raw(
    device float *raw [[buffer(0)]],
    uint gid [[thread_position_in_grid]]) {
    raw[gid] = 0.0f;
}

// ---------------------------------------------------------------------------
// Specialised single-query decode kernels, one per integer bit rate.
//
// The generic kernel resolves the tile word count at run time and redoes the lane
// scatter for every weight. These specialise the bit rate so the tile geometry is
// a literal, and split the accumulation across four independent accumulators to
// break the serial FP dependency chain. All eight call the shared
// `exl3_code_at`, so they agree bit for bit with the generic kernel and with the
// CPU reference.

// Decode one EXL3 code and accumulate it. Generated for integer bit rate 1.
kernel void linear_gemv_fast1(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.out_features) {
        return;
    }
    const uint BITS = 1u;
    const uint TOTAL = 256u;
    const uint WORD_COUNT = 8u;
    const uint local_column = gid & 15u;
    const uint tile_column = gid / 16u;
    const uint tiles = params.in_features / 16u;
    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    for (uint t = 0; t < tiles; ++t) {
        const uint tile_index = t * params.tile_columns + tile_column;
        const uint base = t * 16u;
        a0 += float(input[base + 0u]) * decode_state(exl3_code_at(packed, tile_index, 0u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 1u]) * decode_state(exl3_code_at(packed, tile_index, 1u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 2u]) * decode_state(exl3_code_at(packed, tile_index, 2u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 3u]) * decode_state(exl3_code_at(packed, tile_index, 3u, local_column, WORD_COUNT, BITS), params.codebook);
        a0 += float(input[base + 4u]) * decode_state(exl3_code_at(packed, tile_index, 4u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 5u]) * decode_state(exl3_code_at(packed, tile_index, 5u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 6u]) * decode_state(exl3_code_at(packed, tile_index, 6u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 7u]) * decode_state(exl3_code_at(packed, tile_index, 7u, local_column, WORD_COUNT, BITS), params.codebook);
        a0 += float(input[base + 8u]) * decode_state(exl3_code_at(packed, tile_index, 8u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 9u]) * decode_state(exl3_code_at(packed, tile_index, 9u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 10u]) * decode_state(exl3_code_at(packed, tile_index, 10u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 11u]) * decode_state(exl3_code_at(packed, tile_index, 11u, local_column, WORD_COUNT, BITS), params.codebook);
        a0 += float(input[base + 12u]) * decode_state(exl3_code_at(packed, tile_index, 12u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 13u]) * decode_state(exl3_code_at(packed, tile_index, 13u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 14u]) * decode_state(exl3_code_at(packed, tile_index, 14u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 15u]) * decode_state(exl3_code_at(packed, tile_index, 15u, local_column, WORD_COUNT, BITS), params.codebook);
    }
    output[gid] = (a0 + a1) + (a2 + a3);
}

// Decode one EXL3 code and accumulate it. Generated for integer bit rate 2.
kernel void linear_gemv_fast2(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.out_features) {
        return;
    }
    const uint BITS = 2u;
    const uint TOTAL = 512u;
    const uint WORD_COUNT = 16u;
    const uint local_column = gid & 15u;
    const uint tile_column = gid / 16u;
    const uint tiles = params.in_features / 16u;
    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    for (uint t = 0; t < tiles; ++t) {
        const uint tile_index = t * params.tile_columns + tile_column;
        const uint base = t * 16u;
        a0 += float(input[base + 0u]) * decode_state(exl3_code_at(packed, tile_index, 0u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 1u]) * decode_state(exl3_code_at(packed, tile_index, 1u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 2u]) * decode_state(exl3_code_at(packed, tile_index, 2u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 3u]) * decode_state(exl3_code_at(packed, tile_index, 3u, local_column, WORD_COUNT, BITS), params.codebook);
        a0 += float(input[base + 4u]) * decode_state(exl3_code_at(packed, tile_index, 4u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 5u]) * decode_state(exl3_code_at(packed, tile_index, 5u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 6u]) * decode_state(exl3_code_at(packed, tile_index, 6u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 7u]) * decode_state(exl3_code_at(packed, tile_index, 7u, local_column, WORD_COUNT, BITS), params.codebook);
        a0 += float(input[base + 8u]) * decode_state(exl3_code_at(packed, tile_index, 8u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 9u]) * decode_state(exl3_code_at(packed, tile_index, 9u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 10u]) * decode_state(exl3_code_at(packed, tile_index, 10u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 11u]) * decode_state(exl3_code_at(packed, tile_index, 11u, local_column, WORD_COUNT, BITS), params.codebook);
        a0 += float(input[base + 12u]) * decode_state(exl3_code_at(packed, tile_index, 12u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 13u]) * decode_state(exl3_code_at(packed, tile_index, 13u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 14u]) * decode_state(exl3_code_at(packed, tile_index, 14u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 15u]) * decode_state(exl3_code_at(packed, tile_index, 15u, local_column, WORD_COUNT, BITS), params.codebook);
    }
    output[gid] = (a0 + a1) + (a2 + a3);
}

// Decode one EXL3 code and accumulate it. Generated for integer bit rate 3.
kernel void linear_gemv_fast3(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.out_features) {
        return;
    }
    const uint BITS = 3u;
    const uint TOTAL = 768u;
    const uint WORD_COUNT = 24u;
    const uint local_column = gid & 15u;
    const uint tile_column = gid / 16u;
    const uint tiles = params.in_features / 16u;
    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    for (uint t = 0; t < tiles; ++t) {
        const uint tile_index = t * params.tile_columns + tile_column;
        const uint base = t * 16u;
        a0 += float(input[base + 0u]) * decode_state(exl3_code_at(packed, tile_index, 0u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 1u]) * decode_state(exl3_code_at(packed, tile_index, 1u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 2u]) * decode_state(exl3_code_at(packed, tile_index, 2u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 3u]) * decode_state(exl3_code_at(packed, tile_index, 3u, local_column, WORD_COUNT, BITS), params.codebook);
        a0 += float(input[base + 4u]) * decode_state(exl3_code_at(packed, tile_index, 4u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 5u]) * decode_state(exl3_code_at(packed, tile_index, 5u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 6u]) * decode_state(exl3_code_at(packed, tile_index, 6u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 7u]) * decode_state(exl3_code_at(packed, tile_index, 7u, local_column, WORD_COUNT, BITS), params.codebook);
        a0 += float(input[base + 8u]) * decode_state(exl3_code_at(packed, tile_index, 8u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 9u]) * decode_state(exl3_code_at(packed, tile_index, 9u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 10u]) * decode_state(exl3_code_at(packed, tile_index, 10u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 11u]) * decode_state(exl3_code_at(packed, tile_index, 11u, local_column, WORD_COUNT, BITS), params.codebook);
        a0 += float(input[base + 12u]) * decode_state(exl3_code_at(packed, tile_index, 12u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 13u]) * decode_state(exl3_code_at(packed, tile_index, 13u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 14u]) * decode_state(exl3_code_at(packed, tile_index, 14u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 15u]) * decode_state(exl3_code_at(packed, tile_index, 15u, local_column, WORD_COUNT, BITS), params.codebook);
    }
    output[gid] = (a0 + a1) + (a2 + a3);
}

// Decode one EXL3 code and accumulate it. Generated for integer bit rate 4.
kernel void linear_gemv_fast4(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.out_features) {
        return;
    }
    const uint BITS = 4u;
    const uint TOTAL = 1024u;
    const uint WORD_COUNT = 32u;
    const uint local_column = gid & 15u;
    const uint tile_column = gid / 16u;
    const uint tiles = params.in_features / 16u;
    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    for (uint t = 0; t < tiles; ++t) {
        const uint tile_index = t * params.tile_columns + tile_column;
        const uint base = t * 16u;
        a0 += float(input[base + 0u]) * decode_state(exl3_code_at(packed, tile_index, 0u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 1u]) * decode_state(exl3_code_at(packed, tile_index, 1u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 2u]) * decode_state(exl3_code_at(packed, tile_index, 2u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 3u]) * decode_state(exl3_code_at(packed, tile_index, 3u, local_column, WORD_COUNT, BITS), params.codebook);
        a0 += float(input[base + 4u]) * decode_state(exl3_code_at(packed, tile_index, 4u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 5u]) * decode_state(exl3_code_at(packed, tile_index, 5u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 6u]) * decode_state(exl3_code_at(packed, tile_index, 6u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 7u]) * decode_state(exl3_code_at(packed, tile_index, 7u, local_column, WORD_COUNT, BITS), params.codebook);
        a0 += float(input[base + 8u]) * decode_state(exl3_code_at(packed, tile_index, 8u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 9u]) * decode_state(exl3_code_at(packed, tile_index, 9u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 10u]) * decode_state(exl3_code_at(packed, tile_index, 10u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 11u]) * decode_state(exl3_code_at(packed, tile_index, 11u, local_column, WORD_COUNT, BITS), params.codebook);
        a0 += float(input[base + 12u]) * decode_state(exl3_code_at(packed, tile_index, 12u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 13u]) * decode_state(exl3_code_at(packed, tile_index, 13u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 14u]) * decode_state(exl3_code_at(packed, tile_index, 14u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 15u]) * decode_state(exl3_code_at(packed, tile_index, 15u, local_column, WORD_COUNT, BITS), params.codebook);
    }
    output[gid] = (a0 + a1) + (a2 + a3);
}

// Decode one EXL3 code and accumulate it. Generated for integer bit rate 5.
kernel void linear_gemv_fast5(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.out_features) {
        return;
    }
    const uint BITS = 5u;
    const uint TOTAL = 1280u;
    const uint WORD_COUNT = 40u;
    const uint local_column = gid & 15u;
    const uint tile_column = gid / 16u;
    const uint tiles = params.in_features / 16u;
    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    for (uint t = 0; t < tiles; ++t) {
        const uint tile_index = t * params.tile_columns + tile_column;
        const uint base = t * 16u;
        a0 += float(input[base + 0u]) * decode_state(exl3_code_at(packed, tile_index, 0u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 1u]) * decode_state(exl3_code_at(packed, tile_index, 1u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 2u]) * decode_state(exl3_code_at(packed, tile_index, 2u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 3u]) * decode_state(exl3_code_at(packed, tile_index, 3u, local_column, WORD_COUNT, BITS), params.codebook);
        a0 += float(input[base + 4u]) * decode_state(exl3_code_at(packed, tile_index, 4u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 5u]) * decode_state(exl3_code_at(packed, tile_index, 5u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 6u]) * decode_state(exl3_code_at(packed, tile_index, 6u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 7u]) * decode_state(exl3_code_at(packed, tile_index, 7u, local_column, WORD_COUNT, BITS), params.codebook);
        a0 += float(input[base + 8u]) * decode_state(exl3_code_at(packed, tile_index, 8u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 9u]) * decode_state(exl3_code_at(packed, tile_index, 9u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 10u]) * decode_state(exl3_code_at(packed, tile_index, 10u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 11u]) * decode_state(exl3_code_at(packed, tile_index, 11u, local_column, WORD_COUNT, BITS), params.codebook);
        a0 += float(input[base + 12u]) * decode_state(exl3_code_at(packed, tile_index, 12u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 13u]) * decode_state(exl3_code_at(packed, tile_index, 13u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 14u]) * decode_state(exl3_code_at(packed, tile_index, 14u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 15u]) * decode_state(exl3_code_at(packed, tile_index, 15u, local_column, WORD_COUNT, BITS), params.codebook);
    }
    output[gid] = (a0 + a1) + (a2 + a3);
}

// Decode one EXL3 code and accumulate it. Generated for integer bit rate 6.
kernel void linear_gemv_fast6(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.out_features) {
        return;
    }
    const uint BITS = 6u;
    const uint TOTAL = 1536u;
    const uint WORD_COUNT = 48u;
    const uint local_column = gid & 15u;
    const uint tile_column = gid / 16u;
    const uint tiles = params.in_features / 16u;
    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    for (uint t = 0; t < tiles; ++t) {
        const uint tile_index = t * params.tile_columns + tile_column;
        const uint base = t * 16u;
        a0 += float(input[base + 0u]) * decode_state(exl3_code_at(packed, tile_index, 0u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 1u]) * decode_state(exl3_code_at(packed, tile_index, 1u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 2u]) * decode_state(exl3_code_at(packed, tile_index, 2u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 3u]) * decode_state(exl3_code_at(packed, tile_index, 3u, local_column, WORD_COUNT, BITS), params.codebook);
        a0 += float(input[base + 4u]) * decode_state(exl3_code_at(packed, tile_index, 4u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 5u]) * decode_state(exl3_code_at(packed, tile_index, 5u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 6u]) * decode_state(exl3_code_at(packed, tile_index, 6u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 7u]) * decode_state(exl3_code_at(packed, tile_index, 7u, local_column, WORD_COUNT, BITS), params.codebook);
        a0 += float(input[base + 8u]) * decode_state(exl3_code_at(packed, tile_index, 8u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 9u]) * decode_state(exl3_code_at(packed, tile_index, 9u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 10u]) * decode_state(exl3_code_at(packed, tile_index, 10u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 11u]) * decode_state(exl3_code_at(packed, tile_index, 11u, local_column, WORD_COUNT, BITS), params.codebook);
        a0 += float(input[base + 12u]) * decode_state(exl3_code_at(packed, tile_index, 12u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 13u]) * decode_state(exl3_code_at(packed, tile_index, 13u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 14u]) * decode_state(exl3_code_at(packed, tile_index, 14u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 15u]) * decode_state(exl3_code_at(packed, tile_index, 15u, local_column, WORD_COUNT, BITS), params.codebook);
    }
    output[gid] = (a0 + a1) + (a2 + a3);
}

// Decode one EXL3 code and accumulate it. Generated for integer bit rate 7.
kernel void linear_gemv_fast7(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.out_features) {
        return;
    }
    const uint BITS = 7u;
    const uint TOTAL = 1792u;
    const uint WORD_COUNT = 56u;
    const uint local_column = gid & 15u;
    const uint tile_column = gid / 16u;
    const uint tiles = params.in_features / 16u;
    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    for (uint t = 0; t < tiles; ++t) {
        const uint tile_index = t * params.tile_columns + tile_column;
        const uint base = t * 16u;
        a0 += float(input[base + 0u]) * decode_state(exl3_code_at(packed, tile_index, 0u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 1u]) * decode_state(exl3_code_at(packed, tile_index, 1u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 2u]) * decode_state(exl3_code_at(packed, tile_index, 2u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 3u]) * decode_state(exl3_code_at(packed, tile_index, 3u, local_column, WORD_COUNT, BITS), params.codebook);
        a0 += float(input[base + 4u]) * decode_state(exl3_code_at(packed, tile_index, 4u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 5u]) * decode_state(exl3_code_at(packed, tile_index, 5u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 6u]) * decode_state(exl3_code_at(packed, tile_index, 6u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 7u]) * decode_state(exl3_code_at(packed, tile_index, 7u, local_column, WORD_COUNT, BITS), params.codebook);
        a0 += float(input[base + 8u]) * decode_state(exl3_code_at(packed, tile_index, 8u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 9u]) * decode_state(exl3_code_at(packed, tile_index, 9u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 10u]) * decode_state(exl3_code_at(packed, tile_index, 10u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 11u]) * decode_state(exl3_code_at(packed, tile_index, 11u, local_column, WORD_COUNT, BITS), params.codebook);
        a0 += float(input[base + 12u]) * decode_state(exl3_code_at(packed, tile_index, 12u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 13u]) * decode_state(exl3_code_at(packed, tile_index, 13u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 14u]) * decode_state(exl3_code_at(packed, tile_index, 14u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 15u]) * decode_state(exl3_code_at(packed, tile_index, 15u, local_column, WORD_COUNT, BITS), params.codebook);
    }
    output[gid] = (a0 + a1) + (a2 + a3);
}

// Decode one EXL3 code and accumulate it. Generated for integer bit rate 8.
kernel void linear_gemv_fast8(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.out_features) {
        return;
    }
    const uint BITS = 8u;
    const uint TOTAL = 2048u;
    const uint WORD_COUNT = 64u;
    const uint local_column = gid & 15u;
    const uint tile_column = gid / 16u;
    const uint tiles = params.in_features / 16u;
    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    for (uint t = 0; t < tiles; ++t) {
        const uint tile_index = t * params.tile_columns + tile_column;
        const uint base = t * 16u;
        a0 += float(input[base + 0u]) * decode_state(exl3_code_at(packed, tile_index, 0u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 1u]) * decode_state(exl3_code_at(packed, tile_index, 1u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 2u]) * decode_state(exl3_code_at(packed, tile_index, 2u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 3u]) * decode_state(exl3_code_at(packed, tile_index, 3u, local_column, WORD_COUNT, BITS), params.codebook);
        a0 += float(input[base + 4u]) * decode_state(exl3_code_at(packed, tile_index, 4u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 5u]) * decode_state(exl3_code_at(packed, tile_index, 5u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 6u]) * decode_state(exl3_code_at(packed, tile_index, 6u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 7u]) * decode_state(exl3_code_at(packed, tile_index, 7u, local_column, WORD_COUNT, BITS), params.codebook);
        a0 += float(input[base + 8u]) * decode_state(exl3_code_at(packed, tile_index, 8u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 9u]) * decode_state(exl3_code_at(packed, tile_index, 9u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 10u]) * decode_state(exl3_code_at(packed, tile_index, 10u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 11u]) * decode_state(exl3_code_at(packed, tile_index, 11u, local_column, WORD_COUNT, BITS), params.codebook);
        a0 += float(input[base + 12u]) * decode_state(exl3_code_at(packed, tile_index, 12u, local_column, WORD_COUNT, BITS), params.codebook);
        a1 += float(input[base + 13u]) * decode_state(exl3_code_at(packed, tile_index, 13u, local_column, WORD_COUNT, BITS), params.codebook);
        a2 += float(input[base + 14u]) * decode_state(exl3_code_at(packed, tile_index, 14u, local_column, WORD_COUNT, BITS), params.codebook);
        a3 += float(input[base + 15u]) * decode_state(exl3_code_at(packed, tile_index, 15u, local_column, WORD_COUNT, BITS), params.codebook);
    }
    output[gid] = (a0 + a1) + (a2 + a3);
}










