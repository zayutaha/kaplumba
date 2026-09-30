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

// The three codebooks, one function each, with NO branch on the codebook.
//
// `decode_state` below is the generic form and branches on a runtime `codebook`;
// it exists for the paths that still take the codebook as data (prefill staging,
// the non-specialised kernels, the reference kernels). The per-codebook functions
// exist so a kernel that knows its codebook at compile time can call one of these
// and have no `codebook == k` test in the eight-times-unrolled value loop at all.
//
// This split is the single largest win in the decode matvec. The measurement that
// motivated it, on the real 5376x5376 shape (medians of 15, 384 MB working set,
// exclusive GPU):
//
//   loads only (no decode)        212.2 GB/s
//   production, runtime codebook   62.4 GB/s
//   codebook folded to a constant 163.0 GB/s   (2.61x)
//   branch hoisted, still runtime 128.5 GB/s
//   no decode arithmetic at all    134.3 GB/s
//
// The last two lines are the important ones. Removing the decode *ops* on top of
// the specialization makes it SLOWER (134.3 < 163.0): the arithmetic is free and
// the BRANCH is what serialises the eight values per lane. So the fix is to
// specialise the branch away and leave the arithmetic alone; do not try to make
// the decode cheaper.
//
// Bit-exactness is non-negotiable here -- this is the one path in the engine
// with exactness, and the 2.16e-3 error seen elsewhere comes from fp16 operands,
// never from decode arithmetic. Each function is a transcription of one arm of
// the old `decode_state`, and a unit test compares all three against the engine's
// CPU reference (`decode_state` in crates/engine/src/exl3.rs) for every codebook
// and every bit rate, so a typo in a constant cannot survive.

static float decode_state_plain(uint value) {
    value = value * 89226354u + 64248484u;
    value = 0x3b603b60u ^ (value & 0x8fff8fffu);
    return float(as_type<half>((ushort)value) + as_type<half>((ushort)(value >> 16)));
}

static float decode_state_mcg(uint value) {
    value = value * 0xCBAC1FEDu;
    value = 0x3b603b60u ^ (value & 0x8fff8fffu);
    return float(as_type<half>((ushort)value) + as_type<half>((ushort)(value >> 16)));
}

static float decode_state_mul1(uint value) {
    value = value * 0x83DCD12Du;
    uint sum = 0x6400u + (value & 0xffu) + ((value >> 8) & 0xffu) + ((value >> 16) & 0xffu) + ((value >> 24) & 0xffu);
    half byte_value = as_type<half>((ushort)sum);
    half inverse = as_type<half>((ushort)0x1eeeu);
    half bias = as_type<half>((ushort)0xc931u);
    return float(half(float(byte_value) * float(inverse) + float(bias)));
}

// Generic, runtime-dispatched decode. Bit-identical to the three above; the only
// difference is that the choice happens per code instead of at compile time.
static float decode_state(uint state, uint codebook) {
    if (codebook == 0u) {
        return decode_state_plain(state);
    }
    if (codebook == 1u) {
        return decode_state_mcg(state);
    }
    return decode_state_mul1(state);
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

// The codebook-specialised twin of `exl3_decode8`.
//
// The body is a transcription of the body above with ONE difference: the decode
// call is `decode_state_mcg` instead of `decode_state(s, codebook)`, so the
// `codebook == 0u` / `codebook == 1u` tests -- which sit inside the eight-times
// unrolled value loop, once per decoded weight -- do not exist. Everything else
// is character-for-character the same, so the two are bit-identical by
// construction rather than by argument.
//
// Why the branch and not the arithmetic: on the real 5376x5376 shape (medians of
// 15, 384 MB working set, exclusive GPU)
//
//   loads only, no decode          212.2 GB/s
//   production, runtime codebook    62.4 GB/s
//   codebook folded to a constant  163.0 GB/s   (2.61x)
//   branch hoisted, still runtime  128.5 GB/s
//   no decode arithmetic at all    134.3 GB/s
//
// The last two lines are the point. Removing the decode OPS on top of the
// specialization makes it SLOWER (134.3 < 163.0): the arithmetic is free and the
// BRANCH is what serialises the eight values per lane. So specialise the branch
// away and leave the arithmetic alone.
//
// Two traps this shape is deliberately built to avoid:
//
//  1. Do NOT hold the decoded values in a `uint code[8]` local array written in
//     one `#pragma unroll` loop and read in another. That MISCOMPILED and gave
//     32762/32768 wrong outputs while the harness reported success. `v[]` is
//     thread-local and written in exactly one loop, as before.
//  2. Do NOT let the accumulation re-associate. `acc += x0*v0 + x1*v1 + ...` is
//     an FMA contraction; two variants can each match a baseline alone and
//     differ from each other when combined (30818/32768 differed). The caller's
//     accumulation is left untouched for exactly this reason -- the ONLY change
//     in this file is which decode function is called.
template <bool ALIGNED>
static inline void exl3_decode8_mcg(
    const device ushort *packed,
    uint tile_index,
    uint lane,
    uint ka,
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
    #pragma unroll
    for (uint i = 0u; i < 8u; ++i) {
        const uint r = i * ka + 16u;
        const uint s = (r <= 64u) ? uint(a >> (64u - r)) : uint(c >> (96u - r));
        v[i] = decode_state_mcg(s & 0xffffu);
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

// The same butterfly, one 32-lane simdgroup per 128-element block.
//
// Element `sub*32 + lane` lives in sub-register `sub` of lane `lane`, so the
// five low passes (`width` 1..16) are a `simd_shuffle_xor` away and the two
// high passes (`width` 32 and 64) are register swaps inside the lane -- no
// threadgroup memory, and four floats of registers per thread instead of the
// 128-float array `transform_vector` needs.
//
// Bit-exact with `transform_vector`, pass for pass:
//   * the width loop runs 1, 2, 4, ... 64 in the same order, and each pass reads
//     both partners before writing either, so every add/sub sees the same two
//     values it saw before -- no reassociation;
//   * `width 32` pairs sub-registers 0/1 and 2/3 and `width 64` pairs 0/2 and
//     1/3, which is exactly `i` with bit 5 resp. bit 6 cleared;
//   * a lane holding the *upper* partner must produce `left - right`, so it
//     computes `theirs - mine`, not `mine - theirs`. Those two differ: IEEE-754
//     subtraction is antisymmetric (`a - b` is the exact negation of `b - a`),
//     and half of the output came out sign-flipped when it got that backwards.
//     The extra operand swap is what makes the SIMD form bit-identical rather
//     than merely equal in exact arithmetic.
static inline void transform_simd(thread float &v0, thread float &v1, thread float &v2, thread float &v3, uint lane) {
    #pragma unroll
    for (uint width = 1u; width < 32u; width <<= 1u) {
        const ushort mask = ushort(width);
        float p0 = simd_shuffle_xor(v0, mask);
        float p1 = simd_shuffle_xor(v1, mask);
        float p2 = simd_shuffle_xor(v2, mask);
        float p3 = simd_shuffle_xor(v3, mask);
        const bool upper = (lane & width) != 0u;
        v0 = upper ? p0 - v0 : v0 + p0;
        v1 = upper ? p1 - v1 : v1 + p1;
        v2 = upper ? p2 - v2 : v2 + p2;
        v3 = upper ? p3 - v3 : v3 + p3;
    }
    // width 32: pairs (0,1) and (2,3). Every pair reads both of its inputs
    // before either is written, which is what keeps the pass order irrelevant to
    // the result.
    float t0 = v0;
    float t1 = v1;
    float t2 = v2;
    float t3 = v3;
    v0 = t0 + t1;
    v1 = t0 - t1;
    v2 = t2 + t3;
    v3 = t2 - t3;
    // width 64: pairs (0,2) and (1,3).
    float s0 = v0;
    float s1 = v1;
    v0 = s0 + v2;
    v2 = s0 - v2;
    v1 = s1 + v3;
    v3 = s1 - v3;
    // 1 / sqrt(128), the same constant `transform_vector` scales by, last.
    const float SCALE = 0.08838834764831845f;
    v0 *= SCALE;
    v1 *= SCALE;
    v2 *= SCALE;
    v3 *= SCALE;
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

// EXL3 input transform for decode.
//
// One 128-element block per thread, as before, but the block is transformed by a
// 32-lane simdgroup (four registers per lane) instead of by one thread holding
// 128 floats. At in_features = 5376 this dispatches 42 blocks against a 20-core
// GPU, and each thread had to keep a `thread float values[128]` array alive
// across seven runtime-bounded butterfly passes -- local memory.
//
// `blocks` is `in_features / 128` and the dispatch is `blocks * 32` threads in
// groups of 32 (`LinearKernels::encode_input_transform` is the only caller), so
// every threadgroup is exactly one simdgroup and the guard below can never split
// one. That matters: `simd_shuffle_xor` is undefined if part of the simdgroup is
// inactive, and a 64-thread group with 42 active blocks has an inactive lane in
// its second simdgroup. Under that dispatch this kernel writes *nothing*.
//
// Block index and lane both come from `gid`, so nothing depends on
// `thread_index_in_simdgroup` agreeing with `gid % 32`.
kernel void linear_input_transform(
    const device half *input [[buffer(0)]],
    const device half *input_signs [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &blocks [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= blocks * 32u) {
        return;
    }
    // `input` and `input_signs` are both `in_features` long and share this
    // indexing exactly: block `gid >> 5` reads [base, base + 128).
    const uint lane = gid & 31u;
    const uint base = (gid >> 5) * 128u + lane;
    float v0 = float(input[base]) * float(input_signs[base]);
    float v1 = float(input[base + 32u]) * float(input_signs[base + 32u]);
    float v2 = float(input[base + 64u]) * float(input_signs[base + 64u]);
    float v3 = float(input[base + 96u]) * float(input_signs[base + 96u]);
    transform_simd(v0, v1, v2, v3, lane);
    output[base] = half(v0);
    output[base + 32u] = half(v1);
    output[base + 64u] = half(v2);
    output[base + 96u] = half(v3);
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
// per eight weights.
//
// A lane's eight codes land on rows {r0, r0+1, r0+8, r0+9} and columns {c, c+8},
// so four x values feed all eight, each used twice. The reduction is
// simd_shuffle_xor over four lanes then a threadgroup sum over the eight
// simdgroups. No atomics.
//
// Three axes are specialised, because all three are worth real throughput and
// Metal has no templates for any of them:
//   * KA (1..8), so shift amounts and the ring size fold to constants.
//   * ALIGNED, so a 4-byte-aligned buffer is read with one 32-bit load per word
//     instead of two 16-bit loads. Arena-resident weights are always aligned;
//     only mapped checkpoint bytes are not. The two-16-bit-load form doubles the
//     load instruction count on the hottest path in the engine, and loads -- not
//     the codebook -- dominate: measured 0.336 ms of 0.612 ms for q_proj.
//   * CODEBOOK, only for MCG. The codebook is a per-tensor constant and
//     branching on it eight times per lane cost 61.7% of the kernel; see
//     `exl3_decode8_mcg` for the measurement and for the reason it is the BRANCH
//     and not the arithmetic. `kN`/`kN_u` remain as the runtime-codebook
//     fallback for the plain and mul1 codebooks.
//
// Variants are generated by `scripts/gen_exl3_gemv.py`; do not hand-edit.
//
// GENERATED by scripts/gen_exl3_gemv.py -- do not hand-edit below this line.


kernel void linear_gemv_ishizuki_k1(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant const uint *shape [[buffer(3)]],
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
    constant const uint *shape [[buffer(3)]],
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
    constant const uint *shape [[buffer(3)]],
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
    constant const uint *shape [[buffer(3)]],
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
    constant const uint *shape [[buffer(3)]],
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
    constant const uint *shape [[buffer(3)]],
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
    constant const uint *shape [[buffer(3)]],
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
    constant const uint *shape [[buffer(3)]],
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, params.codebook, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8<false>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, params.codebook, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, params.codebook, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8<false>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, params.codebook, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, params.codebook, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8<false>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, params.codebook, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, params.codebook, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8<false>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, params.codebook, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, params.codebook, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8<false>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, params.codebook, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, params.codebook, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8<false>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, params.codebook, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, params.codebook, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8<false>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, params.codebook, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, params.codebook, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8<false>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, params.codebook, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k1_mcg(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k1_u_mcg(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<false>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k2_mcg(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k2_u_mcg(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<false>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k3_mcg(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k3_u_mcg(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<false>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k4_mcg(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k4_u_mcg(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<false>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k5_mcg(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k5_u_mcg(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<false>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k6_mcg(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k6_u_mcg(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<false>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k7_mcg(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k7_u_mcg(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<false>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k8_mcg(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k8_u_mcg(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 1;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<false>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k1_mcg_r2(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 2;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k1_mcg_r4(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 4;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k1_mcg_r8(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 8;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k2_mcg_r2(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 2;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k2_mcg_r4(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 4;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k2_mcg_r8(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 8;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k3_mcg_r2(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 2;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k3_mcg_r4(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 4;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k3_mcg_r8(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 8;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k4_mcg_r2(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 2;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k4_mcg_r4(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 4;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k4_mcg_r8(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 8;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k5_mcg_r2(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 2;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k5_mcg_r4(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 4;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k5_mcg_r8(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 8;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k6_mcg_r2(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 2;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k6_mcg_r4(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 4;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k6_mcg_r8(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 8;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k7_mcg_r2(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 2;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k7_mcg_r4(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 4;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k7_mcg_r8(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 8;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k8_mcg_r2(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 2;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k8_mcg_r4(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 4;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
        float s = 0.0f;
        for (uint g = 0u; g < simdgroups; ++g) {
            s += red[g][lid];
        }
        output[out_base + lid] = s;
    }
}


kernel void linear_gemv_exl3_k8_mcg_r8(
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
    // NR0 output tile-columns per threadgroup: how many independent weight
    // streams one thread keeps in flight. NR0 = 1 is exactly the previous
    // kernel's shape. The tk order, the accumulator update, the four-lane
    // butterfly and the serial threadgroup sum are all untouched, so every
    // output column sees an identical sequence of fp32 adds and NR0 = 1 is
    // bit-identical to the kernel it replaces by construction. The test still
    // has to check that, because FMA contraction is not stable across
    // recompiles: two variants can each match the baseline alone and differ when
    // combined.
    //
    // Why the axis exists, measured with `examples/gemv_shape.rs` holding the
    // weight bytes constant at 43.4 MB and varying only the rate:
    //
    //   3 bpw 124.6 GB/s   4 bpw 175.4   6 bpw 178.9   8 bpw 156.1
    //
    // Identical bytes, and the rate is the only thing that moved.
    // `exl3_decode8` reads a fixed three ring words per lane whatever KA is, so
    // a lane spends 12 B of load traffic to produce 8*KA bits of codes: three
    // useful bytes at KA = 3, six at KA = 6. The loop is therefore short of
    // distinct bytes in flight rather than short of DRAM, and 3 bpw is 410 of
    // the 411 tensors in this checkpoint. The measured 3 bpw asymptote is
    // 168.6 GB/s against a 240 GB/s ceiling, so the prize is ~1.4x -- and NR0 is
    // the only lever left that adds streams without changing what is loaded.
    //
    // The NR0 streams are CONTIGUOUS tile columns (`tn*NR0 + q`), i.e. 96*NR0
    // adjacent bytes per thread per iteration. The llama.cpp experiment this
    // axis is borrowed from found *adjacent* rows lost 4-11% and rows 32 KiB
    // apart won, but that is a row-major layout where adjacent rows share DRAM
    // pages; here adjacent tile columns are simply adjacent bytes, and DRAM row
    // locality was measured at zero effect from 4 KiB to 3 MiB. Contiguous is
    // therefore the better bet -- and it is a bet, not a measurement, which is
    // why the r-variants are all generated and all measurable.
    const uint NR0 = 8;
    threadgroup float red[8][16 * NR0];
    const uint lane = lid % 32u;
    const uint r0 = (lane % 4u) * 2u;
    const uint tiles_n = params.out_features / 16u;
    const uint tiles_k = params.in_features / 16u;
    const uint simdgroups = tgdim / 32u;
    const uint out_base = tn * (16u * NR0);

    float acc[NR0][2];
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] = 0.0f;
        acc[q][1] = 0.0f;
    }

    for (uint tk = sg; tk < tiles_k; tk += simdgroups) {
        // All NR0 window setups first, so the loads for every stream are in
        // flight before the first FMA consumes any of them. That is the entire
        // point of the axis: at NR0 = 1 the single stream's FMAs sit between its
        // load and its use, and at NR0 = 2 the load for q = 1 is issued before
        // the FMAs for q = 0 rather than after them.
        float v[NR0][8];
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            exl3_decode8_mcg<true>(packed, tk * tiles_n + tn * NR0 + q, lane, KA, v[q]);
        }
        const uint xbase = tk * 16u;
        const float x0 = float(input[xbase + r0 + 0u]);
        const float x1 = float(input[xbase + r0 + 1u]);
        const float x2 = float(input[xbase + r0 + 8u]);
        const float x3 = float(input[xbase + r0 + 9u]);
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }
    }
    #pragma unroll
    for (uint q = 0u; q < NR0; ++q) {
        acc[q][0] += simd_shuffle_xor(acc[q][0], 1u);
        acc[q][0] += simd_shuffle_xor(acc[q][0], 2u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 1u);
        acc[q][1] += simd_shuffle_xor(acc[q][1], 2u);
    }
    if (lane % 4u == 0u) {
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            red[sg][q * 16u + lane / 4u] = acc[q][0];
            red[sg][q * 16u + lane / 4u + 8u] = acc[q][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16u * NR0) {
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
//  8. The y-grid is `ceil(queries / 128)` and NOTHING else. A prompt's tile
//     count is therefore a function of its TOKEN count, not of how the host
//     chose to slice it: 614 tokens is five 128-row tiles whether they arrive
//     as 256+256+102 or as one 614-row dispatch, and the partial last tile
//     costs a full tile because the MMAs are unconditional and only the store
//     is row-guarded. What the host's chunk size does control is how many
//     DISPATCHES those tiles arrive in, and that is worth 6.4% of GEMM on this
//     box: measured per 128-row tile at 614 tokens, one dispatch per tile costs
//     1354.0 ms, two per dispatch 1271.7, four 1222.8, five 1204.1 -- a small
//     grid does not fill 20 GPU cores. Hence PREFILL_CHUNK = 1024 rather than
//     256 (crates/engine/src/model.rs). Do not read "the tail chunk is
//     partially occupied" as a defect to fix with a smaller tile: at 128-row
//     granularity the waste is `(-tokens) mod 128` and cannot be removed, and
//     sub-128-row tiles cannot use the 8x8 MMA shape at all.
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
        } else if (ka == 3u && params.codebook == 1u) {
            // 7. `params.rate` and `params.codebook` are runtime values, and the
            //    body below calls `decode_state` sixteen times per thread per
            //    chunk, so the generic form carries a THREE-WAY CODEBOOK BRANCH
            //    per weight -- 16 branch chains per thread per chunk, on a body
            //    whose real work is ~12 instructions per weight -- plus a
            //    per-weight select between the two overlapping 64-bit windows,
            //    which cannot be resolved because `r = (j + 4*dc)*rate + 16` is
            //    runtime. Both fold away for the rate and codebook the
            //    checkpoint uses (EXL3 3.0 bpw, MCG): `decode_state` collapses
            //    from a branch chain to four instructions, and with `ka == 3`
            //    as a literal `r` is 16..37, so `r <= 64` is ALWAYS TRUE, the
            //    second window `wc` is dead, and the select with it.
            //
            //    This is dequant work, not loop bookkeeping, and it was worth
            //    measuring rather than assuming: deleting the sixteen MMAs from
            //    this kernel leaves 0.33 ms of 23.4 ms, so ~99% of the runtime
            //    is the tensor pipe, and the dequant's 3.0 ms marginal is the
            //    only non-MMA budget there is. Interleaved A/B on the shipped
            //    source (gate_proj 5376x21504, 614 queries, paired A/B/B/A,
            //    median of 6): 23.4 -> 21.9 ms, 1.044x, BIT-IDENTICAL output.
            //    End to end, 68.0 -> 70.6 tok/s (n=4 each, non-overlapping
            //    ranges). Same window, same 16-bit slices, same MCG hash, same
            //    fp16 rounding, so per-element accumulation order and the
            //    weight rounding are both untouched and the worst GEMM error
            //    stays at 2.1603336e-3.
            //
            //    Rates 1,2,4..8 and the Plain/Mul1 codebooks keep the generic
            //    path below, which is unchanged and still correct for them.
            #pragma unroll
            for (uint jw = 0u; jw < 2u; ++jw) {
                const uint w = tid + jw * THREADS;
                const uint t = w / 32u;
                const uint l = w & 31u;
                const uint r0 = (l & 3u) * 2u;
                const uint c0 = l >> 2;
                const uint lo = (l * 8u + 1u) * 3u + 256u * 3u - 16u;
                const uint sh = lo % 32u;
                const ulong g0 = pre[3u * jw + 0u];
                const ulong g1 = pre[3u * jw + 1u];
                const ulong g2 = pre[3u * jw + 2u];
                const ulong wa = ((g0 << 32) | g1) << sh | (sh ? (g2 >> (32 - sh)) : 0ul);
                #pragma unroll
                for (uint j = 0u; j < 4u; ++j) {
                    const uint rr = (t / NTILES) * 16u + r0 + (j & 1u) + ((j >> 1u) << 3u);
                    #pragma unroll
                    for (uint dc = 0u; dc < 2u; ++dc) {
                        const uint r = (j + 4u * dc) * 3u + 16u;
                        const uint s = uint(wa >> (64u - r)) & 0xffffu;
                        uint v = s * 0xCBAC1FEDu;
                        v = 0x3b603b60u ^ (v & 0x8fff8fffu);
                        bstage[rr * BSTRIDE + (t % NTILES) * 16u + c0 + dc * 8u] =
                            half(float(as_type<half>((ushort)v) +
                                       as_type<half>((ushort)(v >> 16))));
                    }
                }
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

// ---------------------------------------------------------------------------
// RESEARCH VARIANT `linear_gemm_mma_regpipe`: the same 128x64 tile, the same
// sixteen accumulators, the same accumulation order and the same bit-identical
// MCG decode as `linear_gemm_mma`, with ONE structural change -- where the
// dequantised weights live while the MMAs run.
//
// `linear_gemm_mma` alternates phases inside one threadgroup:
//   dequant -> threadgroup_barrier (2567) -> MMA -> threadgroup_barrier (2625)
// Both barriers are load-bearing for the single 9216 B `bstage`, so the ALU
// phase and the tensor phase never coexist in time, and a second buffer cannot
// fix that inside a threadgroup because `simdgroup_multiply_accumulate` is a
// 32-lane collective: all 8 simdgroups must be in the MMA together.
//
// Registers break the tie because they are disjoint from `bstage`. Each thread
// decodes its OWN 16 weights per chunk (2 jw x 4 j x 2 dc), so those 16 halfs
// fit in 16 registers. Writing them to `nxt[]` instead of `bstage` means:
//   publish nxt -> bstage; barrier; prefetch; { MMAs; dequant into nxt }
// There is exactly ONE barrier per chunk instead of two, `bstage` is unchanged
// at 64x72 halfs = 9216 B (no occupancy cost), and the dequant for chunk N+1 is
// in the same basic block as the MMAs for chunk N with no memory dependency
// between them -- which is the only arrangement in which the ALU pipe and the
// tensor pipe can actually be busy at the same time.
//
// Host-gated to `rate == Integer(3) && codebook == Mcg` (the folded MCG arm),
// so every other rate keeps the original kernel.
// ---------------------------------------------------------------------------
kernel void linear_gemm_mma_regpipe(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    constexpr uint KCHUNK = 64u;
    constexpr uint NTILES = 4u;
    constexpr uint QTILE = 128u;
    constexpr uint OUTS = 64u;
    constexpr uint BSTRIDE = OUTS + 8u;
    constexpr uint SGROUPS = 8u;
    constexpr uint ROWS_SG = QTILE / SGROUPS;
    const uint sgid = tid / 32u;

    const uint ka = params.rate; // 3: host-gated
    const uint word_count = 256u * ka / 32u;

    threadgroup half bstage[KCHUNK * BSTRIDE];

    const uint rbase = group.y * QTILE + sgid * ROWS_SG;
    const device half *ab0 = input + min(rbase, params.queries - 1u) * params.in_features;
    const device half *ab1 = input + min(rbase + 8u, params.queries - 1u) * params.in_features;
    const uint lane = tid & 31u;
    const uint lane_fm = (lane >> 2 & 4u) + (lane >> 1 & 3u);
    const uint lane_fn = (lane >> 2 & 2u) << 1 | (lane & 1u) << 1;

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

    // This thread's dequant slot, loop-invariant: which 16x16 weight tile, which
    // trellis lane, and therefore which four rows and two columns it owns.
    const uint t0 = tid / 32u;
    const uint l0 = tid & 31u;
    const uint rr0 = (l0 & 3u) * 2u;
    const uint cc0 = l0 >> 2;
    const uint lo0 = (l0 * 8u + 1u) * 3u + 256u * 3u - 16u;
    const uint i00 = lo0 / 32u;
    const uint sh0 = lo0 % 32u;
    const uint col0 = (t0 % NTILES) * 16u;
    const uint row0 = (t0 / NTILES) * 16u;
    // k-tile of the chunk currently being decoded (clamped, see the loop).
    const uint ktiles = params.in_features / 16u;

    thread uint pre[6];
    half nxt[16];

    // Trellis window words for a k-tile index, into `pre`.
#define KAPLUMBA_REGPIPE_PREFETCH(kt)                                             \
    {                                                                             \
        _Pragma("unroll") for (uint jw = 0u; jw < 2u; ++jw) {                      \
            const uint t = (jw == 0u) ? t0 : (t0 + SGROUPS);                       \
            const uint j0 = i00 >= word_count ? i00 - word_count : i00;            \
            const uint j1 = j0 + 1u >= word_count ? j0 + 1u - word_count : j0 + 1u; \
            const uint j2 = j1 + 1u >= word_count ? j1 + 1u - word_count : j1 + 1u; \
            const uint gbase =                                                     \
                2u * (((kt) + t / NTILES) * params.tile_columns +                  \
                     (group.x * NTILES + (t % NTILES))) *                           \
                word_count;                                                        \
            pre[3u * jw + 0u] = trellis_word(packed + gbase, j0);                   \
            pre[3u * jw + 1u] = trellis_word(packed + gbase, j1);                   \
            pre[3u * jw + 2u] = trellis_word(packed + gbase, j2);                   \
        }                                                                          \
    }

    // The MCG decode of one weight, identical to `linear_gemm_mma`: same window,
    // same 16-bit slice, same hash, same fp16 rounding of the pair sum. `r` is a
    // compile-time literal here exactly as it is there, which is what folds
    // `r <= 64` and the second window away.
#define KAPLUMBA_REGPIPE_DECODE(wa, j, dc, slot)                                  \
    {                                                                             \
        const uint r = ((j) + 4u * (dc)) * 3u + 16u;                               \
        const uint s = uint((wa) >> (64u - r)) & 0xffffu;                          \
        uint v = s * 0xCBAC1FEDu;                                                  \
        v = 0x3b603b60u ^ (v & 0x8fff8fffu);                                       \
        (slot) = half(float(as_type<half>((ushort)v) +                             \
                            as_type<half>((ushort)(v >> 16))));                     \
    }

    KAPLUMBA_REGPIPE_PREFETCH(0u)
    #pragma unroll
    for (uint jw = 0u; jw < 2u; ++jw) {
        const uint g0 = pre[3u * jw + 0u];
        const uint g1 = pre[3u * jw + 1u];
        const uint g2 = pre[3u * jw + 2u];
        const ulong wa = ((ulong(g0) << 32) | g1) << sh0 |
                         (sh0 ? (ulong(g2) >> (32 - sh0)) : 0ul);
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) {
            KAPLUMBA_REGPIPE_DECODE(wa, j, 0u, nxt[jw * 8u + j * 2u + 0u])
            KAPLUMBA_REGPIPE_DECODE(wa, j, 1u, nxt[jw * 8u + j * 2u + 1u])
        }
    }

    for (uint kbase = 0; kbase < params.in_features; kbase += KCHUNK) {
        // Publish this chunk's weights. This is the ONLY barrier per chunk, and
        // it is the same publish-then-MMA handoff the original had, so a reader
        // of `bstage` still sees a complete tile.
        #pragma unroll
        for (uint jw = 0u; jw < 2u; ++jw) {
            const uint t = (jw == 0u) ? t0 : (t0 + SGROUPS);
            const uint tcol = (t % NTILES) * 16u;
            const uint trow = (t / NTILES) * 16u;
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) {
                const uint rr = trow + rr0 + (j & 1u) + ((j >> 1u) << 3u);
                bstage[rr * BSTRIDE + tcol + cc0 + 0u] = nxt[jw * 8u + j * 2u + 0u];
                bstage[rr * BSTRIDE + tcol + cc0 + 8u] = nxt[jw * 8u + j * 2u + 1u];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Next chunk's window words, in flight across the whole MMA block below.
        const uint knext = min(kbase + KCHUNK, params.in_features - KCHUNK);
        KAPLUMBA_REGPIPE_PREFETCH((knext / 16u))

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

            // The same chunk's next-step decode, in the same basic block as the
            // MMAs and touching only `nxt` and `pre`. Two of this thread's 16
            // weights per step, eight steps, sixteen weights -- the identical
            // work the original performed before its first barrier.
            const uint jw = step >> 2u;
            const uint j = step & 3u;
            const uint g0 = pre[3u * jw + 0u];
            const uint g1 = pre[3u * jw + 1u];
            const uint g2 = pre[3u * jw + 2u];
            const ulong wa = ((ulong(g0) << 32) | g1) << sh0 |
                             (sh0 ? (ulong(g2) >> (32 - sh0)) : 0ul);
            KAPLUMBA_REGPIPE_DECODE(wa, j, 0u, nxt[jw * 8u + j * 2u + 0u])
            KAPLUMBA_REGPIPE_DECODE(wa, j, 1u, nxt[jw * 8u + j * 2u + 1u])
        }
        // The closing barrier of `linear_gemm_mma:2625` is load-bearing and is
        // NOT removable: a single `bstage` means the next iteration's publish
        // must not begin while any simdgroup is still reading this one. So V5
        // carries the same two barriers as the original; what it changes is that
        // the dequant sits BETWEEN them, in the MMAs' basic block, instead of
        // before the first one.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#undef KAPLUMBA_REGPIPE_DECODE

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










