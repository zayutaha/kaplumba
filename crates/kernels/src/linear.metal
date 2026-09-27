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
    const device uint *tile = (const device uint *)(packed + tile_index * (word_count * 2u));
    ulong w0 = tile[j0];
    ulong w1 = tile[j1];
    ulong w2 = tile[j2];
    ulong a = ((w0 << 32) | w1) << sh | (sh ? (w2 >> (32 - sh)) : 0);
    ulong c = ((w1 << 32) | w2) << sh;
    uint r = i * ka + 16u;
    return (r <= 64u ? uint(a >> (64u - r)) : uint(c >> (96u - r))) & 0xffffu;
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

kernel void linear_gemv_tiled(
    const device ushort *packed [[buffer(0)]],
    const device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &params [[buffer(3)]],
    uint3 local_position [[thread_position_in_threadgroup]],
    uint3 grid_position [[thread_position_in_grid]]) {
    threadgroup half weights[16][16];
    uint local_query = (grid_position.y / 16u) * 16u + local_position.y;
    uint local_output = local_position.x;
    uint tile_column = grid_position.x / 16u;
    uint output_index = tile_column * 16u + local_output;
    uint local_output_in_tile = local_output;
    uint loader = local_position.x + local_position.y * 16u;
    for (uint tile_row = 0; tile_row < params.in_features / 16u; ++tile_row) {
        uint local_row = loader / 16u;
        uint local_column = loader & 15u;
        uint tile_index = tile_row * params.tile_columns + tile_column;
        uint state = packed_state(
            packed,
            tile_index,
            local_row,
            local_column,
            params.packed_words,
            params.rate,
            params.perm_mode
        );
        weights[local_row][local_column] = half(decode_state(state, params.codebook));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (local_query < params.queries) {
            uint query_base = local_query * params.in_features;
            uint output_base = local_query * params.out_features;
            for (uint local_input = 0; local_input < 16u; ++local_input) {
                uint input_index = tile_row * 16u + local_input;
                output[output_base + output_index] += float(input[query_base + input_index]) * float(weights[local_input][local_output_in_tile]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
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

