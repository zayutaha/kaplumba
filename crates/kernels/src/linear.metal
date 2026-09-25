#include <metal_stdlib>
using namespace metal;

struct LinearParams {
    uint in_features;
    uint out_features;
    uint tile_columns;
    uint packed_words;
    uint rate;
    uint codebook;
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

static uint permutation_index(uint row) {
    uint t = row / 8u;
    uint j = row & 7u;
    uint r0 = (t & 3u) * 2u;
    uint r = r0 + ((j & 1u) != 0u ? 1u : 0u) + ((j & 2u) != 0u ? 8u : 0u);
    uint c0 = t >> 2u;
    uint c = c0 + ((j & 4u) != 0u ? 8u : 0u);
    return r * 16u + c;
}

static uint packed_state(const device ushort *packed, uint tile_index, uint local_row, uint local_column, uint words, uint rate) {
    uint lane = permutation_index(local_row * 16u + local_column);
    const device ushort *tile = packed + tile_index * words;
    if (rate < 16u) {
        return integer_window(tile, lane, rate);
    }
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
        uint state = packed_state(packed, tile_index, local_row, local_column, params.packed_words, params.rate);
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
