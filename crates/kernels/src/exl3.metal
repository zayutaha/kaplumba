#include <metal_stdlib>
using namespace metal;

struct CodecParams {
    uint rate;
    uint codebook;
};

static uint read_word(const device ushort *packed, uint index) {
    return uint(packed[index * 2]) | (uint(packed[index * 2 + 1]) << 16);
}

// Extract the 16-bit code for state `state` from the trellis ring.
//
// Each 32-lane group owns eight codes, read out of a 64-bit sliding window that
// advances by one whole rate per code and overlaps by `16 - bits`. A plain
// `bits`-wide read at `state * bits` is wrong: the codes are not stored at their
// own offsets, which is what made the original decoder produce noise.
static uint integer_code_at(const device ushort *packed, uint state, uint bits) {
    uint total = 256u * bits;
    uint word_count = total / 32u;
    uint lane = state / 8u;
    uint i = state % 8u;
    // Window start: one whole rate past this lane's first code, wrapped so the
    // window straddles the end of the ring.
    uint lo = (lane * 8u + 1u) * bits + total - 16u;
    uint w0i = (lo / 32u) % word_count;
    uint w1i = (w0i + 1u) % word_count;
    uint w2i = (w1i + 1u) % word_count;
    uint shift = lo % 32u;
    uint w0 = read_word(packed, w0i);
    uint w1 = read_word(packed, w1i);
    uint w2 = read_word(packed, w2i);
    ulong a = (ulong(w0) << 32) | ulong(w1);
    a = (a << shift) | (shift != 0u ? ulong(w2 >> (32u - shift)) : 0ul);
    ulong c = ((ulong(w1) << 32) | ulong(w2)) << shift;
    uint r = i * bits + 16u;
    uint code = (r <= 64u) ? uint(a >> (64u - r)) : uint(c >> (96u - r));
    return code & 0xffffu;
}

// Inverse of the CPU scatter `out[k * 16 + n] = by_state[lane * 8 + i]`, so a
// thread indexed by tile position can find its own state.
static uint state_for_tile(uint k, uint n) {
    // rows = [r0, r0 + 1, r0 + 8, r0 + 9] with r0 = (lane % 4) * 2 in {0,2,4,6},
    // so r0 = k & 6 and the row within the group is k - r0. lane % 4 == r0 / 2,
    // and n carries lane / 4 in its low three bits and i / 4 above them.
    uint r0 = k & 6u;
    uint delta = k - r0;              // one of 0, 1, 8, 9
    uint low = delta < 8u ? delta : delta - 6u;  // -> i % 4 in 0..3
    uint i = (n / 8u) * 4u + low;
    uint lane = (n % 8u) * 4u + (r0 / 2u);
    return lane * 8u + i;
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
    int start = int(fractional_end(index, bits)) - 16;
    start = normalize_ring(start, depth * 16u);
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
        half low = as_type<half>((ushort)value);
        half high = as_type<half>((ushort)(value >> 16));
        return float(low + high);
    }
    if (codebook == 1u) {
        value = value * 0xCBAC1FEDu;
        value = 0x3b603b60u ^ (value & 0x8fff8fffu);
        half low = as_type<half>((ushort)value);
        half high = as_type<half>((ushort)(value >> 16));
        return float(low + high);
    }
    value = value * 0x83DCD12Du;
    uint sum = 0x6400u + (value & 0xffu) + ((value >> 8) & 0xffu) + ((value >> 16) & 0xffu) + ((value >> 24) & 0xffu);
    half byte_value = as_type<half>((ushort)sum);
    half inverse = as_type<half>((ushort)0x1eeeu);
    half bias = as_type<half>((ushort)0xc931u);
    return float(byte_value * inverse + bias);
}

// EXL3 tiles are stored row-major; see the note in linear.metal.
static uint permutation_index(uint row, uint mode) {
    return row;
}

kernel void exl3_decode_tile(
    const device ushort *packed [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant CodecParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= 256u) {
        return;
    }
    uint state;
    if (params.rate < 16u) {
        state = integer_code_at(packed, state_for_tile(gid / 16u, gid % 16u), params.rate);
    } else {
        state = fractional_window(packed, permutation_index(gid, 0u), params.rate - 16u);
    }
    output[gid] = decode_state(state, params.codebook);
}

kernel void hadamard128_vectors(
    const device half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &vector_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= vector_count) {
        return;
    }
    float values[128];
    uint base = gid * 128u;
    for (uint i = 0; i < 128u; ++i) {
        values[i] = float(input[base + i]);
    }
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
    float scale = 0.08838834764831845f;
    for (uint i = 0; i < 128u; ++i) {
        output[base + i] = half(values[i] * scale);
    }
}
