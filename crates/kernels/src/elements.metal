#include <metal_stdlib>
using namespace metal;

struct EmbeddingParams {
    uint tokens;
    uint hidden;
    uint vocab;
    float scale;
    // Byte offset of the embedding rows inside the bound buffer. Non-zero when
    // the table is offloaded: the view is page-aligned *downward*, so the rows do
    // not begin at its first byte. Expressed in bytes and always odd or even as
    // the file dictates: `embed_tokens` begins at shard offset 133127, giving a
    // window offset of 2055, which is not even 2-byte aligned. Dividing by two to
    // get an element index silently truncates and reads garbage.
    uint offset;
    // When set, the table is BF16 as stored in the checkpoint and must be
    // widened to fp16 here. Doing the conversion per element lets the 2.82 GB
    // table stay mapped instead of being copied into the arena, because only one
    // row per token is ever read.
    uint bf16;
};

struct NormParams {
    uint rows;
    uint width;
    float eps;
};

struct RopeParams {
    uint tokens;
    uint heads;
    uint head_dim;
    uint offset;
    float theta;
    float partial_factor;
    uint proportional;
};

struct ScalarParams {
    uint count;
    float scale;
};

struct SoftcapParams {
    uint count;
    float softcap;
};

static float gelu_tanh_value(float value) {
    constexpr float coefficient = 0.7978845608028654f;
    float cubic = value * value * value;
    float argument = coefficient * (value + 0.044715f * cubic);
    argument = clamp(argument, -20.0f, 20.0f);
    return 0.5f * value * (1.0f + tanh(argument));
}

// BF16 -> FP16. A bf16 is the top half of an fp32, so widening is exact and the
// narrowing back to fp16 rounds exactly as a load-time conversion would.
static inline half bf16_to_half(ushort raw) {
    return half(as_type<float>(uint(raw) << 16));
}

kernel void embedding_scaled(
    const device uint *tokens_data [[buffer(0)]],
    const device half *embedding [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant EmbeddingParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.tokens * params.hidden) {
        return;
    }
    uint token = gid / params.hidden;
    uint column = gid % params.hidden;
    uint token_id = tokens_data[token];
    if (token_id >= params.vocab) {
        return;
    }
    float value;
    if (params.bf16 != 0u) {
        // Offloaded BF16: read the raw bytes so an odd window offset is safe.
        const device uchar *raw = (const device uchar *)embedding;
        uint at = params.offset + (token_id * params.hidden + column) * 2u;
        ushort bits = ushort(raw[at]) | (ushort(raw[at + 1u]) << 8);
        value = float(bf16_to_half(bits));
    } else {
        // Arena copy: already FP16 at a 2-byte-aligned base, offset in elements.
        value = float(embedding[params.offset / 2u + token_id * params.hidden + column]);
    }
    output[gid] = half(value * params.scale);
}

// Parallel RMS reduction, one threadgroup per row.
//
// The kernels below are written "one thread per row", which is correct for
// prefill (many rows) and catastrophic for decode: at `rows == 1` a single GPU
// thread walks 5376 elements twice, serially. Measured 595.7 us against 2.4-3.1
// us for sibling kernels of the same or larger shape -- 200x slower -- and 241
// such dispatches per token cost 156 ms of a 776 ms step.
//
// This variant assigns one 256-thread group per row so the reduction is a
// strided partial sum plus a barrier tree. Only used when `rows` is small; see
// `encode_rms_norm` for the dispatch rule that keeps prefill on the old path.
kernel void rms_norm_weighted_tg(
    const device half *input [[buffer(0)]],
    const device half *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant NormParams &params [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tgdim [[threads_per_threadgroup]]) {
    if (tgid >= params.rows) {
        return;
    }
    threadgroup float partial[256];
    const uint count = params.width;
    const uint base = tgid * count;
    float sum = 0.0f;
    for (uint i = lid; i < count; i += tgdim) {
        const float value = float(input[base + i]);
        sum += value * value;
    }
    partial[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgdim / 2u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            partial[lid] += partial[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inverse = rsqrt(partial[0] / float(count) + params.eps);
    for (uint i = lid; i < count; i += tgdim) {
        output[base + i] = half(float(input[base + i]) * inverse * float(weight[i]));
    }
}

kernel void rms_norm_unweighted_tg(
    const device half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant NormParams &params [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tgdim [[threads_per_threadgroup]]) {
    if (tgid >= params.rows) {
        return;
    }
    threadgroup float partial[256];
    const uint count = params.width;
    const uint base = tgid * count;
    float sum = 0.0f;
    for (uint i = lid; i < count; i += tgdim) {
        const float value = float(input[base + i]);
        sum += value * value;
    }
    partial[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgdim / 2u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            partial[lid] += partial[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inverse = rsqrt(partial[0] / float(count) + params.eps);
    for (uint i = lid; i < count; i += tgdim) {
        output[base + i] = half(float(input[base + i]) * inverse);
    }
}

kernel void rms_norm_weighted(
    const device half *input [[buffer(0)]],
    const device half *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant NormParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.rows) {
        return;
    }
    uint count = params.width;
    uint base = gid * count;
    float sum = 0.0f;
    for (uint i = 0; i < count; ++i) {
        float value = float(input[base + i]);
        sum += value * value;
    }
    float inverse = rsqrt(sum / float(count) + params.eps);
    for (uint i = 0; i < count; ++i) {
        float value = float(input[base + i]) * inverse * float(weight[i]);
        output[base + i] = half(value);
    }
}

kernel void rms_norm_unweighted(
    const device half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant NormParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.rows) {
        return;
    }
    uint count = params.width;
    uint base = gid * count;
    float sum = 0.0f;
    for (uint i = 0; i < count; ++i) {
        float value = float(input[base + i]);
        sum += value * value;
    }
    float inverse = rsqrt(sum / float(count) + params.eps);
    for (uint i = 0; i < count; ++i) {
        output[base + i] = half(float(input[base + i]) * inverse);
    }
}

kernel void residual_add(
    const device half *left [[buffer(0)]],
    const device half *right [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) {
        return;
    }
    output[gid] = half(float(left[gid]) + float(right[gid]));
}

kernel void layer_scale(
    const device half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant ScalarParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) {
        return;
    }
    output[gid] = half(float(input[gid]) * params.scale);
}

kernel void gelu_tanh(
    const device half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) {
        return;
    }
    output[gid] = half(gelu_tanh_value(float(input[gid])));
}

kernel void gated_gelu(
    const device half *gate [[buffer(0)]],
    const device half *up [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) {
        return;
    }
    output[gid] = half(gelu_tanh_value(float(gate[gid])) * float(up[gid]));
}

kernel void rope(
    const device half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant RopeParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    uint vector_width = params.heads * params.head_dim;
    if (gid >= params.tokens * vector_width) {
        return;
    }
    uint token = gid / vector_width;
    uint within = gid % vector_width;
    uint column = within % params.head_dim;
    uint half_width = params.head_dim / 2u;
    uint pair = column < half_width ? column : column - half_width;
    uint rotated_pairs = params.proportional != 0u
        ? uint(params.partial_factor * float(params.head_dim) / 2.0f)
        : half_width;
    if (pair >= rotated_pairs) {
        output[gid] = input[gid];
        return;
    }
    uint position = params.offset + token;
    float exponent = 2.0f * float(pair) / float(params.head_dim);
    float angle = float(position) / pow(params.theta, exponent);
    float cosine = cos(angle);
    float sine = sin(angle);
    uint base = gid - column;
    float value = float(input[base + column]);
    if (column < half_width) {
        output[gid] = half(value * cosine - float(input[base + half_width + pair]) * sine);
    } else {
        output[gid] = half(value * cosine + float(input[base + pair]) * sine);
    }
}

kernel void softcap(
    const device half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant SoftcapParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) {
        return;
    }
    output[gid] = half(tanh(float(input[gid]) / params.softcap) * params.softcap);
}

struct CacheAppendParams {
    uint tokens;
    uint stream_len;
    uint capacity;
    uint slot_start;
    uint position_start;
};

kernel void cache_append(
    const device half *keys [[buffer(0)]],
    const device half *values [[buffer(1)]],
    device half *key_cache [[buffer(2)]],
    device half *value_cache [[buffer(3)]],
    device uint *position_cache [[buffer(4)]],
    constant CacheAppendParams &params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.tokens * params.stream_len) {
        return;
    }
    uint token = gid / params.stream_len;
    uint element = gid % params.stream_len;
    // Sliding layers ring, so the destination slot wraps; full layers have a
    // capacity larger than any position we reach and never wrap.
    uint slot = (params.slot_start + token) % params.capacity;
    key_cache[slot * params.stream_len + element] = keys[gid];
    value_cache[slot * params.stream_len + element] = values[gid];
    if (element == 0) {
        position_cache[slot] = params.position_start + token;
    }
}

kernel void embedding_scaled_buffer(
    const device uint *tokens_data [[buffer(0)]],
    const device half *embedding [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant EmbeddingParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.tokens * params.hidden) {
        return;
    }
    uint token = gid / params.hidden;
    uint column = gid % params.hidden;
    uint token_id = tokens_data[token];
    if (token_id >= params.vocab) {
        return;
    }
    float value;
    if (params.bf16 != 0u) {
        // Offloaded BF16: read the raw bytes so an odd window offset is safe.
        const device uchar *raw = (const device uchar *)embedding;
        uint at = params.offset + (token_id * params.hidden + column) * 2u;
        ushort bits = ushort(raw[at]) | (ushort(raw[at + 1u]) << 8);
        value = float(bf16_to_half(bits));
    } else {
        // Arena copy: already FP16 at a 2-byte-aligned base, offset in elements.
        value = float(embedding[params.offset / 2u + token_id * params.hidden + column]);
    }
    output[gid] = half(value * params.scale);
}

// Copies the final row of a [rows, width] activation into a [width] buffer so
// the vocabulary head only ever runs for the position that is actually sampled.
kernel void copy_last_row(
    const device half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &shape [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    // shape packs (rows - 1) in the low half and width in the high half.
    uint width = shape >> 16u;
    uint row = shape & 0xffffu;
    if (gid >= width) {
        return;
    }
    output[gid] = input[row * width + gid];
}

kernel void copy_buffer(
    const device half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) {
        return;
    }
    output[gid] = input[gid];
}
