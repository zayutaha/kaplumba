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

// One thread per (token, head, rotation PAIR), not per column.
//
// The kernel is called IN PLACE on q and k (`encode_rope(&scratch.q,
// &scratch.q, ...)`, model.rs:1265/1279), and the rotation reads column
// `c + head_dim/2` while writing column `c`. With one thread per column, the
// thread for `c` could write `output[c]` before the thread for
// `c + head_dim/2` had read `input[c]`. That is a cross-warp race whose winner
// depended on scheduling, and it is why greedy decode was nondeterministic in
// ~2 of 8 runs of a 64-token prompt. `KAPLUMBA_MAX_IN_FLIGHT=1` could not fix
// it: the hazard is inside one dispatch, not between command buffers.
//
// Each thread now owns both halves of the pair it writes, so it reads exactly
// the two elements it produces and no thread touches another's. No barrier and
// no scratch are needed. The arithmetic is unchanged, so the result is
// bit-identical to the out-of-place form. Dispatch is halved, and the per-thread
// cos/sin count is halved with it, so the cost nets out.
kernel void rope(
    const device half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant RopeParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    uint half_width = params.head_dim / 2u;
    uint rotated_pairs = params.proportional != 0u
        ? uint(params.partial_factor * float(params.head_dim) / 2.0f)
        : half_width;
    // Non-rotated tail.
    //
    // A column is rotated iff its PAIR is below `rotated_pairs`, and a pair
    // spans BOTH halves -- columns `p` and `p + head_dim/2`. So the unrotated
    // tail is pairs `rotated_pairs .. half_width`, i.e. columns
    // `rotated_pairs .. half_width` in the low half AND the same columns shifted
    // by `half_width` in the high half. Copying only the low half leaves the
    // high half unwritten, which is what the proportional test caught.
    //
    // This range lives AFTER the rotated range: the host dispatches
    // `rotated + tail` threads, so both cannot start at gid 0.
    uint vector_width = params.heads * params.head_dim;
    uint pair_total = params.tokens * params.heads * rotated_pairs;
    if (rotated_pairs < half_width) {
        uint copy_width = half_width - rotated_pairs;
        uint copy_count = params.tokens * params.heads * copy_width;
        if (gid >= pair_total && gid < pair_total + copy_count) {
            uint local = gid - pair_total;
            uint token = local / (params.heads * copy_width);
            uint within = local - token * params.heads * copy_width;
            uint head = within / copy_width;
            uint pair = rotated_pairs + within % copy_width;
            uint base = token * vector_width + head * params.head_dim + pair;
            output[base] = input[base];
            output[base + half_width] = input[base + half_width];
        }
    }
    if (gid >= pair_total) {
        return;
    }
    // The rotated branch reads the partner at `base + half_width`, which for
    // `pair < rotated_pairs` lands in the tail region the copy branch above
    // also touches. The two must not both act: the copy is a straight
    // pass-through of columns `rotated_pairs..half_width`, and the rotation
    // reads (and writes) `base + half_width` only for the pairs it owns, so
    // reading a value the copy already moved is fine but WRITING it is not.
    // Hence the rotation writes only `base` and `base + half_width` for pairs
    // it owns, and the copy owns everything else. That is exactly the original
    // per-column split, reorganised so no thread reads another's output.
    uint pair = gid % rotated_pairs;
    uint within = gid / rotated_pairs;
    uint head = within % params.heads;
    uint token = within / params.heads;
    uint position = params.offset + token;
    float exponent = 2.0f * float(pair) / float(params.head_dim);
    float angle = float(position) / pow(params.theta, exponent);
    float cosine = cos(angle);
    float sine = sin(angle);
    uint base = token * vector_width + head * params.head_dim + pair;
    float lo = float(input[base]);
    float hi = float(input[base + half_width]);
    output[base] = half(lo * cosine - hi * sine);
    output[base + half_width] = half(hi * cosine + lo * sine);
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


// Greedy pick over the capped logits.
//
// Exists so the decode loop never has to bring 262,144 logits to the host and
// sample there. With the pick on the GPU, the next step's forward can read the
// token straight out of device memory, so steps queue back to back and the host
// is not a stall between them. ishizuki measures this exact change at
// 14.35 -> 22.48 tok/s, and attributes it to the serial loop "waiting on the GPU
// twice a token, once for the step and once for the pick".
//
// The key packs a sortable form of the fp16 value in the high 16 bits and the
// complement of the index in the low 16, so one unsigned max yields the largest
// logit and, among ties, the lowest index. That makes the reduction a plain
// atomic_max with no second pass and no comparison branching.
static inline uint greedy_key(half value, uint index) {
    const uint SIGN = 0x8000u;
    uint bits = uint(as_type<ushort>(value));
    // Negative floats compare backwards as unsigned integers; flip every
    // non-negative one so the unsigned ordering matches the float ordering.
    uint sortable = (bits & SIGN) != 0u ? ~bits : (bits ^ SIGN);
    return (sortable << 16) | (0xFFFFu - index);
}

static inline half key_to_half(uint key) {
    const uint SIGN = 0x8000u;
    uint sortable = key >> 16;
    uint bits = (sortable & SIGN) != 0u ? (sortable ^ SIGN) : ~sortable;
    return as_type<half>(ushort(bits));
}

kernel void greedy_pick(const device half *logits [[buffer(0)]],
                        device atomic_uint *best [[buffer(1)]],
                        constant uint &count [[buffer(2)]],
                        uint simd_lane [[thread_index_in_simdgroup]]) {
    uint local = 0u;
    for (uint i = simd_lane; i < count; i += 32u) {
        local = max(local, greedy_key(logits[i], i));
    }
    // 32-lane max via shuffles, so the reduction is one instruction per step.
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        local = max(local, simd_shuffle_xor(local, offset));
    }
    if (simd_lane == 0u) {
        atomic_fetch_max_explicit(best, local, memory_order_relaxed);
    }
}

// Turn the winning key back into a token id, for one query row.
kernel void greedy_token(const device atomic_uint *best [[buffer(0)]],
                         device uint *out [[buffer(1)]],
                         constant uint &row [[buffer(2)]],
                         uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) {
        return;
    }
    out[row] = 0xFFFFu - (*(const device uint *)best) % 0x10000u;
}
