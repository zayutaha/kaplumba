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

struct NormScaleParams {
    uint rows;
    uint width;
    float eps;
    float scale;
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

kernel void rms_norm_residual_tg(
    const device half *input [[buffer(0)]],
    const device half *weight [[buffer(1)]],
    const device half *residual [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant NormParams &params [[buffer(4)]],
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
        const float normed = float(input[base + i]) * inverse * float(weight[i]);
        output[base + i] = half(float(residual[base + i]) + normed);
    }
}

kernel void rms_norm_residual_layer_scale_tg(
    const device half *input [[buffer(0)]],
    const device half *weight [[buffer(1)]],
    const device half *residual [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant NormScaleParams &params [[buffer(4)]],
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
        const float normed = float(input[base + i]) * inverse * float(weight[i]);
        const float res = float(residual[base + i]) + normed;
        output[base + i] = half(res * params.scale);
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
// Device-side argmax with 16-bit slot partitioning (argmax_slots).
// Fixes the 16-bit underflow in greedy_key that clobbered indices >= 65536.
static inline uint sortable16(uint bits) {
    return (bits & 0x8000u) != 0u ? ~bits : (bits ^ 0x8000u);
}

static inline uint value_key(uint bits) {
    const uint EXP = (bits >> 10) & 0x1Fu;
    const uint MAN = bits & 0x3FFu;
    if (EXP == 0x1Fu && MAN != 0u) {
        return 0u; // below -inf (0x03FF): NaN can never win
    }
    if ((bits & 0x7FFFu) == 0u) {
        return 0x8000u; // +0.0: -0.0 folds onto it
    }
    return sortable16(bits);
}

#define ARGMAX_SLOT_SIZE 65536u

static inline uint argmax_key(uint bits, uint index) {
    return (value_key(bits) << 16) | ((0xFFFFu - (index & 0xFFFFu)) & 0xFFFFu);
}

struct ArgmaxParams {
    uint count;
    uint per_slot;
};

kernel void argmax_clear(device atomic_uint *best [[buffer(0)]],
                         constant uint &slots [[buffer(1)]],
                         uint tid [[thread_position_in_grid]]) {
    if (tid < slots) {
        atomic_store_explicit(&best[tid], 0u, memory_order_relaxed);
    }
}

kernel void argmax_slots(const device half *logits [[buffer(0)]],
                         device atomic_uint *best [[buffer(1)]],
                         constant ArgmaxParams &params [[buffer(2)]],
                         uint lid [[thread_position_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint tgid [[threadgroup_position_in_grid]],
                         uint tgdim [[threads_per_threadgroup]]) {
    const uint per_slot = params.per_slot;
    const uint slot = tgid / per_slot;
    const uint lo = slot * ARGMAX_SLOT_SIZE;
    const uint hi = min(lo + ARGMAX_SLOT_SIZE, params.count);
    uint local = 0u;
    const uint stride = per_slot * tgdim;
    for (uint i = lo + (tgid % per_slot) * tgdim + lid; i < hi; i += stride) {
        local = max(local, argmax_key(uint(as_type<ushort>(logits[i])), i));
    }
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        local = max(local, simd_shuffle_xor(local, offset));
    }
    if (lane == 0u) {
        atomic_fetch_max_explicit(&best[slot], local, memory_order_relaxed);
    }
}

kernel void argmax_keys_to_token(
    constant const uint *keys [[buffer(0)]],
    device uint *token_out [[buffer(1)]],
    device uint *host_token_out [[buffer(2)]],
    constant const uint &step_idx [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]]) {
    if (lid == 0) {
        uint best_slot = 0;
        uint best = keys[0];
        for (uint slot = 1; slot < 4; ++slot) {
            if ((keys[slot] >> 16) > (best >> 16)) {
                best = keys[slot];
                best_slot = slot;
            }
        }
        uint in_slot = 0xFFFFu - (best & 0xFFFFu);
        uint token = best_slot * 65536u + in_slot;
        if (token_out != nullptr) {
            token_out[0] = token;
        }
        if (host_token_out != nullptr) {
            host_token_out[step_idx] = token;
        }
    }
}

static inline uint greedy_key(half value, uint index) {
    return argmax_key(uint(as_type<ushort>(value)), index);
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

// MEASUREMENT AID: a deliberately slow dispatch, used to prove that a timing
// instrument can see a cost injected into the region it claims to measure.
//
// A profiler that cannot resolve an injected cost has no coverage claim behind
// it, and this project's decode tables were believed for a long time before
// anyone checked that they could fail. `iters` is a runtime trip count; the loop
// carries a data dependency so it cannot be hoisted, and the result is written so
// the dispatch cannot be elided as dead.
//
// `iters` is at `buffer(2)`, not `buffer(1)`: `MetalEncoder::dispatch_1d` binds
// the caller's buffer list at indices 0..n and then packs the trailing `params`
// bytes at index n, so with one bound buffer the scalar lands at index 1. This
// kernel is dispatched with one buffer, so it must read index 2 -- and getting
// that wrong is silent, because an `iters` read from an unbound or
// zero-filled index is simply 0 and the dispatch costs nothing. It did cost
// nothing, and the only reason it was noticed is that the test comparing the
// instrument's number against the host clock failed.
kernel void spin_burn(device float *out [[buffer(0)]],
                      constant uint &iters [[buffer(2)]],
                      uint tid [[thread_position_in_grid]]) {
    float acc = float(tid) * 1e-3f;
    for (uint i = 0u; i < iters; ++i) {
        acc = fma(acc, 1.0000001f, 0.25f);
    }
    out[tid] = acc;
}

// Q/K: weighted RMSNorm then RoPE. V: unweighted RMSNorm only, no RoPE.
//
// WHY FUSED, AND WHAT THE BIT-EXACTNESS DEPENDS ON
// --------------------------------------------------
// The unfused path is five dispatches per layer (3 norms + 2 ropes). They are
// strictly elementwise and the fp16 buffer between them is a *rounding* step,
// not information the next kernel needs, so they fuse -- PROVIDED the fused
// kernel reproduces that rounding. It does, and it is the whole difficulty:
//
//   rms_norm_weighted_tg writes  half(input * inverse * weight)   <- fp16
//   rope                         reads float(that half)            <- widened
//
// Carrying the normed value in an fp32 register across the rotation instead
// skips the narrowing, and `half(lo*cos - hi*sin)` is not a function of
// `half(half(a)*c - half(half(b))*s)` in general: the two differ by up to half
// an fp16 ULP on almost every rotated element. So the normed value is rounded
// to fp16 HERE (`normed_shared[i] = float(half(normed))`) before the rotation
// reads it, which is the same value `rope` would have read back from DRAM.
//
// The reduction must also match. `rms_norm_weighted_tg` accumulates a strided
// partial per lane and then a tree, so a single-value-per-lane kernel (which is
// what a naive fusion writes) sums the head in a DIFFERENT ORDER and gets a
// different `rsqrt`. The strided loop below is character-for-character the one
// in `rms_norm_weighted_tg` / `rms_norm_unweighted_tg`, and the host dispatches
// 256 threads per group, which is `NORM_GROUP_THREADS`, so every partial sum
// and every tree step agrees. That is what makes this bit-identical rather than
// merely close.
//
// The rotation is one thread per PAIR, never per column, for the reason
// documented on `rope` (elements.metal:337): the partner column belongs to a
// different thread, and a per-column mapping is a cross-warp race on an
// in-place buffer. The staged values live in threadgroup memory and are fully
// written before the barrier, so the rotation reads only the staging array and
// writes only `target` -- no thread ever reads what another wrote.
//
// The non-rotated tail needs BOTH halves. A column is unrotated iff its pair is
// `>= rotated_pairs`, and a pair spans columns `p` and `p + half_width`, so the
// tail is pairs `rotated_pairs..half_width` in both halves. Copying only the low
// half leaves the high half unwritten -- the bug the proportional path in `rope`
// was written to avoid.

struct NormRopeQkvParams {
    uint q_heads;
    uint kv_heads;
    uint head_dim;
    uint offset;
    float eps;
    float theta;
    float partial_factor;
    uint proportional;
};

// One threadgroup per head: `q_heads` groups for Q, `kv_heads` for K, and
// `kv_heads` for V. Head dim is 256 on sliding layers and 512 on full-attention
// layers, so the staging array is sized for 512 and both are handled by the
// strided loops rather than by specialising the kernel twice.
kernel void norm_rope_qkv(
    device half *q [[buffer(0)]],
    device half *k [[buffer(1)]],
    device half *v [[buffer(2)]],
    const device half *q_weight [[buffer(3)]],
    const device half *k_weight [[buffer(4)]],
    constant NormRopeQkvParams &params [[buffer(5)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tgdim [[threads_per_threadgroup]]) {
    const uint kv_span = params.q_heads + params.kv_heads;
    if (tgid >= kv_span + params.kv_heads) {
        return;
    }
    // `NORM_GROUP_THREADS`. The host must dispatch exactly this; the reduction
    // tree below is hard-wired to it, exactly as `rms_norm_weighted_tg` is to
    // the `tgdim` it is dispatched with.
    threadgroup float partial[256];
    // Both heads' normalized values, rounded to fp16, staged so the rotation
    // reads a partner column it did not write. 512 floats = 2 KB, well inside
    // the 32 KB threadgroup budget alongside `partial`.
    threadgroup float normed_shared[512];

    device half *target;
    const device half *weight;
    uint head_idx;
    bool is_v;
    if (tgid < params.q_heads) {
        target = q;
        weight = q_weight;
        head_idx = tgid;
        is_v = false;
    } else if (tgid < kv_span) {
        target = k;
        weight = k_weight;
        head_idx = tgid - params.q_heads;
        is_v = false;
    } else {
        target = v;
        // V is UNWEIGHTED, so `weight` is never read on this path. It is given a
        // valid buffer rather than null anyway: a speculative load of a null
        // device pointer is a fault, and "the compiler must not hoist this" is
        // not a property to rely on.
        weight = q_weight;
        head_idx = tgid - kv_span;
        is_v = true;
    }

    const uint count = params.head_dim;
    const uint base = head_idx * count;

    // Sum of squares, in the same order as `rms_norm_weighted_tg`.
    float sum = 0.0f;
    for (uint i = lid; i < count; i += tgdim) {
        const float value = float(target[base + i]);
        sum += value * value;
    }
    partial[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            partial[lid] += partial[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inverse = rsqrt(partial[0] / float(count) + params.eps);

    if (is_v) {
        // `rms_norm_unweighted_tg`: no weight, no RoPE, straight store.
        for (uint i = lid; i < count; i += tgdim) {
            target[base + i] = half(float(target[base + i]) * inverse);
        }
        return;
    }

    // Weighted norm, ROUNDED TO FP16 before the rotation reads it. The
    // `float(half(...))` is load-bearing; see the note at the top of the kernel.
    for (uint i = lid; i < count; i += tgdim) {
        const float normed = float(target[base + i]) * inverse * float(weight[i]);
        normed_shared[i] = float(half(normed));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint half_width = count / 2u;
    // Same expression as `rope`, deliberately without the host-side
    // `.min(half_width)`: `partial_factor <= 1` already bounds this by
    // `half_width`, and matching the sibling kernel is what keeps the rotated
    // span identical.
    const uint rotated_pairs = params.proportional != 0u
        ? uint(params.partial_factor * float(count) / 2.0f)
        : half_width;

    for (uint pair = lid; pair < rotated_pairs; pair += tgdim) {
        const float exponent = 2.0f * float(pair) / float(count);
        const float angle = float(params.offset) / pow(params.theta, exponent);
        const float cosine = cos(angle);
        const float sine = sin(angle);
        const float lo = normed_shared[pair];
        const float hi = normed_shared[pair + half_width];
        target[base + pair] = half(lo * cosine - hi * sine);
        target[base + pair + half_width] = half(hi * cosine + lo * sine);
    }
    // Non-rotated tail: both halves of every pair `rotated_pairs..half_width`.
    // `half(float(half(x))) == half(x)`, so this is a verbatim copy of what the
    // norm wrote, which is what `rope`'s copy branch does.
    for (uint pair = rotated_pairs + lid; pair < half_width; pair += tgdim) {
        target[base + pair] = half(normed_shared[pair]);
        target[base + pair + half_width] = half(normed_shared[pair + half_width]);
    }
}

struct NormRopeQkvCacheParams {
    uint q_heads;
    uint kv_heads;
    uint head_dim;
    uint offset;
    float eps;
    float theta;
    float partial_factor;
    uint proportional;
    uint slot;
    uint stream_len;
};

kernel void norm_rope_qkv_cache(
    device half *q [[buffer(0)]],
    const device half *raw_k [[buffer(1)]],
    const device half *raw_v [[buffer(2)]],
    device half *key_cache [[buffer(3)]],
    device half *value_cache [[buffer(4)]],
    device uint *position_cache [[buffer(5)]],
    const device half *q_weight [[buffer(6)]],
    const device half *k_weight [[buffer(7)]],
    constant NormRopeQkvCacheParams &params [[buffer(8)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tgdim [[threads_per_threadgroup]]) {
    const uint kv_span = params.q_heads + params.kv_heads;
    if (tgid >= kv_span + params.kv_heads) {
        return;
    }
    threadgroup float partial[256];
    threadgroup float normed_shared[512];

    const device half *src;
    device half *dst;
    const device half *weight;
    uint head_idx;
    bool is_v;
    if (tgid < params.q_heads) {
        src = q;
        dst = q;
        weight = q_weight;
        head_idx = tgid;
        is_v = false;
    } else if (tgid < kv_span) {
        src = raw_k;
        dst = key_cache + params.slot * params.stream_len;
        weight = k_weight;
        head_idx = tgid - params.q_heads;
        is_v = false;
    } else {
        src = raw_v;
        dst = value_cache + params.slot * params.stream_len;
        weight = q_weight;
        head_idx = tgid - kv_span;
        is_v = true;
    }

    const uint count = params.head_dim;
    const uint base = head_idx * count;

    // Sum of squares, in the same order as `rms_norm_weighted_tg`.
    float sum = 0.0f;
    for (uint i = lid; i < count; i += tgdim) {
        const float value = float(src[base + i]);
        sum += value * value;
    }
    partial[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            partial[lid] += partial[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inverse = rsqrt(partial[0] / float(count) + params.eps);

    if (is_v) {
        // `rms_norm_unweighted_tg`: no weight, no RoPE, straight store.
        for (uint i = lid; i < count; i += tgdim) {
            dst[base + i] = half(float(src[base + i]) * inverse);
        }
        return;
    }

    // Weighted norm, ROUNDED TO FP16 before the rotation reads it.
    for (uint i = lid; i < count; i += tgdim) {
        const float normed = float(src[base + i]) * inverse * float(weight[i]);
        normed_shared[i] = float(half(normed));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint half_width = count / 2u;
    const uint rotated_pairs = params.proportional != 0u
        ? uint(params.partial_factor * float(count) / 2.0f)
        : half_width;

    for (uint pair = lid; pair < rotated_pairs; pair += tgdim) {
        const float exponent = 2.0f * float(pair) / float(count);
        const float angle = float(params.offset) / pow(params.theta, exponent);
        const float cosine = cos(angle);
        const float sine = sin(angle);
        const float lo = normed_shared[pair];
        const float hi = normed_shared[pair + half_width];
        dst[base + pair] = half(lo * cosine - hi * sine);
        dst[base + pair + half_width] = half(hi * cosine + lo * sine);
    }
    for (uint pair = rotated_pairs + lid; pair < half_width; pair += tgdim) {
        dst[base + pair] = half(normed_shared[pair]);
        dst[base + pair + half_width] = half(normed_shared[pair + half_width]);
    }

    if (tgid == params.q_heads && lid == 0u) {
        position_cache[params.slot] = params.offset;
    }
}


