#include <metal_stdlib>
using namespace metal;

struct AttentionParams {
    uint queries;
    uint q_heads;
    uint kv_heads;
    uint head_dim;
    uint key_count;
    float scale;
    uint window;
};

template <int HEAD_DIM>
static void online_attention_body(
    const device half *queries [[buffer(0)]],
    const device half *keys [[buffer(1)]],
    const device half *values [[buffer(2)]],
    const device uint *query_positions [[buffer(3)]],
    const device uint *key_positions [[buffer(4)]],
    device half *output [[buffer(5)]],
    constant AttentionParams &params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.queries * params.q_heads) {
        return;
    }
    uint query = gid / params.q_heads;
    uint query_head = gid % params.q_heads;
    uint groups = params.q_heads / params.kv_heads;
    uint key_head = query_head / groups;
    uint query_base = (query * params.q_heads + query_head) * params.head_dim;
    uint query_position = query_positions[query];
    // Compile-time bound, so the loops below unroll and these become registers.
    // With the previous runtime bound this array spilled to local memory and
    // every access in the inner loop paid a local-memory round trip -- which is
    // why prefill attention was ~58% of prefill wall clock for under 1% of the
    // arithmetic.
    float accum[HEAD_DIM];
    for (int i = 0; i < HEAD_DIM; ++i) {
        accum[i] = 0.0f;
    }
    float maximum = -3.402823466e+38f;
    float total = 0.0f;
    for (uint key_index = 0; key_index < params.key_count; ++key_index) {
        uint key_position = key_positions[key_index];
        if (key_position > query_position) {
            continue;
        }
        if (params.window != 0u && query_position - key_position >= params.window) {
            continue;
        }
        uint key_base = (key_index * params.kv_heads + key_head) * params.head_dim;
        float score = 0.0f;
        for (int dimension = 0; dimension < HEAD_DIM; ++dimension) {
            score += float(queries[query_base + dimension]) * float(keys[key_base + dimension]);
        }
        score *= params.scale;
        if (maximum == -3.402823466e+38f) {
            maximum = score;
            total = 1.0f;
            for (uint dimension = 0; dimension < params.head_dim; ++dimension) {
                accum[dimension] = float(values[key_base + dimension]);
            }
        } else if (score > maximum) {
            float factor = exp(maximum - score);
            total = total * factor + 1.0f;
            for (uint dimension = 0; dimension < params.head_dim; ++dimension) {
                accum[dimension] = accum[dimension] * factor + float(values[key_base + dimension]);
            }
            maximum = score;
        } else {
            float factor = exp(score - maximum);
            total += factor;
            for (uint dimension = 0; dimension < params.head_dim; ++dimension) {
                accum[dimension] += factor * float(values[key_base + dimension]);
            }
        }
    }
    uint output_base = query_base;
    if (total == 0.0f) {
        for (uint dimension = 0; dimension < params.head_dim; ++dimension) {
            output[output_base + dimension] = half(0.0f);
        }
        return;
    }
    for (uint dimension = 0; dimension < params.head_dim; ++dimension) {
        output[output_base + dimension] = half(accum[dimension] / total);
    }
}


// ---------------------------------------------------------------------------
// Split-key attention, for the decode shape.
//
// The kernel above dispatches one thread per (query, head), so a decode step
// runs on 32 threads while pulling ~67 MB of cold KV cache per token. Measured
// 153 ms in situ against 132 us for the same shape with an L2-resident buffer,
// so the serialisation, not the arithmetic, is the cost.
//
// This is ordinary flash-decoding in three passes, chosen so that no pass needs
// a per-thread array proportional to head_dim:
//
//   1. scores  : one thread per (head, key). Reads the keys once.
//   2. reduce  : one thread per head. Finds the row max and sum of exp.
//   3. combine : one thread per (head, dimension). Reads the values once.
//
// Peak parallelism is q_heads * key_count and q_heads * head_dim instead of
// q_heads, and the KV data is read once rather than being re-read by a
// single thread.
kernel void attention_scores(
    const device half *queries [[buffer(0)]],
    const device half *keys [[buffer(1)]],
    const device uint *query_positions [[buffer(2)]],
    const device uint *key_positions [[buffer(3)]],
    device float *scores [[buffer(4)]],
    constant AttentionParams &params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    uint key_index = gid % params.key_count;
    uint head = gid / params.key_count;
    if (head >= params.q_heads) {
        return;
    }
    // Stale and future slots score nothing; the window check must precede the
    // key load, as in the serial kernel.
    const uint query_position = query_positions[0];
    const uint key_position = key_positions[key_index];
    if (key_position > query_position) {
        scores[gid] = -3.402823466e38f;
        return;
    }
    if (params.window != 0u && query_position - key_position >= params.window) {
        scores[gid] = -3.402823466e38f;
        return;
    }
    const uint kv_heads = params.kv_heads;
    const uint key_head = head / (params.q_heads / kv_heads);
    const uint key_base = (key_index * kv_heads + key_head) * params.head_dim;
    const uint query_base = head * params.head_dim;
    float score = 0.0f;
    for (uint d = 0; d < params.head_dim; ++d) {
        score += float(queries[query_base + d]) * float(keys[key_base + d]);
    }
    scores[gid] = score * params.scale;
}

kernel void attention_row_reduce(
    const device float *scores [[buffer(0)]],
    device float *row_max [[buffer(1)]],
    device float *row_sum [[buffer(2)]],
    constant AttentionParams &params [[buffer(3)]],
    uint head [[thread_position_in_grid]]) {
    if (head >= params.q_heads) {
        return;
    }
    const device float *row = scores + head * params.key_count;
    float maximum = -3.402823466e38f;
    for (uint k = 0; k < params.key_count; ++k) {
        maximum = max(maximum, row[k]);
    }
    float total = 0.0f;
    for (uint k = 0; k < params.key_count; ++k) {
        total += exp(row[k] - maximum);
    }
    row_max[head] = maximum;
    row_sum[head] = total;
}

kernel void attention_combine(
    const device half *values [[buffer(0)]],
    const device float *scores [[buffer(1)]],
    const device float *row_max [[buffer(2)]],
    const device float *row_sum [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant AttentionParams &params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    uint dimension = gid % params.head_dim;
    uint head = gid / params.head_dim;
    if (head >= params.q_heads) {
        return;
    }
    const device float *row = scores + head * params.key_count;
    const float maximum = row_max[head];
    const float total = row_sum[head];
    const uint kv_heads = params.kv_heads;
    const uint key_head = head / (params.q_heads / kv_heads);
    float accumulator = 0.0f;
    if (total > 0.0f) {
        for (uint k = 0; k < params.key_count; ++k) {
            const float score = row[k];
            if (score <= -3.402823466e38f) {
                continue;
            }
            const uint value_base = (k * kv_heads + key_head) * params.head_dim + dimension;
            accumulator += exp(score - maximum) * float(values[value_base]);
        }
        accumulator /= total;
    }
    output[head * params.head_dim + dimension] = half(accumulator);
}

// Decode attention, one threadgroup per (key, kv_head). The previous mapping put
// key_index on the lane, but keys are stored [key][kv_head][head_dim], so
// neighbouring lanes landed `kv_heads * head_dim` elements apart and every warp
// fetch touched 32 distinct sectors to deliver 64 useful bytes. Here the lane
// owns a dimension instead, so the key row is read once, fully coalesced, and
// the threadgroup reduces the dot products.
kernel void attention_scores_coalesced(
    const device half *queries [[buffer(0)]],
    const device half *keys [[buffer(1)]],
    const device uint *query_positions [[buffer(2)]],
    const device uint *key_positions [[buffer(3)]],
    device float *scores [[buffer(4)]],
    constant AttentionParams &params [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gt [[threadgroup_position_in_grid]]) {
    const uint kv_heads = params.kv_heads;
    const uint key_index = gt / kv_heads;
    const uint key_head = gt - key_index * kv_heads;
    const uint ratio = params.q_heads / kv_heads;
    const uint q_head = key_head * ratio;

    const uint query_position = query_positions[0];
    const uint key_position = key_positions[key_index];
    bool live = key_position <= query_position;
    if (live && params.window != 0u && query_position - key_position >= params.window) {
        live = false;
    }
    if (!live) {
        if (tid < ratio) {
            scores[(q_head + tid) * params.key_count + key_index] = -3.402823466e38f;
        }
        return;
    }

    const uint key_base = (key_index * kv_heads + key_head) * params.head_dim;
    const uint query_base = q_head * params.head_dim;
    // Each q head sharing this kv head has its own query vector, so the key is
    // read once and dotted against all of them. `ratio` is 2 on the sliding
    // layers and 8 on the global ones.
    float partial[8];
    for (uint r = 0; r < ratio; ++r) {
        partial[r] = 0.0f;
    }
    for (uint d = tid; d < params.head_dim; d += 256u) {
        const float k = float(keys[key_base + d]);
        for (uint r = 0; r < ratio; ++r) {
            partial[r] += float(queries[query_base + r * params.head_dim + d]) * k;
        }
    }

    threadgroup float partials[8][256];
    for (uint r = 0; r < ratio; ++r) {
        partials[r][tid] = partial[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        for (uint r = 0; r < ratio; ++r) {
            float total = 0.0f;
            for (uint t = 0; t < 256u; ++t) {
                total += partials[r][t];
            }
            scores[(q_head + r) * params.key_count + key_index] = total * params.scale;
        }
    }
}

// One threadgroup per query head. The previous version dispatched a single
// thread per head -- 32 threads for the whole GPU -- and each walked key_count
// scores twice.
kernel void attention_row_reduce_coalesced(
    const device float *scores [[buffer(0)]],
    device float *row_max [[buffer(1)]],
    device float *row_sum [[buffer(2)]],
    constant AttentionParams &params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint head [[threadgroup_position_in_grid]]) {
    if (head >= params.q_heads) {
        return;
    }
    const device float *row = scores + head * params.key_count;
    float maximum = -3.402823466e38f;
    for (uint k = tid; k < params.key_count; k += 256u) {
        maximum = max(maximum, row[k]);
    }
    threadgroup float shared_max[256];
    shared_max[tid] = maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float all_max = shared_max[0];
    if (tid == 0u) {
        row_max[head] = all_max;
    }

    float total = 0.0f;
    for (uint k = tid; k < params.key_count; k += 256u) {
        total += exp(row[k] - all_max);
    }
    threadgroup float shared_sum[256];
    shared_sum[tid] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        row_sum[head] = shared_sum[0];
    }
}

// Instantiations for the two head dims this model uses: 256 on the 50 sliding
// layers, 512 on the 10 full-attention layers.
kernel void online_attention_256(
    const device half *queries [[buffer(0)]],
    const device half *keys [[buffer(1)]],
    const device half *values [[buffer(2)]],
    const device uint *query_positions [[buffer(3)]],
    const device uint *key_positions [[buffer(4)]],
    device half *output [[buffer(5)]],
    constant AttentionParams &params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    online_attention_body<256>(queries, keys, values, query_positions, key_positions, output, params, gid);
}

kernel void online_attention_512(
    const device half *queries [[buffer(0)]],
    const device half *keys [[buffer(1)]],
    const device half *values [[buffer(2)]],
    const device uint *query_positions [[buffer(3)]],
    const device uint *key_positions [[buffer(4)]],
    device half *output [[buffer(5)]],
    constant AttentionParams &params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    online_attention_body<512>(queries, keys, values, query_positions, key_positions, output, params, gid);
}

// Generic fallback for a head_dim with no specialisation. Kept because the
// kernel's public contract accepts any head_dim up to 512, and the engine tests
// exercise small ones. Slow (runtime bound => the accumulator spills) but
// correct; the model itself only ever hits the 256 and 512 instantiations.
kernel void online_attention(
    const device half *queries [[buffer(0)]],
    const device half *keys [[buffer(1)]],
    const device half *values [[buffer(2)]],
    const device uint *query_positions [[buffer(3)]],
    const device uint *key_positions [[buffer(4)]],
    device half *output [[buffer(5)]],
    constant AttentionParams &params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.queries * params.q_heads) {
        return;
    }
    uint query = gid / params.q_heads;
    uint query_head = gid % params.q_heads;
    uint groups = params.q_heads / params.kv_heads;
    uint key_head = query_head / groups;
    uint query_base = (query * params.q_heads + query_head) * params.head_dim;
    uint query_position = query_positions[query];
    float accum[512];
    for (uint i = 0; i < params.head_dim; ++i) {
        accum[i] = 0.0f;
    }
    float maximum = -3.402823466e+38f;
    float total = 0.0f;
    for (uint key_index = 0; key_index < params.key_count; ++key_index) {
        uint key_position = key_positions[key_index];
        if (key_position > query_position) {
            continue;
        }
        if (params.window != 0u && query_position - key_position >= params.window) {
            continue;
        }
        uint key_base = (key_index * params.kv_heads + key_head) * params.head_dim;
        float score = 0.0f;
        for (uint dimension = 0; dimension < params.head_dim; ++dimension) {
            score += float(queries[query_base + dimension]) * float(keys[key_base + dimension]);
        }
        score *= params.scale;
        if (maximum == -3.402823466e+38f) {
            maximum = score;
            total = 1.0f;
            for (uint dimension = 0; dimension < params.head_dim; ++dimension) {
                accum[dimension] = float(values[key_base + dimension]);
            }
        } else if (score > maximum) {
            float factor = exp(maximum - score);
            total = total * factor + 1.0f;
            maximum = score;
            for (uint dimension = 0; dimension < params.head_dim; ++dimension) {
                accum[dimension] = accum[dimension] * factor + float(values[key_base + dimension]);
            }
        } else {
            float factor = exp(score - maximum);
            total += factor;
            for (uint dimension = 0; dimension < params.head_dim; ++dimension) {
                accum[dimension] += factor * float(values[key_base + dimension]);
            }
        }
    }
    float norm = total > 0.0f ? 1.0f / total : 0.0f;
    for (uint dimension = 0; dimension < params.head_dim; ++dimension) {
        output[query_base + dimension] = half(accum[dimension] * norm);
    }
}

// Prefill attention with one thread per (query, head, DIMENSION).
//
// The serial kernel gave every (query, head) its own `float accum[512]`, which is
// 1024 bytes per thread -- larger than the register file -- so it lived in
// private local memory and every access in the inner loop paid a local-memory
// round trip. That made prefill attention ~58% of prefill wall clock for under
// 1% of the arithmetic.
//
// The fix is that the 256 accumulators are mutually independent: dimension d's
// recurrence never reads any other dimension. So give each dimension its own
// thread and the accumulator is one float in a register. The cost is that the
// score dot product is recomputed once per dimension rather than shared, so it
// runs `head_dim` times redundantly -- affordable precisely because the serial
// version was three orders of magnitude off the arithmetic peak.
//
// Bit-exact: the dot product runs the identical d = 0,1,...,HEAD_DIM-1 sequence,
// and each dimension's online-softmax recurrence sees the identical key order
// with the identical max/total/factor updates, so every output element performs
// the same sequence of fp32 operations as before. No score matrix, no scratch
// beyond the kernel's own registers.
template <int HEAD_DIM>
static void online_attention_dim_body(
    const device half *queries,
    const device half *keys,
    const device half *values,
    const device uint *query_positions,
    const device uint *key_positions,
    device half *output,
    constant AttentionParams &params,
    uint gid) {
    uint span = params.queries * params.q_heads * uint(HEAD_DIM);
    if (gid >= span) {
        return;
    }
    const uint dim = gid % uint(HEAD_DIM);
    const uint head = (gid / uint(HEAD_DIM)) % params.q_heads;
    const uint query = gid / (uint(HEAD_DIM) * params.q_heads);
    const uint groups = params.q_heads / params.kv_heads;
    const uint key_head = head / groups;
    const uint query_base = (query * params.q_heads + head) * uint(HEAD_DIM);
    const uint query_position = query_positions[query];

    float maximum = -3.402823466e+38f;
    float total = 0.0f;
    float accumulator = 0.0f;

    for (uint key_index = 0; key_index < params.key_count; ++key_index) {
        const uint key_position = key_positions[key_index];
        if (key_position > query_position) {
            continue;
        }
        if (params.window != 0u && query_position - key_position >= params.window) {
            continue;
        }
        const uint key_base = (key_index * params.kv_heads + key_head) * uint(HEAD_DIM);
        // Recomputed per dimension rather than shared across the 256 dimensions.
        float score = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d) {
            score += float(queries[query_base + uint(d)]) * float(keys[key_base + uint(d)]);
        }
        score *= params.scale;
        const float value = float(values[key_base + dim]);
        if (maximum == -3.402823466e+38f) {
            maximum = score;
            total = 1.0f;
            accumulator = value;
        } else if (score > maximum) {
            float factor = exp(maximum - score);
            total = total * factor + 1.0f;
            maximum = score;
            accumulator = accumulator * factor + value;
        } else {
            float factor = exp(score - maximum);
            total += factor;
            accumulator += factor * value;
        }
    }
    const float norm = total > 0.0f ? 1.0f / total : 0.0f;
    output[query_base + dim] = half(accumulator * norm);
}

kernel void online_attention_dim_256(
    const device half *queries [[buffer(0)]],
    const device half *keys [[buffer(1)]],
    const device half *values [[buffer(2)]],
    const device uint *query_positions [[buffer(3)]],
    const device uint *key_positions [[buffer(4)]],
    device half *output [[buffer(5)]],
    constant AttentionParams &params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    online_attention_dim_body<256>(queries, keys, values, query_positions, key_positions, output, params, gid);
}

kernel void online_attention_dim_512(
    const device half *queries [[buffer(0)]],
    const device half *keys [[buffer(1)]],
    const device half *values [[buffer(2)]],
    const device uint *query_positions [[buffer(3)]],
    const device uint *key_positions [[buffer(4)]],
    device half *output [[buffer(5)]],
    constant AttentionParams &params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    online_attention_dim_body<512>(queries, keys, values, query_positions, key_positions, output, params, gid);
}

// Prefill attention: one THREADGROUP per (query, head), three phases per key
// chunk. Phase 1 computes the chunk's scores (one thread per key, the dot
// product walked in its original d = 0..HEAD_DIM-1 order). Phase 1b walks the
// chunk's scores in order to advance the online-softmax running max and total
// and to emit one factor per key. Phase 2 gives each of the 256 threads a slice
// of the output dimensions, so its accumulators are registers, and applies the
// factor.
//
// Two earlier attempts were worse and both are instructive:
//   - one thread per (query, head, DIMENSION), recomputing the score per
//     dimension: correct, but 256x the arithmetic and the key traffic.
//   - the same, with the online-softmax recurrence also per dimension: correct,
//     but 256x the exp() calls, which is what dominated.
//
// Here exp() is called once per (query, head, key), as in the original, because
// phase 1b does the recurrence and phase 2 only applies the factor it produced.
//
// Bit-exact: phase 1b performs the original's max/total/factor updates in the
// original key order, and phase 2 applies the identical accumulator update for
// the identical key sequence, so every output element sees the same sequence of
// fp32 operations. Shared scratch is a fixed 256 keys' worth of score, factor
// and rescale flags -- it does not grow with context length.
template <int HEAD_DIM>
static void online_attention_tg2_body(
    const device half *queries,
    const device half *keys,
    const device half *values,
    const device uint *query_positions,
    const device uint *key_positions,
    device half *output,
    constant AttentionParams &params,
    uint gid,
    threadgroup float *sbuf,
    threadgroup float *fbuf,
    threadgroup float *rbuf,
    threadgroup float *stot) {
    uint span = params.queries * params.q_heads * 256u;
    if (gid >= span) {
        return;
    }
    const uint tid = gid & 255u;
    const uint head = (gid / 256u) % params.q_heads;
    const uint query = gid / (256u * params.q_heads);
    const uint groups = params.q_heads / params.kv_heads;
    const uint key_head = head / groups;
    const uint query_base = (query * params.q_heads + head) * uint(HEAD_DIM);
    const uint query_position = query_positions[query];

    constexpr uint DPT = (HEAD_DIM + 255u) / 256u;
    float maximum = -3.402823466e+38f;
    float total = 0.0f;
    float accumulator[DPT];
    for (uint j = 0; j < DPT; ++j) {
        accumulator[j] = 0.0f;
    }
    bool started = false;

    for (uint base = 0; base < params.key_count; base += 256u) {
        const uint n = min(256u, params.key_count - base);
        if (tid < n) {
            const uint key_base = ((base + tid) * params.kv_heads + key_head) * uint(HEAD_DIM);
            float score = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) {
                score += float(queries[query_base + uint(d)]) * float(keys[key_base + uint(d)]);
            }
            sbuf[tid] = score * params.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            for (uint t = 0; t < n; ++t) {
                const uint key_position = key_positions[base + t];
                bool live = key_position <= query_position;
                if (live && params.window != 0u && query_position - key_position >= params.window) {
                    live = false;
                }
                if (!live) {
                    fbuf[t] = -1.0f;
                    rbuf[t] = 0.0f;
                    continue;
                }
                const float score = sbuf[t];
                if (!started) {
                    started = true;
                    maximum = score;
                    total = 1.0f;
                    fbuf[t] = 0.0f;
                    rbuf[t] = 2.0f;
                } else if (score > maximum) {
                    fbuf[t] = exp(maximum - score);
                    total = total * fbuf[t] + 1.0f;
                    maximum = score;
                    rbuf[t] = 1.0f;
                } else {
                    fbuf[t] = exp(score - maximum);
                    total += fbuf[t];
                    rbuf[t] = 0.0f;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = 0; j < DPT; ++j) {
            const uint d = tid + j * 256u;
            if (d >= uint(HEAD_DIM)) {
                continue;
            }
            for (uint t = 0; t < n; ++t) {
                if (fbuf[t] < 0.0f) {
                    continue;
                }
                const float value = float(values[((base + t) * params.kv_heads + key_head) * uint(HEAD_DIM) + d]);
                if (rbuf[t] == 2.0f) {
                    accumulator[j] = value;
                } else if (rbuf[t] != 0.0f) {
                    accumulator[j] = accumulator[j] * fbuf[t] + value;
                } else {
                    accumulator[j] += fbuf[t] * value;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Only thread 0 advances the running total, so it has to be broadcast --
    // otherwise every other thread normalises by its own untouched zero.
    if (tid == 0u) {
        *stot = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float shared_total = *stot;
    const float norm = shared_total > 0.0f ? 1.0f / shared_total : 0.0f;
    for (uint j = 0; j < DPT; ++j) {
        const uint d = tid + j * 256u;
        if (d < uint(HEAD_DIM)) {
            output[query_base + d] = half(accumulator[j] * norm);
        }
    }
}

kernel void online_attention_tg2_256(
    const device half *queries [[buffer(0)]],
    const device half *keys [[buffer(1)]],
    const device half *values [[buffer(2)]],
    const device uint *query_positions [[buffer(3)]],
    const device uint *key_positions [[buffer(4)]],
    device half *output [[buffer(5)]],
    constant AttentionParams &params [[buffer(6)]],
    uint gid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
    threadgroup float sbuf[256];
    threadgroup float fbuf[256];
    threadgroup float rbuf[256];
    threadgroup float stot;
    online_attention_tg2_body<256>(queries, keys, values, query_positions, key_positions, output, params, tgid * 256u + gid, sbuf, fbuf, rbuf, &stot);
}

kernel void online_attention_tg2_512(
    const device half *queries [[buffer(0)]],
    const device half *keys [[buffer(1)]],
    const device half *values [[buffer(2)]],
    const device uint *query_positions [[buffer(3)]],
    const device uint *key_positions [[buffer(4)]],
    device half *output [[buffer(5)]],
    constant AttentionParams &params [[buffer(6)]],
    uint gid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
    threadgroup float sbuf[256];
    threadgroup float fbuf[256];
    threadgroup float rbuf[256];
    threadgroup float stot;
    online_attention_tg2_body<512>(queries, keys, values, query_positions, key_positions, output, params, tgid * 256u + gid, sbuf, fbuf, rbuf, &stot);
}
