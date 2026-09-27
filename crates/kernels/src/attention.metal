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
