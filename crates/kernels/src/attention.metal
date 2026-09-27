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
