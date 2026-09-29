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
    // Decode only (`splits > 0`): the KV axis is cut into `splits` contiguous
    // ranges of `keys_per_split` keys, one workgroup per (split, kv head).
    // Prefill passes 0 and scans `0 .. key_count`.
    uint splits;
    uint keys_per_split;
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
// Decode attention: flash-decoding over the KV axis.
//
// There is no separate decode kernel. The transposed prefill body at the bottom
// of this file runs unchanged for a single query, except that its key range
// comes from `splits`/`keys_per_split` instead of `0 .. key_count` and it emits
// an unnormalised (m, l, O) partial per split instead of a normalised output
// row. `attention_decode_combine_*` then merges the partials.
//
// What it replaces: a three-pass softmax over a materialised
// `q_heads * key_count` fp32 score matrix. That matrix is O(L^2) in the worst
// case, was sized from the cache capacity measured at construction rather than
// from the live key count, and hard-failed the moment a full-attention layer
// outgrew it -- `attention score buffer is 131072 bytes, needs 155520 for 32
// heads x 1215 keys`, i.e. decode died between 914 and 1215 tokens of context.
// AGENTS.md §8 forbids exactly that scratch.

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

// Prefill attention, one 256-thread threadgroup per (query, head), with a fully
// parallel softmax.
//
// The serial version ran the online-softmax bookkeeping on thread 0 alone: one
// serial `exp()` per key while the other 255 threads waited. That was ~25 ms of
// the ~44.7 ms prefill attention cost at 600 tokens (notes §3.1).
//
// Per 256-key chunk, with `n = min(256, remaining)`:
//   1. thread `t < n` scores key `t` (parallel dot product) and masks it to
//      -inf if it is stale or outside the sliding window;
//   2. a simdgroup-shuffle plus 8-way tree reduction finds the chunk maximum;
//   3. the running maximum advances ONCE PER CHUNK:
//      `m' = max(m, cmax)`, `alpha = exp(m - m')`;
//   4. every thread evaluates its own `exp(score - m')` in parallel -- this is
//      the step that used to be serial;
//   5. another reduction sums the weights: `l' = l * alpha + csum`;
//   6. each thread folds its dimensions: `acc = acc * alpha + sum_t w_t * v_t`.
//
// This is the standard flash-attention rescale: the running max advances once
// per 256-key chunk instead of at every key. That is not bit-identical to the
// old serial recurrence, but it is the same mathematics in a different
// association order, and it is *more* numerically stable (every weight is <= 1
// and the rescale is applied once per chunk instead of once per key).
//
// Shared memory is 1 KB (`sbuf[256]`) plus 32 bytes (`red[8]`).
//
// Reductions are simdgroup-shuffle first so only one 32-bit word per simdgroup
// crosses the barrier. `simd_shuffle_xor` is safe because the early return
// above it is threadgroup-uniform: either every lane of the threadgroup is in
// or none is.
template <int HEAD_DIM>
static void online_attention_parallel_body(
    const device half *queries,
    const device half *keys,
    const device half *values,
    const device uint *query_positions,
    const device uint *key_positions,
    device half *output,
    constant AttentionParams &params,
    uint gid,
    threadgroup float *sbuf,
    threadgroup float *red) {
    constexpr float NEG_INF = -3.402823466e+38f;
    const uint span = params.queries * params.q_heads * 256u;
    if (gid >= span) {
        return;
    }
    const uint tid = gid & 255u;
    const uint lane = tid & 31u;
    const uint sg = tid >> 5u;
    const uint head = (gid / 256u) % params.q_heads;
    const uint query = gid / (256u * params.q_heads);
    const uint groups = params.q_heads / params.kv_heads;
    const uint key_head = head / groups;
    const uint query_base = (query * params.q_heads + head) * uint(HEAD_DIM);
    const uint query_position = query_positions[query];

    constexpr uint DPT = (HEAD_DIM + 255u) / 256u;
    float accumulator[DPT];
    for (uint j = 0; j < DPT; ++j) {
        accumulator[j] = 0.0f;
    }
    float running_max = NEG_INF;
    float running_total = 0.0f;

    for (uint base = 0; base < params.key_count; base += 256u) {
        const uint n = min(256u, params.key_count - base);

        // 1. score and mask, in parallel.
        float score = NEG_INF;
        if (tid < n) {
            const uint key_position = key_positions[base + tid];
            bool live = key_position <= query_position;
            if (live && params.window != 0u && query_position - key_position >= params.window) {
                live = false;
            }
            if (live) {
                const uint key_base = ((base + tid) * params.kv_heads + key_head) * uint(HEAD_DIM);
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dot += float(queries[query_base + uint(d)]) * float(keys[key_base + uint(d)]);
                }
                score = dot * params.scale;
            }
        }
        sbuf[tid] = score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. chunk maximum: simdgroup shuffle, then an 8-way tree over `red`.
        float v = score;
        for (uint o = 16u; o > 0u; o >>= 1u) {
            v = max(v, simd_shuffle_xor(v, o));
        }
        if (lane == 0u) {
            red[sg] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float chunk_max = red[0];
        for (uint i = 1u; i < 8u; ++i) {
            chunk_max = max(chunk_max, red[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. advance the running maximum once per chunk.
        const float next_max = max(running_max, chunk_max);
        const float alpha = (running_max > NEG_INF) ? exp(running_max - next_max) : 0.0f;

        // 4. weights: every thread evaluates its own exp, in parallel.
        const float weight =
            (tid < n && score > NEG_INF && next_max > NEG_INF) ? exp(score - next_max) : 0.0f;
        sbuf[tid] = weight;

        // 5. chunk weight sum, same reduction shape.
        float s = weight;
        for (uint o = 16u; o > 0u; o >>= 1u) {
            s += simd_shuffle_xor(s, o);
        }
        if (lane == 0u) {
            red[sg] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float chunk_sum = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            chunk_sum += red[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        running_total = running_total * alpha + chunk_sum;
        running_max = next_max;

        // 6. fold this chunk into the dimension accumulators.
        //
        // `sbuf[t] == 0` for masked keys, but `values` is not zeroed for
        // unwritten slots (only `positions` is), so an unwritten slot can hold
        // NaN and `0.0f * NaN` poisons the accumulator. Skip the load entirely
        // instead: `weight == 0` makes the term exactly `+0.0` for every finite
        // value, so skipping leaves the sum bit-identical for live keys while
        // also removing the load and the multiply. Masking after the load
        // (`0.0f * loaded`) would still be a NaN and would keep the bandwidth.
        for (uint j = 0; j < DPT; ++j) {
            const uint d = tid + j * 256u;
            if (d >= uint(HEAD_DIM)) {
                continue;
            }
            float acc = accumulator[j] * alpha;
            for (uint t = 0; t < n; ++t) {
                const float w = sbuf[t];
                if (w == 0.0f) {
                    continue;
                }
                acc += w * float(values[((base + t) * params.kv_heads + key_head) * uint(HEAD_DIM) + d]);
            }
            accumulator[j] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Every thread computed an identical `running_total`, so no broadcast needed.
    const float norm = running_total > 0.0f ? 1.0f / running_total : 0.0f;
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
    threadgroup float red[8];
    online_attention_parallel_body<256>(queries, keys, values, query_positions, key_positions, output, params, tgid * 256u + gid, sbuf, red);
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
    threadgroup float red[8];
    online_attention_parallel_body<512>(queries, keys, values, query_positions, key_positions, output, params, tgid * 256u + gid, sbuf, red);
}

// ---------------------------------------------------------------------------
// Prefill attention, transposed: lane -> key, simdgroup -> query row.
//
// Geometry (the transposed one from llama.cpp's flash-attention kernel,
// `ggml/src/ggml-metal/kernels/fa_common.metal:411-454`, plus a GQA
// batching step llama.cpp does not do):
//
//   threadgroup   32 * NSG threads, NSG simdgroups.
//   simdgroup sg  owns ONE (query, query-head) row. Query heads are taken as
//                 `head = key_head * GROUPS + sg`, so all NSG rows of a
//                 threadgroup share ONE kv head and therefore read the same K
//                 and V. This is the win over llama.cpp, which puts one
//                 threadgroup per *query* head (`fa_common.metal:131-132`) and
//                 so re-reads the kv cache once per query head. Here the re-read
//                 factor drops from `q_heads` to `q_heads / (32*NSG)` = 1.
//                 Gxma: 2 query heads per kv head on the 50 sliding layers
//                 (H=256, NSG=2) and 8 on the 10 full layers (H=512, NSG=8).
//   lane          owns exactly one key inside a TILE-sized key tile, and
//                 VPT=8 consecutive output dimensions of its row.
//
// Because the simdgroup -- not the threadgroup -- holds `m`/`s` for its row, and
// every lane of that simdgroup shares them, the online-softmax recurrence is
// two hardware cross-lane reductions and nothing else:
//
//     m = simd_max(m, score_lane);
//     s = s * exp(m_old - m) + simd_sum(exp(score_lane - m));
//
// `m` and `s` come out uniform across the 32 lanes, so the O rescale is
// register-local and the score never crosses a simdgroup. The ONLY thing the
// threadgroup has to agree on is the weight vector: the PV phase wants
// lane -> dimension (that is the coalesced V read), while the QK phase produced
// lane -> key, so the weights have to be broadcast. That is a double-buffered
// `wbuf` plus ONE barrier per key tile, against five per 256 keys before.
//
// Scratch is 2 * NSG * 32 floats = 256 * NSG bytes (256 B .. 2 KB) and is
// independent of `key_count`: there is no mask buffer and no score matrix. The
// sliding-window test is arithmetic on values already in registers
// (`k_abs <= q_abs && (window == 0 || q_abs - k_abs < window)`), where
// llama.cpp materialises a 128 MiB f16 mask at L=8192
// (`src/llama-graph.cpp:977`) -- which the project forbids.
//
// A masked key is SKIPPED, never multiplied by zero: `values` is only written
// for live slots, so an unwritten slot can hold NaN and `0.0f * NaN` is NaN.
// The skip is warp-uniform (the weight is a broadcast), so it costs a branch and
// not divergence.
//
// Widen the two `half`s packed in one 32-bit word to a `float2`. `half2` is a
// 4-byte load, so four of them are one 16-byte `uint4` per lane per key.
static inline float2 unpack_half2(uint packed) {
    const half2 pair = as_type<half2>(packed);
    return float2(float(pair.x), float(pair.y));
}

// HEAD_DIM must be a multiple of 32 * VPT so that every lane owns VPT
// consecutive dims and a warp's V read is one contiguous 32*VPT*2-byte run.
//
// `PARTIALS` turns the same body into the decode kernel (flash-decoding over a
// split KV axis). The online softmax, the dot product and the PV fold are
// literally the same code, so prefill and decode cannot drift apart; only the
// key range and the write-out differ:
//
//   key range   PARTIALS: `[split * keys_per_split, min(key_count, +keys_per_split))`
//               prefill: `[0, key_count)`   (identical loop, `splits == 0`)
//   tgid        PARTIALS: `key_head = tgid % kv_heads`, `split = tgid / kv_heads`
//               prefill: `query = tgid / kv_heads`, `key_head = tgid % kv_heads`
//   output      PARTIALS: one unnormalised (m, l, O) row per split in `partials`
//               prefill: the normalised output row
//
// Splitting must not change the scan: the host derives `key_count` from
// `GpuKvCache::key_count(full_attention)`, which for the sliding rings is
// `min(next_position, capacity)` -- exactly the live window -- and each split
// stops at `key_count`, so no split ever reads past the live window and the
// `key_position` / `window` tests still drop anything stale.
template <int HEAD_DIM, int NSG, bool PARTIALS>
static void attention_gqa_body(
    const device half *queries,
    const device half *keys,
    const device half *values,
    const device uint *query_positions,
    const device uint *key_positions,
    device half *output,
    device float *partials,
    constant AttentionParams &params,
    uint tid,
    uint tgid,
    threadgroup float *wbuf,
    threadgroup float *qfbuf) {
    static_assert(HEAD_DIM % 256 == 0, "HEAD_DIM must be a multiple of 32 * 8");
    // -FLT_MAX/2 rather than -inf, exactly as llama.cpp does: it keeps
    // `m_old - m` finite for a fully dead tile, so `exp()` can never return NaN
    // and a live key always has a finite reference point.
    constexpr float NEG = -1.701411733864e+38f;
    constexpr int VPT = 8;
    constexpr int BLOCKS = HEAD_DIM / (32 * VPT);

    const uint lane = tid & 31u;
    const uint sg = tid >> 5u;
    // `groups` is `q_heads / kv_heads`; the dispatch picks NSG >= groups and
    // sizes the threadgroup at 32 * NSG, so this is false for every simdgroup.
    // It must stay a *barrier-safe* test: a simdgroup that returned early would
    // strand the threads still waiting at `wbuf`'s barrier.
    const uint groups = params.q_heads / params.kv_heads;
    const bool active = sg < groups;
    const uint head_in_kv = active ? sg : 0u;

    const uint query = PARTIALS ? 0u : tgid / params.kv_heads;
    const uint key_head = PARTIALS ? tgid - (tgid / params.kv_heads) * params.kv_heads
                                   : tgid - query * params.kv_heads;
    const uint split = PARTIALS ? tgid / params.kv_heads : 0u;
    const uint head = key_head * groups + head_in_kv;
    const uint query_base = (query * params.q_heads + head) * uint(HEAD_DIM);
    const uint q_abs = query_positions[query];
    const uint kv_stride = params.kv_heads * uint(HEAD_DIM);
    const uint head_base = key_head * uint(HEAD_DIM);
    const device half *qrow = queries + query_base;
    // This split's slice of the live window. `min(key_count, ...)` is what keeps
    // a split from running off the end of the cache when `key_count` is not a
    // multiple of `keys_per_split`.
    const uint range_begin = PARTIALS ? split * params.keys_per_split : 0u;
    const uint range_end = PARTIALS ? min(params.key_count, range_begin + params.keys_per_split)
                                    : params.key_count;

    // This lane's slice of its own row: VPT consecutive dims, BLOCKS of them.
    const uint lane_dim = lane * uint(VPT);
    float4 acc[2 * BLOCKS];
    for (int b = 0; b < BLOCKS; ++b) {
        acc[2 * b + 0] = float4(0.0f);
        acc[2 * b + 1] = float4(0.0f);
    }
    // Replicated across the 32 lanes of the simdgroup and uniform over the
    // whole threadgroup column: no lane ever needs another's copy.
    float m = NEG;
    float s = 0.0f;

    // Hoist Q's half->float conversion out of the per-key loop. `qrow` is
    // loop-invariant -- it is this threadgroup's single query row -- yet the
    // scalar QK loop reloaded and reconverted all HEAD_DIM of it for every
    // key, which is half of the loop's instructions and the single largest
    // term in it. All 32 lanes want the same row, so one threadgroup copy,
    // converted once, is read back as plain floats. The values and the
    // accumulation order into `dot` are unchanged, so this is bit-exact.
    threadgroup float *qf = qfbuf + sg * uint(HEAD_DIM);
    for (int c = int(lane); c < HEAD_DIM / 4; c += 32) {
        const uint2 qraw = *reinterpret_cast<const device uint2 *>(qrow + 4 * c);
        const float2 h0 = unpack_half2(qraw.x);
        const float2 h1 = unpack_half2(qraw.y);
        qf[4 * c + 0] = h0.x;
        qf[4 * c + 1] = h0.y;
        qf[4 * c + 2] = h1.x;
        qf[4 * c + 3] = h1.y;
    }
    // Barrier-safe: an inactive simdgroup writes its own (unused) slot and
    // must not return early, or the threads below strand here.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *my_weights = wbuf + sg * 32u;

    uint phase = 0u;
    for (uint base = range_begin; base < range_end; base += 32u) {
        const uint n = min(32u, range_end - base);
        const uint k = base + lane;

        // 1. score this lane's key. The window test is arithmetic, not a
        //    materialised mask, and a dead key never touches `keys` at all.
        float score = NEG;
        bool live = active && (k < range_end);
        if (live) {
            const uint k_abs = key_positions[k];
            live = (k_abs <= q_abs)
                && (params.window == 0u || (q_abs - k_abs) < params.window);
        }
        if (live) {
            const device half *krow = keys + k * kv_stride + head_base;
            float dot = 0.0f;
            for (int i = 0; i < HEAD_DIM / 4; ++i) {
                const uint2 kraw = *reinterpret_cast<const device uint2 *>(krow + 4 * i);
                const float2 kh0 = unpack_half2(kraw.x);
                const float2 kh1 = unpack_half2(kraw.y);
                const float4 qv =
                    *reinterpret_cast<const threadgroup float4 *>(qf + 4 * i);
                // Accumulated in d order, so this dot product is bit-identical
                // to the 256-wide scalar loop it replaces.
                dot += qv.x * kh0.x;
                dot += qv.y * kh0.y;
                dot += qv.z * kh1.x;
                dot += qv.w * kh1.y;
            }
            score = dot * params.scale;
        }

        // 2. online softmax, two hardware reductions, no threadgroup traffic.
        const float m_old = m;
        // `simd_max` is a unary cross-lane reduction; `m` is already uniform
        // across the simdgroup, so folding it in first is exact.
        m = simd_max(max(m, score));
        const float ms = exp(m_old - m);
        const float w = live ? exp(score - m) : 0.0f;
        s = s * ms + simd_sum(w);
        my_weights[phase * (NSG * 32u) + lane] = w;

        // 3. one barrier per tile. Double buffering is what makes a single
        //    barrier sufficient: the next tile writes the OTHER buffer, so no
        //    lane can overwrite a slot a peer is still reading, and the barrier
        //    that publishes tile t+1 orders every lane's reads of tile t before
        //    tile t+2 reuses the buffer.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4. PV. Lane owns VPT consecutive dims, so a warp's read of one key is
        //    32*VPT contiguous halves. `w` is warp-uniform: the branch never
        //    diverges, and skipping is what keeps `0.0f * NaN` out of `acc`.
        const threadgroup float *tile = my_weights + phase * (NSG * 32u);
        for (int b = 0; b < BLOCKS; ++b) {
            const uint d = lane_dim + uint(b) * 32u * uint(VPT);
            float4 lo = acc[2 * b + 0] * ms;
            float4 hi = acc[2 * b + 1] * ms;
            for (uint t = 0; t < n; ++t) {
                const float wt = tile[t];
                if (wt == 0.0f) {
                    continue;
                }
                const device half *vrow =
                    values + (base + t) * kv_stride + head_base + d;
                const uint4 raw = *reinterpret_cast<const device uint4 *>(vrow);
                const float2 vlo0 = unpack_half2(raw.x);
                const float2 vlo1 = unpack_half2(raw.y);
                const float2 vhi0 = unpack_half2(raw.z);
                const float2 vhi1 = unpack_half2(raw.w);
                lo += wt * float4(vlo0, vlo1);
                hi += wt * float4(vhi0, vhi1);
            }
            acc[2 * b + 0] = lo;
            acc[2 * b + 1] = hi;
        }
        phase ^= 1u;
    }

    if (!active) {
        return;
    }
    if (PARTIALS) {
        // Unnormalised partial for this split, laid out as
        //   [m, l, O[head_dim]] at (split * q_heads + head) * (head_dim + 2)
        // so the combine reads a warp's 32 dimensions as 128 contiguous bytes.
        // An empty slice falls out of the loop above with m = NEG, s = 0 and a
        // zero accumulator, which the combine weights to zero. Nothing else has
        // to clear the buffer: every split that is read is a split that wrote.
        device float *row = partials + (split * params.q_heads + head) * (params.head_dim + 2u);
        if (lane == 0u) {
            row[0] = m;
            row[1] = s;
        }
        for (int b = 0; b < BLOCKS; ++b) {
            const uint d = lane_dim + uint(b) * 32u * uint(VPT);
            const float4 lo = acc[2 * b + 0];
            const float4 hi = acc[2 * b + 1];
            *reinterpret_cast<device float4 *>(row + 2u + d + 0u) = lo;
            *reinterpret_cast<device float4 *>(row + 2u + d + 4u) = hi;
        }
        return;
    }
    const float norm = s > 0.0f ? 1.0f / s : 0.0f;
    for (int b = 0; b < BLOCKS; ++b) {
        const uint d = query_base + lane_dim + uint(b) * 32u * uint(VPT);
        device half *orow = output + d;
        const float4 lo = acc[2 * b + 0] * norm;
        const float4 hi = acc[2 * b + 1] * norm;
        *reinterpret_cast<device half2 *>(orow + 0) = half2(half(lo.x), half(lo.y));
        *reinterpret_cast<device half2 *>(orow + 2) = half2(half(lo.z), half(lo.w));
        *reinterpret_cast<device half2 *>(orow + 4) = half2(half(hi.x), half(hi.y));
        *reinterpret_cast<device half2 *>(orow + 6) = half2(half(hi.z), half(hi.w));
    }
}

// Merge the per-split (m, l, O) partials. This is llama.cpp's second pass
// (`ggml-metal` reads `dst1[rid*(2*NWG) + 2*iwg]` back for exactly this):
//
//     M     = max_s m_s
//     den   = sum_s exp(m_s - M) * l_s
//     out[d]= sum_s exp(m_s - M) * O_s[d] / den
//
// `m_s <= M`, so every factor is in [0, 1] and an empty split (m_s = NEG) weighs
// nothing. One thread per (head, dimension) and no threadgroup memory at all.
//
// HEAD_DIM only names the two entry points (the host picks by head_dim); the
// indexing is driven by `params.head_dim`, so one body serves both.
template <int HEAD_DIM>
static void attention_decode_combine_body(
    const device float *partials,
    device half *output,
    constant AttentionParams &params,
    uint gid) {
    const uint dimension = gid % params.head_dim;
    const uint head = gid / params.head_dim;
    if (head >= params.q_heads) {
        return;
    }
    const uint stride = params.head_dim + 2u;
    float global_max = -1.701411733864e+38f;
    for (uint i = 0; i < params.splits; ++i) {
        global_max = max(global_max, partials[(i * params.q_heads + head) * stride]);
    }
    float numerator = 0.0f;
    float denominator = 0.0f;
    for (uint i = 0; i < params.splits; ++i) {
        const device float *row = partials + (i * params.q_heads + head) * stride;
        const float factor = exp(row[0] - global_max);
        denominator += factor * row[1];
        numerator += factor * row[2u + dimension];
    }
    const float value = denominator > 0.0f ? numerator / denominator : 0.0f;
    output[head * params.head_dim + dimension] = half(value);
}

#define KAPLUMBA_FA_GQA(NAME, HEAD_DIM, NSG)                                   \
    kernel void NAME(                                                          \
        const device half *queries [[buffer(0)]],                              \
        const device half *keys [[buffer(1)]],                                 \
        const device half *values [[buffer(2)]],                               \
        const device uint *query_positions [[buffer(3)]],                      \
        const device uint *key_positions [[buffer(4)]],                         \
        device half *output [[buffer(5)]],                                     \
        constant AttentionParams &params [[buffer(6)]],                        \
        uint tid [[thread_position_in_threadgroup]],                           \
        uint tgid [[threadgroup_position_in_grid]]) {                          \
        threadgroup float wbuf[2 * (NSG) * 32];                                \
        threadgroup float qfbuf[(NSG) * (HEAD_DIM)];                           \
        attention_gqa_body<HEAD_DIM, NSG, false>(queries, keys, values,        \
            query_positions, key_positions, output, (device float *)output,    \
            params, tid, tgid, wbuf, qfbuf);                                   \
    }

// Decode: same body, one workgroup per (split, kv head), partial output. The
// partial buffer takes the output slot's index because the kernel has no
// normalised output to write.
#define KAPLUMBA_FA_DECODE(NAME, HEAD_DIM, NSG)                                \
    kernel void NAME(                                                          \
        const device half *queries [[buffer(0)]],                              \
        const device half *keys [[buffer(1)]],                                 \
        const device half *values [[buffer(2)]],                               \
        const device uint *query_positions [[buffer(3)]],                      \
        const device uint *key_positions [[buffer(4)]],                        \
        device float *partials [[buffer(5)]],                                  \
        constant AttentionParams &params [[buffer(6)]],                        \
        uint tid [[thread_position_in_threadgroup]],                           \
        uint tgid [[threadgroup_position_in_grid]]) {                          \
        threadgroup float wbuf[2 * (NSG) * 32];                                \
        threadgroup float qfbuf[(NSG) * (HEAD_DIM)];                           \
        attention_gqa_body<HEAD_DIM, NSG, true>(queries, keys, values,         \
            query_positions, key_positions, (device half *)partials, partials, \
            params, tid, tgid, wbuf, qfbuf);                                   \
    }

#define KAPLUMBA_FA_COMBINE(NAME, HEAD_DIM)                                    \
    kernel void NAME(                                                          \
        const device float *partials [[buffer(0)]],                            \
        device half *output [[buffer(1)]],                                     \
        constant AttentionParams &params [[buffer(2)]],                        \
        uint gid [[thread_position_in_grid]]) {                                \
        attention_decode_combine_body<HEAD_DIM>(partials, output, params, gid); \
    }

// NSG must equal `q_heads / kv_heads` for the model's two layer shapes, so no
// simdgroup idles: 2 on the sliding layers (32/16) and 8 on the full layers (32/4).
// 1 and 4 are the other GQA ratios in the wild (Llama-3 8B is 32/8 = 4).
KAPLUMBA_FA_GQA(online_attention_gqa_256_1, 256, 1)
KAPLUMBA_FA_GQA(online_attention_gqa_256_2, 256, 2)
KAPLUMBA_FA_GQA(online_attention_gqa_256_4, 256, 4)
KAPLUMBA_FA_GQA(online_attention_gqa_256_8, 256, 8)
KAPLUMBA_FA_GQA(online_attention_gqa_512_1, 512, 1)
KAPLUMBA_FA_GQA(online_attention_gqa_512_2, 512, 2)
KAPLUMBA_FA_GQA(online_attention_gqa_512_4, 512, 4)
KAPLUMBA_FA_GQA(online_attention_gqa_512_8, 512, 8)
KAPLUMBA_FA_DECODE(attention_decode_split_256_1, 256, 1)
KAPLUMBA_FA_DECODE(attention_decode_split_256_2, 256, 2)
KAPLUMBA_FA_DECODE(attention_decode_split_256_4, 256, 4)
KAPLUMBA_FA_DECODE(attention_decode_split_256_8, 256, 8)
KAPLUMBA_FA_DECODE(attention_decode_split_512_1, 512, 1)
KAPLUMBA_FA_DECODE(attention_decode_split_512_2, 512, 2)
KAPLUMBA_FA_DECODE(attention_decode_split_512_4, 512, 4)
KAPLUMBA_FA_DECODE(attention_decode_split_512_8, 512, 8)
KAPLUMBA_FA_COMBINE(attention_decode_combine_256, 256)
KAPLUMBA_FA_COMBINE(attention_decode_combine_512, 512)
