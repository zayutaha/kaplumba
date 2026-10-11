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
    // Prefill key-range passes (OQ9): this dispatch covers
    // `[pass_begin, min(key_count, pass_begin + pass_keys))`. `pass_keys == 0`
    // is the one-dispatch behaviour and scans `[0, key_count)`.
    uint pass_begin;
    uint pass_keys;
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
// A `DIVIDE` template parameter was tried here and reverted, because the
// single-pass decode kernel (`attention_decode_single_*`) has to be bit-identical
// to `attention_decode_split` with `splits == 1` followed by
// `attention_decode_combine`, and the combine's epilogue is a *division*
// (`denominator > 0.0f ? numerator / denominator : 0.0f`) where this loop ends in
// a *reciprocal multiply* (`acc * (1 / s)`) -- different f32 operations.
// MEASURED (`crates/engine/tests/attention_single_pass_bit_exact.rs`,
// `the_reciprocal_multiply_is_absorbed_by_the_fp16_store`): the two epilogues
// produce **bit-identical fp16 on 294912 of 294912 elements**, both model shapes
// and twelve key counts. The fp16 store absorbs it, so the parameter bought
// nothing and was removed instead of carried as an unjustified code path. The
// note stays because the reasoning is not obvious and the next person to
// "correct" the epilogue should know it was tried and measured.
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
    const uint d4 = (gid % (params.head_dim / 4u)) * 4u;
    const uint head = gid / (params.head_dim / 4u);
    if (head >= params.q_heads) {
        return;
    }
    const uint stride = params.head_dim + 2u;
    float global_max = -1.701411733864e+38f;
    for (uint i = 0; i < params.splits; ++i) {
        global_max = max(global_max, partials[(i * params.q_heads + head) * stride]);
    }
    float4 num4 = float4(0.0f);
    float den = 0.0f;
    for (uint i = 0; i < params.splits; ++i) {
        const device float *row = partials + (i * params.q_heads + head) * stride;
        const float factor = exp(row[0] - global_max);
        den += factor * row[1];
        const device float4 *val4 = reinterpret_cast<const device float4 *>(row + 2u + d4);
        num4 += factor * (*val4);
    }
    const float inv_den = den > 0.0f ? 1.0f / den : 0.0f;
    device half4 *out4 = reinterpret_cast<device half4 *>(output + head * params.head_dim + d4);
    *out4 = half4(num4 * inv_den);
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

// Single-pass decode: one threadgroup per (query, kv head), the whole live KV
// window in one threadgroup, normalised output written in place.
//
// This is `KAPLUMBA_FA_GQA` under a second name, deliberately. With `splits == 1`
// the split kernel's key range is `[0, key_count)` -- the range this body scans --
// and `attention_decode_combine` then reduces a single partial, so reusing the
// body verbatim is what makes the two bit-identical rather than merely
// mathematically equivalent. Reusing it is also why the epilogue is the
// reciprocal multiply the prefill kernel has rather than the combine's division:
// measured, the fp16 store absorbs the difference (see the `DIVIDE` note above).
//
// The cost of dropping the combine pass is `kv_heads` workgroups instead of
// `splits * kv_heads`, i.e. the whole `splits` factor of parallelism. The saving
// is one dispatch and one `splits * q_heads * (head_dim + 2)` float round trip
// through the partials buffer. Which side wins is a measurement, not an
// argument: see `ATTENTION_SINGLE_PASS_MAX_KEYS` in `attention.rs`.
#define KAPLUMBA_FA_SINGLE(NAME, HEAD_DIM, NSG)                                \
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
// Single-pass decode. Only the two shapes the model uses are instantiated
// (256/NSG=2 on the 50 sliding layers, 512/NSG=8 on the 10 full ones), because
// the host only ever selects it for those: `decode_single_pass_max_keys` is
// reached from `encode_online_attention_split`, which has already resolved
// `gqa_index` and falls back to the prefill kernels for anything else. Adding
// the other six would be six more pipeline compilations for dispatches nothing
// can reach.
KAPLUMBA_FA_SINGLE(attention_decode_single_256_2, 256, 2)
KAPLUMBA_FA_SINGLE(attention_decode_single_512_8, 512, 8)
KAPLUMBA_FA_COMBINE(attention_decode_combine_256, 256)
KAPLUMBA_FA_COMBINE(attention_decode_combine_512, 512)

// ===========================================================================
// T1 (spec/prefill-120tps): simdgroup-MMA QK scoring, prefill path only.
//
// This body is `attention_gqa_body` with step 1 (the per-lane scalar dot at
// :838-850) replaced by 8x8x8 `simdgroup_multiply_accumulate` tiles. Steps 2-4
// (online softmax, double-buffered weight tile, PV) and the epilogue are copied
// verbatim, so the barrier count per 32-key tile is unchanged in the fast path
// and the weight/PV protocol is byte-for-byte the scalar one. The scalar body
// is not modified, so every decode kernel (`KAPLUMBA_FA_DECODE`,
// `KAPLUMBA_FA_SINGLE`) is untouched by construction.
//
// Fragment mapping (verified empirically by
// `crates/engine/tests/attention_qk_mma_matches_scalar.rs`):
//
//   A = K tile, device load, no transpose   A[key][d] = K[k0+key][d8+d]
//       (the proven `linear.metal:10391` pattern)
//   B = Q, threadgroup load, TRANSPOSE      B[d][row] = qhstage[row][d8+d]
//   C = A * B                               C[key][row] = dot(K, Q) pre-scale
//
// M is always the full 8 keys and N the full 8 rows of the threadgroup's wave,
// so neither axis wastes lanes -- the 8x8 instruction executes N=8 whether or
// not the lanes hold real rows, which is why each simdgroup computes all rows'
// scores and keeps column `sg` (rows >= NSG in `qhstage` are deliberately
// unwritten garbage; their C columns are discarded and never read).
//
// Extraction stays in registers: the fp32 8x8 fragment lane map is the one
// the working GEMM uses (linear.metal:10208-10209 + store macro :10427-10433)
// -- lane t holds elements (lane_fm, lane_fn) and (lane_fm, lane_fn + 1) with
// lane_fm = (t >= 16 ? 4 : 0) + ((t >> 1) & 3) and
// lane_fn = (t & 8 ? 4 : 0) + ((t & 1) << 1) -- so key m's value for row `sg`
// sits on one lane (see the formula at the extraction site), one simd_shuffle
// away, and lands directly in the scalar layout `lane = key within the 32-tile`
// that steps 2-4 expect. No score buffer, no publish, no extra barrier.
//
// The tail sub-tile (k0 + 8 > range_end, at most once per pass and uniform
// across the threadgroup) stages guarded rows into `kbuf` instead of reading
// past `key_count`, so a dead key still never touches `keys` out of bounds.
// ===========================================================================
template <int HEAD_DIM, int NSG>
static void attention_gqa_mma_body(
    const device half *queries,
    const device half *keys,
    const device half *values,
    const device uint *query_positions,
    const device uint *key_positions,
    device half *output,
    constant AttentionParams &params,
    uint tid,
    uint tgid,
    threadgroup float *wbuf,
    threadgroup half *qhstage,
    threadgroup half *kbuf,
    threadgroup float *sstage) {
    static_assert(HEAD_DIM % 256 == 0, "HEAD_DIM must be a multiple of 32 * 8");
    // Same constant as the scalar body: finite, so `m_old - m` stays finite for
    // a fully dead tile and `exp()` can never return NaN.
    constexpr float NEG = -1.701411733864e+38f;
    constexpr int VPT = 8;
    constexpr int BLOCKS = HEAD_DIM / (32 * VPT);

    const uint lane = tid & 31u;
    const uint sg = tid >> 5u;
    // Same barrier-safe `active` contract as the scalar body: an inactive
    // simdgroup writes its own slot, runs every loop and every barrier, and
    // only returns at the epilogue.
    const uint groups = params.q_heads / params.kv_heads;
    const bool active = sg < groups;
    const uint head_in_kv = active ? sg : 0u;

    const uint query = tgid / params.kv_heads;
    const uint key_head = tgid - query * params.kv_heads;
    const uint head = key_head * groups + head_in_kv;
    const uint query_base = (query * params.q_heads + head) * uint(HEAD_DIM);
    const uint q_abs = query_positions[query];
    const uint kv_stride = params.kv_heads * uint(HEAD_DIM);
    const uint head_base = key_head * uint(HEAD_DIM);
    const device half *qrow = queries + query_base;
    const uint range_end = params.key_count;

    // This lane's slice of its own row: VPT consecutive dims, BLOCKS of them.
    const uint lane_dim = lane * uint(VPT);
    float4 acc[2 * BLOCKS];
    for (int b = 0; b < BLOCKS; ++b) {
        acc[2 * b + 0] = float4(0.0f);
        acc[2 * b + 1] = float4(0.0f);
    }
    float m = NEG;
    float s = 0.0f;

    // Stage this simdgroup's Q row as halves: the transposed B load below reads
    // qhstage as a half matrix, so unlike the scalar body's float `qfbuf` there
    // is no half->float conversion here at all (the MMA consumes halves
    // directly). Rows [NSG, 8) exist so the 8-row load is in bounds; they are
    // never written and their C columns are discarded.
    threadgroup half *qslot = qhstage + sg * uint(HEAD_DIM);
    for (uint c = lane * 8u; c < uint(HEAD_DIM); c += 256u) {
        *reinterpret_cast<threadgroup uint4 *>(qslot + c) =
            *reinterpret_cast<const device uint4 *>(qrow + c);
    }
    // Barrier-safe exactly as scalar :816-818: every simdgroup writes a slot
    // and every simdgroup's B load below reads all NSG slots.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *my_weights = wbuf + sg * 32u;

    uint phase = 0u;
    for (uint base = 0; base < range_end; base += 32u) {
        const uint n = min(32u, range_end - base);
        const uint k = base + lane;

        // 1. score this lane's key with four 8-key MMA sub-tiles. `score` holds
        //    the unnormalised dot until the live/scale step below, mirroring
        //    scalar's `dot` at :837-851.
        //
        // T2 fix (spec/prefill-120tps): the T1 body had EVERY simdgroup compute
        // all four sub-tiles -- identical inputs, identical 8x8 `dots` -- so
        // the tensor work AND the strided K fragment loads were `NSG` x
        // redundant. That measured 1.7-5x SLOWER than the scalar QK it
        // replaces (micro-bench `attention_qk_mma_bench`). Below, sub-tile
        // `sub` is owned by exactly the simdgroup with `sub % NSG == sg`; its
        // 8x8 result is published to a 1 KB shared score tile and every lane
        // reads its own row back. Total MMA work drops from `4*NSG` sub-tiles
        // to exactly 4, at the cost of one extra barrier per full tile.
        //
        // The final (partial) tile cannot use ownership -- its staging barrier
        // must be taken by every simdgroup -- so it keeps T1's all-simdgroup
        // path. It occurs at most once per pass, so its NSG x cost is noise.
        float score = NEG;
        const bool partial_tile = (base + 32u > range_end);
        if (!partial_tile) {
            for (uint sub = sg; sub < 4u; sub += uint(NSG)) {
                const uint k0 = base + sub * 8u;
                simdgroup_float8x8 dots =
                    make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                for (int d8 = 0; d8 < HEAD_DIM; d8 += 8) {
                    simdgroup_half8x8 kf, qf;
                    simdgroup_load(kf, keys + k0 * kv_stride + head_base + d8,
                                   kv_stride);
                    simdgroup_load(qf, qhstage + d8, HEAD_DIM, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(dots, kf, qf, dots);
                }
                // Publish the 8x8 tile: M = key within the sub, N = query row.
                // Same lane map as the working GEMM store (linear.metal:10427):
                //   fm = (lane>=16?4:0)+((lane>>1)&3)
                //   fn = (lane&8?4:0)+((lane&1)<<1)
                // thread_elements()[0] sits at (fm, fn), [1] at (fm, fn+1).
                const uint fm = ((lane >= 16u) ? 4u : 0u) + ((lane >> 1u) & 3u);
                const uint fn = ((lane & 8u) ? 4u : 0u) + ((lane & 1u) << 1u);
                sstage[(sub * 8u + fm) * 8u + fn] = dots.thread_elements()[0];
                sstage[(sub * 8u + fm) * 8u + fn + 1u] = dots.thread_elements()[1];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            score = sstage[lane * 8u + sg];
        } else {
            for (uint sub = 0u; sub < 4u; ++sub) {
                const uint k0 = base + sub * 8u;
                if (k0 >= range_end) {
                    // Uniform across the whole threadgroup (base and range_end
                    // are loop-invariant), so the break cannot strand a barrier.
                    break;
                }
                simdgroup_float8x8 dots =
                    make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                if (k0 + 8u <= range_end) {
                    for (int d8 = 0; d8 < HEAD_DIM; d8 += 8) {
                        simdgroup_half8x8 kf, qf;
                        simdgroup_load(kf, keys + k0 * kv_stride + head_base + d8,
                                       kv_stride);
                        simdgroup_load(qf, qhstage + d8, HEAD_DIM, ulong2(0, 0), true);
                        simdgroup_multiply_accumulate(dots, kf, qf, dots);
                    }
                } else {
                    // Tail: stage [k0, range_end) with zeros past the end so no
                    // out-of-bounds device read is issued; the scalar `live`
                    // test below forces those positions to NEG.
                    for (uint i = lane; i < 8u * uint(HEAD_DIM); i += 32u) {
                        const uint row = i / uint(HEAD_DIM);
                        const uint col = i % uint(HEAD_DIM);
                        const uint pos = k0 + row;
                        kbuf[i] = pos < range_end
                            ? keys[pos * kv_stride + head_base + col]
                            : half(0.0f);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (int d8 = 0; d8 < HEAD_DIM; d8 += 8) {
                        simdgroup_half8x8 kf, qf;
                        simdgroup_load(kf, kbuf + d8, HEAD_DIM);
                        simdgroup_load(qf, qhstage + d8, HEAD_DIM, ulong2(0, 0), true);
                        simdgroup_multiply_accumulate(dots, kf, qf, dots);
                    }
                }
                // Lane map, decoded from the working GEMM store rather than
                // guessed: linear.metal:10208-10209 defines
                //   lane_fm = (lane >= 16 ? 4 : 0) + ((lane >> 1) & 3)   // M index
                //   lane_fn = (lane & 8 ? 4 : 0) + ((lane & 1) << 1)      // N, even
                // and linear.metal:10427-10433 stores thread_elements()[0] at
                // (lane_fm, lane_fn) and [1] at (lane_fm, lane_fn + 1).
                // Here M = key (A's rows), N = query row (B's cols), so key `m`,
                // row `sg` lives on lane
                //   src = (m >= 4 ? 16 : 0) + ((sg & ~1) >= 4 ? 8 : 0)
                //        + 2 * (m & 3) + (sg & 1)
                // at element index (sg & 1). Out-of-range lanes underflow `m`;
                // src stays <= 31 by construction and their result is discarded
                // by the guard below.
                const float mine = dots.thread_elements()[sg & 1u];
                const uint m = lane - sub * 8u;
                const uint pair = sg >> 1u;
                const uint src = ((m >= 4u) ? 16u : 0u)
                    + ((pair >= 2u) ? 8u : 0u)
                    + 2u * (m & 3u)
                    + (pair & 1u);
                const float v = simd_shuffle(mine, ushort(src));
                if (lane >= sub * 8u && lane < sub * 8u + 8u) {
                    score = v;
                }
            }
        }

        // Live-window test and scale: verbatim arithmetic from scalar
        // :828-834 and :851, with the MMA dot standing in for `dot`.
        bool live = active && (k < range_end);
        if (live) {
            const uint k_abs = key_positions[k];
            live = (k_abs <= q_abs)
                && (params.window == 0u || (q_abs - k_abs) < params.window);
        }
        score = live ? score * params.scale : NEG;

        // 2. online softmax -- copied verbatim from scalar :854-862.
        const float m_old = m;
        m = simd_max(max(m, score));
        const float ms = exp(m_old - m);
        const float w = live ? exp(score - m) : 0.0f;
        s = s * ms + simd_sum(w);
        my_weights[phase * (NSG * 32u) + lane] = w;

        // 3. one barrier per tile, double-buffered -- scalar :864-869.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4. PV -- copied verbatim from scalar :871-896.
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
    // Reciprocal-multiply epilogue, same as the scalar prefill body :924-934.
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

// ===========================================================================
// T3 (`spec/prefill-24k-90tps`) -- flash-style prefill attention for the
// full-attention shape only: HEAD_DIM 512, q_heads/kv_heads == NSG == 8.
//
// `attention_gqa_mma_body` left three structural costs on the table, all three
// read from its source and all three measured here as the reason the 10 full
// layers cost 176.7 s of a 24k prefill at **1.06 TFLOPS**:
//
//   1. QK sub-tiles were owned `sub = sg; sub < 4; sub += NSG`, so with NSG=8
//      the simdgroups 4..7 issued **zero MMAs** and only published/read
//      `sstage`.
//   2. PV was fully scalar, and every one of the 8 simdgroups re-read and
//      re-converted the whole 512-dim V row for every key (each q head's
//      output needs every V dim). 8x redundant device reads and 8x redundant
//      `cvt` per key.
//   3. The Q fragment was reloaded from `qhstage` on all 64 `d8` steps of every
//      key tile although it is key-tile-invariant.
//
// This body fixes all three: a 64-key tile (eight 8-key m-tiles, one owned per
// simdgroup) for QK, and the PV fold as a second `simdgroup_multiply_accumulate`
// -- `P[8 heads x 64 keys] x V[64 keys x 512 dims]` with V loaded straight from
// the device at stride `kv_stride`. No threadgroup V tile: 64 keys x 512 dims of
// halves is 64 KB, twice the 32 KB cliff.
//
// Layouts (all verified against the 8x8 fp32 fragment map the working GEMM
// uses, `linear.metal:10208-10209` + the store at `:10427-10433`):
//   fm = (lane >= 16 ? 4 : 0) + ((lane >> 1) & 3)      // M index = head
//   fn = (lane & 8 ? 4 : 0) + ((lane & 1) << 1)        // N index = dim, even
//   thread_elements()[0] sits at (fm, fn), [1] at (fm, fn + 1).
//
// Query blocking (T4, `spec/prefill-24k-90tps`): a threadgroup owns `BQ`
// consecutive queries of ONE KV head, so every K/V key it streams is reused by
// `BQ` queries. At BQ = 1 each query re-reads its whole causal prefix, which at
// L = 24k is 23.5 TB over the 10 full layers -- ~86 s of pure 273 GB/s streaming
// against ~31 s of tensor work, i.e. K/V bandwidth is the binding constraint.
// BQ = 2 halves it. See implementation_notes.md section 5.
//
// Threadgroup memory at HEAD_DIM=512, NSG=8, BQ=2:
//   qstage  BQ*8*512 halfs = 16 KB (Q rows, one per (query, simdgroup))
//   sth     BQ*64*8 floats =  4 KB (QK scores, [query][key in tile][head])
//   wq      BQ*8*64 halfs  =  2 KB (P, [query][head][key in tile], ld = 64)
//   tail    8*512 halfs    =  8 KB (straddling K or V 8-key step; reused)
//   msb[BQ*8] + ssum[BQ*8] floats = 128 B
//   total 30.1 KB, under the 32 KB cliff.
//
// `tail` is shared between the QK straddle and the PV straddle because the
// straddling m-tile is always read before the softmax barrier and the
// straddling k-step only after it. Everything is threadgroup-uniform (base,
// range_end, key_count, q0), so every barrier below is taken by all 256 threads.
// `PARTIALS` turns this into the DECODE kernel: one query row, a grid of
// `splits * kv_heads` threadgroups, each owning a contiguous key slice, and an
// unnormalised `(m, s, O)` partial per (split, head) in the layout
// `attention_decode_combine_body` already reads. The QK and PV bodies are
// unchanged -- the decode win is that the per-key `simd_sum` + online-softmax
// update of the scalar path is replaced by 8x8 simdgroup MMA, which is what
// oMLX's `sdpa_flash` does for the same reason (`omlx/patches/mimo_v2/
// sdpa_flash.py`, oMLX v0.7.0). Measured there at 1.9-3.6x per decode layer.
//
// `BQ` must be 1 in this mode: a decode step has exactly one query row, so the
// partial layout has no query axis to block over.
//
// `PASSES` is oMLX's key-range-pass mechanism (OQ9), ported from
// `omlx/utils/nax_attention.py:162-170` ("One dispatch covers key blocks
// [kb_begin, kb_end). Unless FIRST, the online-softmax state of every query row
// (fp32 O accumulator, running max, running sum) is loaded from Sin; unless LAST
// it is stored to Sout instead of normalising and writing the output. Every row
// thus runs exactly the operations of one full dispatch, in the same order
// (bit-identical output)") -- `FIRST`/`LAST` here are the runtime tests
// `pass_begin == 0` and `pass_begin + pass_keys >= key_count`.
//
// The reason oMLX has it: "one dispatch over a long key range lets its
// threadgroups drift apart until each streams its KV head from DRAM; per-slice
// dispatches keep the K/V they share in the on-chip caches" (`:30-32`), measured
// as "~85 TFLOPS at 1M keys against ~103 at 64k" (`:89-91`).
//
// The row state lives in `state`, one `(m, s, O[HEAD_DIM])` fp32 row per
// (query, q head) -- the decode partials layout with a query axis in front, so
// nothing new has to be proven about it. `acc[b][j]`'s fragment row `fm` is the
// q head (see the load-layout note above `linear.metal:10208`) and its columns
// are this simdgroup's `NPER * 8` dims (`nt = sg * NPER + j`, T4f), so the eight
// simdgroups together cover each `(query, head)` row exactly once.
template <int HEAD_DIM, int NSG, int BQ, bool PARTIALS, bool PASSES>
static void attention_prefill_flash_body(
    const device half *queries,
    const device half *keys,
    const device half *values,
    const device uint *query_positions,
    const device uint *key_positions,
    device half *output,
    device float *partials,
    device float *state,
    constant AttentionParams &params,
    uint tid,
    uint tgid,
    threadgroup half *qstage,
    threadgroup float *sth,
    threadgroup half *wq,
    threadgroup half *tail,
    threadgroup float *msb,
    threadgroup float *ssum) {
    static_assert(NSG == 8, "the flash body assumes 8 simdgroups == groups");
    static_assert(!PARTIALS || BQ == 1,
                  "the decode partials layout has no query axis to block over");
    static_assert(!PARTIALS || !PASSES,
                  "decode splits and prefill passes are different key-axis cuts");
    constexpr float NEG = -1.701411733864e+38f;
    constexpr uint TILE = 64u;
    constexpr uint NSUB = 8u;
    constexpr uint NTILES = uint(HEAD_DIM) / 8u;
    constexpr uint NPER = NTILES / uint(NSG);
    // Row stride of `state`: fp32 O[HEAD_DIM], then the running max and sum.
    constexpr uint SSTRIDE = uint(HEAD_DIM) + 2u;

    const uint lane = tid & 31u;
    const uint sg = tid >> 5u;
    const uint groups = params.q_heads / params.kv_heads;
    // This threadgroup owns BQ consecutive queries (q0 .. q0+BQ-1) of one KV
    // head; the grid is `ceil(queries/BQ) * kv_heads` threadgroups. Decode
    // instead owns one (split, KV head) pair over a contiguous key slice.
    const uint qg = PARTIALS ? 0u : tgid / params.kv_heads;
    const uint key_head = PARTIALS ? tgid - (tgid / params.kv_heads) * params.kv_heads
                                   : tgid - qg * params.kv_heads;
    const uint q0 = qg * uint(BQ);
    const uint split = PARTIALS ? tgid / params.kv_heads : 0u;
    const uint head = key_head * groups + sg;
    const uint kv_stride = params.kv_heads * uint(HEAD_DIM);
    const uint head_base = key_head * uint(HEAD_DIM);
    // Decode scans only this split's slice. `min(key_count, ...)` keeps a split
    // from running off the end when `key_count` is not a multiple of
    // `keys_per_split`; an empty slice falls out of the loop with m = NEG and a
    // zero accumulator, which the combine pass weights to zero.
    //
    // Prefill passes (OQ9) scan `[pass_begin, pass_end)` instead, where
    // `pass_keys == 0` reproduces the one-dispatch `[0, key_count)` range. The
    // causal and window tests below are per key inside the range, so a key
    // outside it is simply visited by another pass: cutting the index range
    // cannot change any row's arithmetic.
    const uint range_begin =
        PARTIALS ? split * params.keys_per_split : params.pass_begin;
    const uint range_end =
        PARTIALS ? min(params.key_count, range_begin + params.keys_per_split)
        : (params.pass_keys == 0u
               ? params.key_count
               : min(params.key_count, params.pass_begin + params.pass_keys));
    // FIRST / LAST, as oMLX templates them; runtime here so one pipeline serves
    // every pass of the slice chain.
    const bool pass_first = range_begin == 0u;
    const bool pass_last = range_end >= params.key_count;

    // Q rows for all BQ queries of this simdgroup's own head, staged as halves
    // (the QK B operand is loaded transposed from here, so no half->float
    // conversion is needed). The threadgroup holds all 8 heads x BQ queries.
    for (uint b = 0; b < uint(BQ); ++b) {
        if (q0 + b >= params.queries) {
            continue;
        }
        const device half *qsrc =
            queries + ((q0 + b) * params.q_heads + head) * uint(HEAD_DIM);
        threadgroup half *qdst = qstage + (b * uint(NSG) + sg) * uint(HEAD_DIM);
        for (uint c = lane * 8u; c < uint(HEAD_DIM); c += 256u) {
            *reinterpret_cast<threadgroup uint4 *>(qdst + c) =
                *reinterpret_cast<const device uint4 *>(qsrc + c);
        }
    }

    // Running softmax state is per (query, head); simdgroup `sg` owns head `sg`
    // for all BQ queries, so `m`/`s` stay in registers replicated across its 32
    // lanes; only `msb` (the per-(query,head) rescale the PV accumulators need,
    // whose rows are spread over all 8 simdgroups) has to be published.
    float m[BQ];
    float s[BQ];
#pragma unroll
    for (uint b = 0; b < uint(BQ); ++b) {
        m[b] = NEG;
        s[b] = 0.0f;
    }

    simdgroup_float8x8 acc[BQ][NPER];
#pragma unroll
    for (uint b = 0; b < uint(BQ); ++b) {
#pragma unroll
        for (uint j = 0; j < NPER; ++j) {
            acc[b][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    const uint fm = ((lane >= 16u) ? 4u : 0u) + ((lane >> 1u) & 3u);
    const uint fn = ((lane & 8u) ? 4u : 0u) + ((lane & 1u) << 1u);

    // OQ9, resume: this dispatch continues a row an earlier one started. The
    // round trip is fp32 -> device -> fp32 and the key-tile loop below is
    // unchanged, so every row runs exactly the operations of one full dispatch in
    // the same order and the output is bit-identical to it (oMLX's claim, which
    // the `attention_prefill_pass_bit_exact` test checks rather than assumes).
    if (PASSES && !pass_first) {
#pragma unroll
        for (uint b = 0; b < uint(BQ); ++b) {
            if (q0 + b >= params.queries) {
                continue;
            }
            const device float *sbase =
                state + (q0 + b) * params.q_heads * SSTRIDE + key_head * groups * SSTRIDE;
            // `m`/`s` are replicated across the simdgroup and belong to its own
            // head, so a broadcast read is enough and no reduction is needed.
            const device float *mrow = sbase + uint(sg) * SSTRIDE;
            m[b] = mrow[HEAD_DIM];
            s[b] = mrow[HEAD_DIM + 1u];
            const device float *orow = sbase + fm * SSTRIDE;
#pragma unroll
            for (uint j = 0; j < NPER; ++j) {
                // `acc[b][j]`'s columns are dims `nt*8 + fn`, `nt*8 + fn + 1`
                // with `nt = sg * NPER + j` -- the same n-tile order T4f fixed.
                const device float *e =
                    orow + (uint(sg) * uint(NPER) + j) * 8u + fn;
                acc[b][j].thread_elements()[0] = e[0];
                acc[b][j].thread_elements()[1] = e[1];
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint base = range_begin; base < range_end; base += TILE) {
        const uint live_in_tile = min(TILE, range_end - base);
        // Key sub-tile that straddles `range_end`, or NSUB when none does.
        const uint partial_sub =
            (base + TILE > range_end) ? (range_end - 1u - base) / 8u : NSUB;
        const uint partial_kt =
            (live_in_tile % 8u != 0u) ? live_in_tile / 8u : NSUB;

        // ---- QK: simdgroup `sg` owns m-tile `sg` (keys base + 8*sg .. +8) ----
        if (partial_sub < NSUB) {
            // Guard: the previous tile's PV may still be reading `tail`.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint k0 = base + partial_sub * 8u;
            for (uint i = tid; i < 8u * uint(HEAD_DIM); i += 256u) {
                const uint row = i / uint(HEAD_DIM);
                const uint col = i % uint(HEAD_DIM);
                tail[i] = (k0 + row < range_end)
                    ? keys[(k0 + row) * kv_stride + head_base + col]
                    : half(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (sg <= partial_sub) {
            const uint k0 = base + sg * 8u;
            const bool staged = (sg == partial_sub);
            // One dots fragment per blocked query. The K element `kf` is loaded
            // once and reused by every query -- this reuse is the whole point of
            // query blocking, and it halves the K device traffic.
            simdgroup_float8x8 dots[BQ];
#pragma unroll
            for (uint b = 0; b < uint(BQ); ++b) {
                dots[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
            if (staged || k0 + 8u <= range_end) {
                for (int d8 = 0; d8 < HEAD_DIM; d8 += 16) {
                    simdgroup_half8x8 kf0, kf1;
                    if (staged) {
                        simdgroup_load(kf0, tail + d8, HEAD_DIM);
                        simdgroup_load(kf1, tail + d8 + 8, HEAD_DIM);
                    } else {
                        simdgroup_load(kf0, keys + k0 * kv_stride + head_base + d8,
                                       kv_stride);
                        simdgroup_load(kf1, keys + k0 * kv_stride + head_base + d8 + 8,
                                       kv_stride);
                    }
#pragma unroll
                    for (uint b = 0; b < uint(BQ); ++b) {
                        if (q0 + b >= params.queries) {
                            continue;
                        }
                        simdgroup_half8x8 qf0, qf1;
                        simdgroup_load(qf0,
                                       qstage + (b * uint(NSG)) * uint(HEAD_DIM) + d8,
                                       HEAD_DIM, ulong2(0, 0), true);
                        simdgroup_load(qf1,
                                       qstage + (b * uint(NSG)) * uint(HEAD_DIM) + d8 + 8,
                                       HEAD_DIM, ulong2(0, 0), true);
                        simdgroup_multiply_accumulate(dots[b], kf0, qf0, dots[b]);
                        simdgroup_multiply_accumulate(dots[b], kf1, qf1, dots[b]);
                    }
                }
            }
            // Publish [query][key in tile][head]. Keys past `range_end` are
            // forced to NEG by the `live` test below, so a stale or zero slot is
            // harmless.
#pragma unroll
            for (uint b = 0; b < uint(BQ); ++b) {
                if (q0 + b >= params.queries) {
                    continue;
                }
                sth[(b * TILE + sg * 8u + fm) * 8u + fn] = dots[b].thread_elements()[0];
                sth[(b * TILE + sg * 8u + fm) * 8u + fn + 1u] =
                    dots[b].thread_elements()[1];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- online softmax: simdgroup `sg` owns head `sg`, 2 keys per lane,
        //      independently for each of the BQ blocked queries ----
        const uint kt0 = 2u * lane;
        const uint k0 = base + kt0;
        const uint k1 = k0 + 1u;
        const bool in0 = (k0 < range_end);
        const bool in1 = (k1 < range_end);
        const uint p0 = in0 ? key_positions[k0] : 0u;
        const uint p1 = in1 ? key_positions[k1] : 0u;
#pragma unroll
        for (uint b = 0; b < uint(BQ); ++b) {
            if (q0 + b >= params.queries) {
                continue;
            }
            const uint q_abs = query_positions[q0 + b];
            const bool live0 = in0 && (p0 <= q_abs)
                && (params.window == 0u || (q_abs - p0) < params.window);
            const bool live1 = in1 && (p1 <= q_abs)
                && (params.window == 0u || (q_abs - p1) < params.window);
            const float sc0 =
                live0 ? sth[(b * TILE + kt0) * 8u + sg] * params.scale : NEG;
            const float sc1 =
                live1 ? sth[(b * TILE + kt0 + 1u) * 8u + sg] * params.scale : NEG;
            // Same shape as the scalar body: fold the running max in before the
            // cross-lane reduction, and `NEG` (not -inf) so a fully dead tile
            // keeps `m_old - m` finite.
            const float m_new = simd_max(max(max(sc0, sc1), m[b]));
            const float ms = exp(m[b] - m_new);
            const float w0 = live0 ? exp(sc0 - m_new) : 0.0f;
            const float w1 = live1 ? exp(sc1 - m_new) : 0.0f;
            s[b] = s[b] * ms + simd_sum(w0 + w1);
            m[b] = m_new;
            wq[(b * uint(NSG) + sg) * TILE + kt0] = half(w0);
            wq[(b * uint(NSG) + sg) * TILE + kt0 + 1u] = half(w1);
            if (lane == 0u) {
                msb[b * uint(NSG) + sg] = ms;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- PV straddle staging (V rows past `range_end` zeroed) ----
        if (partial_kt < NSUB) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint v0 = base + partial_kt * 8u;
            for (uint i = tid; i < 8u * uint(HEAD_DIM); i += 256u) {
                const uint row = i / uint(HEAD_DIM);
                const uint col = i % uint(HEAD_DIM);
                tail[i] = (v0 + row < range_end)
                    ? values[(v0 + row) * kv_stride + head_base + col]
                    : half(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ---- PV: acc[q][head][dim] = ms * acc + P[head][key] * V[key][dim].
        // The V element `bf` is loaded once and reused by every blocked query,
        // which halves the V device traffic.
        //
        // The rescale is skipped for any (query, head) whose max did not move in
        // this tile: `ms = exp(m_old - m_new)` is then exactly 1, and
        // `x * 1 == x` in IEEE-754, so the guard is bit-exact rather than an
        // approximation. Ported from oMLX v0.7.0
        // (`omlx/patches/mimo_v2/sdpa_flash.py:299-300`, *"A row whose max did not
        // move has factor 1: skipping its rescale is exact"*).
        //
        // `fm` is the fragment row, i.e. the head, and the softmax wrote `ms` for
        // exactly that head -- so this is one branch per (query, head) per tile,
        // guarding `NPER * 2` multiplies. On the first tile of a dispatch every
        // head moves (m starts at NEG), so the branch costs nothing there; on later
        // tiles only the heads that found a new maximum take it. With `BQ = 1` the
        // loop is one iteration, so this is a plain `if` on a threadgroup value
        // all lanes of a simdgroup already computed.
#pragma unroll
        for (uint b = 0; b < uint(BQ); ++b) {
            const float msv = msb[b * uint(NSG) + fm];
            if (msv == 1.0f) {
                continue;
            }
#pragma unroll
            for (uint j = 0; j < NPER; ++j) {
                acc[b][j].thread_elements()[0] *= msv;
                acc[b][j].thread_elements()[1] *= msv;
            }
        }
        // Loop order matters more than anything else in this kernel: a
        // `simdgroup_load` is a real instruction, so the MMAs per load is what
        // caps throughput. Preloading all `BQ` P fragments once per k-step and
        // reusing them across the `NPER` n-tiles costs `BQ + NPER` loads for
        // `BQ * NPER` MMAs -- 9/8 at BQ=1 and 10/16 at BQ=2, versus the
        // `BQ * NPER` P loads a per-(n-tile) load would cost. The same argument
        // is why the K fragment in the QK loop is loaded outside the query
        // loop. (Measured cost of getting this wrong: the T4 bench scored the
        // flash body at ~0.5 MMA/load, 42% below the MMA arm.)
        const uint ksteps = (live_in_tile + 7u) / 8u;
        for (uint kt = 0; kt < ksteps; ++kt) {
            simdgroup_half8x8 af[BQ];
#pragma unroll
            for (uint b = 0; b < uint(BQ); ++b) {
                simdgroup_load(af[b], wq + (b * uint(NSG)) * TILE + kt * 8u, TILE);
            }
            const bool staged = (kt == partial_kt);
#pragma unroll
            for (uint j = 0; j < NPER; ++j) {
                const uint nt = sg * uint(NPER) + j; // KAPLUMBA_NT_BLOCK
                simdgroup_half8x8 bf;
                if (staged) {
                    simdgroup_load(bf, tail + nt * 8u, HEAD_DIM);
                } else {
                    simdgroup_load(bf, values + (base + kt * 8u) * kv_stride
                                       + head_base + nt * 8u, kv_stride);
                }
#pragma unroll
                for (uint b = 0; b < uint(BQ); ++b) {
                    if (q0 + b >= params.queries) {
                        continue;
                    }
                    simdgroup_multiply_accumulate(acc[b][j], af[b], bf, acc[b][j]);
                }
            }
        }
    }

    // Decode: emit the unnormalised `(m, s, O)` partial for this split in the
    // layout `attention_decode_combine_body` reads --
    //   partials[(split * q_heads + head) * (head_dim + 2)] = [m, s, O[head_dim]]
    // so the combine pass is unchanged and stays exact-by-construction (every
    // factor it applies is in [0, 1] because `m_s <= M`). An empty slice writes
    // m = NEG, s = 0 and a zero O, which the combine weights to zero.
    if (PARTIALS) {
        const uint stride = params.head_dim + 2u;
        // `m`/`s` are replicated across the simdgroup and belong to *its* head,
        // so one lane per simdgroup writes them; no `ssum` round trip needed.
        if (lane == 0u) {
            device float *mrow =
                partials + (split * params.q_heads + head) * stride;
            mrow[0] = m[0];
            mrow[1] = s[0];
        }
        device float *orow =
            partials + (split * params.q_heads + key_head * groups + fm) * stride
            + 2u;
#pragma unroll
        for (uint j = 0; j < NPER; ++j) {
            const uint nt = sg * uint(NPER) + j;
            orow[nt * 8u + fn + 0u] = acc[0][j].thread_elements()[0];
            orow[nt * 8u + fn + 1u] = acc[0][j].thread_elements()[1];
        }
        return;
    }

    // OQ9, hand off: this dispatch covered `[range_begin, range_end)` and a
    // later one covers the rest, so store the running row state instead of
    // normalising and writing the output. Symmetric with the resume above.
    if (PASSES && !pass_last) {
#pragma unroll
        for (uint b = 0; b < uint(BQ); ++b) {
            if (q0 + b >= params.queries) {
                continue;
            }
            device float *sbase = state
                + (q0 + b) * params.q_heads * SSTRIDE + key_head * groups * SSTRIDE;
            if (lane == 0u) {
                device float *mrow = sbase + uint(sg) * SSTRIDE;
                mrow[HEAD_DIM] = m[b];
                mrow[HEAD_DIM + 1u] = s[b];
            }
            device float *orow = sbase + fm * SSTRIDE;
#pragma unroll
            for (uint j = 0; j < NPER; ++j) {
                device float *e = orow + (uint(sg) * uint(NPER) + j) * 8u + fn;
                e[0] = acc[b][j].thread_elements()[0];
                e[1] = acc[b][j].thread_elements()[1];
            }
        }
        return;
    }

    // Normalise once at the end: publish each (query, head) `s`, then write fp16.
    if (lane == 0u) {
#pragma unroll
        for (uint b = 0; b < uint(BQ); ++b) {
            ssum[b * uint(NSG) + sg] = s[b];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint out_head = key_head * groups + fm;
#pragma unroll
    for (uint b = 0; b < uint(BQ); ++b) {
        if (q0 + b >= params.queries) {
            continue;
        }
        const float sm = ssum[b * uint(NSG) + fm];
        const float inv = sm > 0.0f ? 1.0f / sm : 0.0f;
#pragma unroll
        for (uint j = 0; j < NPER; ++j) {
            const uint nt = sg * uint(NPER) + j; // KAPLUMBA_NT_BLOCK
            device half *orow = output
                + ((q0 + b) * params.q_heads + out_head) * uint(HEAD_DIM) + nt * 8u;
            orow[fn + 0u] = half(acc[b][j].thread_elements()[0] * inv);
            orow[fn + 1u] = half(acc[b][j].thread_elements()[1] * inv);
        }
    }
}

#define KAPLUMBA_FA_FLASH(NAME, HEAD_DIM, NSG, BQ)                             \
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
        threadgroup half qstage[(BQ) * (NSG) * (HEAD_DIM)];                    \
        threadgroup float sth[(BQ) * 64 * 8];                                  \
        threadgroup half wq[(BQ) * 8 * 64];                                    \
        threadgroup half tail[8 * (HEAD_DIM)];                                 \
        threadgroup float msb[(BQ) * 8];                                       \
        threadgroup float ssum[(BQ) * 8];                                      \
        attention_prefill_flash_body<HEAD_DIM, NSG, BQ, false, false>(          \
            queries, keys, values, query_positions, key_positions, output,     \
            (device float *)output, (device float *)output, params, tid, tgid, \
            qstage, sth, wq, tail, msb, ssum);                                 \
    }

// OQ9: the same body with `PASSES`, so one dispatch covers only
// `[pass_begin, pass_begin + pass_keys)` of the key axis and hands its fp32
// `(m, s, O)` row state to the next dispatch.
//
// **Buffer order.** `dispatch_groups` appends the params blob at
// `bindings.len()`, so `state` is inserted at index 6 and `params` lands at 7
// instead of 6. `output` stays at 5, as in every other attention kernel.
// Getting this wrong binds the state buffer to the params bytes, or leaves the
// kernel's output unbound: neither is an error, both are a plausible and
// completely wrong number.
#define KAPLUMBA_FA_FLASH_PASS(NAME, HEAD_DIM, NSG, BQ)                         \
    kernel void NAME(                                                          \
        const device half *queries [[buffer(0)]],                              \
        const device half *keys [[buffer(1)]],                                 \
        const device half *values [[buffer(2)]],                               \
        const device uint *query_positions [[buffer(3)]],                      \
        const device uint *key_positions [[buffer(4)]],                         \
        device half *output [[buffer(5)]],                                     \
        device float *state [[buffer(6)]],                                     \
        constant AttentionParams &params [[buffer(7)]],                        \
        uint tid [[thread_position_in_threadgroup]],                           \
        uint tgid [[threadgroup_position_in_grid]]) {                          \
        threadgroup half qstage[(BQ) * (NSG) * (HEAD_DIM)];                    \
        threadgroup float sth[(BQ) * 64 * 8];                                  \
        threadgroup half wq[(BQ) * 8 * 64];                                    \
        threadgroup half tail[8 * (HEAD_DIM)];                                 \
        threadgroup float msb[(BQ) * 8];                                       \
        threadgroup float ssum[(BQ) * 8];                                      \
        attention_prefill_flash_body<HEAD_DIM, NSG, BQ, false, true>(           \
            queries, keys, values, query_positions, key_positions, output,     \
            (device float *)output, state, params, tid, tgid, qstage, sth, wq,  \
            tail, msb, ssum);                                                  \
    }

// Decode: the same body with BQ = 1 and PARTIALS, so it scans one split's key
// slice and writes the `(m, s, O)` partial that `attention_decode_combine_512`
// merges. Buffer 5 is the partials buffer, not `output`.
//
// Threadgroup use is the BQ = 1 prefill figure: 8 KB qstage + 2 KB sth + 1 KB
// wq + 8 KB tail = 19.5 KB, under the 32 KB cliff.
#define KAPLUMBA_FA_DECODE_FLASH(NAME, HEAD_DIM, NSG)                         \
    kernel void NAME(                                                          \
        const device half *queries [[buffer(0)]],                              \
        const device half *keys [[buffer(1)]],                                 \
        const device half *values [[buffer(2)]],                               \
        const device uint *query_positions [[buffer(3)]],                      \
        const device uint *key_positions [[buffer(4)]],                         \
        device float *partials [[buffer(5)]],                                  \
        constant AttentionParams &params [[buffer(6)]],                        \
        uint tid [[thread_position_in_threadgroup]],                           \
        uint tgid [[threadgroup_position_in_grid]]) {                          \
        threadgroup half qstage[(NSG) * (HEAD_DIM)];                           \
        threadgroup float sth[64 * 8];                                         \
        threadgroup half wq[8 * 64];                                           \
        threadgroup half tail[8 * (HEAD_DIM)];                                 \
        threadgroup float msb[8];                                              \
        threadgroup float ssum[8];                                             \
        attention_prefill_flash_body<HEAD_DIM, NSG, 1, true, false>(                \
            queries, keys, values, query_positions, key_positions,             \
            (device half *)partials, partials, (device float *)partials, params,\
            tid, tgid, qstage, sth, wq, tail, msb, ssum);                       \
    }

// Only the full-attention shape is instantiated: groups == NSG == 8, HEAD_DIM
// 512. The sliding shape (groups 2, HEAD_DIM 256) keeps the scalar body, and
// `pipeline_warmup_coverage` is not extended to this kernel because it is
// selected by `attention_qk_for`, not by `gqa_index`. BQ must equal
// `attention::FLASH_QUERY_BLOCK` on the Rust side, which sizes the grid.
KAPLUMBA_FA_FLASH(online_attention_flash_512_8, 512, 8, 2)
// The `BQ = 1` twin: same body, one query per threadgroup. Selected below
// `attention::FLASH_QUERY_BLOCK_MIN_KEYS` keys, where query blocking only adds
// critical-path latency (measured at 8k: BQ=2 is ~4% slower than BQ=1).
KAPLUMBA_FA_FLASH(online_attention_flash_bq1_512_8, 512, 8, 1)
// OQ9: the `BQ = 2` and `BQ = 1` key-range-pass twins. Selected only when
// `prefill_pass_enabled` says the shape has enough threadgroups for the slices to
// stay in step (oMLX's `_PASS_MIN_GROUPS = 512` analogue); otherwise the
// one-dispatch kernels above run, unchanged and bit-identical.
KAPLUMBA_FA_FLASH_PASS(online_attention_flash_pass_512_8, 512, 8, 2)
KAPLUMBA_FA_FLASH_PASS(online_attention_flash_pass_bq1_512_8, 512, 8, 1)
// T2b: decode, full-attention shape only (groups == NSG == 8, HEAD_DIM 512),
// same restriction as the prefill flash kernels. BQ is fixed at 1 by the
// partials layout; the split count is a RUNTIME value (`params.splits`), not a
// template parameter -- oMLX measured ~115 ms of JIT per step and -22% decode at
// 8k from templating it (`omlx/patches/mimo_v2/sdpa_flash.py`, v0.7.0).
KAPLUMBA_FA_DECODE_FLASH(attention_decode_flash_512_8, 512, 8)

// Prefill MMA kernel: same buffers as KAPLUMBA_FA_GQA, plus the Q-half staging
// slot and the guarded tail tile. Threadgroup use at the worst instantiated
// shape (NSG=8, HEAD_DIM=512): 2 KB wbuf + 8 KB qhstage + 8 KB kbuf + 1 KB
// sstage (the T2 published score tile) = 19 KB, under the 32 KB cliff.
#define KAPLUMBA_FA_GQA_MMA(NAME, HEAD_DIM, NSG)                               \
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
        threadgroup half qhstage[8 * (HEAD_DIM)];                              \
        threadgroup half kbuf[8 * (HEAD_DIM)];                                 \
        threadgroup float sstage[32 * 8];                                      \
        attention_gqa_mma_body<HEAD_DIM, NSG>(queries, keys, values,           \
            query_positions, key_positions, output, params, tid, tgid,         \
            wbuf, qhstage, kbuf, sstage);                                      \
    }

// Same eight shapes as KAPLUMBA_FA_GQA: the host selects by gqa_index, and
// pipeline_warmup_coverage requires every one of them to exist after new().
KAPLUMBA_FA_GQA_MMA(online_attention_gqa_mma_256_1, 256, 1)
KAPLUMBA_FA_GQA_MMA(online_attention_gqa_mma_256_2, 256, 2)
KAPLUMBA_FA_GQA_MMA(online_attention_gqa_mma_256_4, 256, 4)
KAPLUMBA_FA_GQA_MMA(online_attention_gqa_mma_256_8, 256, 8)
KAPLUMBA_FA_GQA_MMA(online_attention_gqa_mma_512_1, 512, 1)
KAPLUMBA_FA_GQA_MMA(online_attention_gqa_mma_512_2, 512, 2)
KAPLUMBA_FA_GQA_MMA(online_attention_gqa_mma_512_4, 512, 4)
KAPLUMBA_FA_GQA_MMA(online_attention_gqa_mma_512_8, 512, 8)

// ---------------------------------------------------------------------------
// spec/dflash-correctness R8: block attention for the DFlash drafter.
//
// The drafter's query is a 16-row block. Its keys are the concatenation of
// (a) the context rows, which are the target model's fused tap features
// projected by the drafter's own k_proj, and (b) the block's own rows. The
// mask is per-layer:
//
//   * sliding_attention layers: causal within the block AND a `window`-wide
//     trailing window (`q_pos - k_pos < window`).
//   * full_attention layer: NO mask at all -- block rows see every key,
//     including the block rows ahead of them.
//
// `online_attention` above cannot express that: its `live` test is
// unconditional (`k_abs <= q_abs` at attention.metal:836), so a bidirectional
// block needs this kernel. It is O(rows) working set -- one threadgroup per
// (block row, kv head) with an online softmax, no mask buffer and no score
// matrix, which is what AGENTS.md section 8 requires.
//
// One threadgroup per (row, kv head); `nsg = q_heads / kv_heads` simdgroups,
// one per query head in the group. Each lane owns `head_dim / 32` dimensions.
struct DflashBlockParams {
    uint rows;            // block rows, == queries
    uint n_ctx;           // context rows before the block
    uint q_heads;
    uint kv_heads;
    uint head_dim;
    float scale;
    uint window;          // 0 = no window
    uint bidirectional;   // 1 = no causal mask (the full_attention layer)
};

kernel void dflash_block_attn(
    const device half *queries [[buffer(0)]],
    const device half *ctx_keys [[buffer(1)]],
    const device half *ctx_values [[buffer(2)]],
    const device half *block_keys [[buffer(3)]],
    const device half *block_values [[buffer(4)]],
    const device uint *key_positions [[buffer(5)]],
    device half *output [[buffer(6)]],
    constant DflashBlockParams &params [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
    const uint groups = params.q_heads / params.kv_heads;
    const uint row = tgid / params.kv_heads;
    const uint key_head = tgid % params.kv_heads;
    if (row >= params.rows) {
        return;
    }
    const uint nsg = tid / 32u;
    const uint lane = tid % 32u;
    if (nsg >= groups) {
        return;
    }
    const uint q_head = key_head * groups + nsg;
    const uint q_pos = key_positions[params.n_ctx + row];

    // Each lane's own slice of the head dimension.
    const uint dims_per_lane = params.head_dim / 32u;
    const uint dim_base = lane * dims_per_lane;
    const uint q_base = (row * params.q_heads + q_head) * params.head_dim;
    const uint k_base = key_head * params.head_dim;

    float q_reg[16];
    for (uint d = 0; d < dims_per_lane; ++d) {
        q_reg[d] = float(queries[q_base + dim_base + d]);
    }

    const uint total_keys = params.n_ctx + params.rows;
    float m = -INFINITY;
    float l = 0.0f;
    float acc[16];
    for (uint d = 0; d < dims_per_lane; ++d) {
        acc[d] = 0.0f;
    }

    for (uint k = 0; k < total_keys; ++k) {
        const bool past_ok = (params.bidirectional != 0u)
            ? true
            : (key_positions[k] <= q_pos);
        const uint gap = q_pos - key_positions[k];
        const bool window_ok = (params.window == 0u) || (gap < params.window);
        if (!(past_ok && window_ok)) {
            continue;
        }
        const bool in_ctx = k < params.n_ctx;
        const device half *krow = in_ctx
            ? ctx_keys + (k * params.kv_heads + key_head) * params.head_dim
            : block_keys + ((k - params.n_ctx) * params.kv_heads + key_head) * params.head_dim;
        const device half *vrow = in_ctx
            ? ctx_values + (k * params.kv_heads + key_head) * params.head_dim
            : block_values + ((k - params.n_ctx) * params.kv_heads + key_head) * params.head_dim;

        float dot = 0.0f;
        for (uint d = 0; d < dims_per_lane; ++d) {
            dot += q_reg[d] * float(krow[dim_base + d]);
        }
        // Every lane must end up with the SAME total, not just lane 0: `m`, `l`
        // and the `acc[i]` rescale below are per-lane state that the online
        // softmax merges across lanes, so a lane holding a partial sum would
        // silently corrupt the normaliser. `simd_sum` is a butterfly, so all 32
        // lanes end with the true total.
        dot = simd_sum(dot);
        const float score = dot * params.scale;

        // `score` and `m` are both uniform across the simdgroup (the score by
        // the `simd_sum` above, `m` because it is only ever written from
        // `score`), so this is a scalar max -- no reduction. The running max
        // MUST include the current key: taking `simd_max(m)` alone leaves
        // `m == -inf` on the first key and the whole row normalises to zero.
        const float m_new = max(m, score);
        const float alpha = (m == -INFINITY) ? 0.0f : metal::exp(m - m_new);
        const float weight = metal::exp(score - m_new);
        for (uint d = 0; d < dims_per_lane; ++d) {
            acc[d] = acc[d] * alpha + weight * float(vrow[dim_base + d]);
        }
        l = l * alpha + weight;
        m = m_new;
    }

    const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    const uint o_base = (row * params.q_heads + q_head) * params.head_dim;
    for (uint d = 0; d < dims_per_lane; ++d) {
        output[o_base + dim_base + d] = half(acc[d] * inv);
    }
}
