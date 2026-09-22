// Bounded-scratch Gemma 4 GQA attention (T3.5 / T3.7).
//
//   out[pos, h, :] = softmax_{kv in window(pos)} ( q[pos, h, :] · k[kv, kvh, :] ) · v[kv, kvh, :]
//
// Query row `r` of a dispatch sits at absolute position `kv_len - s + r`, and
// attends to `kv <= pos` (causal) plus, when `window != 0`, `kv >= pos - window + 1`.
// That single rule covers both shapes the engine needs: a prefill (`s == kv_len`)
// and a cached decode step (`s == 1`, `kv_len == pos + 1`).
//
// Memory discipline (user directive 2026-09-22, spec T3.7). This kernel keeps
// **the entire attention working set in registers**:
//
//   * the running max, the running softmax denominator and the DW output
//     accumulators are per-thread scalars/arrays — nothing per-sequence;
//   * scores for one KV tile live in a `float[ATILE]` register array;
//   * the slice reduction uses `simd_shuffle_xor` inside the simd group, so
//     there is **no threadgroup memory and no barrier at all**.
//
// Consequently the working set is O(1) in sequence length: no `[heads, L, L]`
// score matrix, no `[heads, q, kv]` mask tensor, no growable scratch. Peak
// resident memory is therefore set by the KV cache alone (T5.1), which is what
// the 24 GB box can actually afford.
//
// GEOMETRY. `DW` (dims per thread) is a **compile-time** constant so the
// per-thread `acc`/`qr` arrays stay in registers with static indexing; the
// kernel is instantiated for the two head dims this model uses
// (`PART = head_dim / DW` = 8 for 256, 16 for 512). A threadgroup covers
// `AROWS` query rows for one query head; the PART threads of a row are
// consecutive lanes of one simd group (PART divides 32 and the threadgroup
// index is `row * PART + slice`), which is what makes the shuffle reduction
// legal. Anything else is rejected by the host wrapper (`KernelError::
// Unsupported`) instead of being silently mis-reduced.

#include <metal_stdlib>
using namespace metal;

struct AttnParams {
    uint s;        // query rows in this dispatch
    uint kv_len;   // keys available (>= s); query row r sits at kv_len - s + r
    uint n_heads;  // query heads
    uint n_kv;     // key/value heads (GQA); n_heads % n_kv == 0
    uint head_dim; // 256 (sliding) or 512 (full) for Gemma 4
    uint window;   // 0 = full causal, else sliding window size
};

constant uint AROWS = 8u;   // query rows per threadgroup
constant uint ATILE = 16u;  // KV entries per tile

// `PART` = simd-group width = number of threads sharing one query row.
template <uint DW, uint PART>
inline void attn_body(device const half *q [[buffer(0)]],
                      device const half *k [[buffer(1)]],
                      device const half *v [[buffer(2)]],
                      device half *out [[buffer(3)]],
                      constant AttnParams &p [[buffer(4)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint2 tp [[thread_position_in_threadgroup]]) {
    const uint hd = p.head_dim;
    const uint d0 = tp.x * DW;
    const uint h = tg.y;
    const uint kvh = h / (p.n_heads / p.n_kv);
    // Global query row: `tp.y` is only the row *within* the threadgroup, so the
    // threadgroup index must be folded in (a missing `tg.x` term silently makes
    // every block past the first recompute block 0's positions).
    const uint row = tg.x * AROWS + tp.y;
    const uint pos = p.kv_len - p.s + row;
    const bool live = row < p.s;

    // Query slice for this row (registers). Padded rows never store, but they
    // must not read out of bounds of `q` either.
    // q/out are laid out by *row* (`[s, n_heads, head_dim]`); only the mask uses
    // the absolute position. The two coincide for a prefill (`s == kv_len`) and
    // differ for a cached decode step.
    float qr[DW];
    if (live) {
        device const half *qp = q + ((ulong)row * p.n_heads + h) * hd + d0;
        for (uint d = 0; d < DW; ++d) {
            qr[d] = float(qp[d]);
        }
    } else {
        for (uint d = 0; d < DW; ++d) {
            qr[d] = 0.0f;
        }
    }

    float acc[DW];
    for (uint d = 0; d < DW; ++d) {
        acc[d] = 0.0f;
    }
    float m_run = -3.0e38f;
    float l_run = 0.0f;

    // KV range covered by this threadgroup's row block. Rows in the block share
    // the range so the loads stay coalesced; out-of-range entries are masked per
    // row inside the tile.
    const uint p0 = p.kv_len - p.s + tg.x * AROWS;
    const uint p1 = min(p0 + AROWS - 1u, p.kv_len - 1u);
    const uint lo = (p.window > 0u && p0 + 1u > p.window) ? (p0 + 1u - p.window) : 0u;
    const uint hi = p1;

    for (uint kb = lo; kb <= hi; kb += ATILE) {
        const uint jn = min(ATILE, hi - kb + 1u);

        // --- scores for the tile: q·k over this thread's dims, then a
        // simd-group all-reduce (all lanes end up with the full dot) ---
        float sc[ATILE];
        for (uint j = 0; j < ATILE; ++j) {
            float dot = 0.0f;
            if (j < jn) {
                device const half *kp = k + ((ulong)(kb + j) * p.n_kv + kvh) * hd + d0;
                for (uint d = 0; d < DW; ++d) {
                    dot += qr[d] * float(kp[d]);
                }
            }
            for (uint m = 1u; m < PART; m <<= 1u) {
                dot += simd_shuffle_xor(dot, (ushort)m);
            }
            sc[j] = dot;
        }

        // --- mask (causal + window) ---
        float tmax = -3.0e38f;
        for (uint j = 0; j < ATILE; ++j) {
            if (j < jn) {
                const uint a = kb + j;
                const bool visible = (a <= pos) && (p.window == 0u || a + p.window > pos);
                if (!visible) {
                    sc[j] = -3.0e38f;
                }
                tmax = max(tmax, sc[j]);
            }
        }

        // --- online softmax update (the rescale is per-thread, on registers) ---
        const float m_new = max(m_run, tmax);
        const float corr = exp(m_run - m_new);
        float psum = 0.0f;
        for (uint j = 0; j < ATILE; ++j) {
            if (j < jn) {
                const float e = exp(sc[j] - m_new);
                sc[j] = e;
                psum += e;
            }
        }
        l_run = l_run * corr + psum;
        for (uint d = 0; d < DW; ++d) {
            acc[d] *= corr;
        }

        // --- accumulate p·v (masked entries carry p = 0 but read in-range v) ---
        for (uint j = 0; j < jn; ++j) {
            const float e = sc[j];
            device const half *vp = v + ((ulong)(kb + j) * p.n_kv + kvh) * hd + d0;
            for (uint d = 0; d < DW; ++d) {
                acc[d] += e * float(vp[d]);
            }
        }
        m_run = m_new;
    }

    if (live) {
        const float inv = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
        device half *op = out + ((ulong)row * p.n_heads + h) * hd + d0;
        for (uint d = 0; d < DW; ++d) {
            op[d] = (half)(acc[d] * inv);
        }
    }
}

// head_dim 256 (sliding layers): 8 slices of 32 dims → 64-thread threadgroups.
kernel void exl3_attn_hd256(device const half *q [[buffer(0)]],
                            device const half *k [[buffer(1)]],
                            device const half *v [[buffer(2)]],
                            device half *out [[buffer(3)]],
                            constant AttnParams &p [[buffer(4)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint2 tp [[thread_position_in_threadgroup]]) {
    attn_body<32u, 8u>(q, k, v, out, p, tg, tp);
}

// head_dim 512 (full-attention layers): 16 slices of 32 dims → 128 threads.
kernel void exl3_attn_hd512(device const half *q [[buffer(0)]],
                            device const half *k [[buffer(1)]],
                            device const half *v [[buffer(2)]],
                            device half *out [[buffer(3)]],
                            constant AttnParams &p [[buffer(4)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint2 tp [[thread_position_in_threadgroup]]) {
    attn_body<32u, 16u>(q, k, v, out, p, tg, tp);
}
