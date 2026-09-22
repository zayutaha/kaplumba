// Bounded-scratch Gemma 4 GQA attention (T3.5 / T3.7).
//
//   out[pos, h, :] = softmax_{kv in window(pos)} ( q[pos, h, :] · k[kv, kvh, :] ) · v[kv, kvh, :]
//
// Query row `r` sits at absolute position `q_pos0 + r` and attends to absolute
// key position `kv_pos0 + j <= pos` (causal) plus, when `window != 0`,
// `kv_pos0 + j >= pos - window + 1`. That single rule covers every shape the
// engine needs, because the two position bases decouple "where the keys start"
// from "where the queries are":
//
//   * prefill:            `s == kv_rows`, `kv_pos0 == q_pos0`
//   * plain cached step:  `s == 1`, `kv_pos0 == 0`, `kv_rows == pos + 1`
//   * windowed SWA cache: `s == 1`, `kv_rows <= window`,
//                         `kv_pos0 == pos + 1 - kv_rows`
//
// The last one is why the bases are explicit: a sliding layer only keeps the
// last `window` keys (a rotated window, not a growing vector), so key row 0 no
// longer corresponds to absolute position 0.
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
    uint kv_rows;  // key rows available (kv_pos0 + kv_rows - 1 is the newest)
    uint q_pos0;   // absolute position of query row 0
    uint kv_pos0;  // absolute position of key row 0 (<= q_pos0)
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
    const uint pos = p.q_pos0 + row;
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

    // KV row range covered by this threadgroup's block of query rows: rows share
    // it so the loads stay coalesced, and anything still invisible is masked per
    // row inside the tile. Bounds are in *key row* space, hence the `- kv_pos0`.
    const uint p_last = pos + min((uint)AROWS, p.s - tg.x * AROWS) - 1u;
    const uint hi = min(p.kv_rows - 1u, p_last - p.kv_pos0);
    const uint lo_abs = (p.window > 0u) ? ((p.q_pos0 + tg.x * AROWS + 1u > p.window)
                                               ? (p.q_pos0 + tg.x * AROWS + 1u - p.window)
                                               : 0u)
                                         : 0u;
    const uint lo = (lo_abs > p.kv_pos0) ? (lo_abs - p.kv_pos0) : 0u;

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
                const uint a = p.kv_pos0 + kb + j;
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
