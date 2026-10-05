// =============================================================================
// crux3 — the decode-arithmetic arms, run at last.
//
// `spec/decode-ceiling/findings/A2-dequant.md` §3 pre-registered exactly three
// arms to answer "is the 3 bpw GEMV **issue**-bound on `decode_state_mcg`, or is
// it already at the memory wall?", wrote down the reading for each, and recorded
// in §10: *"Run the three arms in §3. ~30 min of GPU, no code changes to
// production, and it converts this entire file from arithmetic to a measurement.
// This is the only item on the list that can change the verdict."* They were
// never run. They are run here, and they decide where the last 40% of decode
// lives.
//
// WHY THE THREE ARMS AND WHAT THEY MEAN
//
// Arm A is the production kernel, `linear_gemv_ishizuki_k3_colmajor_2col_sk4` —
// the one `encode_gemv_raw_sk4` dispatches, and the 204 GB/s peak that
// `gemv_bandwidth.rs` measures and `spec/decode-25tok-peak-bandwidth` §16.3
// records. Arms B and C are the *same loop over the same addresses with the same
// loads*, differing only in the body of the extract. They are generated from one
// macro so "identical except the extract" is true by construction and not by
// transcription.
//
// `decode_state_mcg(s)` is 4 ops: `s * 0xCBAC1FED`, `(x & 0x8fff8fff) ^ 0x3b603b60`,
// then `float(half2.h.x + half2.h.y)`. The window is `s & 0xffff`.
//
//   arm | extract body                                   | ops/code | depth
//   -----+------------------------------------------------+----------+------
//   A   | decode_state_mcg(p & 0xffff)  [production]     |    4     |  7
//   B   | float(as_type<half>(ushort((p ^ 0x3b603b60))))  |    2     |  4
//   C   | float(as_type<half>(ushort(p)))                 |    1     |  3
//
// (`p` is already masked to 16 bits by the caller, so the `ushort(...)` is the
// narrowing the production `decode_state_mcg` gets from `as_type<half2>`.)
//
// All three keep **eight independent chains** (the eight compile-time literal
// window shifts of `A_hi`/`A_lo`), which is the property A2 identified as the
// thing the `_nod` experiment destroyed: `_nod` removed ops *and* the ILP filler
// and came out 26.7% **slower**, so it measured ILP starvation, not arithmetic
// cost. B and C are built so they cannot be read that way.
//
// WHY THE OUTPUT IS GARBAGE IN B AND C, AND WHY THAT IS FINE HERE
//
// The decode maps 16 bits to an fp16, so B and C produce wrong numbers. They
// still **consume every loaded word** (the result reaches `y` through the
// accumulator) and **issue the same loads at the same addresses**, so they are
// bandwidth-identical to A and ALU-cheaper. That is the whole measurement.
// Nothing downstream of this file is permitted to call these kernels; they exist
// only for `crates/engine/examples/crux3_decode_algebra.rs`.
// =============================================================================

static inline void extract_v_crux3_armB(uint wa, uint wb, uint sh0, uint rsh0, thread float v[8]) {
    const uint A_hi = (wa << sh0) | (wb >> rsh0);
    const uint A_lo = wb << sh0;
    v[0] = float(as_type<half>(ushort(((A_hi >> 16) & 0xffffu) ^ 0x3b603b60u)));
    v[1] = float(as_type<half>(ushort(((A_hi >> 13) & 0xffffu) ^ 0x3b603b60u)));
    v[2] = float(as_type<half>(ushort(((A_hi >> 10) & 0xffffu) ^ 0x3b603b60u)));
    v[3] = float(as_type<half>(ushort(((A_hi >> 7) & 0xffffu) ^ 0x3b603b60u)));
    v[4] = float(as_type<half>(ushort(((A_hi >> 4) & 0xffffu) ^ 0x3b603b60u)));
    v[5] = float(as_type<half>(ushort(((A_hi >> 1) & 0xffffu) ^ 0x3b603b60u)));
    v[6] = float(as_type<half>(ushort((((A_hi << 2) | (A_lo >> 30)) & 0xffffu) ^ 0x3b603b60u)));
    v[7] = float(as_type<half>(ushort((((A_hi << 5) | (A_lo >> 27)) & 0xffffu) ^ 0x3b603b60u)));
}

static inline void extract_v_crux3_armC(uint wa, uint wb, uint sh0, uint rsh0, thread float v[8]) {
    const uint A_hi = (wa << sh0) | (wb >> rsh0);
    const uint A_lo = wb << sh0;
    v[0] = float(as_type<half>(ushort((A_hi >> 16) & 0xffffu)));
    v[1] = float(as_type<half>(ushort((A_hi >> 13) & 0xffffu)));
    v[2] = float(as_type<half>(ushort((A_hi >> 10) & 0xffffu)));
    v[3] = float(as_type<half>(ushort((A_hi >> 7) & 0xffffu)));
    v[4] = float(as_type<half>(ushort((A_hi >> 4) & 0xffffu)));
    v[5] = float(as_type<half>(ushort((A_hi >> 1) & 0xffffu)));
    v[6] = float(as_type<half>(ushort(((A_hi << 2) | (A_lo >> 30)) & 0xffffu)));
    v[7] = float(as_type<half>(ushort(((A_hi << 5) | (A_lo >> 27)) & 0xffffu)));
}

// One macro, three instantiations. `EXTRACT` is the ONLY thing that differs
// between arm A and arms B and C — verified by
// `crates/engine/tests/probe_crux3_decode_algebra.rs::the_three_arms_differ_only
// _in_the_extract_body`, which reads this file and asserts the generated bodies
// are character-identical outside the extract name.
#define CRUX3_COLMAJOR_2COL_SK4(NAME, EXTRACT)                             \
    kernel void NAME(                                                      \
        const device ushort *packed [[buffer(0)]],                         \
        const device half *x [[buffer(1)]],                                \
        device float *y [[buffer(2)]],                                     \
        constant const uint *shape [[buffer(3)]],                          \
        uint lid [[thread_position_in_threadgroup]],                       \
        uint lane [[thread_index_in_simdgroup]],                           \
        uint sg [[simdgroup_index_in_threadgroup]],                       \
        uint tgid [[threadgroup_position_in_grid]],                        \
        uint ntg [[threadgroups_per_grid]]) {                              \
        constexpr uint KA = 3u;                                            \
        constexpr int CB = 1;                                              \
        constexpr uint RING = 256u * KA;                                   \
        constexpr uint WORDS = RING / 32u;                                 \
        const uint K = shape[0];                                           \
        const uint N = shape[1];                                           \
        const uint tiles_n = N / 16u;                                      \
        const uint tiles_k = K / 16u;                                      \
        const uint pairs_n = tiles_n / 2u;                                 \
        constexpr int SG = 8;                                              \
        constexpr uint SK = 4u;                                            \
        threadgroup float red[2][SG][16];                                  \
        const uint r0 = (lane % 4u) * 2u;                                  \
                                                                             \
        const uint lo0 = (lane * 8u + 1u) * 3u + RING - 16u;               \
        const uint sh0 = lo0 % 32u;                                        \
        const uint rsh0 = 32u - sh0;                                       \
        const uint hi0 = lo0 / 32u;                                        \
        const uint j0 = hi0 >= WORDS ? hi0 - WORDS : hi0;                  \
        const uint j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u;      \
                                                                             \
        const uint total_groups = pairs_n * SK;                            \
        for (uint tg = tgid; tg < total_groups; tg += ntg) {               \
            const uint tn = tg % pairs_n;                                  \
            const uint sk = tg / pairs_n;                                  \
                                                                             \
            const uint k_per_split = tiles_k / SK;                         \
            const uint k_start = sk * k_per_split;                         \
            const uint k_end = k_start + k_per_split;                      \
                                                                             \
            const uint tn0 = tn * 2u;                                      \
            const uint tn1 = tn0 + 1u;                                     \
                                                                             \
            float acc0_0 = 0.0f;                                           \
            float acc0_1 = 0.0f;                                           \
            float acc1_0 = 0.0f;                                           \
            float acc1_1 = 0.0f;                                           \
                                                                             \
            const device uint *col0 = (const device uint *)packed + tn0 * (tiles_k * WORDS); \
            const device uint *col1 = (const device uint *)packed + tn1 * (tiles_k * WORDS); \
            const uint stride = WORDS;                                     \
                                                                             \
            const uint step = SG * 2u;                                     \
            const uint k_unroll_end = k_start + ((k_end - k_start) / step) * step; \
                                                                             \
            uint tk = k_start + sg * 2u;                                   \
            for (; tk < k_unroll_end; tk += step) {                        \
                const device half *xr0 = x + (tk + 0u) * 16u + r0;         \
                const float p00 = float(xr0[0]), p01 = float(xr0[1]), p02 = float(xr0[8]), p03 = float(xr0[9]); \
                const device half *xr1 = x + (tk + 1u) * 16u + r0;         \
                const float p10 = float(xr1[0]), p11 = float(xr1[1]), p12 = float(xr1[8]), p13 = float(xr1[9]); \
                                                                             \
                const device uint *t0_0 = col0 + (tk + 0u) * stride;        \
                const device uint *t1_0 = col0 + (tk + 1u) * stride;        \
                                                                             \
                const device uint *t0_1 = col1 + (tk + 0u) * stride;        \
                const device uint *t1_1 = col1 + (tk + 1u) * stride;        \
                                                                             \
                float v[8];                                                \
                                                                             \
                const uint w0_0_0 = t0_0[j0], w0_0_1 = t0_0[j1];           \
                const uint w1_0_0 = t1_0[j0], w1_0_1 = t1_0[j1];           \
                const uint w0_1_0 = t0_1[j0], w0_1_1 = t0_1[j1];           \
                const uint w1_1_0 = t1_1[j0], w1_1_1 = t1_1[j1];           \
                                                                             \
                EXTRACT(w0_0_0, w0_0_1, sh0, rsh0, v);                     \
                acc0_0 += p00 * v[0] + p01 * v[1] + p02 * v[2] + p03 * v[3];\
                acc0_1 += p00 * v[4] + p01 * v[5] + p02 * v[6] + p03 * v[7];\
                                                                             \
                EXTRACT(w1_0_0, w1_0_1, sh0, rsh0, v);                     \
                acc0_0 += p10 * v[0] + p11 * v[1] + p12 * v[2] + p13 * v[3];\
                acc0_1 += p10 * v[4] + p11 * v[5] + p12 * v[6] + p13 * v[7];\
                                                                             \
                EXTRACT(w0_1_0, w0_1_1, sh0, rsh0, v);                     \
                acc1_0 += p00 * v[0] + p01 * v[1] + p02 * v[2] + p03 * v[3];\
                acc1_1 += p00 * v[4] + p01 * v[5] + p02 * v[6] + p03 * v[7];\
                                                                             \
                EXTRACT(w1_1_0, w1_1_1, sh0, rsh0, v);                     \
                acc1_0 += p10 * v[0] + p11 * v[1] + p12 * v[2] + p13 * v[3];\
                acc1_1 += p10 * v[4] + p11 * v[5] + p12 * v[6] + p13 * v[7];\
            }                                                              \
                                                                             \
            for (tk = k_unroll_end + sg; tk < k_end; tk += SG) {           \
                const device half *xr0 = x + tk * 16u + r0;                \
                const float p0 = float(xr0[0]), p1 = float(xr0[1]), p2 = float(xr0[8]), p3 = float(xr0[9]); \
                                                                             \
                const device uint *t0_0 = col0 + tk * stride;              \
                const uint a0_0 = t0_0[j0], a1_0 = t0_0[j1];               \
                float v_0[8];                                              \
                EXTRACT(a0_0, a1_0, sh0, rsh0, v_0);                       \
                acc0_0 += p0 * v_0[0] + p1 * v_0[1] + p2 * v_0[2] + p3 * v_0[3]; \
                acc0_1 += p0 * v_0[4] + p1 * v_0[5] + p2 * v_0[6] + p3 * v_0[7]; \
                                                                             \
                const device uint *t0_1 = col1 + tk * stride;              \
                const uint a0_1 = t0_1[j0], a1_1 = t0_1[j1];               \
                float v_1[8];                                              \
                EXTRACT(a0_1, a1_1, sh0, rsh0, v_1);                       \
                acc1_0 += p0 * v_1[0] + p1 * v_1[1] + p2 * v_1[2] + p3 * v_1[3]; \
                acc1_1 += p0 * v_1[4] + p1 * v_1[5] + p2 * v_1[6] + p3 * v_1[7]; \
            }                                                              \
                                                                             \
            acc0_0 += simd_shuffle_xor(acc0_0, 1u);                        \
            acc0_0 += simd_shuffle_xor(acc0_0, 2u);                        \
            acc0_1 += simd_shuffle_xor(acc0_1, 1u);                        \
            acc0_1 += simd_shuffle_xor(acc0_1, 2u);                        \
                                                                             \
            acc1_0 += simd_shuffle_xor(acc1_0, 1u);                        \
            acc1_0 += simd_shuffle_xor(acc1_0, 2u);                        \
            acc1_1 += simd_shuffle_xor(acc1_1, 1u);                        \
            acc1_1 += simd_shuffle_xor(acc1_1, 2u);                        \
                                                                             \
            if (lane % 4u == 0u) {                                         \
                red[0][sg][lane / 4u] = acc0_0;                            \
                red[0][sg][lane / 4u + 8u] = acc0_1;                       \
                red[1][sg][lane / 4u] = acc1_0;                            \
                red[1][sg][lane / 4u + 8u] = acc1_1;                       \
            }                                                              \
            threadgroup_barrier(mem_flags::mem_threadgroup);               \
            if (lid < 32u) {                                               \
                const uint col_idx = lid / 16u;                            \
                const uint out_lane = lid % 16u;                           \
                float sum = 0.0f;                                          \
                for (int g = 0; g < SG; ++g) {                             \
                    sum += red[col_idx][g][out_lane];                       \
                }                                                          \
                const uint out_tn = (col_idx == 0u) ? tn0 : tn1;           \
                y[sk * N + out_tn * 16u + out_lane] = sum;                 \
            }                                                              \
            threadgroup_barrier(mem_flags::mem_threadgroup);               \
        }                                                                  \
    }

// Arm A. `extract_v_mcg_32` is the production helper, so this instantiation is
// the shipped kernel with a different name. `crux3_decode_algebra.rs` measures
// THIS one as arm A, not the original, so that all three arms come from one
// macro and the comparison cannot be contaminated by a transcription difference.
CRUX3_COLMAJOR_2COL_SK4(linear_gemv_ishizuki_k3_crux3_armA, extract_v_mcg_32)
CRUX3_COLMAJOR_2COL_SK4(linear_gemv_ishizuki_k3_crux3_armB, extract_v_crux3_armB)
CRUX3_COLMAJOR_2COL_SK4(linear_gemv_ishizuki_k3_crux3_armC, extract_v_crux3_armC)

// PROBE-ONLY KERNELS. Nothing in `linear.rs` selects these names; the production
// path asks for `linear_gemv_ishizuki_k3_colmajor_2col_sk4` by string and this
// macro never defines it. They produce deliberately wrong values in arms B and C
// and exist so that `spec/decode-20tok-crux3` can answer whether the decode
// arithmetic is the wall or the memory system is.
