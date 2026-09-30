// PROTOTYPE ONLY -- NOT COMPILED INTO THE SHIPPED LIBRARY.
//
// `crates/kernels/src/linear.rs` concatenates exactly two Metal sources
// (`include_str!("linear.metal")` at line 13 and `include_str!("linear_repacked.metal")`
// at line 22). This file is in neither, so no Rust build, no `LinearKernels::new`,
// no dispatch table and no production decode path can reach it. It exists to be
// compiled by `xcrun metal -c` as a legality / IR check, and (kernels 1 and 2 only)
// to be A/B'd on a free GPU. Delete it with the spec.
//
// ---------------------------------------------------------------------------
// 2 bpw geometry, derived from scratch and cross-checked against TensorFold
// (MIT, /tmp/kaplumba-refs/TensorFold, src/tensorfold/cuda/exl3/decode.cuh:14-42
// and linear.cu:64-101):
//
//   tile      = 256 codes x 2 bits = 512 bits = 16 u32 words   (their tile_words<4>() = 4*K2)
//   window(c) = bits [2c - 14, 2c + 2) mod 512                 (their stream_end - 16)
//   lane L    = codes 8L+1 .. 8L+8
//   lo0       = (L*8+1)*2 + 512 - 16 = 16L + 498               (their lane_start: first)
//   j0(L)     = (lo0 >> 5) & 15 = (15 + ((L+1) >> 1)) & 15
//   sh0(L)    = lo0 & 31 = 18 (L even), 2 (L odd)
//   j1(L)     = (j0 + 1) & 15
//   => L even needs words {j0, j0+1}; L odd needs {j0} only, because sh0 == 2
//      puts all 30 live bits of A inside one 32-bit word.
//   LPW = 2: 32 lanes map onto 16 distinct words, 2 lanes each, monotone in L
//   (one wrap, at L = 0 -> 1).  That is the "identity" case, and it is exactly
//   TensorFold's `base + lane / LPW` with `prev = word != lane / (8/K2)`.
//
// The two derivations agree on all 32 entries of the table, which is why the
// 2 bpw index is `lane >> 1` (a shift) where the 3 bpw index
// `(24*lane + 755) / 32 % 24` is neither uniform nor lane-linear.
//
// `decode_state_mcg` is copied verbatim from linear.metal:121-126 so this file
// needs nothing from the generated preamble.

#include <metal_stdlib>
using namespace metal;

static inline float decode_state_mcg(uint value) {
    value = value * 0xCBAC1FEDu;
    value = 0x3b603b60u ^ (value & 0x8fff8fffu);
    half2 h = as_type<half2>(value);
    return float(h.x + h.y);
}

#define K2_RING  512u
#define K2_WORDS 16u
#define K2_SG    8

// ---------------------------------------------------------------------------
// K1. Reference port: the live `linear_gemv_ishizuki_k3_colmajor_contig4` with
//     KA := 2.  64 lane-loads per 64 useful bytes per tile => amplification 4.0x.
//     Nothing new; it exists to prove the 2 bpw geometry compiles and to give
//     the scalar 2 bpw baseline the arithmetic needs.
// ---------------------------------------------------------------------------
kernel void proto_k2_scalar(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant const uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]]) {
    constexpr uint KA = 2u;
    threadgroup float red[K2_SG][16];
    const uint r0 = (lane % 4u) * 2u;

    const uint lo0 = (lane * 8u + 1u) * KA + K2_RING - 16u;
    const uint sh0 = lo0 % 32u;
    const uint hi0 = lo0 / 32u;
    const uint j0 = hi0 >= K2_WORDS ? hi0 - K2_WORDS : hi0;
    const uint j1 = j0 + 1u >= K2_WORDS ? j0 + 1u - K2_WORDS : j0 + 1u;

    const uint tiles_n = shape[1] / 16u;
    const uint tiles_k = shape[0] / 16u;

    for (uint tn = tgid; tn < tiles_n; tn += ntg) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        const device uint *column = (const device uint *)packed + tn * (tiles_k * K2_WORDS);
        const uint step = K2_SG * 4u;
        const uint k_end = (tiles_k / step) * step;

        for (uint tk = sg * 4u; tk < k_end; tk += step) {
            const device uint *t0 = column + (tk + 0u) * K2_WORDS;
            const device uint *t1 = column + (tk + 1u) * K2_WORDS;
            const device uint *t2 = column + (tk + 2u) * K2_WORDS;
            const device uint *t3 = column + (tk + 3u) * K2_WORDS;
            const ulong A0 = ((ulong(t0[j0]) << 32) | t0[j1]) << sh0;
            const ulong A1 = ((ulong(t1[j0]) << 32) | t1[j1]) << sh0;
            const ulong A2 = ((ulong(t2[j0]) << 32) | t2[j1]) << sh0;
            const ulong A3 = ((ulong(t3[j0]) << 32) | t3[j1]) << sh0;

            const device half *xr0 = x + (tk + 0u) * 16u + r0;
            const device half *xr1 = x + (tk + 1u) * 16u + r0;
            const device half *xr2 = x + (tk + 2u) * 16u + r0;
            const device half *xr3 = x + (tk + 3u) * 16u + r0;
            const float p00 = float(xr0[0]), p01 = float(xr0[1]), p02 = float(xr0[8]), p03 = float(xr0[9]);
            const float p10 = float(xr1[0]), p11 = float(xr1[1]), p12 = float(xr1[8]), p13 = float(xr1[9]);
            const float p20 = float(xr2[0]), p21 = float(xr2[1]), p22 = float(xr2[8]), p23 = float(xr2[9]);
            const float p30 = float(xr3[0]), p31 = float(xr3[1]), p32 = float(xr3[8]), p33 = float(xr3[9]);

            float v0[8], v1[8], v2[8], v3[8];
            #pragma unroll
            for (uint i = 0u; i < 8u; ++i) {
                const uint r = i * KA + 16u;
                v0[i] = decode_state_mcg(uint(A0 >> (64u - r)) & 0xffffu);
                v1[i] = decode_state_mcg(uint(A1 >> (64u - r)) & 0xffffu);
                v2[i] = decode_state_mcg(uint(A2 >> (64u - r)) & 0xffffu);
                v3[i] = decode_state_mcg(uint(A3 >> (64u - r)) & 0xffffu);
            }
            acc0 += p00 * v0[0] + p01 * v0[1] + p02 * v0[2] + p03 * v0[3];
            acc1 += p00 * v0[4] + p01 * v0[5] + p02 * v0[6] + p03 * v0[7];
            acc0 += p10 * v1[0] + p11 * v1[1] + p12 * v1[2] + p13 * v1[3];
            acc1 += p10 * v1[4] + p11 * v1[5] + p12 * v1[6] + p13 * v1[7];
            acc0 += p20 * v2[0] + p21 * v2[1] + p22 * v2[2] + p23 * v2[3];
            acc1 += p20 * v2[4] + p21 * v2[5] + p22 * v2[6] + p23 * v2[7];
            acc0 += p30 * v3[0] + p31 * v3[1] + p32 * v3[2] + p33 * v3[3];
            acc1 += p30 * v3[4] + p31 * v3[5] + p32 * v3[6] + p33 * v3[7];
        }
        acc0 += simd_shuffle_xor(acc0, 1u);
        acc0 += simd_shuffle_xor(acc0, 2u);
        acc1 += simd_shuffle_xor(acc1, 1u);
        acc1 += simd_shuffle_xor(acc1, 2u);
        if (lane % 4u == 0u) {
            red[sg][lane / 4u] = acc0;
            red[sg][lane / 4u + 8u] = acc1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < 16u) {
            float s = 0.0f;
            for (int g = 0; g < K2_SG; ++g) s += red[g][lid];
            y[tn * 16u + lid] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---------------------------------------------------------------------------
// K2. TensorFold's shuffle loader, transcribed, including their double-buffered
//     `cur`/`nxt` prefetch (linear.cu:140-163) which our reverted `_simd` did
//     not have.  Step = 8 consecutive k-tiles = 128 words = 512 B; lane L loads
//     step[c*32 + L] for c = 0..3, so every word of the step is fetched by
//     exactly one lane (amplification 1.0x).  Tile j's words come from
//     cur[j >> 1] at index (j*16 % 32) + L/2, and even lanes need the word
//     before their own -- which is `i_prev = (i_own + 15) & 15`, still lane-linear.
//     Amplification 1.0x, cost 16 lane-loads + 64 simd_shuffle per tile.
// ---------------------------------------------------------------------------
kernel void proto_k2_shuffle_tf(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant const uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]]) {
    constexpr uint KA = 2u;
    threadgroup float red[K2_SG][16];
    const uint r0 = (lane % 4u) * 2u;
    const uint lo0 = (lane * 8u + 1u) * KA + K2_RING - 16u;
    const uint sh0 = lo0 % 32u;
    const uint i_own = lane >> 1;
    const uint i_prev = (i_own + K2_WORDS - 1u) & (K2_WORDS - 1u);
    const bool prev = (lane & 1u) == 0u;

    const uint tiles_n = shape[1] / 16u;
    const uint tiles_k = shape[0] / 16u;

    for (uint tn = tgid; tn < tiles_n; tn += ntg) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        const device uint *column = (const device uint *)packed + tn * (tiles_k * K2_WORDS);
        const uint steps = tiles_k / 8u;

        for (uint s0 = sg; s0 < steps; s0 += K2_SG) {
            const device uint *step = column + (s0 * 8u) * K2_WORDS;
            uint cur[4];
            uint nxt[4];
            #pragma unroll
            for (uint c = 0u; c < 4u; ++c) cur[c] = step[c * 32u + lane];

            #pragma unroll
            for (uint it = 0u; it < 8u; ++it) {
                if (it < 7u) {
                    const device uint *ns = step + (it + 1u) * 32u;
                    #pragma unroll
                    for (uint c = 0u; c < 4u; ++c) nxt[c] = ns[c * 32u + lane];
                }
                const uint reg = it >> 1;
                const uint base = (it * K2_WORDS) % 32u;   // 0 for even it, 16 for odd
                const uint w0 = prev ? simd_shuffle(cur[reg], base + i_prev)
                                     : simd_shuffle(cur[reg], base + i_own);
                const uint w1 = prev ? simd_shuffle(cur[reg], base + i_own) : 0u;
                const ulong A = ((ulong(w0) << 32) | w1) << sh0;

                const device half *xr = x + (s0 * 8u + it) * 16u + r0;
                const float p0 = float(xr[0]), p1 = float(xr[1]), p2 = float(xr[8]), p3 = float(xr[9]);
                float v[8];
                #pragma unroll
                for (uint i = 0u; i < 8u; ++i) {
                    const uint r = i * KA + 16u;
                    v[i] = decode_state_mcg(uint(A >> (64u - r)) & 0xffffu);
                }
                acc0 += p0 * v[0] + p1 * v[1] + p2 * v[2] + p3 * v[3];
                acc1 += p0 * v[4] + p1 * v[5] + p2 * v[6] + p3 * v[7];
                #pragma unroll
                for (uint c = 0u; c < 4u; ++c) cur[c] = nxt[c];
            }
        }
        acc0 += simd_shuffle_xor(acc0, 1u);
        acc0 += simd_shuffle_xor(acc0, 2u);
        acc1 += simd_shuffle_xor(acc1, 1u);
        acc1 += simd_shuffle_xor(acc1, 2u);
        if (lane % 4u == 0u) {
            red[sg][lane / 4u] = acc0;
            red[sg][lane / 4u + 8u] = acc1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < 16u) {
            float sm = 0.0f;
            for (int g = 0; g < K2_SG; ++g) sm += red[g][lid];
            y[tn * 16u + lid] = sm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---------------------------------------------------------------------------
// K3. The experiment that settles the two prices this whole design is priced
//     against, with no dequantisation in the way.  Every arm moves the SAME
//     512 B per iteration per simdgroup; only the instruction mix changes, so
//     the ratio of arms is the ratio of prices.
//
//       arm 0  32 x `uint`            lane-linear, 512 B useful, 32 lane-loads
//       arm 1   8 x `uint4`           lane-linear, same 512 B,  8 lane-loads
//                                       -> is a 16 B per-lane load priced like a
//                                          4 B one, or like 4 of them?
//       arm 2  32 x `uint` +  4 `simd_shuffle_xor`  (uniform index)
//       arm 3  32 x `uint` + 16 `simd_shuffle_xor`  (uniform index)
//       arm 4  32 x `uint` + 32 `simd_shuffle` (dynamic, lane/2)
//       arm 5  32 x `uint` + 64 `simd_shuffle` (dynamic, lane/2 + i_prev)
//
//     S arms 2..5 against arm 0 is the shuffle price in lane-load units; arm 1
//     against arm 0 is the load-width price.  Both are currently unknown on
//     AGX: `decode-bandwidth-handover/spec.md:236` closed load width only on a
//     DRAM-bound streaming probe, which cannot distinguish them, and the only
//     shuffle datapoint in this repo (180.5 -> 146.5 GB/s) confounds price with
//     latency because `_simd` had no prefetch.
//
//     `cfg[0]` = arm, `cfg[1]` = iterations.  Output is one uint per
//     threadgroup, an xor-checksum that the loads feed, so nothing is dead.
// ---------------------------------------------------------------------------
kernel void proto_price_probe(
    const device uint *src [[buffer(0)]],
    device uint *dst [[buffer(1)]],
    constant const uint *cfg [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]]) {
    const uint arm = cfg[0];
    const uint iters = cfg[1];
    // 128 words of headroom so a simdgroup never reads another one's slot
    const device uint *base = src + tgid * 160u + sg * 32u;
    const uint dyn_a = lane >> 1;
    const uint dyn_b = (dyn_a + 15u) & 15u;
    uint acc = 0u;

    for (uint i = 0u; i < iters; ++i) {
        const device uint *p = base + ((i * 160u) & ~31u);
        uint r[8];
        if (arm == 1u) {
            #pragma unroll
            for (uint c = 0u; c < 8u; ++c) r[c] = p[c * 4u + (lane >> 2u)];
        } else {
            #pragma unroll
            for (uint c = 0u; c < 8u; ++c) r[c] = p[c * 16u + (lane & 15u) * 2u];
        }
        acc ^= r[0];
        if (arm >= 2u && arm <= 3u) {
            uint n = (arm == 2u) ? 4u : 16u;
            #pragma unroll
            for (uint s = 0u; s < 16u; ++s) {
                if (s < n) acc += simd_shuffle_xor(r[s & 7u], (ushort)(1u + (s & 7u)));
            }
        } else if (arm == 4u) {
            #pragma unroll
            for (uint s = 0u; s < 32u; ++s) acc += simd_shuffle(r[s & 7u], (ushort)dyn_a);
        } else if (arm == 5u) {
            #pragma unroll
            for (uint s = 0u; s < 32u; ++s) acc += simd_shuffle(r[s & 7u], (ushort)dyn_a);
            #pragma unroll
            for (uint s = 0u; s < 32u; ++s) acc += simd_shuffle(r[s & 7u], (ushort)dyn_b);
        }
        acc = acc * 0x9E3779B9u + 1u;
    }
    acc += simd_shuffle_xor(acc, 16u);
    if (lane == 0u) dst[tgid] = acc;
    (void)lid;
    (void)ntg;
}

// ---------------------------------------------------------------------------
// DEAD ENDS, recorded so they are not retried.
//
// A) `uint4` per lane with a 4-word block chosen to contain both of the lane's
//    words {j0, j0+1}.  A 4-word block contains the pair iff j0 mod 4 < 3, and
//    the four pairs with j0 mod 4 == 3 are {3,4} {7,8} {11,12} {15,0} -- the last
//    being the ring seam.  There is no 32-byte per-lane load in Metal, and mixing
//    widths costs a second warp instruction, which puts it back at amplification
//    4.0x.  The load-width question is better settled by arm 1 of K3 alone.
//
// B) 16 codes per lane with 16 active lanes, each on the 8-byte-aligned pair at
//    word `L & ~1`.  This has no ring seam and needs only a `uint2` per lane,
//    but it changes which codes a lane owns, hence the k-order of the float sum,
//    hence bit-exactness: `crates/engine/tests/live_kernel_bit_exact.rs` asserts
//    fp16 equality, not a tolerance, and would have to be re-derived.  It also
//    raises live registers from 8 decoded floats to 16.
// ---------------------------------------------------------------------------