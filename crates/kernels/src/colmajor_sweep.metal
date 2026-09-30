// Decode dispatch-geometry sweep: 3 bpw column-major GEMV over the
// (K-split x threads-per-threadgroup) grid.
//
// WHY THIS IS A SEPARATE FILE. The three obvious places to put it are all
// closed to it:
//
//   * `linear.metal` is rewritten between `linear_gemv_exl3_k1(` and
//     `kernel void linear_gemv(` by `scripts/gen_exl3_gemv.py`.
//   * `linear_repacked.metal` holds the existing `_sk2` / `_sk4` colmajor
//     kernels, which are the *baseline* of the sweep -- putting the sweep
//     cells next to them invites exactly the edit this is not allowed to make.
//   * `linear.rs`'s dispatch arms must not grow a case, or the engine would
//     silently acquire a geometry switch.
//
// So this is a standalone library, loaded only by `crates/kernels/src/sweep.rs`
// and dispatched only by `crates/engine/examples/sweep_geom.rs`. Nothing the
// engine runs is reachable from here. Every kernel name starts with
// `gemv_sweep_`, which no dispatch path can produce.
//
// WHAT IS SWEPT. Two axes, and only two:
//
//   SK  K-split: `SK` threadgroups share one output tile-row, each covering
//               `ceil(tiles_k / SK)` K-tiles. The engine's live geometry is 1.
//   TG  threads per threadgroup: 128 / 256 / 512, i.e. 4 / 8 / 16 simdgroups.
//               Every colmajor kernel in this tree hardcodes 8 simdgroups, so
//               128 and 512 have never been dispatched here at all.
//
// TensorFold (MIT, `src/tensorfold/cuda/exl3/linear.py:30-45`) picks a `(SK,
// warps)` pair per shape and admits its own heuristic is "3-19% off the best
// cell of the (K split, warps) grid" on the 3-bit rows of their 27B packs.
// That grid is what the twelve entry points below enumerate. Its hardware is
// not ours -- its `plan()` reasons in SM blocks (192 of them) and an M4 Pro has
// 20 GPU cores -- so the prior it carries is weak, which is a reason to measure
// rather than a reason to expect a win.
//
// A NOTE ON THE ONE FUNCTION COPIED IN. `decode_state_mcg` is transcribed from
// `linear.metal:121-126` rather than shared: MSL has no include path through
// `newLibraryWithSource`, and a shared header would mean editing `linear.metal`,
// which is generated. A copy is a drift risk, so it is checked: the
// `gemv_sweep_mcg_probe` kernel below is dispatched by
// `examples/sweep_geom.rs` for all 256 states and compared against
// `engine::exl3::decode_state_mcg`, and the harness refuses to report a table
// if any state disagrees. This repo has already shipped two false coverage
// claims that came from a test exercising a different entry point than the one
// under test (AGENTS.md §3.4); a private copy of the codebook is exactly that
// shape of mistake waiting to happen.

// The MCG codebook, folded to a compile-time constant. Transcribed from
// `linear.metal:121-126`; see the drift check above.
static inline float decode_state_mcg(uint value) {
    value = value * 0xCBAC1FEDu;
    value = 0x3b603b60u ^ (value & 0x8fff8fffu);
    half2 h = as_type<half2>(value);
    return float(h.x + h.y);
}

// Writes `decode_state_mcg(s)` for `s` in `0..256`, so the harness can prove
// the copy above is the same function the engine's kernels use. A geometry
// table measured with a subtly different codebook would be a table about a
// kernel nobody ships, and it would look entirely normal.
kernel void gemv_sweep_mcg_probe(
    device float *out [[buffer(0)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < 256u) {
        out[gid] = decode_state_mcg(gid);
    }
}

// BEGIN GENERATED: dispatch-geometry sweep
// Two MSL constraints shape this signature, both found the hard way by
// compiling it (a `program_source` error, not a Rust one):
//
//   * a function that is not a kernel gets no `using namespace metal;` prelude,
//     so `min`, `simd_shuffle_xor` and `mem_flags` must be reachable from here
//     -- hence the `using` inside the body;
//   * `threadgroup` variables cannot be *declared* in a non-kernel function, so
//     `red` arrives as a pointer and the kernels below own the storage.
template <uint SK, uint TG>
static inline void sweep_colmajor_k3(
    const device ushort *packed,
    const device half *x,
    device float *y,
    constant const uint *shape, // [K, N, tn_groups]
    threadgroup float *red,
    uint lid,
    uint lane,
    uint sg,
    uint tgid) {
    using namespace metal;
    constexpr uint KA = 3u;
    constexpr uint RING = 256u * KA;
    constexpr uint WORDS = RING / 32u;
    constexpr uint SG = TG / 32u;
    const uint K = shape[0];
    const uint N = shape[1];
    const uint tn_groups = shape[2];
    const uint tiles_n = N / 16u;
    const uint tiles_k = K / 16u;
    const uint r0 = (lane % 4u) * 2u;

    const uint lo0 = (lane * 8u + 1u) * 3u + RING - 16u;
    const uint sh0 = lo0 % 32u;
    const uint hi0 = lo0 / 32u;
    const uint j0 = hi0 >= WORDS ? hi0 - WORDS : hi0;
    const uint j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u;

    // tn_groups threadgroups cover the output tile-rows; SK of them cover the
    // K reduction for each row. The caller passes tn_groups rather than having
    // it recomputed from `ntg`, so the kernel cannot and the dispatch cannot
    // disagree about how the grid was cut up.
    const uint tn0 = tgid % tn_groups;
    const uint sk = tgid / tn_groups;

    // Balanced split. `min` on both ends so an over-large `sk` (only possible
    // if the caller asked for more splits than there are tiles) is an empty
    // range contributing zeros rather than an out-of-bounds read.
    const uint k_per = (tiles_k + SK - 1u) / SK;
    const uint k_start = min(sk * k_per, tiles_k);
    const uint k_end = min(k_start + k_per, tiles_k);

    for (uint tn = tn0; tn < tiles_n; tn += tn_groups) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        const device uint *column = (const device uint *)packed + tn * (tiles_k * WORDS);
        const uint stride = WORDS;

        const uint step = SG * 4u;
        const uint k_unroll_end = k_start + ((k_end - k_start) / step) * step;

        uint tk = k_start + sg * 4u;
        for (; tk < k_unroll_end; tk += step) {
            const device uint *t0 = column + (tk + 0u) * stride;
            const device uint *t1 = column + (tk + 1u) * stride;
            const device uint *t2 = column + (tk + 2u) * stride;
            const device uint *t3 = column + (tk + 3u) * stride;

            const ulong w0a = t0[j0], w0b = t0[j1];
            const ulong w1a = t1[j0], w1b = t1[j1];
            const ulong w2a = t2[j0], w2b = t2[j1];
            const ulong w3a = t3[j0], w3b = t3[j1];

            const ulong A0 = ((w0a << 32) | w0b) << sh0;
            const ulong A1 = ((w1a << 32) | w1b) << sh0;
            const ulong A2 = ((w2a << 32) | w2b) << sh0;
            const ulong A3 = ((w3a << 32) | w3b) << sh0;

            const device half *xr0 = x + (tk + 0u) * 16u + r0;
            const float p00 = float(xr0[0]), p01 = float(xr0[1]), p02 = float(xr0[8]), p03 = float(xr0[9]);
            const device half *xr1 = x + (tk + 1u) * 16u + r0;
            const float p10 = float(xr1[0]), p11 = float(xr1[1]), p12 = float(xr1[8]), p13 = float(xr1[9]);
            const device half *xr2 = x + (tk + 2u) * 16u + r0;
            const float p20 = float(xr2[0]), p21 = float(xr2[1]), p22 = float(xr2[8]), p23 = float(xr2[9]);
            const device half *xr3 = x + (tk + 3u) * 16u + r0;
            const float p30 = float(xr3[0]), p31 = float(xr3[1]), p32 = float(xr3[8]), p33 = float(xr3[9]);

            float v0[8], v1[8], v2[8], v3[8];
            #pragma unroll
            for (uint i = 0; i < 8u; ++i) {
                const uint rad = i * KA + 16u;
                v0[i] = decode_state_mcg(uint(A0 >> (64u - rad)) & 0xffffu);
                v1[i] = decode_state_mcg(uint(A1 >> (64u - rad)) & 0xffffu);
                v2[i] = decode_state_mcg(uint(A2 >> (64u - rad)) & 0xffffu);
                v3[i] = decode_state_mcg(uint(A3 >> (64u - rad)) & 0xffffu);
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

        for (tk = k_unroll_end + sg; tk < k_end; tk += SG) {
            const device uint *t0 = column + tk * stride;
            const ulong a0 = t0[j0], a1 = t0[j1];
            const ulong A0 = ((a0 << 32) | a1) << sh0;
            const device half *xr0 = x + tk * 16u + r0;
            const float p0 = float(xr0[0]), p1 = float(xr0[1]), p2 = float(xr0[8]), p3 = float(xr0[9]);
            float v[8];
            #pragma unroll
            for (uint i = 0; i < 8u; ++i) {
                const uint rad = i * KA + 16u;
                const uint s = uint(A0 >> (64u - rad));
                v[i] = decode_state_mcg(s & 0xffffu);
            }
            acc0 += p0 * v[0] + p1 * v[1] + p2 * v[2] + p3 * v[3];
            acc1 += p0 * v[4] + p1 * v[5] + p2 * v[6] + p3 * v[7];
        }

        acc0 += simd_shuffle_xor(acc0, 1u);
        acc0 += simd_shuffle_xor(acc0, 2u);
        acc1 += simd_shuffle_xor(acc1, 1u);
        acc1 += simd_shuffle_xor(acc1, 2u);
        if (lane % 4u == 0u) {
            red[sg * 16u + (lane / 4u)] = acc0;
            red[sg * 16u + (lane / 4u + 8u)] = acc1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < 16u) {
            float sum = 0.0f;
            for (uint g = 0; g < SG; ++g) {
                sum += red[g * 16u + lid];
            }
            // SK == 1 writes `sk * N == 0`, i.e. the live kernel's layout, so
            // the baseline cell needs no separate reduction pass.
            y[(sk * N) + tn * 16u + lid] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void gemv_sweep_k3_colmajor_sk1_t128(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant const uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
    constexpr uint SG = 128u / 32u;
    threadgroup float red[SG][16];
    // `red` is a 2D threadgroup array and does not decay on its own here;
    // `&red[0][0]` is the flat pointer the body indexes.
    sweep_colmajor_k3<1u, 128u>(packed, x, y, shape, &red[0][0], lid, lane, sg, tgid);
}

kernel void gemv_sweep_k3_colmajor_sk1_t256(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant const uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
    constexpr uint SG = 256u / 32u;
    threadgroup float red[SG][16];
    // `red` is a 2D threadgroup array and does not decay on its own here;
    // `&red[0][0]` is the flat pointer the body indexes.
    sweep_colmajor_k3<1u, 256u>(packed, x, y, shape, &red[0][0], lid, lane, sg, tgid);
}

kernel void gemv_sweep_k3_colmajor_sk1_t512(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant const uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
    constexpr uint SG = 512u / 32u;
    threadgroup float red[SG][16];
    // `red` is a 2D threadgroup array and does not decay on its own here;
    // `&red[0][0]` is the flat pointer the body indexes.
    sweep_colmajor_k3<1u, 512u>(packed, x, y, shape, &red[0][0], lid, lane, sg, tgid);
}

kernel void gemv_sweep_k3_colmajor_sk2_t128(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant const uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
    constexpr uint SG = 128u / 32u;
    threadgroup float red[SG][16];
    // `red` is a 2D threadgroup array and does not decay on its own here;
    // `&red[0][0]` is the flat pointer the body indexes.
    sweep_colmajor_k3<2u, 128u>(packed, x, y, shape, &red[0][0], lid, lane, sg, tgid);
}

kernel void gemv_sweep_k3_colmajor_sk2_t256(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant const uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
    constexpr uint SG = 256u / 32u;
    threadgroup float red[SG][16];
    // `red` is a 2D threadgroup array and does not decay on its own here;
    // `&red[0][0]` is the flat pointer the body indexes.
    sweep_colmajor_k3<2u, 256u>(packed, x, y, shape, &red[0][0], lid, lane, sg, tgid);
}

kernel void gemv_sweep_k3_colmajor_sk2_t512(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant const uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
    constexpr uint SG = 512u / 32u;
    threadgroup float red[SG][16];
    // `red` is a 2D threadgroup array and does not decay on its own here;
    // `&red[0][0]` is the flat pointer the body indexes.
    sweep_colmajor_k3<2u, 512u>(packed, x, y, shape, &red[0][0], lid, lane, sg, tgid);
}

kernel void gemv_sweep_k3_colmajor_sk4_t128(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant const uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
    constexpr uint SG = 128u / 32u;
    threadgroup float red[SG][16];
    // `red` is a 2D threadgroup array and does not decay on its own here;
    // `&red[0][0]` is the flat pointer the body indexes.
    sweep_colmajor_k3<4u, 128u>(packed, x, y, shape, &red[0][0], lid, lane, sg, tgid);
}

kernel void gemv_sweep_k3_colmajor_sk4_t256(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant const uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
    constexpr uint SG = 256u / 32u;
    threadgroup float red[SG][16];
    // `red` is a 2D threadgroup array and does not decay on its own here;
    // `&red[0][0]` is the flat pointer the body indexes.
    sweep_colmajor_k3<4u, 256u>(packed, x, y, shape, &red[0][0], lid, lane, sg, tgid);
}

kernel void gemv_sweep_k3_colmajor_sk4_t512(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant const uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
    constexpr uint SG = 512u / 32u;
    threadgroup float red[SG][16];
    // `red` is a 2D threadgroup array and does not decay on its own here;
    // `&red[0][0]` is the flat pointer the body indexes.
    sweep_colmajor_k3<4u, 512u>(packed, x, y, shape, &red[0][0], lid, lane, sg, tgid);
}

kernel void gemv_sweep_k3_colmajor_sk8_t128(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant const uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
    constexpr uint SG = 128u / 32u;
    threadgroup float red[SG][16];
    // `red` is a 2D threadgroup array and does not decay on its own here;
    // `&red[0][0]` is the flat pointer the body indexes.
    sweep_colmajor_k3<8u, 128u>(packed, x, y, shape, &red[0][0], lid, lane, sg, tgid);
}

kernel void gemv_sweep_k3_colmajor_sk8_t256(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant const uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
    constexpr uint SG = 256u / 32u;
    threadgroup float red[SG][16];
    // `red` is a 2D threadgroup array and does not decay on its own here;
    // `&red[0][0]` is the flat pointer the body indexes.
    sweep_colmajor_k3<8u, 256u>(packed, x, y, shape, &red[0][0], lid, lane, sg, tgid);
}

kernel void gemv_sweep_k3_colmajor_sk8_t512(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant const uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
    constexpr uint SG = 512u / 32u;
    threadgroup float red[SG][16];
    // `red` is a 2D threadgroup array and does not decay on its own here;
    // `&red[0][0]` is the flat pointer the body indexes.
    sweep_colmajor_k3<8u, 512u>(packed, x, y, shape, &red[0][0], lid, lane, sg, tgid);
}

kernel void gemv_sweep_reduce_sk2(
    const device float *partials [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant const uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < n) {
        float sum = partials[gid];
        #pragma unroll
        for (uint s = 1u; s < 2u; ++s) {
            sum += partials[s * n + gid];
        }
        y[gid] = sum;
    }
}

kernel void gemv_sweep_reduce_sk4(
    const device float *partials [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant const uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < n) {
        float sum = partials[gid];
        #pragma unroll
        for (uint s = 1u; s < 4u; ++s) {
            sum += partials[s * n + gid];
        }
        y[gid] = sum;
    }
}

kernel void gemv_sweep_reduce_sk8(
    const device float *partials [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant const uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < n) {
        float sum = partials[gid];
        #pragma unroll
        for (uint s = 1u; s < 8u; ++s) {
            sum += partials[s * n + gid];
        }
        y[gid] = sum;
    }
}

// END GENERATED: dispatch-geometry sweep