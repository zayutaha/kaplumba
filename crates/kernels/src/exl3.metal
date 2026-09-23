// EXL3 trellis GEMV + 128-point Hadamard passes (T2.1, tuned in T2.3).
//
//   y = had_out(had_in(x, suh) @ decode(trellis), svh)
//
// had_in / had_out use a **fast Walsh–Hadamard transform**: 7 butterfly stages
// over the 128-block in threadgroup memory, i.e. 7 × 128 lane-ops instead of a
// full O(n²) 128×128 matrix product per block (measured ~36× fewer ops). The
// butterfly computes the Sylvester matrix H[i][j] = (-1)^popcount(i&j), which is
// symmetric, so `H · block` is the same transform the CPU reference applies —
// only the summation order differs, by ≤1 ulp of f16 after rounding.
//
// exl3_gemv: lane-per-group over the kernel-order permutation. The 256 elements
// of a 16×16 tile form 32 groups of 8 kernel indices; group g maps to row-major
// rows (R0,R1,R2,R3) = ((g%4)*2, +1, +8, +9) of columns cA = g/4, cB = cA + 8.
// Each lane accumulates into exactly two output columns with *static* register
// accumulators, and 4 lanes per column collapse in one shared-memory reduction.
//
// The 4 pairs of a group (pair index t = 4g + q) come from the decoder's
// per-pair funnel shift; the host precomputes (i0, i1, s1) per pair per K in
// `pair_tab`, and the 4 entries a lane needs are consecutive u32s, so they load
// as one 16-byte vector.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint HAD = 128u;
constant uint TILE = 16u;
constant uint LOG2_HAD = 7u;

struct Exl3Params {
    uint in_tiles;
    uint out_tiles;
    uint k;
    // u32 words per packed tile (= 8*k). The trellis is read as 32-bit words
    // because the decoder's funnel window is defined over u32 pairs, exactly
    // like the CPU reference (`unpack_trellis_tile` packs two u16s per word).
    uint twords;
    uint mode; // 0 = Default, 1 = Mcg (Mul1 stays on the CPU path)
    uint aligned; // trellis base is 4-byte aligned -> single u32 loads
    // T2.5 K-split: threadgroups per out-tile (grid.y). Each split accumulates
    // its slice of the in-tile range into y[split * out_tiles + ot]; the host
    // either reads that directly (nsplit == 1 → identical indexing) or sums
    // the splits in `exl3_gemv_reduce`.
    uint nsplit;
};

// Same, plus the row count for the batched prefill GEMM (T3.2).
struct Exl3GemmParams {
    uint in_tiles;
    uint out_tiles;
    uint k;
    uint twords;
    uint mode;
    uint aligned;
    uint rows;
};

// T3.10 K-split extension for the sg path only (scalar exl3_gemm and the
// T3.8 probes keep the 7-field Exl3GemmParams): `nsplit` slices the stage
// range across grid.z and `stride` (elements) is the partial row-padded
// slice pitch — buffer(3) is `y` when nsplit==1 (old flat path, tg.z=0 →
// offset 0) or the shared partial buffer otherwise.
struct Exl3GemmSgParams {
    uint in_tiles;
    uint out_tiles;
    uint k;
    uint twords;
    uint mode;
    uint aligned;
    uint rows;
    uint nsplit;
    uint stride;
};

// Row geometry for the Hadamard passes: rows of `row_len` elements, so the
// 2-D threadgroup grid is (row_len/128, rows) and row 0 with rows=1 reduces to
// the original 1-D dispatch.
struct HadParams {
    uint row_len;
};

// EXL3 procedural codebook, matching the CPU `decode_3inst`:
//   Mcg:     x *= 0xCBAC1FED, then x = 0x3B603B60 ^ (x & 0x8FFF8FFF), then one
//            exact IEEE binary16 add of the two packed halves.
//   Default: x = x*89226354 + 64248484, then the same tail.
// Mul1 is not supported here (the engine keeps it on the CPU path).
inline half exl3_decode(uint cw, uint mode) {
    uint x = cw & 0xFFFFu;
    if (mode == 1u) {
        x = x * 0xCBAC1FEDu;
    } else {
        x = x * 89226354u + 64248484u;
    }
    x = 0x3B603B60u ^ (x & 0x8FFF8FFFu);
    half lo = as_type<half>((ushort)(x & 0xFFFFu));
    half hi = as_type<half>((ushort)(x >> 16));
    return lo + hi;
}

// Vectorized exl3_decode returning float (T3):
// Evaluates both halves as half2 and does one binary16 addition in registers.
inline float exl3_decode_f(uint cw, uint mode) {
    uint x = cw & 0xFFFFu;
    if (mode == 1u) {
        x = x * 0xCBAC1FEDu;
    } else {
        x = x * 89226354u + 64248484u;
    }
    x = 0x3B603B60u ^ (x & 0x8FFF8FFFu);
    half2 h2 = as_type<half2>(x);
    return float(h2.x + h2.y);
}

// One u32 word of the packed trellis = two little-endian u16 words. The paired
// path exists because real artifacts place some trellis tensors (lm_head) at a
// 2-mod-4 byte offset, where a u32 load would not be naturally aligned.
inline uint ld_word(device const ushort *t, uint w, bool aligned) {
    if (aligned) {
        return *(device const uint *)(t + 2u * w);
    }
    return (uint)t[2u * w] | ((uint)t[2u * w + 1u] << 16);
}

struct CodePair {
    half lo;
    half hi;
};

// One funnel window, unpacked to the two 16-bit codewords the pair carries.
// `pk` packs i0 | i1<<8 | s1<<16; the words are read straight from the device
// buffer (L1-resident: a whole tile is 96-192 bytes and 32 lanes share it).
//
// NOTE (T2.3): staging the tile through threadgroup memory first (1-2 global
// loads per lane instead of 8) was tried and produced *wrong* weights — with
// one-hot probes, column cA came back holding column cB's weight, i.e. pairs
// q=0 and q=2 resolved identically. Reverted unproven; see the notes dead-end
// entry before retrying, and use `debug_gemv_rows` to check a one-hot matrix.
inline CodePair funnel_pair(
    device const ushort *tile, uint i0, uint i1, uint s1, uint k, uint mode, bool aligned
) {
    uint a = ld_word(tile, i0, aligned);
    uint b = ld_word(tile, i1, aligned);
    uint w1 = (uint)((((ulong)a) << 32 | (ulong)b) >> s1);
    CodePair r;
    r.lo = exl3_decode((w1 >> k) & 0xFFFFu, mode);
    r.hi = exl3_decode(w1 & 0xFFFFu, mode);
    return r;
}

// ---------------------------------------------------------------------------
// xh = had_r(x * suh)   (one 128-block per threadgroup, 128 threads)
// ---------------------------------------------------------------------------
// grid = (row_len/128, rows): threadgroup (tx, ty) transforms block tx of row
// ty. With rows=1 this is exactly the old 1-D dispatch.
kernel void exl3_had_in(device const half *x [[buffer(0)]],
                        device const half *suh [[buffer(1)]],
                        device half *xh [[buffer(2)]],
                        constant HadParams &hp [[buffer(3)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        uint2 thr [[thread_position_in_threadgroup]]) {
    threadgroup float sh[HAD];
    uint t = thr.x;
    uint base = tg.y * hp.row_len + tg.x * HAD;
    // suh is ONE row of scales shared by every batch row: index it within the
    // row (tg.x*HAD + t), while x/xh are the full [rows, row_len] buffers.
    sh[t] = float(x[base + t]) * float(suh[tg.x * HAD + t]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sylvester FWT, 7 stages; H is symmetric so `H · block` needs no transpose.
    for (uint stage = 1u; stage < HAD; stage <<= 1u) {
        if ((t & stage) == 0u) {
            float a = sh[t];
            float b = sh[t + stage];
            sh[t] = a + b;
            sh[t + stage] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    xh[base + t] = (half)(sh[t] * as_type<float>(0x3DB504F3u));
}

// ---------------------------------------------------------------------------
// y = xh @ decode(trellis)   (raw f32 output, one lane per output column)
// ---------------------------------------------------------------------------
// One out-tile per 32-lane threadgroup. Fatter threadgroups (4 warps sharing a
// threadgroup) measured *slower* on M4 Pro: with few out-tiles there are too
// few threadgroups left to spread across 20 cores. (Superseded by P0 findings:
// the gemv-strided read pattern sustains ≥214 GB/s and splitting raises read
// MLP — the compute is the wall. See `exl3_gemv4` below for the parity shape.)
constant uint WARPS = 1u;

kernel void exl3_gemv(device const half *xh [[buffer(0)]],
                      device const ushort *trellis [[buffer(1)]],
                      device const uint *pair_tab [[buffer(2)]],
                      device float *y [[buffer(3)]],
                      constant Exl3Params &p [[buffer(4)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint2 thr [[thread_position_in_threadgroup]]) {
    uint sub = 0u;
    uint tid = thr.x;
    uint g = tid & 31u;       // lane = kernel-order group
    uint ot = tg.x * WARPS + sub;
    // T2.5 K-split: grid.y selects the in-tile slice (see Exl3Params.nsplit).
    // nsplit == 1 / split == 0 reduces to the old flat indexing exactly.
    uint nsplit = p.nsplit;
    uint chunk = (p.in_tiles + nsplit - 1u) / nsplit;
    uint it0 = tg.y * chunk;
    uint it1 = min(p.in_tiles, it0 + chunk);
    bool live = ot < p.out_tiles;

    uint r0 = (g & 3u) * 2u;
    bool aligned = p.aligned != 0u;

    float accA = 0.0f;
    float accB = 0.0f;

    // Strength-reduced tile address.
    const device ushort *tile =
        trellis + ((ulong)min(it0, p.in_tiles) * p.out_tiles + ot) * p.twords * 2u;
    const ulong tstride = (ulong)p.out_tiles * p.twords * 2u;
    uint it = it0;

    if (p.k != 7u) {
        // T3: dq4 64-bit window path for k in {1..6, 8}.
        // A 4-aligned group of 4 codewords spans 3K+16 bits, which always fits
        // into a single 64-bit window of two words.
        uint ig0_[2];
        uint ig1_[2];
        uint s3_[2];
        int tw = int(p.twords);
        for (uint gg = 0u; gg < 2u; gg++) {
            uint c0 = g * 8u + gg * 4u;
            int e_last = int(c0 + 4u) * int(p.k) + 256 * int(p.k);
            int i_end = (e_last - 1) / 32;
            ig1_[gg] = uint(((i_end % tw) + tw) % tw);
            ig0_[gg] = uint((((i_end - 1) % tw) + tw) % tw);
            s3_[gg] = uint((i_end + 1) * 32 - e_last);
        }

        // 2-way unrolled loop: loads 4 words per tile from device and extracts 8 codewords.
        for (; live && it + 1u < it1; it += 2u) {
            half x_lane = xh[it * 16u + g];
            float x00 = float(simd_shuffle(x_lane, ushort(r0)));
            float x01 = float(simd_shuffle(x_lane, ushort(r0 + 1u)));
            float x02 = float(simd_shuffle(x_lane, ushort(r0 + 8u)));
            float x03 = float(simd_shuffle(x_lane, ushort(r0 + 9u)));
            {
                uint a0 = ld_word(tile, ig0_[0], aligned);
                uint b0 = ld_word(tile, ig1_[0], aligned);
                ulong m0 = ((ulong)a0 << 32) | (ulong)b0;
                uint s0 = s3_[0];
                uint cw0_3 = uint(m0 >> s0) & 0xFFFFu;
                uint cw0_2 = uint(m0 >> (s0 + p.k)) & 0xFFFFu;
                uint cw0_1 = uint(m0 >> (s0 + 2u * p.k)) & 0xFFFFu;
                uint cw0_0 = uint(m0 >> (s0 + 3u * p.k)) & 0xFFFFu;

                uint a1 = ld_word(tile, ig0_[1], aligned);
                uint b1 = ld_word(tile, ig1_[1], aligned);
                ulong m1 = ((ulong)a1 << 32) | (ulong)b1;
                uint s1 = s3_[1];
                uint cw1_3 = uint(m1 >> s1) & 0xFFFFu;
                uint cw1_2 = uint(m1 >> (s1 + p.k)) & 0xFFFFu;
                uint cw1_1 = uint(m1 >> (s1 + 2u * p.k)) & 0xFFFFu;
                uint cw1_0 = uint(m1 >> (s1 + 3u * p.k)) & 0xFFFFu;

                accA += x00 * exl3_decode_f(cw0_0, p.mode) + x01 * exl3_decode_f(cw0_1, p.mode);
                accA += x02 * exl3_decode_f(cw0_2, p.mode) + x03 * exl3_decode_f(cw0_3, p.mode);
                accB += x00 * exl3_decode_f(cw1_0, p.mode) + x01 * exl3_decode_f(cw1_1, p.mode);
                accB += x02 * exl3_decode_f(cw1_2, p.mode) + x03 * exl3_decode_f(cw1_3, p.mode);
            }
            tile += tstride;
            float x10 = float(simd_shuffle(x_lane, ushort(16u + r0)));
            float x11 = float(simd_shuffle(x_lane, ushort(16u + r0 + 1u)));
            float x12 = float(simd_shuffle(x_lane, ushort(16u + r0 + 8u)));
            float x13 = float(simd_shuffle(x_lane, ushort(16u + r0 + 9u)));
            {
                uint a0 = ld_word(tile, ig0_[0], aligned);
                uint b0 = ld_word(tile, ig1_[0], aligned);
                ulong m0 = ((ulong)a0 << 32) | (ulong)b0;
                uint s0 = s3_[0];
                uint cw0_3 = uint(m0 >> s0) & 0xFFFFu;
                uint cw0_2 = uint(m0 >> (s0 + p.k)) & 0xFFFFu;
                uint cw0_1 = uint(m0 >> (s0 + 2u * p.k)) & 0xFFFFu;
                uint cw0_0 = uint(m0 >> (s0 + 3u * p.k)) & 0xFFFFu;

                uint a1 = ld_word(tile, ig0_[1], aligned);
                uint b1 = ld_word(tile, ig1_[1], aligned);
                ulong m1 = ((ulong)a1 << 32) | (ulong)b1;
                uint s1 = s3_[1];
                uint cw1_3 = uint(m1 >> s1) & 0xFFFFu;
                uint cw1_2 = uint(m1 >> (s1 + p.k)) & 0xFFFFu;
                uint cw1_1 = uint(m1 >> (s1 + 2u * p.k)) & 0xFFFFu;
                uint cw1_0 = uint(m1 >> (s1 + 3u * p.k)) & 0xFFFFu;

                accA += x10 * exl3_decode_f(cw0_0, p.mode) + x11 * exl3_decode_f(cw0_1, p.mode);
                accA += x12 * exl3_decode_f(cw0_2, p.mode) + x13 * exl3_decode_f(cw0_3, p.mode);
                accB += x10 * exl3_decode_f(cw1_0, p.mode) + x11 * exl3_decode_f(cw1_1, p.mode);
                accB += x12 * exl3_decode_f(cw1_2, p.mode) + x13 * exl3_decode_f(cw1_3, p.mode);
            }
            tile += tstride;
        }
        for (; live && it < it1; it++) {
            half x_lane = xh[it * 16u + (g & 15u)];
            float x00 = float(simd_shuffle(x_lane, ushort(r0)));
            float x01 = float(simd_shuffle(x_lane, ushort(r0 + 1u)));
            float x02 = float(simd_shuffle(x_lane, ushort(r0 + 8u)));
            float x03 = float(simd_shuffle(x_lane, ushort(r0 + 9u)));

            uint a0 = ld_word(tile, ig0_[0], aligned);
            uint b0 = ld_word(tile, ig1_[0], aligned);
            ulong m0 = ((ulong)a0 << 32) | (ulong)b0;
            uint s0 = s3_[0];
            uint cw0_3 = uint(m0 >> s0) & 0xFFFFu;
            uint cw0_2 = uint(m0 >> (s0 + p.k)) & 0xFFFFu;
            uint cw0_1 = uint(m0 >> (s0 + 2u * p.k)) & 0xFFFFu;
            uint cw0_0 = uint(m0 >> (s0 + 3u * p.k)) & 0xFFFFu;

            uint a1 = ld_word(tile, ig0_[1], aligned);
            uint b1 = ld_word(tile, ig1_[1], aligned);
            ulong m1 = ((ulong)a1 << 32) | (ulong)b1;
            uint s1 = s3_[1];
            uint cw1_3 = uint(m1 >> s1) & 0xFFFFu;
            uint cw1_2 = uint(m1 >> (s1 + p.k)) & 0xFFFFu;
            uint cw1_1 = uint(m1 >> (s1 + 2u * p.k)) & 0xFFFFu;
            uint cw1_0 = uint(m1 >> (s1 + 3u * p.k)) & 0xFFFFu;

            accA += x00 * exl3_decode_f(cw0_0, p.mode) + x01 * exl3_decode_f(cw0_1, p.mode);
            accA += x02 * exl3_decode_f(cw0_2, p.mode) + x03 * exl3_decode_f(cw0_3, p.mode);
            accB += x00 * exl3_decode_f(cw1_0, p.mode) + x01 * exl3_decode_f(cw1_1, p.mode);
            accB += x02 * exl3_decode_f(cw1_2, p.mode) + x03 * exl3_decode_f(cw1_3, p.mode);
            tile += tstride;
        }
    } else {
        // Fallback pair-table path for k=7 (spans 37 bits, crosses 3 words).
        uint pk0 = pair_tab[4u * g];
        uint pk1 = pair_tab[4u * g + 1u];
        uint pk2 = pair_tab[4u * g + 2u];
        uint pk3 = pair_tab[4u * g + 3u];
        uint i0_0 = pk0 & 0xFFu;
        uint i1_0 = (pk0 >> 8) & 0xFFu;
        uint s1_0 = (pk0 >> 16) & 0xFFu;
        uint i0_1 = pk1 & 0xFFu;
        uint i1_1 = (pk1 >> 8) & 0xFFu;
        uint s1_1 = (pk1 >> 16) & 0xFFu;
        uint i0_2 = pk2 & 0xFFu;
        uint i1_2 = (pk2 >> 8) & 0xFFu;
        uint s1_2 = (pk2 >> 16) & 0xFFu;
        uint i0_3 = pk3 & 0xFFu;
        uint i1_3 = (pk3 >> 8) & 0xFFu;
        uint s1_3 = (pk3 >> 16) & 0xFFu;

        for (; live && it < it1; it++) {
            half x_lane = xh[it * 16u + (g & 15u)];
            float x00 = float(simd_shuffle(x_lane, ushort(r0)));
            float x01 = float(simd_shuffle(x_lane, ushort(r0 + 1u)));
            float x02 = float(simd_shuffle(x_lane, ushort(r0 + 8u)));
            float x03 = float(simd_shuffle(x_lane, ushort(r0 + 9u)));
            CodePair p0 = funnel_pair(tile, i0_0, i1_0, s1_0, p.k, p.mode, aligned);
            CodePair p1 = funnel_pair(tile, i0_1, i1_1, s1_1, p.k, p.mode, aligned);
            CodePair p2 = funnel_pair(tile, i0_2, i1_2, s1_2, p.k, p.mode, aligned);
            CodePair p3 = funnel_pair(tile, i0_3, i1_3, s1_3, p.k, p.mode, aligned);
            accA += x00 * float(p0.lo) + x01 * float(p0.hi);
            accA += x02 * float(p1.lo) + x03 * float(p1.hi);
            accB += x00 * float(p2.lo) + x01 * float(p2.hi);
            accB += x02 * float(p3.lo) + x03 * float(p3.hi);
            tile += tstride;
        }
    }

    // SIMD warp reduction (T3): 4 lanes per output column collapse in registers.
    accA += simd_shuffle_down(accA, 1);
    accA += simd_shuffle_down(accA, 2);
    accB += simd_shuffle_down(accB, 1);
    accB += simd_shuffle_down(accB, 2);
    if (live && (g & 3u) == 0u) {
        uint c0 = g >> 2;
        y[(tg.y * p.out_tiles + ot) * TILE + c0] = accA;
        y[(tg.y * p.out_tiles + ot) * TILE + c0 + 8u] = accB;
    }
}

// ---------------------------------------------------------------------------
// exl3_gemv4: PonyExl3 v12 MT=1 parity for rows=1 (spec/decode-20tps P1-T1).
//
// One out-tile per 128-thread threadgroup; NSG=4 simdgroups stride the in-tile
// range (`it += NSG`) with ZERO hot-loop barriers. Each lane keeps 8 static
// accumulator chains (accA[4]/accB[4]) instead of 2, hiding decode latency in
// registers; words are read directly from device (the ds4 pattern), x via
// simd_shuffle, funnel windows computed once (tile-invariant). The per-SG
// column partials collapse in-warp (4 lanes/column), then across the 4
// simdgroups through `tgw[NSG][TILE]` in one barrier at the very end. Same
// bit-level semantics as `exl3_gemv` (same funnel decode, same position map,
// same output indexing) — only the summation groups differ, which the reduce
// path already allows (token-level gates only).
// ---------------------------------------------------------------------------
#define NSG 4u
kernel void exl3_gemv4(device const half *xh [[buffer(0)]],
                       device const ushort *trellis [[buffer(1)]],
                       device const uint *pair_tab [[buffer(2)]],
                       device float *y [[buffer(3)]],
                       constant Exl3Params &p [[buffer(4)]],
                       uint2 tg [[threadgroup_position_in_grid]],
                       uint2 thr [[thread_position_in_threadgroup]]) {
    uint tid = thr.x;
    uint sg = tid >> 5u;
    uint g = tid & 31u;
    uint ot = tg.x;
    uint nsplit = p.nsplit;
    uint chunk = (p.in_tiles + nsplit - 1u) / nsplit;
    uint it0 = tg.y * chunk;
    uint it1 = min(p.in_tiles, it0 + chunk);
    bool live = ot < p.out_tiles;

    uint r0 = (g & 3u) * 2u;
    bool aligned = p.aligned != 0u;

    float accA[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float accB[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const device ushort *tile =
        trellis + ((ulong)min(it0, p.in_tiles) * p.out_tiles + ot) * p.twords * 2u;
    const ulong tstride = (ulong)p.out_tiles * p.twords * 2u;

    if (p.k != 7u) {
        uint ig0_[2];
        uint ig1_[2];
        uint s3_[2];
        int tw = int(p.twords);
        for (uint gg = 0u; gg < 2u; gg++) {
            uint c0 = g * 8u + gg * 4u;
            int e_last = int(c0 + 4u) * int(p.k) + 256 * int(p.k);
            int i_end = (e_last - 1) / 32;
            ig1_[gg] = uint(((i_end % tw) + tw) % tw);
            ig0_[gg] = uint((((i_end - 1) % tw) + tw) % tw);
            s3_[gg] = uint((i_end + 1) * 32 - e_last);
        }
        for (uint it = it0 + sg; live && it < it1; it += NSG) {
            half x_lane = xh[it * 16u + g];
            float x00 = float(simd_shuffle(x_lane, ushort(r0)));
            float x01 = float(simd_shuffle(x_lane, ushort(r0 + 1u)));
            float x02 = float(simd_shuffle(x_lane, ushort(r0 + 8u)));
            float x03 = float(simd_shuffle(x_lane, ushort(r0 + 9u)));
            {
                uint a0 = ld_word(tile, ig0_[0], aligned);
                uint b0 = ld_word(tile, ig1_[0], aligned);
                ulong m0 = ((ulong)a0 << 32) | (ulong)b0;
                uint s0 = s3_[0];
                uint cw0_3 = uint(m0 >> s0) & 0xFFFFu;
                uint cw0_2 = uint(m0 >> (s0 + p.k)) & 0xFFFFu;
                uint cw0_1 = uint(m0 >> (s0 + 2u * p.k)) & 0xFFFFu;
                uint cw0_0 = uint(m0 >> (s0 + 3u * p.k)) & 0xFFFFu;

                uint a1 = ld_word(tile, ig0_[1], aligned);
                uint b1 = ld_word(tile, ig1_[1], aligned);
                ulong m1 = ((ulong)a1 << 32) | (ulong)b1;
                uint s1 = s3_[1];
                uint cw1_3 = uint(m1 >> s1) & 0xFFFFu;
                uint cw1_2 = uint(m1 >> (s1 + p.k)) & 0xFFFFu;
                uint cw1_1 = uint(m1 >> (s1 + 2u * p.k)) & 0xFFFFu;
                uint cw1_0 = uint(m1 >> (s1 + 3u * p.k)) & 0xFFFFu;

                accA[0] += x00 * exl3_decode_f(cw0_0, p.mode);
                accA[1] += x01 * exl3_decode_f(cw0_1, p.mode);
                accA[2] += x02 * exl3_decode_f(cw0_2, p.mode);
                accA[3] += x03 * exl3_decode_f(cw0_3, p.mode);
                accB[0] += x00 * exl3_decode_f(cw1_0, p.mode);
                accB[1] += x01 * exl3_decode_f(cw1_1, p.mode);
                accB[2] += x02 * exl3_decode_f(cw1_2, p.mode);
                accB[3] += x03 * exl3_decode_f(cw1_3, p.mode);
            }
            tile += tstride;
        }
    } else {
        // k=7: pair-table funnel path (spans 37 bits, crosses 3 words).
        uint pk0 = pair_tab[4u * g];
        uint pk1 = pair_tab[4u * g + 1u];
        uint pk2 = pair_tab[4u * g + 2u];
        uint pk3 = pair_tab[4u * g + 3u];
        uint i0_0 = pk0 & 0xFFu;
        uint i1_0 = (pk0 >> 8) & 0xFFu;
        uint s1_0 = (pk0 >> 16) & 0xFFu;
        uint i0_1 = pk1 & 0xFFu;
        uint i1_1 = (pk1 >> 8) & 0xFFu;
        uint s1_1 = (pk1 >> 16) & 0xFFu;
        uint i0_2 = pk2 & 0xFFu;
        uint i1_2 = (pk2 >> 8) & 0xFFu;
        uint s1_2 = (pk2 >> 16) & 0xFFu;
        uint i0_3 = pk3 & 0xFFu;
        uint i1_3 = (pk3 >> 8) & 0xFFu;
        uint s1_3 = (pk3 >> 16) & 0xFFu;
        for (uint it = it0 + sg; live && it < it1; it += NSG) {
            half x_lane = xh[it * 16u + (g & 15u)];
            float x00 = float(simd_shuffle(x_lane, ushort(r0)));
            float x01 = float(simd_shuffle(x_lane, ushort(r0 + 1u)));
            float x02 = float(simd_shuffle(x_lane, ushort(r0 + 8u)));
            float x03 = float(simd_shuffle(x_lane, ushort(r0 + 9u)));
            CodePair p0 = funnel_pair(tile, i0_0, i1_0, s1_0, p.k, p.mode, aligned);
            CodePair p1 = funnel_pair(tile, i0_1, i1_1, s1_1, p.k, p.mode, aligned);
            CodePair p2 = funnel_pair(tile, i0_2, i1_2, s1_2, p.k, p.mode, aligned);
            CodePair p3 = funnel_pair(tile, i0_3, i1_3, s1_3, p.k, p.mode, aligned);
            accA[0] += x00 * float(p0.lo);
            accA[1] += x01 * float(p0.hi);
            accA[2] += x02 * float(p1.lo);
            accA[3] += x03 * float(p1.hi);
            accB[0] += x00 * float(p2.lo);
            accB[1] += x01 * float(p2.hi);
            accB[2] += x02 * float(p3.lo);
            accB[3] += x03 * float(p3.hi);
            tile += tstride;
        }
    }

    // In-warp collapse: 4 lanes per output column (unchanged position map)...
    float a = (accA[0] + accA[1]) + (accA[2] + accA[3]);
    float b = (accB[0] + accB[1]) + (accB[2] + accB[3]);
    a += simd_shuffle_down(a, 1);
    a += simd_shuffle_down(a, 2);
    b += simd_shuffle_down(b, 1);
    b += simd_shuffle_down(b, 2);

    // ...then across the 4 simdgroups (disjoint in-tile subsets).
    threadgroup float tgw[NSG][TILE];
    if (live && (g & 3u) == 0u) {
        tgw[sg][g >> 2] = a;
        tgw[sg][(g >> 2) + 8u] = b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (live && tid < TILE) {
        float s = 0.0f;
        for (uint gg = 0u; gg < NSG; gg++) {
            s += tgw[gg][tid];
        }
        y[((ulong)tg.y * p.out_tiles + ot) * TILE + tid] = s;
    }
}

// ---------------------------------------------------------------------------
// T2.5: sum the K-split partials: y[ot, g] = Σ_s partial[s, ot, g].
// grid = (out_tiles), 32 threads (first TILE active). f32 adds reassociate
// relative to the unsplit loop — token-level gates only (argmax/ulp).
// ---------------------------------------------------------------------------
kernel void exl3_gemv_reduce(device const float *partial [[buffer(0)]],
                             device float *y [[buffer(1)]],
                             constant uint2 &p [[buffer(2)]], // x = out_tiles, y = nsplit
                             uint tg [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]]) {
    if (tid >= TILE) {
        return;
    }
    float s = 0.0f;
    for (uint sp = 0u; sp < p.y; sp++) {
        s += partial[(sp * p.x + tg) * TILE + tid];
    }
    y[tg * TILE + tid] = s;
}

// ---------------------------------------------------------------------------
// out = had_r(y) * svh    (one 128-block per threadgroup; y is the f32 GEMV out)
// ---------------------------------------------------------------------------
// grid = (row_len/128, rows), same layout as exl3_had_in: y is [rows, row_len]
// f32 (the batched GEMM's intermediate), out is [rows, row_len] f16.
kernel void exl3_had_out(device const float *y [[buffer(0)]],
                         device const half *svh [[buffer(1)]],
                         device half *out [[buffer(2)]],
                         constant HadParams &hp [[buffer(3)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint2 thr [[thread_position_in_threadgroup]]) {
    threadgroup float sh[HAD];
    uint t = thr.x;
    uint base = tg.y * hp.row_len + tg.x * HAD;
    // The reference rounds the GEMV slab to f16 before the final Hadamard.
    // svh is ONE row of scales shared by every batch row (see exl3_had_in).
    sh[t] = float((half)y[base + t]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stage = 1u; stage < HAD; stage <<= 1u) {
        if ((t & stage) == 0u) {
            float a = sh[t];
            float b = sh[t + stage];
            sh[t] = a + b;
            sh[t + stage] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[base + t] = (half)(sh[t] * as_type<float>(0x3DB504F3u) * float(svh[tg.x * HAD + t]));
}

// ---------------------------------------------------------------------------
// y = xh @ decode(trellis) for `rows` rows (T3.2 prefill GEMM).
//
// One threadgroup = one out-tile × a chunk of RGT rows. The trellis decode
// depends only on the weight tile and the lane's pair descriptors, so each
// lane decodes its 8 halves **once per in-tile** and reuses them for all RGT
// rows — that reuse is the whole point of the batched path (decode cost / RGT).
// Accumulators live in registers (accA/B[RGT]); the 4-lane column reduction
// collapses through threadgroup memory in one barrier at the end, writing
// y[row * out_features + col]. Rows beyond `rows` are skipped warp-uniformly.
//
// RGT trades register pressure against redundant weight streams: grid.y =
// ceil(rows/RGT), and every row-chunk re-reads the same trellis for its
// out-tile. RGT=8 is the measured sweet spot on M4 Pro — RGT=32 spilled
// registers and collapsed span prefill from 15 t/s to 5.5 t/s (2026-09-22).
// MUST match `RGT` in exl3.rs.
constant uint RGT = 8u;

kernel void exl3_gemm(device const half *xh [[buffer(0)]],
                      device const ushort *trellis [[buffer(1)]],
                      device const uint *pair_tab [[buffer(2)]],
                      device float *y [[buffer(3)]],
                      constant Exl3GemmParams &p [[buffer(4)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint2 thr [[thread_position_in_threadgroup]]) {
    threadgroup float shA[RGT * 32];
    threadgroup float shB[RGT * 32];

    uint tid = thr.x;
    uint g = tid & 31u;
    uint ot = tg.x;
    uint row0 = tg.y * RGT;
    bool live = ot < p.out_tiles && row0 < p.rows;
    uint n_here = 0u;
    if (row0 < p.rows) {
        n_here = min(RGT, p.rows - row0);
    }

    uint nin = p.in_tiles;
    uint nout = p.out_tiles;
    uint K = nin * TILE;
    uint N = nout * TILE;

    uint cA = g >> 2;
    uint cB = cA + 8u;
    uint r0 = (g & 3u) * 2u;

    bool aligned = p.aligned != 0u;
    uint pk0 = pair_tab[4u * g];
    uint pk1 = pair_tab[4u * g + 1u];
    uint pk2 = pair_tab[4u * g + 2u];
    uint pk3 = pair_tab[4u * g + 3u];

    float accA[RGT];
    float accB[RGT];
    for (uint r = 0; r < RGT; r++) {
        accA[r] = 0.0f;
        accB[r] = 0.0f;
    }

    for (uint it = 0; live && it < nin; it++) {
        device const ushort *tile =
            trellis + (ulong)(it * nout + ot) * p.twords * 2u;
        CodePair p0 = funnel_pair(
            tile, pk0 & 0xFFu, (pk0 >> 8) & 0xFFu, (pk0 >> 16) & 0xFFu, p.k, p.mode, aligned
        );
        CodePair p1 = funnel_pair(
            tile, pk1 & 0xFFu, (pk1 >> 8) & 0xFFu, (pk1 >> 16) & 0xFFu, p.k, p.mode, aligned
        );
        CodePair p2 = funnel_pair(
            tile, pk2 & 0xFFu, (pk2 >> 8) & 0xFFu, (pk2 >> 16) & 0xFFu, p.k, p.mode, aligned
        );
        CodePair p3 = funnel_pair(
            tile, pk3 & 0xFFu, (pk3 >> 8) & 0xFFu, (pk3 >> 16) & 0xFFu, p.k, p.mode, aligned
        );

        for (uint r = 0; r < n_here; r++) {
            uint xrow = (row0 + r) * K + it * TILE + r0;
            float x00 = float(xh[xrow]);
            float x01 = float(xh[xrow + 1u]);
            float x02 = float(xh[xrow + 8u]);
            float x03 = float(xh[xrow + 9u]);
            accA[r] += x00 * float(p0.lo) + x01 * float(p0.hi);
            accA[r] += x02 * float(p1.lo) + x03 * float(p1.hi);
            accB[r] += x00 * float(p2.lo) + x01 * float(p2.hi);
            accB[r] += x02 * float(p3.lo) + x03 * float(p3.hi);
        }
    }

    for (uint r = 0; r < n_here; r++) {
        shA[r * 32u + tid] = accA[r];
        shB[r * 32u + tid] = accB[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (live && g < 16u) {
        for (uint r = 0; r < n_here; r++) {
            uint base = r * 32u;
            float s = 0.0f;
            if (g < 8u) {
                for (uint q = 0; q < 4u; q++) {
                    s += shA[base + (g << 2) + q];
                }
            } else {
                for (uint q = 0; q < 4u; q++) {
                    s += shB[base + ((g - 8u) << 2) + q];
                }
            }
            y[(ulong)(row0 + r) * N + ot * TILE + g] = s;
        }
    }
}

// ---------------------------------------------------------------------------
// simdgroup tensor-core trellis GEMM — dense port of PonyExl3 v19c
// `_mm_seg_source` (reference/PonyExl3/ponyexl3/mlx/gemv_metal.py:1544).
//
// One threadgroup = one 64×64 output tile = 4 out-tiles × 64 token rows
// (_MM_BM/_MM_BN = 64). 128 threads = 4 simdgroups in a 2×2 grid
// (sgid = tid>>5, sgr = sgid>>1, sgc = sgid&1); each owns a 32×32 subtile of
// C[4][4] simdgroup_float8x8 accumulators (f32). Per 32-deep k-stage (2
// in-tiles) the 128 threads decode 2 in-tiles × 4 out-tiles ONCE into
// `wblk` (32 in-rows × 64 out-cols, stride 72 — _MM_WS), then 4×
// simdgroup_half8x8 B fragments × device-direct A fragments from `xh` run
// f16 MMA → f32 accumulate. That decode-reuse + tensor MMA replaces the
// scalar FMA+RMW of exl3_gemm (RGT=8 measured 15 t/s prefill; target ≥200).
//
// Decode precompute matches `_decode_full_t_source` (v13t) / v19c verbatim:
//   src = inv_perm[(2*rp+j)*16 + dcol]; t = src>>1;
//   sh  = ((b2-1)/32+1)*32 - b2 + ((src&1) ? 0 : K_BITS);
//   cw  = (merge(i0,i1) >> sh) & 0xFFFF;
// (even src = lo codeword gets the +K_BITS; odd = hi as-is — the funnel
// pairing of exl3_gemv, expressed as one per-codeword shift.)
//
// wblk row = it2*16 + 2*rp + j = row-major in-feature within the stage;
// col = lt*16 + dcol = row-major out-feature within the 64-wide block. So
// `simdgroup_load(bf, wrow, 72)` yields B[k, n] directly — the same
// row-major (in, out) convention xh columns and y use; no transpose.
//
// Early `return` on a dead threadgroup is safe (row0 is threadgroup-uniform,
// so all 128 threads exit together — no barrier is left waiting). The A
// fragment loads are device-direct up to a 64-row boundary: the host pads
// the `xh` scratch to `rows.div_ceil(64)*64` rows, and the store masks
// `gr < blk_len`, so pad rows multiply into discarded outputs only.
//
// Host gate (`sg_eligible` in exl3.rs): rows ≥ 2, in_tiles even,
// out_tiles % 4 == 0, trellis byte_offset % 4 == 0 (funnel words are read as
// u32 pairs — same contract as the reference). lm_head is 2-mod-4 but always
// rows == 1, so it stays on the GEMV path.
constant uint SG_BM = 64u;    // token rows per threadgroup   (_MM_BM)
constant uint SG_BN = 64u;    // out cols per threadgroup     (_MM_BN)
constant uint SG_WS = 72u;    // wblk row stride in halfs     (_MM_WS)
constant uint SG_THREADS = 128u;

kernel void exl3_gemm_sg(device const half *xh [[buffer(0)]],
                         device const ushort *trellis [[buffer(1)]],
                         device const uint *inv_perm [[buffer(2)]],
                         device float *y [[buffer(3)]], // y (nsplit==1) or partial
                         constant Exl3GemmSgParams &p [[buffer(4)]],
                         uint3 tg3 [[threadgroup_position_in_grid]],
                         // uint3 (not uint2): MSL requires every attribute
                         // arg to be the same vector width as tg3.
                         uint3 thr [[thread_position_in_threadgroup]]) {
    threadgroup half wblk[32u * SG_WS]; // 32 in × 64 out, stride 72 (4.5 KB)
    threadgroup float csc[4u * 64u];    // per-simdgroup 8×8 store scratch (1 KB)

    uint tid = thr.x;
    uint lane = tid & 31u;
    uint sgid = tid >> 5u;
    uint sgr = sgid >> 1u; // 0..1 → token rows  0..31 / 32..63
    uint sgc = sgid & 1u;  // 0..1 → out cols    0..31 / 32..31

    uint ot0 = tg3.x * 4u;
    uint row0 = tg3.y * SG_BM;
    if (row0 >= p.rows) {
        return; // threadgroup-uniform: safe before any barrier
    }

    uint nin = p.in_tiles;
    uint nout = p.out_tiles;
    uint K = nin * TILE;
    uint N = nout * TILE;
    uint stages = nin >> 1u; // host gates in_tiles even (≥ 2)
    // T3.10: this grid.z slice of the stage range. nsplit == 1 → ks_lo 0,
    // ks_hi stages — the old loop exactly.
    uint nsplit = p.nsplit;
    uint chunk = (stages + nsplit - 1u) / nsplit;
    uint ks_lo = tg3.z * chunk;
    uint ks_hi = min(ks_lo + chunk, stages);
    // Slice base: y (offset 0) or partial[tg.z] (row-padded pitch).
    device float *dst = y + (ulong)tg3.z * p.stride;
    uint blk_len = p.rows - row0; // live token rows in this TG (1..=64)

    // Decode-position precompute (fixed per thread; `_decode_full_t` mapping).
    uint dcol = tid >> 3u; // out col within tile (0..15)
    uint rp = tid & 7u;    // in row-pair within tile (0..7)
    uint i0_[2];
    uint i1_[2];
    uint sh_[2];
    for (uint j = 0u; j < 2u; j++) {
        uint src = inv_perm[(2u * rp + j) * 16u + dcol];
        uint t = src >> 1u;
        int b0 = int(t) * 2 * int(p.k) + int(p.k) - 16 + 256 * int(p.k);
        int b2 = b0 + int(p.k) + 16;
        i0_[j] = uint(b0 / 32) % p.twords;
        i1_[j] = uint((b2 - 1) / 32) % p.twords;
        sh_[j] =
            uint(((b2 - 1) / 32 + 1) * 32 - b2) + ((src & 1u) ? 0u : p.k);
    }

    simdgroup_float8x8 C[4][4];
    for (uint r = 0u; r < 4u; r++) {
        for (uint c = 0u; c < 4u; c++) {
            C[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    const device half *xrow =
        xh + (ulong)(row0 + sgr * 32u) * K;
    bool aligned = p.aligned != 0u; // host gates byte_offset % 4 == 0

    for (uint ks = ks_lo; ks < ks_hi; ks++) {
        // Previous stage's B fragment loads must finish before overwrite.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // T3.9 phase-split funnel: issue all 32 global loads for the stage
        // back-to-back (independent addresses → memory-level parallelism),
        // then run the shift/decode/write phase as pure ALU. The old
        // load-decode-load-decode chain exposed the ~300-cycle word-load
        // latency serially while C's 32 f32/lane constrained scheduling —
        // the 20.1% tax T3.8 measured (real 11.79 vs ceiling 9.42, isolated
        // fill only 0.21 ms). Same loads, same order → bit-identical.
        uint la0[8]; // it2 * 4 + lt, bit index 0
        uint lb0[8];
        uint la1[8]; // it2 * 4 + lt, bit index 1
        uint lb1[8];
        for (uint it2 = 0u; it2 < 2u; it2++) {
            uint tk = ks * 2u + it2;
            for (uint lt = 0u; lt < 4u; lt++) {
                device const uint *words = (device const uint *)trellis +
                    ((ulong)tk * nout + ot0 + lt) * p.twords;
                uint g = it2 * 4u + lt;
                la0[g] = ld_word((device const ushort *)words, i0_[0], aligned);
                lb0[g] = ld_word((device const ushort *)words, i1_[0], aligned);
                la1[g] = ld_word((device const ushort *)words, i0_[1], aligned);
                lb1[g] = ld_word((device const ushort *)words, i1_[1], aligned);
            }
        }
        for (uint j = 0u; j < 2u; j++) {
            for (uint it2 = 0u; it2 < 2u; it2++) {
                for (uint lt = 0u; lt < 4u; lt++) {
                    uint g = it2 * 4u + lt;
                    uint wa = j == 0u ? la0[g] : la1[g];
                    uint wb = j == 0u ? lb0[g] : lb1[g];
                    uint cw = (uint)((((ulong)wa) << 32 | (ulong)wb) >>
                                     sh_[j]) & 0xFFFFu;
                    wblk[(it2 * 16u + 2u * rp + j) * SG_WS + lt * 16u + dcol] =
                        exl3_decode(cw, p.mode);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k2 = 0u; k2 < 4u; k2++) {
            simdgroup_half8x8 bf0;
            simdgroup_half8x8 bf1;
            simdgroup_half8x8 bf2;
            simdgroup_half8x8 bf3;
            const threadgroup half *wrow = &wblk[k2 * 8u * SG_WS + sgc * 32u];
            simdgroup_load(bf0, wrow, SG_WS);
            simdgroup_load(bf1, wrow + 8u, SG_WS);
            simdgroup_load(bf2, wrow + 16u, SG_WS);
            simdgroup_load(bf3, wrow + 24u, SG_WS);
            for (uint r = 0u; r < 4u; r++) {
                simdgroup_half8x8 a;
                simdgroup_load(a, xrow + (ulong)(r * 8u) * K + ks * 32u + k2 * 8u,
                               K);
                simdgroup_multiply_accumulate(C[r][0], a, bf0, C[r][0]);
                simdgroup_multiply_accumulate(C[r][1], a, bf1, C[r][1]);
                simdgroup_multiply_accumulate(C[r][2], a, bf2, C[r][2]);
                simdgroup_multiply_accumulate(C[r][3], a, bf3, C[r][3]);
            }
        }
    }

    // Store: f32 fragments → tg scratch → y, masked to the live rows. The
    // reference round-trips through scratch to cast f16; y is f32 here
    // (exl3_had_out rounds), so the scatter writes straight through. Only
    // simdgroup_barriers: the ragged-tail `break` is sgr-uniform within a
    // simdgroup, so the 4 simdgroups may exit the loop at different r.
    // T3.10: `dst` is y (nsplit==1, tg.z=0 → offset 0) or this split's
    // partial slice — row-padded pitch keeps slice bases aligned to 64 rows.
    uint cb0 = ot0 * TILE + sgc * 32u;
    uint rbase = sgr * 32u;
    for (uint r = 0u; r < 4u; r++) {
        if (rbase + r * 8u >= blk_len) {
            break;
        }
        for (uint cf = 0u; cf < 4u; cf++) {
            simdgroup_store(C[r][cf], &csc[sgid * 64u], 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lane; i < 64u; i += 32u) {
                uint rr = i >> 3u;
                uint cc = i & 7u;
                uint gr = rbase + r * 8u + rr;
                if (gr < blk_len) {
                    dst[(ulong)(row0 + gr) * N + cb0 + cf * 8u + cc] =
                        csc[sgid * 64u + i];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// T3.10: y[i] = Σ_s partial[s, i] after a K-split sg GEMM. Fixed
// s-ascending order per element → deterministic run-to-run (greedy decode
// determinism forbids atomics). `p` = {nsplit, stride elements, live total}.
// Only live elements [0, rows*N) are read — padded store rows never written.
kernel void exl3_gemm_sg_reduce(device const float *partial [[buffer(0)]],
                                device float *y [[buffer(1)]],
                                constant uint3 &p [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= p.z) {
        return;
    }
    float s = 0.0f;
    for (uint sp = 0u; sp < p.x; sp++) {
        s += partial[(ulong)sp * p.y + gid];
    }
    y[gid] = s;
}

// ---------------------------------------------------------------------------
// T3.8 probe — ceiling: exl3_gemm_sg with the funnel decode replaced by a
// direct device load of a pre-decoded half matrix `w[N, K]` (row-major,
// w[col*K + k]). Grid, thread mapping, wblk fill layout, both barriers, the
// MMA k-loop and the store are IDENTICAL to exl3_gemm_sg — only the source
// of the wblk halves differs. tax = 1 − t(real sg) / t(this kernel) isolates
// the decode+staging cost. Test-only: pipelines are built locally in
// exl3.rs tests; production never binds this kernel.
//   buffer(0) xh [rows, K] f16 | buffer(1) w [N, K] f16
//   buffer(2) y  [rows, N] f32 | buffer(3) Exl3GemmParams
// ---------------------------------------------------------------------------
kernel void exl3_gemm_sg_ceiling(device const half *xh [[buffer(0)]],
                                 device const half *w [[buffer(1)]],
                                 device float *y [[buffer(2)]],
                                 constant Exl3GemmParams &p [[buffer(3)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint2 thr [[thread_position_in_threadgroup]]) {
    threadgroup half wblk[32u * SG_WS];
    threadgroup float csc[4u * 64u];

    uint tid = thr.x;
    uint lane = tid & 31u;
    uint sgid = tid >> 5u;
    uint sgr = sgid >> 1u;
    uint sgc = sgid & 1u;

    uint ot0 = tg.x * 4u;
    uint row0 = tg.y * SG_BM;
    if (row0 >= p.rows) {
        return;
    }

    uint nin = p.in_tiles;
    uint nout = p.out_tiles;
    uint K = nin * TILE;
    uint N = nout * TILE;
    uint stages = nin >> 1u;
    uint blk_len = p.rows - row0;

    uint dcol = tid >> 3u;
    uint rp = tid & 7u;

    simdgroup_float8x8 C[4][4];
    for (uint r = 0u; r < 4u; r++) {
        for (uint c = 0u; c < 4u; c++) {
            C[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    const device half *xrow = xh + (ulong)(row0 + sgr * 32u) * K;

    for (uint ks = 0u; ks < stages; ks++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint it2 = 0u; it2 < 2u; it2++) {
            uint tk = ks * 2u + it2;
            uint k0 = ks * 32u + it2 * 16u + 2u * rp;
            uint base = (it2 * 16u + 2u * rp) * SG_WS;
            for (uint lt = 0u; lt < 4u; lt++) {
                uint col = ot0 * TILE + lt * 16u + dcol;
                device const half *wcol = w + (ulong)col * K + k0;
                wblk[base + lt * 16u + dcol] = wcol[0];
                wblk[base + SG_WS + lt * 16u + dcol] = wcol[1];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k2 = 0u; k2 < 4u; k2++) {
            simdgroup_half8x8 bf0;
            simdgroup_half8x8 bf1;
            simdgroup_half8x8 bf2;
            simdgroup_half8x8 bf3;
            const threadgroup half *wrow = &wblk[k2 * 8u * SG_WS + sgc * 32u];
            simdgroup_load(bf0, wrow, SG_WS);
            simdgroup_load(bf1, wrow + 8u, SG_WS);
            simdgroup_load(bf2, wrow + 16u, SG_WS);
            simdgroup_load(bf3, wrow + 24u, SG_WS);
            for (uint r = 0u; r < 4u; r++) {
                simdgroup_half8x8 a;
                simdgroup_load(a, xrow + (ulong)(r * 8u) * K + ks * 32u + k2 * 8u,
                               K);
                simdgroup_multiply_accumulate(C[r][0], a, bf0, C[r][0]);
                simdgroup_multiply_accumulate(C[r][1], a, bf1, C[r][1]);
                simdgroup_multiply_accumulate(C[r][2], a, bf2, C[r][2]);
                simdgroup_multiply_accumulate(C[r][3], a, bf3, C[r][3]);
            }
        }
    }

    uint cb0 = ot0 * TILE + sgc * 32u;
    uint rbase = sgr * 32u;
    for (uint r = 0u; r < 4u; r++) {
        if (rbase + r * 8u >= blk_len) {
            break;
        }
        for (uint cf = 0u; cf < 4u; cf++) {
            simdgroup_store(C[r][cf], &csc[sgid * 64u], 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lane; i < 64u; i += 32u) {
                uint rr = i >> 3u;
                uint cc = i & 7u;
                uint gr = rbase + r * 8u + rr;
                if (gr < blk_len) {
                    y[(ulong)(row0 + gr) * N + cb0 + cf * 8u + cc] =
                        csc[sgid * 64u + i];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ---------------------------------------------------------------------------
// T3.8 probe — decode-only: the funnel-decode + wblk staging + per-stage
// barriers of exl3_gemm_sg, with the MMA k-loop and store skipped. Times the
// decode/staging half of the pipeline in isolation (barrier structure per
// stage preserved so the staging-vs-MMA overlap behavior matches). Writes
// nothing; safe to exit after the stage loop (all threads leave together).
// Test-only. buffer(0) trellis | buffer(1) inv_perm | buffer(2) params
// ---------------------------------------------------------------------------
kernel void exl3_gemm_sg_decode_only(device const ushort *trellis [[buffer(0)]],
                                     device const uint *inv_perm [[buffer(1)]],
                                     constant Exl3GemmParams &p [[buffer(2)]],
                                     uint2 tg [[threadgroup_position_in_grid]],
                                     uint2 thr [[thread_position_in_threadgroup]]) {
    threadgroup half wblk[32u * SG_WS];

    uint tid = thr.x;
    uint ot0 = tg.x * 4u;
    uint row0 = tg.y * SG_BM;
    if (row0 >= p.rows) {
        return;
    }

    uint nin = p.in_tiles;
    uint nout = p.out_tiles;
    uint stages = nin >> 1u;

    uint dcol = tid >> 3u;
    uint rp = tid & 7u;
    uint i0_[2];
    uint i1_[2];
    uint sh_[2];
    for (uint j = 0u; j < 2u; j++) {
        uint src = inv_perm[(2u * rp + j) * 16u + dcol];
        uint t = src >> 1u;
        int b0 = int(t) * 2 * int(p.k) + int(p.k) - 16 + 256 * int(p.k);
        int b2 = b0 + int(p.k) + 16;
        i0_[j] = uint(b0 / 32) % p.twords;
        i1_[j] = uint((b2 - 1) / 32) % p.twords;
        sh_[j] =
            uint(((b2 - 1) / 32 + 1) * 32 - b2) + ((src & 1u) ? 0u : p.k);
    }
    bool aligned = p.aligned != 0u;

    for (uint ks = 0u; ks < stages; ks++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint it2 = 0u; it2 < 2u; it2++) {
            uint tk = ks * 2u + it2;
            for (uint lt = 0u; lt < 4u; lt++) {
                device const uint *words =
                    (device const uint *)trellis +
                    ((ulong)tk * nout + ot0 + lt) * p.twords;
                for (uint j = 0u; j < 2u; j++) {
                    uint a = ld_word((device const ushort *)words, i0_[j], aligned);
                    uint b = ld_word((device const ushort *)words, i1_[j], aligned);
                    uint cw =
                        (uint)((((ulong)a) << 32 | (ulong)b) >> sh_[j]) & 0xFFFFu;
                    wblk[(it2 * 16u + 2u * rp + j) * SG_WS + lt * 16u + dcol] =
                        exl3_decode(cw, p.mode);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---------------------------------------------------------------------------
// T3.8 probe — MMA peak: fill wblk ONCE (stage-0 geometry, direct half loads
// like the ceiling kernel), then run R identical MMA passes over the SAME
// hot fragments (B in threadgroup memory, A re-read from L2-resident xh).
// No weight streaming, no decode, no per-stage barriers inside the rep loop —
// this is the sustained simdgroup f16→f32 MMA rate of the GPU for the
// production tile configuration. If ≪ what 200 t/s needs (~9.45 TFLOP/s
// average), the sg structure cannot reach the target by staging tweaks.
// p.rows sets the grid (rows/64 × out_tiles/4); R is fixed below.
// Test-only. buffer(0) xh | buffer(1) w | buffer(2) params (y unused)
// ---------------------------------------------------------------------------
constant uint PEAK_REPS = 32u;

kernel void exl3_gemm_sg_mma_peak(device const half *xh [[buffer(0)]],
                                  device const half *w [[buffer(1)]],
                                  device float *y [[buffer(2)]],
                                  constant Exl3GemmParams &p [[buffer(3)]],
                                  uint2 tg [[threadgroup_position_in_grid]],
                                  uint2 thr [[thread_position_in_threadgroup]]) {
    threadgroup half wblk[32u * SG_WS];
    threadgroup float csc[4u * 64u];

    uint tid = thr.x;
    uint lane = tid & 31u;
    uint sgid = tid >> 5u;
    uint sgr = sgid >> 1u;
    uint sgc = sgid & 1u;

    uint ot0 = tg.x * 4u;
    uint row0 = tg.y * SG_BM;
    if (row0 >= p.rows) {
        return;
    }

    uint nin = p.in_tiles;
    uint nout = p.out_tiles;
    uint K = nin * TILE;
    uint N = nout * TILE;
    uint blk_len = p.rows - row0;

    uint dcol = tid >> 3u;
    uint rp = tid & 7u;

    // Fill wblk with stage 0 (k = 0..31) from the pre-decoded w.
    for (uint it2 = 0u; it2 < 2u; it2++) {
        uint k0 = it2 * 16u + 2u * rp;
        uint base = (it2 * 16u + 2u * rp) * SG_WS;
        for (uint lt = 0u; lt < 4u; lt++) {
            uint col = ot0 * TILE + lt * 16u + dcol;
            device const half *wcol = w + (ulong)col * K + k0;
            wblk[base + lt * 16u + dcol] = wcol[0];
            wblk[base + SG_WS + lt * 16u + dcol] = wcol[1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 C[4][4];
    for (uint r = 0u; r < 4u; r++) {
        for (uint c = 0u; c < 4u; c++) {
            C[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    const device half *xrow = xh + (ulong)(row0 + sgr * 32u) * K;

    for (uint rep = 0u; rep < PEAK_REPS; rep++) {
        for (uint k2 = 0u; k2 < 4u; k2++) {
            simdgroup_half8x8 bf0;
            simdgroup_half8x8 bf1;
            simdgroup_half8x8 bf2;
            simdgroup_half8x8 bf3;
            const threadgroup half *wrow = &wblk[k2 * 8u * SG_WS + sgc * 32u];
            simdgroup_load(bf0, wrow, SG_WS);
            simdgroup_load(bf1, wrow + 8u, SG_WS);
            simdgroup_load(bf2, wrow + 16u, SG_WS);
            simdgroup_load(bf3, wrow + 24u, SG_WS);
            for (uint r = 0u; r < 4u; r++) {
                simdgroup_half8x8 a;
                simdgroup_load(a, xrow + (ulong)(r * 8u) * K + k2 * 8u, K);
                simdgroup_multiply_accumulate(C[r][0], a, bf0, C[r][0]);
                simdgroup_multiply_accumulate(C[r][1], a, bf1, C[r][1]);
                simdgroup_multiply_accumulate(C[r][2], a, bf2, C[r][2]);
                simdgroup_multiply_accumulate(C[r][3], a, bf3, C[r][3]);
            }
        }
    }

    // Keep-alive store (same scatter as the ceiling kernel); fixed cost,
    // amortized away by PEAK_REPS in the host-side FLOP count.
    uint cb0 = ot0 * TILE + sgc * 32u;
    uint rbase = sgr * 32u;
    for (uint r = 0u; r < 4u; r++) {
        if (rbase + r * 8u >= blk_len) {
            break;
        }
        for (uint cf = 0u; cf < 4u; cf++) {
            simdgroup_store(C[r][cf], &csc[sgid * 64u], 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lane; i < 64u; i += 32u) {
                uint rr = i >> 3u;
                uint cc = i & 7u;
                uint gr = rbase + r * 8u + rr;
                if (gr < blk_len) {
                    y[(ulong)(row0 + gr) * N + cb0 + cf * 8u + cc] =
                        csc[sgid * 64u + i];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
