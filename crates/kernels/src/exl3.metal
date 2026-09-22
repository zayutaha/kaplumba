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
// few threadgroups left to spread across 20 cores.
constant uint WARPS = 1u;

kernel void exl3_gemv(device const half *xh [[buffer(0)]],
                      device const ushort *trellis [[buffer(1)]],
                      device const uint *pair_tab [[buffer(2)]],
                      device float *y [[buffer(3)]],
                      constant Exl3Params &p [[buffer(4)]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float shA[32 * WARPS];
    threadgroup float shB[32 * WARPS];

    uint sub = 0u;
    uint g = tid & 31u;       // lane = kernel-order group
    uint ot = tg * WARPS + sub;
    bool live = ot < p.out_tiles;

    uint cA = g >> 2;   // 0..7  -> output columns 0..7
    uint cB = cA + 8u;  //       -> output columns 8..15
    uint r0 = (g & 3u) * 2u;

    bool aligned = p.aligned != 0u;
    // The 4 pair descriptors this lane needs. Measured both ways (as 4 scalar
    // loads and as one 16-byte uint4 load): indistinguishable within run-to-run
    // noise, so the simpler scalar form stays.
    uint pk0 = pair_tab[4u * g];
    uint pk1 = pair_tab[4u * g + 1u];
    uint pk2 = pair_tab[4u * g + 2u];
    uint pk3 = pair_tab[4u * g + 3u];

    float accA = 0.0f;
    float accB = 0.0f;

    // `live` is warp-uniform, so the loop is skipped (not divergent) for the
    // padding warp of the last threadgroup while every warp still reaches the
    // barrier below.
    for (uint it = 0; live && it < p.in_tiles; it++) {
        // Stride is in u32 words; the tile is addressed as u16 pairs so that a
        // 2-mod-4 trellis offset still reads correctly.
        device const ushort *tile =
            trellis + (ulong)(it * p.out_tiles + ot) * p.twords * 2u;
        uint xrow = it * TILE + r0;
        float x00 = float(xh[xrow]);
        float x01 = float(xh[xrow + 1u]);
        float x02 = float(xh[xrow + 8u]);
        float x03 = float(xh[xrow + 9u]);

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

        // q = 0,1 feed column cA from rows (r0, r0+1) and (r0+8, r0+9);
        // q = 2,3 do the same for column cB.
        accA += x00 * float(p0.lo) + x01 * float(p0.hi);
        accA += x02 * float(p1.lo) + x03 * float(p1.hi);
        accB += x00 * float(p2.lo) + x01 * float(p2.hi);
        accB += x02 * float(p3.lo) + x03 * float(p3.hi);
    }

    shA[tid] = accA;
    shB[tid] = accB;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (live && g < 16u) {
        uint base = sub << 5;
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
        y[ot * TILE + g] = s;
    }
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
