// EXL3 trellis GEMV + 128-point Hadamard passes (T2.1).
//
// Mirrors the CPU reference path bit-for-bit where the arithmetic allows:
//   y = had_out(had_in(x, suh) @ decode(trellis), svh)
//
// had_in / had_out: one 128-block per thread; the f32 accumulate order over
// j = 0..128 is identical to had_r_128, and the intermediate products are
// rounded to f32 exactly like the CPU's f32 scratch. Those two passes are
// bit-exact against the CPU path.
//
// exl3_gemv: lane-per-group over the kernel-order permutation. The 256
// elements of a 16x16 tile form 32 groups of 8 kernel indices; group g maps to
// row-major rows (R0,R1,R2,R3) = ((g%4)*2, +1, +8, +9) of columns
// cA = g/4 and cB = cA + 8. Each lane therefore accumulates into exactly two
// output columns with *static* register accumulators (no dynamic indexing),
// and the per-lane reduction collapses 4 lanes per column at the end.
//
// The 4 pairs of a group (pair index t = 4g + q) come straight from the
// decoder's per-pair funnel shift; the host precomputes (i0, i1, s1) per pair
// per K in `pair_tab` so the kernel does no index math beyond the loads.

#include <metal_stdlib>
using namespace metal;

constant uint HAD = 128u;
constant uint TILE = 16u;

struct Exl3Params {
    uint in_tiles;
    uint out_tiles;
    uint k;
    // u32 words per packed tile (= 8*k). The trellis is read as 32-bit words
    // because the decoder's funnel window is defined over u32 pairs, exactly
    // like the CPU reference (`unpack_trellis_tile` packs two u16s per word).
    uint twords;
    uint mode; // 0 = Default, 1 = Mcg (Mul1 stays on the CPU path)
    uint pad0;
};

// One u32 word of the packed trellis = two little-endian u16 words.
inline uint load_word(device const ushort *tile, uint w) {
    return (uint)tile[2u * w] | ((uint)tile[2u * w + 1u] << 16);
}

inline float had_sign(uint i, uint j) {
    return (popcount(i & j) & 1u) ? -as_type<float>(0x3DB504F3u) : as_type<float>(0x3DB504F3u);
}

// EXL3 procedural codebook, bit-identical to the CPU `decode_3inst`:
//   Mcg:     x *= 0xCBAC1FED, then x = 0x3B603B60 ^ (x & 0x8FFF8FFF), then
//            one exact IEEE binary16 add of the two packed halves.
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

// ---------------------------------------------------------------------------
// xh = had_r(x * suh)   (one 128-block per threadgroup, 128 threads)
// ---------------------------------------------------------------------------
kernel void exl3_had_in(device const half *x [[buffer(0)]],
                        device const half *suh [[buffer(1)]],
                        device half *xh [[buffer(2)]],
                        uint blk [[threadgroup_position_in_grid]],
                        uint t [[thread_position_in_threadgroup]]) {
    threadgroup float sh[HAD];
    uint base = blk * HAD;
    sh[t] = float(x[base + t]) * float(suh[base + t]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    for (uint j = 0; j < HAD; j++) {
        acc += sh[j] * had_sign(j, t);
    }
    xh[base + t] = (half)acc;
}

// ---------------------------------------------------------------------------
// y = had_r_inner(xh @ inner)   (raw f32 output, one thread per out column)
// ---------------------------------------------------------------------------
kernel void exl3_gemv(device const half *xh [[buffer(0)]],
                      device const ushort *trellis [[buffer(1)]],
                      device const uint *pair_tab [[buffer(2)]],
                      device float *y [[buffer(3)]],
                      constant Exl3Params &p [[buffer(4)]],
                      uint ot [[threadgroup_position_in_grid]],
                      uint g [[thread_position_in_threadgroup]]) {
    threadgroup float shA[32];
    threadgroup float shB[32];

    uint cA = g >> 2;   // 0..7  -> output columns 0..7
    uint cB = cA + 8u;  //       -> output columns 8..15
    uint r0 = (g & 3u) * 2u;

    float accA = 0.0f;
    float accB = 0.0f;

    for (uint it = 0; it < p.in_tiles; it++) {
        // Stride is in u32 words; the tile is addressed as u16 pairs, so a
        // trellis tensor whose byte offset is only 2-aligned (real artifacts:
        // lm_head.trellis) still reads correctly.
        device const ushort *tile =
            trellis + (ulong)(it * p.out_tiles + ot) * p.twords * 2u;
        uint xrow = it * TILE + r0;
        float x00 = float(xh[xrow]);
        float x01 = float(xh[xrow + 1u]);
        float x02 = float(xh[xrow + 8u]);
        float x03 = float(xh[xrow + 9u]);
#pragma unroll
        for (uint q = 0; q < 4u; q++) {
            uint pk = pair_tab[4u * g + q];
            uint i0 = pk & 0xFFu;
            uint i1 = (pk >> 8) & 0xFFu;
            uint s1 = (pk >> 16) & 0xFFu;
            uint a = load_word(tile, i0);
            uint b = load_word(tile, i1);
            uint w1 = (uint)((((ulong)a) << 32 | (ulong)b) >> s1);
            half lo = exl3_decode((w1 >> p.k) & 0xFFFFu, p.mode);
            half hi = exl3_decode(w1 & 0xFFFFu, p.mode);
            // q = 0,2 -> rows (r0, r0+1) of column cA/cB; q = 1,3 -> rows (r0+8, r0+9)
            float v = (q & 1u) == 0u ? (x00 * float(lo) + x01 * float(hi))
                                     : (x02 * float(lo) + x03 * float(hi));
            if (q < 2u) {
                accA += v;
            } else {
                accB += v;
            }
        }
    }

    shA[g] = accA;
    shB[g] = accB;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (g < 16u) {
        float s = 0.0f;
        if (g < 8u) {
            for (uint q = 0; q < 4u; q++) {
                s += shA[(g << 2) + q];
            }
        } else {
            for (uint q = 0; q < 4u; q++) {
                s += shB[((g - 8u) << 2) + q];
            }
        }
        y[ot * TILE + g] = s;
    }
}

// ---------------------------------------------------------------------------
// out = had_r(y) * svh    (one 128-block per threadgroup; y is the f32 GEMV out)
// ---------------------------------------------------------------------------
kernel void exl3_had_out(device const float *y [[buffer(0)]],
                         device const half *svh [[buffer(1)]],
                         device half *out [[buffer(2)]],
                         uint blk [[threadgroup_position_in_grid]],
                         uint t [[thread_position_in_threadgroup]]) {
    threadgroup float sh[HAD];
    uint base = blk * HAD;
    sh[t] = float((half)y[base + t]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    for (uint j = 0; j < HAD; j++) {
        acc += sh[j] * had_sign(j, t);
    }
    out[base + t] = (half)(acc * float(svh[base + t]));
}
