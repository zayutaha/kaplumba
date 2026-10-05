// `spec/decode-20tok-crux2` — independent bandwidth instrument.
//
// WHY THIS EXISTS AND WHY IT IS NOT crux_bw.metal
//
// The prior agent's probe establishes a raster-scan number (~245-253 GB/s). The
// task is to attack that, and the one axis nobody has run is **visit order**:
// every arm in the prior work reads memory front to back, because that is what a
// grid-stride loop does. The real EXL3 decode GEMV does NOT: it walks 32x32
// trellis tiles, so at any instant the grid is touching a small window of memory
// and then jumping. Short bursts plus jumps is a different DRAM access pattern
// from a raster scan, and it is the pattern the machine actually runs.
//
// Two families, both with EXACT full coverage so the CPU-computed reference
// checksum still gates them:
//
//   BURST(B): thread i of buffer b reads element
//                i + k*per + j*per*B        for k in [0,B), j in [0,J)
//             B = 1 is exactly the raster scan (the prior probe's `stream_u4`).
//             B = 24 is 24 uint4 = 384 B per thread per iteration, and 384 B is
//             exactly the payload of one 32x32 tile at 3 bpw (32*32*3 bits), so
//             one parameter sweeps "raster scan" -> "tile at a time".
//             Coverage: J*B*per == count exactly, asserted on the host.
//
//   FEISTEL: visit order is a keyed pseudo-random PERMUTATION of [0, count).
//             A balanced Feistel network over log2(count) bits is a bijection on
//             exactly [0, count) with no cycle walking, so it is O(1) per element
//             and cannot double-visit or skip. This is the "higher entropy in the
//             address stream" axis the task brief flags as untested; the prior
//             multibuffer arm tested *independence of buffers*, not *randomness
//             of order*.
//
// Checksums are order-independent (u32 addition), and the fill is a function of
// POSITION, so every arm must produce the IDENTICAL value, computed on the CPU
// from the fill. An arm that skipped, repeated or mangled a range fails the gate
// instead of reporting a plausible number.
//
// Buffers arrive in fixed slots 0..7 because metal-rs will not let a host build a
// `device uint4* const*`. Unused slots are bound to slot 0 by the host so no
// binding is null; `nbuf` in Params is what decides how many are read, and the
// checksum gate proves only those were touched.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint count;   // uint4 elements PER BUFFER
    uint per;     // threads serving one buffer
    uint burst;   // B: uint4 per thread per iteration
    uint iters;   // J = count / (per * burst)  == number of blocks
    uint odd;     // multiplier for the block permutation
    uint perm;    // 0 => blk = j (no modulo at all); 1 => blk = (j*odd) % iters
    uint nbuf;    // buffers actually read (1..8)
    uint _p0;
};

// Cheap high-entropy visit order.
//
// A 4-round Feistel was the first attempt and it is the WRONG instrument here:
// at 36.6 GB/s it was 6.4x slower than the raster arm, so it was measuring the
// dependent multiply chain, not the memory system. Any permutation cheap enough
// to stay off the critical path is required, or the arm answers the wrong
// question.
//
// `blk_perm(j) = (j * odd) % iters` is a bijection on [0, iters) whenever
// gcd(odd, iters) == 1, costs one multiply and one modulo per BURST (not per
// element), and scatters sequential blocks across the whole buffer -- exactly the
// "higher entropy in the address stream" property under test. `odd = 1` is the
// identity, so the raster arm runs the SAME instruction sequence with no special
// case: the two arms differ only in the multiplier, which is what makes the
// comparison an A/B rather than two different programs.
// `odd` MUST already be reduced mod `iters` by the host. Passing a raw 31-bit
// multiplier does NOT work: `j * odd` overflows u32 for j >= 2, and a wrapped
// product is not congruent to j*odd mod iters, so the "permutation" silently
// stops being one. That failure was caught by the exhaustive multiset proof
// (2,211,840 double-visits) and would otherwise have produced a plausible
// bandwidth for an arm that was skipping and repeating regions.
inline uint blk_perm(uint j, uint odd, uint iters) {
    return (j * odd) % iters;
}

kernel void probe(const device uint4 *s0 [[buffer(0)]],
                  const device uint4 *s1 [[buffer(1)]],
                  const device uint4 *s2 [[buffer(2)]],
                  const device uint4 *s3 [[buffer(3)]],
                  const device uint4 *s4 [[buffer(4)]],
                  const device uint4 *s5 [[buffer(5)]],
                  const device uint4 *s6 [[buffer(6)]],
                  const device uint4 *s7 [[buffer(7)]],
                  device uint *out [[buffer(8)]],
                  constant Params &p [[buffer(9)]],
                  uint gid [[thread_position_in_grid]]) {
    const uint b = gid / p.per;   // which buffer
    const uint lid = gid % p.per; // lane within that buffer

    const device uint4 *data = s0;
    if      (b == 1u) { data = s1; }
    else if (b == 2u) { data = s2; }
    else if (b == 3u) { data = s3; }
    else if (b == 4u) { data = s4; }
    else if (b == 5u) { data = s5; }
    else if (b == 6u) { data = s6; }
    else if (b == 7u) { data = s7; }

    uint4 acc = uint4(0u);
    for (uint j = 0u; j < p.iters; ++j) {
        // `perm` exists so the PERMUTATION COST can be separated from the
        // permutation EFFECT. The stride arms run a multiply + modulo per burst;
        // `raster_burst24_mod` runs the identical multiply + modulo with
        // odd == 1, i.e. identity order. Comparing a stride arm against THAT,
        // rather than against a no-modulo arm, is what makes it an A/B.
        const uint blk = p.perm == 0u ? j : blk_perm(j, p.odd, p.iters);
        const uint base = blk * p.per * p.burst;
        for (uint k = 0u; k < p.burst; ++k) {
            acc += data[base + k * p.per + lid];
        }
    }
    out[gid] = acc.x + acc.y + acc.z + acc.w;
}