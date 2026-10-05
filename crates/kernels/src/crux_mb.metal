#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// `spec/decode-20tok-crux` — the one axis that could break a DRAM wall.
//
// Everything else is already closed, in this repo or in the literature it cites:
// occupancy flat over 32x (decode-bandwidth-240gb: 0.08M-2.62M loads in flight),
// load width 8/16/32 B flat, address streams 1-64 flat, working-set size
// plateauing past 4 GB, contiguity/row-locality zero effect, power/thermal not
// throttled. So the only thing left is:
//
//   AXIS M  **independent ALLOCATIONS**. `crux_bw.metal`'s stream axis carved
//           ONE buffer into S slices. Slices of a single allocation share
//           physical pages, so a DRAM channel/bank imbalance would appear
//           identically in every one of those arms and be invisible. A real
//           decode token touches 411 separate tensor buffers, so the only way to
//           see channel-level parallelism is to read S *separately allocated*
//           buffers at once.
//
// Buffers are bound to fixed slots 0..7 and selected by a **uniform** switch on
// `b` (every thread in a threadgroup agrees, so this costs a couple of cycles
// and cannot masquerade as a bandwidth effect). metal-rs will not let a host
// build a `device uint4* const*` array, so the slot form is how this is
// expressed.
//
// Coverage is exact, not approximate. The grid is linearised: with S buffers and
// T threads, buffer b owns threads [b*P, (b+1)*P) where P = T/S, and each of
// those threads walks `i, i+P, i+2P, ...` over the whole of its buffer. The host
// guarantees `count % P == 0`, so every element is visited exactly once and the
// per-S checksum is identical to the 1-buffer case.
//
//   (The first version of this kernel gave each thread its own `chunk = count/S`
//    sub-slice and stepped by `chunks*threadgroupsize`. That over-reads the tail
//    — `count/S` is rarely a multiple of the step — so different thread counts
//    visited different byte sets. The checksum gate caught it, which is the only
//    reason it was found.)
// ---------------------------------------------------------------------------

#define MB_SLOTS 8

kernel void stream_mb(const device uint4 *d0 [[buffer(0)]],
                      const device uint4 *d1 [[buffer(1)]],
                      const device uint4 *d2 [[buffer(2)]],
                      const device uint4 *d3 [[buffer(3)]],
                      const device uint4 *d4 [[buffer(4)]],
                      const device uint4 *d5 [[buffer(5)]],
                      const device uint4 *d6 [[buffer(6)]],
                      const device uint4 *d7 [[buffer(7)]],
                      device uint *out [[buffer(8)]],
                      constant uint &count [[buffer(9)]],   // uint4 elements PER BUFFER
                      constant uint &sbufs [[buffer(10)]],  // live slots
                      constant uint &per [[buffer(11)]],   // threads per buffer
                      uint gid [[thread_position_in_grid]]) {
    // `b` MUST come from `gid`, not `tgid`. Threadgroups alternate buffers, so
    // `tgid % sbufs` puts buffer 1's threadgroups at odd gids and
    // `gid - b*per` underflows to ~4e9 -> an out-of-bounds read -> SIGSEGV.
    // Deriving it from gid makes each buffer's threads the contiguous range
    // [b*per, (b+1)*per), and `b` is still uniform within a threadgroup because
    // `per` is a multiple of the 256-thread group size.
    uint b = gid / per;

    const device uint4 *p = d0;
    switch (b) {
        case 0: p = d0; break;
        case 1: p = d1; break;
        case 2: p = d2; break;
        case 3: p = d3; break;
        case 4: p = d4; break;
        case 5: p = d5; break;
        case 6: p = d6; break;
        default: p = d7; break;
    }

    uint local = gid - b * per;
    uint4 acc = uint4(0u);
    for (uint i = local; i < count; i += per)
        acc += p[i];
    out[gid] = acc.x + acc.y + acc.z + acc.w;
}
