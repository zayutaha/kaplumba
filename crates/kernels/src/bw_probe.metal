#include <metal_stdlib>
using namespace metal;

// 1. Vectorized uint4 (16 bytes per load), 1 load per iteration
kernel void stream_uint4(const device uint4 *data [[buffer(0)]],
                         device uint4 *out [[buffer(1)]],
                         constant uint &count [[buffer(2)]],
                         uint gid [[thread_position_in_grid]],
                         uint total [[threads_per_grid]]) {
    uint4 sum = uint4(0);
    for (uint i = gid; i < count; i += total) {
        sum += data[i];
    }
    out[gid] = sum;
}

// 2. Vectorized uint4, unrolled 2x (32 bytes per thread per iteration)
kernel void stream_uint4_x2(const device uint4 *data [[buffer(0)]],
                            device uint4 *out [[buffer(1)]],
                            constant uint &count [[buffer(2)]],
                            uint gid [[thread_position_in_grid]],
                            uint total [[threads_per_grid]]) {
    uint4 s0 = uint4(0);
    uint4 s1 = uint4(0);
    uint step = total * 2;
    uint i = gid * 2;
    for (; i + total < count; i += step) {
        s0 += data[i];
        s1 += data[i + total];
    }
    if (i < count) s0 += data[i];
    out[gid] = s0 + s1;
}

// 3. Vectorized uint4, unrolled 4x (64 bytes per thread per iteration)
kernel void stream_uint4_x4(const device uint4 *data [[buffer(0)]],
                            device uint4 *out [[buffer(1)]],
                            constant uint &count [[buffer(2)]],
                            uint gid [[thread_position_in_grid]],
                            uint total [[threads_per_grid]]) {
    uint4 s0 = uint4(0);
    uint4 s1 = uint4(0);
    uint4 s2 = uint4(0);
    uint4 s3 = uint4(0);
    uint step = total * 4;
    uint i = gid * 4;
    for (; i + total * 3 < count; i += step) {
        s0 += data[i];
        s1 += data[i + total];
        s2 += data[i + total * 2];
        s3 += data[i + total * 3];
    }
    for (; i < count; i += total) {
        s0 += data[i];
    }
    out[gid] = s0 + s1 + s2 + s3;
}

// ---------------------------------------------------------------------------
// LOAD-WIDTH SENSITIVITY ARMS
//
// Arms 5, 6 and 1 read the SAME bytes in the SAME grid-stride order. The only
// difference is the width of one load instruction, and therefore how many load
// instructions issue per byte:
//
//   arm 5  stream_uint1   4 bytes/load   1 add per 4 bytes
//   arm 6  stream_uint2   8 bytes/load   2 adds per 8 bytes
//   arm 1  stream_uint4  16 bytes/load   4 adds per 16 bytes
//
// Arithmetic intensity per byte is identical in all three (one u32 add per four
// bytes), so the only variable is load-instruction count. That is what makes
// this a discriminator for the "wide lane window" question rather than just
// another bandwidth number: the wide window does not move DRAM bytes at all, it
// only removes load instructions, so if GB/s is flat across these three arms the
// window cannot pay.
//
// The three must return the SAME checksum, because adding u32s in groups of 1,
// 2 or 4 is the same multiset of additions. That equality is the fail-first
// gate: an arm that read a different number of bytes, or started at a different
// offset, cannot produce it.
// ---------------------------------------------------------------------------

// 5. Scalar uint, 4 bytes per load. Same grid-stride loop as `stream_uint4`.
kernel void stream_uint1(const device uint *data [[buffer(0)]],
                         device uint *out [[buffer(1)]],
                         constant uint &count [[buffer(2)]],
                         uint gid [[thread_position_in_grid]],
                         uint total [[threads_per_grid]]) {
    uint sum = 0;
    for (uint i = gid; i < count; i += total) {
        sum += data[i];
    }
    out[gid] = sum;
}

// 6. uint2, 8 bytes per load. Half the load instructions of arm 5 at the same
// bytes, with no change to the order in which bytes are visited.
kernel void stream_uint2(const device uint2 *data [[buffer(0)]],
                         device uint *out [[buffer(1)]],
                         constant uint &count [[buffer(2)]],
                         uint gid [[thread_position_in_grid]],
                         uint total [[threads_per_grid]]) {
    uint2 sum = uint2(0);
    for (uint i = gid; i < count; i += total) {
        sum += data[i];
    }
    out[gid] = sum.x + sum.y;
}

// 4. Block-contiguous streaming: each SIMD group reads contiguous 4KB/16KB chunk
kernel void stream_block_uint4(const device uint4 *data [[buffer(0)]],
                               device uint4 *out [[buffer(1)]],
                               constant uint &count [[buffer(2)]],
                               uint gid [[thread_position_in_grid]],
                               uint total [[threads_per_grid]]) {
    // Each thread processes contiguous slices of elements
    uint elems_per_thread = (count + total - 1) / total;
    uint start = gid * elems_per_thread;
    uint end = min(start + elems_per_thread, count);
    
    uint4 s0 = uint4(0);
    uint4 s1 = uint4(0);
    uint4 s2 = uint4(0);
    uint4 s3 = uint4(0);
    
    uint i = start;
    for (; i + 3 < end; i += 4) {
        s0 += data[i];
        s1 += data[i + 1];
        s2 += data[i + 2];
        s3 += data[i + 3];
    }
    for (; i < end; ++i) {
        s0 += data[i];
    }
    out[gid] = s0 + s1 + s2 + s3;
}
