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
