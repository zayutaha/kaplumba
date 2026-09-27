#include <metal_stdlib>
using namespace metal;

// Plain streaming read. One thread sums a strided slice; no atomics, no shared
// memory, no barriers -- nothing that could serialise. Partial sums go to
// out[gid], so there is zero contention.
kernel void stream_sum(const device uint *data [[buffer(0)]],
                       device uint *out [[buffer(1)]],
                       constant uint &count [[buffer(2)]],
                       uint gid [[thread_position_in_grid]],
                       uint total [[threads_per_grid]]) {
    uint sum = 0u;
    for (uint i = gid; i < count; i += total) sum += data[i];
    out[gid] = sum;
}
