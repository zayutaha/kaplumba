#include <metal_stdlib>
using namespace metal;

// Separates the op classes so a multiply-limited loop cannot be mistaken for
// general ALU throughput.

// (a) No multiplies: shift / and / xor only. This is the EXL3 *extraction* mix.
kernel void alu_logic(const device uint *in [[buffer(0)]], device uint *out [[buffer(1)]],
                      constant uint &iters [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    uint a0 = in[gid & 1023u] | 1u, a1 = a0 + 7u, a2 = a0 + 13u, a3 = a0 + 29u;
    for (uint i = 0; i < iters; ++i) {
        a0 = (a0 << 3) ^ (a0 >> 5);
        a1 = (a1 << 3) ^ (a1 >> 5);
        a2 = (a2 << 3) ^ (a2 >> 5);
        a3 = (a3 << 3) ^ (a3 >> 5);
    }
    out[gid] = a0 ^ a1 ^ a2 ^ a3;
}

// (b) One multiply per iteration, as the MCG codebook does.
kernel void alu_mul(const device uint *in [[buffer(0)]], device uint *out [[buffer(1)]],
                    constant uint &iters [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    uint a0 = in[gid & 1023u] | 1u, a1 = a0 + 7u, a2 = a0 + 13u, a3 = a0 + 29u;
    for (uint i = 0; i < iters; ++i) {
        a0 = a0 * 0xCBAC1FEDu + 1013904223u;
        a1 = a1 * 0xCBAC1FEDu + 1013904223u;
        a2 = a2 * 0xCBAC1FEDu + 1013904223u;
        a3 = a3 * 0xCBAC1FEDu + 1013904223u;
    }
    out[gid] = a0 ^ a1 ^ a2 ^ a3;
}

// (c) The codebook's fp16 half, packed as half2: two codes per lane.
kernel void alu_half2(const device half *in [[buffer(0)]], device half *out [[buffer(1)]],
                      constant uint &iters [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    half2 a0 = half2(in[gid & 1023u]), a1 = a0, a2 = a0, a3 = a0;
    for (uint i = 0; i < iters; ++i) {
        uint2 v = uint2(i, i + 1u) * 0xCBAC1FEDu;
        v = (v & uint2(0x8fff8fffu)) ^ uint2(0x3b603b60u);
        half2 lo = half2(as_type<half>(ushort(v.x)), as_type<half>(ushort(v.y)));
        half2 hi = half2(as_type<half>(ushort(v.x >> 16)), as_type<half>(ushort(v.y >> 16)));
        a0 += lo + hi; a1 += lo - hi; a2 += hi - lo; a3 += lo + hi;
    }
    out[gid] = half(a0.x + a1.x + a2.x + a3.x);
}

// (d) Plain fp32 FMA, to calibrate against Apple's published TFLOPS.
kernel void alu_fma(const device float *in [[buffer(0)]], device float *out [[buffer(1)]],
                    constant uint &iters [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    float a0 = in[gid & 1023u] + 1.0f, a1 = a0, a2 = a0, a3 = a0;
    for (uint i = 0; i < iters; ++i) {
        a0 = fma(a0, 1.0000001f, 0.5f);
        a1 = fma(a1, 1.0000001f, 0.5f);
        a2 = fma(a2, 1.0000001f, 0.5f);
        a3 = fma(a3, 1.0000001f, 0.5f);
    }
    out[gid] = a0 + a1 + a2 + a3;
}
