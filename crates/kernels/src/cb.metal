#include <metal_stdlib>
using namespace metal;

static inline float codebook_hash(uint s) {
    uint x = s * 0xCBAC1FEDu;
    x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
    return float(as_type<half>(ushort(x)) + as_type<half>(ushort(x >> 16)));
}

// Same function, but from a table built once per threadgroup. At 3 bpw only the
// top `ka` bits of a code carry information, so the table has 2^ka entries.
kernel void cb_hash(const device uint *codes [[buffer(0)]], device float *out [[buffer(1)]],
                    constant uint &n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    float a = 0.0f;
    for (uint i = gid; i < n; ++i) a += codebook_hash(codes[i]);
    out[gid] = a;
}

kernel void cb_lut(const device uint *codes [[buffer(0)]], device float *out [[buffer(1)]],
                   constant uint &n [[buffer(2)]], constant uint &shift16 [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
    // 256-entry table covers every integer rate 1..8.
    float table[256];
    for (uint i = 0; i < 256u; ++i) table[i] = codebook_hash(i << shift16);
    float a = 0.0f;
    for (uint i = gid; i < n; ++i) a += table[codes[i] >> shift16];
    out[gid] = a;
}
