// Elementwise glue for the single-command-buffer decode path (T4.2).
//
// Everything here exists to remove host round trips from a decode step: with the
// whole token encoded into one command buffer, each of these is a pass inside
// that buffer rather than a loop on the host. They mirror the CPU reference in
// `crates/engine/src/gemma4.rs` (`rms_norm_row`, `rope_apply_row`, `gelu_tanh`,
// `add_f16`) — where the CPU uses f64-exact f16 rounding (DE28) the kernels use
// f32 and the difference is documented as at most 1 f16 ulp.
//
// Layout note: all tensors are f16 and addressed as flat slices; the caller
// passes a device span that already points at the right row (`DeviceSpan.byte_offset`),
// so no row arithmetic leaks into the kernels.

#include <metal_stdlib>
using namespace metal;

struct RmsParams {
    uint row_len; // elements per normalized row (hidden, or head_dim)
    float eps;
};

struct RopeParams {
    uint head_dim;
    uint pos; // absolute position of row 0 (rows use pos + row)
    uint rows;
    uint row_stride; // elements per row (n_heads * head_dim)
};

// ---------------------------------------------------------------------------
// kr_rms: RMSNorm over rows of `row_len`, optional f32 weight (length row_len).
//
// grid = rows (one threadgroup per row), threadgroup = 128 threads.
// * hidden norms:  rows = 1 (decode) or M (prefill), row_len = hidden
// * head norms:    rows = n_heads, row_len = head_dim (each head is contiguous)
// * weightless:    `w` is not set (the `k_eq_v` value path)
//
// f32 sum of squares with a warp shuffle + 4-slot threadgroup reduction, so the
// whole row's mean is available in one pass.
// ---------------------------------------------------------------------------
kernel void kr_rms(device const half *x [[buffer(0)]],
                   device const float *w [[buffer(1)]],
                   device half *o [[buffer(2)]],
                   constant RmsParams &p [[buffer(3)]],
                   uint row [[threadgroup_position_in_grid]],
                   uint tid [[thread_position_in_threadgroup]]) {
    const uint n = p.row_len;
    device const half *xr = x + (ulong)row * n;
    device half *orow = o + (ulong)row * n;

    float s = 0.0f;
    for (uint i = tid; i < n; i += 128u) {
        const float v = float(xr[i]);
        s += v * v;
    }
    for (uint m = 16u; m > 0u; m >>= 1u) {
        s += simd_shuffle_xor(s, (ushort)m);
    }
    threadgroup float shw[4];
    if ((tid & 31u) == 0u) {
        shw[tid >> 5] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = shw[0] + shw[1] + shw[2] + shw[3];
    const float ms = total / float(n) + p.eps;
    const float inv = 1.0f / sqrt(ms);

    for (uint i = tid; i < n; i += 128u) {
        float v = float(xr[i]) * inv;
        if (w != nullptr) {
            v *= w[i];
        }
        orow[i] = (half)v;
    }
}

// ---------------------------------------------------------------------------
// kr_rope: neox rotate for `n_heads` heads of `head_dim`, writing to `o`.
//
// The cos/sin tables are `[max_pos, head_dim/2]` f32, built by the host with the
// same `rope_cos_sin` the CPU reference uses (so proportional RoPE's identity
// lanes come along for free: cos=1, sin=0).
//
// `o` may be a *KV cache row* (the caller passes a span already offset to the
// target row), which is how a cached k is roped straight into the cache.
// `rows` > 1 covers prefill: row r is roped at absolute position `pos + r`.
// grid = rows * n_heads * head_dim threads.
// ---------------------------------------------------------------------------
kernel void kr_rope(device const half *x [[buffer(0)]],
                    device const float *ctab [[buffer(1)]],
                    device const float *stab [[buffer(2)]],
                    device half *o [[buffer(3)]],
                    constant RopeParams &p [[buffer(4)]],
                    uint i [[thread_position_in_grid]]) {
    const uint hd = p.head_dim;
    const uint hh = hd >> 1u;
    const uint within = i % p.row_stride;
    const uint row = i / p.row_stride;
    const uint j = within % hd;
    const uint lo = (within / hd) * hd;
    const uint kk = (j < hh) ? j : (j - hh);
    const uint pos = p.pos + row;
    const float c = ctab[(ulong)pos * hh + kk];
    const float s = stab[(ulong)pos * hh + kk];
    const ulong xr = (ulong)row * p.row_stride;
    const float x1 = float(x[xr + lo + kk]);
    const float x2 = float(x[xr + lo + kk + hh]);
    o[i] = (j < hh) ? (half)(x1 * c - x2 * s) : (half)(x1 * s + x2 * c);
}

// ---------------------------------------------------------------------------
// kr_copy: `o[i] = x[i]` (f16). Used to append a value row into the KV cache.
// ---------------------------------------------------------------------------
kernel void kr_copy(device const half *x [[buffer(0)]],
                    device half *o [[buffer(1)]],
                    constant uint &n [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < n) {
        o[i] = x[i];
    }
}

// ---------------------------------------------------------------------------
// kr_gelu_mul: `o = gelu_tanh(g) * u` (the gated MLP inner product).
// ---------------------------------------------------------------------------
inline float gelu_tanh_f(float x) {
    const float k = 0.7978845608f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanh(k * (x + 0.044715f * x * x * x)));
}

kernel void kr_gelu_mul(device const half *g [[buffer(0)]],
                        device const half *u [[buffer(1)]],
                        device half *o [[buffer(2)]],
                        constant uint &n [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i < n) {
        o[i] = (half)(gelu_tanh_f(float(g[i])) * float(u[i]));
    }
}

// ---------------------------------------------------------------------------
// kr_add: `o = x + y` in f16 (the residual adds around attention/MLP).
// f16 inputs summed in f32 then rounded once is exact, matching `add_f16`.
// ---------------------------------------------------------------------------
kernel void kr_add(device const half *x [[buffer(0)]],
                   device const half *y [[buffer(1)]],
                   device half *o [[buffer(2)]],
                   constant uint &n [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
    if (i < n) {
        o[i] = (half)(float(x[i]) + float(y[i]));
    }
}

struct ScaleParams {
    uint n;
    float scalar;
};

// ---------------------------------------------------------------------------
// kr_add_scale: `o = (x + y) * scalar` — the layer's closing residual +
// `layer_scalar`. f32 multiply then one rounding to f16; the CPU reference does
// the same product in f64 and rounds once, so this can differ by at most one
// f16 ulp on a double-rounding boundary (documented in the notes; the fixture
// test bounds it end to end).
// ---------------------------------------------------------------------------
kernel void kr_add_scale(device const half *x [[buffer(0)]],
                         device const half *y [[buffer(1)]],
                         device half *o [[buffer(2)]],
                         constant ScaleParams &p [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        o[i] = (half)((float(x[i]) + float(y[i])) * p.scalar);
    }
}
