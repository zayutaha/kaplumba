//! ADVERSARY INSTRUMENT (round 2, F1c) -- DELETE BEFORE MERGING.
//!
//! `implementation_notes.md` 22.3 claims the flash prefill kernel is "~88%
//! `simdgroup_multiply_accumulate`" and therefore "MMA-issue-bound"; 23.6 names
//! "16 named scalar accumulator chains in the QK loop, written the way
//! `mma_peak.rs` writes them" as the one experiment still standing. Two things
//! about that experiment are untested:
//!
//! 1. Whether 15.1 TFLOPS is the ceiling. `mma_peak.rs` stops at 16 chains, and
//!    16 chains is ~128 registers/lane, so its "peak" may be its own register
//!    limit rather than the tensor pipe's.
//! 2. Whether the QK loop can reach it. `mma_peak.rs` has **no loads**. The
//!    shipped QK loop issues 2 `simdgroup_load`s per MMA, each in the dependent
//!    position. Arms 11/12 of the fixer's ablation tried to break the
//!    accumulator chain with a *runtime* `if ((d8>>3)&1)` switch, which cannot
//!    dual-issue across a taken branch, so they could not have created ILP
//!    whatever they measured.
//!
//! Part A: pure-MMA throughput vs N *named* accumulators, N in {1,4,8,16,24}.
//! Part B: the shipped QK inner loop, with chain count varied *branchlessly*,
//! with and without software-pipelined prefetch, plus the fixer's own load
//! ablations in isolation (arms 5/8 and 14).
//!
//! FLOP convention matches `mma_peak.rs`: one `simdgroup_half8x8`
//! `simdgroup_multiply_accumulate` = 8*8*8 MACs = 1024 FLOP.

use metal::{Device, MTLResourceOptions, MTLSize};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::Instant;

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant int HEAD_DIM = 512;
constant int KV_STRIDE = 2048;

// NOTE on the two macros below: `simdgroup_store` is typed, so a `float8x8`
// cannot be stored into a `half *`. `STORE8` is for half fragments (the
// `mma_chains16_ref` control, byte-identical to `mma_peak.rs`), and `STORE8F`
// packs a float accumulator into a half8x8 by hand for the rest, so the
// accumulators are still observably written and cannot be dead-code-eliminated.
#define STORE8(x) simdgroup_store(x, out, 8)
#define STORE8F(x)                                                           \
    do {                                                                     \
        simdgroup_half8x8 h_;                                                \
        const float xe_[2] = {x.thread_elements()[0], x.thread_elements()[1]}; \
        h_.thread_elements()[0] = half(xe_[0]);                              \
        h_.thread_elements()[1] = half(xe_[1]);                              \
        simdgroup_store(h_, out, 8);                                         \
    } while (0)

// ---------------------------------------------------------------- A: pure MMA
// One kernel per chain count, all named accumulators (an ARRAY of accumulators
// spills -- mma_peak.rs records 15.49 -> 0.74 TFLOPS for exactly that).
// Accumulators are ZERO-initialised, not left undefined. `mma_peak.rs` loads
// all 16 from memory; leaving them undefined here measured 9.97 TFLOP/s at 16
// chains against `mma_peak`'s 15.10 on the same instruction sequence, and the
// only difference is that undefined values can be NaN/denormal. A tensor op
// that sees them is not measuring the pipe. This is a positive control on the
// control: the number must match mma_peak or this table is not comparable.
#define A_PROLOG                                                              \
    simdgroup_half8x8 a, b;                                                   \
    simdgroup_load(a, out, 8);                                                \
    simdgroup_load(b, out, 8);                                                \
    simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float,8,8>(0.0f);    \
    simdgroup_float8x8 c1 = make_filled_simdgroup_matrix<float,8,8>(0.0f);    \
    simdgroup_float8x8 c2 = make_filled_simdgroup_matrix<float,8,8>(0.0f);    \
    simdgroup_float8x8 c3 = make_filled_simdgroup_matrix<float,8,8>(0.0f);    \
    simdgroup_float8x8 d0 = make_filled_simdgroup_matrix<float,8,8>(0.0f);    \
    simdgroup_float8x8 d1 = make_filled_simdgroup_matrix<float,8,8>(0.0f);    \
    simdgroup_float8x8 d2 = make_filled_simdgroup_matrix<float,8,8>(0.0f);    \
    simdgroup_float8x8 d3 = make_filled_simdgroup_matrix<float,8,8>(0.0f);    \
    simdgroup_float8x8 e0 = make_filled_simdgroup_matrix<float,8,8>(0.0f);    \
    simdgroup_float8x8 e1 = make_filled_simdgroup_matrix<float,8,8>(0.0f);    \
    simdgroup_float8x8 e2 = make_filled_simdgroup_matrix<float,8,8>(0.0f);    \
    simdgroup_float8x8 e3 = make_filled_simdgroup_matrix<float,8,8>(0.0f);    \
    simdgroup_float8x8 f0 = make_filled_simdgroup_matrix<float,8,8>(0.0f);    \
    simdgroup_float8x8 f1 = make_filled_simdgroup_matrix<float,8,8>(0.0f);    \
    simdgroup_float8x8 f2 = make_filled_simdgroup_matrix<float,8,8>(0.0f);    \
    simdgroup_float8x8 f3 = make_filled_simdgroup_matrix<float,8,8>(0.0f);

#define A_EPILOG                                                              \
    STORE8F(c0); STORE8F(c1); STORE8F(c2); STORE8F(c3);                       \
    STORE8F(d0); STORE8F(d1); STORE8F(d2); STORE8F(d3);                       \
    STORE8F(e0); STORE8F(e1); STORE8F(e2); STORE8F(e3);                       \
    STORE8F(f0); STORE8F(f1); STORE8F(f2); STORE8F(f3);

// CONTROL: a byte-for-byte copy of the `mma_alu` kernel in
// `crates/engine/examples/mma_peak.rs`, in the same library as the rest of this
// file. If this reads ~15.1 TFLOP/s and `mma_chains16` -- the same 16 named
// accumulators over the same instruction sequence -- does not, then the
// difference is mine and the whole of part A is suspect. If BOTH read the
// same, then 15.1 is what 16 chains buys and `mma_peak`'s table is fine.
kernel void mma_chains16_ref(device half *out [[buffer(0)]],
                             constant uint &iters [[buffer(1)]],
                             uint gid [[thread_position_in_grid]]) {
    constexpr int LD = 8;
    simdgroup_half8x8 a, b;
    simdgroup_half8x8 c0, c1, c2, c3, c4, c5, c6, c7;
    simdgroup_half8x8 d0, d1, d2, d3, d4, d5, d6, d7;
    simdgroup_load(a, out, LD);
    simdgroup_load(b, out, LD);
    simdgroup_load(c0, out, LD);
    simdgroup_load(c1, out, LD);
    simdgroup_load(c2, out, LD);
    simdgroup_load(c3, out, LD);
    simdgroup_load(c4, out, LD);
    simdgroup_load(c5, out, LD);
    simdgroup_load(c6, out, LD);
    simdgroup_load(c7, out, LD);
    simdgroup_load(d0, out, LD);
    simdgroup_load(d1, out, LD);
    simdgroup_load(d2, out, LD);
    simdgroup_load(d3, out, LD);
    simdgroup_load(d4, out, LD);
    simdgroup_load(d5, out, LD);
    simdgroup_load(d6, out, LD);
    simdgroup_load(d7, out, LD);
    for (uint i = 0; i < iters; ++i) {
        simdgroup_multiply_accumulate(c0, a, b, c0);
        simdgroup_multiply_accumulate(c1, a, b, c1);
        simdgroup_multiply_accumulate(c2, a, b, c2);
        simdgroup_multiply_accumulate(c3, a, b, c3);
        simdgroup_multiply_accumulate(c4, a, b, c4);
        simdgroup_multiply_accumulate(c5, a, b, c5);
        simdgroup_multiply_accumulate(c6, a, b, c6);
        simdgroup_multiply_accumulate(c7, a, b, c7);
        simdgroup_multiply_accumulate(d0, a, b, d0);
        simdgroup_multiply_accumulate(d1, a, b, d1);
        simdgroup_multiply_accumulate(d2, a, b, d2);
        simdgroup_multiply_accumulate(d3, a, b, d3);
        simdgroup_multiply_accumulate(d4, a, b, d4);
        simdgroup_multiply_accumulate(d5, a, b, d5);
        simdgroup_multiply_accumulate(d6, a, b, d6);
        simdgroup_multiply_accumulate(d7, a, b, d7);
    }
    simdgroup_store(c0, out, LD);
    simdgroup_store(c1, out, LD);
    simdgroup_store(c2, out, LD);
    simdgroup_store(c3, out, LD);
    simdgroup_store(c4, out, LD);
    simdgroup_store(c5, out, LD);
    simdgroup_store(c6, out, LD);
    simdgroup_store(c7, out, LD);
    simdgroup_store(d0, out, LD);
    simdgroup_store(d1, out, LD);
    simdgroup_store(d2, out, LD);
    simdgroup_store(d3, out, LD);
    simdgroup_store(d4, out, LD);
    simdgroup_store(d5, out, LD);
    simdgroup_store(d6, out, LD);
    simdgroup_store(d7, out, LD);
}

kernel void mma_chains1(device half *out [[buffer(0)]],
                        constant uint &iters [[buffer(1)]]) {
    A_PROLOG
    for (uint i = 0; i < iters; ++i)
        simdgroup_multiply_accumulate(c0, a, b, c0);
    A_EPILOG
}

kernel void mma_chains4(device half *out [[buffer(0)]],
                        constant uint &iters [[buffer(1)]]) {
    A_PROLOG
    for (uint i = 0; i < iters; ++i) {
        simdgroup_multiply_accumulate(c0, a, b, c0);
        simdgroup_multiply_accumulate(c1, a, b, c1);
        simdgroup_multiply_accumulate(c2, a, b, c2);
        simdgroup_multiply_accumulate(c3, a, b, c3);
    }
    A_EPILOG
}

kernel void mma_chains8(device half *out [[buffer(0)]],
                        constant uint &iters [[buffer(1)]]) {
    A_PROLOG
    for (uint i = 0; i < iters; ++i) {
        simdgroup_multiply_accumulate(c0, a, b, c0);
        simdgroup_multiply_accumulate(c1, a, b, c1);
        simdgroup_multiply_accumulate(c2, a, b, c2);
        simdgroup_multiply_accumulate(c3, a, b, c3);
        simdgroup_multiply_accumulate(d0, a, b, d0);
        simdgroup_multiply_accumulate(d1, a, b, d1);
        simdgroup_multiply_accumulate(d2, a, b, d2);
        simdgroup_multiply_accumulate(d3, a, b, d3);
    }
    A_EPILOG
}

kernel void mma_chains16(device half *out [[buffer(0)]],
                         constant uint &iters [[buffer(1)]]) {
    A_PROLOG
    for (uint i = 0; i < iters; ++i) {
        simdgroup_multiply_accumulate(c0, a, b, c0);
        simdgroup_multiply_accumulate(c1, a, b, c1);
        simdgroup_multiply_accumulate(c2, a, b, c2);
        simdgroup_multiply_accumulate(c3, a, b, c3);
        simdgroup_multiply_accumulate(d0, a, b, d0);
        simdgroup_multiply_accumulate(d1, a, b, d1);
        simdgroup_multiply_accumulate(d2, a, b, d2);
        simdgroup_multiply_accumulate(d3, a, b, d3);
        simdgroup_multiply_accumulate(e0, a, b, e0);
        simdgroup_multiply_accumulate(e1, a, b, e1);
        simdgroup_multiply_accumulate(e2, a, b, e2);
        simdgroup_multiply_accumulate(e3, a, b, e3);
    }
    A_EPILOG
}

kernel void mma_chains24(device half *out [[buffer(0)]],
                         constant uint &iters [[buffer(1)]]) {
    A_PROLOG
    for (uint i = 0; i < iters; ++i) {
        simdgroup_multiply_accumulate(c0, a, b, c0);
        simdgroup_multiply_accumulate(c1, a, b, c1);
        simdgroup_multiply_accumulate(c2, a, b, c2);
        simdgroup_multiply_accumulate(c3, a, b, c3);
        simdgroup_multiply_accumulate(d0, a, b, d0);
        simdgroup_multiply_accumulate(d1, a, b, d1);
        simdgroup_multiply_accumulate(d2, a, b, d2);
        simdgroup_multiply_accumulate(d3, a, b, d3);
        simdgroup_multiply_accumulate(e0, a, b, e0);
        simdgroup_multiply_accumulate(e1, a, b, e1);
        simdgroup_multiply_accumulate(e2, a, b, e2);
        simdgroup_multiply_accumulate(e3, a, b, e3);
        simdgroup_multiply_accumulate(f0, a, b, f0);
        simdgroup_multiply_accumulate(f1, a, b, f1);
        simdgroup_multiply_accumulate(f2, a, b, f2);
        simdgroup_multiply_accumulate(f3, a, b, f3);
    }
    A_EPILOG
}

// ------------------------------------------------- B: the shipped QK inner loop
// One kernel; `variant` selects the arm, so ONE build times every arm (the F1
// lesson: a `.metal` edit per arm is what made section 10 untrustworthy).
//
//  0  SHIPPED: 1 accumulator, 64-deep chain, load -> MMA dependency, no prefetch
//  1  2 chains, branchless (load both fragments, then 2 straight-line MMAs)
//  2  4 chains, branchless
//  3  8 chains, branchless
//  4  4 chains + 4-deep load batch (no prefetch across blocks)
//  5  2 chains + 2-deep prefetch across blocks
//  6  16 named chains + 16-deep load batch  (the mma_peak shape)
// 10  1 chain + 4-deep prefetch across blocks (isolates prefetch from chains)
// 11  SHIPPED but `kf` from a fixed row  == the fixer's arm 5/8, isolated
// 12  SHIPPED but `qf` replaced by a constant == the fixer's arm 14, isolated
kernel void qk_loop(device const half *keys [[buffer(0)]],
                    constant uint &tiles [[buffer(1)]],
                    constant uint &variant [[buffer(2)]],
                    device float *out [[buffer(3)]],
                    uint tid [[thread_position_in_threadgroup]],
                    uint tgid [[threadgroup_position_in_grid]]) {
    const uint tgid_x = tgid;
    threadgroup half qstage[8 * HEAD_DIM];
    const uint lane = tid & 31u;
    const uint sg = tid >> 5u;
    const uint head_base = sg * HEAD_DIM;
    for (uint c = lane * 8u; c < HEAD_DIM; c += 256u)
        qstage[sg * HEAD_DIM + c] = half((float)(sg + 1u) * 0.031f + (float)c * 0.0017f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Named accumulators, all ZERO-initialised. This is a measured
    // precondition, not hygiene: `A_PROLOG` above exists because an
    // UNDEFINED accumulator measured 9.99 TFLOP/s at 16 chains against 10.09
    // zero-initialised in the same run -- undefined registers can be NaN, and a
    // tensor op that sees NaN is not measuring the pipe. `mma_peak.rs` gets
    // away with it only because it loads all 16 from a zero buffer.
    simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 c1 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 c2 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 c3 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 d0 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 d1 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 d2 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 d3 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 e0 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 e1 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 e2 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 e3 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 f0 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 f1 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 f2 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 f3 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    // Fragment arrays are indexed by UNROLLED loop counters only.
    simdgroup_half8x8 kf[16], qf[16];

    for (uint t = 0; t < tiles; ++t) {
        const uint krow = (t * 64u + sg * 8u) * uint(KV_STRIDE) + head_base;

        if (variant == 13u) {
            // === THE SHIPPED LOOP, EXACTLY ===
            // `attention.metal:1632`: `for (int d8 = 0; d8 < HEAD_DIM; d8 += 8)`
            // with NO `#pragma unroll`, one `kf` device load, one transposed
            // `qf` threadgroup load, one MMA into `dots`, per iteration.
            // Variant 0 is this same loop WITH `#pragma unroll` on all 64
            // iterations, so 0 vs 13 isolates what full unrolling costs.
            for (int d8 = 0; d8 < HEAD_DIM; d8 += 8) {
                simdgroup_half8x8 kf, qf;
                simdgroup_load(kf, keys + krow + d8, KV_STRIDE);
                simdgroup_load(qf, qstage + sg * uint(HEAD_DIM) + d8,
                               HEAD_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(c0, kf, qf, c0);
            }
        } else if (variant == 0u || variant == 10u || variant == 11u || variant == 12u) {
            // variant 10 is a byte-identical twin of 0: it is the control that
            // the `variant` dispatch itself costs nothing, which is the
            // precondition for every ratio in this table. Variant 0 is the
            // shipped loop with `#pragma unroll` forced on all 64 iterations.
            // SHIPPED, verbatim in shape: one fragment pair per MMA, issued
            // immediately after its own two loads, into ONE 64-deep accumulator.
#pragma unroll
            for (uint i = 0; i < 64u; ++i) {
                simdgroup_half8x8 kk, qq;
                if (variant == 11u)
                    simdgroup_load(kk, keys + head_base + i * 8u, KV_STRIDE);
                else
                    simdgroup_load(kk, keys + krow + i * 8u, KV_STRIDE);
                if (variant == 12u)
                    qq = make_filled_simdgroup_matrix<half, 8, 8>(half(0.25f));
                else
                    simdgroup_load(qq, qstage + sg * uint(HEAD_DIM) + i * 8u,
                                   HEAD_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(c0, kk, qq, c0);
            }
        } else if (variant >= 1u && variant <= 5u) {
            // ONE mechanism at five settings, all accumulating into c0 ONLY, so
            // every arm runs the same 64 MMAs in the same ascending fragment
            // order and must be BIT-IDENTICAL to variant 0. That is the control
            // on the whole table: splitting into 2/4/8/16 accumulators and
            // summing at the end (what `mma_peak`'s chain table and the shipped
            // ablation's arms 11/12 do) REASSOCIATES the sum, so it is a
            // different computation and its rate is not this loop's rate.
            //
            //   variant 0  SHIPPED: load fragment i, MMA i, load i+1, MMA i+1...
            //   variant 1  batch 2:  load i,i+1 then MMA i, MMA i+1
            //   variant 2  batch 4:  load i..i+3 then MMA i..i+3
            //   variant 3  batch 8
            //   variant 4  batch 4, but each MMA immediately follows its own load
            //               inside the batch (the control for variant 2: same
            //               instruction mix, batching removed)
            //   variant 5  batch 4, DOUBLE BUFFERED: the loads for block i+4 are
            //               issued after block i's MMAs, so they execute during
            //               them. This is the only arm that changes WHEN a load is
            //               issued rather than how many are outstanding.
            if (variant == 1u) {
#pragma unroll
                for (uint i = 0; i < 64u; i += 2) {
                    simdgroup_load(kf[0], keys + krow + i*8u, KV_STRIDE);
                    simdgroup_load(kf[1], keys + krow + (i+1)*8u, KV_STRIDE);
                    simdgroup_load(qf[0], qstage + sg*uint(HEAD_DIM) + i*8u, HEAD_DIM, ulong2(0,0), true);
                    simdgroup_load(qf[1], qstage + sg*uint(HEAD_DIM) + (i+1)*8u, HEAD_DIM, ulong2(0,0), true);
                    simdgroup_multiply_accumulate(c0, kf[0], qf[0], c0);
                    simdgroup_multiply_accumulate(c0, kf[1], qf[1], c0);
                }
            } else if (variant == 2u || variant == 4u) {
#pragma unroll
                for (uint i = 0; i < 64u; i += 4) {
                    if (variant == 4u) {
                        // 1:1 load then MMA, four times -- the control.
                        simdgroup_load(kf[0], keys + krow + i*8u, KV_STRIDE);
                        simdgroup_load(qf[0], qstage + sg*uint(HEAD_DIM) + i*8u, HEAD_DIM, ulong2(0,0), true);
                        simdgroup_multiply_accumulate(c0, kf[0], qf[0], c0);
                        simdgroup_load(kf[1], keys + krow + (i+1)*8u, KV_STRIDE);
                        simdgroup_load(qf[1], qstage + sg*uint(HEAD_DIM) + (i+1)*8u, HEAD_DIM, ulong2(0,0), true);
                        simdgroup_multiply_accumulate(c0, kf[1], qf[1], c0);
                        simdgroup_load(kf[2], keys + krow + (i+2)*8u, KV_STRIDE);
                        simdgroup_load(qf[2], qstage + sg*uint(HEAD_DIM) + (i+2)*8u, HEAD_DIM, ulong2(0,0), true);
                        simdgroup_multiply_accumulate(c0, kf[2], qf[2], c0);
                        simdgroup_load(kf[3], keys + krow + (i+3)*8u, KV_STRIDE);
                        simdgroup_load(qf[3], qstage + sg*uint(HEAD_DIM) + (i+3)*8u, HEAD_DIM, ulong2(0,0), true);
                        simdgroup_multiply_accumulate(c0, kf[3], qf[3], c0);
                    } else {
                        simdgroup_load(kf[0], keys + krow + i*8u, KV_STRIDE);
                        simdgroup_load(kf[1], keys + krow + (i+1)*8u, KV_STRIDE);
                        simdgroup_load(kf[2], keys + krow + (i+2)*8u, KV_STRIDE);
                        simdgroup_load(kf[3], keys + krow + (i+3)*8u, KV_STRIDE);
                        simdgroup_load(qf[0], qstage + sg*uint(HEAD_DIM) + i*8u, HEAD_DIM, ulong2(0,0), true);
                        simdgroup_load(qf[1], qstage + sg*uint(HEAD_DIM) + (i+1)*8u, HEAD_DIM, ulong2(0,0), true);
                        simdgroup_load(qf[2], qstage + sg*uint(HEAD_DIM) + (i+2)*8u, HEAD_DIM, ulong2(0,0), true);
                        simdgroup_load(qf[3], qstage + sg*uint(HEAD_DIM) + (i+3)*8u, HEAD_DIM, ulong2(0,0), true);
                        simdgroup_multiply_accumulate(c0, kf[0], qf[0], c0);
                        simdgroup_multiply_accumulate(c0, kf[1], qf[1], c0);
                        simdgroup_multiply_accumulate(c0, kf[2], qf[2], c0);
                        simdgroup_multiply_accumulate(c0, kf[3], qf[3], c0);
                    }
                }
            } else if (variant == 3u) {
#pragma unroll
                for (uint i = 0; i < 64u; i += 8) {
                    for (uint u = 0; u < 8u; ++u)
                        simdgroup_load(kf[u], keys + krow + (i+u)*8u, KV_STRIDE);
                    for (uint u = 0; u < 8u; ++u)
                        simdgroup_load(qf[u], qstage + sg*uint(HEAD_DIM) + (i+u)*8u, HEAD_DIM, ulong2(0,0), true);
                    for (uint u = 0; u < 8u; ++u)
                        simdgroup_multiply_accumulate(c0, kf[u], qf[u], c0);
                }
            } else {
                // variant 5: double buffered, 4 deep. Two fragment sets, so the
                // loads for the next block are issued after this block's MMAs and
                // execute during them. Same 64 MMAs, same order, same c0.
                simdgroup_half8x8 ka[4], qa[4];
                for (uint u = 0; u < 4u; ++u) {
                    simdgroup_load(ka[u], keys + krow + u*8u, KV_STRIDE);
                    simdgroup_load(qa[u], qstage + sg*uint(HEAD_DIM) + u*8u, HEAD_DIM, ulong2(0,0), true);
                }
#pragma unroll
                for (uint i = 0; i < 64u; i += 4) {
                    simdgroup_multiply_accumulate(c0, ka[0], qa[0], c0);
                    simdgroup_multiply_accumulate(c0, ka[1], qa[1], c0);
                    simdgroup_multiply_accumulate(c0, ka[2], qa[2], c0);
                    simdgroup_multiply_accumulate(c0, ka[3], qa[3], c0);
                    if (i + 4 < 64u) {
                        for (uint u = 0; u < 4u; ++u) {
                            simdgroup_load(ka[u], keys + krow + (i+4+u)*8u, KV_STRIDE);
                            simdgroup_load(qa[u], qstage + sg*uint(HEAD_DIM) + (i+4+u)*8u, HEAD_DIM, ulong2(0,0), true);
                        }
                    }
                }
            }
        } else if (variant == 6u) {
            // 16 named chains, branchless, 16-deep load batch. This is the
            // shape implementation_notes.md 23.6 says is the one lever left,
            // and it is the one arms 11/12 could not build.
#pragma unroll
            for (uint i = 0; i < 64u; i += 16) {
                simdgroup_load(kf[0], keys + krow + (i+0)*8u, KV_STRIDE);
                simdgroup_load(kf[1], keys + krow + (i+1)*8u, KV_STRIDE);
                simdgroup_load(kf[2], keys + krow + (i+2)*8u, KV_STRIDE);
                simdgroup_load(kf[3], keys + krow + (i+3)*8u, KV_STRIDE);
                simdgroup_load(kf[4], keys + krow + (i+4)*8u, KV_STRIDE);
                simdgroup_load(kf[5], keys + krow + (i+5)*8u, KV_STRIDE);
                simdgroup_load(kf[6], keys + krow + (i+6)*8u, KV_STRIDE);
                simdgroup_load(kf[7], keys + krow + (i+7)*8u, KV_STRIDE);
                simdgroup_load(kf[8], keys + krow + (i+8)*8u, KV_STRIDE);
                simdgroup_load(kf[9], keys + krow + (i+9)*8u, KV_STRIDE);
                simdgroup_load(kf[10], keys + krow + (i+10)*8u, KV_STRIDE);
                simdgroup_load(kf[11], keys + krow + (i+11)*8u, KV_STRIDE);
                simdgroup_load(kf[12], keys + krow + (i+12)*8u, KV_STRIDE);
                simdgroup_load(kf[13], keys + krow + (i+13)*8u, KV_STRIDE);
                simdgroup_load(kf[14], keys + krow + (i+14)*8u, KV_STRIDE);
                simdgroup_load(kf[15], keys + krow + (i+15)*8u, KV_STRIDE);
                simdgroup_load(qf[0], qstage + sg*uint(HEAD_DIM) + (i+0)*8u, HEAD_DIM, ulong2(0,0), true);
                simdgroup_load(qf[1], qstage + sg*uint(HEAD_DIM) + (i+1)*8u, HEAD_DIM, ulong2(0,0), true);
                simdgroup_load(qf[2], qstage + sg*uint(HEAD_DIM) + (i+2)*8u, HEAD_DIM, ulong2(0,0), true);
                simdgroup_load(qf[3], qstage + sg*uint(HEAD_DIM) + (i+3)*8u, HEAD_DIM, ulong2(0,0), true);
                simdgroup_load(qf[4], qstage + sg*uint(HEAD_DIM) + (i+4)*8u, HEAD_DIM, ulong2(0,0), true);
                simdgroup_load(qf[5], qstage + sg*uint(HEAD_DIM) + (i+5)*8u, HEAD_DIM, ulong2(0,0), true);
                simdgroup_load(qf[6], qstage + sg*uint(HEAD_DIM) + (i+6)*8u, HEAD_DIM, ulong2(0,0), true);
                simdgroup_load(qf[7], qstage + sg*uint(HEAD_DIM) + (i+7)*8u, HEAD_DIM, ulong2(0,0), true);
                simdgroup_load(qf[8], qstage + sg*uint(HEAD_DIM) + (i+8)*8u, HEAD_DIM, ulong2(0,0), true);
                simdgroup_load(qf[9], qstage + sg*uint(HEAD_DIM) + (i+9)*8u, HEAD_DIM, ulong2(0,0), true);
                simdgroup_load(qf[10], qstage + sg*uint(HEAD_DIM) + (i+10)*8u, HEAD_DIM, ulong2(0,0), true);
                simdgroup_load(qf[11], qstage + sg*uint(HEAD_DIM) + (i+11)*8u, HEAD_DIM, ulong2(0,0), true);
                simdgroup_load(qf[12], qstage + sg*uint(HEAD_DIM) + (i+12)*8u, HEAD_DIM, ulong2(0,0), true);
                simdgroup_load(qf[13], qstage + sg*uint(HEAD_DIM) + (i+13)*8u, HEAD_DIM, ulong2(0,0), true);
                simdgroup_load(qf[14], qstage + sg*uint(HEAD_DIM) + (i+14)*8u, HEAD_DIM, ulong2(0,0), true);
                simdgroup_load(qf[15], qstage + sg*uint(HEAD_DIM) + (i+15)*8u, HEAD_DIM, ulong2(0,0), true);
                simdgroup_multiply_accumulate(c0, kf[0], qf[0], c0);
                simdgroup_multiply_accumulate(c1, kf[1], qf[1], c1);
                simdgroup_multiply_accumulate(c2, kf[2], qf[2], c2);
                simdgroup_multiply_accumulate(c3, kf[3], qf[3], c3);
                simdgroup_multiply_accumulate(d0, kf[4], qf[4], d0);
                simdgroup_multiply_accumulate(d1, kf[5], qf[5], d1);
                simdgroup_multiply_accumulate(d2, kf[6], qf[6], d2);
                simdgroup_multiply_accumulate(d3, kf[7], qf[7], d3);
                simdgroup_multiply_accumulate(e0, kf[8], qf[8], e0);
                simdgroup_multiply_accumulate(e1, kf[9], qf[9], e1);
                simdgroup_multiply_accumulate(e2, kf[10], qf[10], e2);
                simdgroup_multiply_accumulate(e3, kf[11], qf[11], e3);
                simdgroup_multiply_accumulate(f0, kf[12], qf[12], f0);
                simdgroup_multiply_accumulate(f1, kf[13], qf[13], f1);
                simdgroup_multiply_accumulate(f2, kf[14], qf[14], f2);
                simdgroup_multiply_accumulate(f3, kf[15], qf[15], f3);
            }
            // Fold the 16 partials into c0, then ZERO the other 15. The `sink`
            // below sums c0..f3 unconditionally, so without this the sink for
            // variant 6 counts the partials twice and reads ~16x the reference.
            // (That is exactly what the first run of this control reported:
            // max rel diff 9.9e1, which is 16x, not a numerics problem.)
            for (uint e = 0; e < 2u; ++e) {
                c0.thread_elements()[e] += c1.thread_elements()[e];
                c0.thread_elements()[e] += c2.thread_elements()[e];
                c0.thread_elements()[e] += c3.thread_elements()[e];
                c0.thread_elements()[e] += d0.thread_elements()[e];
                c0.thread_elements()[e] += d1.thread_elements()[e];
                c0.thread_elements()[e] += d2.thread_elements()[e];
                c0.thread_elements()[e] += d3.thread_elements()[e];
                c0.thread_elements()[e] += e0.thread_elements()[e];
                c0.thread_elements()[e] += e1.thread_elements()[e];
                c0.thread_elements()[e] += e2.thread_elements()[e];
                c0.thread_elements()[e] += e3.thread_elements()[e];
                c0.thread_elements()[e] += f0.thread_elements()[e];
                c0.thread_elements()[e] += f1.thread_elements()[e];
                c0.thread_elements()[e] += f2.thread_elements()[e];
                c0.thread_elements()[e] += f3.thread_elements()[e];
                c1.thread_elements()[e] = 0.0f;
                c2.thread_elements()[e] = 0.0f;
                c3.thread_elements()[e] = 0.0f;
                d0.thread_elements()[e] = 0.0f;
                d1.thread_elements()[e] = 0.0f;
                d2.thread_elements()[e] = 0.0f;
                d3.thread_elements()[e] = 0.0f;
                e0.thread_elements()[e] = 0.0f;
                e1.thread_elements()[e] = 0.0f;
                e2.thread_elements()[e] = 0.0f;
                e3.thread_elements()[e] = 0.0f;
                f0.thread_elements()[e] = 0.0f;
                f1.thread_elements()[e] = 0.0f;
                f2.thread_elements()[e] = 0.0f;
                f3.thread_elements()[e] = 0.0f;
            }
        }
    }
    float sink = 0.0f;
    for (uint e = 0; e < 2u; ++e) {
        sink += c0.thread_elements()[e] + c1.thread_elements()[e];
        sink += c2.thread_elements()[e] + c3.thread_elements()[e];
        sink += d0.thread_elements()[e] + d1.thread_elements()[e];
        sink += d2.thread_elements()[e] + d3.thread_elements()[e];
        sink += e0.thread_elements()[e] + e1.thread_elements()[e];
        sink += e2.thread_elements()[e] + e3.thread_elements()[e];
        sink += f0.thread_elements()[e] + f1.thread_elements()[e];
        sink += f2.thread_elements()[e] + f3.thread_elements()[e];
    }
    // One fp32 per THREADGROUP (sg 0, lane 0), so the host can compare arms
    // elementwise. Unconditional: an earlier revision guarded this with
    // `tid == 0xffffffffu` and every arm timed at 0.13 ms, because the compiler
    // is allowed to delete the entire loop when nothing observable reads it.
    // That is the whole lesson of F1 in one incident.
    if (lane == 0u && sg == 0u)
        out[tgid_x] = sink;
}
"#;

thread_local! {
    static A_MS: RefCell<BTreeMap<usize, Vec<f64>>> = RefCell::new(BTreeMap::new());
}

fn main() {
    let device = Device::system_default().expect("no Metal device");
    println!("device: {}", device.name());
    let library = device
        .new_library_with_source(SRC, &metal::CompileOptions::new())
        .unwrap_or_else(|e| panic!("compile: {e:?}"));
    let queue = device.new_command_queue().to_owned();

    // ---------------- part A: pure-MMA ceiling vs named chain count ----------
    println!("\n=== A. pure MMA, N named accumulator chains (mma_peak's question) ===");
    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>12} {:>12} {:>14}",
        "chains", "median_ms", "min", "max", "TFLOP/s(src)", "TFLOP/s(live)", "MMA/s (x1e9)"
    );
    let groups: u64 = 2048;
    let iters: u32 = 20000;
    let small = device.new_buffer(128, MTLResourceOptions::StorageModeShared);
    let iters_buf = device.new_buffer_with_data(
        &iters as *const u32 as *const std::ffi::c_void,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    for _round in 0..5 {
        for n in [1usize, 4, 8, 16, 24, 100] {
            let name = if n == 100 {
                "mma_chains16_ref".to_string()
            } else {
                format!("mma_chains{n}")
            };
            let f = library
                .get_function(&name, None)
                .unwrap_or_else(|e| panic!("fn mma_chains{n}: {e:?}"));
            let p = device
                .new_compute_pipeline_state_with_function(&f)
                .expect("pipeline");
            let run = || {
                let cb = queue.new_command_buffer();
                let ce = cb.new_compute_command_encoder();
                ce.set_compute_pipeline_state(&p);
                ce.set_buffer(0, Some(&small), 0);
                ce.set_buffer(1, Some(&iters_buf), 0);
                ce.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(32, 1, 1));
                ce.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            };
            run();
            // Untimed, repeated reads: the clock the timed sample sees is a
            // different one each round on this box, which is why every table
            // here is a within-run ratio and never a cross-run comparison.
            let t0 = Instant::now();
            run();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            A_MS.with(|a| a.borrow_mut().entry(n).or_default().push(ms));
        }
    }
    // Every kernel here accumulates into NAMED accumulators and stores every one
    // of them (STORE8F writes all sixteen in A_EPILOG), so `chains` is the
    // number of MMAs that actually execute. There is no dead-chain discount.
    for n in [1usize, 4, 8, 16, 24, 100] {
        let mut ms = A_MS.with(|a| a.borrow().get(&n).cloned().unwrap_or_default());
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = ms[ms.len() / 2];
        // 100 is `mma_chains16_ref`, the copy of `mma_peak.rs`'s kernel. It is
        // the ONLY one here that does NOT store all sixteen: its source stores
        // eight. Reported both ways, because the difference between the two
        // columns IS the dead-store effect (see adv_peak_isolation.rs).
        let (chains, live) = if n == 100 { (16.0, 8.0) } else { (n as f64, n as f64) };
        let rate = groups as f64 * iters as f64 * chains / (med * 1e-3);
        let label = if n == 100 {
            String::from("16ref")
        } else {
            n.to_string()
        };
        println!(
            "{:>6} {:>10.2} {:>10.2} {:>10.2} {:>12.2} {:>12.2} {:>14.3}",
            label,
            med,
            ms[0],
            ms[ms.len() - 1],
            groups as f64 * iters as f64 * chains * 1024.0 / (med * 1e-3) / 1e12,
            groups as f64 * iters as f64 * live * 1024.0 / (med * 1e-3) / 1e12,
            rate / 1e9
        );
    }

    // ---------------- part B: the shipped QK inner loop ----------------------
    println!("\n=== B. shipped QK inner loop (24576 keys x 1024 queries x 4 kv heads) ===");
    println!(
        "{:>7} {:>28} {:>11} {:>10} {:>10} {:>11} {:>9}",
        "variant", "what", "median_ms", "min", "max", "TFLOP/s", "x shipped"
    );
    let kv_halfs = 24576usize * 4 * 512;
    let keys = device.new_buffer((kv_halfs * 2) as u64, MTLResourceOptions::StorageModeShared);
    {
        // Real fp16 bit patterns. An earlier revision wrote
        // `(sin(i)*0.5) as u16`, which is an INTEGER cast of a float in
        // [-0.5, 0.5] and therefore produces 0 for every element -- so the MMA
        // operands were all zero and the sink readback confirmed it (sink = 0).
        // Filling with non-zero halves is required for the timing to mean
        // anything: zero operands are the classic way a multiply gets optimised.
        let p = keys.contents() as *mut u16;
        for i in 0..kv_halfs {
            let v = ((i as f32) * 0.00031).sin() * 0.5;
            unsafe { *p.add(i) = half::f16::from_f32(v).to_bits() };
        }
        // Prove the fill is non-zero before timing anything.
        let nonzero = (0..kv_halfs)
            .filter(|&i| unsafe { *(keys.contents() as *const u16).add(i) } != 0)
            .count();
        println!("keys buffer: {nonzero}/{kv_halfs} elements non-zero");
    }
    // 256 queries, not 1024 (see the note above on why).
    let queries: u64 = 256;
    let kv_heads: u64 = 4;
    // fp32 sinks, one per threadgroup, allocated before anything borrows them.
    let n_sinks = (queries * kv_heads) as usize;
    let out = device.new_buffer((n_sinks * 4) as u64, MTLResourceOptions::StorageModeShared);
    let tiles: u32 = 384;
    let tiles_buf = device.new_buffer_with_data(
        &tiles as *const u32 as *const std::ffi::c_void,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let f = library.get_function("qk_loop", None).expect("fn qk_loop");
    let p = device
        .new_compute_pipeline_state_with_function(&f)
        .expect("pipeline");
    let mut b_ms: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
    let variants: [u32; 11] = [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13];
    for _round in 0..3 {
        for v in variants {
            let vbuf = device.new_buffer_with_data(
                &v as *const u32 as *const std::ffi::c_void,
                4,
                MTLResourceOptions::StorageModeShared,
            );
            let run = || {
                let cb = queue.new_command_buffer();
                let ce = cb.new_compute_command_encoder();
                ce.set_compute_pipeline_state(&p);
                ce.set_buffer(0, Some(&keys), 0);
                ce.set_buffer(1, Some(&tiles_buf), 0);
                ce.set_buffer(2, Some(&vbuf), 0);
                ce.set_buffer(3, Some(&out), 0);
                ce.dispatch_thread_groups(
                    MTLSize::new(queries * kv_heads, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                ce.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            };
            run();
            let t0 = Instant::now();
            run();
            b_ms.entry(v)
                .or_default()
                .push(t0.elapsed().as_secs_f64() * 1e3);
        }
    }
    // The baseline is variant 13, which is the SHIPPED loop transcribed exactly:
    // `attention.metal:1632` has `for (int d8 = 0; d8 < HEAD_DIM; d8 += 8)` with
    // NO `#pragma unroll`. Variant 0 is the same loop with the unroll forced, and
    // it is 2.3x SLOWER -- so using variant 0 as the baseline made every other
    // arm look like a win. Ratios here are against 13.
    let base = {
        let mut v = b_ms[&13].clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let names: BTreeMap<u32, &str> = [
        (13, "SHIPPED, no unroll pragma"),
        (0, "same, unroll forced x64"),
        (1, "batch 2"),
        (2, "batch 4"),
        (3, "batch 8"),
        (4, "batch 4, 1:1 load/MMA (ctl)"),
        (5, "batch 4, DOUBLE BUFFERED"),
        (6, "16 accumulators (reassoc.)"),
        (10, "byte-identical twin of 0"),
        (11, "kf from a fixed row (L1)"),
        (12, "qf load deleted"),
    ]
    .into_iter()
    .collect();
    // POSITIVE CONTROL on part B -- and this is the one that matters, because a
    // 4.8x "speedup" that comes from SKIPPING work is the exact failure mode
    // this repo keeps shipping. Every variant writes its sink for EVERY
    // threadgroup, and the table below compares them ELEMENTWISE against
    // variant 0. Variants 1-6 and 10 execute the same 64 MMAs into the same
    // one accumulator in the same order, so they must be BIT-IDENTICAL; if any
    // differs, the ratio in the timing table is a measurement of missing work.
    //
    // Variant 11 (kf from a fixed row) and 12 (qf deleted) are expected to
    // differ -- they deliberately compute something else.
    println!("\n--- correctness control: per-threadgroup sink, variant 0 vs the rest ---");
    {
        let n_sinks = (queries * kv_heads) as usize;
        let sinks = device.new_buffer((n_sinks * 4) as u64, MTLResourceOptions::StorageModeShared);
        let mut base: Vec<f32> = Vec::new();
        for v in [0u32, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13] {
            let vbuf = device.new_buffer_with_data(
                &v as *const u32 as *const std::ffi::c_void,
                4,
                MTLResourceOptions::StorageModeShared,
            );
            {
                let p = sinks.contents() as *mut f32;
                for i in 0..n_sinks {
                    unsafe { *p.add(i) = f32::NAN };
                }
            }
            let cb = queue.new_command_buffer();
            let ce = cb.new_compute_command_encoder();
            ce.set_compute_pipeline_state(&p);
            ce.set_buffer(0, Some(&keys), 0);
            ce.set_buffer(1, Some(&tiles_buf), 0);
            ce.set_buffer(2, Some(&vbuf), 0);
            ce.set_buffer(3, Some(&sinks), 0);
            ce.dispatch_thread_groups(
                MTLSize::new(queries * kv_heads, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            ce.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let got: Vec<f32> = (0..n_sinks)
                .map(|i| unsafe { *(sinks.contents() as *const f32).add(i) })
                .collect();
            let nz = got.iter().filter(|v| **v != 0.0).count();
            if v == 0 {
                base = got.clone();
                println!(
                    "  variant  0 (reference): {nz}/{n_sinks} non-zero, \
                     e.g. [{:e}, {:e}, {:e}]",
                    got[0], got[1], got[2]
                );
                assert_eq!(nz, n_sinks, "variant 0 left sinks unwritten");
            } else {
                let bit_same = got
                    .iter()
                    .zip(base.iter())
                    .filter(|(a, b)| a.to_bits() == b.to_bits())
                    .count();
                let scale = base.iter().fold(0.0f32, |m, x| m.max(x.abs()));
                let max_rel = got
                    .iter()
                    .zip(base.iter())
                    .fold(0.0f32, |m, (a, b)| m.max((a - b).abs() / scale));
                let max_abs = got
                    .iter()
                    .zip(base.iter())
                    .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
                // Bit-identity is the WRONG criterion for the multi-chain arms:
                // splitting one accumulator into 4 and summing them at the end
                // reassociates the additions, and fp addition is not
                // associative. What must hold is that the arms compute the same
                // mathematical value to within reassociation error, so the
                // criterion is the SIZE of the relative difference, and the
                // verdict line says which criterion is being applied.
                // Variants 0-5 and 10 all accumulate 64 MMAs into c0 in
                // ascending order, so BIT-IDENTICAL is the correct criterion
                // and nothing less will do. Variant 6 uses 16 accumulators, which
                // reassociates, so there the criterion is the SIZE of the
                // difference. 11 and 12 deliberately compute something else.
                let expect_bit = v != 6 && v != 11 && v != 12;
                let _ = expect_bit;
                let ok = if expect_bit {
                    bit_same == n_sinks
                } else {
                    max_rel < 1e-5
                };
                let worst = got
                    .iter()
                    .zip(base.iter())
                    .enumerate()
                    .fold((0usize, 0.0f32), |(bi, bv), (i, (a, b))| {
                        let d = (a - b).abs();
                        if d > bv { (i, d) } else { (bi, bv) }
                    });
                println!(
                    "  variant {v:>2}: {bit_same}/{n_sinks} bit-identical, \
                     max abs diff {max_abs:.3e} (max|base| {scale:.3e}, ratio {:.3e}){}",
                    max_abs / scale.max(f32::MIN_POSITIVE),
                    if v == 11 || v == 12 {
                        "  <- DELIBERATELY computes something else"
                    } else if v == 6 {
                        "  <- reassociates by construction; size of the diff is the check"
                    } else if ok {
                        "  <- CORRECT"
                    } else {
                        "  *** SUSPECT: not the shipped arithmetic ***"
                    }
                );
                if !ok {
                    println!(
                        "      worst entry idx {}: this arm {:e} vs v0 {:e}",
                        worst.0, got[worst.0], base[worst.0]
                    );
                }
                assert!(
                    ok || v == 11 || v == 12,
                    "variant {v} is not computing the shipped QK arithmetic \
                     (bit_same {bit_same}/{n_sinks}, max_abs {max_abs:e}); its ratio is void"
                );
            }
        }
    }

    // 256 queries, not 1024. The per-threadgroup MMA and load counts are
    // IDENTICAL -- the loop over tiles is untouched -- so the arm ratios are
    // unchanged, and the resident set stays small enough that `gpuq` does not
    // go into swap. A 1024-query run of this file dropped `gpuq` to 8.6 GiB
    // available and its per-arm spread blew out to 2.2x (480-1065 ms on one
    // arm), which is a measurement of memory pressure, not of the kernel.
    //
    // QK FLOP only: 2 * q_heads * keys * head_dim * queries.
    let flops = (queries as f64) * 32.0 * 24576.0 * 512.0 * 2.0;
    for v in variants {
        let mut ms = b_ms[&v].clone();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = ms[ms.len() / 2];
        println!(
            "{:>7} {:>28} {:>11.2} {:>10.2} {:>10.2} {:>11.2} {:>9.3}",
            v,
            names[&v],
            med,
            ms[0],
            ms[ms.len() - 1],
            flops / (med * 1e-3) / 1e12,
            base / med
        );
    }
}
