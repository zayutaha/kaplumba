//! ADVERSARY INSTRUMENT (round 2) -- DELETE BEFORE MERGING.
//!
//! Isolation test for one number: `mma_peak.rs` reports the tensor-core peak as
//! **15.1 TFLOPS**, and every "% of peak" in `spec/prefill-24k-90tps` is
//! computed against it (attention "at 12%", AC2's "46% of peak", the GEMM's
//! "43%").
//!
//! In `adv_qk_ceiling.rs` the SAME kernel -- copied byte for byte from
//! `mma_peak.rs` and renamed `mma_chains16_ref` -- reads **7.6 TFLOPS** in the
//! same process, minutes apart, on an idle box. Two candidate explanations:
//!
//!   (a) THERMAL. `mma_peak` runs two dispatches from cold; `adv_qk_ceiling`
//!       runs 30 heavy dispatches of part A first, so the GPU is hot.
//!   (b) COMPILATION CONTEXT. The 16 accumulators, 2 operands and the loop are
//!       identical, but the register allocator sees a different library.
//!
//! `KAPLUMBA_ADV_MODE=cold` runs the reference FIRST, before anything else, and
//! `hot` runs it LAST, after 30 dispatches of the same work. Same binary, same
//! source, same loop. If cold and hot differ, it is (a) and 15.1 is a cold-clock
//! number. If they agree, the difference is (b).
//!
//! This file contains ONLY the reference kernel, so it has no other kernel to be
//! co-resident with: it also answers whether the number survives being the only
//! kernel in its library.

use metal::{Device, MTLResourceOptions, MTLSize};
use std::time::Instant;

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Byte-for-byte the `mma_alu` kernel of crates/engine/examples/mma_peak.rs.
kernel void mma_alu(device half *out [[buffer(0)]],
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
}

// IS THE 7.5-vs-10.1 GAP THE STORES? `mma_alu_16out` (16 MMAs, 16 plain
// `simdgroup_store`s) reads 7.5 TFLOPS. `adv_qk_ceiling.rs`'s 16-chain kernel
// (16 MMAs, 16 accumulators, but stored by packing each float into a half and
// calling `simdgroup_store` on the packed half) reads 10.1. If this kernel --
// identical to `mma_alu_16out` in the loop, differing only in the epilogue --
// also reads 10.1, then the 35% is register pressure from the epilogue, not
// anything about the MMAs, and "peak" is a property of what you do after the
// loop. If it reads 7.5, then the difference is the PROLOGUE (16
// `make_filled_simdgroup_matrix` vs 18 `simdgroup_load`s) and again not the
// MMAs. Either way the ceiling is not a single number.
kernel void mma_alu_16out_packed(device half *out [[buffer(0)]],
                                constant uint &iters [[buffer(1)]],
                                uint gid [[thread_position_in_grid]]) {
    constexpr int LD = 8;
    simdgroup_half8x8 a, b;
    simdgroup_load(a, out, LD);
    simdgroup_load(b, out, LD);
    simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 c1 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 c2 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 c3 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 c4 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 c5 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 c6 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 c7 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 d0 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 d1 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 d2 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 d3 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 d4 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 d5 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 d6 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 d7 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
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
    simdgroup_half8x8 h;
#define PACK(x)                                                              \
    h.thread_elements()[0] = half(x.thread_elements()[0]);                    \
    h.thread_elements()[1] = half(x.thread_elements()[1]);                    \
    simdgroup_store(h, out, LD);
    PACK(c0) PACK(c1) PACK(c2) PACK(c3)
    PACK(c4) PACK(c5) PACK(c6) PACK(c7)
    PACK(d0) PACK(d1) PACK(d2) PACK(d3)
    PACK(d4) PACK(d5) PACK(d6) PACK(d7)
#undef PACK
}

// EXACTLY `adv_qk_ceiling.rs`'s 16-chain kernel, copied. It reads 10.1 TFLOPS
// there against `mma_alu_16out`'s 7.5 in the same process, and both run 16 MMAs
// with 16 live accumulators. If this reproduces 10.1, the difference is the
// accumulator declaration style (`make_filled_simdgroup_matrix` in a comma list
// vs one per statement) and NOT the arithmetic -- so even the "corrected" peak
// is a range, 7.5 to 10.1, for identical MMA work.
kernel void mma_alu_16_macro(device half *out [[buffer(0)]],
                             constant uint &iters [[buffer(1)]],
                             uint gid [[thread_position_in_grid]]) {
    simdgroup_half8x8 a, b;
    simdgroup_load(a, out, 8);
    simdgroup_load(b, out, 8);
    simdgroup_half8x8 h_;
#define D_(x) simdgroup_float8x8 x = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    D_(c0) D_(c1) D_(c2) D_(c3) D_(d0) D_(d1) D_(d2) D_(d3)
    D_(e0) D_(e1) D_(e2) D_(e3) D_(f0) D_(f1) D_(f2) D_(f3)
#undef D_
#define M_(x) simdgroup_multiply_accumulate(x, a, b, x);
    for (uint i = 0; i < iters; ++i) {
        M_(c0) M_(c1) M_(c2) M_(c3) M_(d0) M_(d1) M_(d2) M_(d3)
        M_(e0) M_(e1) M_(e2) M_(e3) M_(f0) M_(f1) M_(f2) M_(f3)
    }
#undef M_
#define S_(x)                                                                 \
    h_.thread_elements()[0] = half(x.thread_elements()[0]);                   \
    h_.thread_elements()[1] = half(x.thread_elements()[1]);                   \
    simdgroup_store(h_, out, 8);
    S_(c0) S_(c1) S_(c2) S_(c3) S_(d0) S_(d1) S_(d2) S_(d3)
    S_(e0) S_(e1) S_(e2) S_(e3) S_(f0) S_(f1) S_(f2) S_(f3)
#undef S_
}

// TWENTY-FOUR chains, all stored. `mma_peak.rs`'s header table stops at 16 and
// records that an array of 32 spilled; nobody has measured a named 24. If 24
// beats 16, the ceiling was never reached at 16 and the "16-chain peak" is an
// artifact of stopping there.
kernel void mma_alu_24out(device half *out [[buffer(0)]],
                          constant uint &iters [[buffer(1)]],
                          uint gid [[thread_position_in_grid]]) {
    constexpr int LD = 8;
    simdgroup_half8x8 a, b;
    simdgroup_load(a, out, LD);
    simdgroup_load(b, out, LD);
    simdgroup_half8x8 c0, c1, c2, c3, c4, c5, c6, c7;
    simdgroup_half8x8 d0, d1, d2, d3, d4, d5, d6, d7;
    simdgroup_half8x8 e0, e1, e2, e3, e4, e5, e6, e7;
    simdgroup_half8x8 f0, f1, f2, f3;
    simdgroup_load(c0, out, LD); simdgroup_load(c1, out, LD);
    simdgroup_load(c2, out, LD); simdgroup_load(c3, out, LD);
    simdgroup_load(c4, out, LD); simdgroup_load(c5, out, LD);
    simdgroup_load(c6, out, LD); simdgroup_load(c7, out, LD);
    simdgroup_load(d0, out, LD); simdgroup_load(d1, out, LD);
    simdgroup_load(d2, out, LD); simdgroup_load(d3, out, LD);
    simdgroup_load(d4, out, LD); simdgroup_load(d5, out, LD);
    simdgroup_load(d6, out, LD); simdgroup_load(d7, out, LD);
    simdgroup_load(e0, out, LD); simdgroup_load(e1, out, LD);
    simdgroup_load(e2, out, LD); simdgroup_load(e3, out, LD);
    simdgroup_load(e4, out, LD); simdgroup_load(e5, out, LD);
    simdgroup_load(e6, out, LD); simdgroup_load(e7, out, LD);
    simdgroup_load(f0, out, LD); simdgroup_load(f1, out, LD);
    simdgroup_load(f2, out, LD); simdgroup_load(f3, out, LD);
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
        simdgroup_multiply_accumulate(e0, a, b, e0);
        simdgroup_multiply_accumulate(e1, a, b, e1);
        simdgroup_multiply_accumulate(e2, a, b, e2);
        simdgroup_multiply_accumulate(e3, a, b, e3);
        simdgroup_multiply_accumulate(e4, a, b, e4);
        simdgroup_multiply_accumulate(e5, a, b, e5);
        simdgroup_multiply_accumulate(e6, a, b, e6);
        simdgroup_multiply_accumulate(e7, a, b, e7);
        simdgroup_multiply_accumulate(f0, a, b, f0);
        simdgroup_multiply_accumulate(f1, a, b, f1);
        simdgroup_multiply_accumulate(f2, a, b, f2);
        simdgroup_multiply_accumulate(f3, a, b, f3);
    }
    simdgroup_store(c0, out, LD); simdgroup_store(c1, out, LD);
    simdgroup_store(c2, out, LD); simdgroup_store(c3, out, LD);
    simdgroup_store(c4, out, LD); simdgroup_store(c5, out, LD);
    simdgroup_store(c6, out, LD); simdgroup_store(c7, out, LD);
    simdgroup_store(d0, out, LD); simdgroup_store(d1, out, LD);
    simdgroup_store(d2, out, LD); simdgroup_store(d3, out, LD);
    simdgroup_store(d4, out, LD); simdgroup_store(d5, out, LD);
    simdgroup_store(d6, out, LD); simdgroup_store(d7, out, LD);
    simdgroup_store(e0, out, LD); simdgroup_store(e1, out, LD);
    simdgroup_store(e2, out, LD); simdgroup_store(e3, out, LD);
    simdgroup_store(e4, out, LD); simdgroup_store(e5, out, LD);
    simdgroup_store(e6, out, LD); simdgroup_store(e7, out, LD);
    simdgroup_store(f0, out, LD); simdgroup_store(f1, out, LD);
    simdgroup_store(f2, out, LD); simdgroup_store(f3, out, LD);
}

// EIGHT MMAs, eight stores: the smallest kernel whose result is fully observed.
// If `mma_alu` (16 MMAs, 8 stores) times the same as this, then eight of its MMAs
// never executed.
kernel void mma_alu_8(device half *out [[buffer(0)]],
                     constant uint &iters [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    constexpr int LD = 8;
    simdgroup_half8x8 a, b;
    simdgroup_half8x8 c0, c1, c2, c3, c4, c5, c6, c7;
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
    for (uint i = 0; i < iters; ++i) {
        simdgroup_multiply_accumulate(c0, a, b, c0);
        simdgroup_multiply_accumulate(c1, a, b, c1);
        simdgroup_multiply_accumulate(c2, a, b, c2);
        simdgroup_multiply_accumulate(c3, a, b, c3);
        simdgroup_multiply_accumulate(c4, a, b, c4);
        simdgroup_multiply_accumulate(c5, a, b, c5);
        simdgroup_multiply_accumulate(c6, a, b, c6);
        simdgroup_multiply_accumulate(c7, a, b, c7);
    }
    simdgroup_store(c0, out, LD);
    simdgroup_store(c1, out, LD);
    simdgroup_store(c2, out, LD);
    simdgroup_store(c3, out, LD);
    simdgroup_store(c4, out, LD);
    simdgroup_store(c5, out, LD);
    simdgroup_store(c6, out, LD);
    simdgroup_store(c7, out, LD);
}

// EIGHT MMAs but SIXTEEN accumulators declared, and all sixteen stored: the
// eight chains with no MMA feeding them are zeros, stored as zeros. Its time is
// the floor for "sixteen accumulators' worth of register pressure with half the
// arithmetic". Comparing it with `mma_alu_16out` separates the cost of the extra
// eight MMAs from the cost of the extra eight live registers.
kernel void mma_alu_8pad(device half *out [[buffer(0)]],
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

// THE FINDING. `mma_peak.rs`'s `mma_alu` runs SIXTEEN `simdgroup_multiply_accumulate`
// per iteration and then stores only EIGHT accumulators (`c0..c7`). The eight
// `d*` accumulators are written by the loop and never read again.
//
// That makes them dead stores, and a compiler is entitled to delete the whole
// dependency chain that produces them: `d4 = d4 + a*b` repeated N times, whose
// result is never observed, is 1/4 of the loop's MMAs. If Metal's compiler does
// that, `mma_peak.rs` divides a wall time covering ~12 MMAs by a FLOP count
// claiming 16, and the "tensor-core peak" is inflated by up to 16/12 = 1.33x --
// or by the whole 16/8 = 2x if the eight dead chains go too.
//
// `mma_alu_16out` runs the identical sixteen MMAs and stores all sixteen. The
// timing difference between the two IS the dead-store effect, and the FLOP
// count per MMA is unchanged, so the ratio is directly interpretable.
kernel void mma_alu_16out(device half *out [[buffer(0)]],
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
"#;

fn main() {
    let mode = std::env::var("KAPLUMBA_ADV_MODE").unwrap_or_else(|_| "cold".into());
    let device = Device::system_default().expect("no Metal device");

    // THE POINT OF THIS FILE. `SRC` holds TWO kernels: the twelve-store
    // `mma_alu` (mma_peak.rs's, byte for byte) and `mma_alu_16out`, which runs
    // the identical sixteen MMAs and writes all sixteen accumulators.
    //
    // `KAPLUMBA_ADV_MODE=alone` compiles a source string containing ONLY the
    // twelve-store kernel -- what `mma_peak.rs` does. `cold` compiles the same
    // kernel with a near-identical second kernel co-resident.
    //
    // If `alone` and `cold` differ, then the register allocator's output for the
    // timed kernel depends on what else is in the library, and the "tensor-core
    // peak" is a property of a compilation unit rather than of the hardware.
    // Every "% of peak" in the spec divides by this number.
    let head = SRC.find("kernel void mma_alu(").expect("mma_alu in SRC");
    let tail = &SRC[head..];
    let end = tail.find("\n}\n").expect("end of mma_alu") + 3;
    let mma_peak_src = format!("{}{}", &SRC[..head], &tail[..end]);
    assert!(
        !mma_peak_src.contains("mma_alu_16out"),
        "the reference kernel must not contain the second kernel"
    );
    let mmas = mma_peak_src.matches("simdgroup_multiply_accumulate").count();
    let stores = mma_peak_src.matches("simdgroup_store").count();
    println!(
        "mma_peak's kernel: {mmas} MMA sites, {stores} stores \
         -> {mmas} accumulators accumulated, {stores} written out"
    );
    assert_eq!(mmas, 16, "reference kernel is not mma_peak's");
    assert_eq!(
        stores, 8,
        "mma_peak's kernel should store only 8 of its 16 accumulators"
    );
    let library = device
        .new_library_with_source(SRC, &metal::CompileOptions::new())
        .expect("compile");
    let function = library.get_function("mma_alu", None).expect("fn");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .expect("pipeline");

    let groups: u64 = 2048;
    // ITERS is settable so the timing can be checked for LINEARITY. A timing
    // that does not scale with the work is not measuring the work -- which is the
    // shape of the 2x discrepancy this file was written to explain.
    let iters: u32 = std::env::var("KAPLUMBA_ADV_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20000);
    let buf = device.new_buffer(128, MTLResourceOptions::StorageModeShared);
    let iters_buf = device.new_buffer_with_data(
        &iters as *const u32 as *const std::ffi::c_void,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let queue = device.new_command_queue();
    let run = || {
        let cb = queue.new_command_buffer();
        let ce = cb.new_compute_command_encoder();
        ce.set_compute_pipeline_state(&pipeline);
        ce.set_buffer(0, Some(&buf), 0);
        ce.set_buffer(1, Some(&iters_buf), 0);
        ce.dispatch_thread_groups(
            MTLSize::new(groups, 1, 1),
            MTLSize::new(32, 1, 1),
        );
        ce.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    };
    let measure = || -> f64 {
        let t0 = Instant::now();
        run();
        t0.elapsed().as_secs_f64() * 1e3
    };

    let flops = groups as f64 * iters as f64 * 16.0 * 1024.0;
    let report = |tag: &str, ms: f64| {
        println!(
            "{tag}: {ms:8.2} ms   {:6.2} TFLOPS   (iters {iters}, {} MMA/s)",
            flops / (ms * 1e-3) / 1e12,
            groups as f64 * iters as f64 * 16.0 / (ms * 1e-3)
        );
    };

    // Time one kernel, chosen by name.
    let time_kernel = |name: &str| -> f64 {
        let f = library.get_function(name, None).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let p = device
            .new_compute_pipeline_state_with_function(&f)
            .expect("pipeline");
        let go = || {
            let cb = queue.new_command_buffer();
            let ce = cb.new_compute_command_encoder();
            ce.set_compute_pipeline_state(&p);
            ce.set_buffer(0, Some(&buf), 0);
            ce.set_buffer(1, Some(&iters_buf), 0);
            ce.dispatch_thread_groups(
                MTLSize::new(groups, 1, 1),
                MTLSize::new(32, 1, 1),
            );
            ce.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        };
        go();
        let t0 = Instant::now();
        go();
        t0.elapsed().as_secs_f64() * 1e3
    };

    // The FLOPS for a kernel are `chains * 1024` per iteration -- counted from
    // the source, not assumed.
    let flops_of = |name: &str, chains: usize| groups as f64 * iters as f64 * chains as f64 * 1024.0;

    if mode == "matrix" {
        // THE TABLE. Four kernels:
        //   16/8   mma_peak's kernel, verbatim: 16 MMAs, 8 stored
        //   16/16  16 MMAs, all 16 stored
        //   8/8    8 MMAs, all 8 stored
        //   8/16   8 MMAs, but 16 accumulators declared and 16 stored, so the
        //          eight dead chains are present in the source and dead in fact
        //
        // Reported two ways: the FLOPs each kernel's SOURCE claims, and the
        // FLOPs it can possibly be executing. If 16/8 runs in the same time as
        // 8/8, the eight `d*` chains were deleted and mma_peak.rs divides a
        // half-length time by a full-length FLOP count.
        println!("\n{:>7} {:>7} {:>10} {:>11} {:>11} {:>11}", "MMAs", "stores", "ms", "TFLOPS/src", "TFLOPS/live", "vs 8/8");
        let mut base8: Option<f64> = None;
        for (name, chains, stores) in [
            ("mma_alu", 16usize, 8usize),
            ("mma_alu_16out", 16, 16),
            ("mma_alu_16out_packed", 16, 16),
            ("mma_alu_16_macro", 16, 16),
            ("mma_alu_8", 8, 8),
            ("mma_alu_8pad", 8, 16),
            ("mma_alu_24out", 24, 24),
        ] {
            let ms = time_kernel(name);
            // `mma_alu` stores only 8 of its 16 accumulators, so at most 8 of its
            // MMAs can be contributing to anything observable. Every other kernel
            // stores all of them.
            let live = if name == "mma_alu" { 8 } else { chains };
            let by_src = flops_of(name, chains) / (ms * 1e-3) / 1e12;
            let by_live = flops_of(name, live) / (ms * 1e-3) / 1e12;
            if name == "mma_alu_8" {
                base8 = Some(ms);
            }
            println!(
                "{chains:>7} {stores:>7} {ms:>10.2} {by_src:>11.2} {by_live:>11.2} {:>10.3}",
                base8.unwrap_or(ms) / ms
            );
        }
    } else if mode == "cold" {
        // Exactly `mma_peak.rs`'s protocol: one warm, then one timed.
        run();
        let ms = measure();
        report("mma_peak's kernel (16 MMAs, 8 stored)", ms);
        println!(
            "  -> if the compiler deleted the 8 unstored chains, the LIVE rate is {:.2} TFLOPS",
            flops_of("mma_alu", 8) / (ms * 1e-3) / 1e12
        );
    } else if mode == "linear" {
        // Linearity: same kernel, 1/4x to 4x the work. If time does not scale
        // with `iters`, the measurement is not of this kernel.
        run();
        for scale in [1u32, 2, 4, 8] {
            let i = iters * scale;
            let ib = device.new_buffer_with_data(
                &i as *const u32 as *const std::ffi::c_void,
                4,
                MTLResourceOptions::StorageModeShared,
            );
            let run_scaled = || {
                let cb = queue.new_command_buffer();
                let ce = cb.new_compute_command_encoder();
                ce.set_compute_pipeline_state(&pipeline);
                ce.set_buffer(0, Some(&buf), 0);
                ce.set_buffer(1, Some(&ib), 0);
                ce.dispatch_thread_groups(
                    MTLSize::new(groups, 1, 1),
                    MTLSize::new(32, 1, 1),
                );
                ce.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            };
            run_scaled();
            let t0 = Instant::now();
            run_scaled();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            println!(
                "  scale {scale}x: {ms:9.2} ms   {:6.2} TFLOPS   {:.2} ms per unit",
                flops * scale as f64 / (ms * 1e-3) / 1e12,
                ms / scale as f64
            );
        }
    } else {
        // Same protocol, but after 30 timed dispatches of the SAME kernel, which
        // is what `adv_qk_ceiling` does before it reaches its reference copy.
        run();
        let mut samples = Vec::new();
        for _ in 0..30 {
            samples.push(measure());
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        report("hot  median of 30", samples[samples.len() / 2]);
        report("hot  first of 30", samples[0]);
        report("hot  last of 30", samples[samples.len() - 1]);
    }
}
