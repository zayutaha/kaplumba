//! P01 probe: the **instruction-level cost of decoding one EXL3 code**, and
//! whether the live 3 bpw decode GEMV is bound by it.
//!
//! # The claim under test
//!
//! From a read-only audit of `crates/kernels/src/linear_repacked.metal` -- the
//! live kernel is `linear_gemv_ishizuki_k3_colmajor_2col`
//! (`linear_repacked.metal:2072-2206`), reached from
//! `LinearKernels::encode_gemv_raw` at `crates/kernels/src/linear.rs:2133-2157`,
//! which fires *before* the `ishizuki_gemv_pipeline` colmajor lookup at
//! `linear.rs:1539-1550` -- the per-8-code instruction budget is:
//!
//! ```text
//! per lane, per 16x16 tile = 8 EXL3 codes:
//!   extract_v_mcg_32 (linear_repacked.metal:53-64)
//!     3  variable 32-bit shifts   wa<<sh0, wb>>rsh0, wb<<sh0   (:54-55)
//!     1  OR                       A_hi = (wa<<sh0)|(wb>>rsh0)  (:54)
//!    10  constant 32-bit shifts   v0..v5 (6) + v6,v7 (4)       (:56-63)
//!     5  AND  &0xffff             v1..v5 only                   (:57-61)
//!     4  OR                       v6, v7                        (:62-63)
//!   decode_state_mcg (linear.metal:121-126)  x8
//!     8  mul.lo.u32                                             (:122)
//!     8  AND  &0x8fff8fff                                        (:123)
//!     8  XOR  ^0x3b603b60                                        (:123)
//!     8  fp16 add  h.x + h.y   (fp16, pinned by exl3.rs:173-179)
//!     8  f16 -> f32 convert                                      (:125)
//!   accumulation (linear_repacked.metal:2138-2139, 2142-2143,
//!                 2150-2151, 2154-2155)
//!     8  fp32 FMA
//!     2  weight loads (t[j0], t[j1])
//!     1  x load + 1 convert  (8+8 per inner iteration, 32 codes)
//!                                ~= 75 lane-instructions / 8 codes
//!                                    of which 45 are integer-class
//!                                    (shift/and/or/xor/mul) = 5.6 per code
//!                                 ~= 9.4 lane-instructions / EXL3 code
//! ```
//!
//! 20 cores x 128 lanes at ~4.3 GHz issues ~5.5e12 lane-instructions/s, so
//! 9.4 instructions/code caps this loop at ~590 Gcodes/s = 221 GB/s at 3 bpw.
//! Measured best is 203 GB/s = 541 Gcodes/s, i.e. ~92% of that. **20 tok/s
//! needs 236.7 GB/s = 631 Gcodes/s = 108% of the machine's issue capability at
//! this instruction count**, so the code-rate ceiling is ~225 GB/s ~ 18.6
//! tok/s of GEMV time, and 20 tok/s is unreachable by making these kernels
//! faster.
//!
//! # What falsifies it
//!
//! | arm | what it is | the claim dies if |
//! |---|---|---|
//! | (A) | production kernel, **real** `Gxma-repacked` trellis bytes | -- (the reference) |
//! | (B) | literal replica of (A)'s decode+accumulate body | B != A bitwise => every other arm is void |
//! | (C) | (B) with weight loads replaced by **tk-varying** synthetic words | C <= 0.60*A => memory/latency bound, not issue bound |
//! | (D) | (B) with `sh0`/`rsh0` pinned to constants (19/13) | D/B < 1.05 => the window setup costs ~nothing |
//! | (E) | (B) with the five `& 0xffff` extraction masks removed | E/B ~= 1.00 => MSL already folds them, real count is 8.5/code |
//! | (G) | register-resident loop: the **identical** per-8-code stream, zero memory | **G < 631 Gcodes/s => 20 tok/s is impossible, unconditionally** |
//!
//! (G) is the decisive arm: it touches no memory, so it measures the machine's
//! instruction ceiling for this exact op mix directly, and needs no Metal
//! disassembler (there is none on this box:
//! `spec/decode-19tok-repacked-250gb/implementation_notes.md:18-21`).
//!
//! # Gates run before any timing is reported (AGENTS.md §3.4)
//!
//! * Every arm must be present in the compiled library. `LinearKernels::new`
//!   has failed open three times in this repo
//!   (`spec/decode-3bpw-amplification/implementation_notes.md:240-243`), so the
//!   probe asserts on `get_function` rather than `unwrap_or_default`.
//! * (B) must be **bitwise equal** to (A) on real weights, or the probe panics.
//! * (C) must change with the probe seed and differ from (A), or the decode has
//!   been optimised away and the arm measures nothing.
//! * Estimator is **min over interleaved rounds**, never the median: this harness
//!   is not covered by the engine's `flock`, so another agent's `kaplumba` steals
//!   rounds, and contention can only make a kernel slower
//!   (`crates/engine/examples/probe_contig4_load_alu.rs:541-548`).
//!
//! # Run
//!
//! ```bash
//! export KAPLUMBA_TEST_MODEL=~/.kaplumba/models/Gxma-repacked
//! scripts/gpuq run -- cargo test --release --test perf_probe_code_rate_p01 -- \
//!     --ignored --nocapture --test-threads=1
//! ```
//!
//! Reads ~90 MiB of trellis off disk and needs ~200 MiB of GPU buffers. It does
//! NOT load the model, but it still must be the only GPU process (AGENTS.md §10).

use half::f16;
use metal::{
    Buffer, CommandQueue, ComputeCommandEncoderRef, ComputePipelineState, Device, Library,
    MTLResourceOptions, MTLSize,
};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

// ===========================================================================
// Metal source
// ===========================================================================

/// The `decode_state*` family transcribed verbatim from
/// `crates/kernels/src/linear.metal:115-147`. `linear_repacked.metal` has no
/// header of its own -- `LinearKernels::new` concatenates it onto `linear.metal`,
/// which opens with `#include <metal_stdlib>` / `using namespace metal;` -- so
/// the probe library reproduces that prefix. Without it every `simd_shuffle_xor`
/// / `mem_flags` in the file is an undeclared identifier.
const PREAMBLE: &str = r#"
#include <metal_stdlib>
using namespace metal;

static float decode_state_plain(uint value) {
    value = value * 89226354u + 64248484u;
    value = 0x3b603b60u ^ (value & 0x8fff8fffu);
    return float(as_type<half>((ushort)value) + as_type<half>((ushort)(value >> 16)));
}

static inline float decode_state_mcg(uint value) {
    value = value * 0xCBAC1FEDu;
    value = 0x3b603b60u ^ (value & 0x8fff8fffu);
    half2 h = as_type<half2>(value);
    return float(h.x + h.y);
}

static float decode_state_mul1(uint value) {
    value = value * 0x83DCD12Du;
    uint sum = 0x6400u + (value & 0xffu) + ((value >> 8) & 0xffu) + ((value >> 16) & 0xffu) + ((value >> 24) & 0xffu);
    half byte_value = as_type<half>((ushort)sum);
    half inverse = as_type<half>((ushort)0x1eeeu);
    half bias = as_type<half>((ushort)0xc931u);
    return float(half(float(byte_value) * float(inverse) + float(bias)));
}

static float decode_state(uint state, uint codebook) {
    if (codebook == 0u) { return decode_state_plain(state); }
    if (codebook == 1u) { return decode_state_mcg(state); }
    return decode_state_mul1(state);
}
"#;

/// `linear_repacked.metal` **unmodified**, so arm (A) is literally the shipping
/// kernel, compiled by the same compiler invocation as every probe arm.
const PRODUCTION: &str = include_str!("../../kernels/src/linear_repacked.metal");

/// The probe arms. Not concatenated into `LinearKernels::new`; nothing in the
/// shipping library sees these kernels and no production dispatch can reach
/// them. (C)/(D)/(E) are **garbage by construction** and only their timing is
/// meaningful.
const PROBE: &str = r#"
// ===========================================================================
// PROBE ARMS. Read crates/engine/tests/perf_probe_code_rate_p01.rs first.
// Every arm is a LITERAL top-level kernel, not a template: a previous attempt
// factored them into one `template<int ARM>` with the reduction array passed in
// as a `threadgroup float *` (MSL forbids that in a non-kernel function) and
// the parameterisation itself perturbed the schedule -- the replica came out
// bitwise identical to production but 12.6% slower
// (spec/decode-19tok-repacked-250gb/implementation_notes.md:1168-1174).
// ===========================================================================

// Loop-varying, so NOTHING in the decode is loop-invariant and LICM cannot hoist
// it. This is the fix for the void probe in commit 49814d4, which fed the decode
// from 64-bit *literals* and therefore measured "loads gone AND decode gone"
// (spec/decode-19tok-repacked-250gb/implementation_notes.md:1144-1162).
static inline uint probe_word(uint seed, uint tile) {
    return seed ^ (tile * 0x9E3779B9u);
}

// (E): extraction with the five `& 0xffff` masks removed. Deliberately
// bit-different from production -- the question is only whether MSL already
// lowered `(x >> k) & 0xffffu` into a single bit-field op on this target.
static inline void extract_v_mcg_32_nomask(uint wa, uint wb, uint sh0, uint rsh0, thread float v[8]) {
    const uint A_hi = (wa << sh0) | (wb >> rsh0);
    const uint A_lo = wb << sh0;
    v[0] = decode_state_mcg(A_hi >> 16);
    v[1] = decode_state_mcg(A_hi >> 13);
    v[2] = decode_state_mcg(A_hi >> 10);
    v[3] = decode_state_mcg(A_hi >> 7);
    v[4] = decode_state_mcg(A_hi >> 4);
    v[5] = decode_state_mcg(A_hi >> 1);
    v[6] = decode_state_mcg((A_hi << 2) | (A_lo >> 30));
    v[7] = decode_state_mcg((A_hi << 5) | (A_lo >> 27));
}

// --- (B) replica: body identical to linear_gemv_ishizuki_k3_colmajor_2col
kernel void probe_2col_ref(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant const uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]]) {
    constexpr uint KA = 3u;
    constexpr uint RING = 256u * KA;
    constexpr uint WORDS = RING / 32u;
    const uint K = shape[0];
    const uint N = shape[1];
    const uint tiles_n = N / 16u;
    const uint tiles_k = K / 16u;
    const uint pairs_n = tiles_n / 2u;
    constexpr int SG = 8;
    threadgroup float red[2][SG][16];
    const uint r0 = (lane % 4u) * 2u;

    const uint lo0 = (lane * 8u + 1u) * 3u + RING - 16u;
    const uint sh0 = lo0 % 32u;
    const uint rsh0 = 32u - sh0;
    const uint hi0 = lo0 / 32u;
    const uint j0 = hi0 >= WORDS ? hi0 - WORDS : hi0;
    const uint j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u;

    for (uint tn = tgid; tn < pairs_n; tn += ntg) {
        const uint tn0 = tn * 2u;
        const uint tn1 = tn0 + 1u;

        float acc0_0 = 0.0f;
        float acc0_1 = 0.0f;
        float acc1_0 = 0.0f;
        float acc1_1 = 0.0f;

        const device uint *col0 = (const device uint *)packed + tn0 * (tiles_k * WORDS);
        const device uint *col1 = (const device uint *)packed + tn1 * (tiles_k * WORDS);
        const uint stride = WORDS;

        const uint step = SG * 2u;
        const uint k_unroll_end = (tiles_k / step) * step;

        uint tk = sg * 2u;
        for (; tk < k_unroll_end; tk += step) {
            const device half *xr0 = x + (tk + 0u) * 16u + r0;
            const float p00 = float(xr0[0]), p01 = float(xr0[1]), p02 = float(xr0[8]), p03 = float(xr0[9]);
            const device half *xr1 = x + (tk + 1u) * 16u + r0;
            const float p10 = float(xr1[0]), p11 = float(xr1[1]), p12 = float(xr1[8]), p13 = float(xr1[9]);

            const device uint *t0_0 = col0 + (tk + 0u) * stride;
            const device uint *t1_0 = col0 + (tk + 1u) * stride;
            const device uint *t0_1 = col1 + (tk + 0u) * stride;
            const device uint *t1_1 = col1 + (tk + 1u) * stride;

            float v[8];

            // PROBE-DELTA none: this is the replica
            const uint w0_0_0 = t0_0[j0], w0_0_1 = t0_0[j1];
            const uint w1_0_0 = t1_0[j0], w1_0_1 = t1_0[j1];

            extract_v_mcg_32(w0_0_0, w0_0_1, sh0, rsh0, v);
            acc0_0 += p00 * v[0] + p01 * v[1] + p02 * v[2] + p03 * v[3];
            acc0_1 += p00 * v[4] + p01 * v[5] + p02 * v[6] + p03 * v[7];

            extract_v_mcg_32(w1_0_0, w1_0_1, sh0, rsh0, v);
            acc0_0 += p10 * v[0] + p11 * v[1] + p12 * v[2] + p13 * v[3];
            acc0_1 += p10 * v[4] + p11 * v[5] + p12 * v[6] + p13 * v[7];

            const uint w0_1_0 = t0_1[j0], w0_1_1 = t0_1[j1];
            const uint w1_1_0 = t1_1[j0], w1_1_1 = t1_1[j1];

            extract_v_mcg_32(w0_1_0, w0_1_1, sh0, rsh0, v);
            acc1_0 += p00 * v[0] + p01 * v[1] + p02 * v[2] + p03 * v[3];
            acc1_1 += p00 * v[4] + p01 * v[5] + p02 * v[6] + p03 * v[7];

            extract_v_mcg_32(w1_1_0, w1_1_1, sh0, rsh0, v);
            acc1_0 += p10 * v[0] + p11 * v[1] + p12 * v[2] + p13 * v[3];
            acc1_1 += p10 * v[4] + p11 * v[5] + p12 * v[6] + p13 * v[7];
        }

        for (tk = k_unroll_end + sg; tk < tiles_k; tk += SG) {
            const device half *xr0 = x + tk * 16u + r0;
            const float p0 = float(xr0[0]), p1 = float(xr0[1]), p2 = float(xr0[8]), p3 = float(xr0[9]);

            const device uint *t0_0 = col0 + tk * stride;
            const uint a0_0 = t0_0[j0], a1_0 = t0_0[j1];
            float v_0[8];
            extract_v_mcg_32(a0_0, a1_0, sh0, rsh0, v_0);
            acc0_0 += p0 * v_0[0] + p1 * v_0[1] + p2 * v_0[2] + p3 * v_0[3];
            acc0_1 += p0 * v_0[4] + p1 * v_0[5] + p2 * v_0[6] + p3 * v_0[7];

            const device uint *t0_1 = col1 + tk * stride;
            const uint a0_1 = t0_1[j0], a1_1 = t0_1[j1];
            float v_1[8];
            extract_v_mcg_32(a0_1, a1_1, sh0, rsh0, v_1);
            acc1_0 += p0 * v_1[0] + p1 * v_1[1] + p2 * v_1[2] + p3 * v_1[3];
            acc1_1 += p0 * v_1[4] + p1 * v_1[5] + p2 * v_1[6] + p3 * v_1[7];
        }

        acc0_0 += simd_shuffle_xor(acc0_0, 1u);
        acc0_0 += simd_shuffle_xor(acc0_0, 2u);
        acc0_1 += simd_shuffle_xor(acc0_1, 1u);
        acc0_1 += simd_shuffle_xor(acc0_1, 2u);
        acc1_0 += simd_shuffle_xor(acc1_0, 1u);
        acc1_0 += simd_shuffle_xor(acc1_0, 2u);
        acc1_1 += simd_shuffle_xor(acc1_1, 1u);
        acc1_1 += simd_shuffle_xor(acc1_1, 2u);

        if (lane % 4u == 0u) {
            red[0][sg][lane / 4u] = acc0_0;
            red[0][sg][lane / 4u + 8u] = acc0_1;
            red[1][sg][lane / 4u] = acc1_0;
            red[1][sg][lane / 4u + 8u] = acc1_1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < 32u) {
            const uint col_idx = lid / 16u;
            const uint out_lane = lid % 16u;
            float sum = 0.0f;
            for (int g = 0; g < SG; ++g) {
                sum += red[col_idx][g][out_lane];
            }
            const uint out_tn = (col_idx == 0u) ? tn0 : tn1;
            y[out_tn * 16u + out_lane] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// --- (C) NO weight loads, tk-varying synthetic words, full decode kept live.
kernel void probe_2col_noload(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant const uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]]) {
    constexpr uint KA = 3u;
    constexpr uint RING = 256u * KA;
    constexpr uint WORDS = RING / 32u;
    const uint K = shape[0];
    const uint N = shape[1];
    const uint tiles_n = N / 16u;
    const uint tiles_k = K / 16u;
    const uint pairs_n = tiles_n / 2u;
    constexpr int SG = 8;
    threadgroup float red[2][SG][16];
    const uint r0 = (lane % 4u) * 2u;
    const uint seed_a = shape[2];
    const uint seed_b = shape[3];

    const uint lo0 = (lane * 8u + 1u) * 3u + RING - 16u;
    const uint sh0 = lo0 % 32u;
    const uint rsh0 = 32u - sh0;
    const uint hi0 = lo0 / 32u;
    const uint j0 = hi0 >= WORDS ? hi0 - WORDS : hi0;
    const uint j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u;

    for (uint tn = tgid; tn < pairs_n; tn += ntg) {
        const uint tn0 = tn * 2u;
        const uint tn1 = tn0 + 1u;

        float acc0_0 = 0.0f;
        float acc0_1 = 0.0f;
        float acc1_0 = 0.0f;
        float acc1_1 = 0.0f;

        const uint step = SG * 2u;
        const uint k_unroll_end = (tiles_k / step) * step;

        uint tk = sg * 2u;
        for (; tk < k_unroll_end; tk += step) {
            const device half *xr0 = x + (tk + 0u) * 16u + r0;
            const float p00 = float(xr0[0]), p01 = float(xr0[1]), p02 = float(xr0[8]), p03 = float(xr0[9]);
            const device half *xr1 = x + (tk + 1u) * 16u + r0;
            const float p10 = float(xr1[0]), p11 = float(xr1[1]), p12 = float(xr1[8]), p13 = float(xr1[9]);

            float v[8];

            // PROBE-DELTA (C): the four weight loads become synthetic words that
            // depend on `tk`, so the eight-times-unrolled decode is not
            // loop-invariant and cannot be hoisted.
            const uint w0_0_0 = probe_word(seed_a, tk + 0u), w0_0_1 = probe_word(seed_b, tk + 0u);
            const uint w1_0_0 = probe_word(seed_a, tk + 1u), w1_0_1 = probe_word(seed_b, tk + 1u);

            extract_v_mcg_32(w0_0_0, w0_0_1, sh0, rsh0, v);
            acc0_0 += p00 * v[0] + p01 * v[1] + p02 * v[2] + p03 * v[3];
            acc0_1 += p00 * v[4] + p01 * v[5] + p02 * v[6] + p03 * v[7];

            extract_v_mcg_32(w1_0_0, w1_0_1, sh0, rsh0, v);
            acc0_0 += p10 * v[0] + p11 * v[1] + p12 * v[2] + p13 * v[3];
            acc0_1 += p10 * v[4] + p11 * v[5] + p12 * v[6] + p13 * v[7];

            const uint w0_1_0 = probe_word(seed_a, tk + 2u), w0_1_1 = probe_word(seed_b, tk + 2u);
            const uint w1_1_0 = probe_word(seed_a, tk + 3u), w1_1_1 = probe_word(seed_b, tk + 3u);

            extract_v_mcg_32(w0_1_0, w0_1_1, sh0, rsh0, v);
            acc1_0 += p00 * v[0] + p01 * v[1] + p02 * v[2] + p03 * v[3];
            acc1_1 += p00 * v[4] + p01 * v[5] + p02 * v[6] + p03 * v[7];

            extract_v_mcg_32(w1_1_0, w1_1_1, sh0, rsh0, v);
            acc1_0 += p10 * v[0] + p11 * v[1] + p12 * v[2] + p13 * v[3];
            acc1_1 += p10 * v[4] + p11 * v[5] + p12 * v[6] + p13 * v[7];
        }

        for (tk = k_unroll_end + sg; tk < tiles_k; tk += SG) {
            const device half *xr0 = x + tk * 16u + r0;
            const float p0 = float(xr0[0]), p1 = float(xr0[1]), p2 = float(xr0[8]), p3 = float(xr0[9]);

            const uint a0_0 = probe_word(seed_a, tk + 8u), a1_0 = probe_word(seed_b, tk + 8u);
            float v_0[8];
            extract_v_mcg_32(a0_0, a1_0, sh0, rsh0, v_0);
            acc0_0 += p0 * v_0[0] + p1 * v_0[1] + p2 * v_0[2] + p3 * v_0[3];
            acc0_1 += p0 * v_0[4] + p1 * v_0[5] + p2 * v_0[6] + p3 * v_0[7];

            const uint a0_1 = probe_word(seed_a, tk + 9u), a1_1 = probe_word(seed_b, tk + 9u);
            float v_1[8];
            extract_v_mcg_32(a0_1, a1_1, sh0, rsh0, v_1);
            acc1_0 += p0 * v_1[0] + p1 * v_1[1] + p2 * v_1[2] + p3 * v_1[3];
            acc1_1 += p0 * v_1[4] + p1 * v_1[5] + p2 * v_1[6] + p3 * v_1[7];
        }

        acc0_0 += simd_shuffle_xor(acc0_0, 1u);
        acc0_0 += simd_shuffle_xor(acc0_0, 2u);
        acc0_1 += simd_shuffle_xor(acc0_1, 1u);
        acc0_1 += simd_shuffle_xor(acc0_1, 2u);
        acc1_0 += simd_shuffle_xor(acc1_0, 1u);
        acc1_0 += simd_shuffle_xor(acc1_0, 2u);
        acc1_1 += simd_shuffle_xor(acc1_1, 1u);
        acc1_1 += simd_shuffle_xor(acc1_1, 2u);

        if (lane % 4u == 0u) {
            red[0][sg][lane / 4u] = acc0_0;
            red[0][sg][lane / 4u + 8u] = acc0_1;
            red[1][sg][lane / 4u] = acc1_0;
            red[1][sg][lane / 4u + 8u] = acc1_1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < 32u) {
            const uint col_idx = lid / 16u;
            const uint out_lane = lid % 16u;
            float sum = 0.0f;
            for (int g = 0; g < SG; ++g) {
                sum += red[col_idx][g][out_lane];
            }
            const uint out_tn = (col_idx == 0u) ? tn0 : tn1;
            y[out_tn * 16u + out_lane] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// --- (D) constant shift amounts. Isolates the cost of the three VARIABLE
//     shifts in the window setup. sh0 is only ever 3/11/19/27 on this target
//     ((24*lane+755) mod 32), so the funnel shift needs no select; 19/13 is
//     lane 0's real pair and equivalent in cost to the other three.
kernel void probe_2col_constshift(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant const uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]]) {
    constexpr uint KA = 3u;
    constexpr uint RING = 256u * KA;
    constexpr uint WORDS = RING / 32u;
    const uint K = shape[0];
    const uint N = shape[1];
    const uint tiles_n = N / 16u;
    const uint tiles_k = K / 16u;
    const uint pairs_n = tiles_n / 2u;
    constexpr int SG = 8;
    threadgroup float red[2][SG][16];
    const uint r0 = (lane % 4u) * 2u;

    // PROBE-DELTA (D): sh0 pinned to a constant.
    const uint sh0 = 19u;
    const uint rsh0 = 13u;
    const uint hi0 = ((lane * 8u + 1u) * 3u + RING - 16u) / 32u;
    const uint j0 = hi0 >= WORDS ? hi0 - WORDS : hi0;
    const uint j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u;

    for (uint tn = tgid; tn < pairs_n; tn += ntg) {
        const uint tn0 = tn * 2u;
        const uint tn1 = tn0 + 1u;

        float acc0_0 = 0.0f;
        float acc0_1 = 0.0f;
        float acc1_0 = 0.0f;
        float acc1_1 = 0.0f;

        const device uint *col0 = (const device uint *)packed + tn0 * (tiles_k * WORDS);
        const device uint *col1 = (const device uint *)packed + tn1 * (tiles_k * WORDS);
        const uint stride = WORDS;

        const uint step = SG * 2u;
        const uint k_unroll_end = (tiles_k / step) * step;

        uint tk = sg * 2u;
        for (; tk < k_unroll_end; tk += step) {
            const device half *xr0 = x + (tk + 0u) * 16u + r0;
            const float p00 = float(xr0[0]), p01 = float(xr0[1]), p02 = float(xr0[8]), p03 = float(xr0[9]);
            const device half *xr1 = x + (tk + 1u) * 16u + r0;
            const float p10 = float(xr1[0]), p11 = float(xr1[1]), p12 = float(xr1[8]), p13 = float(xr1[9]);

            const device uint *t0_0 = col0 + (tk + 0u) * stride;
            const device uint *t1_0 = col0 + (tk + 1u) * stride;
            const device uint *t0_1 = col1 + (tk + 0u) * stride;
            const device uint *t1_1 = col1 + (tk + 1u) * stride;

            float v[8];
            const uint w0_0_0 = t0_0[j0], w0_0_1 = t0_0[j1];
            extract_v_mcg_32(w0_0_0, w0_0_1, sh0, rsh0, v);
            acc0_0 += p00 * v[0] + p01 * v[1] + p02 * v[2] + p03 * v[3];
            acc0_1 += p00 * v[4] + p01 * v[5] + p02 * v[6] + p03 * v[7];
            const uint w1_0_0 = t1_0[j0], w1_0_1 = t1_0[j1];
            extract_v_mcg_32(w1_0_0, w1_0_1, sh0, rsh0, v);
            acc0_0 += p10 * v[0] + p11 * v[1] + p12 * v[2] + p13 * v[3];
            acc0_1 += p10 * v[4] + p11 * v[5] + p12 * v[6] + p13 * v[7];
            const uint w0_1_0 = t0_1[j0], w0_1_1 = t0_1[j1];
            extract_v_mcg_32(w0_1_0, w0_1_1, sh0, rsh0, v);
            acc1_0 += p00 * v[0] + p01 * v[1] + p02 * v[2] + p03 * v[3];
            acc1_1 += p00 * v[4] + p01 * v[5] + p02 * v[6] + p03 * v[7];
            const uint w1_1_0 = t1_1[j0], w1_1_1 = t1_1[j1];
            extract_v_mcg_32(w1_1_0, w1_1_1, sh0, rsh0, v);
            acc1_0 += p10 * v[0] + p11 * v[1] + p12 * v[2] + p13 * v[3];
            acc1_1 += p10 * v[4] + p11 * v[5] + p12 * v[6] + p13 * v[7];
        }

        for (tk = k_unroll_end + sg; tk < tiles_k; tk += SG) {
            const device half *xr0 = x + tk * 16u + r0;
            const float p0 = float(xr0[0]), p1 = float(xr0[1]), p2 = float(xr0[8]), p3 = float(xr0[9]);
            const device uint *t0_0 = col0 + tk * stride;
            const uint a0_0 = t0_0[j0], a1_0 = t0_0[j1];
            float v_0[8];
            extract_v_mcg_32(a0_0, a1_0, sh0, rsh0, v_0);
            acc0_0 += p0 * v_0[0] + p1 * v_0[1] + p2 * v_0[2] + p3 * v_0[3];
            acc0_1 += p0 * v_0[4] + p1 * v_0[5] + p2 * v_0[6] + p3 * v_0[7];
            const device uint *t0_1 = col1 + tk * stride;
            const uint a0_1 = t0_1[j0], a1_1 = t0_1[j1];
            float v_1[8];
            extract_v_mcg_32(a0_1, a1_1, sh0, rsh0, v_1);
            acc1_0 += p0 * v_1[0] + p1 * v_1[1] + p2 * v_1[2] + p3 * v_1[3];
            acc1_1 += p0 * v_1[4] + p1 * v_1[5] + p2 * v_1[6] + p3 * v_1[7];
        }

        acc0_0 += simd_shuffle_xor(acc0_0, 1u);
        acc0_0 += simd_shuffle_xor(acc0_0, 2u);
        acc0_1 += simd_shuffle_xor(acc0_1, 1u);
        acc0_1 += simd_shuffle_xor(acc0_1, 2u);
        acc1_0 += simd_shuffle_xor(acc1_0, 1u);
        acc1_0 += simd_shuffle_xor(acc1_0, 2u);
        acc1_1 += simd_shuffle_xor(acc1_1, 1u);
        acc1_1 += simd_shuffle_xor(acc1_1, 2u);

        if (lane % 4u == 0u) {
            red[0][sg][lane / 4u] = acc0_0;
            red[0][sg][lane / 4u + 8u] = acc0_1;
            red[1][sg][lane / 4u] = acc1_0;
            red[1][sg][lane / 4u + 8u] = acc1_1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < 32u) {
            const uint col_idx = lid / 16u;
            const uint out_lane = lid % 16u;
            float sum = 0.0f;
            for (int g = 0; g < SG; ++g) {
                sum += red[col_idx][g][out_lane];
            }
            const uint out_tn = (col_idx == 0u) ? tn0 : tn1;
            y[out_tn * 16u + out_lane] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// --- (E) the five `& 0xffff` extraction masks removed. Tells us whether MSL
//     already emits a single bit-field op for `(x >> k) & 0xffffu`, which is
//     worth 0.63 instructions per code.
kernel void probe_2col_nomask(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant const uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]]) {
    constexpr uint KA = 3u;
    constexpr uint RING = 256u * KA;
    constexpr uint WORDS = RING / 32u;
    const uint K = shape[0];
    const uint N = shape[1];
    const uint tiles_n = N / 16u;
    const uint tiles_k = K / 16u;
    const uint pairs_n = tiles_n / 2u;
    constexpr int SG = 8;
    threadgroup float red[2][SG][16];
    const uint r0 = (lane % 4u) * 2u;

    const uint lo0 = (lane * 8u + 1u) * 3u + RING - 16u;
    const uint sh0 = lo0 % 32u;
    const uint rsh0 = 32u - sh0;
    const uint hi0 = lo0 / 32u;
    const uint j0 = hi0 >= WORDS ? hi0 - WORDS : hi0;
    const uint j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u;

    for (uint tn = tgid; tn < pairs_n; tn += ntg) {
        const uint tn0 = tn * 2u;
        const uint tn1 = tn0 + 1u;

        float acc0_0 = 0.0f;
        float acc0_1 = 0.0f;
        float acc1_0 = 0.0f;
        float acc1_1 = 0.0f;

        const device uint *col0 = (const device uint *)packed + tn0 * (tiles_k * WORDS);
        const device uint *col1 = (const device uint *)packed + tn1 * (tiles_k * WORDS);
        const uint stride = WORDS;

        const uint step = SG * 2u;
        const uint k_unroll_end = (tiles_k / step) * step;

        uint tk = sg * 2u;
        for (; tk < k_unroll_end; tk += step) {
            const device half *xr0 = x + (tk + 0u) * 16u + r0;
            const float p00 = float(xr0[0]), p01 = float(xr0[1]), p02 = float(xr0[8]), p03 = float(xr0[9]);
            const device half *xr1 = x + (tk + 1u) * 16u + r0;
            const float p10 = float(xr1[0]), p11 = float(xr1[1]), p12 = float(xr1[8]), p13 = float(xr1[9]);

            const device uint *t0_0 = col0 + (tk + 0u) * stride;
            const device uint *t1_0 = col0 + (tk + 1u) * stride;
            const device uint *t0_1 = col1 + (tk + 0u) * stride;
            const device uint *t1_1 = col1 + (tk + 1u) * stride;

            float v[8];
            const uint w0_0_0 = t0_0[j0], w0_0_1 = t0_0[j1];
            extract_v_mcg_32_nomask(w0_0_0, w0_0_1, sh0, rsh0, v);
            acc0_0 += p00 * v[0] + p01 * v[1] + p02 * v[2] + p03 * v[3];
            acc0_1 += p00 * v[4] + p01 * v[5] + p02 * v[6] + p03 * v[7];
            const uint w1_0_0 = t1_0[j0], w1_0_1 = t1_0[j1];
            extract_v_mcg_32_nomask(w1_0_0, w1_0_1, sh0, rsh0, v);
            acc0_0 += p10 * v[0] + p11 * v[1] + p12 * v[2] + p13 * v[3];
            acc0_1 += p10 * v[4] + p11 * v[5] + p12 * v[6] + p13 * v[7];
            const uint w0_1_0 = t0_1[j0], w0_1_1 = t0_1[j1];
            extract_v_mcg_32_nomask(w0_1_0, w0_1_1, sh0, rsh0, v);
            acc1_0 += p00 * v[0] + p01 * v[1] + p02 * v[2] + p03 * v[3];
            acc1_1 += p00 * v[4] + p01 * v[5] + p02 * v[6] + p03 * v[7];
            const uint w1_1_0 = t1_1[j0], w1_1_1 = t1_1[j1];
            extract_v_mcg_32_nomask(w1_1_0, w1_1_1, sh0, rsh0, v);
            acc1_0 += p10 * v[0] + p11 * v[1] + p12 * v[2] + p13 * v[3];
            acc1_1 += p10 * v[4] + p11 * v[5] + p12 * v[6] + p13 * v[7];
        }

        for (tk = k_unroll_end + sg; tk < tiles_k; tk += SG) {
            const device half *xr0 = x + tk * 16u + r0;
            const float p0 = float(xr0[0]), p1 = float(xr0[1]), p2 = float(xr0[8]), p3 = float(xr0[9]);
            const device uint *t0_0 = col0 + tk * stride;
            const uint a0_0 = t0_0[j0], a1_0 = t0_0[j1];
            float v_0[8];
            extract_v_mcg_32_nomask(a0_0, a1_0, sh0, rsh0, v_0);
            acc0_0 += p0 * v_0[0] + p1 * v_0[1] + p2 * v_0[2] + p3 * v_0[3];
            acc0_1 += p0 * v_0[4] + p1 * v_0[5] + p2 * v_0[6] + p3 * v_0[7];
            const device uint *t0_1 = col1 + tk * stride;
            const uint a0_1 = t0_1[j0], a1_1 = t0_1[j1];
            float v_1[8];
            extract_v_mcg_32_nomask(a0_1, a1_1, sh0, rsh0, v_1);
            acc1_0 += p0 * v_1[0] + p1 * v_1[1] + p2 * v_1[2] + p3 * v_1[3];
            acc1_1 += p0 * v_1[4] + p1 * v_1[5] + p2 * v_1[6] + p3 * v_1[7];
        }

        acc0_0 += simd_shuffle_xor(acc0_0, 1u);
        acc0_0 += simd_shuffle_xor(acc0_0, 2u);
        acc0_1 += simd_shuffle_xor(acc0_1, 1u);
        acc0_1 += simd_shuffle_xor(acc0_1, 2u);
        acc1_0 += simd_shuffle_xor(acc1_0, 1u);
        acc1_0 += simd_shuffle_xor(acc1_0, 2u);
        acc1_1 += simd_shuffle_xor(acc1_1, 1u);
        acc1_1 += simd_shuffle_xor(acc1_1, 2u);

        if (lane % 4u == 0u) {
            red[0][sg][lane / 4u] = acc0_0;
            red[0][sg][lane / 4u + 8u] = acc0_1;
            red[1][sg][lane / 4u] = acc1_0;
            red[1][sg][lane / 4u + 8u] = acc1_1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < 32u) {
            const uint col_idx = lid / 16u;
            const uint out_lane = lid % 16u;
            float sum = 0.0f;
            for (int g = 0; g < SG; ++g) {
                sum += red[col_idx][g][out_lane];
            }
            const uint out_tn = (col_idx == 0u) ? tn0 : tn1;
            y[out_tn * 16u + out_lane] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// --- (G) THE DECISIVE ARM. Register-resident: the identical per-8-code
//     instruction stream (extract_v_mcg_32 + 8x decode_state_mcg + 8 adds,
//     four groups sequenced exactly as the production inner loop sequences its
//     four tile-decodes) with ZERO memory traffic. This is the machine's
//     instruction ceiling for this op mix, measured rather than counted, and it
//     needs no Metal disassembler.
//
//     `w` is loop-carried so nothing can be hoisted or strength-reduced, and the
//     accumulators are stored so nothing is dead-code-eliminated. The adds are
//     `v[0]+v[1]+...` rather than `p*v[0]+...` because there is no `x` here;
//     they still cost 7 ops per 8 codes against the production 8 FMAs, so the arm
//     is if anything 1 op per 8 codes CHEAPER than the loop it calibrates.
kernel void probe_code_ceiling(
    const device uint *seed [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant const uint *ctrl [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    const uint iters = ctrl[0];
    const uint sh0 = ctrl[1];
    const uint rsh0 = ctrl[2];
    uint w = seed[gid & 255u] | 1u;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint it = 0u; it < iters; ++it) {
        float v[8];
        extract_v_mcg_32(w, w * 2654435761u, sh0, rsh0, v);
        a0 += v[0] + v[1] + v[2] + v[3];
        a1 += v[4] + v[5] + v[6] + v[7];
        w = w * 1664525u + 1013904223u;

        extract_v_mcg_32(w, w * 2654435761u, sh0, rsh0, v);
        a0 += v[0] + v[1] + v[2] + v[3];
        a1 += v[4] + v[5] + v[6] + v[7];
        w = w * 1664525u + 1013904223u;

        extract_v_mcg_32(w, w * 2654435761u, sh0, rsh0, v);
        a2 += v[0] + v[1] + v[2] + v[3];
        a3 += v[4] + v[5] + v[6] + v[7];
        w = w * 1664525u + 1013904223u;

        extract_v_mcg_32(w, w * 2654435761u, sh0, rsh0, v);
        a2 += v[0] + v[1] + v[2] + v[3];
        a3 += v[4] + v[5] + v[6] + v[7];
        w = w * 1664525u + 1013904223u;
    }
    out[gid] = a0 + a1 + a2 + a3;
}
"#;

/// The GEMV arms, in interleaved order.
const ARMS: [(&str, &str); 5] = [
    ("linear_gemv_ishizuki_k3_colmajor_2col", "(a) production _colmajor_2col"),
    ("probe_2col_ref", "(b) replica (FIDELITY gate)"),
    ("probe_2col_noload", "(c) NO weight loads, decode live"),
    ("probe_2col_constshift", "(d) constant shift amount"),
    ("probe_2col_nomask", "(e) extraction masks removed"),
];

/// 20 tok/s needs this code rate: 236.7 GB/s at 3 bpw.
const GCODES_FOR_20TOK: f64 = 236.7 / 3.0 * 8.0;
/// Best 3 bpw kernel ever measured in this tree.
const MEASURED_PEAK_GBPS: f64 = 203.0;
const MEASURED_PEAK_GCODES: f64 = MEASURED_PEAK_GBPS / 3.0 * 8.0;
/// GB/token, for turning a code rate into tok/s of GEMV time.
const GB_PER_TOKEN: f64 = 12.04;

// ===========================================================================
// Real checkpoint bytes
// ===========================================================================

fn test_model() -> PathBuf {
    PathBuf::from(std::env::var("KAPLUMBA_TEST_MODEL").unwrap_or_else(|_| {
        let home = std::env::var("HOME").expect("HOME");
        format!("{home}/.kaplumba/models/Gxma-repacked")
    }))
}

/// One `.trellis` tensor: name, safetensors `shape`, payload as `u16`.
///
/// safetensors stores a linear weight as `[out_features, in_features]`, so
/// `N = shape[0]` and `K = shape[1]`. That fixes the grid (`pairs = N/32`,
/// i.e. 168 threadgroups for `down_proj`), which is what the bandwidth number
/// depends on; the byte count is the same either way.
struct Tensor {
    name: String,
    n: usize,
    k: usize,
    words: Vec<u16>,
}

impl Tensor {
    fn codes(&self) -> usize {
        self.n * self.k
    }
    /// bytes at 3 bpw
    fn bytes(&self) -> usize {
        self.codes() * 3 / 8
    }
}

struct ShardHeader {
    shard: String,
    tensors: serde_json::Map<String, serde_json::Value>,
}

fn read_shard_headers(model: &Path) -> Vec<ShardHeader> {
    let mut shards: Vec<String> = Vec::new();
    let index_path = model.join("model.safetensors.index.json");
    if index_path.exists() {
        let index: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&index_path).expect("index")).expect("index json");
        if let Some(map) = index["weight_map"].as_object() {
            for v in map.values() {
                if let Some(s) = v.as_str() {
                    if !shards.iter().any(|x| x == s) {
                        shards.push(s.to_string());
                    }
                }
            }
        }
    }
    if shards.is_empty() {
        let mut found: Vec<String> = Vec::new();
        for entry in std::fs::read_dir(model).expect("model dir") {
            let path = entry.expect("dir entry").path();
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            if name.ends_with(".safetensors") {
                found.push(name);
            }
        }
        found.sort();
        shards = found;
    }
    assert!(
        !shards.is_empty(),
        "no .safetensors shards under {model:?}; is KAPLUMBA_TEST_MODEL a checkpoint directory?"
    );

    shards
        .into_iter()
        .map(|shard| {
            let mut file = File::open(model.join(&shard)).expect("shard");
            let mut len = [0u8; 8];
            file.read_exact(&mut len).expect("length prefix");
            let header_len = u64::from_le_bytes(len) as usize;
            let mut header_bytes = vec![0u8; header_len];
            file.read_exact(&mut header_bytes).expect("header");
            let header: serde_json::Value =
                serde_json::from_slice(&header_bytes).expect("header json");
            let mut tensors = serde_json::Map::new();
            if let Some(obj) = header.as_object() {
                for (k, v) in obj {
                    tensors.insert(k.clone(), v.clone());
                }
            }
            ShardHeader { shard, tensors }
        })
        .collect()
}

/// The largest genuinely-3-bpw `.trellis` tensors, biggest first.
///
/// The bit rate is discriminated by exact arithmetic, not by a metadata field:
/// `payload_len * 8 == prod(shape) * 3` holds only at 3 bpw, so the 6 bpw
/// `lm_head.trellis` (where it equals `prod(shape) * 6`) is excluded without
/// trusting `quantization_config.json` -- that file is known stale relative to
/// the payload in this repo
/// (`spec/decode-3bpw-amplification/implementation_notes.md:271-272`).
fn load_real_3bpw_tensors(model: &Path, want: usize) -> Vec<Tensor> {
    let mut found: Vec<Tensor> = Vec::new();
    for shard in read_shard_headers(model) {
        for (name, meta) in shard.tensors {
            if !name.ends_with(".trellis") || meta["dtype"].as_str() != Some("U16") {
                continue;
            }
            let shape: Vec<usize> = match meta["shape"].as_array() {
                Some(a) => a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect(),
                None => continue,
            };
            if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
                continue;
            }
            let prod = shape[0] * shape[1];
            let offsets = match meta["data_offsets"].as_array() {
                Some(a) if a.len() == 2 => a,
                _ => continue,
            };
            let start = offsets[0].as_u64().expect("start");
            let end = offsets[1].as_u64().expect("end");
            let len = (end - start) as usize;
            if len * 8 != prod * 3 {
                continue; // not 3 bpw
            }
            let mut file = File::open(model.join(&shard.shard)).expect("shard");
            file.seek(SeekFrom::Start(start)).expect("seek");
            let mut bytes = vec![0u8; len];
            file.read_exact(&mut bytes).expect("payload");
            found.push(Tensor {
                name,
                n: shape[0],
                k: shape[1],
                words: bytes
                    .chunks_exact(2)
                    .map(|p| u16::from_le_bytes([p[0], p[1]]))
                    .collect(),
            });
        }
    }
    found.sort_by_key(|t| std::cmp::Reverse(t.codes()));
    assert!(
        found.len() >= want,
        "found only {} 3 bpw .trellis tensors under {model:?}, wanted {want}",
        found.len()
    );
    found.truncate(want);
    found
}

// ===========================================================================
// Library / dispatch
// ===========================================================================

struct Ctx {
    queue: CommandQueue,
    library: Library,
    arms: Vec<(String, ComputePipelineState)>,
}

fn build_library(device: &Device) -> Ctx {
    let mut src = String::with_capacity(PREAMBLE.len() + PRODUCTION.len() + PROBE.len() + 3);
    src.push_str(PREAMBLE);
    src.push('\n');
    src.push_str(PRODUCTION);
    src.push('\n');
    src.push_str(PROBE);
    let library = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .expect("compile probe library");
    // The production kernel must be present, or the whole probe measures the
    // wrong thing. `LinearKernels::new` failing open has happened three times in
    // this repo (`spec/decode-3bpw-amplification/implementation_notes.md:240`),
    // so this is an assertion and not a `skip`.
    for (name, _) in ARMS {
        assert!(
            library.get_function(name, None).is_ok(),
            "probe library does not contain {name}"
        );
    }
    let mut arms = Vec::new();
    for (name, _) in ARMS {
        let function = library
            .get_function(name, None)
            .unwrap_or_else(|e| panic!("probe library has no {name}: {e}"));
        let state = device
            .new_compute_pipeline_state_with_function(&function)
            .unwrap_or_else(|e| panic!("pipeline {name}: {e}"));
        arms.push((name.to_string(), state));
    }
    Ctx {
        queue: device.new_command_queue(),
        library,
        arms,
    }
}

fn pipeline<'a>(ctx: &'a Ctx, name: &str) -> &'a ComputePipelineState {
    ctx.arms
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, s)| s)
        .unwrap_or_else(|| panic!("no pipeline {name}"))
}

/// `shape` block: `{K, N, seed_a, seed_b}`. Production reads only the first
/// two; the probe arms read the seeds. 16 bytes for every arm.
fn shape_bytes(k: usize, n: usize, seeds: [u32; 2]) -> [u8; 16] {
    let params: [u32; 4] = [k as u32, n as u32, seeds[0], seeds[1]];
    let mut out = [0u8; 16];
    for (i, v) in params.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    out
}

/// `LinearKernels::gemv_max_groups` (private; `crates/kernels/src/linear.rs:94`):
/// `u64::MAX` when `KAPLUMBA_GEMV_MAX_GROUPS` is unset, which is the default.
fn gemv_max_groups() -> u64 {
    std::env::var("KAPLUMBA_GEMV_MAX_GROUPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(u64::MAX)
}

fn dispatch_gemv(
    ctx: &Ctx,
    name: &str,
    packed: &Buffer,
    x: &Buffer,
    y: &Buffer,
    k: usize,
    n: usize,
    seeds: [u32; 2],
    enc: &ComputeCommandEncoderRef,
) {
    enc.set_compute_pipeline_state(pipeline(ctx, name));
    enc.set_buffer(0, Some(packed), 0);
    enc.set_buffer(1, Some(x), 0);
    enc.set_buffer(2, Some(y), 0);
    enc.set_bytes(
        3,
        16,
        shape_bytes(k, n, seeds).as_ptr() as *const std::ffi::c_void,
    );
    // Grid matches `LinearKernels::encode_gemv_raw`'s `_2col` arm
    // (`crates/kernels/src/linear.rs:2142-2153`): `pairs = out_features / 32`
    // capped by `gemv_max_groups()`, 256 threads per group.
    let pairs = ((n / 32) as u64).min(gemv_max_groups()).max(1);
    enc.dispatch_threads(MTLSize::new(pairs * 256, 1, 1), MTLSize::new(256, 1, 1));
}

/// Upload any `Copy` POD slice as raw bytes. `new_buffer_with_data` copies, so
/// the caller may drop immediately.
fn upload<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    device.new_buffer_with_data(
        data.as_ptr() as *const std::ffi::c_void,
        std::mem::size_of_val(data) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn read_f32(buffer: &Buffer, count: usize) -> Vec<f32> {
    let ptr = buffer.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, count) }.to_vec()
}

/// `x` is small (K halfs) and L1-resident; a sinusoid keeps it away from denormals.
fn sin_x(k: usize) -> Vec<f16> {
    (0..k)
        .map(|i| f16::from_f32(((i as f32) * 0.01).sin()))
        .collect()
}

fn arg_usize(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s[s.len() / 2]
}

fn gbps_from(codes: usize, ms: f64) -> f64 {
    (codes as f64 * 3.0 / 8.0) / (ms * 1e-3) / 1e9
}

fn gcodes_from(codes: usize, ms: f64) -> f64 {
    (codes as f64) / (ms * 1e-3) / 1e9
}

// ===========================================================================
// Test 1: the falsification probe, on real weights
// ===========================================================================

#[test]
#[ignore = "needs sole GPU access and the Gxma-repacked checkpoint"]
fn p01_code_rate_issue_bound() {
    let model = test_model();
    assert!(
        model.is_dir(),
        "KAPLUMBA_TEST_MODEL={model:?} is not a directory"
    );

    let device = Device::system_default().expect("Metal device");
    let ctx = build_library(&device);
    println!(
        "probe library compiled: {} arms, production _colmajor_2col present",
        ctx.arms.len()
    );
    assert!(ctx.library.get_function(ARMS[0].0, None).is_ok());

    let rounds = arg_usize("--rounds", 11);
    let iters = arg_usize("--iters", 12);
    let tensors = load_real_3bpw_tensors(&model, 2);

    for t in &tensors {
        let (k, n) = (t.k, t.n);
        println!(
            "\n=== REAL WEIGHTS {name}: K={k} N={n}  grid={groups} TGs x 256  \
             ({mb:.1} MB of 3 bpw trellis, {codes} codes)",
            name = t.name,
            groups = ((n / 32) as u64).min(gemv_max_groups()).max(1),
            mb = t.bytes() as f64 / 1e6,
            codes = k * n,
        );
        if k % 16 != 0 || n % 32 != 0 {
            println!("  SKIP: not tile-divisible (needs K%16==0, N%32==0)");
            continue;
        }

        let packed = upload(&device, &t.words);
        let xb = upload(&device, &sin_x(k));
        let yb = device.new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared);
        let seeds = [0x1111_1111u32, 0x2222_2222u32];
        let seeds2 = [0x9abc_def0u32, 0x0fed_cba9u32];

        // ---------------- gates, before any timing is reported --------------
        let run_once = |name: &str, sd: [u32; 2]| -> Vec<f32> {
            let cb = ctx.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            dispatch_gemv(&ctx, name, &packed, &xb, &yb, k, n, sd, enc);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            read_f32(&yb, n)
        };

        let y_prod = run_once(ARMS[0].0, seeds);
        assert!(
            y_prod.iter().all(|v| v.is_finite()) && y_prod.iter().any(|v| *v != 0.0),
            "production arm produced non-finite or all-zero output on real weights"
        );
        let y_ref = run_once(ARMS[1].0, seeds);
        assert_eq!(
            y_ref, y_prod,
            "FIDELITY GATE: probe_2col_ref is not bitwise equal to production on REAL weights \
             ({} K={k} N={n}). The probe arms are attributed against (b), so every one of them \
             is void until this holds.",
            t.name
        );
        println!("  FIDELITY: probe_2col_ref bitwise-equal to production on real weights = true");

        let c_a = run_once(ARMS[2].0, seeds);
        let c_b = run_once(ARMS[2].0, seeds2);
        assert!(
            c_a != c_b && c_a != y_prod && c_a.iter().any(|v| *v != 0.0),
            "DCE GATE: probe_2col_noload is seed-independent or matches production -- its decode \
             has been eliminated and the arm measures nothing"
        );
        println!(
            "  DCE: noload seed-dependent = true, differs from production = true, nonzero = true, \
             finite = {}",
            c_a.iter().all(|v| v.is_finite())
        );
        for (arm, label) in [(ARMS[3].0, "(d) const shift"), (ARMS[4].0, "(e) no mask")] {
            let y = run_once(arm, seeds);
            println!(
                "  {label:<20} differs from production = {}  (garbage by construction, intended)",
                y != y_prod
            );
        }

        // ---------------- interleaved A B A B, min estimator ----------------
        for (name, _) in ARMS {
            let cb = ctx.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            dispatch_gemv(&ctx, name, &packed, &xb, &yb, k, n, seeds, enc);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); ARMS.len()];
        for _round in 0..rounds {
            for (ai, (arm, _)) in ARMS.iter().enumerate() {
                let cb = ctx.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                for _ in 0..iters {
                    dispatch_gemv(&ctx, arm, &packed, &xb, &yb, k, n, seeds, enc);
                }
                enc.end_encoding();
                let t = Instant::now();
                cb.commit();
                cb.wait_until_completed();
                samples[ai].push(t.elapsed().as_secs_f64() * 1e3 / iters as f64);
            }
        }

        let min = |i: usize| samples[i].iter().cloned().fold(f64::MAX, f64::min);
        let worst_ratio = samples
            .iter()
            .map(|s| median(s) / s.iter().cloned().fold(f64::MAX, f64::min))
            .fold(0.0f64, f64::max);

        println!(
            "\n  {:<34} {:>9} {:>9} {:>9} {:>10}",
            "arm", "min ms", "med ms", "GB/s", "Gcodes/s"
        );
        let mut tab = String::new();
        for (ai, (_, label)) in ARMS.iter().enumerate() {
            let lo = min(ai);
            tab.push_str(&format!(
                "  {label:<34} {lo:>9.4} {:>9.4} {:>9.1} {:>10.1}\n",
                median(&samples[ai]),
                gbps_from(k * n, lo),
                gcodes_from(k * n, lo),
            ));
        }
        print!("{tab}");

        let (a, b, c, d, e) = (min(0), min(1), min(2), min(3), min(4));
        println!("\n  --- ratios (all against the replica (b) except (c)/(a)) ---");
        println!("  (b)/(a) replica overhead          {:+.1}%", (b - a) / a * 100.0);
        println!(
            "  (c)/(a) load-free floor vs real   {:+.1}%   <- negative means the loads COST that",
            (c - a) / a * 100.0
        );
        println!(
            "  (d)/(b) 3 variable shifts cost     {:+.1}%",
            (d - b) / b * 100.0
        );
        println!(
            "  (e)/(b) 5 extraction masks cost    {:+.1}%",
            (e - b) / b * 100.0
        );
        println!(
            "  estimator: min over {rounds} interleaved rounds, worst med/min = {worst_ratio:.2}x"
        );

        println!("\n  --- VERDICT ---");
        let gc_prod = gcodes_from(k * n, a);
        println!(
            "  production code rate            {gc_prod:.1} Gcodes/s ({:.1} GB/s) = {:.0}% of the \
             {GCODES_FOR_20TOK:.0} Gcodes/s that 20 tok/s needs",
            gbps_from(k * n, a),
            gc_prod / GCODES_FOR_20TOK * 100.0
        );
        if b < a * 0.98 {
            println!(
                "  !! (b) is >2% FASTER than (a) although bitwise equal: the probe body schedules \
                 differently from production. (c)/(d)/(e) are then measured against (b), not (a)."
            );
        }
        if c <= a * 0.60 {
            println!(
                "  HYPOTHESIS REFUTED: the load-free, decode-live floor is only {:.0}% of the real \
                 kernel, so this loop is MEMORY/LATENCY bound, not instruction-issue bound. The \
                 code-rate ceiling is NOT what limits it and this analysis is wrong.",
                c / a * 100.0
            );
        } else {
            println!(
                "  consistent with instruction-bound: deleting every weight load and every DRAM \
                 miss buys only {:.0}% ({:.0} -> {:.0} Gcodes/s).",
                (a - c) / a * 100.0,
                gc_prod,
                gcodes_from(k * n, c)
            );
        }
        let mask_pct = (b - e) / b * 100.0;
        if mask_pct < 2.0 {
            println!(
                "  NOTE: the five `& 0xffff` masks are FREE (MSL already folds them into one op), \
                 so the real count is ~8.5 instr/code, not 9.4, and the issue ceiling rises to \
                 ~{:.0} GB/s.",
                5.5e12 / 8.5 * 3.0 / 8.0 / 1e9
            );
        } else {
            println!(
                "  NOTE: the five `& 0xffff` masks cost {mask_pct:.1}%, confirming they are real \
                 instructions on this target (5.5e12 issue/s model stands)."
            );
        }
        if (b - d) / b * 100.0 > 20.0 {
            println!(
                "  NOTE: the three VARIABLE shifts still cost >20%. Commit d63b705 removed the \
                 64-bit barrel shift and gained only 6% in production; a 32-bit variable shift is \
                 apparently NOT free either, which is new information."
            );
        }
        println!(
            "  reference: best 3 bpw measured in this tree = {MEASURED_PEAK_GBPS:.0} GB/s = \
             {MEASURED_PEAK_GCODES:.0} Gcodes/s; practical copy ceiling ~250 GB/s"
        );
    }

    // ---------------- (G) the decisive arm: no memory at all ---------------
    run_ceiling_arm(&ctx, &device, rounds);
}

/// Arm (G): register-resident instruction ceiling for the exact op mix.
fn run_ceiling_arm(ctx: &Ctx, device: &Device, rounds: usize) {
    let pipe = pipeline(ctx, "probe_code_ceiling");
    let threads = 81_920u64;
    let groups = (threads / 256).max(1);
    let iters = 20_000u32;

    let seeds: Vec<u32> = (0..1024u32)
        .map(|i| i.wrapping_mul(2654435761).wrapping_add(1))
        .collect();
    let seed_buf = upload(device, &seeds);
    let out_buf = device.new_buffer(threads * 4, MTLResourceOptions::StorageModeShared);

    // `sh0 = 19` / `rsh0 = 13` is a real lane value on this target, so this is
    // the production op mix including the three variable shifts. 32 codes per
    // thread per iteration (4 groups x 8), matching `_colmajor_2col`'s inner
    // loop sequencing of its four tile-decodes.
    let ctrl: [u32; 4] = [iters, 19, 13, 0];
    let dispatch = || {
        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&seed_buf), 0);
        enc.set_buffer(1, Some(&out_buf), 0);
        enc.set_bytes(2, 16, ctrl.as_ptr() as *const std::ffi::c_void);
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    };

    dispatch();
    let out = read_f32(&out_buf, threads as usize);
    assert!(
        out.iter().all(|v| v.is_finite()) && out.iter().any(|v| *v != 0.0),
        "arm (G) produced non-finite or all-zero output -- it was optimised away"
    );

    let mut best = f64::MAX;
    for _ in 0..rounds.max(3) {
        let t = Instant::now();
        dispatch();
        best = best.min(t.elapsed().as_secs_f64());
    }

    let codes = threads as f64 * iters as f64 * 32.0;
    let gc = codes / best / 1e9;
    let bw = gc * 3.0 / 8.0;
    let ms_per_token = GB_PER_TOKEN / (bw / 1e9);

    println!("\n=== ARM (G): register-resident instruction ceiling ===");
    println!("  threads {threads}, iters {iters}, codes {codes:.4e}, best {best:.5} s");
    println!("  code rate       {gc:.1} Gcodes/s");
    println!("  == 3 bpw       {bw:.1} GB/s");
    println!("  20 tok/s needs {GCODES_FOR_20TOK:.1} Gcodes/s");
    if gc < GCODES_FOR_20TOK {
        println!(
            "  => **REFUTED for 20 tok/s.** No memory traffic at all, and the IDENTICAL instruction \
             stream still only reaches {gc:.0} Gcodes/s = {ratio:.0}% of what 20 tok/s requires. \
             The instruction stream is the wall; no scheduling, layout or memory change to these \
             same instructions can close the gap.",
            ratio = gc / GCODES_FOR_20TOK * 100.0
        );
        println!(
            "  => implied ceiling: {bw:.1} GB/s at 3 bpw = {ms_per_token:.1} ms for \
             {GB_PER_TOKEN} GB/token = {tps:.1} tok/s of GEMV time, before attention, norms, \
             cache writes and dispatch.",
            tps = 1000.0 / ms_per_token
        );
    } else {
        println!(
            "  => **the instruction-count claim is WRONG.** The identical instruction stream runs \
             at {gc:.0} Gcodes/s with no memory traffic, so the real kernel's code rate is a \
             MEMORY-side limit and there IS instruction headroom to exploit."
        );
    }
}

// ===========================================================================
// Test 2: code rate vs shape, synthetic colmajor weights
// ===========================================================================

/// Confirms the 203 GB/s / ~541 Gcodes/s figure and shows whether the code rate
/// is flat in shape. Contents are pseudo-random `u16`, not valid codes: this
/// measures streaming bandwidth through the production kernel, not correctness
/// (same convention as `gemv_bandwidth.rs:28-31`).
#[test]
#[ignore = "microbenchmark; run explicitly"]
fn p01_code_rate_shape_sweep() {
    let device = Device::system_default().expect("Metal device");
    let ctx = build_library(&device);
    let rounds = arg_usize("--rounds", 9);
    let iters = arg_usize("--iters", 12);
    let arm = ARMS[0].0;

    println!("\n=== synthetic colmajor sweep, production kernel ===");
    for (k, n) in [
        (5376usize, 5376usize),
        (5376, 8192),
        (21504, 5376),
        (5376, 43008),
        (21504, 21504),
    ] {
        // 24 u32 words == 48 u16 per 16x16 tile at 3 bpw
        // (`RING/WORDS`, linear_repacked.metal:2084-2085).
        let tiles = (k / 16) * (n / 16);
        let mut words = vec![0u16; tiles * 48];
        let mut state = 0x1234_5678u32;
        for w in words.iter_mut() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *w = state as u16;
        }
        let packed = upload(&device, &words);
        drop(words);
        let xb = upload(&device, &sin_x(k));
        let yb = device.new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared);
        let seeds = [0x1111_1111u32, 0x2222_2222u32];

        for _ in 0..2 {
            let cb = ctx.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            dispatch_gemv(&ctx, arm, &packed, &xb, &yb, k, n, seeds, enc);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        let mut best = f64::MAX;
        for _ in 0..rounds {
            let cb = ctx.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            for _ in 0..iters {
                dispatch_gemv(&ctx, arm, &packed, &xb, &yb, k, n, seeds, enc);
            }
            enc.end_encoding();
            let t = Instant::now();
            cb.commit();
            cb.wait_until_completed();
            best = best.min(t.elapsed().as_secs_f64() * 1e3 / iters as f64);
        }
        let gbps = gbps_from(k * n, best);
        let gc = gcodes_from(k * n, best);
        println!(
            "  K={k:<6} N={n:<6} {best:8.3} ms  {gbps:7.1} GB/s  {gc:7.1} Gcodes/s  \
             ({:5.1} tok/s-equivalent GEMV-only)",
            GB_PER_TOKEN / (gbps / 1e9)
        );
    }
}