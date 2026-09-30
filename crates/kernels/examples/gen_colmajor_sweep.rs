//! Generator for the decode dispatch-geometry sweep kernels.
//!
//! ```text
//! cargo run -p kaplumba-kernels --example gen_colmajor_sweep
//! ```
//!
//! Writes the block between the `BEGIN/END GENERATED: dispatch-geometry sweep`
//! markers in `src/colmajor_sweep.metal` and leaves everything outside them
//! alone. Idempotent by construction: the block is a pure function of the two
//! axis lists, so running it twice in a row leaves `git diff` empty (spec
//! AC6).
//!
//! Why a generator at all, and why in Rust:
//!
//! * `crates/kernels/src/linear_repacked.metal` already holds the hand-written
//!   `_sk2` / `_sk4` colmajor kernels, and they are the *baseline* of this
//!   sweep -- so the sweep cannot live there, and duplicating 12 near-identical
//!   kernels by hand is how the two families would drift.
//! * AGENTS.md §8 forbids Python in the project. `scripts/gen_exl3_gemv.py`
//!   survives as a recorded exception for `linear.metal`; a new generator is
//!   not worth a second exception, and MSL templating makes the kernel bodies
//!   shareable anyway.
//!
//! The axes, and why these two:
//!
//! * `SK` -- K split: how many threadgroups share one output tile-row, each
//!   covering `ceil(tiles_k / SK)` K-tiles. This is the axis TensorFold calls
//!   `(SK, W)`'s first half and it is the axis T7.6.9 gated behind
//!   `KAPLUMBA_USE_SK=1` after measuring it slower on every shape.
//! * `TG` -- threads per threadgroup. Every colmajor kernel in the tree
//!   hardcodes `SG = 8` (256 threads); nothing in the repo has ever dispatched
//!   one at 128 or 512. TensorFold's `plan()` returns a *warp* count
//!   (`linear.py:30-45`), i.e. the same axis at 32-thread granularity, and its
//!   own note says the heuristic is 3-19% off the best cell of the grid on the
//!   3-bit rows of their packs.
//!
//! What this file deliberately does NOT generate: any change to
//! `linear.rs`'s dispatch, or to the existing kernels. Every kernel here has a
//! new name and is unreachable from the engine; `crates/kernels/src/sweep.rs`
//! is its only loader and `crates/engine/examples/sweep_geom.rs` its only
//! caller.

use std::path::{Path, PathBuf};

/// K-split values. `1` is the engine's live geometry.
const SK_VALUES: [u32; 4] = [1, 2, 4, 8];
/// Threads per threadgroup. `256` is the engine's live geometry (8 simdgroups).
const TG_VALUES: [u32; 3] = [128, 256, 512];

const BEGIN: &str = "// BEGIN GENERATED: dispatch-geometry sweep";
const END: &str = "// END GENERATED: dispatch-geometry sweep";

/// The one kernel body, parameterised on the two axes.
///
/// It is a transcription of `linear_gemv_ishizuki_k3_colmajor_contig4` -- the
/// kernel `KAPLUMBA_TRACE_GEMV=1` shows the engine dispatching -- with three
/// changes, each of which is the point of the sweep:
///
/// 1. `SK` threadgroups per output tile-row instead of one.
/// 2. `TG` threads per threadgroup instead of a hardcoded 8 simdgroups.
/// 3. `k_per = ceil(tiles_k / SK)` instead of the existing
///    `((tiles_k / SK) / 32u) * 32u`. The existing rule rounds *down* to a
///    multiple of 32, so the last split inherits everything the rounding threw
///    away: at `SK = 4`, `tiles_k = 336` (every `K = 5376` shape) it gives
///    `64, 64, 64, 144` -- the last threadgroup does 2.25x the work of the
///    other three. With no floor at all it also collapses to 0 whenever
///    `tiles_k < 32 * SK`, which is a real degenerate tail rather than a
///    hypothetical one (see the `min_tiles` report).
///
/// Accumulation order within a split is contig4's, so a `SK = 1, TG = 256`
/// dispatch accumulates exactly what the live kernel accumulates: the same
/// tiles, in the same order, into the same accumulators. It is bit-identical,
/// not merely close, which is what makes the `(1, 256)` cell a usable baseline
/// for the others.
const BODY: &str = r#"// Two MSL constraints shape this signature, both found the hard way by
// compiling it (a `program_source` error, not a Rust one):
//
//   * a function that is not a kernel gets no `using namespace metal;` prelude,
//     so `min`, `simd_shuffle_xor` and `mem_flags` must be reachable from here
//     -- hence the `using` inside the body;
//   * `threadgroup` variables cannot be *declared* in a non-kernel function, so
//     `red` arrives as a pointer and the kernels below own the storage.
template <uint SK, uint TG>
static inline void sweep_colmajor_k3(
    const device ushort *packed,
    const device half *x,
    device float *y,
    constant const uint *shape, // [K, N, tn_groups]
    threadgroup float *red,
    uint lid,
    uint lane,
    uint sg,
    uint tgid) {
    using namespace metal;
    constexpr uint KA = 3u;
    constexpr uint RING = 256u * KA;
    constexpr uint WORDS = RING / 32u;
    constexpr uint SG = TG / 32u;
    const uint K = shape[0];
    const uint N = shape[1];
    const uint tn_groups = shape[2];
    const uint tiles_n = N / 16u;
    const uint tiles_k = K / 16u;
    const uint r0 = (lane % 4u) * 2u;

    const uint lo0 = (lane * 8u + 1u) * 3u + RING - 16u;
    const uint sh0 = lo0 % 32u;
    const uint hi0 = lo0 / 32u;
    const uint j0 = hi0 >= WORDS ? hi0 - WORDS : hi0;
    const uint j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u;

    // tn_groups threadgroups cover the output tile-rows; SK of them cover the
    // K reduction for each row. The caller passes tn_groups rather than having
    // it recomputed from `ntg`, so the kernel cannot and the dispatch cannot
    // disagree about how the grid was cut up.
    const uint tn0 = tgid % tn_groups;
    const uint sk = tgid / tn_groups;

    // Balanced split. `min` on both ends so an over-large `sk` (only possible
    // if the caller asked for more splits than there are tiles) is an empty
    // range contributing zeros rather than an out-of-bounds read.
    const uint k_per = (tiles_k + SK - 1u) / SK;
    const uint k_start = min(sk * k_per, tiles_k);
    const uint k_end = min(k_start + k_per, tiles_k);

    for (uint tn = tn0; tn < tiles_n; tn += tn_groups) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        const device uint *column = (const device uint *)packed + tn * (tiles_k * WORDS);
        const uint stride = WORDS;

        const uint step = SG * 4u;
        const uint k_unroll_end = k_start + ((k_end - k_start) / step) * step;

        uint tk = k_start + sg * 4u;
        for (; tk < k_unroll_end; tk += step) {
            const device uint *t0 = column + (tk + 0u) * stride;
            const device uint *t1 = column + (tk + 1u) * stride;
            const device uint *t2 = column + (tk + 2u) * stride;
            const device uint *t3 = column + (tk + 3u) * stride;

            const ulong w0a = t0[j0], w0b = t0[j1];
            const ulong w1a = t1[j0], w1b = t1[j1];
            const ulong w2a = t2[j0], w2b = t2[j1];
            const ulong w3a = t3[j0], w3b = t3[j1];

            const ulong A0 = ((w0a << 32) | w0b) << sh0;
            const ulong A1 = ((w1a << 32) | w1b) << sh0;
            const ulong A2 = ((w2a << 32) | w2b) << sh0;
            const ulong A3 = ((w3a << 32) | w3b) << sh0;

            const device half *xr0 = x + (tk + 0u) * 16u + r0;
            const float p00 = float(xr0[0]), p01 = float(xr0[1]), p02 = float(xr0[8]), p03 = float(xr0[9]);
            const device half *xr1 = x + (tk + 1u) * 16u + r0;
            const float p10 = float(xr1[0]), p11 = float(xr1[1]), p12 = float(xr1[8]), p13 = float(xr1[9]);
            const device half *xr2 = x + (tk + 2u) * 16u + r0;
            const float p20 = float(xr2[0]), p21 = float(xr2[1]), p22 = float(xr2[8]), p23 = float(xr2[9]);
            const device half *xr3 = x + (tk + 3u) * 16u + r0;
            const float p30 = float(xr3[0]), p31 = float(xr3[1]), p32 = float(xr3[8]), p33 = float(xr3[9]);

            float v0[8], v1[8], v2[8], v3[8];
            #pragma unroll
            for (uint i = 0; i < 8u; ++i) {
                const uint rad = i * KA + 16u;
                v0[i] = decode_state_mcg(uint(A0 >> (64u - rad)) & 0xffffu);
                v1[i] = decode_state_mcg(uint(A1 >> (64u - rad)) & 0xffffu);
                v2[i] = decode_state_mcg(uint(A2 >> (64u - rad)) & 0xffffu);
                v3[i] = decode_state_mcg(uint(A3 >> (64u - rad)) & 0xffffu);
            }

            acc0 += p00 * v0[0] + p01 * v0[1] + p02 * v0[2] + p03 * v0[3];
            acc1 += p00 * v0[4] + p01 * v0[5] + p02 * v0[6] + p03 * v0[7];
            acc0 += p10 * v1[0] + p11 * v1[1] + p12 * v1[2] + p13 * v1[3];
            acc1 += p10 * v1[4] + p11 * v1[5] + p12 * v1[6] + p13 * v1[7];
            acc0 += p20 * v2[0] + p21 * v2[1] + p22 * v2[2] + p23 * v2[3];
            acc1 += p20 * v2[4] + p21 * v2[5] + p22 * v2[6] + p23 * v2[7];
            acc0 += p30 * v3[0] + p31 * v3[1] + p32 * v3[2] + p33 * v3[3];
            acc1 += p30 * v3[4] + p31 * v3[5] + p32 * v3[6] + p33 * v3[7];
        }

        for (tk = k_unroll_end + sg; tk < k_end; tk += SG) {
            const device uint *t0 = column + tk * stride;
            const ulong a0 = t0[j0], a1 = t0[j1];
            const ulong A0 = ((a0 << 32) | a1) << sh0;
            const device half *xr0 = x + tk * 16u + r0;
            const float p0 = float(xr0[0]), p1 = float(xr0[1]), p2 = float(xr0[8]), p3 = float(xr0[9]);
            float v[8];
            #pragma unroll
            for (uint i = 0; i < 8u; ++i) {
                const uint rad = i * KA + 16u;
                const uint s = uint(A0 >> (64u - rad));
                v[i] = decode_state_mcg(s & 0xffffu);
            }
            acc0 += p0 * v[0] + p1 * v[1] + p2 * v[2] + p3 * v[3];
            acc1 += p0 * v[4] + p1 * v[5] + p2 * v[6] + p3 * v[7];
        }

        acc0 += simd_shuffle_xor(acc0, 1u);
        acc0 += simd_shuffle_xor(acc0, 2u);
        acc1 += simd_shuffle_xor(acc1, 1u);
        acc1 += simd_shuffle_xor(acc1, 2u);
        if (lane % 4u == 0u) {
            red[sg * 16u + (lane / 4u)] = acc0;
            red[sg * 16u + (lane / 4u + 8u)] = acc1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < 16u) {
            float sum = 0.0f;
            for (uint g = 0; g < SG; ++g) {
                sum += red[g * 16u + lid];
            }
            // SK == 1 writes `sk * N == 0`, i.e. the live kernel's layout, so
            // the baseline cell needs no separate reduction pass.
            y[(sk * N) + tn * 16u + lid] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

"#;

/// One `SK`-split reduction kernel: `y[i] = sum_{s} partials[s * N + i]`.
///
/// Split-K's whole cost is this pass plus its dispatch, so the sweep times it
/// rather than hiding it. `partials` is `SK * N` floats -- for the widest shape
/// this sweep touches (`down_proj`, N = 5376) at SK = 8 that is 172 KB against
/// 43 MB of weights, so the pass is ~0.4% of the bytes and the dispatch is the
/// part that matters. That is the point: an alternative reading of T7.6.9 is
/// that split-K lost because the *measured* arm charged it for the reduction and
/// the comparison was fair, or unfairly, depending on whether the dispatch was
/// inside the timing window. Here it is inside.
const REDUCE: &str = r#"kernel void gemv_sweep_reduce_sk{SK}(
    const device float *partials [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant const uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < n) {
        float sum = partials[gid];
        #pragma unroll
        for (uint s = 1u; s < {SK}u; ++s) {
            sum += partials[s * n + gid];
        }
        y[gid] = sum;
    }
}

"#;

const WRAPPER: &str = r#"kernel void gemv_sweep_k3_colmajor_sk{SK}_t{TG}(
    const device ushort *packed [[buffer(0)]],
    const device half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant const uint *shape [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
    constexpr uint SG = {TG}u / 32u;
    threadgroup float red[SG][16];
    // `red` is a 2D threadgroup array and does not decay on its own here;
    // `&red[0][0]` is the flat pointer the body indexes.
    sweep_colmajor_k3<{SK}u, {TG}u>(packed, x, y, shape, &red[0][0], lid, lane, sg, tgid);
}

"#;

fn metal_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/colmajor_sweep.metal")
}

/// The generated block: body once, then one thin entry point per cell.
fn generated_block() -> String {
    let mut out = String::new();
    out.push_str(BODY);
    for sk in SK_VALUES {
        for tg in TG_VALUES {
            out.push_str(
                &WRAPPER
                    .replace("{SK}", &sk.to_string())
                    .replace("{TG}", &tg.to_string()),
            );
        }
    }
    for sk in SK_VALUES.into_iter().filter(|sk| *sk > 1) {
        out.push_str(&REDUCE.replace("{SK}", &sk.to_string()));
    }
    out
}

/// Replace the text between the markers, or append the block if absent.
fn splice(source: &str, block: &str) -> String {
    let start = source
        .find(BEGIN)
        .unwrap_or_else(|| panic!("{} has no `{BEGIN}` marker", metal_path().display()));
    let after = start + BEGIN.len();
    let end = source[after..]
        .find(END)
        .unwrap_or_else(|| panic!("{} has no `{END}` marker", metal_path().display()));
    let mut out = String::with_capacity(source.len() + block.len());
    out.push_str(&source[..after]);
    out.push('\n');
    out.push_str(block);
    out.push_str(&source[after + end..]);
    out
}

fn main() {
    let path = metal_path();
    let source = std::fs::read_to_string(&path).expect("read colmajor_sweep.metal");
    let updated = splice(&source, &generated_block());
    if updated == source {
        println!("{} already up to date", path.display());
        return;
    }
    std::fs::write(&path, updated).expect("write colmajor_sweep.metal");
    println!("wrote {}", path.display());
}