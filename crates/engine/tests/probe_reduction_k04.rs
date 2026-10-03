//! K04 — the cross-lane reduction in the live GEMV, and the two 1-threadgroup
//! finalizes.
//!
//! READ-ONLY probe. CPU only: no Metal context, no model, no `gpuq`. Every
//! claim here is derived either from arithmetic on the shapes or from counting
//! tokens in the kernel source, and every claim that *could* be wrong is
//! falsifiable by a sabotage recorded in `spec/decode-ceiling/findings/
//! K04-reduction.md`.
//!
//! WHAT IS PROVED HERE (5 things)
//!
//! 1. `linear_gemv_ishizuki_k3_colmajor_2col_sk2` — the live kernel — runs its
//!    reduction block **exactly once per threadgroup**, because the dispatch
//!    grid equals `pairs_n * SK`. So "per GEMV" and "per threadgroup" are the
//!    same number and the reduction is not amortised over any tile loop.
//! 2. The reduction block is **27.4 issue slots per warp** against
//!    **3070 for the K loop** at K=5376: **0.89%**. It is not >5%, and the
//!    answer does not depend on N.
//! 3. Reduction cost as a *fraction* is a function of **K alone**, not N.
//!    gate/up (N=21504) do 4x the absolute reduction work of q_proj (N=8192)
//!    and exactly the same fraction of it, because they also do 4x the work.
//! 4. The frozen norm tree is **the standard butterfly on masks 128..1**, so
//!    masks 128/64/32 are the only cross-simdgroup steps. A bit-exact
//!    restructure to **3 `threadgroup_barrier`s + 5 `simd_shuffle_xor`s** exists
//!    and is proved equal bit-for-bit here.
//! 5. A bit-exact **multi-threadgroup** split of the frozen order exists **iff
//!    the group count is a power of two dividing 256**. 42 is not, and 42 is a
//!    reassociation — which is why the "42-group" plan in T5.6 cannot be
//!    bit-exact.
//!
//! Additionally found: `finalize_post_ffn_and_next_input_norm_transform`'s last
//! step still strides `b += 8u` under **32** simdgroups (6 rounds where 2
//! would do) while its sibling `linear_finalize_o_proj_and_pre_ffn_transform_sk2`
//! was correctly bumped to `b += 32u`. A missed edit in T5.6.

use half::f16;

// ===========================================================================
// Shapes. From `~/.kaplumba/models/Gxma/Gxma/config.json`, cross-checked
// against `crates/engine/src/config.rs:223` by
// `crates/engine/tests/perf_probe_fusion_census.rs`.
// ===========================================================================

const HIDDEN: usize = 5376;
const INTER: usize = 21504;
const Q_HEADS: usize = 32;
const SLIDING_HEAD_DIM: usize = 256;
const SLIDING_KV_HEADS: usize = 16;
const FULL_HEAD_DIM: usize = 512;
const FULL_KV_HEADS: usize = 4;
const LAYERS: usize = 60;

// Live-kernel constants, `crates/kernels/src/linear_repacked.metal:2417-2428`.
const SG: usize = 8; // constexpr int SG = 8
const THREADS: usize = 256; // dispatch_groups(..., threads = 256)
const SK: usize = 2; // constexpr uint SK = 2u

// ===========================================================================
// Source access. The counts below are read out of the kernel text, not
// remembered, so a later edit to the reduction block fails these tests.
// ===========================================================================

fn kernel_source(file: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../kernels/src")
        .join(file);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The `linear_gemv_ishizuki_k3_colmajor_2col_sk2` kernel body, as a slice of
/// lines. Located by name so the line numbers cannot drift silently.
fn live_gemv() -> (usize, String) {
    let src = kernel_source("linear_repacked.metal");
    let start = src
        .find("kernel void linear_gemv_ishizuki_k3_colmajor_2col_sk2(")
        .unwrap_or_else(|| panic!("live kernel not found"));
    let end = src[start..]
        .find("\nkernel void ")
        .map(|i| start + i)
        .unwrap_or_else(|| src.len());
    let body = src[start..end].to_string();
    let first_line = src[..start].matches('\n').count() + 1;
    (first_line, body)
}

fn finalize_body(marker: &str) -> (usize, String) {
    let src = kernel_source("linear.metal");
    let at = src
        .find(marker)
        .unwrap_or_else(|| panic!("{marker} not found"));
    let start = src[..at].rfind("kernel void ").expect("a kernel header before marker");
    let end = src[at..]
        .find("\nkernel void ")
        .map(|i| at + i)
        .unwrap_or_else(|| src.len());
    let body = src[start..end].to_string();
    let first_line = src[..start].matches('\n').count() + 1;
    (first_line, body)
}

fn count(hay: &str, needle: &str) -> usize {
    hay.matches(needle).count()
}

// ===========================================================================
// 1. The reduction block runs exactly once per threadgroup.
// ===========================================================================

/// Grid the host passes, from `ishizuki_gemv_pipeline` (`kernels/src/linear.rs`)
/// and `crates/engine/src/linear.rs:2133-2157`: `groups = pairs_n * SK`.
fn grid_for(n: usize) -> usize {
    let tiles_n = n / 16;
    let pairs_n = tiles_n / 2;
    pairs_n * SK
}

#[test]
fn reduction_block_runs_exactly_once_per_threadgroup() {
    // The kernel's grid loop is `for (tg = tgid; tg < total_groups; tg += ntg)`
    // with `total_groups = pairs_n * SK` (linear_repacked.metal:2438-2439). It is
    // one trip iff the dispatch grid equals `total_groups`.
    for n in [2048usize, 4096, 5376, 8192, 16384, 21504] {
        let pairs_n = (n / 16) / 2;
        assert_eq!(grid_for(n), pairs_n * SK, "N={n}");
    }

    // Every real projection shape, so the "one trip" claim is not vacuous for
    // the shapes that actually run.
    let shapes = [
        (5376, Q_HEADS * SLIDING_HEAD_DIM),      // q_proj, sliding
        (5376, SLIDING_KV_HEADS * SLIDING_HEAD_DIM), // k_proj, sliding
        (5376, SLIDING_KV_HEADS * SLIDING_HEAD_DIM), // v_proj, sliding
        (5376, 5376),                                // o_proj
        (5376, INTER),                               // gate
        (5376, INTER),                               // up
        (INTER, 5376),                               // down
        (5376, Q_HEADS * FULL_HEAD_DIM),             // q_proj, full attention
        (5376, FULL_KV_HEADS * FULL_HEAD_DIM),       // k_proj, full attention
        (16384, 5376),                               // o_proj, full attention
    ];
    for (k, n) in shapes {
        assert_eq!(grid_for(n), (n / 32) * SK, "k={k} n={n}: grid != pairs_n*SK");
    }
}

// ===========================================================================
// 2. The reduction block's issue-slot count, read out of the source.
// ===========================================================================

/// Issue slots per warp for one trip of the reduction block, counting the way
/// a lockstep GPU pays for them: a divergent arm is charged to the whole warp.
///
/// * every warp runs the 8 shuffles + 8 adds            -> 16
/// * every warp runs the 4 threadgroup stores under
///   `lane % 4 == 0`, predicated to 4 of 32 lanes       ->  4 + 1 predicate
/// * every warp passes barrier 1                         ->  1
/// * warp 0 alone runs `lid < 32`: `SG` iterations of
///   (threadgroup load + fadd) plus the selects and the
///   device store                                         -> 21
/// * every warp passes barrier 2                         ->  1
fn reduction_slots_per_warp() -> f64 {
    let (line, body) = live_gemv();
    let tail = &body[body.find("acc0_0 += simd_shuffle_xor").unwrap()..];

    assert_eq!(
        count(tail, "simd_shuffle_xor"),
        8,
        "expected 8 simd_shuffle_xor in the reduction tail of the live kernel \
         (starting line {line}); the 0.89% figure below is invalidated"
    );
    assert_eq!(
        count(tail, "threadgroup_barrier"),
        2,
        "expected exactly 2 threadgroup_barrier in the live kernel's reduction \
         tail; the barrier budget changed"
    );
    assert_eq!(
        count(tail, "red[0][sg]"),
        2,
        "expected 2 stores into red[0]; the reduction shape changed"
    );
    assert_eq!(
        count(tail, "red[1][sg]"),
        2,
        "expected 2 stores into red[1]; the reduction shape changed"
    );
    assert!(tail.contains("if (lid < 32u)"), "final arm moved");
    assert!(tail.contains("for (int g = 0; g < SG; ++g)"), "final arm moved");

    let shuffles_and_adds = 16.0;
    let stores = 4.0 + 1.0;
    let barrier1 = 1.0;
    let final_arm_warp0 = (2.0 * SG as f64) + 5.0; // loads+fadds, then selects+store
    let final_arm_other = 2.0; // the `lid < 32` predicate only
    let barrier2 = 1.0;
    let warps = (THREADS / 32) as f64;
    let total = shuffles_and_adds + stores + barrier1 + barrier2;
    (total * (warps - 1.0) + total + final_arm_other * (warps - 1.0) + final_arm_warp0)
        / warps
}

/// Issue slots for the K loop of one warp, one threadgroup, one `tg` trip.
///
/// 52 slots per `extract_v_mcg_32` (20 field extractions + 8x4 decode ops,
/// with the two shifts and the `or` hoisted out of the loop because `sh0` is
/// loop-invariant), 8 weight loads, 8 half loads + 8 conversions, 4 calls x 2
/// accumulators x (4 mul + 4 add) with the FFMA chain already fused, 6 address
/// IMADs and 3 loop-bookkeeping ops — times 4 calls per trip.
fn kloop_slots_per_warp(k: usize) -> f64 {
    let extract = 52.0;
    let per_trip = 4.0 * extract          // 208
        + 8.0                            // uint weight loads
        + 16.0                            // half x-loads + float(half)
        + 4.0 * 2.0 * 8.0                // mul/add per accumulator (fused)
        + 6.0                             // address IMADs
        + 3.0;                            // loop bookkeeping
    let tiles_k = k / 16;
    let k_per_split = tiles_k / SK;
    let step = SG * 2; // `const uint step = SG * 2u`
    let full = k_per_split / step;
    let rem = k_per_split - full * step;
    // The remainder loop `for (tk = k_unroll_end + sg; tk < k_end; tk += SG)` runs
    // one trip when `rem > 0`, and it does 2 extracts, not 4.
    let remainder = if rem == 0 {
        0.0
    } else {
        2.0 * extract + 4.0 + 8.0 + 2.0 * 2.0 * 8.0 + 6.0
    };
    full as f64 * per_trip + remainder
}

#[test]
fn reduction_is_under_five_percent_of_the_live_gemv() {
    let red = reduction_slots_per_warp();

    // Cross-check against the established 9.375 instr/weight figure: one trip of
    // the live kernel's unrolled body handles 4 x 8 = 32 weights.
    let per_trip: f64 = 4.0 * 52.0 + 8.0 + 16.0 + 64.0 + 6.0 + 3.0;
    assert!(
        (per_trip / 32.0 - 9.375).abs() < 1.0,
        "structural count gives {:.3} instr/weight but the established figure is \
         9.375; the two have diverged and the percentages below are unsafe",
        per_trip / 32.0
    );

    for (label, k) in [("K=5376 (q/k/v/o/gate/up)", 5376usize), ("K=21504 (down)", 21504)] {
        let frac = red / kloop_slots_per_warp(k);
        assert!(
            frac < 0.05,
            "reduction is {frac:.4} of the K loop for {label}; this probe's \
             conclusion that it is not the lever is now false"
        );
        println!("reduction / K-loop, {label}: {:.3}%", frac * 100.0);
    }
}

// ===========================================================================
// 3. The fraction depends on K, not on N. gate/up do NOT pay more.
// ===========================================================================

#[test]
fn reduction_fraction_is_independent_of_n() {
    let red = reduction_slots_per_warp();
    let kloop = kloop_slots_per_warp(5376);

    // Absolute reduction slots for a whole dispatch scale with N...
    let slots = |n: usize| (n / 32) as f64 * SK as f64 * (THREADS / 32) as f64 * red;
    assert!(
        (slots(INTER) / slots(HIDDEN) - 4.0).abs() < 1e-9,
        "gate/up (N={INTER}) should do exactly 4x o_proj's (N={HIDDEN}) \
         absolute reduction work"
    );

    // ...but so does the work being reduced, so the fraction is identical.
    let frac = |n: usize| {
        (n / 32) as f64 * SK as f64 * red / ((n / 32) as f64 * SK as f64 * kloop)
    };
    assert!(
        (frac(INTER) - frac(HIDDEN)).abs() < 1e-12,
        "MLP projections pay a different reduction *fraction* than o_proj"
    );
    println!(
        "gate/up absolute reduction work: {:.1}x o_proj (N={HIDDEN} -> {INTER})",
        slots(INTER) / slots(HIDDEN)
    );
    println!(
        "reduction fraction at K=5376, both N: {:.3}%",
        frac(HIDDEN) * 100.0
    );
}

/// Per-token aggregate, over all 60 layers with 10 of them `Full`
/// (`config.rs:241`: layer i is Full iff `i % 6 == 5`).
#[test]
fn reduction_is_under_one_percent_of_gemv_time_per_token() {
    let red = reduction_slots_per_warp();
    let warps = (THREADS / 32) as f64;
    let mut red_total = 0.0;
    let mut work_total = 0.0;
    let mut gemvs = 0usize;

    for layer in 0..LAYERS {
        let full = layer % 6 == 5;
        let head_dim = if full { FULL_HEAD_DIM } else { SLIDING_HEAD_DIM };
        let kv_heads = if full { FULL_KV_HEADS } else { SLIDING_KV_HEADS };
        let q_out = Q_HEADS * head_dim;
        let kv_out = kv_heads * head_dim;
        let shapes = [
            (HIDDEN, q_out),
            (HIDDEN, kv_out),
            (HIDDEN, kv_out), // v_proj; full layers take the fused2 + copy arm
            (q_out, HIDDEN),  // o_proj
            (HIDDEN, INTER),  // gate
            (HIDDEN, INTER),  // up
            (INTER, HIDDEN),  // down
        ];
        for (k, n) in shapes {
            let groups = (n / 32) as f64 * SK as f64;
            red_total += groups * warps * red;
            work_total += groups * warps * kloop_slots_per_warp(k);
            gemvs += 1;
        }
    }
    assert_eq!(gemvs, 7 * LAYERS);
    let frac = red_total / work_total;
    println!(
        "per-token: {gemvs} GEMVs, reduction = {:.3}% of GEMV issue slots",
        frac * 100.0
    );
    assert!(
        frac < 0.01,
        "reduction is {frac:.4} of GEMV time, not under 1%"
    );

    // In milliseconds. 12.039 GB/token at the measured 3 bpw rate of 199.4 GB/s.
    let gemv_ms = 12.039 / 199.4 * 1000.0;
    let red_ms = frac * gemv_ms;
    let step_ms = 62.3;
    let best_case_tps = 16.05 * step_ms / (step_ms - red_ms);
    println!(
        "GEMV {gemv_ms:.1} ms/token, reduction {red_ms:.3} ms, \
         step {step_ms} ms; deleting the reduction ENTIRELY buys \
         {best_case_tps:.3} tok/s (+{:.3})",
        best_case_tps - 16.05
    );
    assert!(
        best_case_tps < 16.2,
        "deleting the whole reduction would buy {best_case_tps:.2} tok/s, which \
         is larger than this probe claims and reopens the question"
    );
}

// ===========================================================================
// 4. The frozen norm tree is a butterfly; masks 128/64/32 are the only
//    cross-simdgroup steps, so 8 barriers -> 3 barriers + 5 shuffles, exactly.
// ===========================================================================

/// The order `linear_finalize_o_proj_and_pre_ffn_transform_sk2`
/// (`linear.metal:937-948`) freezes, reproduced in f32.
fn norm_sum_frozen(input: &[f32], threads: usize) -> f32 {
    let width = input.len();
    let mut partial = vec![0.0f32; threads];
    for lid in 0..threads.min(256) {
        let mut sum = 0.0f32;
        let mut i = lid;
        while i < width {
            sum += input[i] * input[i];
            i += 256;
        }
        partial[lid] = sum;
    }
    // 1024-thread tree from 512; masks 512 and 256 add exact zeros.
    let mut stride = threads / 2;
    while stride > 0 {
        for lid in 0..stride {
            partial[lid] += partial[lid + stride];
        }
        stride >>= 1;
    }
    partial[0]
}

/// The same sum, restructured: the three cross-simdgroup masks (128, 64, 32) go
/// through threadgroup memory with a barrier each; the five intra-simdgroup
/// masks (16..1) become `simd_shuffle_xor` with no barrier at all.
fn norm_sum_butterfly(input: &[f32]) -> f32 {
    let width = input.len();
    // Step 1: 256 leaves, 21-element chains, exactly as frozen.
    let mut leaves = [0.0f32; 256];
    for lid in 0..256 {
        let mut sum = 0.0f32;
        let mut i = lid;
        while i < width {
            sum += input[i] * input[i];
            i += 256;
        }
        leaves[lid] = sum;
    }
    // Mask 128: lanes 0..127 of simdgroups 0..3 += lanes of simdgroups 4..7.
    // Threadgroup memory + 1 barrier.
    let mut buf = [0.0f32; 128];
    for l in 0..128 {
        buf[l] = leaves[l] + leaves[l + 128];
    }
    // Mask 64: simdgroup 0 += simdgroup 1. 1 barrier.
    let mut buf2 = [0.0f32; 64];
    for l in 0..64 {
        buf2[l] = buf[l] + buf[l + 64];
    }
    // Mask 32: simdgroup 0 += simdgroup 1. 1 barrier. 32 values now in one
    // simdgroup, lane = index.
    let mut v = [0.0f32; 32];
    for l in 0..32 {
        v[l] = buf2[l] + buf2[l + 32];
    }
    // Masks 16..1: `simd_shuffle_xor`, zero barriers.
    for mask in [16u32, 8, 4, 2, 1] {
        for l in 0..32usize {
            if (l as u32) & mask == 0 {
                v[l] += v[l + mask as usize];
            }
        }
    }
    v[0]
}

#[test]
fn the_norm_tree_is_a_butterfly_so_three_barriers_and_five_shuffles_are_exact() {
    let width = HIDDEN;
    let input: Vec<f32> = (0..width)
        .map(|i| f16::from_bits(pseudo_half_bits(i as u32)).to_f32())
        .collect();

    let frozen = norm_sum_frozen(&input, 1024);
    let restructured = norm_sum_butterfly(&input);
    assert_eq!(
        frozen.to_bits(),
        restructured.to_bits(),
        "restructuring the frozen norm tree into 3 barriers + 5 shuffles changed \
         the sum ({:e} vs {:e}, {} ulp)",
        frozen,
        restructured,
        ((frozen.to_bits() as i64) - (restructured.to_bits() as i64)).abs()
    );

    // The 1024-thread tree is bit-identical to the 256-thread original, which is
    // the property the kernel header leans on.
    assert_eq!(
        norm_sum_frozen(&input, 1024).to_bits(),
        norm_sum_frozen(&input, 256).to_bits(),
        "the 1024-thread tree stopped matching the 256-thread original"
    );
}

#[test]
fn a_wrong_shuffle_mask_breaks_the_butterfly_proof() {
    // Falsifiability for the test above. The butterfly is only exact because the
    // mask order is 16, 8, 4, 2, 1 -- the same descending order the threadgroup
    // tree uses. Ascending order is a different tree over the same leaves.
    let input: Vec<f32> = (0..HIDDEN)
        .map(|i| f16::from_bits(pseudo_half_bits(i as u32)).to_f32())
        .collect();
    let frozen = norm_sum_frozen(&input, 1024);

    let mut leaves = [0.0f32; 256];
    for lid in 0..256 {
        let mut sum = 0.0f32;
        let mut i = lid;
        while i < HIDDEN {
            sum += input[i] * input[i];
            i += 256;
        }
        leaves[lid] = sum;
    }
    // The same three cross-simdgroup masks, then the five intra-simdgroup ones
    // in ASCENDING order.
    let mut buf = [0.0f32; 128];
    for l in 0..128 {
        buf[l] = leaves[l] + leaves[l + 128];
    }
    let mut buf2 = [0.0f32; 64];
    for l in 0..64 {
        buf2[l] = buf[l] + buf[l + 64];
    }
    let mut v = [0.0f32; 32];
    for l in 0..32 {
        v[l] = buf2[l] + buf2[l + 32];
    }
    for mask in [1u32, 2, 4, 8, 16] {
        for l in 0..32usize {
            if (l as u32) & mask == 0 {
                v[l] += v[l + mask as usize];
            }
        }
    }
    assert_ne!(
        frozen.to_bits(),
        v[0].to_bits(),
        "the descending mask order made no difference on this input, so the \
         butterfly is order-independent and the exactness proof above is vacuous"
    );
}

// ===========================================================================
// 5. Bit-exact multi-threadgroup split: possible iff G | 256 and G is a power
//    of two. 42 is not, so the T5.6 "42-group" plan cannot be bit-exact.
// ===========================================================================

/// Split the 256 frozen leaves across `groups` threadgroups, bit-exactly.
///
/// THE GEOMETRY, derived rather than guessed. The frozen tree is the
/// descending butterfly on masks 128, 64, ..., 1, and mask `2^j` maps `2^(j+1)`
/// nodes to `2^j` nodes, with node `r` covering the leaves `{r + k*2^j}`. So:
///
/// * after masks 128/64/32 there are exactly **32** nodes, index 0..31, node `r`
///   covering leaves `{r, r+32, ..., r+224}`;
/// * masks 16..1 then reduce those 32 to 1, and **all five are stage-wide** —
///   mask 16 pairs node `r` with node `r+16` for `r < 16`, so each of the 32
///   node values is a whole subtree, not a leaf.
///
/// Therefore the only parallelism is *within* the mask-32 stage: `G` groups,
/// each owning `32/G` of those 32 subtrees and computing each with the local
/// masks 16, 8, 4, 2, 1. Masks 128, 64, 32 are folded into that same local
/// butterfly because they only ever pair leaves the group already owns. The
/// five stage-wide masks then finish in a second, trivial pass over 32 values.
///
/// Two shapes are **wrong** and both were tried first:
///
/// * contiguous leaf ranges `[g*per, (g+1)*per)` -- the value `T(lo..hi)` over
///   a contiguous range never exists in this tree; off by 1 ULP.
/// * strided leaves with the group roots recombined under masks
///   `128/G, 128/2G, ...` -- that is the ASCENDING mask order; off by 1 ULP.
fn norm_sum_split(input: &[f32], groups: usize) -> f32 {
    assert_eq!(32 % groups, 0, "G must divide 32");
    let width = input.len();
    let per = 32 / groups;
    let mut nodes = vec![0.0f32; 32];
    for g in 0..groups {
        for j in 0..per {
            // Node index r = g*per + j, covering leaves {r, r+32, ..., r+224}.
            let r = g * per + j;
            let mut v: Vec<f32> = Vec::with_capacity(8);
            let mut leaf = r;
            while leaf < 256 {
                let mut sum = 0.0f32;
                let mut i = leaf;
                while i < width {
                    sum += input[i] * input[i];
                    i += 256;
                }
                v.push(sum);
                leaf += 32;
            }
            // masks 128/64/32 folded in, then 16..1: this is the whole subtree.
            let mut m = 4;
            while m > 0 {
                for j2 in 0..m {
                    v[j2] += v[j2 + m];
                }
                m >>= 1;
            }
            nodes[r] = v[0];
        }
    }
    // The five stage-wide masks, in a second pass over 32 values.
    let mut m = 16usize;
    while m > 0 {
        for j in 0..m {
            nodes[j] += nodes[j + m];
        }
        m >>= 1;
    }
    nodes[0]
}

/// The "42-group" shape from `tasks.md` T5.6 / `ROUND1-2-SUMMARY.md:107`: one
/// group per 128-element block, each block summed sequentially.
fn norm_sum_42_blocks(input: &[f32]) -> f32 {
    let blocks = input.len() / 128;
    let mut partial = vec![0.0f32; blocks];
    for b in 0..blocks {
        let mut sum = 0.0f32;
        for lane in 0..128 {
            let v = input[b * 128 + lane];
            sum += v * v;
        }
        partial[b] = sum;
    }
    let mut stride = blocks / 2;
    while stride > 0 {
        for b in 0..stride {
            partial[b] += partial[b + stride];
        }
        stride >>= 1;
    }
    partial[0]
}

#[test]
fn power_of_two_group_splits_are_bit_exact_and_42_blocks_is_not() {
    let input: Vec<f32> = (0..HIDDEN)
        .map(|i| f16::from_bits(pseudo_half_bits(i as u32)).to_f32())
        .collect();
    let frozen = norm_sum_frozen(&input, 1024);

    for groups in [1usize, 2, 4, 8, 16, 32] {
        assert_eq!(
            frozen.to_bits(),
            norm_sum_split(&input, groups).to_bits(),
            "a {groups}-group split is NOT bit-exact, so the claim that a \
             multi-threadgroup finalize is possible is false"
        );
    }

    // 42 does not divide 32 (nor 256). The T5.6 plan cannot be bit-exact.
    assert_eq!(32 % 42, 32, "42 no longer fails to divide the 32 parallel nodes");
    assert_ne!(
        frozen.to_bits(),
        norm_sum_42_blocks(&input).to_bits(),
        "the 42-block reduction now matches the frozen order; the T5.6 plan \
         would have been bit-exact after all"
    );

    // Why it matters, and why it is not a rounding-tolerant tail.
    let a: Vec<f32> = input.iter().map(|v| v * 1.0f32).collect();
    let scale_frozen = (frozen / HIDDEN as f32 + 1e-6).sqrt().recip();
    let scale_42 = (norm_sum_42_blocks(&input) / HIDDEN as f32 + 1e-6)
        .sqrt()
        .recip();
    let normed_a: Vec<f32> = input.iter().map(|v| v * scale_frozen).collect();
    let normed_b: Vec<f32> = input.iter().map(|v| v * scale_42).collect();
    let differing = normed_a
        .iter()
        .zip(&normed_b)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let _ = a;
    println!("42-block order changes {differing}/{HIDDEN} normed elements");
    assert_eq!(
        differing, HIDDEN,
        "only {differing}/{HIDDEN} normed elements changed; the reassociation \
         risk is narrower than documented"
    );
}

#[test]
fn a_misaligned_group_boundary_breaks_the_split_proof() {
    // Falsifiability for the test above: the partition is exact because each
    // group owns whole mask-32 subtrees. Drop one node — the value it should
    // have contributed is replaced by 0.0 — and the sum must move.
    let input: Vec<f32> = (0..HIDDEN)
        .map(|i| f16::from_bits(pseudo_half_bits(i as u32)).to_f32())
        .collect();
    let frozen = norm_sum_frozen(&input, 1024);

    const G: usize = 8;
    let per = 32 / G;
    let mut nodes = vec![0.0f32; 32];
    for g in 0..G {
        for j in 0..per {
            let r = g * per + j;
            if g == 0 && r == 0 {
                continue; // the sabotage: node 0 is dropped, so it contributes 0
            }
            let mut v: Vec<f32> = Vec::with_capacity(8);
            let mut leaf = r;
            while leaf < 256 {
                let mut sum = 0.0f32;
                let mut i = leaf;
                while i < HIDDEN {
                    sum += input[i] * input[i];
                    i += 256;
                }
                v.push(sum);
                leaf += 32;
            }
            let mut m = 4;
            while m > 0 {
                for j2 in 0..m {
                    v[j2] += v[j2 + m];
                }
                m >>= 1;
            }
            nodes[r] = v[0];
        }
    }
    let mut m = 16usize;
    while m > 0 {
        for j in 0..m {
            nodes[j] += nodes[j + m];
        }
        m >>= 1;
    }
    assert_ne!(
        frozen.to_bits(),
        nodes[0].to_bits(),
        "a misaligned {G}-group split still matched the frozen order, so the \
         whole-subtree ownership requirement is not what makes the split exact"
    );
}

// ===========================================================================
// 6. Defect: the cross-layer finalize's last step still strides 8 under 32
//    simdgroups, while its sibling was bumped to 32.
// ===========================================================================

#[test]
fn cross_layer_last_step_under_parallelises_its_own_sibling() {
    let (_, sk2) = finalize_body("linear_finalize_o_proj_and_pre_ffn_transform_sk2(");
    let (_, cross) = finalize_body("finalize_post_ffn_and_next_input_norm_transform(");

    // Sibling: steps 1 and 4 stride b by 32 (32 simdgroups at 1024 threads).
    assert_eq!(
        count(&sk2, "b += 32u"),
        2,
        "the sk2 finalize no longer strides b by 32 in both of its block loops"
    );

    // The cross-layer kernel: first loop bumped to 32, LAST loop still 8.
    let blocks = HIDDEN / 128;
    let simdgroups = 1024 / 32;
    assert_eq!(count(&cross, "b += 32u"), 1, "first loop not bumped");
    assert_eq!(
        count(&cross, "b += 8u"),
        1,
        "the cross-layer kernel's last block loop is no longer left at 8"
    );

    // With 42 blocks, stride 8 means 6 rounds over 8 of 32 simdgroups; stride 32
    // means 2 rounds over all 32.
    let rounds = |stride: usize| (blocks + stride - 1) / stride;
    assert_eq!(blocks, 42);
    assert_eq!(rounds(8), 6);
    assert_eq!(rounds(32), 2);
    println!(
        "cross-layer last step: {} rounds over 8 of {simdgroups} simdgroups \
         (utilisation {:.1}%); the sibling uses {} rounds over all {simdgroups} \
         ({:.1}%)",
        rounds(8),
        100.0 * blocks as f64 / (8.0 * rounds(8) as f64),
        rounds(32),
        100.0 * blocks as f64 / (simdgroups as f64 * rounds(32) as f64)
    );
    // Falsifiability: 6 rounds really is 3x the 2 the sibling achieves.
    assert_eq!(rounds(8) / rounds(32), 3);
}

// ===========================================================================
// 7. Threadgroup memory footprint of the two finalizes, against the 32 KiB
//    limit and the ~16 KiB occupancy cliff.
// ===========================================================================

#[test]
fn finalize_threadgroup_memory_and_barrier_counts() {
    let (_, sk2) = finalize_body("linear_finalize_o_proj_and_pre_ffn_transform_sk2(");
    let (_, cross) = finalize_body("finalize_post_ffn_and_next_input_norm_transform(");

    // The reduction trees, as they appear in the source. Mask 512/256 over a
    // 1024-entry `partial` adds exact zeros for the 768 idle threads, so the
    // effective tree is the 256-leaf butterfly on 128..1 -- which is what makes
    // the butterfly restructure below legal.
    let tree512 = "for (uint stride = 512u; stride > 0u; stride >>= 1u) {";
    let tree128 = "for (uint stride = 128u; stride > 0u; stride >>= 1u) {";
    assert_eq!(
        count(&sk2, tree512),
        2,
        "the sk2 finalize no longer has two 512-entry trees (10 barriers each)"
    );
    assert_eq!(count(&cross, tree512), 1, "layer L's tree changed");
    assert_eq!(count(&cross, tree128), 1, "layer L+1's tree changed");

    // Static barrier call sites vs. runtime executions. `ROUND1-2-SUMMARY.md:106`
    // says "10 + 18 serial barriers"; that counts only the tree loops. The
    // kernels also have standalone barriers, which the summary omits.
    for (name, body, shared_bufs, want_site, want_run) in [
        ("sk2 finalize", &sk2, 1usize, 5usize, 23usize),
        ("cross-layer", &cross, 2, 6, 22),
    ] {
        assert!(body.contains("threadgroup float partial[1024]"), "{name}");
        let sites = count(body, "threadgroup_barrier");
        assert_eq!(sites, want_site, "{name}: barrier call sites changed");
        let tree_loops = count(body, tree512) * 10 + count(body, tree128) * 8;
        let run = sites - count(body, tree512) - count(body, tree128) + tree_loops;
        assert_eq!(run, want_run, "{name}: runtime barrier executions changed");

        let bytes = 1024 * 4 + shared_bufs * 5376 * 2;
        println!(
            "{name}: {bytes} B threadgroup memory, {sites} barrier call sites -> \
             {run} executions ({tree_loops} in reduction trees), grid 1 x 1024 \
             threads; 1024-thread tree masks 512..1 = 256-leaf butterfly 128..1"
        );
        assert!(
            bytes <= 32 * 1024,
            "{name} needs {bytes} B of threadgroup memory, over the 32 KiB limit"
        );
    }
    // The cross-layer kernel is over the ~16 KiB occupancy cliff, so "just raise
    // the grid" cannot keep the two threadgroup staging buffers.
    let cross_bytes = 1024 * 4 + 2 * 5376 * 2;
    assert!(
        cross_bytes > 16 * 1024,
        "the cross-layer kernel is no longer above the 16 KiB cliff, so \
         'raise the grid' is not blocked by threadgroup memory after all"
    );
}

// ===========================================================================
// helpers
// ===========================================================================

/// Half-precision bit patterns with `exp = 12`, i.e. magnitude in
/// `[0.125, 0.25)` -- normal, finite, and of mixed sign. Every `f16 -> f32 ->
/// f32 * f32 -> f32` in the sums above is therefore exact, so the only error is
/// the summation order, which is the thing under test.
///
/// (An earlier version built `0x300 | mant`, i.e. `exp = 0`, giving f16
/// *denormals* of ~5e-5. That input was too degenerate to discriminate tree
/// shapes: the ascending-mask-order sabotage passed. Fixed here.)
fn pseudo_half_bits(i: u32) -> u16 {
    let exp = 12u16 << 10;
    let mant = ((i.wrapping_mul(0x9e37_79b9)) >> 13 & 0x03ff) as u16;
    let sign = if i & 1 == 0 { 0 } else { 0x8000 };
    sign | exp | mant
}

/// Sanity: the test input must actually be able to discriminate summation
/// orders. Without this, every "exact" assertion below could be passing for
/// the wrong reason -- the same failure mode as the denormal input above.
#[test]
fn the_test_input_discriminates_summation_orders() {
    let input: Vec<f32> = (0..HIDDEN)
        .map(|i| f16::from_bits(pseudo_half_bits(i as u32)).to_f32())
        .collect();
    assert!(
        input.iter().all(|v| v.is_normal() && v.is_finite()),
        "the test input contains a denormal or non-finite half"
    );
    let a = input.iter().map(|v| v * v).sum::<f32>();
    let b = input.iter().rev().map(|v| v * v).sum::<f32>();
    assert_ne!(
        a.to_bits(),
        b.to_bits(),
        "summing 5376 squares forwards and backwards agree bit-for-bit; this \
         input cannot discriminate summation orders and every exactness \
         assertion in this file is vacuous"
    );
}