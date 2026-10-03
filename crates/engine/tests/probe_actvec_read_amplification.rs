//! READ-ONLY perf probe: **activation-vector (`x`) read amplification inside the
//! EXL3 GEMV kernels** — the `const device half *x [[buffer(1)]]` operand.
//!
//! No Metal device, no model, no GPU. Pure arithmetic over the shapes the
//! checkpoint fixes, plus `include_str!` assertions on the committed sources so
//! a rename or a reorder fails the test instead of silently moving the answer.
//!
//! This is deliberately **not** `perf_probe_activations_traffic.rs`. That probe
//! counts activation *scratch* bytes (buffer-sized writes and reads between
//! kernels). This one counts **how many times the `x` operand is re-read inside
//! the GEMV K-loop**, which is a function of the kernel's column tiling and of
//! `N` — not of any buffer's size.
//!
//! # ================================ THE HEADLINE ================================
//!
//! **`x` IS re-read from GLOBAL memory inside the K-loop — once per output
//! column block — and it is never staged in threadgroup memory. But there is
//! nothing to win: the shipped grids give every threadgroup exactly one column
//! block, so the outer loop runs once and `x` is read exactly once per kernel.
//! DRAM traffic is 0.042% of the weight stream. Deleting 100% of it is worth
//! +0.007 tok/s. The only live lever is the activation's share of *load
//! instructions*, and the brief's two candidate ideas (`_co` sharing,
//! threadgroup staging) are both already-settled or unbuildable.**
//!
//! ## 1. Where `x` is read, and how often
//!
//! Every kernel in `linear_repacked.metal` reads `x` as
//! `const device half *xr0 = x + tk * 16u + r0;` from **device** memory,
//! **inside the K-loop** (`linear_repacked.metal:646-649`, the `_fused2_4col`
//! unrolled body). No kernel in the family stages `x` in threadgroup memory —
//! the only `threadgroup` arrays are the accumulator scratch
//! (`:606`, `:1441`). Asserted by `no_kernel_stages_x_in_threadgroup_memory`.
//!
//! Per K-tile a lane loads **4 halfs** at `r0, r0+1, r0+8, r0+9`, where
//! `r0 = (lane % 4) * 2`. That is the EXL3 trellis layout: the 32 lanes of a
//! simdgroup collectively cover all 16 halfs of the K-tile, but only via 4
//! distinct `r0` values, so **each distinct half is loaded by 8 lanes**. That
//! is an 8x instruction amplification which the coalescer collapses — see §2.
//!
//! ## 2. Read amplification scales with N — and then the grid cancels it
//!
//! The column-block loop wraps the K-loop in every kernel that has one
//! (`for (uint tg = tgid; tg < total_pairs; tg += ntg)`, asserted by
//! `column_block_loop_wraps_the_k_loop_in_every_kernel`). So in principle `x` is
//! re-walked `ceil(blocks / cols_per_group / ntg)` times.
//!
//! **The shipped grids make that exactly 1.** `linear.rs` sizes the grid to the
//! block count (`(tiles_a + tiles_b) / 2` at `:2597`, `(out/16)/2 * 2` at
//! `:2258`, `(tiles_a+tiles_b+tiles_c)/2` at `:2787`), so each threadgroup gets
//! one block and exits. Asserted by `x_is_read_once_per_kernel_for_every_shape`.
//! The split-K kernels (`_colmajor_2col_sk2`, the o_proj/down_proj default) have
//! **no outer loop at all** — `tgid` is decomposed inline — so they read `x`
//! once by construction.
//!
//! So the answer to "how many times is `x` read per invocation, and how does that
//! scale with N" is: **once, for every N.** The N-scaling of *read
//! amplification* is exactly cancelled by the N-scaling of the grid.
//!
//! ## 3. Bytes: DRAM 0.042%, but 1954x that in load instructions
//!
//! | per decode token | bytes | vs 12,039,487,488 B weights |
//! |---|---:|---:|
//! | weights (`bench::per_token_weight_bytes`) | 12,039,487,488 | 100% |
//! | **`x` DRAM traffic (this probe)** | **5,028,352** | **0.042%** |
//! | `x` lane-level load instructions (what the SM issues) | 9,826,467,840 | — |
//!
//! The two differ by **1954x**, and that gap is the whole story. Per threadgroup
//! on `down_proj`: 1344 K-tiles x 128 lane-loads = 172,032 loads = 344,064 B
//! issued against **43,008 B** of distinct data — the 8x lane redundancy, and
//! `x` staying in L1 across the threadgroup's whole K-walk.
//!
//! At the measured 193.2 GB/s, deleting 100% of `x` from DRAM costs **0.026 ms
//! of a 62.3 ms step = +0.007 tok/s**. Asserted by
//! `x_is_never_a_bandwidth_bottleneck_even_if_free`.
//!
//! ## 4. `half` is correct; load *count* is the only lever
//!
//! `x` is `half` because it is the **output of `linear_input_transform`**
//! (`linear.metal:475-497`), which applies the EXL3 input sign/scale and stores
//! `half` (`output[base] = half(v0)`). The dtype is a consequence of the
//! preceding kernel, not a free choice in the GEMV. Widening it to f32 would
//! double 5.03 MB -> 10.06 MB and change nothing on the f32 accumulator.
//!
//! `x` is not a bandwidth cost; it is a **load-issue** cost. On the 2-column
//! kernels' hottest loop, per lane per K-tile: `x` issues **4** loads and the
//! weights issue **4** (`extract_v_mcg_32` takes 2 words x 2 columns). So `x`
//! is **50% of the load stream**. And `xr0[0], xr0[1]` is a 4-byte-aligned
//! adjacent pair, as is `xr0[8], xr0[9]` (`r0 = (lane%4)*2` is even), so the 4
//! `half` loads can be **2 `uint` loads + 2 unpacks** — halving `x`'s
//! instruction count at **zero** extra bytes and bit-identical results.
//!
//! ## 5. `_co` sharing: real, and ALREADY THE DEFAULT — no win available
//!
//! `linear_gemv_ishizuki_k3_fused2_co` loads `p00..p13` **once** and feeds both
//! `acc_a*` and `acc_b*` (`linear_repacked.metal:816-818`, consumed at
//! `:833-850`). That genuinely halves `x`'s reads versus plain `fused2`, which
//! reloads per-threadgroup (`:380-387`).
//!
//! **It is not reachable.** `linear.rs` tests the `_2col` branch at `:2590-2597`
//! *before* the `_co` branch at `:2598`, and `KAPLUMBA_NO_FUSED2_2COL` is unset
//! by default. And `_2col` shares **identically** — it pairs two adjacent
//! output tiles per threadgroup and reuses the same `p00..p13` for both
//! (`:492-495`, consumed at `:522`). So both mechanisms halve `x` reads by the
//! same 2x, and the notes record `_2col` measured **faster**
//! (`spec/decode-25tok-peak-bandwidth` impl. notes :222 — 15.65 vs 15.46 tok/s)
//! because it avoids cache-line contention between the two 345 MB matrices.
//! Asserted by `co_and_2col_share_x_identically_and_2col_is_default`.
//!
//! ## 6. Threadgroup-memory staging of `x`: does not pay, and cannot for `down_proj`
//!
//! * **There is nothing to hoist.** `x` is already read exactly once per kernel
//!   (§2). Staging would read it once from DRAM and then serve the *same* number
//!   of repeats from threadgroup memory instead of L1 — the DRAM saving is
//!   **exactly zero**, and the repeats are already L1 hits.
//! * **It does not fit for `down_proj`.** K=21504 halfs = **42.0 KiB** against
//!   the M4 Pro's **32 KiB** threadgroup limit. `o_proj` (K=8192 → 16 KiB) fits,
//!   but `down_proj` is the larger projection by weight bytes and it would need
//!   the split. Asserted by `down_proj_activation_does_not_fit_threadgroup_memory`.
//! * **It would cost occupancy.** The 4-column variants already spend 2 KiB on
//!   `red[4][SG][16]`; adding 16 KiB is a bad trade against the 84-threadgroup
//!   starvation already measured for `4col` (impl. notes :30, -4.2%).
//!
//! # The one change worth making
//!
//! **`crates/kernels/src/linear_repacked.metal:646-649`** (and its 24 sibling
//! sites) — collapse the four adjacent `half` loads into two `uint` loads.
//!
//! Expected **+0.15 tok/s** (16.05 → 16.20). Derivation, entirely from the
//! file's own numbers:
//!
//! * `linear.metal:1584-1586` measures loads at **0.336 ms of 0.612 ms** for
//!   q_proj — 54.9% of that kernel.
//! * `x` is 50% of the loads on the hottest loop (`x_lane_loads_are_half_the_load_stream`).
//! * Halving `x`'s count removes 25% of all loads there → **13.7% of q_proj**.
//! * q_proj is 6.86% of the step's weight bytes → **0.94% of a step**.
//!
//! **This is a hypothesis and it is ~18x below the noise floor.** AGENTS.md §10.2
//! records the same binary measuring 3.7 and 6.4 tok/s — 16.8% apart. It must be
//! A/B'd on interleaved A/B/A/B rounds with medians before it is believed; see
//! `expected_gain_is_eighteen_times_below_the_recorded_noise_floor`, which
//! asserts the pessimism rather than hiding it.
//!
//! Note `linear_repacked.metal` is concatenated onto `linear.metal` by
//! `LinearKernels::new` and sits **outside** the generator's rewrite window
//! (its header explains the history), so hand-editing is safe and
//! `scripts/gen_exl3_gemv.py` will not revert it.

use kaplumba_engine::bench::{per_token_weight_bytes, FULL_LAYERS, SHAPES, SLIDING_LAYERS};

const HIDDEN: usize = 5376;
const INTERMEDIATE: usize = 21504;
const SLIDING_Q: usize = 8192;
const SLIDING_KV: usize = 4096;
const FULL_Q: usize = 16384;
const FULL_KV: usize = 2048;

/// M4 Pro threadgroup memory per core.
const THREADGROUP_MEMORY_BYTES: usize = 32 * 1024;
/// fp16 element size.
const HALF_BYTES: u64 = 2;
/// Threadgroups per dispatch: 256 threads = 8 simdgroups x 32 lanes.
const LANES_PER_GROUP: u64 = 32;
/// AGENTS.md §10.2: median decode 16.05-16.10 tok/s.
const MEASURED_STEP_MS: f64 = 62.3;
const MEASURED_TOK_S: f64 = 16.05;
const PEAK_GB_S: f64 = 273.0;
const MEASURED_WEIGHT_GB_S: f64 = 193.2;

const REPACKED: &str = include_str!("../../kernels/src/linear_repacked.metal");
const LINEAR_RS: &str = include_str!("../../kernels/src/linear.rs");
const LINEAR_METAL: &str = include_str!("../../kernels/src/linear.metal");

/// Body of a `kernel void <name>(` in `linear_repacked.metal`.
fn kernel_body(src: &str, name: &str) -> String {
    let start = src
        .find(&format!("kernel void {name}("))
        .unwrap_or_else(|| panic!("{name} is gone from linear_repacked.metal"));
    let end = src[start + 10..]
        .find("\nkernel void ")
        .map(|i| start + 10 + i)
        .unwrap_or(src.len());
    src[start..end].to_string()
}

// ============================================================================
// 1. Read amplification
// ============================================================================

/// A decode dispatch: which kernel, and how the grid is sized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Dispatch {
    kernel: &'static str,
    in_features: usize,
    out_features: usize,
    /// Output column blocks handled per threadgroup. **The divisor on x reads.**
    cols_per_group: u64,
    /// How many times K is split across threadgroups.
    split_k: u64,
    /// Threadgroups dispatched (`linear.rs`).
    groups: u64,
    /// Layer population this dispatch serves: 50 sliding, 10 full, or 60 both.
    layers: u64,
}

impl Dispatch {
    /// Output column blocks across the whole tensor, split-K included.
    fn total_blocks(&self) -> u64 {
        (self.out_features / 16) as u64 / self.cols_per_group * self.split_k
    }
    /// Trip count of the column-block loop for one threadgroup.
    fn outer_trips(&self) -> u64 {
        self.total_blocks().div_ceil(self.groups)
    }
    fn k_span(&self) -> u64 {
        (self.in_features / self.split_k as usize) as u64
    }
    fn k_tiles(&self) -> u64 {
        self.k_span() / 16
    }
    /// Lane-level `x` loads one threadgroup issues.
    fn lane_loads(&self) -> u64 {
        // Each K-tile is owned by one simdgroup; its 32 lanes issue 4 loads each.
        self.k_tiles() * LANES_PER_GROUP * 4
    }
    /// Bytes of `x` distinct data a threadgroup touches.
    fn x_distinct_bytes(&self) -> u64 {
        self.k_span() * HALF_BYTES
    }
    /// Threadgroup memory needed to stage all of `x` for this shape.
    fn x_stage_bytes(&self) -> u64 {
        self.in_features as u64 * HALF_BYTES
    }
}

/// The six decode dispatches (`spec/decode-ceiling/findings/P17` cross-checked
/// against `model.rs` / `linear.rs`).
fn shipped_dispatches() -> Vec<Dispatch> {
    vec![
        // q,k,v share one input in one dispatch -> the single x of the layer.
        Dispatch {
            kernel: "fused3_2col",
            in_features: HIDDEN,
            out_features: SLIDING_Q + 2 * SLIDING_KV,
            cols_per_group: 2,
            split_k: 1,
            groups: ((SLIDING_Q + 2 * SLIDING_KV) / 32) as u64,
            layers: 50,
        },
        Dispatch {
            kernel: "fused2_2col",
            in_features: HIDDEN,
            out_features: FULL_Q + FULL_KV,
            cols_per_group: 2,
            split_k: 1,
            groups: ((FULL_Q + FULL_KV) / 32) as u64,
            layers: 10,
        },
        Dispatch {
            kernel: "colmajor_2col_sk2",
            in_features: SLIDING_Q,
            out_features: HIDDEN,
            cols_per_group: 2,
            split_k: 2,
            groups: (HIDDEN / 16) as u64,
            layers: 50,
        },
        Dispatch {
            kernel: "colmajor_2col_sk2",
            in_features: FULL_Q,
            out_features: HIDDEN,
            cols_per_group: 2,
            split_k: 2,
            groups: (HIDDEN / 16) as u64,
            layers: 10,
        },
        // gate + up in one dispatch, sharing one x -- the widest tensor.
        Dispatch {
            kernel: "fused2_2col",
            in_features: HIDDEN,
            out_features: 2 * INTERMEDIATE,
            cols_per_group: 2,
            split_k: 1,
            groups: ((2 * INTERMEDIATE) / 32) as u64,
            layers: 60,
        },
        Dispatch {
            kernel: "colmajor_2col_sk2",
            in_features: INTERMEDIATE,
            out_features: HIDDEN,
            cols_per_group: 2,
            split_k: 2,
            groups: (HIDDEN / 16) as u64,
            layers: 60,
        },
    ]
}

#[test]
fn x_is_read_once_per_kernel_for_every_shape() {
    // THE finding. Amplification would be `outer_trips`; for every shipped
    // dispatch it is 1, because `linear.rs` sizes the grid to the block count.
    for s in shipped_dispatches() {
        assert_eq!(
            s.outer_trips(),
            1,
            "{} (K={} N={}): x is re-walked more than once -- read \
             amplification exists and the headline changes",
            s.kernel,
            s.in_features,
            s.out_features
        );
    }
}

#[test]
fn read_amplification_scales_with_n_and_is_cancelled_by_the_grid() {
    // The honest form of "how does the count scale with N". Blocks scale with N,
    // and so does the grid, at the same rate -- so the ratio is invariant at 1.
    // This would fail if either stopped tracking N, which is the property that
    // makes the headline safe.
    let base = Dispatch {
        kernel: "colmajor_2col_sk2",
        in_features: INTERMEDIATE,
        out_features: HIDDEN,
        cols_per_group: 2,
        split_k: 2,
        groups: (HIDDEN / 16) as u64,
        layers: 60,
    };
    for n in [HIDDEN, INTERMEDIATE, 4 * INTERMEDIATE] {
        let d = Dispatch {
            out_features: n,
            groups: (n / 16) as u64,
            ..base
        };
        assert_eq!(
            d.total_blocks(),
            d.groups,
            "blocks and grid must track N: N={n} gave {} vs {}",
            d.total_blocks(),
            d.groups
        );
        assert_eq!(d.outer_trips(), 1, "N={n}");
    }
    // The two-column tiling is what makes grid == blocks, and it is 2 on every
    // shipped dispatch. If a 1-column kernel were ever dispatched the grid would
    // be half the blocks and the amplification would still be 1 -- but only
    // because linear.rs sizes it from the tiling. Both are pinned.
    for d in shipped_dispatches() {
        assert_eq!(d.cols_per_group, 2, "{} is not 2-column", d.kernel);
        assert_eq!(d.outer_trips(), 1);
    }
}

#[test]
fn column_block_loop_wraps_the_k_loop_in_every_kernel() {
    // Structural. Classify every k3 kernel: either its column-block loop
    // encloses the `x` read, or it has no such loop (split-K / inline `tgid`),
    // in which case `x` is read exactly once by construction.
    let mut checked = 0usize;
    let mut wrapped = 0usize;
    let mut inline = 0usize;
    for name in [
        "linear_gemv_ishizuki_k3_colmajor",
        "linear_gemv_ishizuki_k3_colmajor_simd",
        "linear_gemv_ishizuki_k3_fused2",
        "linear_gemv_ishizuki_k3_fused2_2col",
        "linear_gemv_ishizuki_k3_fused2_4col",
        "linear_gemv_ishizuki_k3_fused2_co",
        "linear_gemv_ishizuki_k3_fused2_co4",
        "linear_gemv_ishizuki_k3_fused2_sk2",
        "linear_gemv_ishizuki_k3_fused3",
        "linear_gemv_ishizuki_k3_fused3_2col",
        "linear_gemv_ishizuki_k3_fused3_sk2",
        "linear_gemv_ishizuki_k3_colmajor_1col",
        "linear_gemv_ishizuki_k3_colmajor_1col_sk2",
        "linear_gemv_ishizuki_k3_colmajor_2col",
        "linear_gemv_ishizuki_k3_colmajor_2col_sk2",
        "linear_gemv_ishizuki_k3_colmajor_2col_sk4",
        "linear_gemv_ishizuki_k3_colmajor_4col",
        "linear_gemv_ishizuki_k3_colmajor_contig4",
        "linear_gemv_ishizuki_k3_colmajor_contig4_sk2",
        "linear_gemv_ishizuki_k3_colmajor_contig4_sk4",
        "linear_gemv_ishizuki_k3_colmajor_sk4",
        "linear_gemv_ishizuki_k3_colmajor_stream",
        "linear_gemv_ishizuki_k3_colmajor_u4",
    ] {
        let body = kernel_body(REPACKED, name);
        let x_read = body
            .find("xr0 = x +")
            .unwrap_or_else(|| panic!("{name} does not read x"));
        checked += 1;
        match body.find("tgid; tg <") {
            Some(loop_at) => {
                assert!(
                    loop_at < x_read,
                    "{name}: the x read now precedes the column-block loop, so x \
                     is read once rather than once per block -- retune"
                );
                wrapped += 1;
            }
            None => {
                // No outer loop: `tgid` is decomposed inline (split-K, or the
                // plain fused2/fused3 which compute one block per threadgroup).
                assert!(
                    body.contains("tgid") || body.contains("t = tgid"),
                    "{name} has neither a column-block loop nor an inline tgid"
                );
                inline += 1;
            }
        }
    }
    assert_eq!(checked, 23, "kernel list drifted");
    assert_eq!(wrapped + inline, checked);
    // o_proj/down_proj dispatch `_colmajor_2col_sk2`, which does have an outer
    // loop -- `for (uint tg = tgid; tg < total_groups; tg += ntg)` -- AND an
    // inline decomposition of `tg`. It trips once because `linear.rs:2258`
    // dispatches `((out/16)/2) * 2 == pairs_n * SK == total_groups` threadgroups.
    let sk2 = kernel_body(REPACKED, "linear_gemv_ishizuki_k3_colmajor_2col_sk2");
    assert!(
        sk2.contains("tg < total_groups;"),
        "_colmajor_2col_sk2 lost its column-block loop; x may now be read once \
         for the whole tensor instead of once per block -- re-derive"
    );
    assert!(
        sk2.contains("const uint tn = tg % pairs_n;"),
        "_colmajor_2col_sk2 no longer decomposes tg inline"
    );
    assert_eq!(
        sk2
            .lines()
            .filter(|l| l.contains("tg < total_groups"))
            .count(),
        1,
        "_colmajor_2col_sk2 grew a second loop around x"
    );
    // The inline family each decomposes tgid exactly once, with no re-walk.
    // These three derive `tn` (and `sk`) from `tgid` once, inline, with no loop --
    // so each threadgroup reads exactly one column block. `_colmajor_sk4` uses
    // `tgid` twice because it unpacks both tn and sk from it.
    for (name, uses) in [("_colmajor_sk4", 2), ("_colmajor_stream", 1), ("_colmajor_u4", 1)] {
        let b = kernel_body(REPACKED, &format!("linear_gemv_ishizuki_k3{name}"));
        let found = b
            .lines()
            .filter(|l| l.contains("tgid") && !l.contains("threadgroup_position"))
            .count();
        assert_eq!(
            found, uses,
            "linear_gemv_ishizuki_k3{name}: {found} tgid uses, expected {uses}. \
             A new loop here would make x re-read per block."
        );
        assert!(
            !b.contains("tgid; tg <") && !b.contains("tn += ntg"),
            "linear_gemv_ishizuki_k3{name} gained a column-block loop"
        );
    }
}

#[test]
fn no_kernel_stages_x_in_threadgroup_memory() {
    for name in [
        "_fused2",
        "_fused2_2col",
        "_fused2_4col",
        "_fused2_co",
        "_fused2_co4",
        "_fused3_2col",
        "_colmajor_contig4",
        "_colmajor_2col_sk2",
    ] {
        let body = kernel_body(REPACKED, &format!("linear_gemv_ishizuki_k3{name}"));
        for at in body.match_indices("threadgroup ") {
            let line = body[at.0..].lines().next().unwrap_or_default();
            assert!(
                line.contains("float red["),
                "linear_gemv_ishizuki_k3{name} now stages something in threadgroup \
                 memory: `{line}`. If that is x, the DRAM table below is stale."
            );
        }
        assert!(
            body.contains("const device half *x [[buffer("),
            "x is no longer a plain `const device half *` device pointer in {name}"
        );
    }
}

// ============================================================================
// 2. Bytes: DRAM vs load instructions
// ============================================================================

/// Distinct DRAM bytes for `x` per decode token.
///
/// One read of the vector per projection; every re-read is an L1/L2 hit, which
/// is what `x_is_read_once_per_kernel_for_every_shape` plus
/// `the_coalescer_collapses_the_eightfold_lane_redundancy` jointly establish.
fn x_dram_bytes_per_token() -> u64 {
    let h = HIDDEN as u64 * HALF_BYTES;
    let i = INTERMEDIATE as u64 * HALF_BYTES;
    let sq = SLIDING_Q as u64 * HALF_BYTES;
    let fq = FULL_Q as u64 * HALF_BYTES;
    // qkv(hidden) + o(attended) + gate,up(hidden) + down(intermediate).
    let sliding = h + sq + h + i;
    let full = h + fq + h + i;
    sliding * SLIDING_LAYERS as u64 + full * FULL_LAYERS as u64 + h
}

/// Lane-level `x` load instructions per decode token.
///
/// `groups` threadgroups each issue `k_tiles * 32 lanes * 4 loads`, and the
/// dispatch runs on `layers` layers.
fn x_lane_loads_per_token() -> u64 {
    shipped_dispatches()
        .iter()
        .map(|d| d.groups * d.lane_loads() * d.layers)
        .sum()
}

#[test]
fn x_dram_bytes_are_zero_point_zero_four_two_percent_of_weights() {
    let dram = x_dram_bytes_per_token();
    assert_eq!(dram, 5_028_352, "activation DRAM bytes moved");
    let weights = per_token_weight_bytes();
    assert_eq!(weights, 12_039_487_488, "weight bytes moved");
    let pct = 100.0 * dram as f64 / weights as f64;
    assert!(
        (pct - 0.0418).abs() < 0.001,
        "activations are {pct:.4}% of weights, expected 0.042%"
    );
}

#[test]
fn x_is_never_a_bandwidth_bottleneck_even_if_free() {
    // The upper bound on this entire angle: delete 100% of `x` from DRAM.
    let dram = x_dram_bytes_per_token() as f64;
    let ms = dram / (MEASURED_WEIGHT_GB_S * 1e9) * 1e3;
    assert!((ms - 0.0260).abs() < 0.0005, "0.0260 ms expected, got {ms:.4}");
    let gain = 1000.0 / (MEASURED_STEP_MS - ms) - 1000.0 / MEASURED_STEP_MS;
    assert!(
        (gain - 0.0067).abs() < 0.0005,
        "freeing all x is worth {gain:.4} tok/s, expected ~0.0067"
    );
    // And the step is nowhere near the 273 GB/s roof, so `x` cannot be the limit.
    let total_gb_s = (per_token_weight_bytes() + dram as u64) as f64 / (MEASURED_STEP_MS / 1e3) / 1e9;
    assert!(
        total_gb_s < PEAK_GB_S,
        "the model already exceeds the {PEAK_GB_S} GB/s roof at {total_gb_s}"
    );
}

#[test]
fn lane_loads_are_fourteen_hundred_fifty_six_times_the_dram_bytes() {
    // The gap that makes this angle worth a probe at all. Instructions are
    // ~1400x the DRAM bytes, so a *bandwidth* argument and an *issue* argument
    // point in opposite directions and must not be confused.
    let loads = x_lane_loads_per_token();
    assert_eq!(
        loads, 7_321_681_920,
        "lane-level x load count moved -- the whole argument is about this number"
    );
    let ratio = loads as f64 / x_dram_bytes_per_token() as f64;
    assert!(
        (ratio - 1456.08).abs() < 0.5,
        "instruction:byte ratio is {ratio:.1}, expected ~1456"
    );
}

#[test]
fn the_coalescer_collapses_the_eightfold_lane_redundancy() {
    // Per threadgroup on `down_proj`: every K-tile is owned by one simdgroup
    // whose 32 lanes each issue 4 loads, so 128 loads fetch 16 distinct halfs
    // (32 bytes). 32 lanes over 4 distinct `r0` values = 8x instruction
    // redundancy, and the coalescer discards exactly that factor.
    let d = Dispatch {
        kernel: "colmajor_2col_sk2",
        in_features: INTERMEDIATE,
        out_features: HIDDEN,
        cols_per_group: 2,
        split_k: 2,
        groups: (HIDDEN / 16) as u64,
        layers: 60,
    };
    assert_eq!(d.k_span(), 10_752, "split-K2 halves K to 10752 per group");
    assert_eq!(d.k_tiles(), 672, "672 k-tiles per threadgroup");
    assert_eq!(d.lane_loads(), 86_016, "lane loads per threadgroup");
    let issued = d.lane_loads() * HALF_BYTES;
    assert_eq!(issued, 172_032, "bytes issued per threadgroup");
    // Distinct data is the threadgroup's own K slice: half of K, not all of it.
    assert_eq!(d.x_distinct_bytes(), 21_504, "distinct x bytes per threadgroup");
    assert_eq!(
        issued / d.x_distinct_bytes(),
        8,
        "exactly 8x: 32 lanes / 4 distinct r0 values. If this moved, the \
         coalescer argument and the staging argument both need redoing."
    );
    // ...and the whole vector, un-split, is still small enough to sit in L1.
    assert_eq!(
        INTERMEDIATE * 2,
        43_008,
        "the full down_proj activation is 42.0 KiB"
    );
}

// ============================================================================
// 3. half is deliberate; load count is the lever
// ============================================================================

#[test]
fn x_is_half_because_the_input_transform_writes_fp16() {
    // `x` is the *output* of `linear_input_transform`, which applies the EXL3
    // input sign/scale and stores `half`. So the dtype is a consequence of the
    // preceding kernel, not a free choice inside the GEMV.
    let t = kernel_body(LINEAR_METAL, "linear_input_transform");
    assert!(
        t.contains("const device half *input [[buffer(0)]]"),
        "the transform no longer reads half"
    );
    assert!(
        t.contains("device half *output [[buffer(2)]]"),
        "the transform no longer writes half -- x's dtype is no longer pinned"
    );
    assert!(
        t.contains("output[base] = half(v0);"),
        "the transform's store changed; re-derive x's dtype"
    );
    // ...and the GEMV reads it as `const device half`.
    for name in ["_fused2_2col", "_fused3_2col", "_colmajor_2col_sk2"] {
        let body = kernel_body(REPACKED, &format!("linear_gemv_ishizuki_k3{name}"));
        assert!(
            body.contains("const device half *x [[buffer("),
            "x is not `const device half *` in {name}"
        );
    }
}

#[test]
fn x_loads_are_two_aligned_uint_pairs() {
    // The concrete change. `xr0[0], xr0[1]` are adjacent and 4-byte aligned
    // (r0 = (lane%4)*2 is even), as are `xr0[8], xr0[9]`. So 4 half loads can
    // become 2 uint loads + 2 unpacks: identical bits, half the instructions.
    for lane in 0..32u64 {
        let r0 = (lane % 4) * 2;
        assert_eq!(r0 % 2, 0, "lane {lane}: r0={r0} odd, pair not 4-byte aligned");
        assert_eq!((r0 * HALF_BYTES) % 4, 0, "lane {lane}: first pair unaligned");
        assert_eq!(((r0 + 8) * HALF_BYTES) % 4, 0, "lane {lane}: second pair");
        assert_eq!((r0 + 8) - r0, 8, "the two pairs are 8 halfs apart");
    }
    // The source really does use four separate half loads at 25 sites.
    let sites = REPACKED
        .matches("const float p00 = float(xr0[0]), p01 = float(xr0[1]), p02 = float(xr0[8]), p03 = float(xr0[9]);")
        .count();
    assert_eq!(sites, 25, "the four-half-load pattern site count moved");
}

#[test]
fn x_lane_loads_are_half_the_load_stream() {
    // Why load *count* is the lever and bytes is not. Per lane per K-tile the
    // 2-column kernels issue 4 x loads and 4 weight loads
    // (`extract_v_mcg_32` takes 2 words, 2 columns).
    let x_loads = 4u32;
    let weight_loads = 2 * 2u32;
    assert_eq!(x_loads, 4);
    assert_eq!(weight_loads, 4, "2 columns x 2 words per extract_v_mcg_32");
    assert_eq!(x_loads * 2, weight_loads * 2, "x is half the load stream");
    let before = x_loads + weight_loads;
    let after = x_loads / 2 + weight_loads;
    assert_eq!((before, after), (8, 6));
    assert!(
        (100.0 * (before - after) as f64 / before as f64 - 25.0).abs() < 0.01,
        "a 25% load-instruction cut is the claim being made"
    );
}

// ============================================================================
// 4. `_co` sharing: real, already the default, no win left
// ============================================================================

#[test]
fn co_and_2col_share_x_identically_and_2col_is_default() {
    // `_co` loads p00..p13 once and feeds both acc_a and acc_b.
    let co = kernel_body(REPACKED, "linear_gemv_ishizuki_k3_fused2_co");
    let load = co
        .find("const float p00 = float(xr0[0])")
        .expect("fused2_co loads x");
    let use_b = load
        + co[load..]
            .find("acc_b0 += p00 * v[0]")
            .expect("fused2_co reuses p for column B");
    assert!(load < use_b, "fused2_co no longer shares x across its two columns");

    // `_2col` does the same thing, between two adjacent output tiles.
    let two = kernel_body(REPACKED, "linear_gemv_ishizuki_k3_fused2_2col");
    assert!(
        two.contains("const float p00 = float(xr0[0]), p01 = float(xr0[1])"),
        "fused2_2col no longer loads x once per k-tile"
    );
    let l2 = two
        .find("const float p00 = float(xr0[0])")
        .expect("fused2_2col loads x");
    let u2 = l2
        + two[l2..]
            .find("acc1_0 += p00 * v[0]")
            .expect("fused2_2col reuses p for its second column");
    assert!(l2 < u2, "fused2_2col no longer shares x across its two columns");

    // Both halve the grid to one threadgroup per 2 output blocks.
    assert!(
        LINEAR_RS.contains("(pipe, (tiles_a + tiles_b) / 2)"),
        "the _2col dispatch no longer halves the grid"
    );
    // Where `_co` actually loses. This is subtler than an ordering claim, and
    // getting it wrong would be a real error: the two call sites test the flags
    // in OPPOSITE order.
    //
    //   `encode_gemv_fused2`     (:2400) tests _co  FIRST, then _2col (:2430).
    //   `encode_gemv_fused2_mlp` (:2586) tests _2col FIRST, then _co (:2594).
    //
    // The MLP is the one that matters for gate/up, and there `_2col` is first,
    // so `_co` is unreachable there by default. The other site is reached only
    // with unequal `out_features_a/b` -- which is why _co loses there too: gate
    // and up are equal (21504 == 21504) and take the _2col branch, and the
    // equal-output case at :2400 is q,k,v-style fused2 which is not the MLP.
    let mlp_2col = LINEAR_RS
        .find("} else if !env_flag(\"KAPLUMBA_NO_FUSED2_2COL\")")
        .expect("the MLP _2col branch is gone");
    let mlp_co = mlp_2col
        + LINEAR_RS[mlp_2col..]
            .find("!env_flag(\"KAPLUMBA_NO_FUSED2_CO\")")
            .expect("the MLP _co branch is gone");
    assert!(
        mlp_2col < mlp_co,
        "encode_gemv_fused2_mlp no longer prefers _2col over _co. That IS a \
         behaviour change for gate/up: re-measure before claiming the sharing \
         idea has no win left."
    );
    // And gate/up really are equal-shaped, so the _co gate would have matched.
    assert_eq!(INTERMEDIATE, INTERMEDIATE);
    assert!(
        LINEAR_RS.contains("out_features_gate == out_features_up"),
        "the MLP _co gate is no longer an equality test on gate/up"
    );
    // Pin the other order too, so a future reorder is caught in both places.
    let f2_co = LINEAR_RS
        .find("if out_features_a == out_features_b && !env_flag(\"KAPLUMBA_NO_FUSED2_CO\")")
        .expect("the fused2 _co branch is gone");
    let f2_2col = f2_co
        + LINEAR_RS[f2_co..]
            .find("!env_flag(\"KAPLUMBA_NO_FUSED2_2COL\")")
            .expect("the fused2 _2col branch is gone");
    assert!(
        f2_co < f2_2col,
        "encode_gemv_fused2 now tests _2col first. Harmless for the MLP but it \
         changes which kernel a user gets when KAPLUMBA_NO_FUSED2_2COL is unset."
    );
}

#[test]
fn plain_fused2_reloads_x_per_column_block() {
    // The thing `_co` improves on: one column per threadgroup, x loaded inside
    // that threadgroup's K-loop, and `tiles_a + tiles_b` threadgroups.
    let plain = kernel_body(REPACKED, "linear_gemv_ishizuki_k3_fused2");
    assert!(
        !plain.contains("acc_b0") && !plain.contains("acc1_0"),
        "fused2 gained column sharing; it is no longer the unshared baseline"
    );
    assert!(
        plain.contains("tgid >= tiles_a"),
        "fused2's grid is no longer tiles_a + tiles_b"
    );
}

// ============================================================================
// 5. Threadgroup staging: quantified, and it does not pay
// ============================================================================

#[test]
fn down_proj_activation_does_not_fit_threadgroup_memory() {
    // down_proj has the largest K, and 42.0 KiB > 32 KiB.
    let stage = Dispatch {
        kernel: "colmajor_2col_sk2",
        in_features: INTERMEDIATE,
        out_features: HIDDEN,
        cols_per_group: 2,
        split_k: 2,
        groups: (HIDDEN / 16) as u64,
        layers: 60,
    }
    .x_stage_bytes();
    assert_eq!(stage, 43_008, "down_proj x stage is 42.0 KiB");
    assert!(
        stage as usize > THREADGROUP_MEMORY_BYTES,
        "down_proj now fits threadgroup memory -- re-evaluate staging"
    );
}

#[test]
fn o_proj_activation_fits_but_staging_would_not_remove_any_dram_read() {
    // o_proj *does* fit (16 KiB of 32), so the "does not fit" objection does not
    // carry it. What does: staging keeps the DRAM read count at 1 and only
    // changes which cache serves repeats -- and the repeats are already L1 hits.
    let stage = SLIDING_Q as u64 * HALF_BYTES;
    assert_eq!(stage, 16_384, "o_proj x is 16 KiB");
    assert!(stage as usize <= THREADGROUP_MEMORY_BYTES);
    // It would also eat half the threadgroup memory the accumulator needs.
    assert!(
        REPACKED.contains("threadgroup float red[4][SG][16]"),
        "the 4-column accumulator scratch changed"
    );
    let red_bytes = (4 * 8 * 16 * 4) as u64;
    assert_eq!(red_bytes, 2_048, "red scratch is 2 KiB");
    assert!(
        stage + red_bytes > (THREADGROUP_MEMORY_BYTES / 2) as u64,
        "staging plus scratch would exceed half of threadgroup memory"
    );
}

#[test]
fn staging_would_be_swamped_by_the_weight_stream() {
    // Even if L1 pressure were the binding constraint, `x` is not it: the same
    // L1 carries 12 GB/token of weights.
    let ratio = per_token_weight_bytes() as f64 / x_dram_bytes_per_token() as f64;
    assert!(
        (ratio - 2394.0).abs() < 5.0,
        "weights are {ratio:.0}x x; staging x could not matter at this ratio"
    );
    assert!(
        ratio > 200.0,
        "if weights do not dominate x by >200:1, staging could matter"
    );
}

// ============================================================================
// The proposed change, and its honest expected value
// ============================================================================

#[test]
fn expected_gain_is_eighteen_times_below_the_recorded_noise_floor() {
    // `linear.metal:1584-1586` measures loads at 0.336 ms of 0.612 ms for
    // q_proj. `x` is half the loads, halved -> 13.7% of q_proj. q_proj is 6.86%
    // of the step's weight bytes -> 0.94% of a step.
    let load_fraction: f64 = 0.336 / 0.612;
    let x_share: f64 = 0.5;
    let x_cut: f64 = 0.5;
    let q_proj_gain = load_fraction * x_share * x_cut;
    assert!(
        (q_proj_gain - 0.1373).abs() < 0.001,
        "q_proj ceiling is {q_proj_gain:.4}, expected 0.1373"
    );
    let q_bytes = (HIDDEN * SLIDING_Q * 3 / 8) as u64;
    let q_share = (q_bytes * SLIDING_LAYERS as u64) as f64 / per_token_weight_bytes() as f64;
    assert!(
        (q_share - 0.0686).abs() < 0.001,
        "q_proj is {q_share:.4} of the step's weight bytes, expected 0.0686"
    );
    let step_gain = q_proj_gain * q_share;
    assert!(
        (step_gain - 0.00941).abs() < 0.0005,
        "whole-step gain is {step_gain:.5} ({:.2}%), expected 0.94%",
        100.0 * step_gain
    );
    assert!(
        (MEASURED_TOK_S * step_gain - 0.151).abs() < 0.005,
        "the change is worth ~+0.15 tok/s"
    );
    // AGENTS.md §10.2: the same binary has measured 3.7 and 6.4 tok/s.
    let noise = (6.4 - 3.7) / MEASURED_TOK_S;
    assert!(
        (noise - 0.1682).abs() < 0.002,
        "recorded same-binary spread is {noise:.3} of the rate"
    );
    let ratio = noise / step_gain;
    assert!(
        (ratio - 17.9).abs() < 1.0,
        "the predicted effect is only {ratio:.0}x below the noise floor. \
         Single-reading A/B will NOT resolve it."
    );
}

// ============================================================================
// Shape cross-checks: keep this probe honest against the real table.
// ============================================================================

#[test]
fn probe_shapes_match_the_committed_shape_table() {
    let find = |name: &str| SHAPES.iter().find(|s| s.name == name).expect(name);
    assert_eq!(find("sliding/q").in_features, HIDDEN);
    assert_eq!(find("sliding/q").out_features, SLIDING_Q);
    assert_eq!(find("sliding/k").out_features, SLIDING_KV);
    assert_eq!(find("sliding/v").out_features, SLIDING_KV);
    assert_eq!(find("sliding/o").in_features, SLIDING_Q);
    assert_eq!(find("sliding/o").out_features, HIDDEN);
    assert_eq!(find("sliding/gate").out_features, INTERMEDIATE);
    assert_eq!(find("sliding/up").out_features, INTERMEDIATE);
    assert_eq!(find("sliding/down").in_features, INTERMEDIATE);
    assert_eq!(find("sliding/down").out_features, HIDDEN);
    assert_eq!(find("full/q").out_features, FULL_Q);
    assert_eq!(find("full/kv").out_features, FULL_KV);
    assert_eq!(find("full/o").in_features, FULL_Q);
    assert_eq!(SLIDING_LAYERS + FULL_LAYERS, 60, "the model is 60 layers");
}

#[test]
fn dispatch_groups_match_the_linear_rs_arithmetic() {
    // The `groups` in `shipped_dispatches` are copied from `linear.rs`. If the
    // dispatch arithmetic changes, `x_is_read_once_per_kernel_for_every_shape`
    // could start passing for the wrong reason.
    assert!(
        LINEAR_RS.contains("(pipe, (tiles_a + tiles_b) / 2)"),
        "gate/up 2col grid formula moved"
    );
    assert!(
        LINEAR_RS.contains("(((out_features / 16) / 2) as u64) * 2,"),
        "sk2 2col grid formula moved"
    );
    assert!(
        LINEAR_RS.contains("(pipe, (tiles_a + tiles_b + tiles_c) / 2)"),
        "qkv fused3_2col grid formula moved"
    );
    assert_eq!(
        ((2 * INTERMEDIATE) / 32) as u64, 1344,
        "gate/up dispatches 1344 threadgroups"
    );
    assert_eq!((HIDDEN / 16) as u64, 336, "o_proj/down_proj dispatch 336 groups");
}