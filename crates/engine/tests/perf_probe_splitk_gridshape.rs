//! Split-factor / grid-shape probe for the long-K decode GEMVs.
//!
//! WHY THIS FILE EXISTS
//!
//! The standing claim about `gate_up` / `down_proj` is that they run at
//! 126-129 GB/s ("implied", `spec/decode-25tok-peak-bandwidth/implementation_notes.md`
//! §18.2) and that split-K is the untried lever. Both halves of that are wrong,
//! and this file is the arithmetic that shows it, plus the sweep that can
//! settle the one cell that genuinely has never been measured.
//!
//! **Wrong #1: the shapes.** The brief this probe was written against said
//! "gate_proj, up_proj and down_proj all have K=21504". They do not.
//! `crates/engine/src/config.rs:226-227` pins `hidden_size = 5376` and
//! `intermediate_size = 21504`, so:
//!
//! | projection | K (tokens read) | N (outputs) | tiles_k | tiles_n |
//! |---|---|---|---|---|
//! | gate_proj / up_proj | 5376 | 21504 | 336 | 1344 |
//! | down_proj | 21504 | 5376 | 1344 | 336 |
//! | o_proj | 8192 | 5376 | 512 | 336 |
//!
//! Only `down_proj` has K=21504. `gate_up` has the *short* K and the huge N --
//! the opposite of the starved case. §16.3's sweep puts K=5376 at 168-179 GB/s
//! and K=21504 at 199-203 GB/s, so the projection that looks like the biggest
//! prize is already the best-shaped one in the model.
//!
//! **Wrong #2: the 126-129 GB/s.** That table is `KAPLUMBA_ABLATE` **marginal**
//! cost, and marginal cost is not execution time. Removing `gate_up_gemv`
//! removes 5.202 GB/token of streaming *and* frees that bandwidth for every
//! other kernel in the step, so the delta prices the traffic it denied to
//! everyone else. It cannot be additive: the five deltas sum to 87.6 ms of GEMV
//! inside a 63.1 ms step (1.42x), and the `gate_up` + `down` deltas alone
//! (60.97 ms) leave 2.1 ms for qkv + o_proj + head + attention + norms + 1617
//! dispatch gaps, which is impossible. Rescaled so the table sums to the step it
//! was measured in (x0.708), `gate_up` is 28.6 ms = 182 GB/s and `down` is
//! 14.6 ms = 178 GB/s -- inside 10% of the 203 GB/s measured ceiling, and
//! inside 2% of the profiler's independent 204 / 200 GB/s (§16.2).
//!
//! WHAT THIS FILE ASSERTS (runs everywhere, no GPU)
//!
//! The dispatch equations from `crates/kernels/src/linear.rs` and
//! `crates/engine/src/model.rs` evaluated at the production shapes, plus the
//! four claims that follow and that any change to those equations would
//! falsify:
//!
//! 1. `gate_up` already launches 8.4x the co-resident threadgroup budget, so
//!    split-K cannot buy it occupancy. The one recorded datapoint agrees:
//!    T5.2 measured `gate_up` split-K at -1.04 tok/s.
//! 2. `down_proj` is already at split 2 = 2.1x budget, and split 4 measured
//!    identical in isolation (T5.2) and a wash in production (T5.11).
//! 3. Strict wave quantisation predicts a 7-43% drain. The measured spread
//!    between the 2.1-wave and 4.2-wave `down_proj` cells is +1.9% (§16.3) and
//!    -0.04 tok/s (T5.11): the model over-predicts by more than an order of
//!    magnitude, so a persistent kernel that streams the three projections in
//!    one grid-stride loop cannot buy the ~2 ms it is being asked for.
//! 4. `o_proj` has one unmeasured cell in the (columns x split) design, and it
//!    is the one that matters: `o_proj` is the only projection at half its
//!    peers' bandwidth and it is worth ~8 ms of a 63 ms step.
//!
//! WHAT THIS FILE SWEEPS (`#[ignore]`, GPU)
//!
//! `split_factor_sweep` runs every existing `_1col` / `_2col` / `contig4` /
//! `_sk2` / `_sk4` / `_stream` variant at the three production shapes through
//! the engine's own dispatch code, plus the live `gate_up` configuration via
//! `encode_gemv_fused2_mlp`. It reports GB/s *and* the threadgroup count and
//! bytes-per-threadgroup beside each number, because §16.3 established that this
//! harness's absolute GB/s is trustworthy and its variant *ranking* is inverted
//! relative to production -- so a sweep row is evidence about the shape, never
//! about which variant to ship.
//!
//! Set `KAPLUMBA_PROBE_WEIGHTS=<model dir>` to swap the synthetic buffers for
//! the real `Gxma-repacked` trellises.
//!
//! Run:
//!   cargo test --release --test perf_probe_splitk_gridshape
//!   cargo test --release --test perf_probe_splitk_gridshape -- --ignored --nocapture

use std::ffi::c_void;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::time::Instant;

use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{LinearEncodeRequest, LinearKernels};
use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};

// ---------------------------------------------------------------------------
// Constants that are facts about the checkpoint and the hardware, not choices.
//
// `config.rs:226-227` -- the loader refuses anything else, so these are not
// tunables and asserting against them is asserting a real invariant.
const HIDDEN: usize = 5376;
const INTERMEDIATE: usize = 21504;
const O_IN: usize = 8192; // q_heads(32) * head_dim(256): config.rs:232 + model.rs:525

/// Bytes per 16x16 trellis tile at 3 bpw: 256 codes * 3 bits = 768 bits = 24
/// `u32` = 96 B. Mirrors `RING/WORDS` in `linear_repacked.metal`.
const BYTES_PER_TILE_3BPW: u64 = 96;

/// Threads per threadgroup on every decode GEMV in this tree.
const THREADS_PER_TG: u64 = 256;

/// Candidate co-resident threadgroup budgets for an M4 Pro (20 GPU cores).
///
/// The Metal toolchain is absent on this box
/// (`spec/decode-3bpw-amplification/implementation_notes.md` §1), so no
/// occupancy query can be compiled here and no register/simdgroup-per-core
/// figure can be read off a kernel. These are the three budgets the record
/// actually uses, and every conclusion below is stated against all three so it
/// does not depend on picking one:
///
/// * 80  = 20 cores x 4 TGs of 256 threads. A 1024-thread/core budget is the
///   register-limited figure T4.3 reached with <=30 registers.
/// * 160 = 20 cores x 8 TGs. The figure `spec/decode-20tps/
///   implementation_notes.md` §A.2 uses: "336 TGs ~= 2 waves of ~160".
/// * 320 = 20 cores x 16 TGs. The most permissive reading; T4.2's "84 TGs is
///   only 4.2 per core" implies 16/core was believed reachable.
const RESIDENT_TG_BUDGETS: [u64; 3] = [80, 160, 320];

/// One dispatch configuration: which kernel, at which shape, split how far.
///
/// `threadgroups()` is not a parameter -- it is *derived* from the dispatch
/// equations in `linear.rs` / `model.rs`, and
/// `dispatch_equations_match_the_hand_computed_production_grids` pins those
/// derivations against hand-computed production values, so a change to the
/// equations surfaces as a test failure rather than as a silently different
/// sweep.
#[derive(Clone, Debug)]
struct Cell {
    label: String,
    kernel: String,
    k: usize,
    /// Output width. For a two-matrix cell this is the width of *each* matrix
    /// and `bytes()` counts both.
    n: usize,
    /// Output tile-columns handled by one threadgroup.
    cols: usize,
    /// Split-K factor.
    splits: usize,
    /// How many packed matrices the dispatch reads (1, or 2 for `fused2`).
    matrices: usize,
    /// True when this is what the engine dispatches today with no env set.
    live: bool,
    /// The byte count, for shapes whose tensors are not all `n` wide. Only q/k/v
    /// needs it: 8192 + 4096 + 4096, not three times one width. Same contract as
    /// `threadgroups_override` -- the handwritten equation is the assertion.
    bytes_override: Option<u64>,
    /// The grid, for shapes that are not a uniform-width fusion and so cannot be
    /// derived from `n` and `matrices`. Only q/k/v needs it: its three tensors
    /// are 8192, 4096 and 4096 wide, not `n` each. `Some` here is asserted to
    /// agree with the handwritten equation in
    /// `dispatch_equations_match_the_hand_computed_production_grids`, so it
    /// cannot rot silently.
    threadgroups_override: Option<u64>,
}

impl Cell {
    fn tiles_n(&self) -> u64 {
        (self.n / 16) as u64
    }

    fn tiles_k(&self) -> u64 {
        (self.k / 16) as u64
    }

    /// The dispatch equation, transcribed from the call site.
    ///
    /// * `encode_ishizuki_gemv_named` (`linear.rs:1156-1487`): `_1col`/`_2col`
    ///   -> `min(n/16 or n/32, KAPLUMBA_GEMV_MAX_GROUPS)`; `_stream` ->
    ///   `(n/16 + 7)/8`; `_2col_sk{2,4}` -> `(n/32)*sk`; `_1col_sk2`, bare
    ///   `_sk{2,4}` and `_contig4*` -> `(n/16)*sk`; else `n/16`.
    /// * `encode_gemv_fused2_mlp` (`linear.rs:2492-2683`): the `_fused2_2col`
    ///   arm is `((n_a/16) + (n_b/16)) / 2` at `:2592`.
    /// * `gemv_max_groups()` is `u64::MAX` when `KAPLUMBA_GEMV_MAX_GROUPS` is
    ///   unset, which is the default, so `min` is the identity here.
    ///
    /// `_fused2_co*` is deliberately absent: `encode_gemv_fused2_mlp` tests the
    /// `_2col` arm *before* the co arm (`:2590`), so co is unreachable from the
    /// live MLP dispatch, and T5.2 measured it at 15.46 against 2col's 15.65.
    fn threadgroups(&self) -> u64 {
        if let Some(fixed) = self.threadgroups_override {
            return fixed;
        }
        let tiles = self.tiles_n();
        let matrices = self.matrices as u64;
        let splits = self.splits as u64;
        if self.kernel.ends_with("_stream") {
            return tiles.div_ceil(8);
        }
        if self.kernel.contains("_fused2") {
            // `_fused2_2col` at `linear.rs:2592` is `((n_a/16)+(n_b/16))/2` and
            // has no split factor. The only split-K arm of this dispatch is the
            // 1-column `_fused2_sk2` at `linear.rs:2533`, whose grid is
            // `(tiles_a + tiles_b) * 2` -- note no `/2`, because that kernel is
            // one column per threadgroup. So a split on a fused2 cell switches
            // kernel, not just a multiplier.
            return if self.cols == 2 && splits == 1 {
                (tiles * matrices) / 2
            } else {
                (tiles * matrices) * splits
            };
        }
        if self.cols == 2 {
            return ((tiles / 2) * splits).max(1);
        }
        (tiles * splits).max(1)
    }

    /// Distinct weight bytes one threadgroup streams.
    fn bytes_per_tg(&self) -> u64 {
        self.cols as u64 * (self.tiles_k() / self.splits.max(1) as u64) * BYTES_PER_TILE_3BPW
    }

    /// Distinct weight bytes the whole dispatch reads.
    fn bytes(&self) -> u64 {
        self.bytes_override
            .unwrap_or(self.matrices as u64 * self.k as u64 * self.n as u64 * 3 / 8)
    }

    fn waves(&self, resident: u64) -> f64 {
        self.threadgroups() as f64 / resident as f64
    }

    /// Upper bound on the fraction of the dispatch lost to a trailing partial
    /// wave, under the pessimistic assumption that threadgroups retire in
    /// lockstep waves and the tail does not overlap. The real number is much
    /// smaller; see `strict_wave_quantisation_over_predicts_the_measured_drain`.
    fn strict_wave_loss(&self, resident: u64) -> f64 {
        let tgs = self.threadgroups() as f64;
        let waves = (tgs / resident as f64).ceil() * resident as f64;
        (waves - tgs) / tgs
    }
}

/// Every production decode GEMV dispatch, as the engine builds it today.
fn production_cells() -> Vec<Cell> {
    vec![
        // gate_up: `model.rs:2188-2216` -> `encode_gemv_fused2_mlp` with no env
        // set -> `linear.rs:2590` `_fused2_2col` at `(1344 + 1344)/2`.
        Cell {
            label: "gate_up  (live)".into(),
            kernel: "linear_gemv_ishizuki_k3_fused2_2col".into(),
            k: HIDDEN,
            n: INTERMEDIATE,
            cols: 2,
            splits: 1,
            matrices: 2,
            live: true,
            threadgroups_override: None,
            bytes_override: None,
        },
        // down_proj: `model.rs:2250-2262` -> `use_sk2` -> `linear.rs:2262`
        // `colmajor_2col_sk2` at `(5376/32)*2`.
        Cell {
            label: "down     (live)".into(),
            kernel: "linear_gemv_ishizuki_k3_colmajor_2col_sk2".into(),
            k: INTERMEDIATE,
            n: HIDDEN,
            cols: 2,
            splits: 2,
            matrices: 1,
            live: true,
            threadgroups_override: None,
            bytes_override: None,
        },
        // o_proj: `model.rs:1943-1949` -> `use_sk2`, the same shared helper, so
        // it launches down_proj's grid from half the K.
        Cell {
            label: "o_proj   (live)".into(),
            kernel: "linear_gemv_ishizuki_k3_colmajor_2col_sk2".into(),
            k: O_IN,
            n: HIDDEN,
            cols: 2,
            splits: 2,
            matrices: 1,
            live: true,
            threadgroups_override: None,
            bytes_override: None,
        },
        // q/k/v: `model.rs` -> `encode_gemv_fused3` -> `_fused3_2col`.
        // Recorded for the wave table; the sweep does not re-measure it, because
        // the fused3 dispatch takes three packed matrices of *different* widths,
        // which `n` and `matrices` cannot express -- hence the override, checked
        // against the handwritten equation below.
        Cell {
            label: "qkv      (live)".into(),
            kernel: "linear_gemv_ishizuki_k3_fused3_2col".into(),
            k: HIDDEN,
            n: 8192,
            cols: 2,
            splits: 1,
            matrices: 3,
            live: true,
            threadgroups_override: Some(512),
            // (8192 + 4096 + 4096) x 5376 x 3/8 = 33 030 144, not three times
            // one width.
            bytes_override: Some(33_030_144),
        },
    ]
}

// ---------------------------------------------------------------------------
// Test 1: the dispatch equations, pinned against hand-computed values.
//
// If this fails, a kernel's grid equation changed and every number derived from
// `threadgroups()` below is stale. That is the point: the arithmetic tests in
// this file are only meaningful while the equations are pinned.
// ---------------------------------------------------------------------------

#[test]
fn dispatch_equations_match_the_hand_computed_production_grids() {
    let cell = |prefix: &str| -> Cell {
        production_cells()
            .into_iter()
            .find(|c| c.label.starts_with(prefix))
            .unwrap_or_else(|| panic!("no production cell labelled {prefix}"))
    };

    assert_eq!(
        cell("gate_up").threadgroups(),
        1344,
        "gate_up dispatches (21504/16 + 21504/16)/2 = 1344 TGs (linear.rs:2592)"
    );
    assert_eq!(
        cell("down").threadgroups(),
        336,
        "down_proj dispatches (5376/32)*2 = 336 TGs (linear.rs:2262)"
    );
    assert_eq!(
        cell("o_proj").threadgroups(),
        336,
        "o_proj shares down_proj's helper, so it launches the same 336 TGs from \
         half the K (model.rs:1943 -> linear.rs:2262)"
    );
    assert_eq!(
        cell("qkv").threadgroups(),
        (8192 / 16 + 4096 / 16 + 4096 / 16) as u64 / 2,
        "qkv dispatches (512 + 256 + 256)/2 = 512 TGs (linear.rs:2778). The cell \
         carries this as an override because its three tensors differ in width, \
         so the handwritten equation above is the real assertion."
    );

    // The byte accounting has to close, or the GB/s column is meaningless.
    // These are the same figures as §18.2's GB/token column divided by 60, and
    // they are the denominator of every bandwidth claim in the repo.
    //
    // §18.2 prints 5.202 / 2.601 / 1.982 / 0.991 GB per token for gate_up /
    // down / qkv / o_proj, i.e. 86.70 / 43.35 / 33.03 / 16.52 MB per layer. The
    // exact values are one part in 4300 lower than that rounding, so the table
    // is right to two decimals and the `5.202 / 0.129` style division against it
    // is what produced the "129 GB/s" figure this file is correcting.
    let all = production_cells();
    for c in &all {
        if c.bytes_override.is_some() {
            continue;
        }
        assert_eq!(
            c.bytes(),
            (c.k * c.n * 3 / 8 * c.matrices) as u64,
            "{} byte count disagrees with itself",
            c.label
        );
    }
    assert_eq!(cell("gate_up").bytes(), 86_704_128, "86.70 MB/layer");
    assert_eq!(cell("down").bytes(), 43_352_064, "43.35 MB/layer");
    assert_eq!(cell("o_proj").bytes(), 16_515_072, "16.52 MB/layer");
    assert_eq!(cell("qkv").bytes(), 33_030_144, "33.03 MB/layer");

    // The per-token totals §16.4 lists, recomputed from the layer bytes. This
    // closes on 11.834 GB/token only after `lm_head`'s 1.057 GB, which is 6 bpw
    // and out of scope here.
    let layer_sum: u64 = all.iter().map(|c| c.bytes()).sum();
    println!(
        "layer sum {} bytes = {:.2} MB/layer = {:.3} GB/token over 60 layers",
        layer_sum,
        layer_sum as f64 / 1e6,
        layer_sum as f64 * 60.0 / 1e9
    );
    assert_eq!(layer_sum, 179_601_408, "179.60 MB/layer");
    assert_eq!(layer_sum * 60, 10_776_084_480, "10.776 GB/token of layers");
}

// ---------------------------------------------------------------------------
// Test 2: gate_up's split factor cannot buy occupancy, because occupancy is
// already spent 8x over.
//
// This is the falsifiable form of "T5.2 measured gate_up split-K at -1.04
// tok/s". The mechanism claim is the threadgroup count; the measurement only
// confirmed it. If the grid equation changes so gate_up drops below the budget,
// this test fails and the -1.04 datapoint has to be re-derived.
// ---------------------------------------------------------------------------

#[test]
fn gate_up_grid_is_already_oversubscribed_at_every_candidate_resident_budget() {
    let gate_up = production_cells()
        .into_iter()
        .find(|c| c.kernel.contains("_fused2"))
        .expect("gate_up cell");
    assert_eq!(gate_up.threadgroups(), 1344);

    // 1344 TGs is 16.8 waves at 80, 8.4 at 160, 4.2 at 320. The claim is not
    // "many waves at every budget" -- it is that split-K's *only* mechanism is
    // raising the wave count, and even at the most generous budget (320, which
    // T4.2's "84 TGs is 4.2 per core" implies 16 TGs/core was believed
    // reachable) the grid is already past 4 waves, so the mechanism is spent.
    for resident in RESIDENT_TG_BUDGETS {
        assert!(
            gate_up.waves(resident) >= 4.0,
            "at a {resident}-TG budget gate_up is only {:.1} waves, so split-K \
             could still buy parallelism and the -1.04 tok/s gate_up split-K \
             result would need re-deriving",
            gate_up.waves(resident)
        );
    }

    // Splitting K multiplies the grid by the split factor and does nothing else:
    // the resident set is already at the budget.
    for splits in [2usize, 4, 8] {
        let split_cell = Cell { splits, ..gate_up.clone() };
        assert!(
            split_cell.threadgroups() > gate_up.threadgroups(),
            "split {splits} did not raise the grid, so this test is not \
             exercising what it claims to"
        );
        assert!(
            split_cell.waves(RESIDENT_TG_BUDGETS[1]) >= gate_up.waves(RESIDENT_TG_BUDGETS[1]),
            "split {splits} *lowered* occupancy pressure; the mechanism claim is wrong"
        );
    }

    // And what split-K actually costs at gate_up's K: halving the K loop halves
    // the number of unrolled iterations each threadgroup amortises its fixed
    // epilogue (two `threadgroup_barrier`s plus an 8-deep serial reduce over
    // `red[2][8][16]`) over. tiles_k / step, with step = SG*2 = 16.
    assert_eq!(gate_up.tiles_k(), 336, "K=5376 -> 336 tiles");
    assert_eq!(336 / 16, 21, "21 unrolled iterations per threadgroup");
    let down = production_cells()
        .into_iter()
        .find(|c| c.label.starts_with("down"))
        .expect("down cell");
    assert_eq!(down.tiles_k() / down.splits as u64, 672, "down split 2 -> 672");
    assert_eq!(
        down.tiles_k() / down.splits as u64,
        2 * gate_up.tiles_k(),
        "down_proj already amortises its epilogue over twice gate_up's iteration \
         count, which is the whole reason §16.3's sweep ranks K=21504 \
         (199-203 GB/s) above K=5376 (168-179 GB/s)"
    );
}

// ---------------------------------------------------------------------------
// Test 3: down_proj's split ladder is climbed; o_proj's is not.
//
// down_proj: 168 (split 1) -> 336 (split 2, live) -> 672 (split 4). T5.2
// measured split 2 as a 2.7 ms win over split 1 and split 4 as a wash; T5.11
// measured `KAPLUMBA_O_SK4` at -0.04 tok/s in production.
// o_proj: the same ladder, but the number of output tile-columns per threadgroup
// is a second knob, and `encode_gemv_raw_sk2` chooses it from flags that
// down_proj also reads.
// ---------------------------------------------------------------------------

#[test]
fn down_proj_split_ladder_is_exhausted_and_o_projs_columns_knob_is_not() {
    let cells = production_cells();
    let down = cells.iter().find(|c| c.label.starts_with("down")).unwrap();
    let o_proj = cells.iter().find(|c| c.label.starts_with("o_proj")).unwrap();

    let ladder = |base: &Cell| -> [u64; 3] {
        [1usize, 2, 4].map(|s| Cell { splits: s, ..base.clone() }.threadgroups())
    };
    assert_eq!(ladder(down), [168, 336, 672], "down 168/336/672");
    assert_eq!(ladder(o_proj), [168, 336, 672], "o_proj 168/336/672");

    // The live choice sits one rung above one full wave at the 160 budget, and
    // 168 is the only cell that does not.
    assert_eq!(down.splits, 2);
    assert!(
        down.waves(160) > 1.0,
        "down split 1 (168 TGs) is the only cell at one wave, and it is the one \
         T5.2 measured 2.7 ms slower"
    );

    // The (columns x splits) design has six cells. `encode_gemv_raw_sk2` picks
    // columns from `KAPLUMBA_SK2_1COL` / `KAPLUMBA_SK2_CONTIG4`
    // (linear.rs:2243-2257) and `encode_gemv_raw_sk4` is hardwired to `_2col`
    // (linear.rs:2296), so from the flags as they stand o_proj reaches only
    // three grids: 168 (`KAPLUMBA_O_NOSK`), 336 (default), 672 (`KAPLUMBA_O_SK4`).
    let cells_of_interest = [
        (2usize, 1usize, "KAPLUMBA_O_NOSK", false),
        (2, 2, "default", true),
        (2, 4, "KAPLUMBA_O_SK4", true),
        (1, 2, "KAPLUMBA_SK2_1COL (also hits down_proj)", false),
    ];
    let grids: Vec<u64> = cells_of_interest
        .iter()
        .map(|(cols, splits, _, _)| {
            Cell {
                cols: *cols,
                splits: *splits,
                ..o_proj.clone()
            }
            .threadgroups()
        })
        .collect();
    assert_eq!(grids, vec![168, 336, 672, 672]);

    // The duplicate 672 *is* the finding: two configurations reach the grid
    // T5.11 measured, and they are not the same configuration.
    let sk4 = Cell { cols: 2, splits: 4, ..o_proj.clone() };
    let one_col_sk2 = Cell { cols: 1, splits: 2, ..o_proj.clone() };
    assert_eq!(sk4.threadgroups(), one_col_sk2.threadgroups());
    assert_eq!(sk4.threadgroups(), 672);
    // (2,4) pays for 4 fp32 partial planes and halves each threadgroup's K run;
    // (1,2) pays for 2 planes and halves only the per-threadgroup K run. So they
    // differ in reduction traffic *and* in bytes per threadgroup, and neither
    // difference is visible in the threadgroup count T5.11 logged.
    assert_eq!(sk4.splits, 4);
    assert_eq!(one_col_sk2.splits, 2);
    // And the sharper form of why they are not the same configuration: they move
    // the *identical* number of weight bytes per threadgroup (2 cols x 128 tiles
    // vs 1 col x 256 tiles), because doubling the columns halves the split. So
    // `bytes_per_tg` -- the quantity the whole "long K amortises a fixed
    // per-threadgroup cost" story rests on (§16.3, §16.4) -- cannot tell them
    // apart. What differs is the reduction: 4 fp32 partial planes against 2, so
    // `sk4` writes and re-reads twice as many partial bytes, and its finalize is
    // `linear_finalize_*_sk4` against `_sk2`.
    assert_eq!(
        one_col_sk2.bytes_per_tg(),
        sk4.bytes_per_tg(),
        "2 cols x 128 tiles and 1 col x 256 tiles must move the same bytes; if \
         they stopped doing so the two cells would differ in more than reduction \
         traffic and the argument below would need rewording"
    );
    assert_eq!(one_col_sk2.bytes_per_tg(), BYTES_PER_TILE_3BPW * 256);
    // The reduction traffic the two cells do NOT share: `splits` planes of N
    // fp32, written by the GEMV and read by the finalize.
    let partial_bytes = |c: &Cell| c.splits as u64 * c.n as u64 * 4;
    assert_eq!(partial_bytes(&sk4), 2 * partial_bytes(&one_col_sk2));
    assert_eq!(partial_bytes(&one_col_sk2), 2 * 5376 * 4);
    // §16.3's isolated sweep at K=8192/N=5376 puts these 52% apart
    // (sk2 120.7 vs sk4 183.3 GB/s); §16.3 also says that sweep inverts variant
    // rankings in production. So this cell needs a production A/B, not a number
    // from this file.
}

// ---------------------------------------------------------------------------
// Test 4: the drain a persistent kernel would remove is already small.
//
// The pessimistic bound is the strict-wave loss. The measured bound is the
// 2.1-wave vs 4.2-wave `down_proj` pair. The gap between them is the point: a
// megakernel streaming gate_up, down_proj and o_proj in one grid-stride loop
// could remove at most the pessimistic number, and the pessimistic number is
// what the measurement refutes.
// ---------------------------------------------------------------------------

#[test]
fn strict_wave_quantisation_over_predicts_the_measured_drain_by_an_order_of_magnitude() {
    for c in production_cells() {
        for resident in RESIDENT_TG_BUDGETS {
            println!(
                "{:<16} {resident:>4} TG budget: {:>6.1} waves, strict-wave loss \
                 {:>5.1}%, {} B/threadgroup",
                c.label,
                c.waves(resident),
                100.0 * c.strict_wave_loss(resident),
                c.bytes_per_tg(),
            );
        }
    }

    // down_proj at 336 TGs against a 160-TG budget is the 2.1-wave cell; split 4
    // is the 4.2-wave cell. §16.3 measured 199.3 -> 203.0 GB/s (+1.9%) and T5.11
    // measured 15.781 vs 15.818 tok/s (-0.04, a wash).
    let down = production_cells()
        .into_iter()
        .find(|c| c.label.starts_with("down"))
        .unwrap();
    let split4 = Cell { splits: 4, ..down.clone() };
    let bound = 100.0 * split4.strict_wave_loss(160);
    let measured_bound = 1.9_f64;
    println!("\ndown_proj split-4 strict-wave bound at 160 TG: {bound:.1}%");
    println!("measured split2 -> split4 delta: +{measured_bound:.1}% isolated, -0.04 tok/s in production");
    println!(
        "over-prediction factor: {:.1}x",
        bound / measured_bound
    );

    assert!(
        bound > 10.0 * measured_bound,
        "the strict-wave bound for down_proj split 4 is {bound:.1}% against a \
         measured +{measured_bound:.1}%; if it has closed to within 10x the wave \
         model has become usable and the drain estimate must be re-derived from \
         the sweep rather than from the stale measurement"
    );
    // For scale: one `down_proj` layer at the 203 GB/s ceiling is 43.35 MB /
    // 203 GB/s = 214 us, x60 layers = 12.8 ms of a 63.1 ms step. If the drain
    // really were the pessimistic number, splitting into 4.2 waves would hand
    // back tens of percent of that -- and it hands back 1.9%.
    let down_layer_us = down.bytes() as f64 / 203e9 * 1e6;
    println!(
        "down_proj: {down_layer_us:.1} us/layer at 203 GB/s = {:.2} ms/token over 60 layers",
        60.0 * down_layer_us / 1e3
    );
    assert!(
        (200.0..225.0).contains(&down_layer_us),
        "down_proj came out at {down_layer_us:.1} us/layer at 203 GB/s; the byte \
         count or the ceiling moved"
    );

    // And the ceiling the persistent-kernel idea is chasing. A megakernel can
    // only merge dispatches that are not sequentially dependent. Within a layer
    // the chain is strict:
    //   pre_attn_norm -> qkv -> attention -> o_proj -> post_attn_norm
    //                  -> gate_up -> gelu -> down_proj -> residual
    // and layer L+1's qkv reads layer L's residual. So the only pairs that can
    // ever share a dispatch are (q,k,v) and (gate,up) -- and both are already
    // fused, which is why the kernels are named `fused2` and `fused3`. Metal has
    // no grid-wide barrier (ishizuki `README.md:386-387`: "a megakernel buys
    // nothing on Metal, which has no grid-wide barrier"), and `down_proj` cannot
    // start until all 21504 values of the gated activation exist, so it cannot be
    // pipelined against gate_up's tail either.
    let all = production_cells();
    let fused: Vec<&str> = all
        .iter()
        .filter(|c| c.live && (c.kernel.contains("fused2") || c.kernel.contains("fused3")))
        .map(|c| c.kernel.as_str())
        .collect();
    assert_eq!(fused.len(), 2, "qkv (fused3) and gate_up (fused2) are the only two");
}

// ---------------------------------------------------------------------------
// Test 5: the bandwidth arithmetic, stated so it cannot be misread again.
//
// The ABLATE-delta table in §18.2 and the 63.1 ms step are not simultaneously
// true. This asserts the contradiction, so the next reader sees it as a
// contradiction rather than as two facts to multiply.
// ---------------------------------------------------------------------------

#[test]
fn the_ablate_implied_bandwidths_cannot_be_the_kernels_bandwidth() {
    // §18.1 marginal costs: GB/token and ms/step.
    let ablate: [(&str, f64, f64); 5] = [
        ("gate_up", 5.202, 40.34),
        ("down", 2.601, 20.63),
        ("qkv", 1.982, 15.82),
        ("o_proj", 0.991, 6.63),
        ("head", 1.057, 4.17),
    ];
    let gemv_sum: f64 = ablate.iter().map(|(_, _, ms)| ms).sum();
    let elements_ms = 1.52;
    let step_ms = 63.1_f64;
    // GB/token for one projection, from the layer bytes this file already
    // checks in `dispatch_equations_match_the_hand_computed_production_grids`.
    let cell_bytes = |prefix: &str| -> f64 {
        production_cells()
            .into_iter()
            .find(|c| c.label.starts_with(prefix))
            .unwrap_or_else(|| panic!("no production cell labelled {prefix}"))
            .bytes() as f64
            * 60.0
            / 1e9
    };

    assert!(
        gemv_sum + elements_ms > step_ms * 1.3,
        "if the marginal costs summed to less than the step they would be additive \
         and directly usable; they sum to {:.1} ms inside a {step_ms} ms step \
         ({:.2}x), so at least {:.0}% of every delta is bandwidth the ablated \
         kernel was denying to the others",
        gemv_sum + elements_ms,
        (gemv_sum + elements_ms) / step_ms,
        100.0 * (1.0 - step_ms / (gemv_sum + elements_ms)),
    );

    // Rescaled so the table sums to the step it was measured in, the long-K
    // projections land at 178-182 GB/s -- within 10% of the 203 GB/s measured
    // ceiling and within 2% of the profiler's independent 204 / 200 (§16.2).
    let scale = step_ms / (gemv_sum + elements_ms);
    let gate_up_gbps = 5.202 / (40.34 * scale / 1e3);
    let down_gbps = 2.601 / (20.63 * scale / 1e3);
    println!("rescale factor {scale:.3}");
    println!("gate_up {gate_up_gbps:.0} GB/s, down {down_gbps:.0} GB/s (was 129 / 126)");
    assert!(
        (175.0..190.0).contains(&gate_up_gbps) && (170.0..190.0).contains(&down_gbps),
        "rescaled gate_up/down came out at {gate_up_gbps:.0}/{down_gbps:.0} GB/s, \
         outside the 170-190 band the profiler and the microbenchmark agree on"
    );

    // The prize, computed properly. Bringing gate_up + down from the rescaled
    // 178-182 GB/s to the 203 GB/s measured ceiling saves:
    let gate_up_save = 5.202 / 182.0 * 1e3 - 5.202 / 203.0 * 1e3;
    let down_save = 2.601 / 178.0 * 1e3 - 2.601 / 203.0 * 1e3;
    let saving_ms = gate_up_save + down_save;
    let new_step = step_ms - saving_ms;
    println!(
        "gate_up+down at 203 GB/s: {saving_ms:.2} ms saved ({gate_up_save:.2} + \
         {down_save:.2}) -> {new_step:.1} ms = {:.2} tok/s (from {:.2})",
        1e3 / new_step,
        1e3 / step_ms
    );
    assert!(
        new_step > 50.0,
        "reaching 20 tok/s from these two projections alone would need a step of \
         {new_step:.1} ms; anything under 50 ms means the 203 GB/s ceiling itself \
         has moved and the whole ceiling argument has to be redone"
    );
    // 63.1 ms is 15.85 tok/s and ~4.8 ms is ~1.3 tok/s. The reason the whole
    // gate_up+down prize is only ~1.3 tok/s: §16.3 measured the best 3-bpw
    // kernel in this tree at 203 GB/s and `head/gemv` at 257 GB/s is 6 bpw, so
    // it is not a template. Getting the *only* two under-performing projections
    // to the best already-measured 3-bpw rate lands at 17.1 tok/s -- which is
    // §16.4's ceiling, arrived at from the opposite direction. The two routes to
    // the same number is the check that this arithmetic is not double-counting.
    assert!(
        (16.8_f64..18.0).contains(&(1e3 / new_step)),
        "the gate_up+down prize landed at {:.2} tok/s, outside the 16.8-18.0 band \
         that §16.4's independent 11.834/203 = 58.3 ms ceiling implies",
        1e3 / new_step
    );
    assert!(
        new_step > 50.0,
        "reaching 20 tok/s from these two projections alone would need a step of \
         {new_step:.1} ms; anything under 50 ms means the 203 GB/s ceiling itself \
         has moved and the whole ceiling argument has to be redone"
    );

    // And the number that actually matters, because o_proj is the only
    // projection more than 10% off its peers. §16.3 measured 120.7 GB/s at
    // K=8192/N=5376 sk2 and validated that cell against production's ~100-120.
    let o_gbps_micro = 120.7_f64;
    let o_ms = 0.991 / o_gbps_micro * 1e3;
    // The comparison has to be per byte, not per token: o_proj moves 4.7x less
    // data than gate_up, so 8.2 ms against 28.6 ms is o_proj being *faster* in
    // absolute terms. The deficit only shows up normalised.
    let gate_up_ms = 5.202 / 182.0 * 1e3;
    let o_bytes = cell_bytes("o_proj");
    let gate_up_bytes = cell_bytes("gate_up");
    let o_norm = o_ms / o_bytes;
    let gate_up_norm = gate_up_ms / gate_up_bytes;
    println!("o_proj  {o_ms:.1} ms/token over {o_bytes:.3} GB -> {o_norm:.4} ms/GB");
    println!("gate_up {gate_up_ms:.1} ms/token over {gate_up_bytes:.3} GB -> {gate_up_norm:.4} ms/GB");
    // ms/GB is *inverse* bandwidth, so the worse one has the *larger* number.
    println!(
        "o_proj is {:.2}x worse per byte than gate_up ({o_gbps_micro:.1} vs 182 GB/s)",
        o_norm / gate_up_norm
    );
    assert!(
        o_norm / gate_up_norm > 1.4,
        "o_proj normalised is only {:.2}x gate_up's; below 1.4x there is no \
         meaningful hole left and the recommendation in the report is stale",
        o_norm / gate_up_norm
    );
    // And the prize: if o_proj ran at 182 GB/s like the long-K projections, its
    // cost falls by the same factor. That is where the ~1 tok/s in the report
    // comes from, and it is why o_proj rather than gate_up is the target.
    let o_if_peers_ms = o_bytes * gate_up_norm;
    println!(
        "o_proj at 182 GB/s would be {o_if_peers_ms:.2} ms/token, saving {:.2} ms \
         = +{:.2} tok/s",
        o_ms - o_if_peers_ms,
        1e3 / (step_ms - (o_ms - o_if_peers_ms)) - 1e3 / step_ms
    );
    assert!(
        o_ms - o_if_peers_ms > 2.0 && o_ms - o_if_peers_ms < 5.0,
        "the o_proj prize came out at {:.2} ms; outside 2-5 ms, re-derive",
        o_ms - o_if_peers_ms
    );
}

// ---------------------------------------------------------------------------
// The sweep. `#[ignore]` because it needs the GPU and takes minutes.
// ---------------------------------------------------------------------------

/// Aim for this much GPU time per cell. Long, because §16.3 recorded a 4.7 ms
/// measurement with a 3.5x spread between adjacent K -- clock ramp, not kernel.
const TARGET_MS: f64 = 40.0;
/// Rotating-buffer bounds, so no cell can be served from L2 and no cell can
/// exhaust memory.
const MAX_WORKING_SET: usize = 768 << 20;
const MIN_WORKING_SET: usize = 192 << 20;

fn synthetic(device: &Device, bytes: usize, seed: u32) -> Buffer {
    let mut state = seed | 1;
    let words = bytes / 2;
    let mut data: Vec<u16> = Vec::with_capacity(words);
    for _ in 0..words {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        data.push(state as u16);
    }
    device.new_buffer_with_data(
        data.as_ptr() as *const c_void,
        (data.len() * size_of::<u16>()) as u64,
        // Private, not Shared: `concurrent_stream_economics.rs` §"Sizing" records
        // a Shared buffer reporting 804 GB/s against a 273 GB/s wire.
        MTLResourceOptions::StorageModePrivate,
    )
}

/// Where the packed trellises come from.
///
/// Synthetic buffers of pseudo-random `u16` measure *streaming bandwidth*, which
/// is what this probe is for: the kernel extracts bit patterns and the code
/// values cannot change how many bytes cross the bus.
/// `rate4_colmajor_bit_exact.rs:746` already asserts DRAM line traffic is 1x at
/// every rate, so buffer contents cannot flatter the absolute numbers.
/// Correctness of the variants is `live_kernel_bit_exact.rs` and
/// `splitk_bit_exact.rs`, not this file.
///
/// `Real` reads the genuine trellis out of a checkpoint. Set
/// `KAPLUMBA_PROBE_WEIGHTS=/path/to/Gxma-repacked`.
#[derive(Debug)]
enum Packed {
    Synthetic,
    Real(PathBuf),
}

impl Packed {
    fn from_env() -> Self {
        match std::env::var("KAPLUMBA_PROBE_WEIGHTS") {
            Ok(dir) if !dir.trim().is_empty() => Packed::Real(PathBuf::from(dir)),
            _ => Packed::Synthetic,
        }
    }

    /// Byte range of one tensor inside its shard.
    fn real_range(dir: &Path, tensor: &str) -> Result<(u64, u64, PathBuf), String> {
        let meta = kaplumba_engine::safetensors::inspect_model_dir(dir)
            .map_err(|error| format!("{dir:?}: {error}"))?;
        let shard = meta
            .selection
            .shard_for(tensor)
            .ok_or_else(|| format!("{tensor} is not in the text selection"))?;
        let (start, end) = meta
            .selection
            .absolute_range(tensor)
            .ok_or_else(|| format!("{tensor} has no byte range"))?;
        Ok((start, end - start, dir.join(shard)))
    }

    /// One buffer of at least `bytes`, read from the real checkpoint when
    /// `KAPLUMBA_PROBE_WEIGHTS` is set.
    fn one(&self, device: &Device, tensor: &str, bytes: usize) -> Result<Buffer, String> {
        match self {
            Packed::Synthetic => Ok(synthetic(device, bytes, 0x1234_5678)),
            Packed::Real(dir) => {
                let (start, len, shard) = Self::real_range(dir, tensor)?;
                if len < bytes as u64 {
                    return Err(format!(
                        "{tensor} is {len} B but this cell wants {bytes} B of packed \
                         stream -- wrong tensor name for this shape"
                    ));
                }
                use std::io::{Read, Seek};
                let mut file = std::fs::File::open(&shard).map_err(|e| e.to_string())?;
                let mut host = vec![0u8; bytes];
                file.seek(std::io::SeekFrom::Start(start))
                    .map_err(|e| e.to_string())?;
                file.read_exact(&mut host).map_err(|e| e.to_string())?;
                // The host copy is dropped immediately and never read back.
                Ok(device.new_buffer_with_data(
                    host.as_ptr() as *const c_void,
                    bytes as u64,
                    MTLResourceOptions::StorageModePrivate,
                ))
            }
        }
    }
}

/// The trellis tensor names one cell reads. Two-matrix cells read gate and up;
/// everything else reads the one named here. Names follow `weights.rs:1177` and
/// `model.rs:3425-3477`.
fn tensor_names(c: &Cell) -> Vec<String> {
    let layer = "model.language_model.layers.0";
    if c.matrices >= 2 {
        vec![
            format!("{layer}.mlp.gate_proj.trellis"),
            format!("{layer}.mlp.up_proj.trellis"),
        ]
    } else if c.n == O_IN {
        vec![format!("{layer}.self_attn.o_proj.trellis")]
    } else {
        vec![format!("{layer}.mlp.down_proj.trellis")]
    }
}

struct Bench {
    packed: Vec<Buffer>,
    transformed: Buffer,
    raw: Buffer,
    output: Buffer,
    output_signs: Buffer,
    input: Buffer,
    input_signs: Buffer,
}

fn prepare(device: &Device, source: &Packed, c: &Cell) -> Result<Bench, String> {
    let per_matrix = c.k * c.n * 3 / 8;
    let total = per_matrix * c.matrices;
    let nbuf = (MAX_WORKING_SET / total.max(1))
        .clamp(2, 8)
        .min((MIN_WORKING_SET / total.max(1)).max(2));

    let names = tensor_names(c);
    let mut packed = Vec::with_capacity(nbuf * c.matrices);
    for i in 0..nbuf {
        for name in &names {
            packed.push(source.one(device, name, per_matrix)?);
        }
        // Distinct seeds per rotation when synthetic, so a rotated cell cannot
        // be served from a line the previous rotation left hot.
        let _ = i;
    }

    // `transformed` is fed as pre-transformed garbage: the probe measures
    // streaming, and x is 10-43 KB so it is L2-resident either way.
    // `pretransformed: true` keeps the input dispatch out of the span.
    let splits = c.splits.max(1);
    Ok(Bench {
        packed,
        transformed: synthetic(device, c.k * 2, 0x9abc_def0),
        raw: synthetic(device, (c.n * splits) * 4, 0xfeed_beef),
        output: synthetic(device, c.n * 2, 0x0bad_f00d),
        output_signs: synthetic(device, c.n * 2, 0x1111_2222),
        input: synthetic(device, c.k * 2, 0x3333_4444),
        input_signs: synthetic(device, c.k * 2, 0x5555_6666),
    })
}

/// Dispatch one repetition of `c`.
fn dispatch(kernels: &LinearKernels, enc: &mut MetalEncoder, bench: &Bench, c: &Cell) {
    let stride = c.matrices.max(1);
    let packed = |i: usize| -> &Buffer { &bench.packed[(i * stride) % bench.packed.len()] };

    if c.matrices >= 2 {
        // The live gate_up path, so the grid is `(1344 + 1344)/2` and not the
        // half-size a single-matrix 2col dispatch would give at N=21504.
        kernels
            .encode_gemv_fused2_mlp(
                enc,
                packed(0),
                0,
                c.n,
                &bench.output_signs,
                packed(1),
                0,
                c.n,
                &bench.output_signs,
                None,
                &bench.transformed,
                &bench.input,
                &bench.input_signs,
                &bench.transformed,
                &bench.raw,
                &bench.raw,
                c.k,
                true,
                0,
            )
            .expect("encode_gemv_fused2_mlp");
        return;
    }

    let request = LinearEncodeRequest {
        packed: packed(0),
        packed_offset: 0,
        perm_mode: 1,
        input: &bench.input,
        input_signs: &bench.input_signs,
        output_signs: &bench.output_signs,
        transformed: &bench.transformed,
        raw: &bench.raw,
        output: &bench.output,
        in_features: c.k,
        out_features: c.n,
        rate: Exl3Rate::Integer(3),
        codebook: Exl3Codebook::Mcg,
        pretransformed: true,
    };
    kernels
        .encode_ishizuki_gemv_named(enc, &c.kernel, &request)
        .unwrap_or_else(|error| panic!("{}: {error}", c.kernel));
}

/// Run one cell and return achieved GB/s, or `None` if the GPU clock is
/// unavailable (`GPUStartTime` reads back as zero without the engine's timing
/// trait -- §16.3 got this wrong first and reported 1600-2000 GB/s).
fn run(
    kernels: &LinearKernels,
    device: &Device,
    queue: &CommandQueue,
    source: &Packed,
    c: &Cell,
) -> Option<f64> {
    let bench = prepare(device, source, c).ok()?;
    let nbuf = bench.packed.len() / c.matrices.max(1);

    // Calibrate on wall time to size the run. This only picks a length; every
    // number reported comes from the GPU clock below.
    let reps = {
        let mut enc = MetalEncoder::new(queue).expect("encoder");
        enc.set_max_dispatches(u32::MAX);
        let t0 = Instant::now();
        for i in 0..4 {
            dispatch(kernels, &mut enc, &bench, c);
            let _ = i;
        }
        enc.commit_and_wait().expect("calibrate");
        let per = (t0.elapsed().as_secs_f64() * 1e3 / 4.0).max(1e-4);
        (((TARGET_MS / per).round() as usize).clamp(4, 2000)).min(nbuf.max(4) * 8)
    };

    // Warm the clocks and the page tables; discarded.
    {
        let mut enc = MetalEncoder::new(queue).expect("encoder");
        for i in 0..reps.min(8) {
            dispatch(kernels, &mut enc, &bench, c);
            let _ = i;
        }
        enc.commit_and_wait().expect("warmup");
    }

    let mut enc = MetalEncoder::new(queue).expect("encoder");
    enc.set_max_dispatches(u32::MAX);
    for i in 0..reps {
        dispatch(kernels, &mut enc, &bench, c);
        let _ = i;
    }
    enc.commit_stamped(&c.label).expect("commit");
    let gpu_ms: f64 = enc.take_gpu_stamps().iter().map(|s| s.own_ms).sum();
    enc.commit_and_wait().ok();

    if gpu_ms <= 0.0 {
        return None;
    }
    Some((c.bytes() as f64 * reps as f64) / (gpu_ms * 1e-3) / 1e9)
}

/// The single-matrix variants reachable at a given width.
///
/// `_4col` is excluded on purpose: `encode_ishizuki_gemv_named` classifies by
/// `ends_with("_2col")`, false for `_4col`, so it falls through to
/// `encode_ishizuki_gemv` and launches `n/16` threadgroups instead of `n/64` --
/// a wrong grid, not a slow kernel. The `fused3` family takes three packed
/// matrices and is not driven through this path.
fn variants(n: usize) -> Vec<Cell> {
    [
        ("1col", "_colmajor", 1usize, 1usize),
        ("2col", "_colmajor_2col", 2, 1),
        ("1col_sk2", "_colmajor_1col_sk2", 1, 2),
        ("2col_sk2", "_colmajor_2col_sk2", 2, 2),
        ("2col_sk4", "_colmajor_2col_sk4", 2, 4),
        ("colmajor_sk4", "_colmajor_sk4", 1, 4),
        ("contig4", "_colmajor_contig4", 1, 1),
        ("contig4_sk2", "_colmajor_contig4_sk2", 1, 2),
        ("contig4_sk4", "_colmajor_contig4_sk4", 1, 4),
        ("stream", "_colmajor_stream", 1, 1),
    ]
    .into_iter()
    .map(|(label, suffix, cols, splits)| Cell {
        label: label.to_string(),
        kernel: format!("linear_gemv_ishizuki_k3{suffix}"),
        k: 0,
        n,
        cols,
        splits,
        matrices: 1,
        live: false,
        threadgroups_override: None,
        bytes_override: None,
    })
    .collect()
}

#[test]
#[ignore = "microbenchmark; needs the GPU and minutes of runtime"]
fn split_factor_sweep() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let kernels = LinearKernels::new().expect("linear kernels compile");
    let source = Packed::from_env();
    let available = kernels.ishizuki_gemv_kernel_names();

    println!("\nsource: {source:?}");
    println!(
        "\n{:<13} {:>7} {:>7} {:<13} {:>5} {:>7} {:>6} {:>9} {:>8}",
        "shape", "K", "N", "variant", "cols", "splits", "TGs", "B/TG", "GB/s"
    );

    let shapes: [(&str, usize, usize); 3] = [
        ("gate_up(one)", HIDDEN, INTERMEDIATE),
        ("down", INTERMEDIATE, HIDDEN),
        ("o_proj", O_IN, HIDDEN),
    ];

    for (shape_label, k, n) in shapes {
        let mut cells: Vec<Cell> = variants(n);
        if shape_label.starts_with("gate_up") {
            // The live configuration, and the only cell whose grid matches what
            // the engine dispatches for gate_up.
            cells.push(Cell {
                label: "fused2_2col".into(),
                kernel: "linear_gemv_ishizuki_k3_fused2_2col".into(),
                k,
                n,
                cols: 2,
                splits: 1,
                matrices: 2,
                live: true,
                threadgroups_override: None,
                bytes_override: None,
            });
        }
        for mut c in cells {
            c.k = k;
            if !available.iter().any(|name| name == &c.kernel) {
                println!("{shape_label:<13} {k:>7} {n:>7} {} not in this library, skipped", c.label);
                continue;
            }
            let gbps = run(&kernels, &device, &queue, &source, &c);
            println!(
                "{:<13} {:>7} {:>7} {:<13} {:>5} {:>7} {:>6} {:>9} {:>8}",
                shape_label,
                k,
                n,
                c.label,
                c.cols,
                c.splits,
                c.threadgroups(),
                c.bytes_per_tg(),
                gbps.map(|v| format!("{v:.1}")).unwrap_or_else(|| "--".into()),
            );
        }
        println!();
    }

    println!(
        "\nRead the TGs column first. §16.3 established that this harness's \
         absolute GB/s is trustworthy and its variant ranking is inverted \
         relative to production, so a row is evidence about the *shape* -- grid \
         size, bytes per threadgroup, tiles per threadgroup -- and never about \
         which variant to ship. Only interleaved production A/B decides that, \
         and only one model process may run at a time (AGENTS.md §10)."
    );
}

/// Every variant the sweep names must resolve, so a sweep that silently skipped
/// half its rows cannot look complete.
#[test]
#[ignore = "needs the GPU only to compile the library"]
fn every_variant_the_sweep_names_resolves() {
    let kernels = LinearKernels::new().expect("linear kernels compile");
    let available = kernels.ishizuki_gemv_kernel_names();
    let mut missing = Vec::new();
    for n in [HIDDEN, INTERMEDIATE, O_IN] {
        for c in variants(n) {
            if !available.iter().any(|name| name == &c.kernel) {
                missing.push(c.kernel);
            }
        }
    }
    if !available
        .iter()
        .any(|name| name == "linear_gemv_ishizuki_k3_fused2_2col")
    {
        missing.push("linear_gemv_ishizuki_k3_fused2_2col".to_string());
    }
    assert!(
        missing.is_empty(),
        "the sweep would skip these kernels: {missing:?}"
    );
    println!("{} ishizuki kernels in the library", available.len());
}
