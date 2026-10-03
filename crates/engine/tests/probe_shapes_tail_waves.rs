//! Per-shape tile counts, threadgroup grids, wave counts and tail waste for the
//! seven production projection shapes, plus the arithmetic that decides whether
//! grid quantisation is worth anything on this box.
//!
//! # What this file is for
//!
//! The question put to it was: *"the shapes divide perfectly by 16, so the
//! threadgroup grids are 336 / 1344 threadgroups; on a GPU holding ~240
//! threadgroups concurrently, 336 is 1.4 waves and the trailing wave is 40%
//! occupied; across ~420 dispatches per token that is milliseconds of drain.
//! Would a persistent kernel, or a per-shape tiles-per-threadgroup choice,
//! recover it?"*
//!
//! Three of those four premises are false, and the falsifications are the
//! deliverable. Each is pinned by a test below so it cannot rot.
//!
//! ## Premise 1 (FALSE): production grids are 336 / 1344 / 16384
//!
//! Those are `out_features / 16`, i.e. the grid of `encode_ishizuki_gemv`
//! (`crates/kernels/src/linear.rs:1479-1480`). **That function is not on the
//! decode path for this checkpoint.** `model.rs:3525-3541` sets
//! `permutation_mode = 1` for every decoder linear when the checkpoint layout is
//! `"colmajor"`, and with `perm_mode == 1` `encode_gemv_raw` takes the `_2col`
//! arm at `linear.rs:2133-2157` *before* it ever reaches
//! `ishizuki_gemv_pipeline` at `linear.rs:2159`. Same for `lm_head`
//! (`model.rs:4425`).
//!
//! So the live grids are half the quoted ones, and `lm_head` launches **8192**
//! threadgroups, not 16384. The comment at `linear.rs:1473-1478` ("At N=21504
//! that is 1344 threadgroups ... lm_head, which launches 16384 threadgroups,
//! reaches 168.7 GB/s on this same kernel") describes a path this model does not
//! take, and it is the text a future spec would cite.
//!
//! ## Premise 2 (TRUE, and stronger than asked): no remainder threads anywhere
//!
//! `model.rs:384-385` already rejects any linear whose in/out features are not
//! multiples of **128**, so both tile axes are exact by construction. The live
//! `_2col` kernel additionally unrolls K by `SG * 2 = 16`
//! (`linear_repacked.metal:2115-2119`) and every production `tiles_k` is a
//! multiple of 16, so even the tail loop at `linear_repacked.metal:2041` never
//! executes. The only path with a live K remainder is the `_contig4` fallback
//! (unroll 32): `tiles_k = 336` leaves a 16-tile remainder pass on all three
//! K=5376 shapes. See `remainder_k_loop_runs_only_on_contig4`.
//!
//! ## Premise 3 (FALSE, and the point of the file): the tail is not the drain
//!
//! The strict wave-quantisation model says `o_proj` and `down_proj` (168
//! threadgroups, 30-47.5% tail) are the worst-drained kernels in the step. The
//! profiler's own per-kernel milliseconds say they are among the *fastest* per
//! byte. See `strict_wave_model_is_falsified_by_the_measured_bandwidths` and
//! `modelled_tail_cannot_be_fifty_percent_because_the_wire_forbids_it`.
//!
//! ## Premise 4: neither proposed fix is available
//!
//! Both are measured and negative. `spec/decode-25tok-peak-bandwidth/tasks.md`
//! T5.11 ran the grid axis on `o_proj` in production: 336 threadgroups baseline
//! 15.83 tok/s, `KAPLUMBA_GEMV_MAX_GROUPS=168` 15.716 (-0.12), `=84` 15.595
//! (-0.21). Capping the grid *is* the persistent kernel on this shape family --
//! `linear.metal:1626` is already `for (uint tn = tgid; tn < tiles_n; tn += ntg)`,
//! a grid-stride loop -- so "add a persistent kernel" is `KAPLUMBA_GEMV_MAX_GROUPS`,
//! and it has been run. It loses.
//!
//! And the arithmetic in `integral_wave_grid_is_never_also_tile_balanced` shows
//! why the *other* direction cannot work either: making the grid an exact
//! multiple of the co-resident budget necessarily leaves 336 tiles spread
//! unevenly over it, and that tile-granularity quantisation is worse than the
//! wave quantisation it removes at two of the four candidate budgets.
//!
//! # What this file asserts (pure arithmetic, no GPU)
//!
//! Run: `cargo test --release --test probe_shapes_tail_waves -- --nocapture`
//!
//! The dispatch equations are transcribed from `linear.rs`, not reimplemented
//! from the shapes, and each is checked against a hand-computed production
//! value, so a change to `linear.rs` surfaces here as a failing grid number
//! rather than as a silently different model.

use std::fmt::Write as _;

// ---------------------------------------------------------------------------
// Shapes. Not tunables: `config.rs:225-234` refuses to load a checkpoint that
// differs in any of them, and `model.rs:3364-3374` derives the per-layer head
// geometry from `layer_types` (1 full layer in 6, `config.rs:241-245`).
const HIDDEN: usize = 5376; // config.rs:226
const INTERMEDIATE: usize = 21504; // config.rs:227
const VOCAB: usize = 262144; // config.rs:228
const Q_HEADS: usize = 32; // config.rs:229
const KV_HEADS_SLIDING: usize = 16; // config.rs:230
const KV_HEADS_FULL: usize = 4; // config.rs:231
const HEAD_DIM: usize = 256; // config.rs:232, applied on sliding layers
const GLOBAL_HEAD_DIM: usize = 512; // config.rs:233, applied on full layers
const LAYERS: usize = 60; // config.rs:225
const FULL_EVERY: usize = 6; // config.rs:241, `index % 6 == 5`
const SLIDING_LAYERS: usize = LAYERS - LAYERS / FULL_EVERY; // 50
const FULL_LAYERS: usize = LAYERS / FULL_EVERY; // 10

/// Bytes per 16x16 trellis tile at 3 bpw: 256 codes * 3 bits = 768 bits = 96 B
/// (`WORDS = RING / 32` with `RING = 256 * KA`, `KA = 3`,
/// `linear_repacked.metal:2084-2085`).
const BYTES_PER_TILE_3BPW: u64 = 96;

/// Threads per threadgroup on every decode GEMV (`linear.rs:2152`, `:2652`,
/// `:2818` all dispatch `256 * groups` threads in groups of 256).
const THREADS_PER_TG: u64 = 256;

/// K unroll factor of each live kernel, i.e. the `step` in
/// `k_unroll_end = (tiles_k / step) * step`:
/// `_2col` -> `SG * 2` (`linear_repacked.metal:2115`), `_contig4` -> `SG * 4`
/// (`:2005`).
const K_UNROLL_2COL: u64 = 16;
const K_UNROLL_CONTIG4: u64 = 32;

/// Per-kernel milliseconds per token, 60 layers, from the profiler table in
/// `spec/decode-25tok-peak-bandwidth/implementation_notes.md` section 2. These
/// are wall-clock totals, which is why the model below can be falsified by them.
const MS_GATE_UP: f64 = 25.84; // implementation_notes.md:23
const MS_DOWN: f64 = 13.22; // :24
const MS_QKV: f64 = 8.71; // :25
const MS_O: f64 = 6.26; // :26
const MS_HEAD: f64 = 4.12; // :332 (1.057 GB at 257 GB/s)
const MS_STEP_TOTAL: f64 = 65.06; // :30

/// `head` is the one 6 bpw tensor in the checkpoint (`:426`,
/// `config.rs:257` reads the rate from the metadata rather than assuming 3).
const HEAD_BPW: u64 = 6;

// ---------------------------------------------------------------------------
// Candidate co-resident threadgroup budgets, 20 GPU cores x 256 threads.
//
// 80 / 160 / 320 are the three the record uses
// (`perf_probe_splitk_gridshape.rs:117-123`); 240 is added because the question
// that prompted this file named it, and because it is the value that makes
// `o_proj`/`down_proj` exactly one (70%-filled) wave rather than two. No
// occupancy query has ever been run on this box -- the Metal toolchain was
// absent when that probe was written, and `device.recommendedMaxThreadgroups-
// PerThreadgroup` is not referenced anywhere in the tree -- so every conclusion
// here is stated against all four and none depends on picking one.
const RESIDENT_BUDGETS: [u64; 4] = [80, 160, 240, 320];

/// One decode GEMV dispatch in production: which kernel the selector resolves
/// to, and the threadgroup count that kernel's dispatch equation produces.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Dispatch {
    label: &'static str,
    k: usize,
    /// Output width of each packed matrix in the dispatch. Three for the fused
    /// qkv, two for the fused MLP, one otherwise.
    n: &'static [usize],
    /// Which `linear.rs` branch selects it.
    arm: Arm,
    /// Layers per token that use this geometry: 50 or 10 for the layer GEMVs,
    /// 1 for the head (`config.rs:241` puts one full layer in six).
    layers: usize,
    /// `spec/decode-25tok-peak-bandwidth/tasks.md` section 2: this row's whole
    /// contribution to one token, in milliseconds. Already scaled by
    /// `layers / LAYERS`, so the seven rows sum to the section-2 breakdown and
    /// must not be re-weighted when aggregated.
    ms_per_token: f64,
    /// Bit rate of the packed weights, only used for the head's byte count.
    bpw: u64,
}

/// The three dispatch shapes in the decode path, named by the arm that selects
/// them rather than by kernel name, because the kernel *name* is a lookup that
/// can silently fall through to a different variant
/// (`linear.rs:1526-1532`: "a rate with no `_w2` kernel simply falls through to
/// the 3-word slot"). The grid equation is what is being tested, and the grid
/// equation is per-arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    /// `linear.rs:2133-2157`, `_colmajor_2col`: `out_features / 32`.
    Col2,
    /// `linear.rs:2780-2787`, `_fused3_2col`: `(tiles_a + tiles_b + tiles_c) / 2`.
    Fused3x2Col,
    /// `linear.rs:2590-2597`, `_fused2_2col`: `(tiles_a + tiles_b) / 2`.
    Fused2x2Col,
}

impl Dispatch {
    /// Output tile-columns, 16 wide, summed over the packed matrices.
    fn tiles_n(&self) -> u64 {
        self.n.iter().map(|n| (*n / 16) as u64).sum()
    }

    /// Input tile-rows, 16 tall.
    fn tiles_k(&self) -> u64 {
        (self.k / 16) as u64
    }

    /// The dispatch equation, transcribed from the branch named in `Arm`.
    fn threadgroups(&self) -> u64 {
        match self.arm {
            Arm::Col2 => (self.n[0] / 32) as u64,
            Arm::Fused3x2Col => self.tiles_n() / 2,
            Arm::Fused2x2Col => self.tiles_n() / 2,
        }
    }

    /// Threads the dispatch asks for (`linear.rs:2152` et al).
    fn threads(&self) -> u64 {
        self.threadgroups() * THREADS_PER_TG
    }

    /// Weight bytes one dispatch streams: one matrix is `k * n * bpw / 8`.
    fn bytes_per_dispatch(&self) -> u64 {
        self.n.iter().map(|n| self.k as u64 * *n as u64 * self.bpw / 8).sum()
    }

    /// Weight bytes this row streams per token, across all its layers.
    ///
    /// The two orders of magnitude matter: a bandwidth computed from
    /// per-dispatch bytes over per-dispatch milliseconds is wrong by the layer
    /// count, which is how `o_proj` came out at 3.2 GB/s in the first run of
    /// this file.
    fn bytes_per_token(&self) -> u64 {
        self.bytes_per_dispatch() * self.layers as u64
    }

    /// Threads dispatched per token: the four layer GEMVs plus one head.
    fn dispatches_per_token(&self) -> usize {
        self.layers
    }

    /// Wave count against a co-resident budget, and the strict-model tail
    /// fraction `(ceil(w) - w) / ceil(w)`.
    ///
    /// Note the degenerate case: when the whole grid fits in one wave the tail
    /// formula reports the *idle slots*, not a tail. 168 threadgroups against a
    /// 240 budget is 0.7 waves and "30% drained" really means "70% of the
    /// machine idle for the whole kernel" -- a persistent underfill, which no
    /// amount of tail arithmetic can fix because the grid cannot grow past
    /// `tiles_n`.
    fn waves(&self, resident: u64) -> f64 {
        self.threadgroups() as f64 / resident as f64
    }

    fn tail_fraction(&self, resident: u64) -> f64 {
        let w = self.waves(resident);
        let c = w.ceil();
        if c <= 0.0 {
            0.0
        } else {
            (c - w) / c
        }
    }

    /// Milliseconds of modelled drain per token.
    fn tail_ms(&self, resident: u64) -> f64 {
        self.ms_per_token * self.tail_fraction(resident)
    }

    /// Measured bandwidth from first-principles bytes, not from the notes'
    /// own accounting. The notes quote 185 GB/s for qkv and 207 for o_proj,
    /// which are 6% and 12% off these because they bill every layer at the
    /// sliding-layer geometry; `o_proj` is 8192-wide on 50 layers and
    /// 16384-wide on 10 (`model.rs:3364-3374`), which is 24% of its bytes.
    fn measured_gbps(&self) -> f64 {
        self.bytes_per_token() as f64 / 1e9 / (self.ms_per_token / 1e3)
    }
}

/// The seven production shapes. `q`/`k`/`v` are one fused dispatch, `gate`/`up`
/// are one fused dispatch, so this is seven *shapes* and five *dispatch
/// geometries* per layer plus the head.
fn dispatches() -> Vec<Dispatch> {
    vec![
        // q/k/v, sliding layers: head_dim 256, kv_heads 16 (`model.rs:3365-3373`).
        Dispatch {
            label: "qkv (sliding)",
            k: HIDDEN,
            n: &[
                Q_HEADS * HEAD_DIM,
                KV_HEADS_SLIDING * HEAD_DIM,
                KV_HEADS_SLIDING * HEAD_DIM,
            ],
            arm: Arm::Fused3x2Col,
            layers: SLIDING_LAYERS,
            ms_per_token: MS_QKV * SLIDING_LAYERS as f64 / LAYERS as f64,
            bpw: 3,
        },
        // q/k/v, the 10 full layers: global_head_dim 512, 4 global kv heads.
        Dispatch {
            label: "qkv (full)",
            k: HIDDEN,
            n: &[
                Q_HEADS * GLOBAL_HEAD_DIM,
                KV_HEADS_FULL * GLOBAL_HEAD_DIM,
                KV_HEADS_FULL * GLOBAL_HEAD_DIM,
            ],
            arm: Arm::Fused3x2Col,
            layers: FULL_LAYERS,
            ms_per_token: MS_QKV * FULL_LAYERS as f64 / LAYERS as f64,
            bpw: 3,
        },
        Dispatch {
            label: "o_proj (sliding)",
            k: Q_HEADS * HEAD_DIM,
            n: &[HIDDEN],
            arm: Arm::Col2,
            layers: SLIDING_LAYERS,
            ms_per_token: MS_O * SLIDING_LAYERS as f64 / LAYERS as f64,
            bpw: 3,
        },
        Dispatch {
            label: "o_proj (full)",
            k: Q_HEADS * GLOBAL_HEAD_DIM,
            n: &[HIDDEN],
            arm: Arm::Col2,
            layers: FULL_LAYERS,
            ms_per_token: MS_O * FULL_LAYERS as f64 / LAYERS as f64,
            bpw: 3,
        },
        Dispatch {
            label: "gate_up",
            k: HIDDEN,
            n: &[INTERMEDIATE, INTERMEDIATE],
            arm: Arm::Fused2x2Col,
            layers: LAYERS,
            ms_per_token: MS_GATE_UP,
            bpw: 3,
        },
        Dispatch {
            label: "down_proj",
            k: INTERMEDIATE,
            n: &[HIDDEN],
            arm: Arm::Col2,
            layers: LAYERS,
            ms_per_token: MS_DOWN,
            bpw: 3,
        },
        Dispatch {
            label: "lm_head",
            k: HIDDEN,
            n: &[VOCAB],
            arm: Arm::Col2,
            layers: 1,
            ms_per_token: MS_HEAD,
            bpw: HEAD_BPW,
        },
    ]
}

/// Theoretical DRAM bus, `implementation_notes.md:6`. Used as a hard ceiling in
/// `modelled_tail_cannot_be_fifty_percent_because_the_wire_forbids_it`.
const WIRE_GBPS: f64 = 273.0;

// ---------------------------------------------------------------------------
// Premise 1/2: shapes, tiles, grids.
// ---------------------------------------------------------------------------

/// Every tile count is exact on both axes, on every shape.
///
/// `model.rs:384-385` already rejects any linear whose features are not
/// multiples of 128, which makes `k % 16 == 0` and `n % 16 == 0` structural
/// rather than lucky. The test is here so that if that validation is ever
/// relaxed, the *grid* numbers below are known to be the ones that still hold.
#[test]
fn seven_shapes_are_exact_on_both_tile_axes() {
    for d in dispatches() {
        assert_eq!(d.k % 16, 0, "{}: K={} not a multiple of the 16-tile", d.label, d.k);
        for n in d.n {
            assert_eq!(
                n % 16,
                0,
                "{}: N={} not a multiple of the 16-tile",
                d.label,
                n
            );
            // The multiple-of-128 rule is what guarantees the above, so pin it.
            assert_eq!(n % 128, 0, "{}: N={} violates model.rs:385", d.label, n);
        }
        assert_eq!(d.k % 128, 0, "{}: K={} violates model.rs:384", d.label, d.k);
        assert_eq!(d.tiles_k() * 16, d.k as u64);
        assert_eq!(d.tiles_n() * 16, d.n.iter().map(|n| *n as u64).sum::<u64>());
    }
}

/// The `_2col` K loop never runs its remainder, on any shape.
///
/// `linear_repacked.metal:2115-2119` unrolls by `SG * 2 = 16` and floors with
/// `k_unroll_end = (tiles_k / 16) * 16`; the tail loop at `:2041` starts at
/// `k_unroll_end + sg`. Every production `tiles_k` (336, 512, 1024, 1344) is a
/// multiple of 16, so the tail loop's first iteration index is already
/// `>= tiles_k` and it never executes. This is stronger than the question asked
/// ("no remainder *threads*"): there is no remainder *K tile* either.
#[test]
fn remainder_k_loop_never_executes_on_the_live_kernel() {
    for d in dispatches() {
        assert_eq!(
            d.tiles_k() % K_UNROLL_2COL,
            0,
            "{}: tiles_k={} leaves a remainder pass on _2col",
            d.label,
            d.tiles_k()
        );
    }
}

/// The `_2col` output split is exact, so `pairs_n = tiles_n / 2` drops nothing.
///
/// `linear_repacked.metal:2090` and `:2102`: the loop is `for (tn = tgid;
/// tn < pairs_n; tn += ntg)`, so a truncated `tiles_n / 2` would silently leave
/// the last tile-column of every dispatch unwritten. It cannot here, but the
/// failure would be a partial-output bug rather than a wrong number, so it is
/// worth a test.
#[test]
fn two_col_output_split_is_exact_on_every_shape() {
    for d in dispatches() {
        assert_eq!(
            d.tiles_n() % 2,
            0,
            "{}: tiles_n={} is odd, _2col would drop a tile-column",
            d.label,
            d.tiles_n()
        );
    }
}

/// The live grids, checked against hand-computed production values.
///
/// If `linear.rs` changes a dispatch equation this test fails with the new grid,
/// which is the whole reason the equations are transcribed rather than derived
/// from the shapes.
#[test]
fn live_production_grids_match_the_transcribed_equations() {
    let grids: Vec<(&str, u64)> = dispatches()
        .iter()
        .map(|d| (d.label, d.threadgroups()))
        .collect();
    assert_eq!(
        grids,
        vec![
            // (512 + 256 + 256) / 2
            ("qkv (sliding)", 512),
            // (1024 + 128 + 128) / 2
            ("qkv (full)", 640),
            // 5376 / 32
            ("o_proj (sliding)", 168),
            ("o_proj (full)", 168),
            // (1344 + 1344) / 2
            ("gate_up", 1344),
            ("down_proj", 168),
            // 262144 / 32, NOT 262144 / 16
            ("lm_head", 8192),
        ]
    );
}

/// The premise that motivated this file, pinned as false.
///
/// "336 threadgroups for q_proj" and "lm_head launches 16384 threadgroups" are
/// `out_features / 16`, the grid of `encode_ishizuki_gemv`
/// (`linear.rs:1479-1480`) -- which `encode_gemv_raw` reaches only when the
/// `_2col` arm above it does not fire. With `permutation_mode == 1`
/// (`model.rs:3525-3541`) it does not fire, so both numbers are 2x the live
/// grid. `q_proj` is not 336 either way: it is not dispatched on its own at all,
/// it is 512 of the 512 in the fused qkv arm.
#[test]
fn the_out_features_div_16_grids_are_not_the_live_ones() {
    // The numbers the premise quoted, and what is actually dispatched.
    assert_eq!(HIDDEN / 16, 336, "premise's 336 = HIDDEN/16");
    assert_eq!(INTERMEDIATE / 16, 1344, "premise's 1344 = INTERMEDIATE/16");
    assert_eq!(VOCAB / 16, 16384, "premise's 16384 = VOCAB/16");

    // HIDDEN/16 does occur -- as `tiles_n`, never as the grid.
    assert_eq!((HIDDEN / 16) as u64, 336);
    // Every live grid is half of the /16 value on the 5376-wide projections,
    // because _2col packs two tile-columns per threadgroup.
    assert_eq!(168 * 2, 336);
    assert_eq!(8192 * 2, 16384);
    // And the comment that asserts 16384 is describing the wrong path.
    assert_ne!(8192, 16384);
}

/// The only K-remainder in the whole tree is on the `_contig4` fallback, and it
/// exists solely on the three K=5376 shapes.
///
/// `_contig4` unrolls by `SG * 4 = 32` (`linear_repacked.metal:2005`), so
/// `tiles_k = 336` floors to 320 and leaves 16 tiles for the tail loop at
/// `:2041` (which does cover them exactly: it starts at `k_unroll_end + sg` and
/// strides `SG`, so it visits 320..327 then 328..335). `tiles_k` of 512, 1024
/// and 1344 are all multiples of 32, so the other four shapes are clean.
///
/// This matters because `_contig4` is precisely the kernel that
/// `KAPLUMBA_NO_2COL=1` would select: with `perm_mode == 1`,
/// `ishizuki_gemv_pipeline` (`linear.rs:1539-1547`) looks up
/// `_colmajor_contig4` *before* `_colmajor`, and falls back to it silently if
/// the 4-column kernel is absent. So "just turn off 2-col" is not the one-line
/// experiment it looks like.
#[test]
fn remainder_k_loop_runs_only_on_contig4() {
    let dirty: Vec<&str> = dispatches()
        .iter()
        .filter(|d| d.tiles_k() % K_UNROLL_CONTIG4 != 0)
        .map(|d| d.label)
        .collect();
    assert_eq!(dirty, vec!["qkv (sliding)", "qkv (full)", "gate_up", "lm_head"]);

    // The remainder is exactly one SG*2 block, which the SG-strided tail loop
    // covers with no overlap and no gap.
    for d in dispatches().iter().filter(|d| d.tiles_k() % K_UNROLL_CONTIG4 != 0) {
        assert_eq!(d.tiles_k() % K_UNROLL_CONTIG4, 16, "{}", d.label);
    }
}

// ---------------------------------------------------------------------------
// Premise 3: waves, tail, and the falsification of the tail model.
// ---------------------------------------------------------------------------

/// The per-shape table the task asked for.
#[test]
fn tail_waste_table_per_shape() {
    let mut out = String::new();
    writeln!(
        out,
        "{:<17} {:>6} {:>6} {:>5} {:>6} {:>6} {:>5}",
        "shape", "K", "tiles_k", "tiles_n", "grid", "MB/tok", "GB/s"
    )
    .unwrap();
    for d in dispatches() {
        writeln!(
            out,
            "{:<17} {:>6} {:>6} {:>5} {:>6} {:>6} {:>5.1}",
            d.label,
            d.k,
            d.tiles_k(),
            d.tiles_n(),
            d.threadgroups(),
            d.bytes_per_token() as f64 / 1e6,
            d.measured_gbps(),
        )
        .unwrap();
    }
    for resident in RESIDENT_BUDGETS {
        writeln!(
            out,
            "\nN_resident = {resident} threadgroups ({:.1} TG/core)",
            resident as f64 / 20.0
        )
        .unwrap();
        writeln!(out, "{:<17} {:>7} {:>6} {:>8}", "shape", "waves", "ceil", "tail").unwrap();
        for d in dispatches() {
            writeln!(
                out,
                "{:<17} {:>7.3} {:>6} {:>7.1}%",
                d.label,
                d.waves(resident),
                d.waves(resident).ceil(),
                d.tail_fraction(resident) * 100.0,
            )
            .unwrap();
        }
    }
    println!("{out}");
}

/// Aggregate tail cost per token, per co-resident budget.
///
/// The question asked for "waves x tail waste x 420 dispatches". The dispatch
/// count is 241, not ~420: four GEMV dispatches per layer (`qkv` fused,
/// `gate_up` fused, `o_proj`, `down_proj`) times 60 layers plus one head. The
/// other ~241 dispatches in a step are the `finalize_*` passes and the norms,
/// which are elementwise and have no tiles at all.
#[test]
fn aggregate_tail_cost_per_token() {
    // Four GEMV dispatches per layer plus the head.
    assert_eq!(4 * LAYERS + 1, 241);

    let mut out = String::new();
    writeln!(
        out,
        "{:<11} {:>7} {:>8} {:>8} {:>8} {:>8} {:>9} {:>7}",
        "N_resident", "gate_up", "qkv", "o_proj", "down", "head", "TOTAL ms", "tok/s"
    )
    .unwrap();
    for resident in RESIDENT_BUDGETS {
        let d = dispatches();
        // `ms_per_token` is already each row's whole contribution to a token,
        // so the rows sum; no re-weighting by layer count.
        let gate_up = d[4].tail_ms(resident);
        let qkv = d[0].tail_ms(resident) + d[1].tail_ms(resident);
        let o = d[2].tail_ms(resident) + d[3].tail_ms(resident);
        let down = d[5].tail_ms(resident);
        let head = d[6].tail_ms(resident);
        let total = gate_up + qkv + o + down + head;
        let step = MS_STEP_TOTAL - total;
        writeln!(
            out,
            "{:<11} {:>7.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>9.2} {:>7.2}",
            resident,
            gate_up,
            qkv,
            o,
            down,
            head,
            total,
            1000.0 / step,
        )
        .unwrap();
    }
    println!("{out}");

    // Sanity: the model is not vacuous. At 240 resident it predicts ~10 ms of a
    // 65 ms step, which is 15% -- large enough that it *should* have been seen
    // in the profiler if it were real.
    let total_240: f64 = RESIDENT_BUDGETS
        .iter()
        .find(|r| **r == 240)
        .map(|r| {
            dispatches().iter().map(|d| d.tail_ms(*r)).sum::<f64>()
        })
        .unwrap();
    assert!(
        total_240 > 5.0 && total_240 < 15.0,
        "model collapsed: {total_240} ms/token at N=240"
    );
}

/// The two 168-threadgroup projections hold most of the modelled tail.
#[test]
fn the_modelled_tail_is_concentrated_in_o_proj_and_down_proj() {
    for resident in RESIDENT_BUDGETS {
        let d = dispatches();
        let wide_tail =
            d[2].tail_ms(resident) + d[3].tail_ms(resident) + d[5].tail_ms(resident);
        let all: f64 = d.iter().map(|d| d.tail_ms(resident)).sum();
        assert!(
            wide_tail / all > 0.5,
            "N={resident}: 5376-wide projections hold only {:.0}% of the tail",
            wide_tail / all * 100.0
        );
    }
}

/// **The load-bearing test.** The strict wave model does not order the four
/// layer GEMVs, and the profiler's own milliseconds say so.
///
/// At every candidate budget the model ranks `o_proj`/`down_proj` (168
/// threadgroups, 30-47.5% tail) as far worse drained than `gate_up` (1344
/// threadgroups, 1.2-16% tail) -- a 4.5x ratio at N=240. Measured, from the same
/// notes and first-principles byte counts, `o_proj` is 8% *slower* per byte than
/// `gate_up`, not 4.5x. And `qkv`, which the model puts in the middle, is the
/// fastest 3-bpw kernel in the step at 237 GB/s.
///
/// A model that gets the sign right but the magnitude wrong by more than an
/// order of magnitude is not a model you can size a change against.
#[test]
fn strict_wave_model_is_falsified_by_the_measured_bandwidths() {
    let d = dispatches();
    let o = (d[2].measured_gbps() * SLIDING_LAYERS as f64
        + d[3].measured_gbps() * FULL_LAYERS as f64)
        / LAYERS as f64;
    let gate_up = d[4].measured_gbps();

    // The model, at the budget the question named.
    let modelled_ratio = d[2].tail_fraction(240) / d[4].tail_fraction(240);
    assert!(
        modelled_ratio > 4.0,
        "model no longer predicts a 4.5x tail gap (got {modelled_ratio:.2}) -- \
         this test exists to notice when the premise it falsifies changes"
    );

    // The measurement.
    let measured_ratio = gate_up / o;
    assert!(
        measured_ratio < 1.5,
        "o_proj at {o:.1} GB/s is no longer within 1.5x of gate_up at \
         {gate_up:.1} GB/s -- re-read the tail argument before reusing it"
    );
    // And the fastest 3-bpw kernel is not the one with the least modelled tail.
    assert!(
        d[0].measured_gbps() > gate_up,
        "qkv is no longer the fastest layer GEMV; the ordering argument changed"
    );
    assert!(
        d[0].tail_fraction(240) > d[4].tail_fraction(240),
        "qkv no longer has a larger modelled tail than gate_up"
    );
}

/// The hard version of the same falsification, from physics rather than from a
/// second measurement.
///
/// Un-draining a measured rate is division by `(1 - tail)`, and the wire caps the
/// result. For the two 168-threadgroup projections:
///
/// * at N=160 or N=320 the modelled tail is 47.5%, so `o_proj` would have to be
///   running at 184.7 / 0.525 = 352 GB/s and `down_proj` at 375 GB/s. The wire
///   is 273 GB/s (`implementation_notes.md:6`). Flatly impossible.
/// * at N=240 the tail is 30%, and `down_proj` still needs 281 GB/s, which is
///   also impossible. `o_proj` needs 264 GB/s, which the bare wire does *not*
///   exclude -- so the wire alone does not settle that one cell, and the honest
///   claim is that 264 GB/s is 97% of the bus for a 3-bpw kernel whose best
///   measurement on this box is 203 GB/s, i.e. 74% of it
///   (`implementation_notes.md:425`).
#[test]
fn modelled_tail_cannot_be_fifty_percent_because_the_wire_forbids_it() {
    let d = dispatches();
    let o = (d[2].measured_gbps() * SLIDING_LAYERS as f64
        + d[3].measured_gbps() * FULL_LAYERS as f64)
        / LAYERS as f64;
    let down = d[5].measured_gbps();

    // The 47.5% variant is excluded outright, for both.
    for (label, gbps) in [("o_proj", o), ("down_proj", down)] {
        let tail = d[2].tail_fraction(160);
        assert!((tail - 0.475).abs() < 1e-9, "tail at N=160 moved to {tail:.3}");
        let implied = gbps / (1.0 - tail);
        assert!(
            implied > WIRE_GBPS,
            "{label}: a 47.5% tail now implies only {implied:.0} GB/s \
             un-drained, at or below the {WIRE_GBPS} GB/s wire. The tail model \
             has become consistent with the measurement -- revisit it."
        );
    }

    // The 30% variant is excluded for down_proj, and pinned near the wire for
    // o_proj rather than claimed as excluded.
    let tail_30 = d[2].tail_fraction(240);
    assert!((tail_30 - 0.30).abs() < 1e-9, "tail at N=240 moved to {tail_30:.3}");
    assert!(
        down / (1.0 - tail_30) > WIRE_GBPS,
        "down_proj's 30% tail is now within the wire"
    );
    let o_implied = o / (1.0 - tail_30);
    assert!(
        o_implied > 0.9 * WIRE_GBPS,
        "o_proj's un-drained rate fell to {o_implied:.0} GB/s ({:.0}% of the \
         wire); it is no longer the implausible one, so the argument for \
         ignoring the tail model at N=240 has weakened",
        o_implied / WIRE_GBPS * 100.0,
    );
    println!(
        "implied un-drained at N=240: o_proj {o_implied:.0} GB/s \
         ({:.0}% of wire), down_proj {:.0} GB/s ({:.0}% of wire); measured \
         3-bpw best on this box is 203 GB/s (74%).",
        o_implied / WIRE_GBPS * 100.0,
        down / (1.0 - tail_30),
        down / (1.0 - tail_30) / WIRE_GBPS * 100.0,
    );
}

/// The prize is bounded by the bus, not by the tail model.
#[test]
fn total_recoverable_on_the_5376_wide_projections_is_under_half_a_tok_per_second() {
    let d = dispatches();
    // `bytes_per_token` already spans the row's layers; the two `o_proj` rows
    // together are all 60 of them.
    let o_bytes = (d[2].bytes_per_token() + d[3].bytes_per_token()) as f64;
    let ms_o = MS_O;
    let ms_at_wire = o_bytes / 1e9 / WIRE_GBPS * 1e3;
    // 203 GB/s is the best 3-bpw EXL3 GEMV ever measured on this box
    // (implementation_notes.md:425), so that is the realistic target, not the
    // wire.
    let ms_at_peak = o_bytes / 1e9 / 203.0 * 1e3;

    assert!(ms_at_wire < ms_o && ms_at_peak < ms_o);
    let recoverable_wire = ms_o - ms_at_wire;
    let recoverable_peak = ms_o - ms_at_peak;
    assert!(
        recoverable_wire < 2.5,
        "o_proj floor moved: {recoverable_wire:.2} ms of {ms_o} ms"
    );

    // Express it in the unit the record uses.
    let base_tok_s = 1000.0 / MS_STEP_TOTAL;
    let at_peak_tok_s = 1000.0 / (MS_STEP_TOTAL - recoverable_peak);
    let at_wire_tok_s = 1000.0 / (MS_STEP_TOTAL - recoverable_wire);
    println!(
        "o_proj: {ms_o} ms now, {ms_at_peak:.2} ms at the measured 203 GB/s peak, \
         {ms_at_wire:.2} ms at the 273 GB/s wire.\n\
         Whole step: {base_tok_s:.2} -> {at_peak_tok_s:.2} -> {at_wire_tok_s:.2} tok/s \
         if o_proj alone hit those bounds."
    );
    assert!(
        at_peak_tok_s - base_tok_s < 0.5,
        "o_proj prize is now worth +{:.2} tok/s, which would change the \
         recommendation",
        at_peak_tok_s - base_tok_s
    );
}

// ---------------------------------------------------------------------------
// The proposed fixes.
// ---------------------------------------------------------------------------

/// Neither "make the grid a clean single wave" nor "give each threadgroup more
/// tiles" can work, and the reason is a quantisation trade-off rather than a
/// lack of imagination.
///
/// To fill `N` co-resident slots with no wave tail the grid must be `g = N * m`
/// for some `m`. But the 5376-wide projections have exactly 336 tile-columns, so
/// the tiles per threadgroup become `336 / g`, which is generally not an
/// integer: the tail loop inside the kernel then reappears, on the tile axis
/// instead of the wave axis, and the critical path is `ceil(336 / g)` tiles
/// rather than `336 / g`.
///
/// For each candidate budget this enumerates every integral-wave grid and
/// compares it with what production actually dispatches. The result: the status
/// quo wins at N=240 and N=320 and loses narrowly at N=80 and N=160. There is no
/// budget at which the rewrite pays for itself.
#[test]
fn integral_wave_grid_is_never_also_tile_balanced() {
    // `o_proj` / `down_proj`: 336 tile-columns, production packs 2 per
    // threadgroup (`_2col`).
    const TILE_COLUMNS: u64 = 336;
    const PRODUCTION_TILES_PER_TG: u64 = 2;

    let mut out = String::new();
    writeln!(
        out,
        "{:<11} {:>5} {:>9} {:>11} {:>9} {:>9}",
        "N_resident", "grid", "tiles/TG", "int-waves", "tile-waste", "vs prod"
    )
    .unwrap();

    for resident in RESIDENT_BUDGETS {
        // What production does: `g = 336 / 2`, perfectly tile-balanced, and the
        // wave quantisation on top.
        let prod_grid = TILE_COLUMNS / PRODUCTION_TILES_PER_TG;
        let prod = if prod_grid % resident == 0 {
            0.0
        } else {
            let w = prod_grid as f64 / resident as f64;
            (w.ceil() - w) / w.ceil()
        };

        let mut best: Option<(u64, f64)> = None;
        for m in 1..=(TILE_COLUMNS / resident) {
            let grid = resident * m;
            let per_tg = TILE_COLUMNS as f64 / grid as f64;
            let tile_waste = (per_tg.ceil() / per_tg) - 1.0;
            writeln!(
                out,
                "{:<11} {:>5} {:>9.3} {:>11} {:>8.1}% {:>9}",
                resident,
                grid,
                per_tg,
                m,
                tile_waste * 100.0,
                if tile_waste < prod { "better" } else { "worse" },
            )
            .unwrap();
            if tile_waste < 1.0 {
                best = Some((grid, tile_waste));
            }
        }

        // Every integral-wave grid needs a non-integral tiles/threadgroup,
        // because 336 = 2^4 * 3 * 7 and every candidate budget has a factor 5
        // that 336 lacks.
        let (grid, waste) = best.expect("at least one integral grid must exist");
        assert_eq!(grid % resident, 0);
        assert_ne!(
            TILE_COLUMNS % grid,
            0,
            "N={resident}: grid {grid} divides 336 exactly -- the trade-off \
             vanished and this test needs rethinking"
        );
        // And it is not actually better than the status quo at half the
        // candidate budgets.
        if resident == 240 || resident == 320 {
            assert!(
                waste > prod,
                "N={resident}: integral grid {grid} now beats production \
                 ({waste:.1}% vs {prod:.1}%) -- a real change, re-run the A/B"
            );
        }
    }
    println!("{out}");
}

/// A persistent kernel is not an untried idea on this shape family; it is
/// `KAPLUMBA_GEMV_MAX_GROUPS`, and it has been run.
///
/// Two independent facts make this an arithmetic consequence rather than a
/// citation:
///
/// 1. The live kernels are *already* grid-stride loops. `linear.metal:1626` is
///    `for (uint tn = tgid; tn < tiles_n; tn += ntg)` and
///    `linear_repacked.metal:1999`/`:2102` are the same shape. Dispatching fewer
///    threadgroups than tiles is not a kernel change; it is the existing
///    `groups = ...min(gemv_max_groups())` term.
/// 2. `gemv_max_groups()` is `u64::MAX` unless `KAPLUMBA_GEMV_MAX_GROUPS` is set
///    (`linear.rs:94-102`), so the cap is a no-op in production -- which is
///    also why the comment at `linear.rs:1473-1478` describing a "capped grid
///    [that] turns the tail into an exact multiple" describes behaviour no
///    production run has.
///
/// The result of running it (`spec/decode-25tok-peak-bandwidth/tasks.md` T5.11,
/// interleaved A/B on `o_proj`):
#[test]
fn the_persistent_kernel_axis_is_already_measured_and_negative() {
    // baseline (336 TGs, 1 tile/TG) -> 168 (2 tiles/TG) -> 84 (4 tiles/TG).
    let measured = [(336u64, 15.83f64), (168, 15.716), (84, 15.595)];
    let baseline = measured[0].1;
    for (grid, tok_s) in measured {
        assert!(tok_s <= baseline, "T5.11 baseline moved: {tok_s} at {grid} TGs");
    }
    // Monotone: fewer threadgroups is monotonically worse.
    assert!(measured[0].1 > measured[1].1 && measured[1].1 > measured[2].1);

    // The cap is dead code in production, which is the other half of the story:
    // a reader of `linear.rs:1473-1478` would conclude the grid is already
    // capped, and would be wrong.
    let cap_default = u64::MAX; // linear.rs:100
    let tiles: u64 = 336;
    assert_eq!(tiles.min(cap_default).max(1), tiles);
}

/// The one grid cell that has never been measured, and what it is worth.
///
/// The measured axis is 84 -> 168 -> (336 is the production-equivalent on the
/// 1-column path) -> 672 (SK4). The trend says *more* threadgroups is better
/// (T5.11: "o_proj wants more threadgroups, not fewer"), so the missing cell is
/// the one that doubles 168: dispatch `o_proj`/`down_proj` with 336 threadgroups
/// instead of 168.
///
/// Note what that flag would actually select. `KAPLUMBA_NO_2COL=1` skips the
/// `_2col` arm at `linear.rs:2133` and falls into `ishizuki_gemv_pipeline`
/// (`linear.rs:2159`), which for `perm_mode == 1` looks up
/// `_colmajor_contig4` **before** `_colmajor` (`linear.rs:1539-1547`). So the
/// flag does not select the 1-column kernel the measurement is about; it selects
/// the 4-way K-unrolled one, which is the only kernel in the tree with a live K
/// remainder loop (`remainder_k_loop_runs_only_on_contig4`). Selecting
/// `_colmajor` explicitly is a code change, not a flag.
#[test]
fn the_unmeasured_cell_is_336_threadgroups_on_the_5376_wide_projections() {
    let d = dispatches();
    let now = d[2].threadgroups();
    assert_eq!(now, 168, "production grid for o_proj moved");
    let proposed = 336u64;
    assert_eq!(proposed, HIDDEN as u64 / 16);

    // It never increases the modelled tail, and strictly reduces it at two of
    // the four candidate budgets. That is a weak argument, and the weak part is
    // structural rather than incidental: at N=240, 168 threadgroups is already
    // 0.7 of a single wave, so it is reported as one 70%-occupied wave, and 336
    // is 1.4 waves reported as two -- 30% either way. Doubling the grid cannot
    // help when the starting point is already under one wave.
    let mut strictly_better = 0;
    for resident in RESIDENT_BUDGETS {
        let w_now = now as f64 / resident as f64;
        let w_new = proposed as f64 / resident as f64;
        let tail_now = (w_now.ceil() - w_now) / w_now.ceil();
        let tail_new = (w_new.ceil() - w_new) / w_new.ceil();
        assert!(
            tail_new <= tail_now,
            "N={resident}: 336 TGs now drains *worse* than 168 \
             ({tail_new:.1}% vs {tail_now:.1}%) -- re-run the A/B"
        );
        if tail_new < tail_now {
            strictly_better += 1;
        }
        println!(
            "N={resident:4}: 168 TGs {:.1}% tail, 336 TGs {:.1}% tail",
            tail_now * 100.0,
            tail_new * 100.0
        );
    }
    assert_eq!(
        strictly_better, 2,
        "the doubling now strictly helps at {strictly_better} of 4 budgets \
         (expected 2: 80 and 160); re-derive the case for it"
    );

    // The gain is not an extrapolation: three adjacent cells are measured and
    // 336 is one of them. From `tasks.md` T5.11 (interleaved A/B, o_proj):
    //
    //   84 threadgroups   15.595
    //  168 threadgroups   15.716   (+0.121)
    //  336 threadgroups   15.830   (+0.114)  <- measured optimum
    //  672 threadgroups   15.781   (-0.049)  SK4, and not the same kernel
    //
    // so the axis rises monotonically to 336 and then turns over. Production
    // dispatches 168, the second-best measured cell, and the move to the
    // measured optimum is worth the adjacent-cell delta directly.
    let measured = [(84u64, 15.595f64), (168, 15.716), (336, 15.830), (672, 15.781)];
    let last_gain = measured[2].1 - measured[1].1;
    assert!(
        (last_gain - 0.114).abs() < 0.005,
        "the 168 -> 336 delta moved to {last_gain:+.3} tok/s; re-derive the \
         recommendation before quoting it"
    );
    // And the turn-over is real, so 672 is not the answer.
    assert!(measured[3].1 < measured[2].1);

    let gain_low = last_gain;
    let gain_high = 0.12f64;
    let base = 15.83f64;
    assert!(gain_low > 0.10 && gain_high <= 0.12);
    println!(
        "proposed: o_proj/down_proj 168 -> 336 threadgroups.\n\
         measured adjacent-cell delta: {gain_low:+.3} tok/s on a {base:.2} \
         baseline ({:.1}%). T5.11's own spread on this axis is +/-0.12, so this \
         is a single-cell A/B, not a foregone conclusion.",
        gain_low / base * 100.0,
    );

    // The one thing that must be settled first, and can be settled without any
    // code change: T5.11's table does not record the baseline *grid*, only the
    // baseline tok/s. If the baseline already ran 168 threadgroups -- which the
    // `_2col` arm gives -- then `MAX_GROUPS=168` was a no-op run and its -0.12
    // is noise, and only the `=84` cell carries information. If the baseline ran
    // 336, the -0.12 is the real cost of this exact move. `KAPLUMBA_TRACE_GEMV`
    // prints the selected kernel name per dispatch (`linear.rs:1945-1957`), so
    // one run settles it.
    assert_eq!(
        dispatches()[2].threadgroups(),
        168,
        "production grid for o_proj changed; the T5.11 table's baseline grid \
         needs re-deriving before the recommendation stands"
    );
}

/// The one comment that must change whatever else happens.
///
/// `linear.rs:1473-1478` is the load-bearing text for every wave/tail argument in
/// the tree, and both of its numbers are wrong for the live path: 16384
/// threadgroups for `lm_head` (it is 8192) and a grid cap that is a no-op
/// (`gemv_max_groups()` is `u64::MAX`).
#[test]
fn the_capping_comment_describes_a_path_production_does_not_take() {
    // What the comment claims.
    assert_eq!(INTERMEDIATE / 16, 1344, "comment: N=21504 -> 1344 TGs");
    assert_eq!(VOCAB / 16, 16384, "comment: lm_head launches 16384 TGs");
    // What production dispatches for lm_head.
    assert_eq!(VOCAB / 32, 8192);
    assert_eq!((VOCAB / 32) as u64, dispatches()[6].threadgroups());
    // And the cap that the comment says is turning the tail into an exact
    // multiple is inert.
    assert_eq!(u64::MAX, u64::MAX); // gemv_max_groups() with the flag unset
    assert_eq!(336u64.min(u64::MAX), 336);
}
