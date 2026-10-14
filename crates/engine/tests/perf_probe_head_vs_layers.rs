//! Why is `lm_head` 2x the bandwidth of every layer GEMV?
//!
//! # The claim under test
//!
//! Measured in production (interleaved A/B ablation, `spec/decode-25tok-peak-bandwidth/`
//! implementation_notes.md 18.2):
//!
//! | projection | GB/token | ms/step | implied GB/s |
//! |---|---|---|---|
//! | gate_up | 5.202 | 40.34 | 129 |
//! | down | 2.601 | 20.63 | 126 |
//! | qkv | 1.982 | 15.82 | 125 |
//! | o_proj | 0.991 | 6.63 | 150 |
//! | **head** | **1.057** | **4.17** | **253** |
//!
//! The head is the only kernel anywhere near the 273 GB/s wire. Four candidate
//! explanations, and this file exists to separate them rather than to pick one:
//!
//! * **(a) parallelism** -- the head launches 16384 threadgroups (N = 262144);
//!   every layer projection launches 336-1344 (N = 5376-21504).
//! * **(b) contiguity** -- the head is row-major `[tiles_k, tiles_n, 96]`; every
//!   layer tensor is colmajor `[tiles_n, tiles_k, 48]` after `repack_model.rs`.
//! * **(c) K is large** -- the microbenchmark in `gemv_bandwidth.rs` reported
//!   efficiency monotone in K.
//! * **(d) it reads more bytes per code** -- the head is 6 bpw, the layers 3 bpw,
//!   and `decode_state_mcg` costs the same per code either way.
//!
//! Arithmetic already favours (d): head codes/s = 1.409e9 / 4.17e-3 = 338 G/s,
//! layer codes/s = 1.387e10 / (40.34+20.63+15.82+6.63)e-3 = 344 G/s. Same code
//! rate, 2x the bytes, 2x the bandwidth. But arithmetic built on somebody else's
//! numbers is a hypothesis, and (a) and (b) have each looked plausible in this
//! repo before, so each of the four gets its own experiment below.
//!
//! # What is measured
//!
//! `head_vs_layers_run_the_production_dispatch` runs each shape through the
//! **exact entry point `model.rs` calls**, on the checkpoint's own bytes, with
//! the grid that entry point builds. The three controls then move one variable
//! each:
//!
//! * **Control R (`head_shape_at_three_bpw`)** -- the head's shape, the head's
//!   kernel family, the head's grid, the head's row-major layout, at **3 bpw**.
//!   Only the bit rate differs. If bandwidth halves and codes/s is unchanged,
//!   (d) is the whole story and (a)/(b)/(c) carry no weight.
//! * **Control G (`head_grid_thickness_sweep`)** -- the head's shape at 6 bpw
//!   with the grid forced from 16384 groups down to 336, i.e. the thinnest grid
//!   in the model. `KAPLUMBA_GEMV_MAX_GROUPS` is read through a `OnceLock`, so
//!   this re-execs the test binary.
//! * **Control S (`layer_shape_at_six_bpw`)** -- `down_proj`'s shape read at
//!   6 bpw. If head-like GB/s appears at a layer shape, the advantage is the
//!   shape and (d) is wrong.
//!
//! # Correctness is not what this tests
//!
//! The controls deliberately pair kernels with weights they were not written
//! for: a 3 bpw tensor read by `_k6`, a colmajor tensor read by the row-major
//! `_k6`. That is the same trade `gemv_bandwidth.rs` makes and for the same
//! reason -- the measurement is bytes moved per second, and the bytes are real.
//! What must not happen is a *plausible* number from a kernel that read nothing,
//! so `assert_live` refuses to report a bandwidth for an all-zero or non-finite
//! output. Correctness is `linear_metal.rs` and `live_kernel_bit_exact.rs`.
//!
//! # Running it
//!
//! ```text
//! KAPLUMBA_TEST_MODEL=~/.kaplumba/models/Gxma-repacked \
//!   cargo test --release --test perf_probe_head_vs_layers -- --ignored --nocapture
//! ```
//!
//! Reads ~1.25 GB of real weights (the whole 1.06 GB head plus one sliding
//! layer) and needs the GPU to itself. **It must not run while the model is
//! loaded** -- AGENTS.md 10.3 records two instances on this box costing 3.9 GiB
//! of swap and 1.4 tok/s. The two arithmetic tests at the top need neither, and
//! run in a plain `cargo test`.

use std::ffi::c_void;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

use kaplumba_engine::bench::{Layout, SHAPES, Shape};
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{LinearEncodeRequest, LinearKernels};
use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};

const CHILD_GROUPS: &str = "KAPLUMBA_PERF_PROBE_GROUPS";
const LAYER0: &str = "model.language_model.layers.0.";

fn shape(name: &str) -> Shape {
    *SHAPES
        .iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("SHAPES has no {name}"))
}

fn layout_name(s: &Shape) -> &'static str {
    match s.layout {
        Layout::RowMajor => "row-major",
        Layout::Colmajor => "colmajor",
    }
}

// ---------------------------------------------------------------------------
// 1. The dispatch arithmetic, as pure functions.
//
// Written out rather than read off the running engine so that a change to the
// engine shows up here as a failing number instead of as a silently different
// benchmark. `perf_probe_head_dispatch_arithmetic` is what holds them to it.
// ---------------------------------------------------------------------------

/// `u16` of packed trellis data in one 16x16 tile: `16 * bits`. Mirrors
/// `Shape::words_per_tile` in `crates/engine/src/bench.rs`, which is what the
/// byte accounting below actually uses; this spelling is here so the tile
/// arithmetic reads the same as `RING/WORDS` in the kernels.
#[allow(dead_code)]
const fn words_per_tile(bits: u8) -> usize {
    16 * bits as usize
}

/// `linear_gemv_ishizuki_k{bits}` grid (`encode_ishizuki_gemv`, linear.rs:1479):
/// one threadgroup per output tile, grid-strided. Also what `_contig4` and the
/// `_1col` names get, because `encode_ishizuki_gemv_named` falls through to the
/// same function for them (linear.rs:1415-1426).
const fn groups_1col(out_features: usize) -> u64 {
    (out_features / 16) as u64
}

/// `linear_gemv_ishizuki_k{bits}_colmajor_2col` grid (linear.rs:2142-2143) and
/// `encode_gemv_raw_sk2`'s (linear.rs:2258): two output tiles per threadgroup,
/// so `N/32` groups, doubled again by the split-K factor.
const fn groups_2col(out_features: usize, splits: u64) -> u64 {
    ((out_features / 16) / 2) as u64 * splits
}

/// `fused2_2col` grid (linear.rs:2597): gate and up concatenated, two output
/// tiles per threadgroup, no split-K.
const fn groups_fused2_2col(out_a: usize, out_b: usize) -> u64 {
    ((out_a / 16 + out_b / 16) / 2) as u64
}

/// `fused3_2col` grid (linear.rs:2787): q, k and v concatenated.
const fn groups_fused3_2col(out_a: usize, out_b: usize, out_c: usize) -> u64 {
    ((out_a / 16 + out_b / 16 + out_c / 16) / 2) as u64
}

/// Threadgroups resident device-wide, from the measured one-wave point for a
/// 256-thread group (`decode_one_wave_workgroups`, crates/kernels/src/attention.rs:324:
/// 54 is one wave, 55 is two). Only used to print waves-per-dispatch; nothing
/// asserts on it, because 54 is bracketed for the attention kernel's shape and a
/// 256-thread group with a different `red` footprint may pack differently.
const ONE_WAVE_256: u64 = 54;

/// Which production entry point a row is dispatched through.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Route {
    /// `model.rs:4420` -> `linear.rs:1960` (`encode_gemv`). Head only: the fused
    /// and split-K arms all require `rate == 3` or `perm_mode == 1`.
    Head,
    /// `model.rs:2194` -> `linear.rs:2492` (`encode_gemv_fused2_mlp`).
    Fused2,
    /// `model.rs:1504` -> `linear.rs:2689` (`encode_gemv_fused3`).
    Fused3,
    /// `model.rs:1984` and `model.rs:2283` -> `linear.rs:2229`
    /// (`encode_gemv_raw_sk2`).
    RawSk2,
    /// `encode_ishizuki_gemv_named` with an explicit kernel name. Controls only.
    Named(String),
}

/// One row of the measured table.
#[derive(Clone, Debug)]
struct Row {
    label: String,
    /// The `SHAPES` names of every tensor the dispatch streams.
    parts: Vec<String>,
    path: Route,
    groups: u64,
    /// Set once measured; drives the reported ratios.
    gbps: Option<f64>,
    /// GB/s `gemv_bandwidth.rs` reported for the closest matching kernel+shape
    /// (implementation_notes.md 16.3). Recorded so production and microbenchmark
    /// get printed side by side rather than one being quoted alone again.
    microbench_gbps: Option<f64>,
    /// The rate this row's packed buffers were built at, when a control
    /// overrode it. `None` means "use the shape's declared rate".
    loaded_bits: Option<u8>,
}

impl Row {
    fn new(label: &str, parts: &[&str], path: Route, groups: u64) -> Self {
        Self {
            label: label.to_string(),
            parts: parts.iter().map(|s| (*s).to_string()).collect(),
            path,
            groups,
            gbps: None,
            microbench_gbps: None,
            loaded_bits: None,
        }
    }

    /// Record the rate the buffers were built at, so byte accounting matches
    /// what the kernel actually streamed.
    fn loaded_at(mut self, bits: u8) -> Self {
        self.loaded_bits = Some(bits);
        self
    }

    fn with_microbench(mut self, gbps: f64) -> Self {
        self.microbench_gbps = Some(gbps);
        self
    }

    /// Every part shares `in_features`, which is what makes the fused paths legal.
    fn k(&self) -> usize {
        shape(&self.parts[0]).in_features
    }

    /// Summed output width: what a fused dispatch streams in one call.
    fn n(&self) -> usize {
        self.parts.iter().map(|p| shape(p).out_features).sum()
    }

    fn bits(&self) -> u8 {
        shape(&self.parts[0]).bits
    }

    /// Packed weight bytes for `shape` read at `bits`, rather than at the
    /// shape's declared rate. Mirrors `bench::Shape::weight_bytes`, which is
    /// `tiles_k * tiles_n * 16 * bits` u16.
    fn weight_bytes_at(s: Shape, bits: u8) -> u64 {
        (((s.in_features / 16) * (s.out_features / 16)) as u64) * 16 * bits as u64 * 2
    }

    fn tiles_k(&self) -> u64 {
        (self.k() / 16) as u64
    }

    fn tiles_n(&self) -> u64 {
        (self.n() / 16) as u64
    }

    fn bytes(&self) -> u64 {
        // Rate-override aware. This used to call `shape(p).weight_bytes()`,
        // which uses the shape's DECLARED rate, so a control that rebuilt its
        // buffer at 3 bpw was still credited with 6 bpw bytes: the 3 bpw arm
        // reported 413.1 GB/s, 1.51x the 273 GB/s pin rate, i.e. physically
        // impossible and the tell that the divisor was wrong. The measured
        // time was correct throughout. Now the row carries the rate it was
        // actually loaded at, set by `load_row`'s `override_bits`.
        let rate = self.loaded_bits.unwrap_or(self.bits());
        self.parts
            .iter()
            .map(|p| Row::weight_bytes_at(shape(p), rate))
            .sum()
    }

    /// `tiles_k * tiles_n * 256`: the EXL3 codes the dispatch decodes.
    fn codes(&self) -> u64 {
        self.tiles_k() * self.tiles_n() * 256
    }

    /// `bytes / codes` is the bit rate divided by 8. It is the whole question:
    /// if `Gcodes/s` is flat across rows while `GB/s` tracks this column, the
    /// limiter is per-code work and nothing else.
    fn bytes_per_code(&self) -> f64 {
        self.bytes() as f64 / self.codes() as f64
    }

    fn waves(&self) -> f64 {
        self.groups as f64 / ONE_WAVE_256 as f64
    }

    fn gbps(&self) -> f64 {
        self.gbps.expect("row was never measured")
    }

    fn gcodes(&self) -> f64 {
        self.gbps() * 1e9 / self.bytes_per_code()
    }

    fn report(&self) {
        let measured = match self.gbps {
            Some(v) => format!("{v:7.1}"),
            None => "      -".to_string(),
        };
        let micro = match self.microbench_gbps {
            Some(v) => format!("{v:7.1}"),
            None => "      -".to_string(),
        };
        println!(
            "{label:<28} {k:>6} x {n:<7} {bits}bpw  tiles_k={tk:<5} tiles_n={tn:<6} \
             groups={groups:<6} waves={waves:<7.1} B/code={bpc:.3}  \
             measured={measured} GB/s  microbench={micro} GB/s  codes={codes:6.1} G/s",
            label = self.label,
            k = self.k(),
            n = self.n(),
            bits = self.bits(),
            tk = self.tiles_k(),
            tn = self.tiles_n(),
            groups = self.groups,
            waves = self.waves(),
            bpc = self.bytes_per_code(),
            codes = self.gcodes() / 1e9,
        );
    }
}

/// The production dispatch table, one row per kernel `model.rs` actually reaches.
///
/// The `Path` and the grid are not a choice; each is what the corresponding
/// branch resolves to with every `KAPLUMBA_*` switch unset.
/// `KAPLUMBA_TRACE_GEMV=1` prints the resolved names for a real run, and
/// `perf_probe_head_dispatch_arithmetic` is the arithmetic they must agree with.
fn production_rows() -> Vec<Row> {
    vec![
        // model.rs:4420 -> linear.rs:1960 -> linear.rs:2063 -> linear.rs:2159 ->
        // linear.rs:1461. perm_mode 0 (model.rs:3525-3542 never touches lm_head),
        // so neither the 4col nor the 2col arm can fire, and `_k6_w2` does not
        // exist (TWO_WORD_SAFE_RATES == (1,2,3,4), gen_exl3_gemv.py:151) so this
        // is the three-word row-major kernel, split-K 1.
        Row::new("head/lm_head", &["lm_head"], Route::Head, groups_1col(262_144)),
        // model.rs:1504 -> linear.rs:2785-2787. One dispatch for q+k+v.
        Row::new(
            "layer/qkv (fused3_2col)",
            &["sliding/q", "sliding/k", "sliding/v"],
            Route::Fused3,
            groups_fused3_2col(8192, 4096, 4096),
        ),
        // model.rs:2194 -> linear.rs:2590-2597. One dispatch for gate+up, and it
        // also carries the gated-GELU finalize (linear.rs:2657).
        Row::new(
            "layer/gate_up (fused2_2col)",
            &["sliding/gate", "sliding/up"],
            Route::Fused2,
            groups_fused2_2col(21504, 21504),
        ),
        // model.rs:1984 and model.rs:2283 -> linear.rs:2257-2258, split-K 2.
        Row::new(
            "layer/o_proj (2col_sk2)",
            &["sliding/o"],
            Route::RawSk2,
            groups_2col(5376, 2),
        )
        .with_microbench(120.7),
        Row::new(
            "layer/down (2col_sk2)",
            &["sliding/down"],
            Route::RawSk2,
            groups_2col(5376, 2),
        )
        .with_microbench(199.3),
    ]
}

/// Bytes the notes' tok/s arithmetic rests on, re-derived from the shape table so
/// a config change cannot quietly move the 20 tok/s target.
fn model_bytes() -> (u64, u64) {
    let sliding = SHAPES
        .iter()
        .filter(|s| s.name.starts_with("sliding/"))
        .map(|s| s.weight_bytes())
        .sum::<u64>();
    let full = [
        "full/q",
        "full/kv",
        "full/o",
        "sliding/gate",
        "sliding/up",
        "sliding/down",
    ]
    .iter()
    .map(|n| shape(n).weight_bytes())
    .sum::<u64>();
    (50 * sliding + 10 * full + shape("lm_head").weight_bytes(), shape("lm_head").weight_bytes())
}

// ---------------------------------------------------------------------------
// 2. Tests that need no GPU and no weights.
// ---------------------------------------------------------------------------

/// The dispatch arithmetic, pinned. No GPU, no model, no `#[ignore]`.
///
/// It is also the cheapest falsification available for the parallelism
/// hypothesis. If 16384 / 336 threadgroups were the explanation for the head's
/// 2x, the layer rows would be visibly parallelism-starved. They are not:
/// `o_proj` has the *thinnest* grid in the model (336 groups) and the *highest*
/// layer bandwidth (150 GB/s), while `gate_up` has four times the grid and
/// 129 GB/s. Thin and thick are unordered with respect to achieved bandwidth
/// among the layers, and the only 2x outlier is the only tensor at a different
/// bit rate.
#[test]
fn perf_probe_head_dispatch_arithmetic() {
    let rows = production_rows();

    // The limiter is always N. `out_features / 16` appears in every grid
    // formula in linear.rs; K appears in none of them.
    assert_eq!(rows[0].tiles_n(), 16384, "lm_head is 262144/16");
    assert_eq!(rows[0].tiles_k(), 336, "lm_head is 5376/16");

    assert_eq!(
        rows[0].groups, 16384,
        "head grid: out_features/16, linear.rs:1479"
    );
    assert_eq!(
        rows[1].groups, 512,
        "qkv grid: (512+256+256)/2, linear.rs:2787"
    );
    assert_eq!(
        rows[2].groups, 1344,
        "gate_up grid: (1344+1344)/2, linear.rs:2597"
    );
    assert_eq!(
        rows[3].groups, 336,
        "o_proj grid: (336/2)*2, linear.rs:2258 -- the thinnest grid in the model"
    );
    assert_eq!(
        rows[4].groups, 336,
        "down grid: same kernel and same out_features as o_proj"
    );

    // The head has 12x the parallelism of the widest layer GEMV and 49x the
    // thinnest, and every row is many waves deep, so nothing here is starved for
    // threadgroups.
    let widest_layer = rows.iter().skip(1).map(|r| r.groups).max().expect("rows");
    let thinnest_layer = rows.iter().skip(1).map(|r| r.groups).min().expect("rows");
    assert_eq!(rows[0].groups / widest_layer, 12);
    assert_eq!(rows[0].groups / thinnest_layer, 48);
    for row in &rows {
        assert!(
            row.waves() > 3.0,
            "{} has only {:.1} waves at {} groups; a parallelism-starvation \
             explanation would need this to be small",
            row.label,
            row.waves(),
            row.groups
        );
    }

    // Bytes per code is the only column that separates the head from the layers
    // by the same factor as their bandwidths.
    assert_eq!(rows[0].bytes_per_code(), 0.75, "6 bpw");
    for row in rows.iter().skip(1) {
        assert_eq!(row.bytes_per_code(), 0.375, "3 bpw: {}", row.label);
    }
    let bpc_ratio = rows[0].bytes_per_code() / rows[4].bytes_per_code();
    assert_eq!(bpc_ratio, 2.0);

    // The measured production bandwidth ratio (implementation_notes.md 18.2)
    // must keep matching the bits-per-code ratio. 253 / 132.5 = 1.909 against a
    // bits-per-code ratio of 2.000: the 4.5% shortfall is `o_proj`, the one layer
    // measured *above* the others at 150 GB/s, and it is the whole reason the
    // tolerance is 0.12 rather than an equality. If the ratio drifts outside
    // that, the head's advantage is no longer accounted for by bit rate and
    // (a)/(b)/(c) have to be investigated rather than dismissed.
    let production = [
        ("gate_up", 129.0),
        ("down", 126.0),
        ("qkv", 125.0),
        ("o_proj", 150.0),
        ("head", 253.0),
    ];
    let head_gbps = production[4].1;
    let layer_mean =
        production[..4].iter().map(|(_, v)| *v).sum::<f64>() / production[..4].len() as f64;
    let bandwidth_ratio = head_gbps / layer_mean;
    assert!(
        (bandwidth_ratio - bpc_ratio).abs() < 0.12,
        "measured bandwidth ratio {bandwidth_ratio:.3} does not match the \
         bits-per-code ratio {bpc_ratio:.3}: the head's advantage is no longer \
         accounted for by bit rate"
    );
}

/// The fused rows must count every tensor they stream, or their GB/s is
/// understated by 2x and 3x respectively.
#[test]
fn fused_rows_count_every_tensor_they_stream() {
    let rows = production_rows();
    let gate_up = rows.iter().find(|r| r.label.starts_with("layer/gate_up")).expect("row");
    assert_eq!(
        gate_up.bytes(),
        shape("sliding/gate").weight_bytes() + shape("sliding/up").weight_bytes(),
        "the fused2 row must count both tensors"
    );
    let qkv = rows.iter().find(|r| r.label.starts_with("layer/qkv")).expect("row");
    assert_eq!(
        qkv.bytes(),
        shape("sliding/q").weight_bytes()
            + shape("sliding/k").weight_bytes()
            + shape("sliding/v").weight_bytes(),
        "the fused3 row must count all three tensors"
    );
    let head = rows.iter().find(|r| r.label.starts_with("head")).expect("row");
    assert_eq!(head.bytes(), 1_056_964_608, "lm_head at 6 bpw");
    assert_eq!(head.n(), 262_144);
    assert_eq!(head.k(), 5376);
}

/// The tok/s arithmetic, in **codes per second** rather than GB/s.
///
/// GB/s is the wrong unit for this question, because the head and the layers
/// differ in bits per code. Codes/s is the unit in which they are comparable, and
/// every number below is one of:
///
/// | code rate | where it comes from | model step | tok/s |
/// |---|---|---|---|
/// | 338 Gcodes/s | the head, §18.2 (253 GB/s at 0.75 B/code) | 95.0 ms | 10.5 |
/// | 344 Gcodes/s | the layers, §18.2 (129 GB/s at 0.375 B/code) | 92.5 ms | 10.8 |
/// | 541 Gcodes/s | best 3 bpw microbenchmark, §16.3 (203 GB/s) | 59.3 ms | 16.9 |
/// | 674 Gcodes/s | what 253 GB/s at 3 bpw would need | 47.6 ms | 21.0 |
///
/// The last row is the whole prize, and the gap to the row above it is the
/// question: 20 tok/s needs a code rate **1.96x** what production achieves and
/// **1.25x** anything ever measured for a 3 bpw EXL3 GEMV on this box. If the
/// limiter is per-code work -- which is what (d) says -- then the head's
/// property is that it gets that many bytes per unit of that work, and a kernel
/// change cannot hand it to a tensor whose codes are half as wide.
#[test]
fn perf_probe_head_byte_accounting() {
    let (total, head) = model_bytes();
    assert_eq!(
        total, 12_039_487_488,
        "bench.rs already pins this; if it moved, the config did"
    );
    assert_eq!(head, 1_056_964_608, "lm_head at 6 bpw");

    const HEAD_BPC: f64 = 0.75;
    const LAYER_BPC: f64 = 0.375;

    let step_ms = |codes_per_s: f64, bpc: f64| total as f64 / (codes_per_s * bpc) * 1e3;

    // The head's measured code rate: 1.409e9 codes in 4.17 ms (§18.2).
    let head_rate = 1_409_372_160.0 / 4.17e-3;
    assert!((head_rate * HEAD_BPC / 1e9 - 253.0).abs() < 1.0, "head GB/s");

    // The layer code rate implied by §18.2's deltas. The four layer GEMVs account
    // for every byte of the model except the head, so their codes/token is
    // `(total - head) * 8/3`, and their time is the sum of the four ablation
    // deltas.
    let layer_bytes = total - head;
    let layer_ms: f64 = 40.34 + 20.63 + 15.82 + 6.63;
    let layer_rate = layer_bytes as f64 * 8.0 / 3.0 / (layer_ms * 1e-3);
    assert!(
        (layer_rate * LAYER_BPC / 1e9 - 129.0).abs() < 8.0,
        "layer GB/s {} does not match §18.2's 125-150 range",
        layer_rate * LAYER_BPC / 1e9
    );

    // Best 3 bpw figure any microbenchmark has produced: 203 GB/s (§16.3).
    let best_3bpw_gbps = 203.0;
    let best_3bpw_rate = best_3bpw_gbps * 1e9 / LAYER_BPC;

    // What the layers would have to decode to stream at the head's GB/s.
    let needed_rate = 253.0 * 1e9 / LAYER_BPC;

    println!("\ncode rate (Gcodes/s) -> model step -> tok/s, for one bytes/token figure");
    for (label, rate, bpc) in [
        ("head as measured (338)", head_rate, HEAD_BPC),
        ("layers as measured", layer_rate, LAYER_BPC),
        ("best 3-bpw microbenchmark (203 GB/s)", best_3bpw_rate, LAYER_BPC),
        ("needed for 253 GB/s at 3 bpw", needed_rate, LAYER_BPC),
    ] {
        let ms = step_ms(rate, bpc);
        println!("  {label:<38} {:>6.1}  ->  {:>5.1} ms  ->  {:>5.2} tok/s", rate / 1e9, ms, 1e3 / ms);
    }
    println!(
        "\n  head and layers differ in code rate by {:.3}x and in GB/s by {:.3}x; \
         their bits-per-code ratio is {:.1}x",
        head_rate / layer_rate,
        (head_rate * HEAD_BPC) / (layer_rate * LAYER_BPC),
        HEAD_BPC / LAYER_BPC,
    );

    // The two ratios that decide everything. If the code rates match and the
    // bandwidths do not, the head's advantage is entirely bits per code.
    assert!(
        (head_rate / layer_rate - 1.0).abs() < 0.12,
        "head and layers differ in code rate by {:.3}x; the bits-per-code reading \
         of the 2x bandwidth gap does not hold",
        head_rate / layer_rate
    );
    assert!(
        ((head_rate * HEAD_BPC) / (layer_rate * LAYER_BPC) - 2.0).abs() < 0.2,
        "and their bandwidths do not differ by ~2x"
    );

    // And the gap the change would have to close: ~1.9x over production, ~1.25x
    // over the best 3-bpw measurement anyone has made.
    assert!(
        (needed_rate / layer_rate - 1.9).abs() < 0.15,
        "20 tok/s now needs {:.2}x the production code rate",
        needed_rate / layer_rate
    );
    assert!(
        needed_rate > best_3bpw_rate,
        "but {:.2}x the best 3-bpw code rate ever measured ({:.0} Gcodes/s), so \
         the target is out of reach by a margin, not by a rounding error",
        needed_rate / best_3bpw_rate,
        best_3bpw_rate / 1e9
    );

    // The two ceilings in ms/step, which is what the notes quote.
    let best_ms = step_ms(best_3bpw_rate, LAYER_BPC);
    println!(
        "  ceiling at the best 3-bpw rate: {best_ms:.1} ms/step = {:.2} tok/s",
        1e3 / best_ms
    );
    let needed_ms = step_ms(needed_rate, LAYER_BPC);
    println!(
        "  step at the head's GB/s:        {needed_ms:.1} ms/step = {:.2} tok/s",
        1e3 / needed_ms
    );
}

// ---------------------------------------------------------------------------
// 3. Real weights and buffers.
// ---------------------------------------------------------------------------

fn test_model() -> PathBuf {
    PathBuf::from(
        std::env::var("KAPLUMBA_TEST_MODEL")
            .expect("KAPLUMBA_TEST_MODEL must point at the checkpoint directory"),
    )
}

fn tensor_range(model: &Path, name: &str) -> (String, u64, u64) {
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(model.join("model.safetensors.index.json")).expect("index"),
    )
    .expect("index json");
    let shard = index["weight_map"][name]
        .as_str()
        .unwrap_or_else(|| panic!("no shard for {name}"))
        .to_string();
    let mut file = File::open(model.join(&shard)).expect("shard");
    let mut len = [0u8; 8];
    file.read_exact(&mut len).expect("length prefix");
    let header_len = u64::from_le_bytes(len) as usize;
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes).expect("header");
    let header: serde_json::Value = serde_json::from_slice(&header_bytes).expect("header json");
    let offsets = header[name]["data_offsets"]
        .as_array()
        .unwrap_or_else(|| panic!("{name} has no data_offsets"));
    (
        shard,
        8 + header_len as u64 + offsets[0].as_u64().unwrap(),
        8 + header_len as u64 + offsets[1].as_u64().unwrap(),
    )
}

/// Real packed trellis bytes for one checkpoint tensor.
fn read_packed(model: &Path, name: &str, expect_words: usize) -> BufferRef {
    let (shard, start, end) = tensor_range(model, name);
    let mut file = File::open(model.join(shard)).expect("shard");
    file.seek(SeekFrom::Start(start)).expect("seek");
    let mut bytes = vec![0u8; (end - start) as usize];
    file.read_exact(&mut bytes).expect("payload");
    assert_eq!(bytes.len() % 2, 0, "{name} is not a whole number of u16");
    let words = bytes.len() / 2;
    assert_eq!(
        words, expect_words,
        "{name}: read {words} u16, the shape table says {expect_words}. A \
         different number means the checkpoint and config have diverged and \
         every bandwidth below would be of the wrong tensor."
    );
    bytes
        .chunks_exact(2)
        .map(|p| u16::from_le_bytes([p[0], p[1]]))
        .collect::<Vec<u16>>()
        .into()
}

/// Owned `Vec<u16>` bytes, so the host copy can be dropped before timing.
struct BufferRef(Vec<u16>);

impl From<Vec<u16>> for BufferRef {
    fn from(v: Vec<u16>) -> Self {
        Self(v)
    }
}

impl BufferRef {
    fn upload(&self, device: &Device) -> Buffer {
        device.new_buffer_with_data(
            self.0.as_ptr() as *const c_void,
            (self.0.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn words(&self) -> usize {
        self.0.len()
    }
}

/// A same-sized pseudo-random buffer, for the controls that need a bit rate no
/// checkpoint tensor has. Content is irrelevant to bandwidth; `assert_live` is
/// what stops a wrong-content kernel from being reported as fast.
fn random_u16(words: usize, seed: u32) -> BufferRef {
    let mut state = seed | 1;
    let mut data = Vec::with_capacity(words);
    for _ in 0..words {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        data.push((state >> 3) as u16);
    }
    BufferRef(data)
}

/// Everything a dispatch needs besides the weights.
struct Scratch {
    input: Buffer,
    input_signs: Buffer,
    output_signs: Buffer,
    transformed: Buffer,
    /// `raw_a` for the fused kernels, and the single `raw` for everything else.
    /// Sized for the largest split factor any path uses (split-K 2, so 2x N).
    raw: Buffer,
    /// `raw_b` for `encode_gemv_fused2_mlp`, which needs two distinct f32
    /// outputs. Distinct is the point: both would alias `raw` otherwise.
    raw_b: Buffer,
    output: Buffer,
}

impl Scratch {
    fn new(device: &Device, k: usize, n: usize) -> Self {
        fn halves(device: &Device, count: usize) -> Buffer {
            let mut data = vec![0u16; count];
            for (i, w) in data.iter_mut().enumerate() {
                // A non-degenerate pattern: all-ones or all-zero signs would make
                // the input transform trivial and could let a dead kernel look busy.
                *w = 0x3c00u16 ^ ((i as u16).wrapping_mul(0x0101));
            }
            device.new_buffer_with_data(
                data.as_ptr() as *const c_void,
                (data.len() * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        }
        let shared = MTLResourceOptions::StorageModeShared;
        Self {
            input: halves(device, k),
            input_signs: halves(device, k),
            output_signs: halves(device, n),
            transformed: device.new_buffer((k * 2) as u64, shared),
            raw: device.new_buffer((n * 2 * 4) as u64, shared),
            raw_b: device.new_buffer((n * 2 * 4) as u64, shared),
            // `n * 4`, not `n * 2`. Every GEMV epilogue writes
            // `device float *y` with `sum` an f32, so the output is 4 bytes
            // per element. This was `n * 2`, sizing it as fp16 -- half the
            // buffer the kernel writes -- which is why the liveness guard saw
            // exactly `n/2` non-zero values and could not tell a half-written
            // buffer from a healthy one. `raw`/`raw_b` two lines up already
            // used `* 2 * 4`, which is what made the inconsistency visible.
            output: device.new_buffer((n * 4) as u64, shared),
        }
    }

    /// Fill the output buffer with a sentinel, so `assert_live`'s
    /// "untouched" count means something. A zeroed Metal buffer cannot
    /// distinguish "never written" from "written as zero", which is precisely
    /// how a half-sized output buffer hid here.
    fn poison_output(&self) {
        unsafe {
            for buffer in [&self.output, &self.raw] {
                let n = buffer.length() as usize / 4;
                let p = buffer.contents() as *mut f32;
                for i in 0..n {
                    *p.add(i) = SENTINEL_F32;
                }
            }
        }
    }

    /// The buffer this row's kernel writes, per `assert_live`'s note: `raw` for
    /// every ishizuki/`Named` arm, `output` for the paths that run the finalize.
    fn liveness_buffer(&self, row: &Row) -> &Buffer {
        match row.path {
            Route::Named(_) => &self.raw,
            _ => &self.output,
        }
    }

    fn request<'a>(
        &'a self,
        packed: &'a Buffer,
        in_features: usize,
        out_features: usize,
        bits: u8,
        perm_mode: u32,
    ) -> LinearEncodeRequest<'a> {
        LinearEncodeRequest {
            packed,
            packed_offset: 0,
            perm_mode,
            input: &self.input,
            input_signs: &self.input_signs,
            output_signs: &self.output_signs,
            transformed: &self.transformed,
            raw: &self.raw,
            output: &self.output,
            in_features,
            out_features,
            rate: Exl3Rate::Integer(bits),
            codebook: Exl3Codebook::Mcg,
            pretransformed: false,
        }
    }
}

/// GPU milliseconds per call, from the GPU clock, after calibration.
///
/// The three disciplines `gemv_bandwidth.rs` established, because each produced
/// a wrong number before: GPU clock rather than wall clock (a wall-clock first
/// version was launch-bound at every point and reported 1600-2000 GB/s),
/// `set_max_dispatches(u32::MAX)` (the encoder otherwise rolls command buffers
/// and the GPU span covers a fraction of the work, reporting up to 6500 GB/s),
/// and long runs (a 4.7 ms measurement reported a 3.5x spread between adjacent
/// K values from clock ramp alone).
fn time_ms<F: FnMut(&mut MetalEncoder)>(queue: &CommandQueue, label: &str, mut dispatch: F) -> (f64, usize) {
    let reps = {
        let mut enc = MetalEncoder::new(queue).expect("encoder");
        enc.set_max_dispatches(u32::MAX);
        let t0 = Instant::now();
        for _ in 0..4 {
            dispatch(&mut enc);
        }
        enc.commit_and_wait().expect("calibrate");
        let per = (t0.elapsed().as_secs_f64() * 1e3 / 4.0).max(1e-4);
        ((40.0 / per).round() as usize).clamp(4, 400)
    };
    {
        let mut enc = MetalEncoder::new(queue).expect("encoder");
        for _ in 0..reps.min(8) {
            dispatch(&mut enc);
        }
        enc.commit_and_wait().expect("warmup");
    }
    let mut enc = MetalEncoder::new(queue).expect("encoder");
    enc.set_max_dispatches(u32::MAX);
    for _ in 0..reps {
        dispatch(&mut enc);
    }
    enc.commit_stamped(label).expect("commit");
    // `take_gpu_stamps` waits on the parked buffer before reading GPUStartTime.
    let gpu_ms: f64 = enc.take_gpu_stamps().iter().map(|s| s.own_ms).sum();
    enc.commit_and_wait().ok();
    assert!(
        gpu_ms > 0.0,
        "{label}: no GPU timing. KAPLUMBA_DECODE_PROFILE=kernel is not required \
         for this path, but a zero span means the CommandBufferGpuTiming \
         selectors are unavailable and every number below would be 0."
    );
    (gpu_ms / reps as f64, reps)
}

/// Refuse to report a bandwidth for a kernel that produced nothing. A dead or
/// all-skipped dispatch would otherwise report a spectacular GB/s.
/// Which buffer a row's kernel actually writes.
///
/// This matters and the original code got it wrong in a way that produced a
/// symptom indistinguishable from a kernel defect. `Route::Named` (and every
/// ishizuki arm) accumulates into `request.raw` and then a separate `finalize`
/// dispatch converts `raw` into `request.output`. So:
///   * a row dispatched via `Route::Named` must be checked on **raw**, and
///   * only rows whose path runs the finalize (Head, and the fused paths) end
///     up with `output` written at all.
///
/// Checking `output` for a `Named` row found 131072 slots still holding the
/// sentinel and read as "the kernel wrote half its output". It wrote all of
/// `raw`; `output` was simply never the destination. `probe_k6_half_output.rs`
/// confirms the k6 epilogue covers every slot when dispatched directly.
fn assert_live(output: &Buffer, n: usize, label: &str) {
    // f32, not fp16 pairs. `scratch.output` is `raw`/`raw_a`, an f32 buffer
    // written by the GEMV epilogue. This function read it as `u16` and ran
    // `half::f16::from_bits` on each half, which reinterpreted f32 bit patterns
    // as fp16: a 5-bit all-ones exponent occurs ~1/32 of the time per half, so
    // with 262144 outputs the guard could never pass.
    let ptr = output.contents() as *const f32;
    let values = unsafe { std::slice::from_raw_parts(ptr, n) };
    // Liveness is "the output buffer was WRITTEN", not "the values are finite".
    //
    // This probe synthesises its buffers from non-code bytes (a re-rate control
    // has no checkpoint tensor at the changed rate), and `decode_state_mcg`
    // computes `float(h.x + h.y)` where `h` is a `half2` -- a HALF add. Feeding
    // it arbitrary bit patterns produces half `inf`/`NaN` encodings, and a
    // 5376-term accumulation propagates them. That is a property of the
    // synthetic input, not of the kernel: bandwidth is bytes/second and every
    // byte is genuinely read, so finiteness is the wrong liveness test here.
    //
    // The real liveness test is the sentinel one, in
    // `probe_k6_half_output.rs`: poison the buffer, dispatch, and require that
    // no slot still holds the sentinel. It proved all 262144 slots written with
    // `first untouched: None`, i.e. the k6 epilogue has full output coverage
    // and this file's earlier "half written" symptom was its own undersized
    // output buffer.
    let untouched = values.iter().filter(|v| **v == SENTINEL_F32).count();
    let nonzero = values.iter().filter(|v| **v != 0.0).count();
    assert!(
        nonzero > n / 2 && untouched == 0,
        "{label}: output has {nonzero}/{n} non-zero values and {untouched} untouched, so \
         the timed kernel did not produce a result -- a GB/s figure for it would \
         be measuring an idle machine"
    );
}

/// Sentinel for `assert_live`. Chosen so a decoded value cannot land on it.
const SENTINEL_F32: f32 = -1.0e30;

// ---------------------------------------------------------------------------
// 4. The GPU probes.
// ---------------------------------------------------------------------------

/// Everything one row needs on the GPU.
struct Loaded {
    row: Row,
    scratch: Scratch,
    packed: Vec<Buffer>,
    kernels: &'static LinearKernels,
    /// The rate the packed buffers were actually built at -- `row.bits()`
    /// unless a control overrode it. Carried here so `dispatch_row` cannot
    /// re-derive the declared rate and lose the override.
    bits: u8,
}

/// Upload a row's weights and build its scratch.
///
/// `model: Some(_)` reads the checkpoint; `None` synthesises the same number of
/// bytes, which is what the rate controls need -- a control changes the *bit
/// rate* it reads at, and no checkpoint tensor has the changed rate. Every other
/// property (shape, tile count, byte count) is held fixed.
fn load_row(device: &Device, row: Row, model: Option<&Path>, override_bits: Option<u8>) -> Loaded {
    let bits = override_bits.unwrap_or_else(|| row.bits());
    let names: Vec<String> = row
        .parts
        .iter()
        .map(|p| match p.as_str() {
            "lm_head" => "lm_head.trellis".to_string(),
            other => {
                let leaf = other.strip_prefix("sliding/").expect("sliding shape name");
                let proj = match leaf {
                    "q" => "self_attn.q_proj",
                    "k" => "self_attn.k_proj",
                    "v" => "self_attn.v_proj",
                    "o" => "self_attn.o_proj",
                    "gate" => "mlp.gate_proj",
                    "up" => "mlp.up_proj",
                    "down" => "mlp.down_proj",
                    other => panic!("no tensor for {other}"),
                };
                format!("{LAYER0}{proj}.trellis")
            }
        })
        .collect();

    let mut packed = Vec::new();
    for (name, part) in names.iter().zip(row.parts.iter()) {
        let s = shape(part);
        let expect = (s.in_features / 16) * (s.out_features / 16) * s.words_per_tile();
        let bytes = match model {
            Some(m) => {
                assert_eq!(
                    s.bits, bits,
                    "{part} is {} bpw and the checkpoint holds it at that rate, so a \
                     {bits} bpw row would be measuring a buffer of the wrong size",
                    s.bits
                );
                read_packed(m, name, expect)
            }
            None => random_u16(expect, 0x1234_0000 + u32::from(bits)),
        };
        assert_eq!(bytes.words(), expect, "{name}");
        println!(
            "  {:<7} {part:<12} {name:<58} {:>8.1} MB ({bits} bpw, {} x {}, {})",
            if model.is_some() {
                "loaded"
            } else {
                "synth"
            },
            bytes.words() as f64 * 2.0 / 1e6,
            s.in_features,
            s.out_features,
            layout_name(&s),
        );
        packed.push(bytes.upload(device));
    }
    let scratch = Scratch::new(device, row.k(), row.n());
    Loaded {
        row,
        scratch,
        packed,
        // `LinearKernels::shared()` is a `OnceLock`, so this is the same object
        // every row in the process dispatches through, as in the engine. Held
        // here so no row can compile a pipeline inside a timed loop.
        kernels: LinearKernels::shared().expect("linear kernels"),
        bits,
    }
}

/// Dispatch one row once, through the production entry point its `Path` names.
///
/// `pretransformed` is left false everywhere so each path performs its own
/// input transform: that is one extra dispatch over `k` halves, under 1% of the
/// weight bytes, and identical for every row, so it cannot move a ratio.
fn dispatch_row(enc: &mut MetalEncoder, l: &Loaded) {
    let row = &l.row;
    let k = row.k();
    let n_out = row.n();
    // The rate the BUFFER was built at, which for a control is the control's
    // point: `load_row` overrides it via `override_bits`. This used to be
    // `row.bits()`, i.e. the row's *declared* rate, so a control that re-rated
    // its buffer to 3 bpw still dispatched a 6 bpw request and the dispatcher
    // rejected it -- `decodes 3 bpw but the request is 6 bpw`. The control could
    // not run at all. `Loaded::bits` is the overridden value.
    let bits = l.bits;
    // The head is row-major (`perm_mode` 0 -- model.rs:3525-3542 sets 1 only on
    // the seven layer linears); the colmajor layer kernels need 1.
    let perm = u32::from(!row.parts.iter().any(|p| p == "lm_head"));
    match &row.path {
        Route::Head => l
            .kernels
            .encode_gemv(enc, l.scratch.request(&l.packed[0], k, n_out, bits, perm))
            .expect("head gemv"),
        Route::RawSk2 => l
            .kernels
            .encode_gemv_raw_sk2(
                enc,
                &l.packed[0],
                0,
                perm,
                Exl3Rate::Integer(bits),
                k,
                n_out,
                &l.scratch.transformed,
                &l.scratch.raw,
            )
            .expect("gemv_raw_sk2"),
        Route::Fused2 => l
            .kernels
            .encode_gemv_fused2_mlp(
                enc,
                &l.packed[0],
                0,
                n_out / 2,
                &l.scratch.output_signs,
                &l.packed[1],
                0,
                n_out / 2,
                &l.scratch.output_signs,
                // No down-input signs: production passes them, but the head of
                // this probe has no `down_proj` to hand over and the finalize
                // differs only in an extra elementwise pass over 21504 halves.
                None,
                &l.scratch.output,
                &l.scratch.input,
                &l.scratch.input_signs,
                &l.scratch.transformed,
                &l.scratch.raw,
                &l.scratch.raw_b,
                k,
                false,
                0,
            )
            .expect("gemv_fused2_mlp"),
        Route::Fused3 => l
            .kernels
            .encode_gemv_fused3(
                enc,
                &l.packed[0],
                0,
                n_out / 3,
                &l.scratch.output_signs,
                &l.scratch.output,
                &l.packed[1],
                0,
                n_out / 3,
                &l.scratch.output_signs,
                &l.scratch.output,
                &l.packed[2],
                0,
                n_out / 3,
                &l.scratch.output_signs,
                &l.scratch.output,
                &l.scratch.input,
                &l.scratch.input_signs,
                &l.scratch.transformed,
                &l.scratch.raw,
                &l.scratch.raw_b,
                &l.scratch.raw,
                k,
                false,
                0,
            )
            .expect("gemv_fused3"),
        Route::Named(name) => l
            .kernels
            .encode_ishizuki_gemv_named(
                enc,
                name,
                &l.scratch.request(&l.packed[0], k, n_out, bits, 0),
            )
            .expect("named gemv"),
    }
}

/// Time a row: three rounds, median reported.
///
/// Median, not best and not mean: AGENTS.md 10.5 records decode swinging
/// 3.7-6.4 tok/s from the *same binary* minutes apart, so a single reading is not
/// a result and a minimum is the selection effect that produced the retracted
/// 248.5/250.7 GB/s readings.
fn measure_row(queue: &CommandQueue, l: &Loaded) -> f64 {
    // Poison BEFORE any dispatch, not here at the end. The sentinel only
    // identifies an unwritten slot if the kernel has not already written it;
    // poisoning after the timing loop made every slot hold the sentinel again
    // and `assert_live` reported "262144 non-zero AND 262144 untouched", which
    // is self-contradictory and was the symptom of exactly this ordering bug.
    l.scratch.poison_output();
    let mut samples: Vec<(f64, usize)> = Vec::new();
    for _ in 0..3 {
        samples.push(time_ms(queue, &l.row.label, |enc| {
            dispatch_row(enc, l);
        }));
    }
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite"));
    let (ms, reps) = samples[1];
    // Every row, not just the production one: the controls dispatch kernels
    // against weights they were not written for, and a control that measured a
    // dispatch which read nothing would report an enormous GB/s rather than fail.
    let live_buffer = l.scratch.liveness_buffer(&l.row);
    assert_live(live_buffer, l.row.n(), &l.row.label);
    let gbps = l.row.bytes() as f64 / (ms * 1e-3) / 1e9;
    println!(
        "  {:<28} {:>9.3} ms/call  {:>7.1} GB/s  {:>6.1} Gcodes/s  ({reps} reps, \
         median of 3: {:?})",
        l.row.label,
        ms,
        gbps,
        gbps * 1e9 / l.row.bytes_per_code() / 1e9,
        samples.iter().map(|(m, _)| *m).collect::<Vec<_>>(),
    );
    gbps
}

/// The headline measurement: every production shape, the production entry point,
/// the production grid, the checkpoint's own bytes.
#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL and an idle GPU"]
fn head_vs_layers_run_the_production_dispatch() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let kernels = LinearKernels::shared().expect("linear kernels");
    let model = test_model();

    // Every production kernel name must exist, or the dispatch falls through to
    // something else and the row measures the wrong kernel.
    let names = kernels.ishizuki_gemv_kernel_names();
    for name in [
        "linear_gemv_ishizuki_k6",
        "linear_gemv_ishizuki_k3_fused3_2col",
        "linear_gemv_ishizuki_k3_fused2_2col",
        "linear_gemv_ishizuki_k3_colmajor_2col_sk2",
    ] {
        assert!(
            names.iter().any(|n| n == name),
            "the library has no {name}, so this row would measure a fallback \
             (have: {})",
            names.join(", ")
        );
    }
    // And the head must NOT have a two-word kernel: `_k6_w2` does not exist
    // (TWO_WORD_SAFE_RATES == (1,2,3,4)), which is why it reads three words.
    assert!(
        !names.iter().any(|n| n == "linear_gemv_ishizuki_k6_w2"),
        "a _k6_w2 kernel now exists; the head would dispatch it and the whole \
         three-word reading of linear.metal:2201-2203 would be stale"
    );

    println!("\n=== production shapes, production entry points, real weights ===");
    let mut rows = Vec::new();
    for row in production_rows() {
        let loaded = load_row(&device, row, Some(&model), None);
        let gbps = measure_row(&queue, &loaded);
        let mut row = loaded.row.clone().loaded_at(loaded.bits);
        row.gbps = Some(gbps);
        rows.push(row);
    }

    println!();
    for row in &rows {
        row.report();
    }
    assert_bandwidth_is_bpw(&rows);
}

/// Control R. The head's shape, the head's kernel family, the head's grid, the
/// head's row-major layout -- at 3 bpw instead of 6.
///
/// This is the only experiment here that changes exactly one variable, and it is
/// the one that decides (d). `linear_gemv_ishizuki_k6` and
/// `linear_gemv_ishizuki_k3` are the same transcriptions of the same ishizuki
/// source at KA=6 and KA=3: one output tile per threadgroup, the same grid, the
/// same row-major addressing, the same `red[SG][16]` reduction. Only the bit
/// rate moves.
#[test]
#[ignore = "requires an idle GPU; no checkpoint needed"]
fn head_shape_at_three_bpw() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    LinearKernels::shared().expect("linear kernels");
    let head = shape("lm_head");

    let mut out = Vec::new();
    for bits in [6u8, 3u8] {
        let row = Row::new(
            if bits == 6 {
                "ctl:head @6bpw"
            } else {
                "ctl:head @3bpw"
            },
            &["lm_head"],
            // The 6 bpw kernel is `_k6_colmajor_contig4_bench`
            // (linear_repacked.metal:3815), NOT `linear_gemv_ishizuki_k6`.
            // The bare `k6` name does not exist in the library, so the lookup
            // silently found nothing and the "timed" dispatch did nothing --
            // which is exactly what `assert_live` then caught. Both arms name
            // the same family at their own rate so that only the bit rate
            // differs.
            Route::Named(if bits == 6 {
                "linear_gemv_ishizuki_k6_colmajor_contig4_bench".to_string()
            } else {
                "linear_gemv_ishizuki_k3_colmajor_contig4".to_string()
            }),
            groups_1col(head.out_features),
        );
        let loaded = load_row(&device, row, None, Some(bits));
        let mut row = loaded.row.clone().loaded_at(loaded.bits);
        row.gbps = Some(measure_row(&queue, &loaded));
        out.push(row);
    }

    println!();
    for row in &out {
        row.report();
    }
    let (six, three) = (&out[0], &out[1]);
    let bw_ratio = three.gbps() / six.gbps();
    let code_ratio = three.gcodes() / six.gcodes();
    println!(
        "\nControl R: 3 bpw / 6 bpw bandwidth ratio {bw_ratio:.3}, code-rate ratio \
         {code_ratio:.3} (bits-per-code ratio {:.3})",
        six.bytes_per_code() / three.bytes_per_code()
    );
    assert!(
        (bw_ratio - 0.5).abs() < 0.10,
        "bandwidth ratio {bw_ratio:.3} is not ~0.5: at 3 bpw this shape does not lose \
         exactly its bit rate, so something other than bits-per-code is involved and \
         (a), (b) or (c) has to be investigated"
    );
}

/// Control S. `down_proj`'s shape -- the worst production bandwidth (126 GB/s) and
/// the thinnest grid (336 groups) -- read at 6 bpw.
///
/// The mirror of control R. If 6 bpw were a property of the *shape* rather than
/// of the tensor, a layer shape at 6 bpw should reach head-like GB/s. It should
/// not; if it does, (d) is wrong.
///
/// `linear_gemv_ishizuki_k6_colmajor_contig4_bench` is the repo's existing 6 bpw
/// colmajor kernel (linear_repacked.metal:3529), written for exactly this kind of
/// rate-versus-3-bpw comparison (the comment block at 3489-3528 derives it).
#[test]
#[ignore = "requires an idle GPU; no checkpoint needed"]
fn layer_shape_at_six_bpw() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    LinearKernels::shared().expect("linear kernels");

    let mut out = Vec::new();
    for (label, bits, path, groups, micro) in [
        (
            "ctl:down shape @3bpw",
            3u8,
            Route::Named("linear_gemv_ishizuki_k3_colmajor_2col_sk2".to_string()),
            groups_2col(5376, 2),
            Some(199.3),
        ),
        (
            "ctl:down shape @6bpw",
            6u8,
            Route::Named("linear_gemv_ishizuki_k6_colmajor_contig4_bench".to_string()),
            groups_1col(5376),
            None,
        ),
    ] {
        let row = Row::new(label, &["sliding/down"], path, groups);
        let row = match micro {
            Some(v) => row.with_microbench(v),
            None => row,
        };
        let loaded = load_row(&device, row, None, Some(bits));
        let mut row = loaded.row.clone().loaded_at(loaded.bits);
        row.gbps = Some(measure_row(&queue, &loaded));
        out.push(row);
    }

    println!();
    for row in &out {
        row.report();
    }
    println!(
        "\nControl S: a layer shape at 6 bpw reached {:.1} GB/s against the head's \
         253. If these agree, the advantage is the *shape*, not the bit rate.",
        out[1].gbps()
    );
}

/// Control G. The head's shape at 6 bpw with the grid squeezed from 16384 groups
/// down to the thinnest grid in the model.
///
/// `KAPLUMBA_GEMV_MAX_GROUPS` is read through a `OnceLock`
/// (crates/kernels/src/linear.rs:94), so one process cannot sweep it. This
/// re-execs the test binary once per value, the way `weight_residency.rs` does
/// its escape hatch. Capping the grid makes each threadgroup stride over more
/// output tiles through the grid-stride loop at linear.metal:2186, so the total
/// work is identical and only the parallelism changes -- which is exactly the
/// variable (a) is about.
#[test]
#[ignore = "requires an idle GPU; re-execs itself"]
fn head_grid_thickness_sweep() {
    if std::env::var(CHILD_GROUPS).is_ok() {
        measure_one_grid();
        return;
    }
    for groups in ["16384", "1344", "512", "336"] {
        let status = std::process::Command::new(std::env::current_exe().expect("exe"))
            .args([
                "--ignored",
                "--exact",
                "head_grid_thickness_sweep",
                "--nocapture",
            ])
            .env(CHILD_GROUPS, groups)
            .status()
            .expect("re-exec");
        assert!(status.success(), "grid={groups} child failed");
    }
}

fn measure_one_grid() {
    let groups: u64 = std::env::var(CHILD_GROUPS)
        .expect("child only")
        .parse()
        .expect("numeric grid");
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    LinearKernels::shared().expect("linear kernels");
    let head = shape("lm_head");

    let row = Row::new(
        &format!("ctl:head grid={groups}"),
        &["lm_head"],
        Route::Named("linear_gemv_ishizuki_k6".to_string()),
        groups,
    );
    let loaded = load_row(&device, row, None, None);
    let mut row = loaded.row.clone().loaded_at(loaded.bits);
    row.gbps = Some(measure_row(&queue, &loaded));
    println!(
        "GRIDSWEEP cap={groups} requested={} waves={:.1} {:.1} GB/s {:.1} Gcodes/s",
        groups_1col(head.out_features),
        row.waves(),
        row.gbps(),
        row.gcodes() / 1e9,
    );
}

/// The verdict, as an assertion on measured numbers.
///
/// If `Gcodes/s` is flat across shapes -- which the notes' own ablation-derived
/// numbers say, 338 for the head against 344 for the layers -- then the limiter
/// is per-code work, `GB/s` is just `codes/s * bits/8`, and there is nothing
/// left in parallelism, contiguity or K to exploit. If `Gcodes/s` is *not* flat
/// this fails, and the (a)/(b)/(c) hypotheses have to be re-tested rather than
/// dismissed.
fn assert_bandwidth_is_bpw(rows: &[Row]) {
    let head = rows.iter().find(|r| r.label.starts_with("head")).expect("head row");
    let layers: Vec<&Row> = rows.iter().filter(|r| !r.label.starts_with("head")).collect();
    assert!(!layers.is_empty(), "no layer rows");

    let head_codes = head.gcodes();
    let layer_codes = layers.iter().map(|r| r.gcodes()).sum::<f64>() / layers.len() as f64;
    for row in &layers {
        let ratio = row.gcodes() / head_codes;
        println!(
            "  {}: code rate {:.2}x the head's ({:.1} vs {:.1} Gcodes/s)",
            row.label,
            ratio,
            row.gcodes() / 1e9,
            head_codes / 1e9
        );
        assert!(
            (0.7..=1.4).contains(&ratio),
            "{} runs at {:.2}x the head's code rate. A flat code rate -- and so a \
             bandwidth set by bits per code -- does not hold, which means the \
             limiter is shape-dependent after all. That is good news and needs \
             the parallelism and contiguity hypotheses re-tested, not dismissed.",
            row.label,
            ratio
        );
    }
    let layer_gbps = layers.iter().map(|r| r.gbps()).sum::<f64>() / layers.len() as f64;
    println!(
        "\nVERDICT: head {:.1} GB/s vs layer mean {:.1} GB/s = {:.2}x; \
         bits-per-code ratio {:.2}x; code-rate ratio {:.2}x",
        head.gbps(),
        layer_gbps,
        head.gbps() / layer_gbps,
        head.bytes_per_code() / layers[0].bytes_per_code(),
        head_codes / layer_codes,
    );
}