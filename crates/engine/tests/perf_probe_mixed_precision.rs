//! PERF PROBE (investigation, not a regression gate): MIXED PRECISION ACROSS LAYERS.
//!
//! **Question.** The model is uniformly EXL3 3 bpw (`lm_head` 6 bpw). Can a
//! per-layer rate assignment — some layers at 2 bpw, attention at 4, the head
//! elsewhere — reach **20 tok/s (50 ms/step)**, given that the baseline is
//! 62.3 ms/step (16.05–16.10 tok/s), 12.04 GB/token, 193 GB/s achieved, and the
//! best 3-bpw kernel ever measured on this box is 203 GB/s?
//!
//! **Answer, up front, because it is blunt: no. Nothing on the 410 layer tensors
//! helps at all, and the one tensor whose rate does matter is worth 1.9 ms.**
//!
//! Decode GEMV time is set by how many 32-bit ring words the matvec reads per
//! lane per 16x16 tile, and that count is `lane_words(rate)`, not `rate`. So
//! `time ∝ Σ tiles · lane_words` with `tiles = K·N/256`, while `bytes = tiles ·
//! 8·rate`, and **`bytes/rate` cancels exactly**:
//!
//! | rate | lane words/lane (colmajor family) | amp = 32·LW/(8·rate) | time vs 3 bpw | end-to-end GB/s |
//! |---:|---:|---:|---:|---:|
//! | 1 | 2 | 8.00 | **1.00x** | 76 |
//! | 2 | 2 | 4.00 | **1.00x** | 135 |
//! | 3 | 2 | 2.67 | 1.00x (measured) | 193 |
//! | 4 | 2 | 2.00 | **1.00x** (measured 1.068x) | 252 |
//! | 6 | 3 | 2.00 | **1.50x** (measured 1.586x) | 256 |
//!
//! The 4→6 row is the load-bearing one: 1.5 is what the model predicts and 1.586
//! is what the repo's own rate sweep measured at fixed tile count
//! (`spec/decode-3bpw-amplification/spec.md` §2 — 199.4 / 249.0 / 251.4 GB/s on
//! 96 / 128 / 192 B tiles). That is the only independent check available here
//! and it is the one that closes the 3→4 and, by the same identity, the 3→2 end
//! of the axis.
//!
//! `no_integer_rate_assignment_beats_the_current_uniform_3_bpw` searches all
//! 8^6 = 262,144 assignments of one rate per projection group. The 410 layer
//! tensors are a **plateau over rates 1..=4**: at a 6 bpw head, layers at 1, 2, 3
//! or 4 bpw all cost exactly the shipped 62.3 ms. The only rate that moves the
//! clock is `lm_head`, which runs the EXL3 family rather than the column-major
//! one and whose per-tile cost is genuinely rate-dependent (124.6 / 175.4 /
//! 178.9 / 156.1 GB/s at 3 / 4 / 6 / 8 bpw, `linear.metal:4655`): 6 → 4 bpw is
//! worth **1.89 ms, 16.05 → 16.55 tok/s**. Best case overall is **16.55 tok/s
//! against a 20 tok/s target**, still 10.4 ms short.
//!
//! **This is corroborated, not invented.** `spec/decode-19tok-repacked-250gb/
//! findings/02-2bpw-shuffle-refuted.md:15-16` reached the same identity
//! independently ("`rate(ka) ∝ ka / lane_words`, so `time ∝ bytes/rate ∝
//! (K·N·ka/8)/(ka/LW) = K·N·LW/8`. **bpw cancels exactly.**") and
//! `spec/decode-3bpw-amplification/spec.md:70-73` states the amplification form
//! `64/(8·KA)`. This file adds (a) the exhaustive search, (b) the exact byte map
//! including the *full-attention* layers, which are the **largest** layers and not
//! the smallest, and (c) the format-support answer to "is this even possible".
//!
//! ---
//!
//! **The format already supports per-module rates. The loader refuses them.**
//! This is the genuinely new engineering finding, and it is positive.
//!
//! `quantization_config.json` carries a **`bits_per_weight` on every individual
//! module**, not just the global `bits` (`Gxma-repacked/quantization_config.json`
//! line 9: `"lm_head": { "bits_per_weight": 6, ... }` against a global
//! `"bits": 3.0`). The engine's own parser exposes it per module
//! (`config.rs:446`, `TensorStorage::bits_per_weight`) and
//! `TensorStorage::validate` already range-checks *each* one
//! (`config.rs:463-470`). So the on-disk format is mixed-precision-capable and
//! the data model models it.
//!
//! Uniformity is imposed by **three** places, all in the loader, and all of them
//! compare a module's own `bits_per_weight` against the global and refuse a
//! mismatch:
//!
//! | site | what it does |
//! |---|---|
//! | `safetensors.rs:577-588` (`validate_storage`) | `expected_bits = ... else if module.starts_with("model.language_model.layers.") { quantization.bits }`, then `if bits != expected_bits { Err }` |
//! | `safetensors.rs:942` (`linear_dimensions`) | `Some((k, n, quantization.bits as f64))` — feeds the trellis-shape check |
//! | `model.rs:3357` | `let layer_rate = integer_rate(metadata.quantization.bits, "quantization.bits")?;` — **one** rate computed before the layer loop and passed to all 7 projections of all 60 layers |
//!
//! `model.rs:3351-3356` says out loud that it reads rates from the checkpoint and
//! not from constants, and that this was changed to make a 4-bpw export
//! loadable. It changed *where* the number comes from (metadata instead of a
//! literal 3) but not *how many* numbers there are. Turning uniformity off is
//! therefore a three-line change, not a format change.
//!
//! **It is still the wrong lever, and the reason is not plumbing.** With the
//! gates lifted, a 2-bpw layer on this checkpoint would (i) fail to load,
//! correctly, because `repack_model.rs:162` only transposes tensors with
//! `shape[2] == 48`, and (ii) if forced through, dispatch a **row-major** kernel
//! over **column-major** weights and return finite, plausible, wrong numbers with
//! no error at all. See
//! `a_non_3_bpw_layer_would_run_a_row_major_kernel_over_column_major_weights`.
//!
//! **What these tests prove, and what they do not.**
//!
//! - **Proven by arithmetic on this repo's own sources and checkpoint:** the byte
//!   map, the three loader gates, the colmajor-only-at-3-bpw kernel coverage,
//!   the row-major fallthrough, the `Integer(3)` fast-path gates, the
//!   repacker's 48-u16 hardcode, and the lane-load identity including the
//!   exhaustive search over rate assignments.
//! - **Not measured here:** anything requiring the GPU. No test in this file
//!   creates a Metal device, loads the model, or takes the GPU lock, so the whole
//!   file is safe to run concurrently with a benchmark — a probe that perturbs
//!   what it measures is worse than no probe (AGENTS.md §10).
//! - **Not decided here:** quality. Every mixed-precision scheme *changes
//!   greedy output by definition*, and AGENTS.md forbids quality loss. See
//!   `quality_is_a_user_decision_not_an_engineering_one` — that test is a
//!   statement, not a measurement, and it is the most important thing in this
//!   file.
//!
//! Commands:
//!
//! ```text
//! cargo test -p kaplumba-engine --test perf_probe_mixed_precision -- --nocapture
//! ```
//!
//! (reads `~/.kaplumba/models/Gxma-repacked/quantization_config.json` through the
//! engine's own parser if present; the checkpoint-dependent test skips cleanly,
//! with a printed reason, if it is not.)
//!
//! ---
//!
//! ## Sabotage evidence (AGENTS.md §3.4)
//!
//! Every source-pinned claim was broken and observed to fail. The failures
//! observed, and the message each produced:
//!
//! | sabotage | test | observed failure |
//! |---|---|---|
//! | rename `_k3_colmajor` to `_k2_colmajor` in `linear_repacked.metal` | `only_rate_3_has_a_colmajor_decode_kernel` | `the live column-major family must contain rate 3 only; found {2, 3}` |
//! | `if bits != expected_bits` -> `if false` in `safetensors.rs` | `the_loader_forces_uniformity_in_exactly_three_places` | `safetensors.rs:584 rejection gate changed` |
//! | read `layer_rate` from a per-module field in `model.rs:3357` | `the_loader_forces_uniformity_in_exactly_three_places` | `model.rs:3357 single global layer rate changed` |
//! | widen the repacker predicate to `shape[2] == 48 \|\| == 64` | `the_repacker_only_transposes_3_bpw_tensors` | `the repacker's 3-bpw-only trellis predicate must appear exactly twice` |
//! | the same widened to `\|\| == 32` | `the_repacker_only_transposes_3_bpw_tensors` | `a 2-bpw width appeared in the repacker` |
//! | replace one `matches!(o_proj.rate, Integer(3))` with `&& true` | `fused_fast_paths_are_gated_on_integer_3` | gate count fell below the pinned floor |
//! | `codebook_multiplier = 4` -> `8` in `exl3_bytes` | `every_checkpoint_tensor_matches_the_byte_formula` | `mlp.gate_proj (sliding): formula disagrees with the checkpoint's n_bytes` |
//! | `Colmajor if bits <= 4 => 2` -> `bits <= 2` | five tests at once | flatness assertions fire in `lane_load_cost_is_flat_across_rates_1_to_4`, `the_cost_is_separable_so_a_finer_scheme_cannot_win`, `no_integer_rate_assignment_beats_the_current_uniform_3_bpw`, `as_the_code_stands_today_only_rate_3_is_even_correct`, `the_bytes_over_bandwidth_arithmetic_that_reaches_20_tok_s_is_invalid` |
//! | drop `(7, 10)` — the full-attention `q_proj` — from group 0 | `the_cost_is_separable_so_a_finer_scheme_cannot_win` | `411 quantized tensors, no more, no fewer` |
//!
//! The last one is the one worth keeping: an earlier revision of this file had
//! exactly that bug, which silently deleted 0.703 GB/token (5.8% of the budget)
//! and made every proposal look 5.8% cheaper than it is. The `counted == 411`
//! assertion exists because that failure was invisible.
//!
//! One assertion was **too weak** and the sabotage table caught it: counting the
//! substring `"shape[2] == 48"` survived widening the repacker predicate,
//! because the widened expression still contains it. It is now pinned whole.

use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Constants. Every number here is either a `config.json` field of the shipped
// checkpoint or a byte count read out of its `quantization_config.json`.
// ---------------------------------------------------------------------------

const LAYERS: usize = 60;
/// `config.json:layer_types` — indices 5, 11, ... 59 are `full_attention`, and
/// `config.rs:241-245` re-derives the same rule as `index % 6 == 5`.
const FULL_LAYERS: usize = 10;
const SLIDING_LAYERS: usize = LAYERS - FULL_LAYERS;

const HIDDEN: usize = 5376;
const INTERMEDIATE: usize = 21_504;
/// `num_attention_heads * head_dim` on a sliding layer.
const Q_OUT_SLIDING: usize = 8192;
/// `num_key_value_heads * head_dim` on a sliding layer.
const KV_OUT_SLIDING: usize = 4096;
/// `num_attention_heads * global_head_dim` on a full layer.
const Q_OUT_FULL: usize = 16_384;
/// `num_global_key_value_heads * global_head_dim` on a full layer.
const KV_OUT_FULL: usize = 2048;
const VOCAB: usize = 262_144;

const LAYER_BPW: u8 = 3;
const HEAD_BPW: u8 = 6;

/// Baseline of record for this investigation (the brief's figure, and
/// `spec/decode-3bpw-amplification/implementation_notes.md:296` records the
/// same 62.3 ms / 16 tok/s neighbourhood).
const BASELINE_MS: f64 = 62.3;
const BASELINE_TOK_S: f64 = 1000.0 / BASELINE_MS;
/// The best 3-bpw GEMV microbenchmark ever measured on this box,
/// `spec/decode-25tok-peak-bandwidth/implementation_notes.md:425`.
const BEST_3BPW_GB_S: f64 = 203.0;
/// What the brief calls the target bandwidth. Pure-streaming 240 GB/s has never
/// been achieved; 273 GB/s is the wire peak.
const ASPIRATIONAL_GB_S: f64 = 250.0;

// ---------------------------------------------------------------------------
// 1. The byte map, derived from one exact formula rather than transcribed.
// ---------------------------------------------------------------------------

/// Bytes one EXL3 module occupies on disk, exactly as the checkpoint stores it.
///
/// Verified against every distinct `(shape, rate)` in the checkpoint by
/// `every_checkpoint_tensor_matches_this_formula`; there is no fudge factor.
///
/// * trellis — `(in/16) x (out/16) x (16·bits)` **u16**. For the colmajor
///   layout `safetensors.rs:894-898` stores it as `[out/16, in/16, 16·bits]`,
///   i.e. the first two axes are swapped; the byte count is unaffected.
/// * `suh` — one f16 per in-feature. `svh` — one f16 per out-feature.
/// * `mcg` — one i32 codebook multiplier (`MCG_MULTIPLIER`).
fn exl3_bytes(in_features: usize, out_features: usize, bits: u8) -> u64 {
    let tiles = (in_features / 16) * (out_features / 16);
    let trellis = tiles * (16 * bits as usize) * 2;
    let input_signs = 2 * in_features;
    let output_signs = 2 * out_features;
    let codebook_multiplier = 4;
    (trellis + input_signs + output_signs + codebook_multiplier) as u64
}

/// One quantized projection class, shared by `layers` layers.
#[derive(Debug, Clone, Copy)]
struct Projection {
    name: &'static str,
    in_features: usize,
    out_features: usize,
    layers: usize,
    bits: u8,
}

impl Projection {
    /// `tiles = K·N/256`: the count of 16x16 weight tiles, which is what the
    /// lane-load cost is proportional to.
    fn tiles(&self) -> u64 {
        ((self.in_features / 16) * (self.out_features / 16)) as u64
    }
    fn bytes_each(&self) -> u64 {
        exl3_bytes(self.in_features, self.out_features, self.bits)
    }
    fn bytes_total(&self) -> u64 {
        self.bytes_each() * self.layers as u64
    }
    fn effective_bpw(&self) -> f64 {
        let weights = (self.in_features * self.out_features) as f64;
        self.bytes_each() as f64 / weights
    }
}

/// Every distinct quantized tensor in the checkpoint, one row per shape.
///
/// Note the shape split that a "60 identical layers" mental model hides: the ten
/// `full_attention` layers have a **wider** `q_proj`/`o_proj` (16384 vs 8192) and
/// a **narrower** `k_proj` (2048 vs 4096), because `global_head_dim` is 512 and
/// `num_global_key_value_heads` is 4 against `head_dim` 256 and 16 heads. The
/// full-attention layers are therefore *bigger*, which is the opposite of the
/// intuitive guess and the reason a per-layer byte map is worth writing down.
fn projections() -> Vec<Projection> {
    vec![
        Projection {
            name: "q_proj (sliding)",
            in_features: HIDDEN,
            out_features: Q_OUT_SLIDING,
            layers: SLIDING_LAYERS,
            bits: LAYER_BPW,
        },
        Projection {
            name: "k_proj (sliding)",
            in_features: HIDDEN,
            out_features: KV_OUT_SLIDING,
            layers: SLIDING_LAYERS,
            bits: LAYER_BPW,
        },
        Projection {
            name: "v_proj (sliding)",
            in_features: HIDDEN,
            out_features: KV_OUT_SLIDING,
            layers: SLIDING_LAYERS,
            bits: LAYER_BPW,
        },
        Projection {
            name: "o_proj (sliding)",
            in_features: Q_OUT_SLIDING,
            out_features: HIDDEN,
            layers: SLIDING_LAYERS,
            bits: LAYER_BPW,
        },
        Projection {
            name: "mlp.gate_proj",
            in_features: HIDDEN,
            out_features: INTERMEDIATE,
            layers: LAYERS,
            bits: LAYER_BPW,
        },
        Projection {
            name: "mlp.up_proj",
            in_features: HIDDEN,
            out_features: INTERMEDIATE,
            layers: LAYERS,
            bits: LAYER_BPW,
        },
        Projection {
            name: "mlp.down_proj",
            in_features: INTERMEDIATE,
            out_features: HIDDEN,
            layers: LAYERS,
            bits: LAYER_BPW,
        },
        Projection {
            name: "q_proj (full)",
            in_features: HIDDEN,
            out_features: Q_OUT_FULL,
            layers: FULL_LAYERS,
            bits: LAYER_BPW,
        },
        Projection {
            name: "k_proj (full)",
            in_features: HIDDEN,
            out_features: KV_OUT_FULL,
            layers: FULL_LAYERS,
            bits: LAYER_BPW,
        },
        Projection {
            name: "o_proj (full)",
            in_features: Q_OUT_FULL,
            out_features: HIDDEN,
            layers: FULL_LAYERS,
            bits: LAYER_BPW,
        },
        Projection {
            name: "lm_head",
            in_features: HIDDEN,
            out_features: VOCAB,
            layers: 1,
            bits: HEAD_BPW,
        },
    ]
}

/// The exact per-token byte total of the shipped 3 bpw checkpoint.
fn bytes_per_token_at_uniform_3bpw() -> u64 {
    projections().iter().map(|p| p.bytes_total()).sum()
}

// ---------------------------------------------------------------------------
// 2. The lane-load cost model.
// ---------------------------------------------------------------------------

/// Which decode-matvec family executes a tensor.
///
/// The distinction matters enormously and is invisible at the type level:
/// `QuantizedLinear` carries one `rate` and no layout, and the layout is
/// inferred downstream from `permutation_mode`, which `model.rs:3525-3542` sets
/// from `quantization_config.json`'s `layout` field — once, for every tensor in
/// the model, with no per-tensor exception.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    /// `linear_gemv_ishizuki_k3_colmajor*` — the repacked column-major layout
    /// this checkpoint uses. Reads exactly **two** ring words per lane per tile
    /// (`linear_repacked.metal:121`: `const device uint *t0 = ...; const ulong
    /// w0a = t0[j0], w0b = t0[j1];` — there is no `j2`).
    Colmajor,
    /// `linear_gemv_ishizuki_k<N>` and `_w2` — the reference packer's tile-major
    /// layout. Reads **three** (`linear.metal:1853-1859`: `j0`, `j1`, `j2`, and
    /// `column = packed + tn * WORDS; stride = tiles_n * WORDS`).
    RowMajor,
    /// `linear_gemv_exl3_k<N>` — `exl3_decode8` reads a fixed three ring words
    /// per lane whatever `KA` is. Recorded in the kernel's own comment,
    /// `linear.metal:4661-4662`.
    Exl3,
}

/// 32-bit ring words read per lane, per 16x16 tile.
///
/// For `Colmajor` this is an extrapolation, and it is labelled as one: the
/// 3-bpw implementation reads 2 (`linear_repacked.metal:121`), and the
/// 3→4→6 rate sweep at fixed tile count (`spec/decode-3bpw-amplification/spec.md`
/// §2: 199.4 / 249.0 / 251.4 GB/s on 96 / 128 / 192 B tiles, i.e. time ratios
/// 1.000 / 1.067 / 1.485 against 3 bpw) pins 4 bpw at 2 and 6 bpw at 3, which is
/// what the two kN colmajor `_bench` kernels were written to measure
/// (`linear_repacked.metal:3356`, `:3529`).
///
/// For `RowMajor` and `Exl3` it is read directly off the source above and is
/// rate-independent — which is the whole problem with those families.
fn lane_words(family: Family, bits: u8) -> u32 {
    match family {
        Family::Colmajor if bits <= 4 => 2,
        Family::Colmajor => 3,
        Family::RowMajor => 3,
        Family::Exl3 => 3,
    }
}

/// Per-tile cost of the **EXL3** family, in arbitrary units.
///
/// This family does *not* have a rate-invariant cost, and that is the one place
/// where a bit-rate change buys time. `linear.metal:4655-4659` records the
/// repo's own equal-bytes sweep of it (`examples/gemv_shape.rs`, 43.4 MB held
/// constant, rate the only thing that moved):
///
/// ```text
///   3 bpw 124.6 GB/s    4 bpw 175.4    6 bpw 178.9    8 bpw 156.1
/// ```
///
/// `GB/s = 8·rate / cost`, so `cost = 8·rate / GB_s` gives the per-tile cost
/// below. Costs are **held flat outside the measured rates, downward only where
/// a lower measured rate is cheaper** — so an unmeasured rate can never be
/// credited with a gain the data does not support. The result is that `lm_head`
/// at 6 bpw is paying ~1.47x the per-tile cost it pays at 4 bpw, which is the
/// single largest bit-rate effect available anywhere in this model.
fn exl3_tile_cost(bits: u8) -> f64 {
    let measured = [(3u8, 124.6f64), (4, 175.4), (6, 178.9), (8, 156.1)];
    if let Some((_, gb_s)) = measured.iter().find(|(rate, _)| *rate == bits) {
        return 8.0 * bits as f64 / gb_s;
    }
    // Conservative: an unmeasured rate takes the WORST (most expensive) cost
    // among the measured rates at or below it, so it can never be credited with
    // a gain the data does not support. With nothing below, take the nearest
    // measured rate.
    let below: Vec<(u8, f64)> = measured
        .iter()
        .filter(|(rate, _)| *rate <= bits)
        .map(|(rate, gb_s)| (*rate, 8.0 * *rate as f64 / gb_s))
        .collect();
    if let Some((_rate, cost)) = below.iter().max_by(|a, b| a.1.total_cmp(&b.1)) {
        return *cost;
    }
    let (rate, gb_s) = measured
        .iter()
        .min_by_key(|(rate, _)| rate.abs_diff(bits))
        .copied()
        .expect("the table is never empty");
    8.0 * rate as f64 / gb_s
}

// --- Putting the two families on one scale ---------------------------------
//
// `lane_words` is not comparable across families and `exl3_tile_cost` is not
// either: the colmajor family is tuned for this layout and the EXL3 family is
// not. Both are anchored on the repo's own rate sweeps, converted to ms per
// 16x16 tile, so that a rate change on one family is not silently free and a
// rate change on the other is not silently charged the wrong price.
//
// colmajor, `spec/decode-3bpw-amplification/spec.md` §2: 3 bpw, 96 B tiles,
// 199.4 GB/s  ->  96 / 199.4e9 s  =  4.814e-7 ms/tile
// EXL3,      `linear.metal:4655`: 6 bpw, 192 B tiles, 178.9 GB/s  ->  1.0732e-6
//
// Sanity check of the resulting split against production
// (`spec/decode-25tok-peak-bandwidth/implementation_notes.md` §18.2): layers
// 55.1 ms + head 5.9 ms = 61.0 ms of GEMV in a 62.3 ms step, i.e. the ~1.3 ms
// remainder is attention, norms and cache writes. The head is ~6.7% of the
// step in the measurement and ~9.7% in the model, which is the right order and
// is the reason a head-rate change is worth pricing at all.

const COLMAJOR_MS_PER_TILE_AT_3BPW: f64 = 96.0 / 199.4e9 * 1e3;
const EXL3_MS_PER_TILE_AT_6BPW: f64 = 192.0 / 178.9e9 * 1e3;

/// Lane-load amplification: how many words are requested per word of useful
/// weight data. `32 lanes x lane_words` requested over `8 x bits` useful.
fn amplification(family: Family, bits: u8) -> f64 {
    32.0 * lane_words(family, bits) as f64 / (8.0 * bits as f64)
}

/// The independently-rateable groups of the model.
///
/// Six groups, each a set of projections that share one rate. Grouping by
/// projection *role* rather than by projection is what makes the group
/// assignment exhaustive in the only sense that matters: the cost function
/// `ms_per_tile(Colmajor, rate)` does not depend on the shape, so the total cost
/// is `Σ_g tiles_g · cost(rate_g)` and any finer partition of the 410 tensors
/// attains its minimum at the same per-group minimum. That separability is
/// asserted directly in `the_cost_is_separable_so_a_finer_scheme_cannot_win`.
///
/// Note that `q_proj`, `k_proj` and `o_proj` each span TWO shapes, because the
/// ten full-attention layers are wider in q/o and narrower in k. Dropping the
/// second shape — which an earlier revision of this file did — silently removes
/// 0.703 GB/token, i.e. 5.8% of the budget.
const GROUPS: usize = 6;

/// Indices into [`projections`], with multiplicity, that make up each group.
fn group_members(group: usize) -> &'static [(usize, u64)] {
    match group {
        0 => &[(0, 50), (7, 10)],          // q_proj: sliding + full
        1 => &[(1, 50), (8, 10)],          // k_proj: sliding + full
        2 => &[(2, 50)],                   // v_proj: sliding only (full layers are k_eq_v)
        3 => &[(3, 50), (9, 10)],          // o_proj: sliding + full
        4 => &[(4, 60), (5, 60), (6, 60)], // mlp gate/up/down
        _ => &[(10, 1)],                   // lm_head
    }
}

fn group_tiles(group: usize) -> u64 {
    let all = projections();
    group_members(group)
        .iter()
        .map(|(index, copies)| all[*index].tiles() * copies)
        .sum()
}

fn group_name(group: usize) -> &'static str {
    match group {
        0 => "q_proj",
        1 => "k_proj",
        2 => "v_proj",
        3 => "o_proj",
        4 => "mlp.gate/up/down",
        _ => "lm_head",
    }
}

/// Per-tile wall-clock cost in ms, in a scale common to both families.
fn ms_per_tile(family: Family, bits: u8) -> f64 {
    match family {
        Family::Colmajor => {
            COLMAJOR_MS_PER_TILE_AT_3BPW * lane_words(Family::Colmajor, bits) as f64 / 2.0
        }
        Family::RowMajor => COLMAJOR_MS_PER_TILE_AT_3BPW * 3.0 / 2.0,
        Family::Exl3 => EXL3_MS_PER_TILE_AT_6BPW * exl3_tile_cost(bits) / exl3_tile_cost(6),
    }
}

/// GEMV time in ms for a family/rate assignment.
fn gemv_ms_with(families: &[Family; GROUPS], rates: [u8; GROUPS]) -> f64 {
    (0..GROUPS)
        .map(|g| group_tiles(g) as f64 * ms_per_tile(families[g], rates[g]))
        .sum()
}

/// The family assignment of the shipped checkpoint: 410 layer linears on the
/// column-major family, `lm_head` on the EXL3 family (its trellis lands at
/// `packed_offset % 4 == 2`, which `ishizuki_gemv_pipeline` refuses —
/// `spec/decode-3bpw-amplification/spec.md` §6.1).
const BASELINE_FAMILIES: [Family; GROUPS] = [
    Family::Colmajor,
    Family::Colmajor,
    Family::Colmajor,
    Family::Colmajor,
    Family::Colmajor,
    Family::Exl3,
];

/// Baseline GEMV time: uniform 3 bpw layers, 6 bpw head.
fn baseline_gemv_ms() -> f64 {
    gemv_ms_with(&BASELINE_FAMILIES, [3, 3, 3, 3, 3, HEAD_BPW])
}

/// Non-GEMV time in the measured 62.3 ms step: attention, norms, KV writes,
/// dispatch. Taken as the remainder so the model reproduces the measurement and
/// then prices changes against it.
fn non_gemv_ms() -> f64 {
    BASELINE_MS - baseline_gemv_ms()
}

/// Modelled step time in ms for a family/rate assignment.
///
/// Calibrated, not absolute: the GEMV part comes from the two rate sweeps and
/// the non-GEMV part is the measured remainder, so `ms(3,3,3,3,3,6)` reproduces
/// the 62.3 ms baseline by construction and every other assignment is read as a
/// delta on a measurement rather than as a prediction from first principles.
fn modelled_ms_with(families: &[Family; GROUPS], rates: [u8; GROUPS]) -> f64 {
    gemv_ms_with(families, rates) + non_gemv_ms()
}

/// The hypothetical in which a column-major kernel existed at every rate, with
/// the EXL3 head priced from its own measured rate curve.
fn modelled_ms(rates: [u8; GROUPS]) -> f64 {
    modelled_ms_with(&BASELINE_FAMILIES, rates)
}

/// The code as it stands: only rate 3 has a column-major kernel, so any other
/// rate on a layer tensor falls through to the row-major family over
/// column-major weights — wrong answers *and* 3 words/lane.
fn modelled_ms_as_it_stands(rates: [u8; GROUPS]) -> f64 {
    let mut families = [Family::Colmajor; GROUPS];
    for (g, family) in families.iter_mut().enumerate() {
        *family = if g == 5 {
            Family::Exl3
        } else if rates[g] == 3 {
            Family::Colmajor
        } else {
            Family::RowMajor
        };
    }
    modelled_ms_with(&families, rates)
}

/// Bytes per token for an assignment of one rate per group.
fn bytes_per_token(rates: [u8; GROUPS]) -> u64 {
    let all = projections();
    let mut total = 0u64;
    for (g, rate) in rates.iter().enumerate() {
        for (index, copies) in group_members(g) {
            let p = all[*index];
            total += exl3_bytes(p.in_features, p.out_features, *rate) * copies;
        }
    }
    total
}

// ---------------------------------------------------------------------------
// 3. Tests: the byte map.
// ---------------------------------------------------------------------------

#[test]
fn every_checkpoint_tensor_matches_the_byte_formula() {
    // The exact `stored_tensors[].n_bytes` sums read out of
    // ~/.kaplumba/models/Gxma-repacked/quantization_config.json, grouped by
    // (full_attention?, projection, rate). Nothing here is transcribed from a
    // report; if `exl3_bytes` is wrong, these fail.
    let observed: &[(&str, usize, usize, u8, u64)] = &[
        (
            "mlp.gate_proj (sliding)",
            HIDDEN,
            INTERMEDIATE,
            3,
            43_405_828,
        ),
        ("mlp.up_proj (sliding)", HIDDEN, INTERMEDIATE, 3, 43_405_828),
        (
            "mlp.down_proj (sliding)",
            INTERMEDIATE,
            HIDDEN,
            3,
            43_405_828,
        ),
        ("q_proj (sliding)", HIDDEN, Q_OUT_SLIDING, 3, 16_542_212),
        ("k_proj (sliding)", HIDDEN, KV_OUT_SLIDING, 3, 8_276_484),
        ("v_proj (sliding)", HIDDEN, KV_OUT_SLIDING, 3, 8_276_484),
        ("o_proj (sliding)", Q_OUT_SLIDING, HIDDEN, 3, 16_542_212),
        ("q_proj (full)", HIDDEN, Q_OUT_FULL, 3, 33_073_668),
        ("k_proj (full)", HIDDEN, KV_OUT_FULL, 3, 4_143_620),
        ("o_proj (full)", Q_OUT_FULL, HIDDEN, 3, 33_073_668),
        ("lm_head", HIDDEN, VOCAB, 6, 1_057_499_652),
    ];
    for (name, k, n, bits, bytes) in observed {
        assert_eq!(
            exl3_bytes(*k, *n, *bits),
            *bytes,
            "{name}: formula disagrees with the checkpoint's n_bytes"
        );
    }
    // And the same shapes at a hypothetical 2 bpw, so the mixed-precision
    // proposals below rest on the formula rather than on a scaled constant.
    for p in projections() {
        let at_2 = exl3_bytes(p.in_features, p.out_features, 2);
        assert!(
            at_2 * 3 < p.bytes_each() * 3,
            "{}: 2 bpw is not smaller than {} bpw",
            p.name,
            p.bits
        );
    }
}

#[test]
fn per_layer_byte_map() {
    let all = projections();
    let mut layer_linear = 0u64;
    let mut mlp = 0u64;
    let mut attention = 0u64;
    let mut head = 0u64;

    println!(
        "\n{:<22} {:>7} {:>9} {:>7} {:>13} {:>15} {:>8}",
        "projection", "K", "N", "layers", "bytes each", "bytes total", "% of tok"
    );
    println!("{}", "-".repeat(88));
    for p in &all {
        let total = p.bytes_total();
        let share = 100.0 * total as f64 / bytes_per_token_at_uniform_3bpw() as f64;
        println!(
            "{:<22} {:>7} {:>9} {:>7} {:>13} {:>15} {:>7.2}%",
            p.name,
            p.in_features,
            p.out_features,
            p.layers,
            p.bytes_each(),
            total,
            share
        );
        if p.name == "lm_head" {
            head = total;
        } else if p.name.starts_with("mlp.") {
            mlp += total;
            layer_linear += total;
        } else {
            attention += total;
            layer_linear += total;
        }
    }
    println!("{}", "-".repeat(88));
    println!(
        "layer linears (3 bpw)          {:>48} {:>7.2}%",
        layer_linear,
        100.0 * layer_linear as f64 / bytes_per_token_at_uniform_3bpw() as f64
    );
    println!(
        "  of which MLP (gate/up/down)  {:>48} {:>7.2}%",
        mlp,
        100.0 * mlp as f64 / bytes_per_token_at_uniform_3bpw() as f64
    );
    println!(
        "  of which attention          {:>48} {:>7.2}%",
        attention,
        100.0 * attention as f64 / bytes_per_token_at_uniform_3bpw() as f64
    );
    println!(
        "lm_head (6 bpw)                {:>48} {:>7.2}%",
        head,
        100.0 * head as f64 / bytes_per_token_at_uniform_3bpw() as f64
    );
    println!(
        "TOTAL per token                {:>48}",
        bytes_per_token_at_uniform_3bpw()
    );

    assert_eq!(layer_linear, 10_997_828_200, "3 bpw layer linears");
    assert_eq!(head, 1_057_499_652, "6 bpw head");
    assert_eq!(bytes_per_token_at_uniform_3bpw(), 12_055_327_852);

    // Per-layer rollup, the form a sensitivity study would consume.
    let sliding = 179_854_876u64;
    let full = 200_508_440u64;
    assert_eq!(sliding, 179_854_876);
    assert_eq!(full, 200_508_440);
    assert_eq!(
        sliding * SLIDING_LAYERS as u64 + full * FULL_LAYERS as u64,
        10_997_828_200
    );
    println!(
        "\nper layer: sliding {:.3} MB x {}   full {:.3} MB x {}",
        sliding as f64 / 1e6,
        SLIDING_LAYERS,
        full as f64 / 1e6,
        FULL_LAYERS
    );
    println!(
        "MLP is {:.1}% of every layer; attention is {:.1}%. {} bpw is {} bpw.",
        100.0 * (3.0 * 43_405_828.0) / sliding as f64,
        100.0 * (sliding - 3 * 43_405_828) as f64 / sliding as f64,
        LAYER_BPW,
        LAYER_BPW
    );

    // Sign tables and the codebook multiplier are the only overhead over the
    // nominal rate, and they are ~0.1%: the headline "3.0 bpw" is honest.
    for p in &all {
        let overhead = p.effective_bpw() - p.bits as f64;
        assert!(
            overhead < 0.25,
            "{}: effective {:.3} bpw, more than 0.25 above nominal {}",
            p.name,
            p.effective_bpw(),
            p.bits
        );
    }
}

#[test]
fn full_attention_layers_are_the_largest_not_the_smallest() {
    // The 10 `full_attention` layers carry no `v_proj` at all
    // (`attention_k_eq_v: true`, `config.json`; the loader skips it at
    // `safetensors.rs:708` and `model.rs:3439-3450`) — which invites the
    // conclusion that they are the *cheap* layers. They are not. Their
    // q_proj/o_proj are 16384 wide against 8192, and only the k_proj shrinks.
    let sliding = 179_854_876u64;
    let full = 200_508_440u64;
    assert!(
        full > sliding,
        "full-attention layer {full} should exceed sliding {sliding}"
    );
    assert_eq!(
        full - sliding,
        2 * 33_073_668 + 4_143_620 - 2 * 16_542_212 - 2 * 8_276_484
    );

    // There is no `v_proj` module for exactly the ten full layers — verified
    // against the real file in `the_checkpoint_format_already_carries_a_per_module_rate`.
    let absent: Vec<usize> = (0..LAYERS).filter(|i| i % 6 == 5).collect();
    assert_eq!(absent, vec![5, 11, 17, 23, 29, 35, 41, 47, 53, 59]);
    assert_eq!(absent.len(), FULL_LAYERS);
}

#[test]
fn total_per_token_bytes_matches_the_measured_12_04_gb() {
    let modelled = bytes_per_token_at_uniform_3bpw() as f64;
    let measured = 12.04e9; // GB/token, the brief's baseline figure
    let error = (modelled - measured).abs() / measured;
    println!(
        "\nmodelled bytes/token {:.0} ({:.3} GB) vs measured {:.2} GB -> {:.2}% apart",
        modelled,
        modelled / 1e9,
        measured / 1e9,
        100.0 * error
    );
    assert!(
        error < 0.005,
        "the byte model must account for the measured 12.04 GB/token to within 0.5%, got {:.2}%",
        100.0 * error
    );
    // 12.055 GB at 62.3 ms is 193.5 GB/s, the brief's achieved figure. The
    // model is therefore complete: there is no unmodelled weight traffic.
    let achieved = modelled / (BASELINE_MS / 1000.0);
    println!("implied achieved bandwidth {:.1} GB/s", achieved / 1e9);
    assert!((achieved / 1e9 - 193.0).abs() < 2.0);
}

// ---------------------------------------------------------------------------
// 4. Tests: does the format support per-layer rates? (the key question)
// ---------------------------------------------------------------------------

fn checkpoint_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let path = PathBuf::from(home)
        .join(".kaplumba")
        .join("models")
        .join("Gxma-repacked");
    path.join("quantization_config.json")
        .is_file()
        .then_some(path)
}

#[test]
fn the_checkpoint_format_already_carries_a_per_module_rate() {
    let Some(dir) = checkpoint_dir() else {
        println!("SKIP: ~/.kaplumba/models/Gxma-repacked not present");
        return;
    };
    // Parsed with the ENGINE'S OWN parser, not serde_json directly: the point is
    // that the engine's data model already carries a per-module rate.
    let cfg =
        kaplumba_engine::config::load_quantization_config(&dir.join("quantization_config.json"))
            .expect("the engine must parse this checkpoint");
    assert_eq!(cfg.bits, 3.0);
    assert_eq!(cfg.head_bits, 6);

    let quantized: Vec<(&String, f64)> = cfg
        .tensor_storage
        .iter()
        .filter(|(_, s)| s.bits_per_weight.is_some())
        .map(|(name, s)| (name, s.bits_per_weight.expect("filtered")))
        .collect();
    assert_eq!(quantized.len(), 411, "410 layer linears + lm_head");

    // THE FINDING: `bits_per_weight` is a per-module field and it already
    // *differs* from the global `bits` for exactly one module. A field that is a
    // verbatim copy of a global scalar could never do that.
    let differing: Vec<&str> = quantized
        .iter()
        .filter(|(_, bits)| *bits != cfg.bits as f64)
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(
        differing,
        vec!["lm_head"],
        "the per-module rate field must already be able to disagree with the global"
    );
    assert_eq!(
        quantized.iter().find(|(n, _)| *n == "lm_head").unwrap().1,
        6.0
    );

    // Every layer module currently agrees with the global — i.e. the field
    // exists, is honoured, and is uniform. That is the whole gap.
    let layer_modules: Vec<&(&String, f64)> = quantized
        .iter()
        .filter(|(name, _)| name.starts_with("model.language_model.layers."))
        .collect();
    assert_eq!(layer_modules.len(), 410);
    assert!(layer_modules.iter().all(|(_, bits)| *bits == 3.0));

    // 60 x 7 - 10 v_proj: the ten `full_attention` layers have no v_proj module,
    // which is what `attention_k_eq_v: true` means on disk.
    let without_v: Vec<u32> = (0..LAYERS as u32)
        .filter(|i| {
            !cfg.tensor_storage
                .contains_key(&format!("model.language_model.layers.{i}.self_attn.v_proj"))
        })
        .collect();
    assert_eq!(without_v, vec![5, 11, 17, 23, 29, 35, 41, 47, 53, 59]);

    // The bf16 leftovers are only norms plus the embedding; the embedding is not
    // read per decode token (lm_head is), so it does not enter the byte budget.
    let embed = cfg
        .tensor_storage
        .get("model.language_model.embed_tokens")
        .and_then(|s| {
            s.stored_tensors
                .get("model.language_model.embed_tokens.weight")
        })
        .map(|t| t.n_bytes)
        .unwrap_or(0);
    assert_eq!(embed, 2_818_572_288);

    println!(
        "\nformat verdict: PER-MODULE bits_per_weight is present on all 411 quantized \
         modules, parsed and range-checked per module by the engine \
         (config.rs:442-470). The format is mixed-precision-capable."
    );
}

const SAFETENSORS_RS: &str = include_str!("../src/safetensors.rs");
const MODEL_RS: &str = include_str!("../src/model.rs");
const CONFIG_RS: &str = include_str!("../src/config.rs");
const LINEAR_RS: &str = include_str!("../../kernels/src/linear.rs");
const LINEAR_METAL: &str = include_str!("../../kernels/src/linear.metal");
const REPACKED_METAL: &str = include_str!("../../kernels/src/linear_repacked.metal");
const REPACK_BIN_RS: &str = include_str!("../src/bin/repack_model.rs");

/// Exact, non-overlapping occurrence count.
///
/// Hand-rolled rather than `str::matches` because the counts are *pinned*: a
/// source edit that adds or removes one of these sites must fail the test rather
/// than silently change what the report claims.
fn occurrences_is_exact(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    let mut count = 0usize;
    let mut from = 0usize;
    while let Some(relative) = haystack[from..].find(needle) {
        count += 1;
        from += relative + needle.len();
    }
    count
}

#[test]
fn the_loader_forces_uniformity_in_exactly_three_places() {
    // Gate 1 — safetensors.rs:577-588. Rejects any layer module whose own
    // `bits_per_weight` disagrees with the global, *after* reading it.
    assert!(
        SAFETENSORS_RS.contains("} else if module.starts_with(\"model.language_model.layers.\") {"),
        "safetensors.rs:580 uniformity gate text changed"
    );
    assert!(
        SAFETENSORS_RS.contains("quantization.bits as f64"),
        "safetensors.rs:580 global-bits comparison changed"
    );
    assert!(
        SAFETENSORS_RS.contains("if bits != expected_bits {"),
        "safetensors.rs:584 rejection gate changed"
    );
    assert_eq!(
        occurrences_is_exact(SAFETENSORS_RS, "if bits != expected_bits {"),
        1,
        "there must be exactly one such gate"
    );

    // Gate 2 — safetensors.rs:942. The trellis-geometry check is handed the
    // global rate rather than the module's, so it would reject a mixed
    // checkpoint's shapes even with gate 1 removed.
    assert!(
        SAFETENSORS_RS.contains("Some((dimensions.0, dimensions.1, quantization.bits as f64))"),
        "safetensors.rs:942 uniform dimension helper changed"
    );

    // Gate 3 — model.rs:3357. ONE rate, computed before the layer loop, shared by
    // every projection of every layer. This is the actual plumbing point:
    // `QuantizedLinear::rate` is per-linear (`model.rs:297`), so the type
    // supports mixed precision and only this call site collapses it.
    assert!(
        MODEL_RS.contains(
            "let layer_rate = integer_rate(metadata.quantization.bits, \"quantization.bits\")?;"
        ),
        "model.rs:3357 single global layer rate changed"
    );
    assert!(
        MODEL_RS.contains("pub rate: Exl3Rate,"),
        "QuantizedLinear::rate must stay per-linear for mixed precision to be possible"
    );
    assert_eq!(
        occurrences_is_exact(MODEL_RS, "layer_rate,"),
        7,
        "one `layer_rate` argument per projection per layer; if this changes, a \
         per-module rate lookup has been threaded through"
    );

    // And the per-module field is ALREADY validated per module — this is the
    // half of the pipeline that needs no change.
    assert!(CONFIG_RS.contains("pub bits_per_weight: Option<f64>,"));
    assert!(
        CONFIG_RS.contains("unsupported EXL3 bitrate {bits} for {module}"),
        "config.rs:466 range-checks the per-module rate; that is what makes the \
         format mixed-precision-capable"
    );

    println!(
        "\nloader verdict: the FORMAT supports per-module rates; three loader sites \
         impose uniformity (safetensors.rs:577-588, safetensors.rs:942, \
         model.rs:3357). config.rs already range-checks each module's own rate."
    );
}

#[test]
fn only_rate_3_has_a_colmajor_decode_kernel() {
    // Every kernel in the repacked (column-major) file, with its rate parsed out
    // of the name. If a `k2_colmajor` or `k4_colmajor` appears, the mixed
    // precision story gets materially better and this fails.
    let colmajor: Vec<(String, u8)> = REPACKED_METAL
        .match_indices("linear_gemv_ishizuki_k")
        .filter_map(|(index, _)| {
            let prefix_len = "linear_gemv_ishizuki_k".len();
            let rest = &REPACKED_METAL[index + prefix_len..];
            let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
            if digits.is_empty() {
                return None;
            }
            let after_rate = &rest[digits.len()..];
            if !after_rate.starts_with("_colmajor") {
                return None;
            }
            // The whole identifier, so a `_bench` suffix is visible to the filter
            // below. A kernel name ends at the first non-identifier character.
            let tail: String = after_rate
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            Some((
                format!("linear_gemv_ishizuki_k{digits}{tail}"),
                digits.parse::<u8>().expect("kernel name has a rate"),
            ))
        })
        .collect();

    assert!(!colmajor.is_empty(), "no colmajor kernels found");
    // `_bench` variants exist for 4 and 6 but are never on the dispatch table
    // (`LinearKernels::new` gives a rate slot to the first *unsuffixed* kernel
    // it sees, `linear.rs:456-467`), so they are excluded here: only the
    // kernels a rate assignment could actually reach count.
    let live: Vec<(String, u8)> = colmajor
        .iter()
        .filter(|(name, _)| !name.ends_with("_bench"))
        .cloned()
        .collect();
    let live_rates: std::collections::BTreeSet<u8> = live.iter().map(|(_, b)| *b).collect();
    assert_eq!(
        live_rates,
        [3u8].into_iter().collect(),
        "the live column-major family must contain rate 3 only; found {live_rates:?}"
    );
    let live_names: std::collections::BTreeSet<&str> =
        live.iter().map(|(n, _)| n.as_str()).collect();
    println!(
        "live colmajor kernels ({}): {}",
        live_names.len(),
        live_names.into_iter().collect::<Vec<_>>().join(", ")
    );
    println!(
        "non-live colmajor `_bench` names: {}",
        colmajor
            .iter()
            .filter(|(name, _)| name.ends_with("_bench"))
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );

    // The two `_bench` extras exist for 4 and 6 but are named `_bench`, so the
    // rate table does not pick them up and they were never on a hot path.
    assert!(REPACKED_METAL.contains("linear_gemv_ishizuki_k4_colmajor_contig4_bench"));
    assert!(REPACKED_METAL.contains("linear_gemv_ishizuki_k6_colmajor_contig4_bench"));

    // Contrast: the EXL3 fallback family has every rate 1..=8 (linear.rs:491-494),
    // and the row-major ishizuki family has k1..k8 (linear.metal:1598..2382).
    // So "every rate has a kernel" is true and irrelevant: the wrong layout is
    // worse than no kernel.
    for bits in 1..=8u32 {
        assert!(
            LINEAR_RS.contains(&format!("linear_gemv_exl3_k{bits}{{suffix}}{{family}}"))
                || LINEAR_RS.contains("for bits in 1..=8_u32"),
            "EXL3 family no longer covers rate {bits}"
        );
    }
    assert!(LINEAR_RS.contains("for bits in 1..=8_u32"));

    println!(
        "kernel verdict: colmajor decode family = rate 3 ONLY. EXL3 fallback = 1..=8. \
         Row-major ishizuki = k1..k8. A 2-bpw layer has no correct kernel."
    );
}

#[test]
fn a_non_3_bpw_layer_would_run_a_row_major_kernel_over_column_major_weights() {
    // The silent-wrong-answer path, traced through
    // `LinearKernels::ishizuki_gemv_pipeline` (linear.rs:1508-1560) as
    // `encode_gemv` would call it with `perm_mode == 1` (set for every tensor by
    // `model.rs:3525-3542` because `layout == "colmajor"`):
    //
    //   step 1: `linear_gemv_ishizuki_k2_colmajor_contig4`  -> absent (test above)
    //   step 2: `linear_gemv_ishizuki_k2_colmajor`           -> absent
    //   step 3: `linear_gemv_ishizuki_k2_w2`                 -> PRESENT, and ROW-MAJOR
    //
    // Step 3 is not gated on `perm_mode`. `linear.metal:8571` defines it with
    // `stride = tiles_n * WORDS` (tile-major) where `linear_repacked.metal:111`
    // uses `stride = WORDS` with `column = packed + tn * (tiles_k * WORDS)`
    // (column-major). Same bytes, transposed indexing.
    //
    // The failure is not an error. The kernel runs, returns finite values, and
    // every output is wrong — exactly the class of bug
    // `rate4_8_kernel_coverage.rs:11-15` was written to prevent for rates.
    let row_major_marker = "const uint stride = tiles_n * WORDS;";
    let col_major_marker = "const uint stride = WORDS;";

    assert!(
        LINEAR_METAL.contains("kernel void linear_gemv_ishizuki_k2_w2("),
        "the row-major 2-bpw fallthrough kernel is gone; re-check this claim"
    );
    assert!(
        LINEAR_METAL.contains(row_major_marker),
        "row-major stride form changed"
    );
    assert!(
        REPACKED_METAL.contains(col_major_marker),
        "colmajor stride form changed"
    );

    // The fallthrough is unconditional: the `_w2` lookup sits outside the
    // `perm_mode == 1` guard, so a colmajor tensor at a rate with no colmajor
    // kernel takes it.
    let pipeline_body = LINEAR_RS
        .split("fn ishizuki_gemv_pipeline(")
        .nth(1)
        .and_then(|rest| {
            rest.split("/// Slot of a `(codebook, aligned, bits)`")
                .next()
        })
        .expect("ishizuki_gemv_pipeline body");
    assert!(pipeline_body.contains("_colmajor_contig4"));
    assert!(
        pipeline_body.contains("linear_gemv_ishizuki_k{bits}_w2"),
        "the unguarded row-major `_w2` fallthrough changed"
    );
    let guard = pipeline_body
        .find("if perm_mode == 1 || env_flag(\"KAPLUMBA_REPACKED_GEMV\") {")
        .expect("perm_mode guard");
    let w2 = pipeline_body
        .find("linear_gemv_ishizuki_k{bits}_w2")
        .expect("_w2 lookup");
    assert!(guard < w2, "the colmajor branch must still open the lookup");

    println!(
        "\nSILENT-WRONG-ANSWER PATH: a 2-bpw (or 4/5/6/7/8-bpw) layer on this \
         colmajor checkpoint dispatches `linear_gemv_ishizuki_k2_w2`, which strides \
         by `tiles_n * WORDS` instead of `WORDS`. Finite, plausible, wrong. There is \
         no error path and no layout check on the rate."
    );
    println!(
        "Cost consequence: the row-major family reads 3 ring words per lane where \
         the colmajor family reads 2, so a 2-bpw MLP costs 1.50x what the 3-bpw MLP \
         costs today, for 2/3 of the bytes."
    );
}

#[test]
fn fused_fast_paths_are_gated_on_integer_3() {
    // Even if a colmajor kernel existed at the new rate, these gates would drop
    // the layer onto a slower path. Counted so that a change is visible.
    let engine_gates = occurrences_is_exact(MODEL_RS, "Exl3Rate::Integer(3)");
    let kernel_gates = occurrences_is_exact(LINEAR_RS, "Exl3Rate::Integer(3)");
    println!(
        "`matches!(.., Exl3Rate::Integer(3))` sites: engine {engine_gates}, kernels {kernel_gates}"
    );
    assert!(engine_gates >= 3, "expected the down_proj/o_proj SK guards");
    assert!(kernel_gates >= 1, "expected the encode_gemv `use_sk` guard");
    assert!(MODEL_RS.contains("matches!(self.weights.down_proj.rate, Exl3Rate::Integer(3))"));
    assert!(MODEL_RS.contains("matches!(self.weights.o_proj.rate, Exl3Rate::Integer(3))"));
    assert!(LINEAR_RS.contains("&& matches!(rate, Exl3Rate::Integer(3))"));

    println!(
        "these gate the SK2/SK4 paths for o_proj and down_proj and the cross-layer \
         fusion (model.rs:1283), i.e. the throughput work already banked. A \
         non-3-bpw layer forfeits all of it on top of running the wrong kernel."
    );
}

#[test]
fn the_repacker_only_transposes_3_bpw_tensors() {
    // `layout: "colmajor"` is produced by this offline tool, not by the loader.
    // It hardcodes 48 u16 per tile = 3 bpw, so a 2-bpw (32) or 4-bpw (64)
    // tensor would keep the reference packer's tile-major ordering and then be
    // read by colmajor kernels — the same transposition bug, upstream.
    //
    // The predicate is pinned WHOLE, not by counting `"shape[2] == 48"`:
    // counting the substring survives widening the predicate to
    // `shape[2] == 48 || shape[2] == 64`, which is precisely the change that
    // would make this test lie. (Verified by sabotage — see the module docs.)
    const PREDICATE: &str =
        "entry.name.ends_with(\".trellis\") && entry.shape.len() == 3 && entry.shape[2] == 48";
    assert_eq!(
        occurrences_is_exact(REPACK_BIN_RS, PREDICATE),
        2,
        "the repacker's 3-bpw-only trellis predicate must appear exactly twice"
    );
    assert!(
        !REPACK_BIN_RS.contains("shape[2] == 32"),
        "a 2-bpw width appeared in the repacker"
    );
    assert!(
        !REPACK_BIN_RS.contains("shape[2] == 64"),
        "a 4-bpw width appeared in the repacker"
    );
    // And the output shape it writes is likewise 3-bpw-only.
    assert_eq!(
        occurrences_is_exact(REPACK_BIN_RS, "vec![tiles_n, tiles_k, 48]"),
        1
    );
    println!(
        "repack_model.rs transposes only 3-bpw trellis tensors (48 u16 = 16 x 3) at \
         both of its two sites. A mixed-precision checkpoint cannot be made colmajor \
         by the current repacker."
    );
}

// ---------------------------------------------------------------------------
// 5. Tests: the cost model, and the exhaustive search that closes the axis.
// ---------------------------------------------------------------------------

#[test]
fn the_cost_is_separable_so_a_finer_scheme_cannot_win() {
    // The exhaustive search runs over six rate groups. A reader is entitled to
    // ask whether a scheme that sets a *different* rate on, say, layer 7's MLP
    // than on layer 8's could do better. It cannot, and this is the proof rather
    // than an assertion of it:
    //
    //   total(rates) = non_gemv + head_tiles·cost_exl3(rate_head)
    //                + Σ_{layer tensors} tiles · cost_colmajor(rate)
    //
    // `cost_colmajor` does not depend on the tensor's shape, so splitting any
    // layer group in two replaces `tiles·min_r cost(r)` by
    // `tiles_a·min_r cost(r) + tiles_b·min_r cost(r)`, which is the same number.
    // The minimum over a finer partition therefore equals the minimum over the
    // six groups, and the 262,144-assignment search is exhaustive.
    let min_colmajor = (1..=8u8)
        .map(|b| ms_per_tile(Family::Colmajor, b))
        .fold(f64::INFINITY, f64::min);
    for bits in 1..=4u8 {
        assert_eq!(ms_per_tile(Family::Colmajor, bits), min_colmajor);
    }
    assert!(ms_per_tile(Family::Colmajor, 5) > min_colmajor);

    let mut layer_tiles = 0u64;
    for g in 0..GROUPS - 1 {
        layer_tiles += group_tiles(g);
    }
    let fine_optimum = non_gemv_ms()
        + layer_tiles as f64 * min_colmajor
        + group_tiles(5) as f64
            * (1..=8u8)
                .map(|b| ms_per_tile(Family::Exl3, b))
                .fold(f64::INFINITY, f64::min);
    let searched_optimum = modelled_ms([1, 1, 1, 1, 1, 4]);
    assert!(
        (fine_optimum - searched_optimum).abs() < 1e-9,
        "the separable minimum {fine_optimum} must equal the searched minimum \
         {searched_optimum}"
    );

    // And the tile accounting is complete: every one of the 411 quantized
    // tensors is in exactly one group.
    let mut counted = 0u64;
    for g in 0..GROUPS {
        counted += group_members(g)
            .iter()
            .map(|(_, copies)| *copies)
            .sum::<u64>();
    }
    assert_eq!(counted, 411, "411 quantized tensors, no more, no fewer");
    println!(
        "\nlayer tiles {} + head tiles {} = {}; total per-token bytes {}",
        layer_tiles,
        group_tiles(5),
        layer_tiles + group_tiles(5),
        bytes_per_token([3, 3, 3, 3, 3, HEAD_BPW])
    );
    for g in 0..GROUPS {
        println!(
            "  group {g} {:<20} tiles {:>12}",
            group_name(g),
            group_tiles(g)
        );
    }
}

#[test]
fn lane_load_cost_is_flat_across_rates_1_to_4() {
    println!(
        "\n{:<6} {:>12} {:>10} {:>14} {:>16}",
        "rate", "lane words", "amp", "time vs 3", "eff GB/s (end-to-end)"
    );
    println!("{}", "-".repeat(64));
    for bits in 1..=8u8 {
        let lw = lane_words(Family::Colmajor, bits);
        let time_ratio = lw as f64 / lane_words(Family::Colmajor, LAYER_BPW) as f64;
        // bytes/token at this rate for the 410 layer linears, so GB/s follows
        let rates = [bits, bits, bits, bits, bits, HEAD_BPW];
        let bytes = bytes_per_token(rates) as f64;
        let gbs = bytes / (modelled_ms(rates) / 1000.0) / 1e9;
        println!(
            "{:<6} {:>12} {:>10.2} {:>14.3} {:>16.1}",
            bits,
            lw,
            amplification(Family::Colmajor, bits),
            time_ratio,
            gbs
        );
    }

    for bits in 1..=4u8 {
        assert_eq!(
            lane_words(Family::Colmajor, bits),
            lane_words(Family::Colmajor, LAYER_BPW),
            "{bits} bpw must cost the same lane-loads as 3 bpw"
        );
        assert_eq!(
            amplification(Family::Colmajor, bits),
            8.0 / bits as f64,
            "amp = 4*LW/KA = 8/{bits} with LW = 2"
        );
    }
    for bits in 5..=8u8 {
        assert_eq!(lane_words(Family::Colmajor, bits), 3);
    }
    assert_eq!(lane_words(Family::Colmajor, 6) as f64 / 2.0, 1.5);

    // Cross-check the model against the ONE rate transition the repo measured at
    // fixed tile count (spec/decode-3bpw-amplification/spec.md §2).
    //
    //   3 bpw 96 B/tile @ 199.4 GB/s ; 4 bpw 128 B @ 249.0 ; 6 bpw 192 B @ 251.4
    //   time ratio 4 vs 3 = (128/249.0) / (96/199.4) = 1.067   model 1.000
    //   time ratio 6 vs 3 = (192/251.4) / (96/199.4) = 1.590   model 1.500
    let measured_4: f64 = (128.0 / 249.0) / (96.0 / 199.4);
    let measured_6: f64 = (192.0 / 251.4) / (96.0 / 199.4);
    println!(
        "\nmodel vs the repo's fixed-tile-count rate sweep:\n  4 vs 3 bpw: model \
         1.000, measured {measured_4:.3}  (delta {:.1}%)\n  6 vs 3 bpw: model \
         1.500, measured {measured_6:.3}  (delta {:.1}%)",
        100.0 * (measured_4 - 1.0).abs(),
        100.0 * (measured_6 - 1.5).abs() / 1.5
    );
    assert!(
        (measured_4 - 1.0).abs() < 0.10,
        "4 bpw should be time-neutral; measured {measured_4}"
    );
    assert!(
        (measured_6 - 1.5).abs() < 0.15,
        "6 bpw should cost 1.5x; measured {measured_6}"
    );
}

#[test]
fn no_integer_rate_assignment_beats_the_current_uniform_3_bpw() {
    // Exhaustive over one rate per group, 8^6 = 262,144 assignments. Groups:
    // q, k, v, o, mlp(x3), lm_head. `lane_words` is the only thing the rate
    // changes, so this is a complete search of the mixed-precision space that
    // the EXL3 format and this kernel family can express.
    let mut best_ms = f64::INFINITY;
    let mut best: Vec<[u8; GROUPS]> = Vec::new();
    let mut bytes_at_best = u64::MAX;
    let uniform3 = modelled_ms([3, 3, 3, 3, 3, HEAD_BPW]);

    let mut rates = [1u8; GROUPS];
    loop {
        let ms = modelled_ms(rates);
        if ms < best_ms - 1e-9 {
            best_ms = ms;
            best.clear();
            best.push(rates);
            bytes_at_best = bytes_per_token(rates);
        } else if (ms - best_ms).abs() < 1e-9 {
            best.push(rates);
            bytes_at_best = bytes_at_best.min(bytes_per_token(rates));
        }
        // odometer
        let mut carry = true;
        for slot in rates.iter_mut() {
            if *slot < 8 {
                *slot += 1;
                carry = false;
                break;
            }
            *slot = 1;
        }
        if carry {
            break;
        }
    }

    println!("\nsearched 262144 assignments over rates 1..=8 for (q,k,v,o,mlp,head)");
    println!(
        "best modelled step time {best_ms:.3} ms = {:.3} tok/s",
        1000.0 / best_ms
    );
    println!(
        "uniform 3 bpw today   {uniform3:.3} ms = {:.3} tok/s",
        1000.0 / uniform3
    );
    println!("assignments attaining the minimum: {}", best.len());
    println!(
        "  layer-group rates present in the minimising set: {:?}  (flat = rate-invariant)",
        (1..=8u8)
            .filter(|b| best.iter().any(|a| a[..GROUPS - 1].contains(b)))
            .collect::<Vec<_>>()
    );
    println!(
        "  head rates present in the minimising set:      {:?}",
        (1..=8u8)
            .filter(|b| best.iter().any(|a| a[GROUPS - 1] == *b))
            .collect::<Vec<_>>()
    );
    println!(
        "  smallest bytes/token among minimisers: {:.3} GB (baseline {:.3} GB)",
        bytes_at_best as f64 / 1e9,
        bytes_per_token_at_uniform_3bpw() as f64 / 1e9
    );

    // The whole gain over uniform 3 bpw comes from `lm_head`. Prove that by
    // construction: hold the head at its shipped 6 bpw, let every layer group
    // take any rate in 1..=4, and the time does not move at all.
    for rates in [[1u8; 5], [2; 5], [3; 5], [4; 5]] {
        let mut full = [0u8; GROUPS];
        full[..GROUPS - 1].copy_from_slice(&rates);
        full[GROUPS - 1] = HEAD_BPW;
        assert!(
            (modelled_ms(full) - BASELINE_MS).abs() < 1e-9,
            "layer rates {rates:?} at a 6 bpw head must cost exactly today's \
             {:.3} ms, got {:.3}",
            BASELINE_MS,
            modelled_ms(full)
        );
    }
    // And the separable optimum the proof in
    // `the_cost_is_separable_so_a_finer_scheme_cannot_win` predicts.
    assert!((best_ms - modelled_ms([1, 1, 1, 1, 1, 4])).abs() < 1e-9);

    // The five layer groups are a plateau over 1..=4; only `lm_head` has a rate
    // that affects time, and its cheapest measured per-tile cost is at 4 bpw.
    let fastest_head = (1..=8u8)
        .min_by(|a, b| exl3_tile_cost(*a).total_cmp(&exl3_tile_cost(*b)))
        .expect("non-empty");
    assert_eq!(
        fastest_head, 4,
        "the EXL3 rate curve's cheapest point moved"
    );
    assert!(best.iter().all(|a| a[GROUPS - 1] == fastest_head));
    println!(
        "\nVERDICT: the 410 layer tensors are a plateau over rates 1..=4 — every \
         integer-rate mixed-precision scheme on them is at best exactly as fast as \
         uniform 3 bpw. The only tensor whose rate moves the clock is `lm_head`, \
         and the cheapest measured per-tile cost is at {fastest_head} bpw, worth \
         {:.2} ms = +{:.2} tok/s. 20 tok/s (50 ms) is not among the outcomes; the \
         gap to it is {:.1} ms, and the entire mixed-precision space closes {:.1}% \
         of it.",
        uniform3 - best_ms,
        1000.0 / best_ms - BASELINE_TOK_S,
        best_ms - 50.0,
        100.0 * (uniform3 - best_ms) / (uniform3 - 50.0)
    );
    assert!(
        best_ms > 50.0,
        "the whole mixed-precision space must miss 20 tok/s; got {best_ms} ms"
    );
    // The only reason the best case beats the shipped configuration is the head.
    assert!(
        (best_ms - modelled_ms([3, 3, 3, 3, 3, 4])).abs() < 1e-9,
        "the optimum must be exactly 'whatever the layers like, head at 4 bpw'"
    );
    assert!(
        best_ms > 60.0,
        "even the head at its cheapest rate leaves the step above 60 ms; got {best_ms}"
    );
    // The head effect is real but small: 1.89 ms, which is 15% of the 12.3 ms gap
    // between the shipped configuration and 20 tok/s.
    let head_effect = uniform3 - best_ms;
    assert!(
        (head_effect - 1.89).abs() < 0.05,
        "the whole head effect is one tensor's measured per-tile cost difference, \
         about 1.9 ms; got {head_effect}"
    );
}

#[test]
fn as_the_code_stands_today_only_rate_3_is_even_correct() {
    // Same search, but with the family each group would actually reach TODAY:
    // only rate 3 has a colmajor kernel, so every other rate lands on the
    // row-major (3 words/lane) family over column-major weights — wrong answers
    // AND 1.5x the cost.
    let mut best_ms = f64::INFINITY;
    let mut best_rates: Vec<[u8; GROUPS]> = Vec::new();
    let mut rates = [1u8; GROUPS];
    loop {
        let ms = modelled_ms_as_it_stands(rates);
        if ms < best_ms - 1e-9 {
            best_ms = ms;
            best_rates.clear();
            best_rates.push(rates);
        } else if (ms - best_ms).abs() < 1e-9 {
            best_rates.push(rates);
        }
        let mut carry = true;
        for slot in rates.iter_mut() {
            if *slot < 8 {
                *slot += 1;
                carry = false;
                break;
            }
            *slot = 1;
        }
        if carry {
            break;
        }
    }
    println!(
        "\nas-the-code-stands: best {best_ms:.3} ms = {:.3} tok/s, attained by {} \
         assignment(s)",
        1000.0 / best_ms,
        best_rates.len()
    );
    assert!(
        best_rates
            .iter()
            .all(|a| a[..GROUPS - 1] == [3, 3, 3, 3, 3])
    );
    assert!((best_ms - modelled_ms([3, 3, 3, 3, 3, 4])).abs() < 1e-9);

    // And the concrete cost of the tempting mistake: MLP at 2 bpw today.
    let mlp_at_2 = [3u8, 3, 3, 3, 2, HEAD_BPW];
    let ratio = ms_per_tile(Family::RowMajor, 2) / ms_per_tile(Family::Colmajor, 3);
    println!(
        "MLP at 2 bpw as the code stands: {:.2}x the 3-bpw MLP cost (row-major, \
         3 words/lane) for 2/3 of the bytes -> {:.1} ms, {:.2} tok/s (i.e. 25% SLOWER \
         than shipping 3 bpw, with wrong answers)",
        ratio,
        modelled_ms_as_it_stands(mlp_at_2),
        1000.0 / modelled_ms_as_it_stands(mlp_at_2)
    );
    assert_eq!(ratio, 1.5);
    assert!(
        modelled_ms_as_it_stands(mlp_at_2) > modelled_ms(mlp_at_2) + 15.0,
        "a 2-bpw MLP as the code stands must be far worse than the same bytes would \
         be if a colmajor kernel existed"
    );
    assert!((modelled_ms(mlp_at_2) - BASELINE_MS).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// 6. The trap: the arithmetic that appears to reach 20 tok/s, and why it lies.
// ---------------------------------------------------------------------------

#[test]
fn the_bytes_over_bandwidth_arithmetic_that_reaches_20_tok_s_is_invalid() {
    let mlp_at_2 = [3u8, 3, 3, 3, 2, HEAD_BPW];
    let all_at_2 = [2u8, 2, 2, 2, 2, HEAD_BPW];
    let all_at_2_head_4 = [2u8, 2, 2, 2, 2, 4];

    println!(
        "\n{:<34} {:>10} {:>12} {:>12} {:>12}",
        "scheme", "GB/token", "naive ms", "naive tok/s", "modelled tok/s"
    );
    println!("{}", "-".repeat(84));
    let rows: [(&str, [u8; GROUPS]); 4] = [
        ("uniform 3 bpw (today)", [3, 3, 3, 3, 3, HEAD_BPW]),
        ("MLP -> 2 bpw", mlp_at_2),
        ("all 410 -> 2 bpw", all_at_2),
        ("all 410 -> 2 bpw, head -> 4", all_at_2_head_4),
    ];
    for (name, rates) in rows {
        let bytes = bytes_per_token(rates) as f64;
        let naive_ms = bytes / (BEST_3BPW_GB_S * 1e9) * 1000.0;
        let naive_tps = 1000.0 / naive_ms;
        let modelled_tps = 1000.0 / modelled_ms(rates);
        println!(
            "{:<34} {:>10.3} {:>12.2} {:>12.2} {:>12.2}",
            name,
            bytes / 1e9,
            naive_ms,
            naive_tps,
            modelled_tps
        );
        assert!(
            modelled_tps < 17.0,
            "{name} must not be projected above the measured 3-bpw ceiling"
        );
    }
    println!("{}", "-".repeat(84));

    // The trap, stated exactly.
    let naive_reaches_20 = bytes_per_token(mlp_at_2) as f64 / (BEST_3BPW_GB_S * 1e9) * 1000.0;
    let mlp_at_2_gbs = bytes_per_token(mlp_at_2) as f64 / (modelled_ms(mlp_at_2) / 1000.0) / 1e9;
    println!(
        "\nTHE TRAP: 'MLP at 2 bpw is {:.3} GB/token; at the best-ever 3-bpw kernel of \
         {:.0} GB/s that is {:.1} ms = {:.1} tok/s, so mixed precision reaches 20 \
         tok/s.'",
        bytes_per_token(mlp_at_2) as f64 / 1e9,
        BEST_3BPW_GB_S,
        naive_reaches_20,
        1000.0 / naive_reaches_20
    );
    println!(
        "It is wrong because 203 GB/s is a *3-bpw* bandwidth: it is bytes/time at rate \
         3, and in the 1..=4 range the kernel's cost is set by lane-loads, not bytes. \
         A 2-bpw MLP has the same lane-load count as a 3-bpw MLP, so its time is \
         unchanged and its effective bandwidth falls from 193 to {:.0} GB/s. Bytes and \
         time fall together; tok/s does not move.",
        mlp_at_2_gbs
    );

    // What the same arithmetic gives at the two aspirational bandwidths, so the
    // question is answered on its own terms as well as on the real ones.
    for bw in [BEST_3BPW_GB_S, ASPIRATIONAL_GB_S, 273.0] {
        let ms = bytes_per_token_at_uniform_3bpw() as f64 / (bw * 1e9) * 1000.0;
        println!(
            "uniform 3 bpw at {bw:.0} GB/s: {ms:.1} ms = {:.1} tok/s",
            1000.0 / ms
        );
    }
    println!(
        "so even at the 273 GB/s wire peak, uniform 3 bpw is {:.1} tok/s — the target \
         is out of reach on bytes alone, which is why the mixed-precision question \
         was worth asking.",
        1000.0 / (bytes_per_token_at_uniform_3bpw() as f64 / (273.0 * 1e9) * 1000.0)
    );
}

#[test]
fn proposals_price_bytes_and_time() {
    struct Proposal {
        name: &'static str,
        /// (first layer inclusive, last layer exclusive) whose MLP changes.
        mlp_layers: Option<(usize, usize)>,
        mlp_rate: u8,
        attention_rate: u8,
        head_rate: u8,
        note: &'static str,
    }

    let proposals = [
        Proposal {
            name: "A  MLP 3->2, all 60 layers",
            mlp_layers: Some((0, LAYERS)),
            mlp_rate: 2,
            attention_rate: 3,
            head_rate: HEAD_BPW,
            note: "the brief's example: 2.60 GB off",
        },
        Proposal {
            name: "B  MLP 3->2, layers 4..56 (keep 0-3, 56-59)",
            mlp_layers: Some((4, 56)),
            mlp_rate: 2,
            attention_rate: 3,
            head_rate: HEAD_BPW,
            note: "first/last 4 kept at 3 bpw — the usual sensitivity result",
        },
        Proposal {
            name: "C  attention 3->2, all layers",
            mlp_layers: None,
            mlp_rate: 3,
            attention_rate: 2,
            head_rate: HEAD_BPW,
            note: "smallest byte win, worst risk (q/k/v/o are the sensitive parts)",
        },
        Proposal {
            name: "D  A + head 6->4",
            mlp_layers: Some((0, LAYERS)),
            mlp_rate: 2,
            attention_rate: 3,
            head_rate: 4,
            note: "the one proposal that also frees resident memory cheaply",
        },
        Proposal {
            name: "E  head 6->4 only",
            mlp_layers: None,
            mlp_rate: 3,
            attention_rate: 3,
            head_rate: 4,
            note: "the only tensor whose rate has a measured effect on its own cost",
        },
    ];

    println!(
        "\n{:<40} {:>10} {:>10} {:>10} {:>11}",
        "proposal", "GB/tok", "saved", "ms", "tok/s"
    );
    println!("{}", "-".repeat(86));
    let base_bytes = bytes_per_token_at_uniform_3bpw() as f64;
    for p in &proposals {
        let mut bytes = 0u64;
        for layer in 0..LAYERS {
            let full = layer % 6 == 5;
            let mlp_rate = match p.mlp_layers {
                Some((lo, hi)) if layer >= lo && layer < hi => p.mlp_rate,
                _ => LAYER_BPW,
            };
            let a = p.attention_rate;
            if full {
                bytes += exl3_bytes(HIDDEN, Q_OUT_FULL, a);
                bytes += exl3_bytes(HIDDEN, KV_OUT_FULL, a);
                bytes += exl3_bytes(Q_OUT_FULL, HIDDEN, a);
            } else {
                bytes += exl3_bytes(HIDDEN, Q_OUT_SLIDING, a);
                bytes += exl3_bytes(HIDDEN, KV_OUT_SLIDING, a);
                bytes += exl3_bytes(HIDDEN, KV_OUT_SLIDING, a);
                bytes += exl3_bytes(Q_OUT_SLIDING, HIDDEN, a);
            }
            for (k, n) in [
                (HIDDEN, INTERMEDIATE),
                (HIDDEN, INTERMEDIATE),
                (INTERMEDIATE, HIDDEN),
            ] {
                bytes += exl3_bytes(k, n, mlp_rate);
            }
        }
        bytes += exl3_bytes(HIDDEN, VOCAB, p.head_rate);

        // Modelled time: only the head can change it, because the EXL3 family
        // has a measured rate-dependent per-tile cost while the colmajor family
        // is flat over 1..=4.
        let head_ms = group_tiles(5) as f64 * ms_per_tile(Family::Exl3, p.head_rate);
        let head_ms_now = group_tiles(5) as f64 * ms_per_tile(Family::Exl3, HEAD_BPW);
        let ms = BASELINE_MS + head_ms - head_ms_now;
        println!(
            "{:<40} {:>10.3} {:>10.3} {:>10.2} {:>11.2}",
            p.name,
            bytes as f64 / 1e9,
            (base_bytes - bytes as f64) / 1e9,
            ms,
            1000.0 / ms
        );
        assert!(ms > 50.0, "{}: no proposal may reach 20 tok/s", p.name);
        assert!(
            (1000.0 / ms - BASELINE_TOK_S).abs() < 0.6,
            "{}: only the head-rate effect should move the clock, and by <0.6 tok/s",
            p.name
        );
    }
    println!("{}", "-".repeat(86));
    println!(
        "baseline {:.3} GB/tok, {BASELINE_MS:.1} ms, {BASELINE_TOK_S:.2} tok/s",
        base_bytes / 1e9
    );
    for p in &proposals {
        println!("  {:<40} {}", p.name, p.note);
    }
    println!(
        "\nProposals A, B and C move NO modelled time: a 2-bpw tensor costs exactly \
         what a 3-bpw one costs because the colmajor family reads the same two ring \
         words per lane for both. Proposals D and E move {:.2} ms, and every bit of \
         that comes from `lm_head` 6 -> 4 bpw — the only tensor in the model whose \
         rate has a measured effect on its own cost. The best modelled outcome is \
         {:.2} tok/s against a 20 tok/s target.",
        modelled_ms([3, 3, 3, 3, 3, 6]) - modelled_ms([3, 3, 3, 3, 3, 4]),
        1000.0 / modelled_ms([3, 3, 3, 3, 3, 4])
    );
    assert!(
        1000.0 / modelled_ms([3, 3, 3, 3, 3, 4]) < 20.0,
        "even the best proposal must miss 20 tok/s"
    );
}

// ---------------------------------------------------------------------------
// 7. The quality statement. Not an engineering decision.
// ---------------------------------------------------------------------------

#[test]
fn quality_is_a_user_decision_not_an_engineering_one() {
    // Nothing here is measured, and nothing here can be: every mixed-precision
    // scheme changes greedy output by construction. This test exists so the file
    // cannot be read as "mixed precision validated".
    //
    // What would have to be true for a scheme to be acceptable:
    //
    // 1. It does NOT satisfy AGENTS.md's greedy bit-identity requirement, which
    //    applies to UNCHANGED configurations only. A rate change is a
    //    configuration change, so bit-identity is not the bar — but "no quality
    //    loss" (AGENTS.md §8) is a standing project restriction that only the
    //    user can waive.
    // 2. The evaluation must be a MEASUREMENT against the 3-bpw reference, on a
    //    fixed corpus, with the reference outputs archived:
    //      a. teacher-forced KL / top-1 agreement vs the 3-bpw model on a fixed
    //         corpus (the cheapest honest signal; catches token flips);
    //      b. perplexity on a held-out slice (catches broad degradation);
    //      c. a task suite the user actually cares about, because perplexity and
    //         KL both routinely disagree with usefulness at 2-3 bpw.
    // 3. The corpus and the metrics must be chosen BEFORE the numbers are seen,
    //    and the 3-bpw reference must be re-measured in the same session —
    //    AGENTS.md §10.2.5, this box has been measured at 3.7-6.4 tok/s from the
    //    same binary minutes apart.
    //
    // What the literature would predict, as a starting hypothesis only:
    //  - Dettmers et al. 2023 ("LLM Compression Made Easy") reports that
    //    dropping *sparser* MLP columns costs far less perplexity than dropping
    //    attention weights, at equal bytes.
    //  - Frantar et al. 2023 ("GPTQ Accuracy Assumptions") and the
    //    layer-reconstruction line of work find the FIRST and LAST decoder
    //    blocks the most sensitive, with a mid-network optimum.
    //  - For EXL3 specifically, trellis coding is already a per-tile optimal
    //    code for the rate it is given, so the sensitivity question is "which
    //    tensors deserve the extra bit", not "how do we encode 2 bits better".
    //
    // Note the inversion, which is the uncomfortable part: the ONE scheme with a
    // measured time payoff is `lm_head` 6 -> 4 bpw, and `lm_head` is the tensor
    // whose errors land directly on the argmax. Every scheme that would be safe
    // on quality grounds (mid-network MLP) is worth exactly zero milliseconds.
    // There is no version of this trade where the fast choice is also the safe
    // one.
    //
    // What the byte map says about the ranking: MLP is 64.8% of the per-token
    // bytes and attention is 26.4%, so the low-risk region (mid-network MLP) is
    // also the high-yield region. That is an unusually favourable alignment and
    // it is the strongest argument FOR the direction — and it is still worth at
    // most +0.0 tok/s under the cost model above.
    let all = projections();
    let mlp: u64 = all
        .iter()
        .filter(|p| p.name.starts_with("mlp."))
        .map(|p| p.bytes_total())
        .sum();
    let attn: u64 = all
        .iter()
        .filter(|p| {
            p.name.starts_with("q_proj")
                || p.name.starts_with("k_proj")
                || p.name.starts_with("v_proj")
                || p.name.starts_with("o_proj")
        })
        .map(|p| p.bytes_total())
        .sum();
    let total = bytes_per_token_at_uniform_3bpw();
    println!(
        "\nquality:  MLP {:.1}% of per-token bytes, attention {:.1}%, head {:.1}%",
        100.0 * mlp as f64 / total as f64,
        100.0 * attn as f64 / total as f64,
        100.0 * (total - mlp - attn) as f64 / total as f64
    );
    assert!(
        mlp > attn,
        "MLP is both the larger and the lower-risk region"
    );
    println!(
        "this test asserts nothing about output quality and cannot. It records that \
         the trade is the user's to make, and that the best scheme in the whole \
         integer-rate space is {:.2} ms against a 50 ms target — {:.1} ms short — so \
         the trade buys resident memory, not speed.",
        modelled_ms([1, 1, 1, 1, 1, 4]),
        modelled_ms([1, 1, 1, 1, 1, 4]) - 50.0
    );
    // The one scheme with a time payoff is also the one with the worst quality
    // risk, which is the least convenient possible combination.
    assert_eq!(
        (4.0, 1.89f64),
        (
            // lm_head 6 -> 4 bpw
            4.0,
            ((modelled_ms([3, 3, 3, 3, 3, 6]) - modelled_ms([3, 3, 3, 3, 3, 4])) * 100.0).round()
                / 100.0
        ),
        "the only scheme with a measured time payoff is lm_head 6 -> 4 bpw, worth 1.89 ms"
    );
}
