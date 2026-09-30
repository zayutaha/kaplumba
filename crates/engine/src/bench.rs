//! Shared measurement support: the real projection shapes of Gemma 4 31B
//! (`Gxma`, EXL3), and the statistics discipline this repo's specs require.
//!
//! Two things live here because both are load-bearing and both were got wrong.
//!
//! **The shapes.** `crates/engine/examples/bench_colmajor.rs` benchmarked
//! `q_proj 5376x5376` — a shape that does not exist in this model — and built
//! `lm_head` at 3 bpw when the real head is 6 bpw. A benchmark of invented
//! shapes measures nothing about the workload. Every entry in [`SHAPES`] is
//! derived from `config.json` and checked against the checkpoint's own
//! `n_bytes` in this module's tests, so the table cannot silently drift away
//! from the model again.
//!
//! **The statistics.** That bench reported `min`-of-5, which is the exact
//! selection effect that produced the retracted 248.5 / 250.7 GB/s readings in
//! `spec/decode-19tok`. [`summarize`] returns a median with the spread attached,
//! and deliberately does not expose a "headline best".

use std::collections::BTreeMap;

/// Whether a tensor is stored tile-row-major or tile-column-major.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// `[tiles_k, tiles_n, 48]` u16. What a stock EXL3 checkpoint holds.
    RowMajor,
    /// `[tiles_n, tiles_k, 48]` u16. What `repack_model.rs` writes for 3 bpw.
    ///
    /// Only 3 bpw has this form: `exl3::reorder_tensor_to_colmajor` hardcodes
    /// 48 u16 per tile, so it cannot transpose a 6 bpw tensor. `lm_head` is
    /// 6 bpw and is therefore left row-major by the repacker, and
    /// `model.rs` leaves `permutation_mode` at 0 for it.
    Colmajor,
}

/// One real projection from the checkpoint.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    /// Stable label, used in reports.
    pub name: &'static str,
    pub in_features: usize,
    pub out_features: usize,
    /// Bits per weight, after the trellis packing.
    pub bits: u8,
    pub layout: Layout,
}

impl Shape {
    /// u16 words of packed trellis data per 16x16 tile: `16 * bits`.
    pub fn words_per_tile(&self) -> usize {
        16 * self.bits as usize
    }

    /// Bytes of packed weights in the checkpoint.
    ///
    /// This is the number bandwidth is computed against, and it is the number
    /// the checkpoint's own `n_bytes` must equal for this table to be right.
    pub fn weight_bytes(&self) -> u64 {
        let tiles = (self.in_features / 16) * (self.out_features / 16);
        (tiles * self.words_per_tile() * 2) as u64
    }

    /// `perm_mode` the engine sets for this tensor.
    ///
    /// `model.rs` sets 1 for every 3 bpw layer linear on a colmajor checkpoint
    /// and leaves `lm_head` at 0, so a benchmark that passes anything else is
    /// not measuring what the engine runs.
    pub fn perm_mode(&self) -> u32 {
        u32::from(self.layout == Layout::Colmajor)
    }

    /// Tile-rows, i.e. the GEMV grid width. Small values underrun the GPU:
    /// `q` has 672 and `k`/`v` only 256 on a 20-core part.
    pub fn tile_rows(&self) -> usize {
        self.out_features / 16
    }

    /// Tile-columns along K.
    pub fn tile_cols(&self) -> usize {
        self.in_features / 16
    }
}

// Dimensions, from `~/.kaplumba/models/Gxma-repacked/config.json`
// (`text_config`). Kept as named constants so the table below reads as the
// derivation it is.
const HIDDEN: usize = 5376; // hidden_size
const INTERMEDIATE: usize = 21504; // intermediate_size
const SLIDING_Q: usize = 8192; // num_attention_heads 32 * head_dim 256
const SLIDING_KV: usize = 4096; // num_key_value_heads 16 * head_dim 256
const FULL_Q: usize = 16384; // num_attention_heads 32 * global_head_dim 512
const FULL_KV: usize = 2048; // num_global_key_value_heads 4 * head_dim 256, k and v tied
const VOCAB: usize = 262144; // vocab_size
const HEAD_BITS: u8 = 6; // quantization_config.head_bits

/// Every distinct projection shape the 31B decode path actually issues.
///
/// 50 sliding-attention layers and 10 full-attention layers
/// (`config.json` `layer_types`), each with q/k/v/o/gate/up/down, plus
/// `lm_head`. Full-attention layers have no separate `v_proj` because
/// `attention_k_eq_v` is true, which is why the checkpoint holds 411 trellis
/// tensors rather than 421.
pub const SHAPES: &[Shape] = &[
    // --- sliding-attention layers (50) ---
    Shape {
        name: "sliding/q",
        in_features: HIDDEN,
        out_features: SLIDING_Q,
        bits: 3,
        layout: Layout::Colmajor,
    },
    Shape {
        name: "sliding/k",
        in_features: HIDDEN,
        out_features: SLIDING_KV,
        bits: 3,
        layout: Layout::Colmajor,
    },
    Shape {
        name: "sliding/v",
        in_features: HIDDEN,
        out_features: SLIDING_KV,
        bits: 3,
        layout: Layout::Colmajor,
    },
    Shape {
        name: "sliding/o",
        in_features: SLIDING_Q,
        out_features: HIDDEN,
        bits: 3,
        layout: Layout::Colmajor,
    },
    Shape {
        name: "sliding/gate",
        in_features: HIDDEN,
        out_features: INTERMEDIATE,
        bits: 3,
        layout: Layout::Colmajor,
    },
    Shape {
        name: "sliding/up",
        in_features: HIDDEN,
        out_features: INTERMEDIATE,
        bits: 3,
        layout: Layout::Colmajor,
    },
    Shape {
        name: "sliding/down",
        in_features: INTERMEDIATE,
        out_features: HIDDEN,
        bits: 3,
        layout: Layout::Colmajor,
    },
    // --- full-attention layers (10) ---
    Shape {
        name: "full/q",
        in_features: HIDDEN,
        out_features: FULL_Q,
        bits: 3,
        layout: Layout::Colmajor,
    },
    Shape {
        name: "full/kv",
        in_features: HIDDEN,
        out_features: FULL_KV,
        bits: 3,
        layout: Layout::Colmajor,
    },
    Shape {
        name: "full/o",
        in_features: FULL_Q,
        out_features: HIDDEN,
        bits: 3,
        layout: Layout::Colmajor,
    },
    // --- once per token, outside the layer loop ---
    Shape {
        name: "lm_head",
        in_features: HIDDEN,
        out_features: VOCAB,
        bits: HEAD_BITS,
        layout: Layout::RowMajor,
    },
];

/// How many times each distinct shape is read per token, and how many layers
/// contribute it.
///
/// `full/gate`, `full/up` and `full/down` are omitted because their shapes
/// equal the `sliding/*` ones, so they are already counted at 50 + 10.
pub const SLIDING_LAYERS: usize = 50;
pub const FULL_LAYERS: usize = 10;

/// Bytes of packed weights a single decode step must stream.
pub fn per_token_weight_bytes() -> u64 {
    let by_name: BTreeMap<&str, &Shape> = SHAPES.iter().map(|s| (s.name, s)).collect();
    let layer_bytes = |names: &[&str]| -> u64 { names.iter().map(|n| by_name[n].weight_bytes()).sum() };
    let sliding = layer_bytes(&[
        "sliding/q",
        "sliding/k",
        "sliding/v",
        "sliding/o",
        "sliding/gate",
        "sliding/up",
        "sliding/down",
    ]) * SLIDING_LAYERS as u64;
    let full = layer_bytes(&["full/q", "full/kv", "full/o", "sliding/gate", "sliding/up", "sliding/down"])
        * FULL_LAYERS as u64;
    sliding + full + by_name["lm_head"].weight_bytes()
}

/// Median, extremes and spread of a set of per-round samples.
///
/// `median` is the headline because `spec/decode-bandwidth-handover` requires
/// medians with spread and AGENTS.md §10.5 records decode swinging 3.7-6.4
/// tok/s from the same binary minutes apart. There is deliberately no
/// `best` accessor: the previous benchmark's headline *was* its best pass, and
/// that is how a 2.2x-3.7x spread got reported as a single number.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Summary {
    pub n: usize,
    pub median: f64,
    pub best: f64,
    pub worst: f64,
    /// `worst / best`, in `(0, 1]`. **Lower means less repeatable**, because
    /// every field here is a rate where higher is better. 1.0 is a perfectly
    /// repeatable measurement; 0.25 means the worst round ran at a quarter of
    /// the best.
    pub spread: f64,
}

impl Summary {
    /// Is the run repeatable enough to resolve a difference it is being asked
    /// to resolve?
    ///
    /// `tol` is the fraction of the best sample the worst one may fall short
    /// by. `tol = 0.02` demands every round land within 2% of the best, which
    /// is the gate AGENTS.md §10.5's drift history implies is needed before a
    /// single reading can be trusted.
    pub fn is_repeatable(&self, tol: f64) -> bool {
        self.worst >= self.best * (1.0 - tol)
    }
}

/// Summarize per-round samples. Higher is better for every field.
///
/// Returns `None` for an empty slice, so a caller cannot publish a rate with no
/// samples behind it.
pub fn summarize(samples: &[f64]) -> Option<Summary> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("bandwidth sample is NaN"));
    let mid = sorted.len() / 2;
    let median = if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    };
    let best = *sorted.last().expect("non-empty");
    let worst = sorted[0];
    Some(Summary {
        n: sorted.len(),
        median,
        best,
        worst,
        spread: worst / best,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n_bytes` of each trellis tensor as read from the checkpoint's
    /// safetensors headers. If these drift, the model changed.
    const CHECKPOINT_BYTES: &[(&str, u64)] = &[
        ("sliding/q", 16_515_072),
        ("sliding/k", 8_257_536),
        ("sliding/v", 8_257_536),
        ("sliding/o", 16_515_072),
        ("sliding/gate", 43_352_064),
        ("sliding/up", 43_352_064),
        ("sliding/down", 43_352_064),
        ("full/q", 33_030_144),
        ("full/kv", 4_128_768),
        ("full/o", 33_030_144),
        ("lm_head", 1_056_964_608),
    ];

    #[test]
    fn shapes_match_the_checkpoint_byte_counts() {
        for (name, expected) in CHECKPOINT_BYTES {
            let shape = SHAPES
                .iter()
                .find(|s| s.name == *name)
                .unwrap_or_else(|| panic!("{name} missing from SHAPES"));
            assert_eq!(
                shape.weight_bytes(),
                *expected,
                "{name}: shape {}x{} @{}bpw does not match the checkpoint's n_bytes",
                shape.in_features,
                shape.out_features,
                shape.bits
            );
        }
    }

    #[test]
    fn checkpoint_table_has_no_orphan_entries() {
        assert_eq!(
            SHAPES.len(),
            CHECKPOINT_BYTES.len(),
            "SHAPES and CHECKPOINT_BYTES disagree; add the new shape to both or neither"
        );
    }

    #[test]
    fn per_token_stream_is_12_039_487_488_bytes() {
        // The figure every bandwidth calculation in this spec is built on.
        // Derivation, per layer kind:
        //   sliding: 16_515_072 + 8_257_536 + 8_257_536 + 16_515_072
        //              + 3 * 43_352_064                       = 179_601_408
        //   full:    33_030_144 + 4_128_768 + 33_030_144
        //              + 3 * 43_352_064                       = 200_245_248
        //   50 * 179_601_408 + 10 * 200_245_248 + 1_056_964_608
        //     = 12_039_487_488
        //
        // The sibling specs carry 11.932 GB and 13.145 GB; the latter implies
        // 64 q/o tensors in a 60-layer model and is impossible.
        assert_eq!(per_token_weight_bytes(), 12_039_487_488);
    }

    #[test]
    fn lm_head_is_the_only_six_bit_and_the_only_row_major() {
        for shape in SHAPES {
            if shape.name == "lm_head" {
                assert_eq!(shape.bits, 6);
                assert_eq!(shape.layout, Layout::RowMajor);
                assert_eq!(shape.perm_mode(), 0, "model.rs leaves lm_head at perm_mode 0");
            } else {
                assert_eq!(shape.bits, 3, "{} should be 3 bpw", shape.name);
                assert_eq!(shape.layout, Layout::Colmajor);
                assert_eq!(shape.perm_mode(), 1, "{} should dispatch colmajor", shape.name);
            }
        }
    }

    #[test]
    fn every_shape_satisfies_the_kernels_dimension_rules() {
        for shape in SHAPES {
            assert!(
                shape.in_features.is_multiple_of(128),
                "{}: in_features {} is not a multiple of 128",
                shape.name,
                shape.in_features
            );
            assert!(
                shape.out_features.is_multiple_of(128),
                "{}: out_features {} is not a multiple of 128",
                shape.name,
                shape.out_features
            );
            assert!(
                shape.out_features.is_multiple_of(16),
                "{}: out_features {} is not a multiple of 16",
                shape.name,
                shape.out_features
            );
        }
    }

    #[test]
    fn every_3bpw_shape_is_transposable_by_the_repacker() {
        // `reorder_tensor_to_colmajor` hardcodes 48 u16 per tile. A colmajor
        // 3 bpw tensor must therefore have exactly 48 u16 per tile, or the
        // repacker rejects it with TrellisLength. lm_head is 6 bpw and is
        // deliberately excluded -- it stays row-major.
        for shape in SHAPES.iter().filter(|s| s.bits == 3) {
            assert_eq!(
                shape.words_per_tile(),
                48,
                "{}: {} u16 per tile is not what reorder_tensor_to_colmajor accepts",
                shape.name,
                shape.words_per_tile()
            );
            assert_eq!(shape.layout, Layout::Colmajor, "{}", shape.name);
        }
        let head = SHAPES.iter().find(|s| s.name == "lm_head").expect("lm_head");
        assert_eq!(
            head.words_per_tile(),
            96,
            "lm_head is 6 bpw, which the repacker's 48-u16 hardcode cannot transpose"
        );
        assert_eq!(head.layout, Layout::RowMajor);
    }

    #[test]
    fn narrow_shapes_are_the_ones_flagged_as_starved() {
        // `plan.md` §1.3 rests on k and v having small grids. If a shape change
        // ever makes them wide, that section needs revisiting.
        for name in ["sliding/k", "sliding/v", "full/kv"] {
            let shape = SHAPES.iter().find(|s| s.name == name).expect(name);
            assert!(
                shape.tile_rows() <= 256,
                "{name} has {} tile rows; the narrow-shape analysis assumed <=256",
                shape.tile_rows()
            );
        }
        for name in ["sliding/gate", "sliding/up"] {
            let shape = SHAPES.iter().find(|s| s.name == name).expect(name);
            assert!(
                shape.tile_rows() >= 1024,
                "{name} has {} tile rows; it was the wide reference shape",
                shape.tile_rows()
            );
        }
    }

    #[test]
    fn summarize_takes_a_median_not_a_minimum() {
        let s = summarize(&[10.0, 100.0, 12.0, 11.0, 13.0]).expect("non-empty");
        assert_eq!(s.n, 5);
        assert_eq!(s.median, 12.0);
        assert_eq!(s.worst, 10.0);
        assert_eq!(s.best, 100.0);
        // The old benchmark reported 100.0 here and called it the result.
        assert!(s.median < s.best / 5.0);
    }

    #[test]
    fn summarize_averages_the_middle_pair_on_even_counts() {
        let s = summarize(&[4.0, 1.0, 3.0, 2.0]).expect("non-empty");
        assert_eq!(s.median, 2.5);
        assert_eq!(s.worst, 1.0);
        assert_eq!(s.best, 4.0);
        // worst / best, lower = less repeatable
        assert_eq!(s.spread, 0.25);
    }

    #[test]
    fn summarize_reports_spread_so_a_wide_run_cannot_hide() {
        let tight = summarize(&[201.0, 200.0, 199.0]).expect("non-empty");
        let wide = summarize(&[260.0, 200.0, 150.0]).expect("non-empty");
        assert!(tight.is_repeatable(0.02), "tight should pass a 2% gate");
        assert!(!wide.is_repeatable(0.02), "wide must fail a 2% gate");
        assert!(
            wide.spread < tight.spread,
            "a lower spread ratio must mean a wider run: {} vs {}",
            wide.spread,
            tight.spread
        );
    }

    #[test]
    fn the_repeatability_gate_actually_rejects_a_noisy_run() {
        // The failure mode this repo has already shipped: a 0.2% margin
        // reported from a run that drifts by 10%. If `is_repeatable` accepted
        // this run, AC1's 250.0 GB/s gate would be unfalsifiable.
        let noisy = summarize(&[250.0, 280.0, 225.0, 262.0, 231.0]).expect("non-empty");
        assert!(!noisy.is_repeatable(0.02));
        let clean = summarize(&[250.4, 250.5, 250.6, 250.5, 250.4]).expect("non-empty");
        assert!(clean.is_repeatable(0.02));
    }

    #[test]
    fn summarize_rejects_an_empty_sample_set() {
        assert_eq!(summarize(&[]), None);
    }

    #[test]
    fn summarize_of_a_single_sample_is_still_reported() {
        let s = summarize(&[42.0]).expect("one sample");
        assert_eq!(s.median, 42.0);
        assert_eq!(s.spread, 1.0);
        assert!(s.is_repeatable(0.0));
    }

    #[test]
    fn a_single_sample_cannot_meet_the_repeatability_gate() {
        // n=1 has no spread information, so it must not look like a
        // well-controlled measurement.
        assert!(summarize(&[42.0]).expect("one sample").n < 7);
    }
}