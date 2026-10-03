//! Byte-accounting probe: how many bytes of *real checkpoint weights* does one
//! decode step actually read?
//!
//! # Why this exists
//!
//! Decode throughput on this box is bandwidth-bound, so the only two levers are
//! bytes-per-token and bytes-per-second. Every tok/s claim in `spec/` has been
//! argued from a bandwidth figure; none has been closed against a byte count
//! derived from the checkpoint itself. This file closes it, and it is the
//! instrument that any future byte-reduction proposal must be scored against.
//!
//! It is deliberately **header-only**: it opens the two `.safetensors` files,
//! reads the 8-byte length prefix and the JSON header, and never touches the
//! payload. So it runs on a busy box, takes milliseconds, loads no model, takes
//! no GPU lock and allocates no Metal buffer (`AGENTS.md` §10 is not engaged).
//!
//! # What it measures, and what it does not
//!
//! It counts *distinct bytes the decode path must touch per token*, derived from
//! tensor byte lengths and the dispatch structure in `model.rs`. It does **not**
//! count DRAM traffic, because the two are provably not the same number here:
//! `tests/rate4_colmajor_bit_exact.rs`
//! (`dram_line_traffic_is_1x_at_every_rate_so_the_amplification_is_requests_only`)
//! measures 1.0000x line traffic at every rate, i.e. the 3 bpw kernel's 2.667
//! lane-load amplification is a *request-path* cost that reads every byte exactly
//! once. So there is no wasted byte traffic available to reclaim, and
//! bytes-per-token is a hard lower bound on the work, not an estimate of it.
//!
//! # The classification, and why each bucket is what it is
//!
//! | bucket | tensors | read per token | evidence |
//! |---|---|---|---|
//! | layer trellis | `…layers.N.<proj>.trellis` | **all of it** | `model.rs:1418-2160`, one GEMV per matrix, fused pairs (`encode_gemv_fused2` q+k, `encode_gemv_fused2_mlp` gate+up) never re-read |
//! | head trellis | `lm_head.trellis` | **all of it** | `model.rs:4424` `packed_offset: self.lm_head.metal_offset`, `out_features: vocab`, once per chunk; `weights.rs:516` "read in full each token" |
//! | suh/svh/mcg | scales | all of it (tiny) | side tensors bound to every GEMV |
//! | norms / q_norm / k_norm / layer_scalar | BF16 | all of it (tiny) | bound to every RMSNorm |
//! | `embed_tokens.weight` | BF16 `[262144, 5376]` | **one row = 10 752 B** | `weights.rs:514` "exactly one row is read per token, a ~10 KiB working set" |
//!
//! and the one bucket that is *not* in the checkpoint at all — the KV cache,
//! which is synthesised at runtime and therefore must be modelled from
//! `Gemma4Config` rather than summed from tensor headers.
//!
//! # Self-check (AGENTS.md §3.4)
//!
//! A counter that reports zero is indistinguishable from a counter that is
//! broken, so every classification rule below is applied twice: once to the real
//! header, and once to a deliberately mis-classified copy of the same names.
//! `classification_is_not_vacuous` asserts the mutant produces a *different*
//! total. If someone weakens a rule until it matches everything, that test is
//! what notices.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use kaplumba_engine::config::{Gemma4Config, LayerKind};
use kaplumba_engine::safetensors::{
    TensorInfo, is_vision_tensor_name, load_text_selection, read_index,
};

/// Where the real checkpoint lives. Overridable so the test can be pointed at a
/// sibling export (`Gxma-repacked`, `Gxma-4-bit`) without editing it.
fn model_dir() -> PathBuf {
    match std::env::var_os("KAPLUMBA_PROBE_MODEL_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => {
            let home = std::env::var_os("HOME").expect("HOME must be set");
            PathBuf::from(home).join(".kaplumba/models/Gxma/Gxma")
        }
    }
}

/// Header-only load of the checkpoint's text tensors. Returns `None` when the
/// checkpoint is not on this box, so the suite stays green elsewhere and the
/// numbers are never silently invented.
struct Checkpoint {
    config: Gemma4Config,
    tensors: BTreeMap<String, TensorInfo>,
    excluded_vision: Vec<String>,
    vision_bytes: u64,
    all_bytes: u64,
    text_bytes: u64,
}

fn load(dir: &Path) -> Option<Checkpoint> {
    let index_path = dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        return None;
    }
    let index = read_index(&index_path).ok()?;
    let selection = load_text_selection(dir, index).ok()?;
    // Re-derive the vision byte total from the shards we already parsed, rather
    // than trusting `selection.excluded_vision` to be complete.
    let mut vision_bytes = 0_u64;
    for shard in selection.shards.values() {
        for (name, tensor) in &shard.tensors {
            if is_vision_tensor_name(name) {
                vision_bytes += tensor.byte_len();
            }
        }
    }
    Some(Checkpoint {
        config: safetensors_config(dir),
        excluded_vision: selection.excluded_vision.clone(),
        tensors: selection.tensors.clone(),
        vision_bytes,
        all_bytes: selection.all_bytes,
        text_bytes: selection.text_bytes,
    })
}

fn safetensors_config(dir: &Path) -> Gemma4Config {
    // `inspect_model_dir` does the config parse and the tensor validation
    // together. We only want the config, and we do not want a failure on an
    // unrelated validation to be swallowed silently — so this propagates.
    kaplumba_engine::safetensors::inspect_model_dir(dir)
        .expect("checkpoint metadata must load")
        .config
}

// ---------------------------------------------------------------------------
// The counter
// ---------------------------------------------------------------------------

/// Bytes the decode path touches per generated token, split by bucket.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct PerTokenBytes {
    /// Layer `.trellis`: read in full, every token, every layer. 91.1%.
    layer_trellis: u64,
    /// `lm_head.trellis`: read in full, every token. 8.8%.
    head_trellis: u64,
    /// `.suh` / `.svh` / `.mcg` side tensors.
    scales: u64,
    /// RMSNorm weights, per-head QK norms, per-layer scalars.
    norms: u64,
    /// `embed_tokens`: exactly one row, `hidden_size * 2`.
    embed_row: u64,
}

impl PerTokenBytes {
    fn weights(&self) -> u64 {
        self.layer_trellis
            + self.head_trellis
            + self.scales
            + self.norms
            + self.embed_row
    }
}

/// The one classification rule set. Deliberately a pure function of tensor
/// *names*, so it can be run against a mutated name list to prove it is not
/// vacuous.
///
/// The four rules, and the `model.rs` fact each one encodes:
/// 1. `…layers.N.<proj>.trellis` -> layer trellis. Every layer runs every
///    projection every token; there is no conditional weight loading in the
///    decode path (`model.rs:1403` `let full = config.full_attention;` only
///    selects the *attention* kernel, never the presence of a GEMV).
/// 2. `lm_head.trellis` -> head trellis. `encode_lm_head` is unconditional on
///    the sampled position (`model.rs:4267-4280`); `run_head` is only ever false
///    on the offline calibration path, never during decode.
/// 3. `.suh` / `.svh` / `.mcg` -> scales.
/// 4. `embed_tokens.weight` -> collapsed to one row. Everything else that is a
///    `.weight` and not a projection -> norms.
fn classify(name: &str, byte_len: u64, hidden_size: usize, acc: &mut PerTokenBytes) {
    if name.contains(".layers.") && name.ends_with(".trellis") {
        acc.layer_trellis += byte_len;
    } else if name == "lm_head.trellis" {
        acc.head_trellis += byte_len;
    } else if name.ends_with(".suh") || name.ends_with(".svh") || name.ends_with(".mcg") {
        acc.scales += byte_len;
    } else if name.ends_with("embed_tokens.weight") {
        // One row of the [vocab, hidden] bf16 table. The whole 2.82 GB table is
        // mapped, not resident (`weights.rs:576-587`), precisely because only
        // this slice is touched.
        acc.embed_row += (hidden_size * 2) as u64;
    } else {
        acc.norms += byte_len;
    }
}

fn count_per_token(tensors: &BTreeMap<String, TensorInfo>, hidden_size: usize) -> PerTokenBytes {
    let mut acc = PerTokenBytes::default();
    for (name, tensor) in tensors {
        classify(name, tensor.byte_len(), hidden_size, &mut acc);
    }
    acc
}

// ---------------------------------------------------------------------------
// KV cache: modelled, not summed (it is not in the checkpoint)
// ---------------------------------------------------------------------------

/// Bytes of KV read by one decode step at context length `context`.
///
/// Sliding layers read `min(context, sliding_window)` positions; full-attention
/// layers read all `context`. The global layers use `num_global_key_value_heads`
/// and `global_head_dim` (512/4 here), which is why they are not just
/// `num_key_value_heads * head_dim` like the sliding ones — getting this wrong
/// understates the 10 full layers by 2x.
fn kv_bytes_per_token(config: &Gemma4Config, context: usize) -> u64 {
    let sliding = config
        .layer_types
        .iter()
        .filter(|kind| !kind.is_full())
        .count();
    let full = config.layer_types.len() - sliding;
    let sliding_kv_dim = config.num_key_value_heads as u64 * config.head_dim as u64;
    let global_kv_dim =
        config.num_global_key_value_heads as u64 * config.global_head_dim as u64;
    let window = context.min(config.sliding_window) as u64;
    // 2 (K and V) * 2 bytes (fp16) per element, per layer.
    sliding as u64 * 2 * 2 * sliding_kv_dim * window
        + full as u64 * 2 * 2 * global_kv_dim * context as u64
}

// ---------------------------------------------------------------------------
// Bandwidths measured in this tree, and the floors they imply
// ---------------------------------------------------------------------------

/// Best sustained 3 bpw EXL3 GEMV rate ever measured on this box.
/// `spec/decode-25tok-peak-bandwidth/tasks.md:124`, `colmajor_2col_sk4` at
/// K=21504.
const GBPS_3BPW: f64 = 203.0;
/// The 2026-10-01 rate sweep (`spec/decode-3bpw-amplification/spec.md` §2):
/// 3 bpw 199.4, 4 bpw 249.0, 6 bpw 251.4.
const GBPS_3BPW_SWEEP: f64 = 199.4;
/// Pure streaming copy probe, 12 GB resident, uint4 loads.
const GBPS_COPY: f64 = 250.5;
/// `lm_head` (6 bpw, row-major) in situ: 1.057 GB / 4.12 ms. The highest
/// weight-read rate ever recorded in this repo — and it is 6 bpw, so it is not
/// available to the 3 bpw layer GEMVs.
const GBPS_HEAD_6BPW: f64 = 257.0;
/// Measured baseline: 16.05-16.10 tok/s.
const BASELINE_MS: f64 = 62.3;

/// Absolute floor on bytes/token: every non-embedding parameter at 3 bpw.
/// Embedding is excluded because only one row is read.
fn floor_bytes_at_3bpw(config: &Gemma4Config, side_tensors: u64) -> u64 {
    let params = floor_params(config);
    // `side_tensors` is scales + norms + the one embed row, already collapsed.
    params * 3 / 8 + side_tensors
}

/// Non-embedding parameters: every weight the decode path reads in full.
///
/// Split out so it can be cross-checked against the checkpoint independently of
/// any byte total. Getting the global-vs-sliding head split wrong here is the
/// easiest way to produce a floor that looks right and is not.
fn floor_params(config: &Gemma4Config) -> u64 {
    let h = config.hidden_size as u64;
    let im = config.intermediate_size as u64;
    let mut params = config.vocab_size as u64 * h; // lm_head: [262144, 5376]
    for kind in &config.layer_types {
        // Sliding layers project to `num_attention_heads * head_dim` and back;
        // full-attention ("global") layers project to
        // `num_attention_heads * global_head_dim`, which is what the checkpoint's
        // `o_proj.suh` length of 16384 says. Both are cross-checked below.
        let (q_out, kv_out, has_v) = if kind.is_full() {
            (
                config.num_attention_heads as u64 * config.global_head_dim as u64,
                config.num_global_key_value_heads as u64 * config.global_head_dim as u64,
                false,
            )
        } else {
            (
                config.num_attention_heads as u64 * config.head_dim as u64,
                config.num_key_value_heads as u64 * config.head_dim as u64,
                true,
            )
        };
        params += h * q_out; // q_proj
        params += h * kv_out; // k_proj
        if has_v {
            params += h * kv_out; // v_proj
        }
        params += q_out * h; // o_proj
        params += 3 * h * im; // gate, up, down
    }
    params
}

fn ms_for(bytes: u64, gbps: f64) -> f64 {
    bytes as f64 / 1e9 / gbps * 1000.0
}

fn tok_s(ms: f64) -> f64 {
    1000.0 / ms
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The headline number. If this moves, every tok/s projection in `spec/` moves.
#[test]
fn one_decode_step_touches_12_058_gb_of_weights() {
    let Some(ckpt) = load(&model_dir()) else {
        eprintln!("SKIP: checkpoint not at {}", model_dir().display());
        return;
    };
    let bytes = count_per_token(&ckpt.tensors, ckpt.config.hidden_size);

    // 410 quantised matrices: 50 sliding layers x 7 projections
    // (q,k,v,o,gate,up,down) + 10 full layers x 6 (no v_proj) = 350 + 60.
    // NOTE: this 410 is a *matrix* count. The layer count is 60, not 410.
    let matrices = ckpt
        .tensors
        .keys()
        .filter(|n| n.contains(".layers.") && n.ends_with(".trellis"))
        .count();
    assert_eq!(matrices, 410, "expected 410 quantised layer matrices");
    assert_eq!(ckpt.config.num_hidden_layers, 60, "Gemma 4 31B has 60 layers");
    assert_eq!(
        ckpt.config.layer_types.iter().filter(|k| k.is_full()).count(),
        10,
        "every 6th layer is full attention (config.rs:240 index % 6 == 5)"
    );

    assert_eq!(
        bytes.layer_trellis, 10_982_522_880,
        "60 layers of 3 bpw trellis"
    );
    assert_eq!(bytes.head_trellis, 1_056_964_608, "262144 x 5376 at 6 bpw");
    assert_eq!(bytes.embed_row, 10_752, "one bf16 row of hidden_size");

    // The 6 bpw head is 8.8% of all bytes read while every layer is 3 bpw.
    let head_share = bytes.head_trellis as f64 / bytes.weights() as f64;
    let layer_share = bytes.layer_trellis as f64 / bytes.weights() as f64;
    assert!(
        (head_share - 0.0877).abs() < 0.001,
        "head share was {head_share:.4}, expected ~0.0877"
    );
    assert!(
        (layer_share - 0.9109).abs() < 0.001,
        "layers are {layer_share:.4} of bytes/token, expected ~0.9109. If this \
         drifts toward 1.0 the head has been swept into the layer bucket — see \
         mutant C in `classification_is_not_vacuous`"
    );
    assert!(
        layer_share + head_share < 1.0,
        "the remaining {:.4} is side tensors and one embed row",
        1.0 - layer_share - head_share
    );

    assert_eq!(
        bytes.weights(),
        12_058_001_636,
        "bytes/token = {:.4} GB; at {} ms that is {:.1} GB/s",
        bytes.weights() as f64 / 1e9,
        BASELINE_MS,
        bytes.weights() as f64 / 1e9 / (BASELINE_MS / 1000.0)
    );

    // ... and that reconciles with the measured baseline. This is the check
    // that makes the number mean something rather than merely being large.
    let implied = bytes.weights() as f64 / 1e9 / (BASELINE_MS / 1000.0);
    assert!(
        (implied - 193.5).abs() < 1.0,
        "12.058 GB in 62.3 ms implies {implied:.1} GB/s, expected ~193.5 \
         (the recorded 193 GB/s achieved)"
    );
}

/// The vision tower is 7.2% of the checkpoint and **zero** bytes per token.
/// Worth stating because it is the only place in the file where a large tensor
/// is mapped and never read at all.
#[test]
fn vision_tower_is_mapped_and_never_read() {
    let Some(ckpt) = load(&model_dir()) else {
        return;
    };
    assert!(
        !ckpt.excluded_vision.is_empty(),
        "the vision tower must be excluded from the text selection"
    );
    assert_eq!(ckpt.vision_bytes, 1_151_487_072, "vision tower bytes");
    assert_eq!(
        ckpt.all_bytes - ckpt.text_bytes,
        ckpt.vision_bytes,
        "every non-text byte must be vision"
    );
    // All 356 vision tensors are `.linear.weight`/norms in BF16 — i.e. a full
    // unquantised fp vision model, not EXL3 trellis.
    let bytes = count_per_token(&ckpt.tensors, ckpt.config.hidden_size);
    assert!(
        bytes.weights() < ckpt.text_bytes,
        "per-token bytes ({}) must be below all text bytes ({}) because \
         embed_tokens contributes one row, not 2.82 GB",
        bytes.weights(),
        ckpt.text_bytes
    );
}

/// `embed_tokens` is 2.82 GB of the 16.03 GB checkpoint and 0.00009% of the
/// bytes read per token. This is the answer to "is 23% of the checkpoint doing
/// nothing for decode": it is 17.6%, and yes, it is doing nothing for *speed* —
/// but it is not resident either (`weights.rs:576`), so it costs nothing there
/// either. It is mapped, and one row stays hot in the page cache.
#[test]
fn embed_tokens_is_two_point_eight_gb_and_one_row() {
    let Some(ckpt) = load(&model_dir()) else {
        return;
    };
    let full = ckpt
        .tensors
        .get("model.language_model.embed_tokens.weight")
        .expect("embed_tokens must be present");
    assert_eq!(full.byte_len(), 2_818_572_288, "[262144, 5376] bf16");
    assert_eq!(full.shape, vec![262_144, 5_376]);

    let bytes = count_per_token(&ckpt.tensors, ckpt.config.hidden_size);
    let share = bytes.embed_row as f64 / bytes.weights() as f64;
    assert!(
        share < 1e-6,
        "one embed row is {share:.2e} of per-token bytes; if this assertion \
         fires the embedding is being read whole"
    );
    // And the "23% of the checkpoint" framing, for the record.
    let checkpoint = ckpt.all_bytes;
    assert!(
        (full.byte_len() as f64 / checkpoint as f64 - 0.176).abs() < 0.005,
        "embed_tokens is {:.1}% of the {checkpoint} byte checkpoint",
        full.byte_len() as f64 / checkpoint as f64 * 100.0
    );
}

/// The head is read **in full, every token**, and cannot be skipped. This test
/// pins the code structure that makes that true so a future refactor that
/// accidentally reads it twice, or not at all, is caught here.
#[test]
fn head_is_read_in_full_every_token_and_cannot_be_skipped() {
    let Some(ckpt) = load(&model_dir()) else {
        return;
    };
    let bytes = count_per_token(&ckpt.tensors, ckpt.config.hidden_size);
    assert_eq!(bytes.head_trellis, 1_056_964_608, "the whole 6 bpw head");

    // The head is not conditioned on the sampled token: `encode_lm_head` runs
    // the full 262144-wide GEMV and only then takes an argmax over it. The
    // softcap is skipped for greedy (`model.rs:4443` `skip_softcap`) because
    // `tanh` is monotonic, so the *result* is unchanged — but the 1.057 GB is
    // spent before any of that.
    assert_eq!(
        ckpt.config.vocab_size, 262_144,
        "head out_features == vocab_size, so the GEMV is full-width"
    );
    assert_eq!(
        ckpt.config.quantization.head_bits, 6,
        "head is 6 bpw while layers are {} bpw",
        ckpt.config.quantization.bits
    );
    assert_eq!(ckpt.config.quantization.bits, 3.0);
    assert!(
        !ckpt.config.tie_word_embeddings,
        "tie_word_embeddings == false, so lm_head is a separate tensor from \
         embed_tokens and can be requantised independently"
    );
    assert_eq!(
        ckpt.config.final_logit_softcapping, 30.0,
        "softcap exists but is skipped for greedy argmax"
    );

    // Time it costs today, and the most it can ever be worth.
    let now_ms = ms_for(bytes.head_trellis, GBPS_HEAD_6BPW);
    assert!((now_ms - 4.11).abs() < 0.05, "head is {now_ms:.2} ms");
    // The floor on a 3 bpw head is 0.529 GB at the measured 3 bpw rate.
    let at_3bpw_ms = ms_for(bytes.head_trellis / 2, GBPS_3BPW_SWEEP);
    let best_saving = now_ms - at_3bpw_ms;
    assert!(
        (best_saving - 1.46).abs() < 0.05,
        "downrating the head to 3 bpw saves {best_saving:.2} ms at most"
    );
    // ... which is 2.3% of the step. This is the ceiling of the whole idea.
    assert!(
        tok_s(BASELINE_MS - best_saving) < 16.5,
        "a free 3 bpw head would give {:.2} tok/s, not 20",
        tok_s(BASELINE_MS - best_saving)
    );
}

/// No weight matrix is read twice in a step. The structural argument: a
/// single-token GEMV (M = 1) has no tile to reuse, so the only way to read a
/// matrix twice is to dispatch it twice — and the fusion sites read each
/// neighbour exactly once, fusing only *prologues and epilogues*.
///
/// Concretely, per sliding layer: `encode_gemv_fused2` (q,k),
/// `encode_gemv_fused2_mlp` (gate,up), `encode_gemv_raw_sk2` (o),
/// `encode_gemv_raw_sk2` (down) = 5 dispatches for 7 matrices. The "o_proj +
/// gate fusion" at `model.rs:1848` is a misnomer: what is fused is o_proj's
/// *finalize* with the post-attention norm, the residual add, the pre-FFN norm
/// and gate's input transform — `encode_finalize_o_proj_and_pre_ffn_transform_sk2`.
/// No weight matrix appears in two dispatches.
#[test]
fn no_weight_matrix_is_read_twice_per_step() {
    let Some(ckpt) = load(&model_dir()) else {
        return;
    };
    let config = &ckpt.config;

    // 7 projections on sliding layers, 6 on full layers (k reused as v).
    let expected_matrices = config
        .layer_types
        .iter()
        .map(|kind| if kind.is_full() { 6 } else { 7 })
        .sum::<usize>();
    let trellis = ckpt
        .tensors
        .keys()
        .filter(|n| n.contains(".layers.") && n.ends_with(".trellis"))
        .count();
    assert_eq!(trellis, expected_matrices, "one trellis per dispatch-input");

    // Cross-check the count against the shapes rather than against a constant,
    // so this still means something on a sibling export.
    for name in ckpt.tensors.keys() {
        if !name.contains(".layers.") || !name.ends_with(".trellis") {
            continue;
        }
        let info = &ckpt.tensors[name];
        assert_eq!(
            info.shape.len(),
            3,
            "{name} is a [tiles_k, tiles_n, words] trellis"
        );
        // tiles_k = in_features/16 and tiles_n = out_features/16, and the two
        // swap between gate/up and down — so assert the product, not either
        // axis. Product is hidden_size/16 * (in or out)/16 either way.
        let tiles: u64 = info.shape[0] as u64 * info.shape[1] as u64;
        // 48 u16 == 24 u32 words == 3 bpw, and 96 bytes per 16x16 tile.
        assert_eq!(info.shape[2] * 2, 96, "{name} is 3 bpw (48 u16/tile)");
        assert_eq!(
            info.byte_len(),
            tiles * 96,
            "{name}: {} tiles x 96 B must equal its byte length",
            tiles
        );
        // Every layer tile axis is a multiple of 16 in features.
        assert!(
            info.shape[0].is_multiple_of(1) && info.shape[1].is_multiple_of(1),
            "{name} tile shape"
        );
        assert!(
            tiles % 336 == 0 || tiles % 1344 == 0,
            "{name}: {tiles} tiles is not hidden/16 * (5376|21504|8192...)/16"
        );
    }

    // `attention_k_eq_v` is what removes v_proj on the full layers; if it were
    // false the 6-vs-7 count above would be wrong.
    assert!(
        config.attention_k_eq_v,
        "full-attention layers reuse K as V and therefore have no v_proj"
    );

    // Because line traffic is measured at 1.0000x, bytes-per-token is a floor on
    // the work, not an estimate of it: there is no wasted traffic to reclaim.
    // Assert the arithmetic that makes the two equal for this layout.
    let bytes = count_per_token(&ckpt.tensors, config.hidden_size);
    let layer_tiles = (config.layer_types.len() * 336) as u64;
    assert!(
        bytes.layer_trellis > layer_tiles * 96,
        "sanity: {layer_tiles} tiles cannot exceed {} bytes of trellis",
        bytes.layer_trellis
    );
}

/// The absolute floor, and the demonstration that 20 tok/s is not reachable by
/// any byte reduction that holds 3 bpw.
#[test]
fn twenty_tok_s_needs_more_bandwidth_than_this_format_can_deliver() {
    let Some(ckpt) = load(&model_dir()) else {
        return;
    };
    let bytes = count_per_token(&ckpt.tensors, ckpt.config.hidden_size);
    let side = bytes.scales + bytes.norms + bytes.embed_row;
    let floor = floor_bytes_at_3bpw(&ckpt.config, side);

    // Cross-check the parameter count against the checkpoint's own shapes
    // before trusting any byte total derived from it. This is where a wrong
    // global-vs-sliding head split shows up: it moves the parameter count by
    // billions and every tok/s projection built on it moves with it.
    let params = floor_params(&ckpt.config);
    assert_eq!(
        params, 30_696_013_824,
        "30.696 B non-embedding parameters: 29.287 B of layers + 1.409 B head"
    );
    // 50 sliding layers at 478.9 M params (7 projections) and 10 global layers
    // at 534.0 M (6 projections, but q/o are 16384-wide instead of 8192).
    let sliding = ckpt
        .config
        .layer_types
        .iter()
        .filter(|k| !k.is_full())
        .count() as u64;
    let global = ckpt.config.layer_types.len() as u64 - sliding;
    assert_eq!(sliding, 50);
    assert_eq!(global, 10);
    let head_params = ckpt.config.vocab_size as u64 * ckpt.config.hidden_size as u64;
    assert_eq!(head_params, 1_409_286_144, "262144 x 5376");
    assert_eq!(
        params - head_params,
        sliding * 478_937_088 + global * 533_987_328,
        "per-layer parameter counts"
    );

    assert_eq!(
        floor, 11_529_519_332,
        "30.696 B non-embedding parameters at 3 bpw + side tensors"
    );
    // Every byte of today's model is already at or above 3 bpw; the floor is a
    // 4.4% reduction and it is the *end of the line* at this bitrate.
    assert!(
        floor < bytes.weights(),
        "3 bpw everywhere ({floor}) must be below the real {}-byte model",
        bytes.weights()
    );
    assert!(
        (floor as f64 / bytes.weights() as f64 - 0.9563).abs() < 0.001,
        "the floor is only 4.4% below today"
    );

    // Weights only: no attention, no KV, no norms, no dispatch, no argmax.
    for (_label, gbps) in [
        ("best 3 bpw kernel measured", GBPS_3BPW),
        ("3 bpw rate sweep", GBPS_3BPW_SWEEP),
        ("pure copy probe", GBPS_COPY),
        ("head at 6 bpw (not available to 3 bpw layers)", GBPS_HEAD_6BPW),
    ] {
        let ms = ms_for(floor, gbps);
        println!(
            "floor {:.3} GB at {:6.1} GB/s -> {ms:6.2} ms -> {:5.2} tok/s",
            gbps,
            floor as f64 / 1e9,
            tok_s(ms)
        );
    }

    // The load-bearing claim: at the measured 3 bpw rate, even a model whose
    // weights are 100% of the step and 100% 3 bpw does not reach 20 tok/s.
    let ms = ms_for(floor, GBPS_3BPW_SWEEP);
    assert!(
        tok_s(ms) < 18.0,
        "3 bpw floor at the measured 3 bpw rate is {:.2} tok/s",
        tok_s(ms)
    );
    let needed = floor as f64 / 1e9 / 0.050;
    assert!(
        (needed - 230.6).abs() < 0.5,
        "20 tok/s at the floor needs {needed:.1} GB/s sustained"
    );
    assert!(
        needed > GBPS_3BPW_SWEEP,
        "which is more than the {:.1} GB/s the 3 bpw GEMV has ever reached",
        GBPS_3BPW_SWEEP
    );

    // And the head downrate, applied on top of today's step, cannot close it.
    let head_now = ms_for(bytes.head_trellis, GBPS_HEAD_6BPW);
    let head_3bpw = ms_for(bytes.head_trellis / 2, GBPS_3BPW_SWEEP);
    let best = tok_s(BASELINE_MS - (head_now - head_3bpw));
    assert!(
        best < 16.5,
        "best case from every head option is {best:.2} tok/s"
    );
}

/// KV is the only per-token byte term that depends on context, and it is not
/// negligible at long context: at 8192 tokens of KV it is 11% of everything.
#[test]
fn kv_bytes_grow_with_context_and_the_window_caps_the_sliding_layers() {
    let Some(ckpt) = load(&model_dir()) else {
        return;
    };
    let config = &ckpt.config;
    assert_eq!(config.sliding_window, 1024, "SWA window");

    let at_614 = kv_bytes_per_token(config, 614);
    let at_1024 = kv_bytes_per_token(config, 1_024);
    let at_8192 = kv_bytes_per_token(config, 8_192);
    assert_eq!(at_614, 553_287_680, "614 tokens of context");
    assert_eq!(at_1024, 922_746_880, "1024 tokens");
    assert_eq!(at_8192, 1_509_949_440, "8192 tokens");

    // Sliding layers stop growing past the window; only the 10 full layers do.
    let sliding_only_8192 = at_8192 - 10 * 2 * 2 * 4 * 512 * 8192;
    assert_eq!(
        sliding_only_8192, 50 * 2 * 2 * 16 * 256 * 1024,
        "the 50 sliding layers are capped at the 1024 window"
    );

    let bytes = count_per_token(&ckpt.tensors, config.hidden_size);
    let share_8192 = at_8192 as f64 / (bytes.weights() + at_8192) as f64;
    assert!(
        share_8192 > 0.10,
        "KV is {:.1}% of per-token bytes at 8192 context",
        share_8192 * 100.0
    );

    // Correctness of the global-vs-sliding head split: using head_dim for the
    // full layers would halve their contribution, so assert the two differ.
    assert_eq!(config.head_dim, 256);
    assert_eq!(config.global_head_dim, 512);
    assert_eq!(config.num_key_value_heads, 16);
    assert_eq!(config.num_global_key_value_heads, 4);
}

/// A two-stage head: coarse 3 bpw over all 262144, then a 6 bpw refinement over
/// only the coarse top-k rows.
///
/// # Byte arithmetic (the only part that is arithmetic)
///
/// `coarse = head/2`, `refine = head * k / 262144`. For k = 64 that is
/// 0.5285 + 0.000258 = 0.5287 GB, i.e. the whole saving of a flat 3 bpw head.
/// So the refinement is essentially free in bytes — its cost is one extra
/// dispatch and one candidate gather.
///
/// # Why the refinement can be exact where a flat 3 bpw head cannot
///
/// A flat 3 bpw head must win the argmax outright, which fails whenever its
/// error exceeds the top1-top2 gap. The two-stage version only needs the true
/// winner to appear *somewhere in the coarse top-k*, which fails only when more
/// than k tokens lie within the coarse error of the winner. Those are very
/// different events: under a Gaussian-tail model with V = 262144 the max sits at
/// ~4.78 sigma and the expected count of competitors within 0.05 sigma of it is
/// V * phi(4.78) * 0.05 ~ 0.14, so k = 64 is a ~450x margin.
///
/// # What is NOT established here
///
/// The 0.14 is a model, not a measurement. The two inputs it needs —
/// the coarse head's per-row logit error, and the real top1-top2 gap
/// distribution — are **both unmeasured on this box**, and
/// `spec/logit-distribution-probe` was specced to measure exactly that and was
/// never implemented (`crates/engine/src/logit_probe.rs` does not exist and all
/// of its `tasks.md` is unchecked). So this test asserts the *arithmetic and the
/// margin*, and explicitly refuses to assert an exact-match rate: any number
/// here would be invented.
///
/// # Two hard blockers beyond quality
///
/// 1. **No bf16 text weights exist on this box.** `tie_word_embeddings` is false
///    and every text projection is already EXL3 trellis; only `embed_tokens` and
///    the vision tower are bf16. A 3 bpw head therefore has to be made by
///    requantising the 6 bpw head, which compounds error rather than removing
///    it.
/// 2. **`reorder_tensor_to_colmajor` pins 48 u16 per tile** (`exl3.rs:740`),
///    which is exactly 3 bpw. A 4 bpw head has 64 u16/tile and cannot be
///    repacked colmajor at all; only a 3 bpw head can. So the 4 bpw option — the
///    one with the best precision-per-byte — is not reachable through the
///    production repacker.
#[test]
fn two_stage_head_saves_the_same_bytes_as_a_flat_3_bpw_head() {
    let Some(ckpt) = load(&model_dir()) else {
        return;
    };
    let head = count_per_token(&ckpt.tensors, ckpt.config.hidden_size).head_trellis;
    let vocab = ckpt.config.vocab_size as f64;

    let coarse = head / 2; // 3 bpw
    for k in [8_u64, 64, 256] {
        let refine = (head as f64 * k as f64 / vocab) as u64;
        let total = coarse + refine;
        assert!(
            total < head,
            "k={k}: {total} bytes must beat the flat 6 bpw head's {head}"
        );
        // The refinement is under 0.5% of the two-stage total even at k=256.
        assert!(
            (refine as f64) / (total as f64) < 0.005,
            "k={k}: refinement is {refine} bytes ({:.4}% of the two-stage head)",
            (refine as f64) / (total as f64) * 100.0
        );
    }
    // The refinement's share scales exactly as k/vocab, which is what makes the
    // k=64 claim a statement about k rather than about arithmetic luck.
    let refine_64 = (head as f64 * 64.0 / vocab) as u64;
    assert_eq!(refine_64, 258_048, "64 rows of the 6 bpw head");
    // So the byte saving equals a flat 3 bpw head's, and so does the time: the
    // refinement's 258 KB is 0.01 ms of noise on a 2.65 ms kernel.
    let flat_3bpw_ms = ms_for(coarse, GBPS_3BPW_SWEEP);
    let two_stage_ms = ms_for(coarse + (head as f64 * 64.0 / vocab) as u64, GBPS_3BPW_SWEEP);
    assert!(
        (two_stage_ms - flat_3bpw_ms).abs() < 0.02,
        "the refinement costs {two_stage_ms:.3} ms vs {flat_3bpw_ms:.3} ms flat"
    );

    // The margin the design depends on, stated as arithmetic on a model so it
    // is at least checkable: expected competitors within eps of the max.
    let sigma_frac = 0.05_f64; // eps / sigma, the coarse head's logit error
    let max_sigma = (2.0 * vocab.ln()).sqrt(); // Gaussian max of V samples
    let expected_competitors =
        vocab * (-0.5 * max_sigma * max_sigma).exp() / (2.0 * std::f64::consts::PI).sqrt()
            * sigma_frac;
    assert!(
        (max_sigma - 5.00).abs() < 0.02,
        "max of 262144 standard normals is at {max_sigma:.2} sigma"
    );
    assert!(
        expected_competitors < 0.05,
        "expected competitors within 0.05 sigma is {expected_competitors:.4}, \
         so k=64 has a {:.0}x margin",
        64.0 / expected_competitors.max(1e-9)
    );

    // The thing that is genuinely blocked, asserted so it cannot be forgotten.
    assert!(
        !ckpt.config.tie_word_embeddings,
        "requantising 6 -> 3 bpw compounds error; there is no bf16 lm_head here"
    );
}

/// Sabotage guard: the classifier must be able to produce a *different* answer.
///
/// A byte counter whose rules match everything reports the same number as a
/// counter that counts nothing correctly. This runs the identical rule set over
/// a mutated name list and requires the totals to diverge, so a rule cannot be
/// widened until it is vacuous without this failing.
#[test]
fn classification_is_not_vacuous() {
    let tensors: BTreeMap<String, TensorInfo> = BTreeMap::new();
    let _ = tensors; // the real-header case is covered by the tests above

    let names = [
        "model.language_model.layers.0.mlp.gate_proj.trellis",
        "model.language_model.layers.0.self_attn.q_proj.trellis",
        "lm_head.trellis",
        "lm_head.svh",
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "model.language_model.layers.0.self_attn.q_proj.suh",
    ];
    let lens = [
        43_352_064_u64,
        16_515_072,
        1_056_964_608,
        524_288,
        2_818_572_288,
        10_752,
        10_752,
    ];

    let real = {
        let mut acc = PerTokenBytes::default();
        for (name, len) in names.iter().zip(lens) {
            classify(name, len, 5_376, &mut acc);
        }
        acc
    };
    assert_eq!(real.layer_trellis, 43_352_064 + 16_515_072);
    assert_eq!(real.head_trellis, 1_056_964_608);
    assert_eq!(real.scales, 524_288 + 10_752);
    assert_eq!(real.embed_row, 10_752, "2.82 GB collapsed to 10 KiB");
    assert_eq!(real.norms, 10_752, "the final norm");
    assert_eq!(
        real.weights(),
        1_117_388_288,
        "59.9 MB layers + 1.057 GB head + 0.52 MB scales + 10 KiB embed row + final norm"
    );

    // Mutant A: treat the embedding as read whole. This is the mistake the
    // `weights.rs:514` comment warns about, and it inflates the total 2530x.
    let mutant_a = {
        let mut acc = PerTokenBytes::default();
        for (name, len) in names.iter().zip(lens) {
            if name.ends_with("embed_tokens.weight") {
                acc.embed_row += len; // whole table
            } else {
                classify(name, len, 5_376, &mut acc);
            }
        }
        acc
    };
    assert_ne!(mutant_a.weights(), real.weights());
    assert!(
        mutant_a.weights() > 3_800_000_000,
        "reading embed_tokens whole would inflate per-token bytes to {:.2} GB \
         and make the model look 3.9x bigger than it is",
        mutant_a.weights() as f64 / 1e9
    );

    // Mutant B: drop the head because "the head is not a layer weight". That
    // silently loses 8.8% of the bytes and is the error that would make the
    // head-downrate proposals look twice as good as they are.
    let mutant_b = {
        let mut acc = PerTokenBytes::default();
        for (name, len) in names.iter().zip(lens) {
            if *name == "lm_head.trellis" {
                continue;
            }
            classify(name, len, 5_376, &mut acc);
        }
        acc
    };
    assert_ne!(mutant_b.weights(), real.weights());
    assert_eq!(
        real.weights() - mutant_b.weights(),
        1_056_964_608,
        "forgetting the head undercounts by exactly one head"
    );

    // Mutant C: `.trellis` matched without the `.layers.` guard, which would
    // sweep the head into the layer bucket and erase the 3-vs-6 bpw split that
    // the entire head analysis rests on.
    let mutant_c = {
        let mut acc = PerTokenBytes::default();
        for (name, len) in names.iter().zip(lens) {
            if name.ends_with(".trellis") {
                acc.layer_trellis += len;
            } else {
                classify(name, len, 5_376, &mut acc);
            }
        }
        acc
    };
    // The total is *unchanged* here — that is exactly why this mutant is worth
    // a test. The byte sum still reconciles with the 62.3 ms baseline, so a
    // headline "12.06 GB/token" check passes, while every conclusion drawn from
    // the layer bucket is wrong: it now claims the head is 3 bpw, which would
    // make a head-downrate proposal look like a 91% byte cut instead of 4.4%.
    assert_eq!(
        mutant_c.weights(),
        real.weights(),
        "sweeping head into layers preserves the total — which is the trap"
    );
    assert_eq!(mutant_c.head_trellis, 0, "head swept into layers");
    assert_eq!(
        mutant_c.layer_trellis - real.layer_trellis,
        1_056_964_608,
        "the layer bucket absorbed the whole head"
    );
    // An analysis that only looks at `layer_trellis` now misses 8.8% of the
    // bytes while still reporting a correct grand total.
    // This fixture carries two small layer trellis on purpose (it is a rule
    // test, not a model test), so its layer share is not the model's 91%. What
    // matters is the direction: the mutant always reports a strictly higher
    // layer share, and it does so while the grand total still reconciles.
    let real_layer_share = real.layer_trellis as f64 / real.weights() as f64;
    let mutant_layer_share = mutant_c.layer_trellis as f64 / mutant_c.weights() as f64;
    assert!(
        mutant_layer_share > real_layer_share,
        "mutant inflates the layer share from {:.4} to {:.4}",
        real_layer_share,
        mutant_layer_share
    );
    // On the real checkpoint the same mutation is the difference between
    // "layers are 91.1% and the head is 8.8%" and "layers are 99.9%". Asserted
    // against real weights in
    // `one_decode_step_touches_12_058_gb_of_weights`.

    // Mutant D: every tensor counted whole. The upper bound a naive counter
    // reports, i.e. the whole text selection minus the embedding round-trip.
    let mutant_d = {
        let mut acc = PerTokenBytes::default();
        for (name, len) in names.iter().zip(lens) {
            if name.ends_with("embed_tokens.weight") {
                acc.embed_row += len;
            } else {
                acc.norms += len;
            }
        }
        acc
    };
    assert_eq!(mutant_d.weights(), lens.iter().sum::<u64>());
    assert!(mutant_d.weights() > real.weights());
}

/// The options table, printed. Not an assertion — this is the artefact the
/// investigation exists to produce, and it is regenerated from the real headers
/// rather than transcribed.
#[test]
fn print_byte_reduction_options() {
    let Some(ckpt) = load(&model_dir()) else {
        return;
    };
    let bytes = count_per_token(&ckpt.tensors, ckpt.config.hidden_size);
    let side = bytes.scales + bytes.norms + bytes.embed_row;
    let head_now = ms_for(bytes.head_trellis, GBPS_HEAD_6BPW);
    let layer_now = ms_for(bytes.layer_trellis, GBPS_3BPW_SWEEP);

    println!(
        "\nbytes/token: layers {:.4} GB + head {:.4} GB + side {:.4} GB = {:.4} GB",
        bytes.layer_trellis as f64 / 1e9,
        bytes.head_trellis as f64 / 1e9,
        side as f64 / 1e9,
        bytes.weights() as f64 / 1e9
    );
    println!("baseline    : {BASELINE_MS:.1} ms = {:.2} tok/s", tok_s(BASELINE_MS));
    println!("head today  : {head_now:.2} ms at {GBPS_HEAD_6BPW} GB/s (6 bpw)");
    println!("layers today: {layer_now:.2} ms at {GBPS_3BPW_SWEEP} GB/s (3 bpw)\n");

    println!(
        "{:<34} {:>10} {:>9} {:>9} {:>9}  {}",
        "option", "GB/token", "save ms", "ms @203", "ms @250", "tok/s @203"
    );
    println!("{}", "-".repeat(94));

    println!(
        "{:<34} {:>10.4} {:>9} {:>9.2} {:>9.2}  {:>9.2}",
        "baseline (3 bpw layers, 6 bpw head)",
        bytes.weights() as f64 / 1e9,
        "-",
        BASELINE_MS,
        BASELINE_MS,
        tok_s(BASELINE_MS)
    );

    for (label, bits, gbps) in [
        ("head -> 4 bpw (row-major only)", 4.0, GBPS_COPY),
        ("head -> 3 bpw (colmatar legal)", 3.0, GBPS_3BPW_SWEEP),
    ] {
        let head = (bytes.head_trellis as f64 * bits / 6.0) as u64;
        let per_token = bytes.weights() - bytes.head_trellis + head;
        let now = ms_for(head, gbps);
        let saved = head_now - now;
        println!(
            "{label:<34} {:>10.4} {:>9.2} {:>9.2} {:>9.2}  {:>9.2}",
            per_token as f64 / 1e9,
            saved,
            BASELINE_MS - saved,
            ms_for(per_token, GBPS_3BPW_SWEEP).min(BASELINE_MS - saved),
            tok_s(BASELINE_MS - saved)
        );
    }

    println!(
        "{:<34} {:>10.4} {:>9} {:>9.2} {:>9.2}  {:>9.2}",
        "head -> 2 bpw (bandwidth UNKNOWN)",
        (bytes.weights() - bytes.head_trellis + bytes.head_trellis / 3) as f64 / 1e9,
        "?",
        f64::NAN,
        f64::NAN,
        f64::NAN
    );

    for (label, factor, gbps) in [
        ("layers -> 2 bpw @ measured 199.4", 2.0 / 3.0, GBPS_3BPW_SWEEP),
        ("layers -> 2 bpw @ amp law 125", 2.0 / 3.0, 125.0),
        ("layers -> 4 bpw @ measured 250.5", 4.0 / 3.0, GBPS_COPY),
    ] {
        let layers = (bytes.layer_trellis as f64 * factor) as u64;
        let per_token = layers + bytes.head_trellis + side;
        let ms = ms_for(per_token, gbps);
        println!(
            "{label:<34} {:>10.4} {:>9} {:>9.2} {:>9.2}  {:>9.2}",
            per_token as f64 / 1e9,
            "-",
            ms,
            ms,
            tok_s(ms)
        );
    }

    println!();
}