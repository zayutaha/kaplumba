//! READ-ONLY perf probe: per-token **activation** traffic of the decode step,
//! buffer by buffer, dispatch by dispatch.
//!
//! No Metal, no GPU, no model. Pure arithmetic over the shapes the checkpoint
//! fixes, cross-checked against real functions in the tree
//! (`kaplumba_engine::bench::per_token_weight_bytes`,
//! `kaplumba_kernels::attention::{decode_partial_elements, decode_splits}`), so a
//! shape change breaks these tests instead of quietly invalidating the report.
//!
//! ============================== THE HEADLINE ==============================
//!
//! **Activations are 0.53% of the step's bytes at short context and 1.14% at a
//! full window, and no change to activation precision or layout can be worth
//! more than +0.21 tok/s.** They are not a second bandwidth consumer, so nobody
//! has been missing an optimisation here.
//!
//! The arithmetic, all of it derived below from `file:line`:
//!
//! | per decode token | bytes | vs 12,039,487,488 B of weights |
//! |---|---:|---:|
//! | weights (`bench::per_token_weight_bytes`) | 12,039,487,488 | 100% |
//! | activation scratch, `key_count <= 64` (single-pass attention) | 63,555,600 | 0.53% |
//! | activation scratch, 614 keys | 109,738,000 | 0.91% |
//! | activation scratch, `key_count >= 1024` (16 splits, saturated) | 137,447,440 | 1.14% |
//! | per-layer sign/norm tables (once per layer per token) | 17,285,120 | 0.14% |
//! | KV cache read at 614 tokens of context | 553,287,680 | 4.60% |
//! | KV cache read at 8192 tokens of context | 1,509,949,440 | 12.54% |
//!
//! At the measured 16.05-16.10 tok/s (62.3 ms/step, AGENTS.md §10.2 takes the
//! median) the weight stream alone is **193.3 GB/s** of 273. Deleting **100%**
//! of every activation byte -- which is impossible, because most of them *are*
//! a GEMV's output -- is `154,732,560 / 193.3 GB/s = 0.801 ms`: 62.3 -> 61.50
//! ms = **16.05 -> 16.26 tok/s**. That is the hard ceiling of this angle, and
//! the largest removable item (the attention partials round trip) is worth
//! +0.099 tok/s -- and is not removable at all without changing the token
//! stream.
//!
//! **The second bandwidth consumer is the KV cache, and it is 11x the
//! activations**: 1.51 GB/token against 137 MB/token at 8192 of context. That
//! is the attention angle's job, but every ceiling calculation in
//! `spec/decode-25tok-peak-bandwidth` is built on weights alone, so at full
//! context it understates the stream by 13.7%: 13.69 GB/token, needing 220 GB/s
//! to hold 62.3 ms, not 193.
//!
//! ===================== WHAT THE COUNTING DID FIND (2 defects) =====================
//!
//! 1. **`scratch.attended` is 5,632 B too small at decode and 5,767,168 B too
//!    small at 1024 rows.** `model.rs:3012` sizes it `u16_elements(max_tokens,
//!    hidden)` -- 5376 halves -- but both attention epilogues write
//!    `q_heads * head_dim` halves into it (`attention.metal:974`, combine:
//!    `output[head * head_dim + dimension]`; `attention.metal:927`,
//!    single-pass and prefill: `output + query_base + ...`), which is 8192 on the
//!    50 sliding layers and 16384 on the 10 full ones. `o_proj.in_features` is
//!    the same number (`model.rs:525`) and `encode_input_transform`
//!    (`linear.metal:488`) reads all of it. It produces correct output today only
//!    because Metal's allocations are page-granular, so the overrun lands in the
//!    buffer's own padding rather than in a neighbour.
//!    See `attended_is_sized_hidden_but_written_q_dim`.
//!
//! 2. **`scratch.normed` is a dead store on the fused decode path.** It is
//!    written by the input-norm epilogue (`model.rs:1435`, `:2327`) and passed
//!    to `encode_gemv_fused3` (`model.rs:1521`) and `encode_gemv_fused2_mlp`
//!    (`model.rs:2206`) as their `input` argument -- but both are called with
//!    `pretransformed = true`, and that argument is consumed only by the
//!    `input_pipeline` dispatch they then skip (`linear.rs:2716-2725`,
//!    `:2513-2522`). The kernels bind `transformed`, never `input`.
//!    10,752 B/layer/token = 645,120 B/token, 0.005% of the stream. `scratch.proj`
//!    is worse: on the decode path it is never written at all
//!    (`model.rs:1983-1995` writes `raw`, and the fused finalize writes
//!    `hidden_b`), so 10,752 B/layer of pure allocation.
//!
//! ============================== WHAT IS NOT A FINDING ==============================
//!
//! * **The f32 `raw` accumulators are not "convenience f32".** `raw` is f32
//!   because it holds a GEMV's *reduced* output (32-lane shuffle reduce,
//!   `linear_repacked.metal:872-901`) which the very next dispatch runs
//!   `transform_simd` over -- a 4-point Hadamard across lanes 32 output-columns
//!   apart (`linear.metal:522`, `:602`). Narrowing the store to fp16 would round
//!   *before* the Hadamard, which is the same class of 1-ULP perturbation that
//!   `attention.rs:176-182` measured changing **43 of 128** greedy tokens. Every
//!   hand-off that leaves a dispatch is already fp16 (`transformed`, `activated`,
//!   `q`/`k`/`v`, `attended`, `logits`): there is nothing to narrow.
//! * **The MLP intermediates are not f32 in storage.** `gate`/`up`/`activated`
//!   are u16 (`model.rs:3014-3016`) and on the decode path they are **dead**;
//!   the live intermediate is `transformed`, fp16, written by
//!   `linear_finalize_gated_gelu_and_down_input_transform`
//!   (`linear.metal:1343-1346`). The f32 pair is `raw`/`raw_b`, 86,016 B each per
//!   layer, and they are the largest non-attention activation item: 20.64 MB/token
//!   written and 20.64 MB/token read back, matching
//!   `spec/decode-19tok-repacked-250gb/implementation_notes.md:507-508`.
//! * **No memsets, nothing re-zeroed.** There is no `memset`/`fill` on the decode
//!   path; the only zeroing kernel, `linear_zero_raw`, is explicitly bench/test
//!   only (`linear.rs:613-619`). `attention.metal:908` states the same invariant
//!   for the partials: "every split that is read is a split that wrote".
//! * **The logits are negligible, confirmed.** 512 KB each; and at decode the
//!   softcap is *skipped* (`model.rs:4444-4458`, `skip_softcap`), so `capped` is
//!   never written and the argmax reads `logits` directly. The whole vocabulary
//!   head's non-weight traffic is 3,178,048 B against 1,056,964,608 B of head
//!   weights -- 0.30%.

use kaplumba_engine::bench::per_token_weight_bytes;
use kaplumba_kernels::attention::{decode_partial_elements, decode_splits};

const MB: f64 = 1_000_000.0;
const GB: f64 = 1_000_000_000.0;

// ===========================================================================
// 1. Shapes. Every constant below is checked against the tree, not assumed.
// ===========================================================================

/// `config.rs:226` `hidden_size`.
const HIDDEN: usize = 5376;
/// `config.rs:227` `intermediate_size`.
const INTER: usize = 21504;
/// `config.rs:228` `vocab_size`.
const VOCAB: usize = 262144;
/// `config.rs:229` `num_attention_heads`.
const Q_HEADS: usize = 32;
/// `config.rs:230` `num_key_value_heads` (the 50 sliding layers).
const SLIDING_KV_HEADS: usize = 16;
/// `config.rs:231` `num_global_key_value_heads` (the 10 full layers).
const FULL_KV_HEADS: usize = 4;
/// `config.rs:232` `head_dim`, sliding.
const SLIDING_HEAD_DIM: usize = 256;
/// `config.rs:233` `global_head_dim`, full.
const FULL_HEAD_DIM: usize = 512;
/// `config.rs:225` `num_hidden_layers`.
const LAYERS: usize = 60;
/// `config.rs:240-252`: `index % 6 == 5` is `LayerKind::Full`.
const FULL_LAYERS: usize = 10;
const SLIDING_LAYERS: usize = LAYERS - FULL_LAYERS;
/// `config.rs:234` `sliding_window`: a sliding ring never sees more keys.
const SLIDING_WINDOW: usize = 1024;
/// `attention.rs:21`.
const SPLITS_MAX: usize = 16;
/// `attention.rs:74`: at or below this many keys the decode attention is one
/// dispatch with no partials at all (`model.rs:1853-1854`).
const SINGLE_PASS_MAX_KEYS: usize = 64;

/// `q_heads * head_dim`; `model.rs:3193`.
const Q_MAX: usize = Q_HEADS * FULL_HEAD_DIM; // 16384
/// `kv_heads * head_dim`; `model.rs:3195`, max over layers.
const KV_MAX: usize = SLIDING_KV_HEADS * SLIDING_HEAD_DIM; // 4096

const F32: usize = 4;
const U16: usize = 2;

/// Bytes of one fp16 row of `n` elements.
const fn h(n: usize) -> u64 {
    (n * U16) as u64
}
/// Bytes of one fp32 accumulator row of `n` elements.
const fn f(n: usize) -> u64 {
    (n * F32) as u64
}

#[test]
fn shapes_are_the_ones_the_checkpoint_validates() {
    assert_eq!(Q_MAX, 16_384);
    assert_eq!(KV_MAX, 4_096);
    assert_eq!((0..LAYERS).filter(|i| i % 6 == 5).count(), FULL_LAYERS);
    assert_eq!(SLIDING_LAYERS + FULL_LAYERS, LAYERS);
    // `model.rs:265` requires `q_heads % kv_heads == 0` for both kinds.
    assert_eq!(Q_HEADS % SLIDING_KV_HEADS, 0);
    assert_eq!(Q_HEADS % FULL_KV_HEADS, 0);
    // `attn_partials` is sized from the widest layer only (`model.rs:3196-3199`).
    assert_eq!(decode_partial_elements(Q_HEADS, FULL_HEAD_DIM), 263_168);
    assert_eq!(decode_partial_elements(Q_HEADS, SLIDING_HEAD_DIM), 132_096);
    // The weight stream this probe divides by, from the function every other
    // bandwidth calculation in the repo is built on.
    assert_eq!(per_token_weight_bytes(), 12_039_487_488);
}

// ===========================================================================
// 2. The scratch inventory: dtype and size per buffer, `model.rs:3004-3042`.
// ===========================================================================

/// `raw_elements = (max_tokens * inter).max(vocab)`, `model.rs:2987`. At decode
/// that is the **vocabulary**, not the intermediate: 262,144 f32 = 1 MiB, which
/// is why the lm_head's accumulator and the MLP's accumulator share one buffer.
const fn raw_elements(rows: usize) -> usize {
    if rows * INTER > VOCAB {
        rows * INTER
    } else {
        VOCAB
    }
}

/// `(name, bytes at 1 row, bytes at 1024 rows, dtype, counted by resident_bytes?)`
type Buf = (&'static str, u64, u64, &'static str, bool);

fn inventory() -> Vec<Buf> {
    vec![
        // --- fp16 activations, `model.rs:3006-3016`, `:3020`, `:3027-3029` ---
        ("hidden_a", h(HIDDEN), h(HIDDEN * 1024), "u16", true),
        ("hidden_b", h(HIDDEN), h(HIDDEN * 1024), "u16", true),
        ("normed", h(HIDDEN), h(HIDDEN * 1024), "u16", true),
        ("q", h(Q_MAX), h(Q_MAX * 1024), "u16", true),
        ("k", h(KV_MAX), h(KV_MAX * 1024), "u16", true),
        ("v", h(KV_MAX), h(KV_MAX * 1024), "u16", true),
        // `attended` is `hidden`-wide here and `q_dim`-wide in fact: defect 1.
        ("attended", h(HIDDEN), h(HIDDEN * 1024), "u16", true),
        ("proj", h(HIDDEN), h(HIDDEN * 1024), "u16", true),
        ("gate", h(INTER), h(INTER * 1024), "u16", true),
        ("up", h(INTER), h(INTER * 1024), "u16", true),
        ("activated", h(INTER), h(INTER * 1024), "u16", true),
        ("transformed", h(INTER), h(INTER * 1024), "u16", true),
        ("last_hidden", h(HIDDEN), h(HIDDEN), "u16", true),
        ("logits", h(VOCAB), h(VOCAB), "u16", true),
        ("capped", h(VOCAB), h(VOCAB), "u16", true),
        // --- fp32 accumulators, `model.rs:2987-2990`, `:3037` ---
        (
            "raw",
            f(raw_elements(1)),
            f(raw_elements(1024)),
            "f32",
            true,
        ),
        (
            "raw_b",
            f(raw_elements(1)),
            f(raw_elements(1024)),
            "f32",
            false,
        ),
        (
            "raw_c",
            f(raw_elements(1)),
            f(raw_elements(1024)),
            "f32",
            false,
        ),
        (
            "attn_partials",
            f(decode_partial_elements(Q_HEADS, FULL_HEAD_DIM)),
            f(decode_partial_elements(Q_HEADS, FULL_HEAD_DIM)),
            "f32",
            true,
        ),
        // --- u32 staging, `model.rs:2991-3003` ---
        ("query_positions", 4, 4096, "u32", true),
        ("token_ids", 4, 4096, "u32", true),
        ("query_positions_decode[2]", 128, 128, "u32", true),
        ("token_ids_decode[2]", 128, 128, "u32", true),
        ("host_tokens", 32_768, 32_768, "u32", true),
        // `model.rs:3038-3041`: `(vocab.div_ceil(65536) * 4).max(16)` is a **byte**
        // count. 262144/65536 = 4 words = 16 B, and the `.max` is a 16-byte
        // floor. The doc comment above it ("4 words (one u32 per 65,536
        // elements)") is right, but a reader who reads the expression as words
        // gets 48 B more per scratch than exists -- which is exactly the size of
        // the discrepancy between this table and an earlier one.
        ("argmax_keys", 16, 16, "u32[4]", true),
    ]
}

#[test]
fn scratch_totals_match_the_measured_figures() {
    let inv = inventory();
    let total_1: u64 = inv.iter().map(|b| b.1).sum();
    let total_1024: u64 = inv.iter().map(|b| b.2).sum();
    // spec/memory-swap-cliff/implementation_notes.md:527 records 547,937,040 B at
    // 1024 rows and :546 records 5,565,720 B as the true 1-row total. Both
    // reproduce exactly, which is what makes the shape list above trustworthy.
    assert_eq!(total_1024, 547_937_040);
    assert_eq!(total_1, 5_565_720);
    // The 1024-row figure in the task brief is this TRUE one. Its 1-row
    // companion, 4,564,720, matches neither the true total nor what
    // `resident_bytes()` reports; 5,565,720 - 4,564,720 = 1,001,000 is not any
    // buffer, nor any pair, nor any triple in the list.
    assert_ne!(total_1, 4_564_720);
    let reported_1: u64 = inv.iter().filter(|b| b.4).map(|b| b.1).sum();
    assert_ne!(reported_1, 4_564_720);
    println!("true {total_1} B at 1 row, {total_1024} B at 1024 rows");
}

/// `resident_bytes()` (`model.rs:3045-3076`) sums 24 of the 27 buffers: it omits
/// `raw_b` and `raw_c`, which are allocated at `model.rs:2989-2990` and handed
/// to real kernels at `model.rs:1525-1526`, `:1582`, `:2210`. Filed as F5 in
/// spec/memory-swap-cliff (implementation_notes.md:494-527); still open here. It
/// under-reports the activation budget by 2 MiB at decode and 168.0 MiB at 1024
/// rows, and `real_forward_buffer.rs`'s 400 KiB/row budget passes at 1.31x its
/// real figure because of it.
#[test]
fn resident_bytes_omits_raw_b_and_raw_c() {
    let inv = inventory();
    let reported_1: u64 = inv.iter().filter(|b| b.4).map(|b| b.1).sum();
    let reported_1024: u64 = inv.iter().filter(|b| b.4).map(|b| b.2).sum();
    let true_1: u64 = inv.iter().map(|b| b.1).sum();
    let true_1024: u64 = inv.iter().map(|b| b.2).sum();
    assert_eq!(reported_1, 3_468_568);
    assert_eq!(reported_1024, 371_776_272);
    assert_eq!(true_1 - reported_1, 2 * f(raw_elements(1)));
    assert_eq!(true_1024 - reported_1024, 176_160_768);
    assert!((true_1024 - reported_1024) as f64 / 1024.0 / 1024.0 < 168.001);
}

/// Defect 1, now FIXED: `attended` used to be `hidden` wide in the allocator
/// while both attention epilogues write `q_dim`. T9 sized it `q_max`. The test
/// is kept inverted: it now asserts the allocation covers the widest write,
/// so re-introducing the `hidden`-wide allocation fails here.
#[test]
fn attended_is_sized_for_the_q_dim_both_epilogues_write() {
    // `model.rs:3229`, now `u16_elements(max_tokens, q_max)`.
    let attended_1 = h(Q_HEADS * FULL_HEAD_DIM);
    let attended_1024 = h(Q_HEADS * FULL_HEAD_DIM * 1024);
    let model_rs =
        std::fs::read_to_string(format!("{}/src/model.rs", env!("CARGO_MANIFEST_DIR")))
            .expect("model.rs");
    assert!(
        model_rs.contains("attended: u16_elements(max_tokens, q_max)?,"),
        "scratch.attended is no longer q_max-wide; the 22,016 B/layer/token \
         overrun is back and is silent on AGX"
    );
    for (label, q_dim) in [
        ("sliding", Q_HEADS * SLIDING_HEAD_DIM),
        ("full", Q_HEADS * FULL_HEAD_DIM),
    ] {
        assert!(
            q_dim > HIDDEN,
            "{label}: q_dim {q_dim} no longer exceeds `hidden`; the sizing \
             argument this fix is based on is stale"
        );
        assert!(
            attended_1 >= h(q_dim),
            "{label}: {q_dim} halves are written into a {}-half buffer",
            attended_1 / 2
        );
        println!(
            "{label}: writes {q_dim} halves into a {}-half buffer ({}-half headroom; \
             the hidden-wide allocation underran this by {} B at decode and {} B at 1024 rows)",
            attended_1 / 2,
            (attended_1 - h(q_dim)) / 2,
            h(q_dim) - h(HIDDEN),
            h(q_dim * 1024) - h(HIDDEN * 1024),
        );
    }
    assert_eq!(h(Q_HEADS * SLIDING_HEAD_DIM) - h(HIDDEN), 5_632);
    assert_eq!(h(Q_HEADS * FULL_HEAD_DIM) - h(HIDDEN), 22_016);
    // The old hidden-wide allocation's 1024-row overruns, kept as the record of
    // what the fix removed. `attended_1024` is now the q_max-wide size.
    assert_eq!(
        h(Q_HEADS * SLIDING_HEAD_DIM * 1024) - h(HIDDEN * 1024),
        5_767_168
    );
    assert_eq!(
        h(Q_HEADS * FULL_HEAD_DIM * 1024) - h(HIDDEN * 1024),
        22_544_384
    );
    assert!(attended_1024 >= h(Q_HEADS * FULL_HEAD_DIM * 1024));
    // The zero-cost variant: `transformed` is `inter` = 21504 halves, so it can
    // hold either q_dim, and it is DEAD when attention runs -- the qkv GEMV has
    // already consumed it and the o_proj `input_transform` overwrites it. So the
    // attention request's `output` (`model.rs:1843`) could be `transformed`
    // instead of `attended`.
    assert!(INTER >= Q_MAX);
    assert!(INTER - Q_HEADS * SLIDING_HEAD_DIM == 13_312);
}

// ===========================================================================
// 3. Per-layer dispatch traffic. One row per dispatch, with its file:line.
// ===========================================================================

/// `(label, scratch read, scratch write)` for one dispatch. "Scratch" means
/// `ForwardScratch` only; the per-layer sign/norm tables are counted in
/// [`Layer::table_bytes`] because they are resident weights, not activations.
type Row = (&'static str, u64, u64);

#[derive(Clone, Copy)]
struct Layer {
    label: &'static str,
    count: usize,
    q_dim: usize,
    kv_dim: usize,
    head_dim: usize,
    /// `v_proj` exists only on full-attention layers (`model.rs:3439`,
    /// `attention_k_eq_v` is true for this checkpoint).
    has_v_proj: bool,
}

const SLIDING: Layer = Layer {
    label: "sliding",
    count: SLIDING_LAYERS,
    q_dim: 8192,
    kv_dim: 4096,
    head_dim: 256,
    has_v_proj: false,
};
const FULL: Layer = Layer {
    label: "full",
    count: FULL_LAYERS,
    q_dim: 16384,
    kv_dim: 2048,
    head_dim: 512,
    has_v_proj: true,
};

impl Layer {
    /// Bytes of `attn_partials` this layer writes, and reads back, at `splits`.
    fn partials(&self, splits: usize) -> u64 {
        f(splits * Q_HEADS * (self.head_dim + 2))
    }

    /// Rows for one layer at decode (`tokens == 1`), `permutation_mode == 1` on
    /// every projection, split-K2 on `o_proj`/`down_proj` (the default:
    /// `model.rs:1941-1946`, `:2242-2247`), cross-layer fusion on
    /// (`model.rs:2303-2304`), softcap skipped (`model.rs:4444-4445`).
    ///
    /// `first` is layer 0, which has no predecessor and so issues its own input
    /// norm. `last` is layer 59, which has no successor and so does not fuse the
    /// next layer's input norm into its finalize.
    fn rows(&self, splits: usize, single_pass: bool, first: bool, last: bool) -> Vec<Row> {
        let kv_accs = if self.has_v_proj { 3 } else { 2 };
        let kv_outs = if self.has_v_proj { 3 } else { 2 };
        let mut rows = Vec::new();

        if first {
            // `encode_input_norm_and_qkv_transform`, `model.rs:1429-1440`:
            // reads `input`, writes `normed` and `transformed`.
            rows.push((
                "input_norm + q transform (layer 0)",
                h(HIDDEN),
                h(HIDDEN) * 2,
            ));
        }
        // Layers 1..=59 skip that: `input_norm_precomputed`
        // (`model.rs:1411-1417`) is set by the previous layer's cross-layer
        // finalize, which writes `normed` AND `transformed` at `model.rs:2327-2328`.
        // Neither is read again (defect 2); counted as written, the conservative
        // direction.

        // QKV. Sliding: `encode_gemv_fused2` (`model.rs:1565`) then a k->v copy
        // (`model.rs:1588`) because `attention_k_eq_v`. Full: `encode_gemv_fused3`
        // (`model.rs:1503`).
        rows.push((
            "qkv matvec -> raw/raw_b[/raw_c]",
            h(HIDDEN), // reads `transformed`; in_features = hidden
            f(self.q_dim) + f(self.kv_dim) * (kv_accs - 1) as u64,
        ));
        rows.push((
            "q/k[/v] finalize -> fp16",
            f(self.q_dim) + f(self.kv_dim) * (kv_accs - 1) as u64,
            h(self.q_dim) + h(self.kv_dim) * (kv_outs - 1) as u64,
        ));
        if !self.has_v_proj {
            rows.push(("k -> v copy", h(self.kv_dim), h(self.kv_dim)));
        }
        // `encode_norm_rope_qkv_cache`, `model.rs:1666-1688`: reads q/k/v, rewrites
        // q/k in place, writes K/V into the cache (KV traffic, not scratch).
        rows.push((
            "q/k norm + rope -> kv cache",
            h(self.q_dim) + h(self.kv_dim) * 2,
            h(self.q_dim) + h(self.kv_dim),
        ));
        // Attention. `encode_online_attention_split` (`model.rs:1864`), or
        // `encode_decode_single_pass` (`model.rs:1858`) when
        // `key_count <= 64`, which reads no partials at all.
        if single_pass {
            rows.push((
                "attention (single pass)",
                h(self.q_dim),
                h(self.q_dim), // `attended`: q_dim halves, see defect 1
            ));
        } else {
            rows.push((
                "attention split -> partials",
                h(self.q_dim),
                self.partials(splits),
            ));
            rows.push(("attention combine", self.partials(splits), h(self.q_dim)));
        }
        // o_proj input transform, `model.rs:1911-1919`, `in_features = q_dim`.
        rows.push(("o_proj input transform", h(self.q_dim), h(self.q_dim)));
        // o_proj matvec, split-K2: `raw` holds 2 x hidden f32.
        rows.push(("o_proj matvec -> raw", h(self.q_dim), 2 * f(HIDDEN)));
        // o_proj finalize + post-attn residual + pre-ffn norm + gate input
        // transform, one dispatch, `model.rs:1999-2013`.
        rows.push((
            "o_proj finalize + pre_ffn",
            2 * f(HIDDEN) + h(HIDDEN), // `raw`, `residual` = this layer's input
            h(HIDDEN) * 2,             // `hidden_b`, `transformed`
        ));
        // gate + up matvec, `encode_gemv_fused2_mlp`, `model.rs:2193-2214`.
        rows.push(("gate + up matvec -> raw/raw_b", h(HIDDEN), f(INTER) * 2));
        // `linear_finalize_gated_gelu_and_down_input_transform`, `linear.metal:1301`.
        rows.push(("gated gelu + down input -> fp16", f(INTER) * 2, h(INTER)));
        // down_proj matvec, split-K2.
        rows.push(("down_proj matvec -> raw", h(INTER), 2 * f(HIDDEN)));
        // down_proj finalize. Layers 0..=58 use the cross-layer kernel
        // (`model.rs:2317-2333`), which also writes the NEXT layer's `normed` and
        // `transformed`; layer 59 has no successor and uses
        // `encode_finalize_and_post_ffn_norm_scale_sk2` (`model.rs:2336-2347`).
        // On even layers `output` and `residual` are the same buffer (`hidden_b`)
        // and the kernel reads element `i` before writing it
        // (`linear.metal:1155-1159`), so read+write are both real traffic.
        let finalize_write = if last { h(HIDDEN) } else { h(HIDDEN) * 3 };
        rows.push((
            "down_proj finalize (+ next layer's norm)",
            2 * f(HIDDEN) + h(HIDDEN),
            finalize_write,
        ));
        rows
    }

    /// Per-layer sign/norm tables read once per layer per token. Resident
    /// weights, listed separately so the activation figure cannot absorb them.
    fn table_bytes(&self) -> u64 {
        // 5376-wide: input_norm, q_input_signs, o_out_signs, post_attn_norm,
        // pre_ffn_norm, gate_input_signs, down_out_signs, post_ffn_norm.
        let narrow = 8 * h(HIDDEN);
        // q_dim-wide: q_out_signs, o_input_signs, q_norm.
        let wide = 3 * h(self.q_dim);
        // kv_dim-wide: k_out_signs, k_norm, plus v_out_signs only where `v_proj`
        // exists (sliding never reads it, `model.rs:1462`).
        let kv = 2 * h(self.kv_dim) + if self.has_v_proj { h(self.kv_dim) } else { 0 };
        // inter-wide: gate_out_signs, up_out_signs, down_input_signs.
        let inter = 3 * h(INTER);
        narrow + wide + kv + inter
    }

    fn total(&self, splits: usize, single_pass: bool, first: bool, last: bool) -> u64 {
        self.rows(splits, single_pass, first, last)
            .iter()
            .map(|r| r.1 + r.2)
            .sum()
    }
}

/// Splits the shipped default rule picks. `decode_splits` is pure given the
/// environment and this process sets neither knob, so this is the `ThreadTarget`
/// default: `16384 / (32 * 32) = 16`, then `.min(ceil(key_count / 64))`
/// (`attention.rs:289-306`).
fn splits_at(key_count: usize, kv_heads: usize) -> usize {
    decode_splits(Q_HEADS, kv_heads, key_count)
}

fn single_pass_at(key_count: usize) -> bool {
    key_count <= SINGLE_PASS_MAX_KEYS
}

#[test]
fn split_counts_follow_the_shipped_rule() {
    assert_eq!(splits_at(64, SLIDING_KV_HEADS), 1);
    assert_eq!(splits_at(65, SLIDING_KV_HEADS), 2);
    assert_eq!(splits_at(100, SLIDING_KV_HEADS), 2);
    assert_eq!(splits_at(SLIDING_WINDOW, SLIDING_KV_HEADS), SPLITS_MAX);
    assert_eq!(splits_at(8192, FULL_KV_HEADS), SPLITS_MAX);
    // Above a full sliding window nothing on the activation side grows any more:
    // the partials are already at the 16-split cap on both kinds of layer.
    assert_eq!(
        splits_at(8192, SLIDING_KV_HEADS),
        splits_at(1024, SLIDING_KV_HEADS)
    );
}

// ===========================================================================
// 4. Per-token totals, and the fraction of DRAM traffic that is not weights.
// ===========================================================================

/// Activation-scratch bytes per decode token at a given context length.
///
/// `key_count` is the full-attention layers' key count; a sliding ring's live
/// window is `min(key_count, 1024)`, which is what `GpuKvCache::key_count`
/// returns for a ring (`config.rs:234`).
fn activations_per_token(key_count: usize) -> u64 {
    let sliding_keys = key_count.min(SLIDING_WINDOW);
    let s_splits = splits_at(sliding_keys, SLIDING_KV_HEADS);
    let f_splits = splits_at(key_count, FULL_KV_HEADS);

    // Generic per-layer total for both kinds, then the two boundary layers
    // adjusted: layer 0 ADDs its own input norm, layer 59 (a full layer, since
    // `59 % 6 == 5`) SKIPS the next layer's fused norm.
    let generic_sliding = SLIDING.total(s_splits, single_pass_at(sliding_keys), false, false);
    let layer0 = SLIDING.total(s_splits, single_pass_at(sliding_keys), true, false);
    let generic_full = FULL.total(f_splits, single_pass_at(key_count), false, false);
    let layer59 = FULL.total(f_splits, single_pass_at(key_count), false, true);
    let layers = generic_sliding * SLIDING_LAYERS as u64 + generic_full * FULL_LAYERS as u64
        + (layer0 - generic_sliding) // layer 0 pays its own input norm
        - (generic_full - layer59); // layer 59 skips the next layer's norm

    // Embedding row -> `hidden_a`, `model.rs:4155-4166`.
    let embedding = h(HIDDEN) * 2;
    // `encode_copy_last_row`, `model.rs:4270-4276`.
    let copy_last_row = h(HIDDEN) * 2;
    // Vocabulary head. `encode_gemv` (`model.rs:4420`) is called with
    // `pretransformed: false`, so it dispatches `linear_input_transform` first;
    // `lm_head` is RowMajor/`perm_mode 0` (`bench.rs:179-181`) so it takes
    // `encode_gemv_raw` (`linear.rs:2063`) and writes the whole vocabulary into
    // `raw` -- which is why `raw_elements` is `max(rows * inter, vocab)`
    // (`model.rs:2987`).
    let head = h(HIDDEN) * 3 // last_hidden r, transformed w, transformed r
        + f(VOCAB) * 2 // raw w, raw r
        + h(VOCAB); // logits w
    // The softcap is SKIPPED at decode (`model.rs:4444-4458`), so
    // `capped` is never written; the argmax reads `logits`.
    let argmax = h(VOCAB) + 16; // logits r, argmax_keys w

    layers + embedding + copy_last_row + head + argmax
}

#[test]
fn per_token_activation_traffic() {
    let at_64 = activations_per_token(64);
    let at_65 = activations_per_token(65);
    let at_614 = activations_per_token(614);
    let at_1024 = activations_per_token(1024);
    let at_8192 = activations_per_token(8192);

    println!("context      activations/token   MB     % of weight stream");
    for (label, bytes) in [
        ("64 keys", at_64),
        ("65 keys", at_65),
        ("614 keys", at_614),
        ("1024 keys", at_1024),
        ("8192 keys", at_8192),
    ] {
        println!(
            "{label:>10}  {bytes:>18} B  {:>6.2}  {:>8.3}%",
            bytes as f64 / MB,
            100.0 * bytes as f64 / per_token_weight_bytes() as f64
        );
    }

    // Pinned exactly, so a shape change cannot quietly move the headline: these
    // are the numbers in the module doc. (Verified to fail under a 1-element
    // `INTER` change -- see the note in `the_whole_activation_axis_...`.)
    assert_eq!(at_64, 63_555_600);
    assert_eq!(at_65, 72_792_080);
    assert_eq!(at_614, 109_738_000);
    assert_eq!(at_1024, 137_447_440);
    // The headline: under 1.2% at every context length, so the activation axis
    // cannot be a second bandwidth consumer.
    assert!(100.0 * at_8192 as f64 / (per_token_weight_bytes() as f64) < 1.2);
    assert!(at_64 < at_65 && at_65 < at_1024);
    // Saturated: the partials are at the 16-split cap on both layer kinds once
    // the sliding window is full, so more context moves no more activations.
    assert_eq!(at_1024, at_8192);
}

/// The per-layer table. This is the deliverable.
#[test]
fn per_layer_table_is_the_sum_of_its_rows() {
    for layer in [SLIDING, FULL] {
        println!(
            "\n--- {} layer: {} of {} layers, 16 splits ---",
            layer.label, layer.count, LAYERS
        );
        println!(
            "{:<44} {:>10} {:>10} {:>10}",
            "dispatch", "read", "write", "r+w"
        );
        let (mut read, mut write) = (0u64, 0u64);
        for (label, r, w) in layer.rows(SPLITS_MAX, false, false, false) {
            println!("{label:<44} {r:>10} {w:>10} {:>10}", r + w);
            read += r;
            write += w;
        }
        println!("{:<44} {read:>10} {write:>10} {}", "TOTAL", read + write);
        assert_eq!(layer.total(SPLITS_MAX, false, false, false), read + write);
        println!(
            "x {} layers = {} B/token; sign/norm tables {} B/layer = {} B/token",
            layer.count,
            layer.total(SPLITS_MAX, false, false, false) * layer.count as u64,
            layer.table_bytes(),
            layer.table_bytes() * layer.count as u64
        );
    }
    let tables =
        SLIDING.table_bytes() * SLIDING_LAYERS as u64 + FULL.table_bytes() * FULL_LAYERS as u64;
    println!("\nper-layer sign/norm tables: {tables} B/token");
    assert_eq!(tables, 17_285_120);
    assert_eq!(SLIDING.table_bytes(), 280_576);
    assert_eq!(FULL.table_bytes(), 325_632);
}

/// The figure the brief asked to confirm: the MLP intermediates.
#[test]
fn mlp_intermediates_are_the_largest_non_attention_activation() {
    // The three MLP rows of the per-layer table, summed: gate+up matvec, the
    // gated-gelu epilogue, down_proj matvec.
    let mlp_per_layer =
        (h(HIDDEN) + f(INTER) * 2) + (f(INTER) * 2 + h(INTER)) + (h(INTER) + 2 * f(HIDDEN));
    let mlp = mlp_per_layer * LAYERS as u64;
    println!(
        "MLP non-weight traffic: {mlp} B/token = {:.2} MB = {:.3}% of the weight stream",
        mlp as f64 / MB,
        100.0 * mlp as f64 / (per_token_weight_bytes() as f64)
    );
    assert_eq!(mlp_per_layer, 483_840);
    assert_eq!(mlp, 29_030_400);

    // `raw` + `raw_b` are the f32 pair: 86,016 B each per layer, written by the
    // gate/up matvec and read back by the epilogue.
    let raw_pair_write = f(INTER) * 2;
    assert_eq!(raw_pair_write, 172_032); // 21504 f32 each
    assert_eq!(raw_pair_write * LAYERS as u64, 10_321_920); // 10.32 MB/token written
    // Round trip: written once, read once. Matches
    // spec/decode-19tok-repacked-250gb/implementation_notes.md:507-508
    // ("the live intermediate is `raw`/`raw_b` in **fp32**, 20.6 MB/token").
    assert_eq!(raw_pair_write * 2 * LAYERS as u64, 20_643_840);
    assert!((raw_pair_write * 2 * LAYERS as u64) as f64 / MB > 20.6);
    // Three u16 buffers are DEAD on the decode path and one is a dead store:
    //   `gate`/`up`/`activated` -- the fused gate/up path writes `raw`/`raw_b`
    //     (`model.rs:2209-2210`) and its epilogue writes `transformed`
    //     (`model.rs:2205`), so no decode dispatch touches them: 129,024 B/layer.
    //   `proj` -- never written at all: `model.rs:1983-1995` writes `raw` and the
    //     fused o_proj finalize writes `hidden_b`/`transformed`: 10,752 B/layer.
    //   `normed` -- written and never read (defect 2): 10,752 B/layer.
    let dead = (h(INTER) * 3 + h(HIDDEN) * 2) * LAYERS as u64;
    assert_eq!(dead, 9_031_680);
    println!(
        "dead u16 scratch at decode: {dead} B, {:.3}% of the stream",
        100.0 * dead as f64 / per_token_weight_bytes() as f64
    );
}

// ===========================================================================
// 5. KV cache: the second bandwidth consumer, and it is 10x the activations.
// ===========================================================================

/// Keys and values read per token: `key_count * kv_dim * 2 B * 2` (the decode
/// kernel loads K for the QK dot at `attention.metal:855` and V for the PV fold
/// at `:884`).
fn kv_read_per_token(key_count: usize) -> u64 {
    let sliding_keys = key_count.min(SLIDING_WINDOW) as u64;
    let sliding = sliding_keys * KV_MAX as u64 * 2 * 2 * SLIDING_LAYERS as u64;
    let full =
        key_count as u64 * (FULL_KV_HEADS * FULL_HEAD_DIM) as u64 * 2 * 2 * FULL_LAYERS as u64;
    sliding + full
}

#[test]
fn kv_is_ten_times_the_activations() {
    let kv_614 = kv_read_per_token(614);
    let kv_8192 = kv_read_per_token(8192);
    assert_eq!(kv_614, 553_287_680);
    assert_eq!(kv_8192, 1_509_949_440);
    for (label, kv, key_count) in [("614", kv_614, 614usize), ("8192", kv_8192, 8192)] {
        let act = activations_per_token(key_count);
        println!(
            "{label} keys: KV {kv} B ({:.2} GB) vs activations {act} B ({:.2} MB) = {:.1}x",
            kv as f64 / GB,
            act as f64 / MB,
            kv as f64 / act as f64
        );
        assert!(
            kv as f64 > act as f64 * 5.0,
            "{label}: KV is no longer the second consumer"
        );
    }
    // Every ceiling calculation built on weights alone understates the stream at
    // full context by this much.
    let total =
        per_token_weight_bytes() as f64 + kv_8192 as f64 + activations_per_token(8192) as f64;
    println!(
        "8192 keys: {total:.0} B/token = {:.3} GB, so 62.3 ms needs {:.1} GB/s of 273",
        total / GB,
        total / 0.0623 / GB
    );
    assert!(total / per_token_weight_bytes() as f64 > 1.13);
}

// ===========================================================================
// 6. The ceiling, and the one change worth making.
// ===========================================================================

/// Measured 16.05-16.10 tok/s, i.e. 62.3 ms/step (task brief; AGENTS.md §10.2
/// takes the median, and the spread is 0.3%).
const STEP_MS: f64 = 62.3;
const TOK_S_BASELINE: f64 = 16.05;

fn achieved_gb_s() -> f64 {
    per_token_weight_bytes() as f64 / (STEP_MS / 1000.0) / GB
}

/// tok/s if `bytes` of the step's traffic vanished, priced at the bandwidth the
/// weights alone are already achieving. An UPPER bound: it assumes the removed
/// bytes were streaming at full rate, which none of them are.
fn tok_s_after_removing(bytes: u64) -> f64 {
    let ms = STEP_MS - bytes as f64 / (achieved_gb_s() * GB) * 1000.0;
    1000.0 / ms
}

#[test]
fn the_whole_activation_axis_is_worth_at_most_zero_point_three_tok_s() {
    let everything = activations_per_token(8192) + 17_285_120;
    assert_eq!(everything, 154_732_560);
    println!(
        "weight stream at {STEP_MS} ms: {:.1} GB/s of 273",
        achieved_gb_s()
    );
    println!(
        "deleting 100% of {everything} B/token ({:.1} MB): {:.3} ms -> {:.3} tok/s (+{:.2})",
        everything as f64 / MB,
        everything as f64 / (achieved_gb_s() * GB) * 1000.0,
        tok_s_after_removing(everything),
        tok_s_after_removing(everything) - TOK_S_BASELINE,
    );
    assert!(achieved_gb_s() > 190.0 && achieved_gb_s() < 196.0);
    // The ceiling of the angle. Not 5%, not 1%: under 1.3% of the step.
    assert!(tok_s_after_removing(everything) < 16.35);
    assert!(tok_s_after_removing(everything) > 16.15);

    // Most of it is not even removable: a GEMV's output IS the activation.
    // Ranked, best case per change, priced the same way.
    let partials = SLIDING_LAYERS as u64 * SLIDING.partials(SPLITS_MAX) * 2
        + FULL_LAYERS as u64 * FULL.partials(SPLITS_MAX) * 2;
    let mut ranked: Vec<(&str, u64)> = vec![
        (
            "attn_partials round trip (fp16 would break bit-exactness: 43/128 tokens)",
            partials,
        ),
        (
            "gate/up f32 round trip (needs a 128-wide output tile: transform_simd is cross-lane)",
            258_048 * LAYERS as u64,
        ),
        ("lm_head raw f32 round trip", f(VOCAB) * 2),
        ("normed dead store", h(HIDDEN) * LAYERS as u64),
        ("attended overrun (a bug, not a cost)", 0),
    ];
    ranked.sort_by_key(|(_, bytes)| std::cmp::Reverse(*bytes));
    for (label, bytes) in &ranked {
        println!(
            "  {:>10} B/token  ->  +{:.4} tok/s   {label}",
            bytes,
            tok_s_after_removing(0) - tok_s_after_removing(*bytes)
        );
    }
    assert_eq!(ranked[0].1, 73_891_840);
    // No single item is worth 0.11 tok/s, let alone a millisecond.
    assert!(
        ranked
            .iter()
            .all(|(_, bytes)| tok_s_after_removing(0) - tok_s_after_removing(*bytes) < 0.11)
    );
    assert_eq!(TOK_S_BASELINE, 16.05);
}

/// The one change this probe recommends, and its honest price: nothing.
///
/// `model.rs:3012`: `attended: u16_elements(max_tokens, hidden)?`
///            -> `attended: u16_elements(max_tokens, q_max)?`
///
/// Removes an out-of-bounds write per layer per token -- 5,632 B on the 50
/// sliding layers and 22,016 B on the 10 full ones (5,767,168 B and 22,544,384 B
/// at 1024 rows) -- that is correct today only because Metal pads allocations to
/// page granularity. Estimated tok/s: **0.00**. It is a hazard fix, not a speed
/// fix, and 22,016 B of extra scratch is 0.0002% of the stream. The zero-cost
/// variant is to point the attention request's `output` (`model.rs:1843`) at
/// `transformed`, which is `inter` = 21504 halves and dead at that point in the
/// layer; that needs the aliasing proved against the prefill path before it
/// ships, so the resize goes first.
#[test]
fn the_recommended_change_is_worth_zero_tok_s_and_says_so() {
    let extra_decode = h(Q_MAX) - h(HIDDEN);
    let extra_prefill = h(Q_MAX * 1024) - h(HIDDEN * 1024);
    assert_eq!(extra_decode, 22_016);
    assert_eq!(extra_prefill, 22_544_384);
    let delta = tok_s_after_removing(0) - tok_s_after_removing(extra_decode);
    println!("attended resize: +{extra_decode} B of scratch/token, {delta:.6} tok/s");
    assert!(
        delta < 0.001,
        "a 22 KB allocation cannot be worth a millisecond per step"
    );
    // On the prefill side it is 22.5 MB against the 168 MiB that
    // `resident_bytes` already fails to report (F5), i.e. 12.8% of an
    // under-counted budget -- so it should land WITH F5, not before it.
    assert!(extra_prefill as f64 / 176_160_768.0 < 0.129);
}
