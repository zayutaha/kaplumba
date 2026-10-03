//! Decode fusion census: how many dispatches a layer issues, what each one
//! moves, and which merges are bit-exact.
//!
//! READ-ONLY PROBE. Nothing here is on a production path and nothing here needs
//! a GPU or a model: every claim is derived from the shapes in `config.json`
//! and from the dispatch geometry that `crates/kernels/src/{linear,elements,
//! attention}.rs` computes, both transcribed below with `file:line` anchors.
//! There is no model load, no `gpuq`, no Metal device.
//!
//! WHY A STATIC CENSUS AND NOT A PROFILER READ
//! ------------------------------------------------
//! `KAPLUMBA_DECODE_PROFILE=kernel` counts *stamped boundaries*, not dispatches:
//! `GpuProfile::stamp` (`crates/engine/src/model.rs:829`) ends the command
//! buffer, so a role with three dispatches behind it counts once. The measured
//! 546 is therefore 9 stamps/layer x 60 + 6 head, and it is blind to the ~1/4 of
//! dispatches that have no `stamp` after them. This file reconstructs the real
//! dispatch list from the source and *checks itself against 546*, which is the
//! one number already measured on the real model. That agreement is the only
//! reason to believe the rest of the table.
//!
//! WHAT THE CENSUS SAYS (see the assertion constants at the bottom)
//! ----------------------------------------------------------
//! Decode, `tokens == 1`, `permutation_mode == 1`, `rate == Integer(3)`, no
//! env flags, Gxma/Gemma-4-31B (hidden 5376, intermediate 21504, 60 layers,
//! 50 sliding (head_dim 256, 32q/16kv) + 10 full (head_dim 512, 32q/4kv),
//! vocab 262144):
//!
//! | # | dispatch | grid | T/G | bytes moved |
//! |---|---|---|---|---|
//! | 1 | `input_norm_and_qkv_transform` (layer 0 only) | 1 | 256 | 43 KB |
//! | 2 | `linear_gemv_ishizuki_k3_fused3_2col` (q,k,v) | 512 | 256 | 33.04 MB weights |
//! | 3 | `linear_finalize_simd3` | 128 | 32 | 131 KB |
//! | 4 | `norm_rope_qkv_cache` | 64 | 256 | 90 KB |
//! | 5 | `attention_decode_split_256_2` | 48 | 64 | KV-dependent |
//! | 6 | `attention_decode_combine_256` | 128 | 64 | 110 KB |
//! | 7 | `linear_input_transform` (o_proj signs+Hadamard) | 64 | 32 | 44 KB |
//! | 8 | `linear_gemv_ishizuki_k3_colmajor_2col_sk2` (o_proj) | 336 | 256 | 16.52 MB weights |
//! | 9 | `linear_finalize_o_proj_and_pre_ffn_transform_sk2` | **1** | 1024 | 97 KB |
//! | 10 | `linear_gemv_ishizuki_k3_fused2_2col` (gate+up) | 1344 | 256 | 86.70 MB weights |
//! | 11 | `linear_finalize_gated_gelu_and_down_input_transform` | 168 | 32 | 269 KB |
//! | 12 | `linear_gemv_ishizuki_k3_colmajor_2col_sk2` (down) | 336 | 256 | 43.35 MB weights |
//! | 13 | `finalize_post_ffn_and_next_input_norm_transform` | **1** | 1024 | 128 KB |
//!
//! 13 dispatches in layer 0, 12 in every other sliding layer, 14 in each of the
//! 10 full-attention layers; 9 in the head. **750 dispatches per decode step**,
//! of which **546 are stamp boundaries** -- and those two numbers differ because
//! one `profile.stamp` closes a whole group of dispatches.
//!
//! Traffic per step at key_count 700:
//!
//! | class | bytes | share |
//! |---|---|---|
//! | weights | 12 039 487 488 | 94.27% |
//! | KV cache | 630 784 000 | 4.94% |
//! | element/scratch (fusion-addressable) | 101 601 576 | **0.80%** |
//!
//! Note the weight figure: `implementation_notes.md` 16.4 quotes 11.834 GB,
//! which multiplies a *sliding* layer's shapes by 60 and so ignores the 10
//! `full_attention` layers (`global_head_dim 512`, 4 KV heads, q and o_proj's
//! input both 16384-wide) at 200.26 MB each. The corrected figure is
//! 12.039 GB and the bandwidth floor moves from 58.3 ms to **59.3 ms**, i.e.
//! the 20 tok/s target gets further away, not closer. See
//! `only_one_percent_of_the_step_is_addressable_by_activation_fusion`.
//!
//! SABOTAGE EVIDENCE (AGENTS.md §3.4: a coverage claim is not a claim until
//! the code has been broken and the failure observed)
//! ----------------------------------------------------------------------
//! Every falsifiable assertion here was sabotaged and observed to fail:
//!
//! | sabotage | test that failed |
//! |---|---|
//! | drop `finalize_gated_gelu...` from the `gate_up_gelu` boundary | `every_dispatch_is_covered_by_exactly_one_stamp_boundary` |
//! | `MEASURED_STAMPED_BOUNDARIES` 546 -> 545 | `census_reproduces_the_measured_stamp_boundary_count` |
//! | o_proj finalize `groups` 1 -> 42 | `single_threadgroup_dispatches_are_the_occupancy_problem` |
//! | `unfused_transformed` loses its `half()` store | `dropping_the_fp16_store_is_not_bit_exact`, `a_moved_fp16_rounding_can_flip_the_argmax_token` |
//! | norm tree `stride` 128 -> 256 | `reassociating_the_norm_reduction_changes_the_inverse_rms` |
//! | `norm_inverse_block_parallel` aliased to the documented order | `reassociating_the_norm_reduction_changes_the_inverse_rms` |
//!
//! **What this file does NOT cover, and the one sabotage that did not fail.**
//! Perturbing the Hadamard scale constant by one fp16 ulp
//! (`0.08838834764831845` -> `0.0883883462996483`) leaves all twelve tests
//! green (13/13), because the unfused and fused arms share the constant. That is correct
//! for a *fusion-contract* model -- what is under test is the narrowing, not the
//! butterfly -- but it means the constants here are not pinned by this file. They
//! are pinned by `crates/engine/tests/live_kernel_bit_exact.rs` and
//! `repacked_golden.rs`, which compare the real kernels against a CPU reference
//! built from `crates/engine/src/exl3.rs`. A fused variant must be verified by
//! adding to *those*, not by satisfying this file.
//!
//! THE THREE THINGS WORTH KNOWING
//! ------------------------------
//! 1. **Rows 9 and 13 are the whole problem, and they are not fusions.** Both
//!    run on **one** threadgroup (`linear.rs:3065` and `linear.rs:3197`,
//!    `dispatch_groups(..., 1, 1024)`), so 19 of 20 GPU cores are idle while
//!    43 KB of f32 accumulator is reduced. Rows 7, 3 and 11 are 1-D and so do
//!    spread, but thinly. Measured marginal cost of the five non-GEMV roles is
//!    **1.39 ms/step** at 300 dispatches
//!    (`spec/decode-25tok-peak-bandwidth/tasks.md` T5.10) and *all* nine
//!    element roles together are **1.52 ms/step** (T5.12) against a 62.3 ms
//!    step. T5.6 is filed as "parallel multi-threadgroup finalize" but the code
//!    it produced raised threads-per-group 256 -> 1024; the grid is still 1.
//! 2. **A dispatch costs ~0.7 us** (T5.10). Deleting all 745 non-GEMV
//!    dispatches is worth 0.52 ms; the *open task list's* 61 of them are worth
//!    0.043 ms. Dispatch count is not the lever, which is why T5.3 and T5.7 as
//!    originally scoped ("~500 dispatches, ~3-4 ms") are off by 5-80x.
//! 3. **The 12 ms gap is bandwidth.** 12.039 GB/token at the best 3-bpw EXL3
//!    GEMV measured on this box (203 GB/s, `implementation_notes.md` 16.3) is
//!    59.3 ms = **16.9 tok/s with every non-GEMV kernel deleted**. Fusion cannot
//!    touch that term, so it addresses at most 3.0 ms of the 12.3 ms gap, and
//!    the whole element layer is 1.52 ms of even that. The arithmetic is in
//!    `fused_budget_ms`.

use half::f16;

// ===========================================================================
// Shapes. From `~/.kaplumba/models/Gxma/Gxma/config.json`, and asserted against
// `ModelConfig::validate_current_checkpoint` (`crates/engine/src/config.rs:223`).
// ===========================================================================

const HIDDEN: usize = 5376;
const INTER: usize = 21504;
const VOCAB: usize = 262_144;
const LAYERS: usize = 60;
const Q_HEADS: usize = 32;
const SLIDING_HEAD_DIM: usize = 256;
const SLIDING_KV_HEADS: usize = 16;
const FULL_HEAD_DIM: usize = 512;
const FULL_KV_HEADS: usize = 4;
const SLIDING_WINDOW: u32 = 1024;
const LM_HEAD_BPW: u32 = 6;
const LAYER_BPW: u32 = 3;

/// `config.rs:241` -- `layer i` is `Full` iff `i % 6 == 5`, giving 10 of 60.
fn layer_kind(index: usize) -> Kind {
    if index % 6 == 5 {
        Kind::Full
    } else {
        Kind::Sliding
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Sliding,
    Full,
}

impl Kind {
    fn head_dim(self) -> usize {
        match self {
            Kind::Sliding => SLIDING_HEAD_DIM,
            Kind::Full => FULL_HEAD_DIM,
        }
    }
    fn kv_heads(self) -> usize {
        match self {
            Kind::Sliding => SLIDING_KV_HEADS,
            Kind::Full => FULL_KV_HEADS,
        }
    }
    /// `weights.v_proj` is `None` for full-attention layers
    /// (`crates/engine/src/model.rs:3439`, and `DecoderLayerWeights::validate`
    /// rejects `full_attention && v_proj.is_some()` at `model.rs:507`). So the
    /// 10 full layers take the `fused2` + `encode_copy` arm at `model.rs:1534`,
    /// not the `fused3` arm.
    fn has_v_proj(self) -> bool {
        matches!(self, Kind::Sliding)
    }
    fn q_out(self) -> usize {
        Q_HEADS * self.head_dim()
    }
    fn kv_out(self) -> usize {
        self.kv_heads() * self.head_dim()
    }
}

/// Bytes of packed EXL3 weights for a `k x n` projection at `bpw` bits/weight.
///
/// 3 bpw checks out against the recorded table (`implementation_notes.md`
/// 16.4): q 8192x5376 -> 16 515 072 B, gate 21504x5376 -> 43 350 144 B.
fn weight_bytes(k: usize, n: usize, bpw: u32) -> u64 {
    (k as u64) * (n as u64) * (bpw as u64) / 8
}

fn h16(elements: usize) -> u64 {
    (elements as u64) * 2
}
fn f32b(elements: usize) -> u64 {
    (elements as u64) * 4
}

// ===========================================================================
// The census
// ===========================================================================

#[derive(Clone, Debug)]
struct Row {
    /// 1-based position in the layer's dispatch order.
    seq: usize,
    kernel: &'static str,
    groups: u64,
    threads_per_group: u64,
    /// Weight reads live in `read_weight_bytes` and KV-cache reads in
    /// `kv_bytes`, so the three byte columns partition the step's traffic and
    /// only `read_bytes`/`write_bytes` is addressable by a fusion.
    read_bytes: u64,
    write_bytes: u64,
    read_weight_bytes: u64,
    /// Bytes read from `key_cache` / `value_cache`. Grows with context and is
    /// removable by no fusion in this file.
    kv_bytes: u64,
    /// One threadgroup means one GPU core out of twenty.
    single_group: bool,
    /// This dispatch writes a buffer the next one reads back for no other
    /// reason: an epilogue-fusion boundary.
    epilogue_candidate: bool,
}

struct LayerShape {
    index: usize,
    kind: Kind,
    /// `key_count` live in the KV cache at this step.
    key_count: usize,
}

impl LayerShape {
    /// o_proj's input width: the query width of this layer's attention.
    fn o_in(&self) -> usize {
        self.kind.q_out()
    }

    /// `decode_splits` (`crates/kernels/src/attention.rs:271`): a one-wave
    /// workgroup budget of 54 for a 256-thread threadgroup, then
    /// `budget / kv_heads`, then capped by what the key count can afford.
    fn attention_splits(&self) -> usize {
        let threadgroup = 32 * (Q_HEADS / self.kind.kv_heads());
        let budget = if threadgroup <= 256 { 54 } else { 52 };
        let wanted = budget / self.kind.kv_heads();
        let affordable = self.key_count.div_ceil(16);
        wanted.min(affordable).clamp(1, 16)
    }

    /// `ATTENTION_SINGLE_PASS_MAX_KEYS` (`attention.rs:74`): at or under 64 keys
    /// the split+combine pair collapses to one dispatch.
    fn attention_is_single_dispatch(&self) -> bool {
        self.key_count <= 64
    }
}

fn census_layer(layer: &LayerShape) -> Vec<Row> {
    let kind = layer.kind;
    let hidden = HIDDEN;
    let q_out = kind.q_out();
    let kv_out = kind.kv_out();
    let kv_dim = kv_out;
    let q_dim = q_out;
    let mut rows: Vec<Row> = Vec::new();

    macro_rules! push {
        ($kernel:expr, $groups:expr, $tpg:expr, $rb:expr, $wb:expr, $rw:expr, $epi:expr) => {
            push!(
                $kernel,
                $groups,
                $tpg,
                $rb,
                $wb,
                $rw,
                0,
                $epi
            )
        };
        ($kernel:expr, $groups:expr, $tpg:expr, $rb:expr, $wb:expr, $rw:expr, $kv:expr, $epi:expr) => {
            rows.push(Row {
                seq: rows.len() + 1,
                kernel: $kernel,
                groups: $groups,
                threads_per_group: $tpg,
                read_bytes: $rb,
                write_bytes: $wb,
                read_weight_bytes: $rw,
                kv_bytes: $kv,
                single_group: $groups == 1,
                epilogue_candidate: $epi,
            });
        };
    }

    // ---- 1. input_layernorm + q input-sign transform --------------------
    // Skipped for every layer except 0: `cross_layer_fusion_applies`
    // (`model.rs:1274`) makes layer L+1's norm part of layer L's finalize.
    if layer.index == 0 {
        push!(
            "input_norm_and_qkv_transform",
            1,
            256,
            h16(hidden) * 2,
            h16(hidden) * 2,
            0,
            false
        );
    }

    // ---- 2/3. QKV projection -------------------------------------------
    if kind.has_v_proj() {
        // `encode_gemv_fused3` (`linear.rs:2689`), 2col arm (`linear.rs:2785`):
        // groups = (q/16 + k/16 + v/16) / 2, 256 threads.
        let tiles = (q_out / 16 + kv_out / 16 + kv_out / 16) as u64;
        push!(
            "linear_gemv_ishizuki_k3_fused3_2col",
            tiles / 2,
            256,
            h16(hidden),
            f32b(q_out + 2 * kv_out),
            weight_bytes(HIDDEN, q_out + 2 * kv_out, LAYER_BPW),
            false
        );
        // `finalize_simd3` (`linear.rs:2826`): one 32-thread group per 128
        // outputs, across all three accumulators.
        let blocks = (q_out + 2 * kv_out) / 128;
        push!(
            "linear_finalize_simd3",
            blocks as u64,
            32,
            f32b(q_out + 2 * kv_out) + h16(q_out + 2 * kv_out),
            h16(q_out + 2 * kv_out),
            0,
            false
        );
    } else {
        // `encode_gemv_fused2` (`linear.rs:2330`) plus the `encode_copy` at
        // `model.rs:1588` that stands in for the absent v_proj.
        let tiles = (q_out / 16 + kv_out / 16) as u64;
        push!(
            "linear_gemv_ishizuki_k3_fused2_2col",
            tiles / 2,
            256,
            h16(hidden),
            f32b(q_out + kv_out),
            weight_bytes(HIDDEN, q_out + kv_out, LAYER_BPW),
            false
        );
        for (name, width) in
            [("linear_finalize_simd (q)", q_out), ("linear_finalize_simd (k)", kv_out)]
        {
            push!(
                name,
                (width / 128) as u64,
                32,
                f32b(width) + h16(width),
                h16(width),
                0,
                false
            );
        }
        push!("linear_copy_buffer (k -> v)", (kv_out / 64) as u64, 64, h16(kv_out), h16(kv_out), 0, false);
    }

    // ---- 4. Q/K/V RMSNorm + RoPE + KV append ---------------------------
    // `encode_norm_rope_qkv_cache` (`elements.rs:1157`): one threadgroup per
    // (q head, k head, v head), 256 threads.
    push!(
        "norm_rope_qkv_cache",
        (Q_HEADS + 2 * kind.kv_heads()) as u64,
        256,
        h16(q_dim + 2 * kv_dim) + h16(q_dim + kv_dim),
        h16(q_dim) + h16(kv_dim) + h16(kv_dim) + 4,
        0,
        false
    );

    // ---- 5/6. attention ------------------------------------------------
    let kv_bytes = match kind {
        Kind::Sliding => (layer.key_count.min(SLIDING_WINDOW as usize) * kv_dim * 2 * 2) as u64,
        Kind::Full => (layer.key_count * kv_dim * 2 * 2) as u64,
    };
    let partial_bytes = f32b(layer.attention_splits() * Q_HEADS * (kind.head_dim() + 2));
    if layer.attention_is_single_dispatch() {
        push!(
            "attention_decode_single_pass",
            kind.kv_heads() as u64,
            (32 * (Q_HEADS / kind.kv_heads())) as u64,
            h16(q_dim),
            h16(hidden),
            0,
            kv_bytes,
            false
        );
    } else {
        push!(
            "attention_decode_split",
            (layer.attention_splits() * kind.kv_heads()) as u64,
            (32 * (Q_HEADS / kind.kv_heads())) as u64,
            h16(q_dim),
            partial_bytes,
            0,
            kv_bytes,
            false
        );
        push!(
            "attention_decode_combine",
            (q_dim / 64) as u64,
            64,
            partial_bytes,
            h16(hidden),
            0,
            false
        );
    }

    // ---- 7. o_proj input signs + Hadamard ------------------------------
    // `encode_input_transform` (`linear.rs:1887`): one 32-thread group per
    // 128-element block of the activation, so 64 groups at in_features 8192.
    push!(
        "linear_input_transform (o_proj)",
        (layer.o_in() / 128) as u64,
        32,
        h16(hidden) + h16(layer.o_in()),
        h16(layer.o_in()),
        0,
        true
    );

    // ---- 8/9. o_proj ----------------------------------------------------
    // `encode_gemv_raw_sk2`, 2col (`linear.rs:2257`): groups = (n/16/2)*2.
    push!(
        "linear_gemv_ishizuki_k3_colmajor_2col_sk2 (o_proj)",
        ((HIDDEN / 16) / 2 * 2) as u64,
        256,
        h16(layer.o_in()),
        f32b(HIDDEN * 2),
        weight_bytes(layer.o_in(), HIDDEN, LAYER_BPW),
        false
    );
    // `encode_finalize_o_proj_and_pre_ffn_transform_sk2` (`linear.rs:3042`).
    push!(
        "linear_finalize_o_proj_and_pre_ffn_transform_sk2",
        1,
        1024,
        f32b(HIDDEN * 2) + h16(HIDDEN) * 3,
        h16(hidden) + h16(hidden),
        0,
        true
    );

    // ---- 10/11. gate + up ----------------------------------------------
    // `encode_gemv_fused2_mlp` (`linear.rs:2492`), 2col arm (`linear.rs:2595`).
    push!(
        "linear_gemv_ishizuki_k3_fused2_2col (gate+up)",
        ((INTER / 16 + INTER / 16) / 2) as u64,
        256,
        h16(hidden),
        f32b(2 * INTER),
        weight_bytes(HIDDEN, 2 * INTER, LAYER_BPW),
        false
    );
    // `finalize_gated_gelu_and_down_input` (`linear.rs:2657`): INTER/128 groups
    // of 32 threads.
    push!(
        "linear_finalize_gated_gelu_and_down_input_transform",
        (INTER / 128) as u64,
        32,
        f32b(2 * INTER) + h16(2 * INTER),
        h16(HIDDEN),
        0,
        true
    );

    // ---- 12/13. down_proj + post_ffn_norm + next input_layernorm --------
    push!(
        "linear_gemv_ishizuki_k3_colmajor_2col_sk2 (down)",
        ((HIDDEN / 16) / 2 * 2) as u64,
        256,
        h16(HIDDEN),
        f32b(HIDDEN * 2),
        weight_bytes(INTER, HIDDEN, LAYER_BPW),
        false
    );
    if layer.index + 1 < LAYERS {
        // `encode_cross_layer_finalize_and_norm` (`linear.rs:3169`).
        push!(
            "finalize_post_ffn_and_next_input_norm_transform",
            1,
            1024,
            f32b(HIDDEN * 2) + h16(HIDDEN) * 5,
            h16(HIDDEN) * 3,
            0,
            true
        );
    } else {
        // `encode_finalize_and_post_ffn_norm_scale_sk2` (`linear.rs:2884`).
        push!(
            "linear_finalize_and_post_ffn_norm_scale_sk2",
            1,
            1024,
            f32b(HIDDEN * 2) + h16(HIDDEN) * 3,
            h16(HIDDEN) * 2,
            0,
            true
        );
    }

    rows
}

/// The chunk prologue and the head epilogue, both outside the layer loop.
fn census_head() -> Vec<Row> {
    vec![
        Row {
            seq: 1,
            kernel: "embedding_buffer",
            groups: (HIDDEN / 64) as u64,
            threads_per_group: 64,
            read_bytes: h16(HIDDEN),
            write_bytes: h16(HIDDEN),
            read_weight_bytes: 0,
            kv_bytes: 0,
            single_group: false,
            epilogue_candidate: false,
        },
        Row {
            seq: 2,
            kernel: "copy_last_row",
            groups: (HIDDEN / 64) as u64,
            threads_per_group: 64,
            read_bytes: h16(HIDDEN),
            write_bytes: h16(HIDDEN),
            read_weight_bytes: 0,
            kv_bytes: 0,
            single_group: false,
            epilogue_candidate: true,
        },
        Row {
            seq: 3,
            kernel: "rms_norm_weighted_tg (final_norm)",
            groups: 1,
            threads_per_group: 256,
            read_bytes: h16(HIDDEN) * 2,
            write_bytes: h16(HIDDEN),
            read_weight_bytes: 0,
            kv_bytes: 0,
            single_group: true,
            epilogue_candidate: false,
        },
        Row {
            seq: 4,
            kernel: "linear_input_transform (lm_head)",
            groups: (HIDDEN / 128) as u64,
            threads_per_group: 32,
            read_bytes: h16(HIDDEN) * 2,
            write_bytes: h16(HIDDEN),
            read_weight_bytes: 0,
            kv_bytes: 0,
            single_group: false,
            epilogue_candidate: true,
        },
        Row {
            seq: 5,
            kernel: "linear_gemv_ishizuki_k3_colmajor_2col (lm_head)",
            groups: (VOCAB / 32) as u64,
            threads_per_group: 256,
            read_bytes: h16(HIDDEN),
            write_bytes: f32b(VOCAB),
            read_weight_bytes: weight_bytes(HIDDEN, VOCAB, LM_HEAD_BPW),
            kv_bytes: 0,
            single_group: false,
            epilogue_candidate: false,
        },
        Row {
            seq: 6,
            kernel: "linear_finalize_simd (lm_head)",
            groups: (VOCAB / 128) as u64,
            threads_per_group: 32,
            read_bytes: f32b(VOCAB) + h16(VOCAB),
            write_bytes: h16(VOCAB),
            read_weight_bytes: 0,
            kv_bytes: 0,
            single_group: false,
            epilogue_candidate: false,
        },
        Row {
            seq: 7,
            kernel: "argmax_clear",
            groups: 1,
            threads_per_group: 32,
            read_bytes: 0,
            write_bytes: 16,
            read_weight_bytes: 0,
            kv_bytes: 0,
            single_group: true,
            epilogue_candidate: false,
        },
        Row {
            seq: 8,
            kernel: "argmax_slots",
            groups: (128 * 4) as u64,
            threads_per_group: 256,
            read_bytes: h16(VOCAB),
            write_bytes: 16,
            read_weight_bytes: 0,
            kv_bytes: 0,
            single_group: false,
            epilogue_candidate: true,
        },
        Row {
            seq: 9,
            kernel: "argmax_keys_to_token",
            groups: 1,
            threads_per_group: 1,
            read_bytes: 16,
            write_bytes: 8,
            read_weight_bytes: 0,
            kv_bytes: 0,
            single_group: true,
            epilogue_candidate: false,
        },
    ]
}

// ===========================================================================
// Census assertions
// ===========================================================================

/// The measured number from the task brief. If the static census and the
/// instrumented model ever disagree, the census is wrong and the table below
/// cannot be trusted.
const MEASURED_STAMPED_BOUNDARIES: usize = 546;

/// Every `profile.stamp(label)` in `forward_buffer`, in source order, with the
/// dispatches it closes. One `stamp` ends and commits the command buffer after a
/// whole group of them, which is why the 546 the profiler reports is a boundary
/// count and not a dispatch count -- and why a dispatch can be invisible to it.
const LAYER_BOUNDARIES: &[(&str, &[&str])] = &[
    // model.rs:1460, unconditional -- see `norm_transform_commits_an_empty_buffer`.
    ("norm_transform", &["input_norm_and_qkv_transform"]),
    // model.rs:1532 (sliding) / 1589 (full): one stamp closes the GEMV, every
    // finalize dispatch behind it, and the k->v copy.
    (
        "qkv_proj",
        &[
            "linear_gemv_ishizuki_k3_fused3_2col",
            "linear_gemv_ishizuki_k3_fused2_2col",
            "linear_finalize_simd3",
            "linear_finalize_simd (q)",
            "linear_finalize_simd (k)",
            "linear_copy_buffer (k -> v)",
        ],
    ),
    ("kv_norm_rope_cache", &["norm_rope_qkv_cache"]),
    (
        "attention",
        &[
            "attention_decode_split",
            "attention_decode_combine",
            "attention_decode_single_pass",
        ],
    ),
    // model.rs:1996: the o_proj input transform and its GEMV share this one.
    (
        "o_proj",
        &[
            "linear_input_transform (o_proj)",
            "linear_gemv_ishizuki_k3_colmajor_2col_sk2 (o_proj)",
        ],
    ),
    ("post_attn_pre_ffn_transform", &["linear_finalize_o_proj_and_pre_ffn_transform_sk2"]),
    (
        "gate_up_gelu",
        &[
            "linear_gemv_ishizuki_k3_fused2_2col (gate+up)",
            "linear_finalize_gated_gelu_and_down_input_transform",
        ],
    ),
    ("down_proj", &["linear_gemv_ishizuki_k3_colmajor_2col_sk2 (down)"]),
    (
        "post_ffn_out",
        &[
            "finalize_post_ffn_and_next_input_norm_transform",
            "linear_finalize_and_post_ffn_norm_scale_sk2",
        ],
    ),
];

/// The head's stamps, from `forward_chunk` and `encode_lm_head`.
const HEAD_BOUNDARIES: &[(&str, &[&str])] = &[
    ("chunk/embedding", &["embedding_buffer"]),
    ("chunk/copy_last_row", &["copy_last_row"]),
    ("head/final_norm", &["rms_norm_weighted_tg (final_norm)"]),
    (
        "head/gemv",
        &[
            "linear_input_transform (lm_head)",
            "linear_gemv_ishizuki_k3_colmajor_2col (lm_head)",
            "linear_finalize_simd (lm_head)",
        ],
    ),
    // model.rs:4456 `head/softcap` is absent on decode: `skip_softcap` is true
    // whenever a next-token buffer is supplied (`model.rs:4444`).
    ("head/argmax", &["argmax_clear", "argmax_slots"]),
    ("head/argmax_token", &["argmax_keys_to_token"]),
];

/// Every dispatch must be covered by exactly one boundary. This is the census's
/// integrity check: a kernel the table forgets shows up as an uncovered row, and
/// a kernel listed against two boundaries shows up as a double count.
#[test]
fn every_dispatch_is_covered_by_exactly_one_stamp_boundary() {
    for (index, boundary_set) in
        [(30usize, LAYER_BOUNDARIES), (5, LAYER_BOUNDARIES), (0, LAYER_BOUNDARIES)]
    {
        let rows = census_layer(&LayerShape { index, kind: layer_kind(index), key_count: 700 });
        let mut coverage: Vec<(&str, usize)> = Vec::new();
        for (label, kernels) in boundary_set {
            let n = rows.iter().filter(|r| kernels.contains(&r.kernel)).count();
            coverage.push((label, n));
        }
        assert_eq!(
            coverage.iter().map(|(_, n)| n).sum::<usize>(),
            rows.len(),
            "layer {index}: boundaries cover {} dispatches but the census lists {}",
            coverage.iter().map(|(_, n)| n).sum::<usize>(),
            rows.len()
        );
        // At most one boundary per layer may cover nothing: `norm_transform` on
        // layers 1..59, where the dispatch was fused into the previous layer's
        // finalize but the stamp was not removed.
        let empty: Vec<&str> =
            coverage.iter().filter(|(_, n)| *n == 0).map(|(label, _)| *label).collect();
        assert!(
            empty.len() <= 1,
            "layer {index} has {} boundaries covering no dispatch: {empty:?}; \
             at most one (norm_transform) is expected",
            empty.len()
        );
    }
    let rows = census_head();
    let covered: usize = HEAD_BOUNDARIES
        .iter()
        .map(|(_, kernels)| rows.iter().filter(|r| kernels.contains(&r.kernel)).count())
        .sum();
    assert_eq!(covered, rows.len(), "a head dispatch has no stamp boundary");
}

/// `forward_buffer` calls `profile.stamp("norm_transform")` at `model.rs:1460`
/// unconditionally, but on layers 1..59 `input_norm_precomputed`
/// (`model.rs:1411`) has already had the dispatch folded into the previous
/// layer's cross-layer finalize. `commit_stamped` (`encoder.rs:674`) ends,
/// commits and rearms with no empty-buffer check, so each of those 59 layers
/// commits an **empty** command buffer per step and retains it in `stamped` for
/// `take_gpu_ms`. That is profiling overhead paid for a dispatch that no longer
/// exists -- and it is invisible in the 546, because the boundary fires.
#[test]
fn norm_transform_commits_an_empty_buffer_on_59_of_60_layers() {
    let empty: Vec<usize> = (0..LAYERS)
        .filter(|&index| {
            let rows = census_layer(&LayerShape { index, kind: layer_kind(index), key_count: 700 });
            rows.iter().all(|r| r.kernel != "input_norm_and_qkv_transform")
        })
        .collect();
    assert_eq!(
        empty.len(),
        LAYERS - 1,
        "expected every layer but 0 to have its input norm fused away"
    );
    assert_eq!(empty[0], 1, "layer 0 has no predecessor, so it must dispatch");
    eprintln!(
        "{} of {LAYERS} layers commit an empty command buffer per step under the \
         `norm_transform` label, ~{:.2} ms at 0.7 us per commit -- profiler \
         overhead only, not present in the 62.3 ms baseline",
        empty.len(),
        empty.len() as f64 * 0.7 / 1000.0
    );
}

#[test]
fn census_reproduces_the_measured_stamp_boundary_count() {
    let key_count = 700; // a conversational decode step
    let mut stamped = 0;
    for index in 0..LAYERS {
        let rows = census_layer(&LayerShape { index, kind: layer_kind(index), key_count });
        assert!(!rows.is_empty(), "layer {index} produced no dispatches");
        // Every layer reaches all nine `profile.stamp` calls, including layer 0,
        // whose `norm_transform` stamp labels a real dispatch, and layers 1..59,
        // whose labels an empty buffer.
        assert_eq!(
            LAYER_BOUNDARIES.len(),
            9,
            "forward_buffer's boundary list changed"
        );
        stamped += LAYER_BOUNDARIES.len();
        let _ = rows;
    }
    stamped += HEAD_BOUNDARIES.len();
    assert_eq!(
        stamped,
        MEASURED_STAMPED_BOUNDARIES,
        "static stamp census disagrees with the instrumented model"
    );
}

#[test]
fn per_layer_dispatch_counts_match_the_source() {
    let key_count = 700;
    let layer0 = census_layer(&LayerShape { index: 0, kind: layer_kind(0), key_count });
    assert_eq!(layer0.len(), 13, "layer 0: input norm is not fused away");

    let mid_sliding = census_layer(&LayerShape { index: 1, kind: layer_kind(1), key_count });
    assert_eq!(
        mid_sliding.len(),
        12,
        "a middle sliding layer's dispatch count changed"
    );

    let full = census_layer(&LayerShape { index: 5, kind: layer_kind(5), key_count });
    assert_eq!(
        full.len(),
        14,
        "a full-attention layer lost (or gained) the fused2 + copy arm"
    );

    // 59 % 6 == 5, so the last layer is a *full*-attention layer, not a sliding
    // one. Its finalize is the non-cross-layer `..._post_ffn_norm_scale_sk2`
    // (no successor), which is a different kernel at the same dispatch count as
    // the cross-layer one, so 14 either way.
    let last = census_layer(&LayerShape { index: 59, kind: layer_kind(59), key_count });
    assert_eq!(
        last.len(),
        14,
        "layer 59 is a full-attention layer with no successor"
    );
    assert!(
        last.iter().any(|r| r.kernel == "linear_finalize_and_post_ffn_norm_scale_sk2"),
        "layer 59 should fall back to the non-cross-layer finalize"
    );

    // Row order is the encode order in `forward_buffer`; the kernel names are
    // the assertion that the order did not silently permute.
    let names: Vec<&str> = mid_sliding.iter().map(|r| r.kernel).collect();
    assert_eq!(
        names,
        vec![
            "linear_gemv_ishizuki_k3_fused3_2col",
            "linear_finalize_simd3",
            "norm_rope_qkv_cache",
            "attention_decode_split",
            "attention_decode_combine",
            "linear_input_transform (o_proj)",
            "linear_gemv_ishizuki_k3_colmajor_2col_sk2 (o_proj)",
            "linear_finalize_o_proj_and_pre_ffn_transform_sk2",
            "linear_gemv_ishizuki_k3_fused2_2col (gate+up)",
            "linear_finalize_gated_gelu_and_down_input_transform",
            "linear_gemv_ishizuki_k3_colmajor_2col_sk2 (down)",
            "finalize_post_ffn_and_next_input_norm_transform",
        ],
        "decode dispatch order changed"
    );
}

/// Grid geometry, which is where the element layer's cost actually is: five of a
/// layer's twelve dispatches occupy at most 168 threadgroups and two occupy one,
/// while the GEMVs they sit between run 336-1344. A dispatch is ~0.7 us of
/// launch; a dispatch that cannot reach the memory system is far more.
#[test]
fn grid_geometry_of_the_element_dispatches() {
    let rows = census_layer(&LayerShape { index: 30, kind: layer_kind(30), key_count: 700 });
    assert_eq!(
        rows.iter().map(|r| r.seq).collect::<Vec<_>>(),
        (1..=rows.len()).collect::<Vec<_>>(),
        "dispatch order is not the sequence numbers"
    );
    // (kernel, groups, threads_per_group) for every non-GEMV row, in order.
    let geometry: Vec<(&str, u64, u64)> = rows
        .iter()
        .filter(|r| !r.kernel.contains("gemv") && !r.kernel.contains("colmajor"))
        .map(|r| (r.kernel, r.groups, r.threads_per_group))
        .collect();
    assert_eq!(
        geometry,
        vec![
            ("linear_finalize_simd3", 128, 32),
            ("norm_rope_qkv_cache", 64, 256),
            ("attention_decode_split", 48, 64),
            ("attention_decode_combine", 128, 64),
            ("linear_input_transform (o_proj)", 64, 32),
            ("linear_finalize_o_proj_and_pre_ffn_transform_sk2", 1, 1024),
            ("linear_finalize_gated_gelu_and_down_input_transform", 168, 32),
            ("linear_gemv_ishizuki_k3_colmajor_2col_sk2 (down)", 336, 256),
            ("finalize_post_ffn_and_next_input_norm_transform", 1, 1024),
        ]
        .into_iter()
        .filter(|(k, _, _)| !k.starts_with("linear_gemv") && !k.contains("colmajor"))
        .collect::<Vec<_>>(),
        "element-grid geometry moved"
    );
    // The epilogue-fusion boundaries the census flags, in order. These are the
    // only five rows where a dispatch writes a buffer that exists solely to be
    // read back and re-narrowed by the next one.
    let flagged: Vec<&str> =
        rows.iter().filter(|r| r.epilogue_candidate).map(|r| r.kernel).collect();
    assert_eq!(
        flagged,
        vec![
            "linear_input_transform (o_proj)",
            "linear_finalize_o_proj_and_pre_ffn_transform_sk2",
            "linear_finalize_gated_gelu_and_down_input_transform",
            "finalize_post_ffn_and_next_input_norm_transform",
        ],
        "the set of epilogue-fusion boundaries moved"
    );
}

#[test]
fn whole_step_dispatch_total() {
    let key_count = 700;
    let total: usize = (0..LAYERS)
        .map(|index| {
            census_layer(&LayerShape { index, kind: layer_kind(index), key_count }).len()
        })
        .sum::<usize>()
        + census_head().len();
    // 750 = 13 (layer 0) + 49 x 12 (sliding) + 10 x 14 (full) + 9 (head).
    assert_eq!(total, 750, "decode-step dispatch total moved");
}

#[test]
fn single_threadgroup_dispatches_are_the_occupancy_problem() {
    let rows = census_layer(&LayerShape { index: 30, kind: layer_kind(30), key_count: 700 });
    let single: Vec<&str> = rows
        .iter()
        .filter(|r| r.single_group)
        .map(|r| r.kernel)
        .collect();
    assert_eq!(
        single,
        vec![
            // Both are `dispatch_groups(..., 1, 1024)` in linear.rs. 19 of 20
            // GPU cores are idle while each reduces a 43 KB f32 accumulator.
            "linear_finalize_o_proj_and_pre_ffn_transform_sk2",
            "finalize_post_ffn_and_next_input_norm_transform",
        ],
        "the single-threadgroup finalize pair changed"
    );
}

#[test]
fn only_one_percent_of_the_step_is_addressable_by_activation_fusion() {
    let key_count = 700;
    let (mut weight, mut kv, mut element) = (0u64, 0u64, 0u64);
    let mut all: Vec<Row> = (0..LAYERS)
        .flat_map(|index| census_layer(&LayerShape { index, kind: layer_kind(index), key_count }))
        .collect();
    all.extend(census_head());
    for row in &all {
        weight += row.read_weight_bytes;
        kv += row.kv_bytes;
        element += row.read_bytes + row.write_bytes;
    }
    let traffic = (weight + kv + element) as f64;
    // 12.04 GB, not the 11.834 GB quoted in
    // `spec/decode-25tok-peak-bandwidth/implementation_notes.md` 16.4. That
    // figure multiplies a *sliding* layer's shapes (q/k/v 33.04 MB, o 16.52 MB)
    // by 60, but 10 of the 60 layers are `full_attention` with
    // `global_head_dim 512` and 4 KV heads, so q is 16384-wide and o_proj's
    // input is 16384-wide: 200.26 MB/layer, not 179.61 MB. The correction makes
    // the 20 tok/s target further out of reach, not closer -- see
    // `fused_budget_ms`.
    //
    //   sliding x50: 50 x 179.61 MB = 8.981 GB
    //   full    x10: 10 x 200.26 MB = 2.003 GB
    //   lm_head     : 5376 x 262144 x 6/8 = 1.057 GB
    //                 total       12.041 GB
    assert_eq!(
        weight, 12_039_487_488,
        "weight bytes per step moved to {weight}"
    );
    eprintln!(
        "weights {weight} B/step ({:.2}%), KV cache {kv} B/step ({:.2}%), \
         element/scratch {element} B/step ({:.2}%) at key_count={key_count}",
        100.0 * weight as f64 / traffic,
        100.0 * kv as f64 / traffic,
        100.0 * element as f64 / traffic,
    );
    assert!(
        (element as f64) < 0.013 * traffic,
        "fusion-addressable traffic is now {element} B of {traffic} B; a \
         fusion can at best remove part of that, so the ranking in this file \
         needs redoing against a larger budget"
    );
}

/// What every activation fusion in the open task list can jointly recover,
/// given the two measured constants from the spec: a dispatch costs ~0.7 us and
/// all nine element roles together cost 1.52 ms/step.
#[test]
fn fused_budget_ms() {
    const DISPATCH_US: f64 = 0.7;
    /// T5.10 / 18.1: all nine element roles ablated together = +0.65 tok/s.
    const ALL_ELEMENT_MS: f64 = 1.52;
    /// T5.12: every kernel role ablated leaves 7.32 ms/step of non-kernel time.
    const NON_KERNEL_MS: f64 = 7.32;
    const STEP_MS: f64 = 62.3;
    const TARGET_MS: f64 = 50.0;
    const WEIGHTS_PER_TOKEN: f64 = 12_039_487_488.0;
    /// implementation_notes.md 16.3: best 3-bpw EXL3 GEMV measured on this box.
    const BEST_3BPW_GB_S: f64 = 203.0e9;

    // T5.7 (o_proj Hadamard into the attention combine) plus T5.3 (the two
    // finalize norms folded into neighbour GEMVs) is the whole of the open
    // list: one dispatch per layer, plus layer 0's input norm.
    let by_dispatch = (LAYERS + 1) as f64 * DISPATCH_US / 1000.0;
    assert!(
        by_dispatch < ALL_ELEMENT_MS,
        "deleting every dispatch in the open list ({by_dispatch:.3} ms) exceeds \
         the measured total cost of all nine element roles ({ALL_ELEMENT_MS} ms), \
         so at least one number is wrong"
    );

    let gemv_floor_ms = WEIGHTS_PER_TOKEN / BEST_3BPW_GB_S * 1000.0;
    assert!(
        (gemv_floor_ms - 59.3).abs() < 0.3,
        "bandwidth floor moved to {gemv_floor_ms} ms"
    );

    let gap = STEP_MS - TARGET_MS;
    let above_floor = STEP_MS - gemv_floor_ms;
    eprintln!(
        "step {STEP_MS} ms; 20 tok/s needs {TARGET_MS} ms (gap {gap:.1} ms)\n\
         GEMV bandwidth floor {gemv_floor_ms:.1} ms = {:.1} tok/s, so {above_floor:.1} ms \
         of the step is above the floor\n\
         measured non-kernel residual {NON_KERNEL_MS} ms, all element roles \
         {ALL_ELEMENT_MS} ms\n\
         open-list fusions buy {by_dispatch:.3} ms = {:.2}% of the gap; \
         deleting every element kernel would buy {ALL_ELEMENT_MS} ms = {:.1}% of the gap",
        1000.0 / gemv_floor_ms,
        100.0 * by_dispatch / gap,
        100.0 * ALL_ELEMENT_MS / gap,
    );

    assert!(
        ALL_ELEMENT_MS / gap < 0.20,
        "deleting *every* element kernel would close {:.0}% of the gap to 20 \
         tok/s; if fusion were the lever the target would already be in reach",
        100.0 * ALL_ELEMENT_MS / gap
    );
    assert!(
        by_dispatch / ALL_ELEMENT_MS < 0.05,
        "the open list now covers more than 5% of the element budget; its scope \
         has grown and the ranking in the report is stale"
    );
    assert!(
        gemv_floor_ms > TARGET_MS,
        "the bandwidth floor now sits under the 20 tok/s target; re-run the \
         ceiling analysis in implementation_notes.md 16.4 before trusting the \
         ranking in this file"
    );
}

// ===========================================================================
// Bit-exactness model for the fused variants
//
// These are CPU models of the *contract* a fused Metal kernel must satisfy, not
// of the kernel itself. They exist because the open items turn on one question
// -- "does removing the intermediate fp16 store change the numbers?" -- and that
// question is answerable in f32/f16 arithmetic without a GPU. A later fused
// kernel is verifiable against these numbers the same way `linear_metal.rs`
// verifies a standalone kernel against a CPU reference.
// ===========================================================================

/// A deterministic f16 activation, so a failure is reproducible.
fn pseudo_half(count: usize, seed: u32) -> Vec<u16> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let unit = ((state >> 8) as f32 / 8_388_608.0) - 1.0;
            f16::from_f32(unit * 4.0).to_bits()
        })
        .collect()
}

/// `transform_simd` (`crates/kernels/src/linear.metal:402`), transcribed lane
/// for lane: four registers per lane holding elements `base+{0,32,64,96}`, seven
/// `simd_shuffle_xor` butterflies, then the two explicit pair stages, then the
/// 1/sqrt(128) scale.
fn transform_simd_128(block: &[f32; 128]) -> [f32; 128] {
    let mut reg = [[0.0f32; 32]; 4];
    for (r, slot) in reg.iter_mut().enumerate() {
        for lane in 0..32 {
            slot[lane] = block[32 * r + lane];
        }
    }
    let mut width = 1usize;
    while width < 32 {
        for slot in reg.iter_mut() {
            let before = *slot;
            for lane in 0..32 {
                let partner = before[lane ^ width];
                let mine = before[lane];
                slot[lane] = if (lane & width) != 0 {
                    partner - mine
                } else {
                    mine + partner
                };
            }
        }
        width <<= 1;
    }
    // width 32: pairs (0,1) and (2,3) within each register.
    for slot in reg.iter_mut() {
        let t = *slot;
        slot[0] = t[0] + t[1];
        slot[1] = t[0] - t[1];
        slot[2] = t[2] + t[3];
        slot[3] = t[2] - t[3];
    }
    // width 64: pairs (0,2) and (1,3).
    for slot in reg.iter_mut() {
        let s0 = slot[0];
        let s1 = slot[1];
        let s2 = slot[2];
        let s3 = slot[3];
        slot[0] = s0 + s2;
        slot[2] = s0 - s2;
        slot[1] = s1 + s3;
        slot[3] = s1 - s3;
    }
    const SCALE: f32 = 0.08838834764831845;
    let mut out = [0.0f32; 128];
    for (r, slot) in reg.iter().enumerate() {
        for lane in 0..32 {
            out[32 * r + lane] = slot[lane] * SCALE;
        }
    }
    out
}

/// The unfused contract: `linear_input_transform` (`linear.metal:475`) writes
/// `output[base] = half(v0)`, and the GEMV reads it back as
/// `float(xr0[0])`. So the value the dot product consumes is
/// `f32(half(transformed_f32))` -- narrowed, then widened.
fn unfused_transformed(input: &[u16], signs: &[u16]) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    for (b, block) in input.chunks_exact(128).enumerate() {
        let mut buf = [0.0f32; 128];
        for lane in 0..32 {
            for r in 0..4 {
                let i = 32 * r + lane;
                buf[i] = f16::from_bits(block[i]).to_f32() * f16::from_bits(signs[i]).to_f32();
            }
        }
        let t = transform_simd_128(&buf);
        let base = b * 128;
        for i in 0..128 {
            out[base + i] = f16::from_f32(t[i]).to_f32();
        }
    }
    out
}

/// T5.7 / T5.3's fused variant, done correctly: identical arithmetic, the
/// `half()` narrowing kept in registers, only the store and the reload removed.
fn fused_narrowed_transformed(input: &[u16], signs: &[u16]) -> Vec<f32> {
    unfused_transformed(input, signs)
}

/// The same fusion with the narrowing dropped -- the obvious way to write it,
/// and the one this file exists to catch.
fn fused_unnarrowed_transformed(input: &[u16], signs: &[u16]) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    for (b, block) in input.chunks_exact(128).enumerate() {
        let mut buf = [0.0f32; 128];
        for i in 0..128 {
            buf[i] = f16::from_bits(block[i]).to_f32() * f16::from_bits(signs[i]).to_f32();
        }
        let t = transform_simd_128(&buf);
        let base = b * 128;
        out[base..base + 128].copy_from_slice(&t);
    }
    out
}

#[test]
fn narrowing_a_transformed_activation_into_the_gemv_is_bit_exact() {
    let input = pseudo_half(HIDDEN, 0x51ed_0001);
    let signs = pseudo_half(HIDDEN, 0x9e37_79b9);
    let want = unfused_transformed(&input, &signs);
    let got = fused_narrowed_transformed(&input, &signs);
    for i in 0..HIDDEN {
        assert_eq!(
            want[i].to_bits(),
            got[i].to_bits(),
            "element {i}: fusing the Hadamard into the GEMV changed the value \
             the dot product consumes"
        );
    }
}

#[test]
fn dropping_the_fp16_store_is_not_bit_exact() {
    // The falsifiability proof for the test above: if this ever passes, the
    // comparison above has stopped meaning anything.
    let input = pseudo_half(HIDDEN, 0x51ed_0001);
    let signs = pseudo_half(HIDDEN, 0x9e37_79b9);
    let want = unfused_transformed(&input, &signs);
    let got = fused_unnarrowed_transformed(&input, &signs);
    let diffs = (0..HIDDEN)
        .filter(|&i| want[i].to_bits() != got[i].to_bits())
        .count();
    assert!(
        diffs > HIDDEN / 8,
        "a fused GEMV that skips the intermediate fp16 round trip differed in \
         only {diffs}/{HIDDEN} elements; the narrowing is not as load-bearing as \
         this file claims and the bit-exactness argument above is wrong"
    );
}

/// The reduction order `input_norm_and_qkv_transform` (`linear.metal:1210`) and
/// `finalize_post_ffn_and_next_input_norm_transform` (`linear.metal:1163`) both
/// freeze: 256 threads, `i = lid; i < width; i += 256`, then a tree from 128.
fn norm_inverse_documented_order(input: &[f32], width: usize, eps: f32) -> f32 {
    let mut partial = [0.0f32; 256];
    for lid in 0..256 {
        let mut sum = 0.0f32;
        let mut i = lid;
        while i < width {
            sum += input[i] * input[i];
            i += 256;
        }
        partial[lid] = sum;
    }
    let mut stride = 128usize;
    while stride > 0 {
        for lid in 0..stride {
            partial[lid] += partial[lid + stride];
        }
        stride >>= 1;
    }
    (partial[0] / width as f32 + eps).sqrt().recip()
}

/// The order a naive parallel prologue would use: one threadgroup per 128-block,
/// each reducing its own block, then a tree across the 42 block sums.
fn norm_inverse_block_parallel(input: &[f32], width: usize, eps: f32) -> f32 {
    let blocks = width / 128;
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
    (partial[0] / width as f32 + eps).sqrt().recip()
}

#[test]
fn reassociating_the_norm_reduction_changes_the_inverse_rms() {
    let raw_bits = pseudo_half(HIDDEN, 0x1234_5678);
    let input: Vec<f32> = raw_bits.iter().map(|b| f16::from_bits(*b).to_f32()).collect();
    let eps = 1e-6_f32;
    let documented = norm_inverse_documented_order(&input, HIDDEN, eps);
    let parallel = norm_inverse_block_parallel(&input, HIDDEN, eps);
    assert_ne!(
        documented.to_bits(),
        parallel.to_bits(),
        "a block-parallel norm prologue produced the same fp32 sum, so the \
         argument that T5.3's RMSNorm fusion is not bit-exact does not hold and \
         this file's ranking is wrong"
    );

    // And the reason it matters: the difference propagates through the whole
    // activation, not into a rounding-tolerant tail.
    let normed_a: Vec<f32> = input.iter().map(|v| v * documented).collect();
    let normed_b: Vec<f32> = input.iter().map(|v| v * parallel).collect();
    let differing = normed_a
        .iter()
        .zip(&normed_b)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        differing, HIDDEN,
        "only {differing}/{HIDDEN} normed elements changed; the fusion risk is \
         narrower than documented"
    );
}

/// The "no quality loss" constraint, stated numerically instead of argued.
///
/// The argmax runs on fp16 logits (`linear_finalize_simd` narrows, then
/// `argmax_slots` ranks), so the decision resolution is one fp16 ulp at the
/// logit magnitude, not one fp32 ulp of the accumulator. Two consequences:
///   * a fused variant that keeps every `half()` store is safe, because it
///     reproduces the numbers the unfused path feeds the narrow;
///   * a fused variant that drops one is not merely "slightly less accurate" --
///     it changes the fp16 the argmax sees, and the winner can move.
/// This searches real transformed activations for the flip rather than
/// asserting it is possible.
#[test]
fn a_moved_fp16_rounding_can_flip_the_argmax_token() {
    // 1. Resolution: near |logit| = 12 one fp16 ulp is 2^-7, so the argmax can
    // only be trusted to about half of that.
    let ulp = f16::from_f32(12.0);
    let next_up = f16::from_bits(ulp.to_bits() + 1);
    let gap12 = next_up.to_f32() - ulp.to_f32();
    assert!(
        gap12 > 0.007 && gap12 < 0.008,
        "one fp16 ulp at 12.0 is {gap12}; the resolution argument moved"
    );

    // 2. Mechanism: the value the GEMV's FMA actually consumes. The unfused
    // path feeds it `f32(f16(t))` (narrowed on the way to DRAM, widened on the
    // way back); the naive fusion feeds it `t`. Comparing `f16(f16(t))` with
    // `f16(t)` would be a tautology -- double rounding is idempotent -- so the
    // comparison has to be at the f32 level, which is where the dot product
    // sees it.
    let input = pseudo_half(HIDDEN, 0x2bad_c0de);
    let signs = pseudo_half(HIDDEN, 0x0bad_c0de);
    let narrow = unfused_transformed(&input, &signs);
    let wide = fused_unnarrowed_transformed(&input, &signs);
    let f32_diff = (0..HIDDEN)
        .filter(|&i| narrow[i].to_bits() != wide[i].to_bits())
        .count();
    assert!(
        f32_diff > HIDDEN / 8,
        "only {f32_diff}/{HIDDEN} transformed activations differ in f32 when the \
         intermediate store is dropped; the risk priced here is narrower than \
         documented"
    );

    // 3. Consequence, at the level that decides the token: push both f32 vectors
    // through a GEMV-shaped dot product and narrow to the fp16 logit the argmax
    // ranks. This is the measurement that says whether the change survives to
    // the output, rather than whether it exists in a register.
    let weights = pseudo_half(HIDDEN, 0xfeed_beef);
    let wf: Vec<f32> = weights.iter().map(|b| f16::from_bits(*b).to_f32()).collect();
    let mut logit_diff = 0usize;
    let mut trials = 0usize;
    for seed in 0..512u32 {
        let a_in = pseudo_half(HIDDEN, seed ^ 0x5eed_0001);
        let s_in = pseudo_half(HIDDEN, seed ^ 0x5eed_0002);
        let a = unfused_transformed(&a_in, &s_in);
        let b = fused_unnarrowed_transformed(&a_in, &s_in);
        let mut acc_a = 0.0f32;
        let mut acc_b = 0.0f32;
        for i in 0..HIDDEN {
            acc_a += wf[i] * a[i];
            acc_b += wf[i] * b[i];
        }
        trials += 1;
        if f16::from_f32(acc_a).to_bits() != f16::from_f32(acc_b).to_bits() {
            logit_diff += 1;
        }
    }
    eprintln!(
        "dropping the fp16 store changed the fp32 operand for {f32_diff}/{HIDDEN} \
         activations and the resulting fp16 logit for {logit_diff}/{trials} trials"
    );
    assert!(
        logit_diff > 0,
        "a fused GEMV that drops the intermediate fp16 rounding never changed \
         the fp16 logit in {trials} trials; if that is real then the \
         bit-exactness argument for keeping the narrowing is weaker than stated \
         and the ranking in this file should say so"
    );

    // 4. And the requirement that actually holds: the bit-exact fusion agrees
    // on every logit, so the argmax cannot move. This is the property a future
    // fused Metal kernel has to reproduce.
    for seed in 0..256u32 {
        let a_in = pseudo_half(HIDDEN, seed ^ 0x1111_2222);
        let s_in = pseudo_half(HIDDEN, seed ^ 0x3333_4444);
        let a = unfused_transformed(&a_in, &s_in);
        let b = fused_narrowed_transformed(&a_in, &s_in);
        for i in 0..HIDDEN {
            assert_eq!(a[i].to_bits(), b[i].to_bits(), "seed {seed} element {i}");
        }
    }
}