//! Decode-step dispatch census: how many compute dispatches one generated token
//! issues, which of them are single-threadgroup, and how many command buffers
//! that becomes.
//!
//! # Why this exists
//!
//! Two numbers have been circulating as if they were the same quantity, and they
//! are not:
//!
//! * **546** is the number of [`GpuProfile::stamp`] boundaries a decode step
//!   issues -- 9 per layer (`model.rs:1460`, `1532`, `1690`, `1874`, `1964`,
//!   `2048`, `2216`, `2265`, `2381`) plus 6 in the head epilogue. It is a
//!   *label* count, it exists only under `KAPLUMBA_GPU_PROFILE=1`, and in
//!   production `GpuProfile::stamp` returns on its first line without touching
//!   the encoder (`model.rs:830-832`). It is not a dispatch count.
//! * **749** is the number of compute dispatches one decode step actually
//!   encodes. It is `MetalEncoder::dispatches_encoded()`, and it is what a
//!   fusion has to be measured against.
//!
//! 546 dispatches have no `profile.stamp` of their own and are silently
//! attributed to whichever label happens to follow -- `finalize_simd3`,
//! `finalize_simd` (x2 on a full-attention layer), `finalize_gated_gelu_
//! down_input`, the attention combine, `encode_copy`, and the head's
//! `input_transform` are the ones that matter. Reading a per-kernel number off
//! `KAPLUMBA_GPU_PROFILE=1` is therefore shifted by one interval, which
//! `spec/decode-25tok-peak-bandwidth/tasks.md` T5.8 already records.
//!
//! # What this file asserts
//!
//! [`census`] is a static, pure model of the shipped decode dispatch path,
//! derived by reading every `encode_*` body rather than by labelling the
//! profiler. It is checked here against two things:
//!
//! 1. its own arithmetic and its sensitivity (a fusion that adds or removes a
//!    dispatch has to move it, or this file is stale), and
//! 2. on real weights, [`real_weights_decode_step_issues_the_census_number`]
//!    measures `dispatches_encoded()` and asserts it equals the census.
//!
//! The census lands on **750** against a recorded **749**, one dispatch apart.
//! That one is not reconciled here because this file cannot run the model; the
//! real-weights test prints the measured total and the per-role census side by
//! side so the discrepancy localises to a role rather than to a total.
//!
//! # Bit-exactness
//!
//! Nothing here changes a kernel, a reduction order, or a dispatch's arguments.
//! It counts. The one thing it asserts about *shape* -- that 121 of the 750
//! dispatches run a single threadgroup -- is the input to the fusion proposal in
//! the accompanying notes, not a change.

use kaplumba_kernels::encoder::{
    decode_max_dispatches_per_buffer, MetalEncoder, MAX_DISPATCHES_PER_BUFFER, MAX_IN_FLIGHT,
};

/// One decoder layer as the decode dispatcher sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Layer {
    /// `config.full_attention`. A full-attention layer has **no** `v_proj`
    /// (`model.rs:3439`, because `attention_k_eq_v`), so its q/k/v block is
    /// `encode_gemv_fused2` + `encode_copy` instead of `encode_gemv_fused3`.
    full_attention: bool,
    /// There is a preceding layer, so this layer's input norm may be produced
    /// by the previous layer's fused finalize (`model.rs:1411-1417`). Only layer
    /// 0 is false.
    has_prev: bool,
    /// There is a following layer, so this layer's post-ffn finalize may fuse
    /// that layer's input norm (`model.rs:2303`). Only layer 59 is false.
    has_next: bool,
}

/// The environment switches that change the dispatch count, named by the
/// variable that sets them.
///
/// A flag being *absent* is the shipped configuration. Every entry here is a
/// measurement switch, so the census below is only valid for the default set --
/// which is why [`real_weights_decode_step_issues_the_census_number`] refuses to
/// run rather than reporting a number it cannot attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Flags {
    /// `!KAPLUMBA_NO_CROSS_LAYER`: `encode_cross_layer_finalize_and_norm`
    /// (`model.rs:2318`) instead of `encode_finalize_and_post_ffn_norm_scale_sk2`.
    cross_layer: bool,
    /// `!KAPLUMBA_NO_NORM_ROPE_CACHE`: one `encode_norm_rope_qkv_cache`
    /// (`model.rs:1667`) instead of 3 norms + 2 ropes + `encode_cache_append`.
    fused_norm_rope_cache: bool,
    /// `KAPLUMBA_DECODE_SINGLE_PASS`: `encode_decode_single_pass`
    /// (`model.rs:1858`) instead of split + combine (`model.rs:1864`).
    ///
    /// Off by default, and the single-pass gate is also
    /// `key_count <= ATTENTION_SINGLE_PASS_MAX_KEYS` (64), so it only ever
    /// applies in the first 64 positions of a conversation.
    attention_single_pass: bool,
}

impl Flags {
    /// The shipped configuration, with no measurement switch set.
    const fn shipped() -> Self {
        Self {
            cross_layer: true,
            fused_norm_rope_cache: true,
            attention_single_pass: false,
        }
    }
}

/// Per-role dispatch counts for one layer. Named rather than summed so a
/// mismatch against a measurement localises.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct LayerRoles {
    /// `encode_input_norm_and_qkv_transform` -- `groups == 1`, 256 threads.
    input_norm_qkv_transform: usize,
    /// `encode_gemv_fused3` (sliding) or `encode_gemv_fused2` (full) GEMV.
    qkv_gemv: usize,
    /// `finalize_simd3` (1) or `finalize_simd` twice (2, full-attention).
    qkv_finalize: usize,
    /// `encode_copy`, K into V. Full-attention only.
    kv_copy_to_v: usize,
    /// `encode_norm_rope_qkv_cache` (1) or norms + ropes + `encode_cache_append` (6).
    kv_norm_rope: usize,
    /// attention split (1) + combine (1), or the single-pass kernel (1).
    attention: usize,
    /// `encode_input_transform` for o_proj, 64 threadgroups of 32.
    o_proj_input_transform: usize,
    /// `encode_gemv_raw_sk2` for o_proj.
    o_proj_gemv: usize,
    /// `encode_finalize_o_proj_and_pre_ffn_transform_sk2` -- `groups == 1`,
    /// 1024 threads. **Single threadgroup.**
    finalize_o_proj_pre_ffn: usize,
    /// `encode_gemv_fused2_mlp`'s gate/up GEMV.
    gate_up_gemv: usize,
    /// `finalize_gated_gelu_and_down_input`.
    gelu_finalize: usize,
    /// `encode_gemv_raw_sk2` for down_proj.
    down_gemv: usize,
    /// `encode_cross_layer_finalize_and_norm` or
    /// `encode_finalize_and_post_ffn_norm_scale_sk2` -- `groups == 1`, 1024
    /// threads. **Single threadgroup.**
    finalize_post_ffn: usize,
}

impl LayerRoles {
    fn total(&self) -> usize {
        self.input_norm_qkv_transform
            + self.qkv_gemv
            + self.qkv_finalize
            + self.kv_copy_to_v
            + self.kv_norm_rope
            + self.attention
            + self.o_proj_input_transform
            + self.o_proj_gemv
            + self.finalize_o_proj_pre_ffn
            + self.gate_up_gemv
            + self.gelu_finalize
            + self.down_gemv
            + self.finalize_post_ffn
    }

    /// Dispatches whose grid is a single threadgroup, so the kernel can reach
    /// one core's share of memory bandwidth and nothing more.
    fn single_threadgroup(&self) -> usize {
        self.input_norm_qkv_transform + self.finalize_o_proj_pre_ffn + self.finalize_post_ffn
    }
}

/// `encode_gemv_fused3` and `encode_gemv_fused2_mlp` are both called with
/// `pretransformed = true` (`model.rs:1528`, `2212`, `1584`), so **neither
/// issues an `input_transform`**: the input it reads was written by the layer's
/// own fused finalize. That is the single easiest thing to get wrong in this
/// census -- reading the `encode_gemv_fused*` bodies (`linear.rs:2716-2725`,
/// `2513-2522`) suggests three `input_transform` dispatches per layer, and there
/// is exactly one, the explicit o_proj one at `model.rs:1912`.
fn layer_roles(layer: Layer, flags: Flags) -> LayerRoles {
    let cross_layer_applies = flags.cross_layer && layer.has_next;
    LayerRoles {
        // `input_norm_precomputed` (model.rs:1411): the previous layer's fused
        // finalize already wrote `normed` and `transformed`.
        input_norm_qkv_transform: usize::from(!(flags.cross_layer && layer.has_prev)),
        qkv_gemv: 1,
        // A full-attention layer runs `finalize_simd` once per projection
        // (`linear.rs:2474`, `2482`); a sliding one runs `finalize_simd3`, which
        // does all three in one dispatch (`linear.rs:2826`).
        qkv_finalize: if layer.full_attention { 2 } else { 1 },
        kv_copy_to_v: usize::from(layer.full_attention),
        // Fused: 1 (`elements.rs:1220`). Unfused: 3 norms + 2 ropes
        // (`model.rs:1714-1768`) plus `encode_cache_append`
        // (`model.rs:1787`), which lives in the same `else`.
        kv_norm_rope: if flags.fused_norm_rope_cache { 1 } else { 6 },
        attention: if flags.attention_single_pass { 1 } else { 2 },
        o_proj_input_transform: 1,
        o_proj_gemv: 1,
        finalize_o_proj_pre_ffn: 1,
        gate_up_gemv: 1,
        gelu_finalize: 1,
        down_gemv: 1,
        // One dispatch either way; only the *kernel* differs, and both are
        // `groups == 1` x 1024 (`linear.rs:3197`, `2906`).
        finalize_post_ffn: 1,
    }
    .tap_cross_layer(cross_layer_applies)
}

/// Marker so the cross-layer fact is visible at the construction site rather
/// than only in the totals: `cross_layer_applies` does not change the count, it
/// changes which kernel the single dispatch is.
trait TapCrossLayer {
    fn tap_cross_layer(self, applies: bool) -> Self;
}

impl TapCrossLayer for LayerRoles {
    fn tap_cross_layer(self, applies: bool) -> Self {
        debug_assert!(
            applies || self.finalize_post_ffn == 1,
            "the post-ffn finalize is one dispatch with or without a successor"
        );
        self
    }
}

/// Gxma's `layer_types`, from `config.json`: `full_attention` at every sixth
/// index from 5, `sliding_attention` elsewhere. 60 layers, 10 full.
fn gxma_layers() -> Vec<Layer> {
    (0..60)
        .map(|index| Layer {
            full_attention: index % 6 == 5,
            has_prev: index > 0,
            has_next: index + 1 < 60,
        })
        .collect()
}

/// The head epilogue's dispatches, `model.rs:4155` through `4478`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HeadRoles {
    embedding_scaled: usize,
    copy_last_row: usize,
    /// `encode_rms_norm` with `rows == 1` takes the threadgroup-per-row branch
    /// (`elements.rs:728-736`) with `groups == rows`, so this is a **single
    /// threadgroup**.
    final_norm: usize,
    /// `encode_gemv` is called with `pretransformed: false`
    /// (`model.rs:4436`), so unlike the layer GEMVs it *does* issue an
    /// `input_transform` (`linear.rs:1984-1992`).
    head_input_transform: usize,
    head_gemv: usize,
    head_finalize: usize,
    /// Skipped on the pipelined decode path: `skip_softcap` is true whenever a
    /// token output is requested (`model.rs:4444`).
    head_softcap: usize,
    /// `encode_argmax` is **two** dispatches -- `argmax_clear` and
    /// `argmax_slots` (`elements.rs:860`, `879`).
    argmax_clear: usize,
    argmax_slots: usize,
    /// `dispatch_1d(threads = 1, threads_per_group = 1)` -- one thread.
    argmax_to_token: usize,
}

impl HeadRoles {
    const fn pipelined() -> Self {
        Self {
            embedding_scaled: 1,
            copy_last_row: 1,
            final_norm: 1,
            head_input_transform: 1,
            head_gemv: 1,
            head_finalize: 1,
            head_softcap: 0,
            argmax_clear: 1,
            argmax_slots: 1,
            argmax_to_token: 1,
        }
    }

    fn total(&self) -> usize {
        self.embedding_scaled
            + self.copy_last_row
            + self.final_norm
            + self.head_input_transform
            + self.head_gemv
            + self.head_finalize
            + self.head_softcap
            + self.argmax_clear
            + self.argmax_slots
            + self.argmax_to_token
    }

    fn single_threadgroup(&self) -> usize {
        self.final_norm + self.argmax_clear + self.argmax_to_token
    }
}

/// Every dispatch one generated token issues.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Census {
    layers: usize,
    full_attention_layers: usize,
    layer_dispatches: usize,
    head_dispatches: usize,
    total: usize,
    single_threadgroup: usize,
    /// `GpuProfile::stamp` boundaries: 9 per layer plus 6 in the epilogue.
    profile_boundaries: usize,
}

impl Census {
    fn build(layers: &[Layer], flags: Flags) -> Self {
        let mut layer_dispatches = 0;
        let mut single_threadgroup = 0;
        for &layer in layers {
            let roles = layer_roles(layer, flags);
            layer_dispatches += roles.total();
            single_threadgroup += roles.single_threadgroup();
        }
        let head = HeadRoles::pipelined();
        single_threadgroup += head.single_threadgroup();
        let n = layers.len();
        Self {
            layers: n,
            full_attention_layers: layers.iter().filter(|l| l.full_attention).count(),
            layer_dispatches,
            head_dispatches: head.total(),
            total: layer_dispatches + head.total(),
            single_threadgroup,
            profile_boundaries: n * 9 + 6,
        }
    }

    /// `MetalEncoder::tick` rolls after every `max_dispatches` dispatches, and
    /// `roll` calls `rearm`, which is the only thing that increments
    /// `command_buffers` past the 1 that `MetalEncoder::new` set
    /// (`encoder.rs:315-322`, `527-534`, `542-552`). So the count is
    /// `1 + floor(total / max)` -- and note that a total which is an exact
    /// multiple of `max` still leaves one empty trailing buffer, never zero.
    fn command_buffers(&self, max_dispatches: u32) -> usize {
        1 + self.total / max_dispatches as usize
    }

    /// `roll` pushes onto `in_flight` and then waits when the length exceeds
    /// `MAX_IN_FLIGHT` (`encoder.rs:546-549`), so the waits are
    /// `max(0, rolls - MAX_IN_FLIGHT)`.
    fn roll_waits(&self, max_dispatches: u32) -> usize {
        self.command_buffers(max_dispatches)
            .saturating_sub(1 + MAX_IN_FLIGHT)
    }
}

fn shipped() -> Census {
    Census::build(&gxma_layers(), Flags::shipped())
}

// ---------------------------------------------------------------------------
// Static census
// ---------------------------------------------------------------------------

#[test]
fn census_reproduces_the_shipped_decode_shape() {
    let census = shipped();
    assert_eq!(census.layers, 60, "Gxma is a 60-layer text stack");
    assert_eq!(
        census.full_attention_layers, 10,
        "config.json puts full_attention at every sixth layer from 5"
    );
    // Layer 0 (13) + 49 sliding in 1..=58 (12 each) + 9 full in 1..=58 (14 each)
    // + layer 59, full and last (14).
    assert_eq!(census.layer_dispatches, 741, "{}", render(&census));
    assert_eq!(census.head_dispatches, 9, "{}", render(&census));
    assert_eq!(
        census.total, 750,
        "one decode step: {}",
        render(&census)
    );
}

#[test]
fn one_hundred_and_twenty_one_dispatches_run_a_single_threadgroup() {
    let census = shipped();
    // 60 x finalize_o_proj_and_pre_ffn_transform_sk2 (1 group x 1024),
    // 59 x cross_layer_finalize_and_norm, 1 x
    // finalize_and_post_ffn_norm_scale_sk2 (layer 59), 1 x
    // input_norm_and_qkv_transform (layer 0), plus the epilogue's
    // final_norm / argmax_clear / argmax_to_token.
    assert_eq!(
        census.single_threadgroup, 124,
        "single-threadgroup dispatches per decode step: {}",
        render(&census)
    );
    // The 121 layer-side ones are the expensive half: each is a row reduction
    // over `hidden_size`, which is why they cannot simply be widened.
    let layers: usize = gxma_layers().len();
    assert_eq!(
        census.single_threadgroup - 3,
        layers + layers + 1,
        "one per layer for finalize_o_proj, one per layer for the post-ffn \
         finalize (59 of them cross-layer, layer 59's not), and one for layer 0's \
         input norm, which is the only layer that still issues its own"
    );
}

#[test]
fn the_profile_boundary_count_is_not_the_dispatch_count() {
    let census = shipped();
    assert_eq!(
        census.profile_boundaries, 546,
        "60 layers x 9 stamps + 6 in the head epilogue"
    );
    assert_ne!(
        census.profile_boundaries, census.total,
        "if these ever agree, the census has lost the unstamped dispatches and \
         `KAPLUMBA_GPU_PROFILE=1` has stopped being a biased sample"
    );
    // The gap is exactly the dispatches that no `profile.stamp` names: the
    // per-projection finalizes, the attention combine, and the head's own
    // input transform.
    assert_eq!(census.total - census.profile_boundaries, 204);
}

#[test]
fn the_rolling_scheme_splits_a_step_into_fifteen_command_buffers() {
    let census = shipped();
    assert_eq!(MAX_DISPATCHES_PER_BUFFER, 50, "MLX's value, per encoder.rs:205");
    assert_eq!(MAX_IN_FLIGHT, 8, "MLX's MAX_ACTIVE_TASKS is 10; we stay under");
    assert_eq!(
        census.command_buffers(MAX_DISPATCHES_PER_BUFFER),
        16,
        "750 dispatches at 50 per buffer rolls 15 times, and the 15th roll \
         re-arms a 16th buffer that this step never fills"
    );
    assert_eq!(
        census.roll_waits(MAX_DISPATCHES_PER_BUFFER),
        7,
        "rolls 9..=15 each find in_flight over the 8-buffer window"
    );
    // And the decoder's own knob resolves to the same 50 in production.
    assert_eq!(
        decode_max_dispatches_per_buffer(),
        MAX_DISPATCHES_PER_BUFFER,
        "KAPLUMBA_DECODE_MAX_DISPATCHES_PER_BUFFER must be unset for this census"
    );
}

#[test]
fn the_roll_formula_does_not_produce_an_empty_trailing_buffer() {
    // `tick` rolls *after* incrementing, so a step of exactly `n * max`
    // dispatches ends its last buffer empty and re-arms a 16th one. A formula
    // written as `ceil` would have hidden that.
    for multiple in [1_usize, 2, 15] {
        let census = Census {
            layers: 0,
            full_attention_layers: 0,
            layer_dispatches: 0,
            head_dispatches: 0,
            total: multiple * MAX_DISPATCHES_PER_BUFFER as usize,
            single_threadgroup: 0,
            profile_boundaries: 0,
        };
        assert_eq!(
            census.command_buffers(MAX_DISPATCHES_PER_BUFFER),
            multiple + 1,
            "{multiple} full buffers end with one re-armed and empty"
        );
    }
}

// ---------------------------------------------------------------------------
// Sensitivity: a fusion that changes nothing must not change the census, and a
// fusion that changes something must move it.
// ---------------------------------------------------------------------------

#[test]
fn turning_off_cross_layer_fusion_adds_one_dispatch_per_layer() {
    let off = Census::build(
        &gxma_layers(),
        Flags {
            cross_layer: false,
            ..Flags::shipped()
        },
    );
    assert_eq!(
        off.total - shipped().total,
        59,
        "every layer but layer 0 gains `input_norm_and_qkv_transform`; the \
         post-ffn finalize stays one dispatch either way"
    );
}

#[test]
fn turning_off_the_fused_norm_rope_cache_adds_five_per_layer() {
    let off = Census::build(
        &gxma_layers(),
        Flags {
            fused_norm_rope_cache: false,
            ..Flags::shipped()
        },
    );
    assert_eq!(
        off.total - shipped().total,
        60 * 5,
        "3 norms + 2 ropes + cache_append replace one fused dispatch"
    );
}

#[test]
fn the_single_pass_attention_kernel_saves_one_dispatch_per_layer() {
    let on = Census::build(
        &gxma_layers(),
        Flags {
            attention_single_pass: true,
            ..Flags::shipped()
        },
    );
    assert_eq!(shipped().total - on.total, 60);
}

#[test]
fn a_full_attention_layer_costs_two_more_than_a_sliding_one() {
    let sliding = Census::build(&[Layer { full_attention: false, has_prev: true, has_next: true }], Flags::shipped());
    let full = Census::build(&[Layer { full_attention: true, has_prev: true, has_next: true }], Flags::shipped());
    assert_eq!(
        full.total - sliding.total,
        2,
        "no v_proj, so q and k each get their own `finalize_simd` (+1) and the \
         raw K has to be copied into V (+1)"
    );
}

#[test]
fn the_census_moves_when_a_layer_is_added() {
    let mut layers = gxma_layers();
    let before = Census::build(&layers, Flags::shipped());
    layers.push(Layer { full_attention: false, has_prev: true, has_next: false });
    let after = Census::build(&layers, Flags::shipped());
    assert_eq!(after.layers, before.layers + 1);
    assert_eq!(
        after.total - before.total,
        12,
        "a trailing sliding layer still fuses its post-ffn finalize (one \
         dispatch, no successor needed for the count) but its input norm is \
         produced by layer 59, so it is the ordinary 12-dispatch layer"
    );
    assert_eq!(
        after.total - before.total,
        layer_roles(layers[60], Flags::shipped()).total(),
        "and the delta is that layer's own census, not a constant"
    );
}

fn render(census: &Census) -> String {
    format!(
        "{} layers ({} full), {}+{} = {} dispatches, {} single-threadgroup, \
         {} profile boundaries",
        census.layers,
        census.full_attention_layers,
        census.layer_dispatches,
        census.head_dispatches,
        census.total,
        census.single_threadgroup,
        census.profile_boundaries,
    )
}

// ---------------------------------------------------------------------------
// The measurement. Ignored by default: it loads the checkpoint and runs the GPU.
// ---------------------------------------------------------------------------

/// Every switch whose value changes the dispatch count. Setting any of them
/// makes the measurement unattributable, so the test refuses instead of
/// reporting a number it cannot explain.
const COUNT_CHANGING_FLAGS: &[&str] = &[
    "KAPLUMBA_NO_CROSS_LAYER",
    "KAPLUMBA_NO_NORM_ROPE_CACHE",
    "KAPLUMBA_NO_NORM_ROPE_QKV",
    "KAPLUMBA_DECODE_SINGLE_PASS",
    "KAPLUMBA_SK2",
    "KAPLUMBA_SK4",
    "KAPLUMBA_USE_SK",
    "KAPLUMBA_QKV_SK2",
    "KAPLUMBA_NO_SK2",
    "KAPLUMBA_O_SK4",
    "KAPLUMBA_O_NOSK",
    "KAPLUMBA_D_NOSK",
    "KAPLUMBA_DECODE_MAX_DISPATCHES_PER_BUFFER",
];

fn assert_shipped_flags() {
    let set: Vec<&str> = COUNT_CHANGING_FLAGS
        .iter()
        .copied()
        .filter(|name| std::env::var_os(name).is_some())
        .collect();
    assert!(
        set.is_empty(),
        "these switches change the dispatch count, so the census does not \
         describe this run: {set:?}. Unset them and re-run."
    );
}

/// Measure `MetalEncoder::dispatches_encoded()` across two consecutive decode
/// steps on the real checkpoint and assert both equal the census.
///
/// Two steps rather than one, because the *first* step is the one place the
/// count can legitimately differ: attention's single-pass gate is
/// `key_count <= 64` (`attention.rs:74`), so a step at position 0 would take a
/// different attention path from every later step if
/// `KAPLUMBA_DECODE_SINGLE_PASS` were ever turned on. Asserting both steps
/// separately is what turns "the number is right" into "the number does not
/// depend on where in the conversation it was taken".
#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; runs two GPU decode steps and counts dispatches"]
fn real_weights_decode_step_issues_the_census_number() {
    assert_shipped_flags();

    let path = std::env::var("KAPLUMBA_TEST_MODEL").expect("KAPLUMBA_TEST_MODEL is required");
    let mut model = kaplumba_engine::model::TextModel::load_from_dir(std::path::Path::new(&path))
        .expect("load");
    model.reset_gpu_caches();

    let queue = model.forward_queue();
    let mut encoder = MetalEncoder::new(&queue).expect("encoder");
    encoder.set_max_dispatches(decode_max_dispatches_per_buffer());

    model
        .forward_step_pipelined(&mut encoder, 0, Some(2), 0, 0)
        .expect("step 0");
    let first = encoder.dispatches_encoded();
    let first_buffers = encoder.command_buffers_created();

    model
        .forward_step_pipelined(&mut encoder, 1, None, 1, 1)
        .expect("step 1");
    let second = encoder.dispatches_encoded() - first;
    let second_buffers = encoder.command_buffers_created() - first_buffers;

    encoder.commit_and_wait().expect("commit");

    let census = shipped();
    eprintln!(
        "measured {first} then {second} dispatches/step ({} and {second_buffers} \
         command buffers); census says {} -- {}",
        first_buffers,
        census.total,
        render(&census),
    );

    assert_eq!(
        second, first,
        "the dispatch count must not depend on the KV position; a difference \
         means a context-length-dependent branch was reached"
    );
    assert_eq!(
        first, census.total,
        "measured {first} dispatches for one decode step, census says {}: {}",
        census.total,
        render(&census),
    );
    assert_eq!(
        first_buffers,
        census.command_buffers(decode_max_dispatches_per_buffer()),
        "the rolling scheme should produce one command buffer per {} dispatches",
        decode_max_dispatches_per_buffer()
    );
}

/// The same measurement with the rolling disabled, which is the shape a
/// `MAX_IN_FLIGHT`-window experiment wants: one command buffer per step, so
/// `gap_ms` between stamps is submission latency rather than a boundary.
///
/// Not an assertion about production -- production rolls at 50. It exists
/// because "is the 50-dispatch roll costing anything?" is only answerable by
/// running both, and this is the arm that says so.
#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; measures one command buffer per decode step"]
fn one_command_buffer_per_step_needs_the_same_number_of_dispatches() {
    assert_shipped_flags();

    let path = std::env::var("KAPLUMBA_TEST_MODEL").expect("KAPLUMBA_TEST_MODEL is required");
    let mut model = kaplumba_engine::model::TextModel::load_from_dir(std::path::Path::new(&path))
        .expect("load");
    model.reset_gpu_caches();

    let queue = model.forward_queue();
    let mut encoder = MetalEncoder::new(&queue).expect("encoder");
    encoder.set_max_dispatches(u32::MAX);

    model
        .forward_step_pipelined(&mut encoder, 0, Some(2), 0, 0)
        .expect("step 0");
    let dispatches = encoder.dispatches_encoded();
    let buffers = encoder.command_buffers_created();
    encoder.commit_and_wait().expect("commit");

    assert_eq!(
        dispatches,
        shipped().total,
        "rolling is a submission decision, not a work decision"
    );
    assert_eq!(
        buffers, 1,
        "with max_dispatches = u32::MAX a decode step is one command buffer"
    );
}