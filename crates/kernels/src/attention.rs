use std::ffi::c_void;
use std::mem::{size_of, size_of_val};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

pub use crate::exl3::KernelError;

const SOURCE: &str = include_str!("attention.metal");
const MAX_HEAD_DIM: usize = 512;

/// Most splits of the KV axis the decode path ever takes.
///
/// This is a hard cap, and it is what makes the decode scratch *fixed-size*: the
/// engine allocates `ATTENTION_SPLITS_MAX * q_heads * (head_dim + 2)` floats
/// once, at construction, and no prompt length can change it. 16 is what fills
/// the GPU at Gxma's 32 query heads -- `16 * 32 * 32` = 16,384 threads, ~6 waves
/// over a 20-core M4 Pro's resident thread capacity.
pub const ATTENTION_SPLITS_MAX: usize = 16;

/// Fewest keys a split may be given. Below this the combine pass's extra read
/// of the partials costs more than the parallelism buys.
const ATTENTION_MIN_KEYS_PER_SPLIT: usize = 64;

/// Threads the decode partial kernel aims to run, before the per-split cap.
///
/// **Retained only as the fallback for threadgroup widths whose one-wave
/// workgroup budget was not measured** -- see [`decode_splits`] and
/// [`decode_one_wave_workgroups`]. It is a thread target, and a thread target
/// is the wrong unit: what quantises is workgroups, and the two are not the
/// same thing once `kv_heads` differs between layer shapes.
const ATTENTION_DECODE_THREADS_TARGET: usize = 16_384;

/// Key counts at or below which the decode path prefers one single-pass dispatch
/// to split+combine, when `KAPLUMBA_DECODE_SINGLE_PASS` is set to a truthy value
/// that is not a number of keys. `None` (the default) means the production path
/// is untouched.
///
/// MEASURED, not the intersection of two cost models. See
/// `spec/decode-19tok-repacked-250gb/implementation_notes.md` ("single-pass
/// decode attention") for the sweep, the interleaved A/B medians and the negative
/// results. The short version: the two paths move *exactly the same DRAM bytes* --
/// a workgroup owns one kv head and its `q_heads / kv_heads` simdgroups read that
/// head's K and V out of the same L1, so a split never duplicates a byte. The
/// single-pass path therefore saves one dispatch and one
/// `splits * q_heads * (head_dim + 2)` float round trip, and pays for it with the
/// entire `splits` factor of parallelism.
///
/// MEASURED: **64 keys**, which is exactly `ATTENTION_MIN_KEYS_PER_SPLIT` and
/// therefore exactly where `decode_splits` already returns 1. That the crossover
/// lands on that number is structural, not a coincidence, and it is worth being
/// explicit about because it means the window is as narrow as it looks:
///
///   * At `splits == 1` both paths launch `kv_heads` workgroups of
///     `32 * (q_heads / kv_heads)` threads over the identical key range. The
///     *only* difference is the second dispatch and the
///     `q_heads * (head_dim + 2)` float round trip through the partials buffer,
///     so single-pass wins by exactly that: measured 1.059x (sliding) and
///     1.034x (full) at 64 keys, and 1.14x / 1.10x at 16 keys where the combine
///     is a larger share of a smaller kernel.
///   * At `splits == 2` the split path has twice the workgroups for the identical
///     byte count, and it wins immediately: 0.68x (sliding) and 0.63x (full) at 65
///     keys, falling to 0.13x and 0.14x by the shapes' maxima. The gap widens
///     monotonically with `splits` because the loss is the whole `splits` factor
///     of occupancy.
///
/// So this is a real but very short win: the first 64 tokens of a conversation,
/// worth 0.097 ms/token summed over 60 layers (0.08 sliding + 0.017 full), and
/// nothing at all after that. See
/// `spec/decode-19tok-repacked-250gb/implementation_notes.md` ("single-pass
/// decode attention") for the full interleaved tables and the negative results.
pub const ATTENTION_SINGLE_PASS_MAX_KEYS: usize = 64;

/// Highest key count `KAPLUMBA_DECODE_SINGLE_PASS=<n>` may be swept to. Far above
/// any crossover, and deliberately so: the sweep has to be able to reach the point
/// where single-pass is *much* worse, or it cannot see the shape of the curve.
pub const ATTENTION_SINGLE_PASS_SWEEP_MAX_KEYS: usize = 65_536;

/// Parse a `KAPLUMBA_DECODE_SINGLE_PASS` value into a key limit. `0` means off.
///
/// Exposed so the parsing rules are testable without the process-wide cache in
/// [`decode_single_pass_max_keys`], which is deliberately read once.
///
/// Total by construction: every value maps to a limit, and anything unrecognised
/// maps to off. This is consulted on the decode hot path, where a panic would
/// take the process down and where an unrecognised value must not silently widen
/// the grid.
pub fn parse_single_pass_limit(raw: &str) -> Option<usize> {
    let raw = raw.trim();
    match raw {
        // Explicitly off, spelled out so a shell that exports the variable empty
        // does not silently turn it on.
        "" | "0" | "off" | "false" | "no" => Some(0),
        // Truthy but not a number: on, up to the measured threshold. Kept as a
        // spelling of "on" so the default can be re-derived from a sweep without
        // editing and rebuilding.
        "1" | "on" | "true" | "yes" | "auto" => Some(ATTENTION_SINGLE_PASS_MAX_KEYS),
        // Any other number is that many keys. Clamped to the sweep range, which is
        // about the sweep reaching the whole curve and about a typo not becoming
        // an unbounded grid; this path does not touch the partials buffer, so the
        // clamp is not a scratch-safety limit.
        _ => Some(
            raw.parse::<usize>()
                .ok()
                .map_or(0, |keys| keys.min(ATTENTION_SINGLE_PASS_SWEEP_MAX_KEYS)),
        ),
    }
}

/// How many keys of decode KV context may use the single-pass dispatch, or `None`
/// to always use split+combine. **Default is `None`** -- the production decode
/// path is unchanged unless this environment variable is set.
///
/// `KAPLUMBA_DECODE_SINGLE_PASS`:
///
/// | value | meaning |
/// |---|---|
/// | unset, `0`, `off`, `false`, `no`, empty | off (default) |
/// | `1`, `on`, `true`, `yes`, `auto` | on, up to [`ATTENTION_SINGLE_PASS_MAX_KEYS`] keys |
/// | `N` (an integer) | on, up to `N` keys -- how the crossover was swept |
///
/// Read once per process: it is consulted once per layer per token (60x/token)
/// and `getenv` takes a lock in some libc builds. Every layer reads the same
/// answer, which is also why the number has to be right rather than lucky.
pub fn decode_single_pass_max_keys() -> Option<usize> {
    static CACHE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        let raw = std::env::var("KAPLUMBA_DECODE_SINGLE_PASS").ok()?;
        parse_single_pass_limit(&raw).filter(|limit| *limit > 0)
    })
}

/// Floats the decode partial buffer holds for a layer: two softmax statistics
/// and one unnormalised accumulator per (split, query head).
pub fn decode_partial_elements(q_heads: usize, head_dim: usize) -> usize {
    let rows = ATTENTION_SPLITS_MAX.saturating_mul(q_heads);
    rows.saturating_mul(head_dim)
        .saturating_add(rows.saturating_mul(2))
}

/// How many ways to cut `key_count` keys for the decode path.
///
/// **MEASUREMENT KNOB / DEFAULT PENDING.** The split count used to come from a
/// *thread* target, `ATTENTION_DECODE_THREADS_TARGET / (32 * q_heads)`, and the
/// missing `kv_heads` argument is what made that wrong: a threadgroup is
/// `32 * (q_heads / kv_heads)` threads, so holding the thread count at 16384
/// lands both of Gxma's layer shapes on 16 splits while giving them workgroup
/// counts that differ 4x -- 16 kv heads x 16 = **256** workgroups of 64 threads
/// on the 50 sliding layers, versus 4 x 16 = **64** workgroups of 256 threads on
/// the 10 full-attention ones. What a GPU scheduler actually packs onto a core is
/// workgroups, not threads, so the thread target silently assumed a residency
/// that the wide-threadgroup shape does not have.
///
/// The rule is therefore a *one-wave workgroup budget*,
/// [`decode_one_wave_workgroups`], and the result is still in
/// `1 ..= ATTENTION_SPLITS_MAX` so the scratch stays fixed-size. See
/// `spec/decode-19tok/implementation_notes.md` ("the decode split count, from a
/// workgroup budget rather than a thread count") for the sweep, including the
/// measurement that the sliding shape is unchanged at 16 and why that is
/// structural rather than luck.
///
/// The budget is stated for the **whole device** rather than per core because
/// the measurement is: 54 workgroups of 256 threads is one wave on this box and
/// 55 is two, which is `18 * 3` rather than `20 * 3` or `20 * 2.7`, and picking
/// a core count to make the arithmetic come out would be inventing a fact.
///
/// **The one-wave rule is NOT the default, and the reason is bit-exactness
/// rather than doubt about the speed.** Changing the split count changes the
/// order in which the combine pass sums the keys, and floating-point addition is
/// not associative, so 13 splits is a different -- equally valid, not equally
/// identical -- answer to the same question.
///
/// The precedent for shipping default-off is in this repo and it cuts against
/// the obvious argument. `spec/decode-19tok-repacked-250gb` §21.2-21.4 measured
/// a *larger* perturbation elsewhere -- the gate/up GEMV fusion is within 1 fp16
/// ULP on 0.78% of activations per layer, across all 60 layers -- and it changed
/// **zero** of 128 greedy tokens. This change is smaller (1 ULP on 3-4 of 16384
/// attention outputs, on 10 of 60 layers) and changes **43 of 128**. So "it is
/// only 1 ULP, and we already accept 1 ULP" is not a valid argument: 1 ULP is
/// not a unit of comparison across sites, and only the measurement settles it.
///
/// The rule therefore ships behind `KAPLUMBA_DECODE_SPLIT_RULE=workgroups` and
/// the default stays the thread target. The measured consequence is in
/// `spec/decode-19tok/implementation_notes.md`: 1.34x on the full-attention
/// layers at 8k of context, and a different token stream.
///
/// `KAPLUMBA_DECODE_SPLITS` still pins the count, so the two can be compared
/// without a rebuild.
pub fn decode_splits(q_heads: usize, kv_heads: usize, key_count: usize) -> usize {
    if q_heads == 0 || key_count == 0 {
        return 1;
    }
    // MEASUREMENT KNOB. `KAPLUMBA_DECODE_SPLITS=N` pins the split count so the
    // choice can be swept without a rebuild, which is how the workgroups-per-core
    // rule below was measured at all. It outranks the rule selector, so a sweep
    // works under either rule.
    let pinned = std::env::var("KAPLUMBA_DECODE_SPLITS")
        .ok()
        .as_deref()
        .and_then(parse_pinned_splits);
    decode_splits_with(q_heads, kv_heads, key_count, pinned, decode_split_rule())
}

/// Which arithmetic [`decode_splits`] uses. See [`decode_split_rule`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecodeSplitRule {
    /// The historical `ATTENTION_DECODE_THREADS_TARGET / (32 * q_heads)`. The
    /// default, and bit-identical to what shipped before `kv_heads` was passed in.
    ThreadTarget,
    /// One wave of workgroups, sized by [`decode_one_wave_workgroups`].
    WorkgroupsPerCore,
}

/// `KAPLUMBA_DECODE_SPLIT_RULE` -> the arithmetic to use. Default
/// [`DecodeSplitRule::ThreadTarget`].
///
/// | value | rule |
/// |---|---|
/// | unset, empty, `0`, `off`, `false`, `no`, `threads`, `legacy` | thread target (default) |
/// | `workgroups`, `waves`, `one_wave`, `1`, `on`, `true`, `yes`, `auto` | one-wave workgroup budget |
///
/// Parsed as a function so it is testable without touching the process
/// environment, which Rust 2024 makes `unsafe` and which every other test in
/// this process shares. Total by construction: an unrecognised value is the
/// *default*, never an error, because this is read on the decode hot path where
/// a panic would take the process down and where a typo must not silently
/// change the numerics of the model.
pub fn parse_split_rule(raw: &str) -> DecodeSplitRule {
    match raw.trim() {
        "workgroups" | "waves" | "one_wave" | "1" | "on" | "true" | "yes" | "auto" => {
            DecodeSplitRule::WorkgroupsPerCore
        }
        _ => DecodeSplitRule::ThreadTarget,
    }
}

/// The process's rule, read once. It cannot meaningfully change mid-process --
/// the split count is baked into every decode step's numerics -- and `getenv`
/// takes a lock in some libc builds, so this is consulted once per layer per
/// token (60x/token) rather than 60 times a second of `getenv`.
pub fn decode_split_rule() -> DecodeSplitRule {
    static CACHE: std::sync::OnceLock<DecodeSplitRule> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("KAPLUMBA_DECODE_SPLIT_RULE")
            .ok()
            .as_deref()
            .map_or(DecodeSplitRule::ThreadTarget, parse_split_rule)
    })
}

/// `KAPLUMBA_DECODE_SPLITS` -> a pinned split count, or `None` for "use the
/// heuristic".
///
/// Split out of [`decode_splits`] so the parsing is testable without mutating the
/// process environment, which Rust 2024 makes `unsafe` and which other tests in
/// this process share. Total by construction: anything unrecognised, out of
/// range, or unparseable is `None` (the heuristic) rather than an error, because
/// this is read on the decode hot path where a panic would take the process down
/// and where a typo must not be able to request a grid the scratch cannot hold.
fn parse_pinned_splits(raw: &str) -> Option<usize> {
    raw.trim()
        .parse::<usize>()
        .ok()
        .filter(|value| (1..=ATTENTION_SPLITS_MAX).contains(value))
}

/// The split-count rule with the environment already resolved. Pure, so every
/// boundary it has is testable without a GPU and without the environment.
fn decode_splits_with(
    q_heads: usize,
    kv_heads: usize,
    key_count: usize,
    pinned: Option<usize>,
    rule: DecodeSplitRule,
) -> usize {
    if let Some(pinned) = pinned {
        return pinned;
    }
    if q_heads == 0 || key_count == 0 {
        return 1;
    }
    // `kv_heads == 0` is not reachable from the encode path, which rejects it
    // before it gets here, but the arithmetic must not divide by zero if a
    // future caller forgets: `max(1)` keeps it finite and the result is then
    // decided entirely by the cap, which is harmless.
    let kv_heads = kv_heads.max(1);
    let affordable = key_count.div_ceil(ATTENTION_MIN_KEYS_PER_SPLIT);
    // One wave: `splits * kv_heads <= budget`. Integer division rounds *down*,
    // which is the safe direction -- the alternative is a grid that spills into
    // a second wave, and a second wave is not a small penalty: measured, 56
    // workgroups of 256 threads costs 1.50x the time of 52 for the identical
    // byte count.
    let workgroup_budget = decode_one_wave_workgroups(32 * (q_heads / kv_heads));
    let wanted = match (rule, workgroup_budget) {
        (DecodeSplitRule::WorkgroupsPerCore, Some(budget)) => {
            (budget / kv_heads).clamp(1, ATTENTION_SPLITS_MAX)
        }
        // Either the default rule, or a width whose budget was never measured:
        // the historical thread target, which is what ships.
        _ => {
            (ATTENTION_DECODE_THREADS_TARGET / (32 * q_heads.max(1))).clamp(1, ATTENTION_SPLITS_MAX)
        }
    };
    wanted.min(affordable).max(1)
}

/// The largest decode grid, in workgroups, that still runs in **one wave** on
/// this box for a threadgroup of `threads_per_group` threads. `None` for a
/// width that was not measured, which keeps the historical thread target.
///
/// A scheduler reserves per-core resources for a whole threadgroup, so a core
/// holds far fewer of a wide one, and that is the entire content of this fix.
/// Measured by sweeping every split count 1..=16 for three shapes that all use
/// a 256-thread threadgroup and differ only in `kv_heads`, so the grid size
/// `splits * kv_heads` walks multiples of 4, 5 and 6 and the transition is
/// bracketed from both sides:
///
/// | grid | 48 | 50 | **52** | **54** | **55** | **56** | 60 | 64 |
/// |---|---|---|---|---|---|---|---|---|
/// | waves | 1 | 1 | **1** | **1** | 2 | 2 | 2 | 2 |
///
/// 54 of 256 threads is one wave and 55 is two, so the budget is **54**. It is
/// `18 * 3` rather than `20 * 3`; which of those it is does not change any
/// answer here, and picking a core count to make the arithmetic tidy would be
/// inventing a fact, so the constant is stated as the measurement.
///
/// The rule `budget / kv_heads` then reproduces the *measured* optimum for all
/// three shapes it was measured on, which is the check that it is a rule and not
/// a fit to one number: `kv_heads 4 -> 13` (measured best 13), `kv_heads 5 ->
/// 10` (measured best 10, 50 workgroups), `kv_heads 6 -> 9` (measured best 9,
/// 54 workgroups).
///
/// The 64-thread entry is the important negative result. 288 is `18 * 16`, and
/// it is a *lower* bound: the sliding curve is still improving at the 256
/// workgroups the [`ATTENTION_SPLITS_MAX`] cap allows, so the honest statement
/// is "at least 288", which for 16 kv heads is 18 splits -- above the cap, so
/// the cap decides and the shipped 16 is unchanged. **That is why 83% of the
/// layers cannot regress from this rule: for them it provably evaluates to the
/// number that ships today**, and `the_pinned_sweep_reproduces_the_reference_table`
/// checks it against the reference measurement rather than against a constant
/// typed twice.
pub const fn decode_one_wave_workgroups(threads_per_group: usize) -> Option<usize> {
    match threads_per_group {
        64 => Some(288),
        256 => Some(54),
        _ => None,
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AttentionParams {
    queries: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    key_count: u32,
    scale: f32,
    window: u32,
    /// Decode only: KV splits and keys per split. Zero means "scan every key",
    /// which is what the prefill kernels get.
    splits: u32,
    keys_per_split: u32,
}

#[derive(Clone, Copy)]
pub struct AttentionRequest<'a> {
    pub queries: &'a [u16],
    pub keys: &'a [u16],
    pub values: &'a [u16],
    pub query_positions: &'a [u32],
    pub key_positions: &'a [u32],
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
    pub window: usize,
}

/// GPU-resident attention request: every buffer is caller-owned so attention
/// can be encoded into the same command buffer as the rest of the layer.
#[derive(Clone, Copy)]
pub struct AttentionEncodeRequest<'a> {
    pub queries: &'a Buffer,
    pub keys: &'a Buffer,
    pub values: &'a Buffer,
    pub query_positions: &'a Buffer,
    pub key_positions: &'a Buffer,
    pub output: &'a Buffer,
    pub queries_count: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub key_count: usize,
    pub scale: f32,
    pub window: usize,
}

pub struct AttentionKernels {
    device: Device,
    queue: CommandQueue,
    _library: Library,
    online_attention_pipeline: ComputePipelineState,
    online_attention_256_pipeline: ComputePipelineState,
    online_attention_512_pipeline: ComputePipelineState,
    attention_dim_256_pipeline: ComputePipelineState,
    attention_dim_512_pipeline: ComputePipelineState,
    attention_tg2_256_pipeline: ComputePipelineState,
    attention_tg2_512_pipeline: ComputePipelineState,
    /// Transposed prefill kernel, indexed by `gqa_index(head_dim, nsg)`.
    /// `head_dim` is 256 or 512; `nsg` is the GQA ratio `q_heads / kv_heads`,
    /// which the kernel needs verbatim because it sizes the threadgroup.
    attention_gqa_pipelines: [ComputePipelineState; GQA_PIPELINES],
    /// The same prefill shapes with the simdgroup-MMA QK scoring (T1,
    /// `spec/prefill-120tps`), indexed by `gqa_index` like the scalar array.
    /// Built eagerly in `new()` whether or not `attention_qk()` selects them:
    /// `pipeline_warmup_coverage` fails on any pipeline a dispatch compiles
    /// after construction.
    attention_gqa_mma_pipelines: [ComputePipelineState; GQA_PIPELINES],
    /// The same body run over one split of the KV axis (flash-decoding).
    decode_split_pipelines: [ComputePipelineState; GQA_PIPELINES],
    /// The same body run over the whole KV axis in one threadgroup, writing the
    /// normalised output directly. Only `(256, 2)` and `(512, 8)` -- the two
    /// shapes this model has -- because the host only selects it after
    /// `gqa_index` has already resolved, and the metal side instantiates
    /// nothing else.
    decode_single_pipelines: [Option<ComputePipelineState>; GQA_PIPELINES],
    decode_combine_256_pipeline: ComputePipelineState,
    decode_combine_512_pipeline: ComputePipelineState,
}

/// `q_heads / kv_heads` values the transposed kernel is instantiated for, and
/// therefore also the simdgroup counts. Anything else falls back to the
/// 256-thread `online_attention_tg2_*` kernel, which makes no such assumption.
const GQA_RATIOS: [usize; 4] = [1, 2, 4, 8];
const GQA_HEAD_DIMS: [usize; 2] = [256, 512];
const GQA_PIPELINES: usize = GQA_RATIOS.len() * GQA_HEAD_DIMS.len();

/// Position of a (head_dim, nsg) pair in `attention_gqa_pipelines`, or `None`
/// when the shape has no specialisation.
fn gqa_index(head_dim: usize, nsg: usize) -> Option<usize> {
    let dim = GQA_HEAD_DIMS.iter().position(|d| *d == head_dim)?;
    let ratio = GQA_RATIOS.iter().position(|r| *r == nsg)?;
    Some(dim * GQA_RATIOS.len() + ratio)
}

fn gqa_pipeline_name(head_dim: usize, nsg: usize) -> String {
    format!("online_attention_gqa_{head_dim}_{nsg}")
}

/// Kernel name of the MMA-QK specialisation for one prefill shape.
fn gqa_mma_pipeline_name(head_dim: usize, nsg: usize) -> String {
    format!("online_attention_gqa_mma_{head_dim}_{nsg}")
}

/// The pipeline-array index a decode shape resolves to, or `None` when it has no
/// specialisation and the prefill kernels take it. Public because
/// [`AttentionKernels::encode_decode_single_pass`] and
/// [`AttentionKernels::encode_decode_split_with_splits`] both take the index
/// rather than deriving it, and a caller that wants to compare the two paths
/// needs the same value `encode_online_attention_split` would have computed.
pub fn decode_gqa_index(head_dim: usize, q_heads: usize, kv_heads: usize) -> Option<usize> {
    if kv_heads == 0 || q_heads % kv_heads != 0 {
        return None;
    }
    gqa_index(head_dim, q_heads / kv_heads)
}

fn decode_split_pipeline_name(head_dim: usize, nsg: usize) -> String {
    format!("attention_decode_split_{head_dim}_{nsg}")
}

/// `(head_dim, nsg)` pairs the single-pass decode kernel is instantiated for.
/// See the `AttentionKernels::decode_single_pipelines` field.
fn decode_single_pass_name(head_dim: usize, nsg: usize) -> Option<String> {
    match (head_dim, nsg) {
        (256, 2) | (512, 8) => Some(format!("attention_decode_single_{head_dim}_{nsg}")),
        _ => None,
    }
}

/// Which prefill QK implementation `encode_online_attention` dispatches to.
///
/// The two arms are the T1 A/B (`spec/prefill-120tps`): `Scalar` is the
/// per-lane dot product in `attention_gqa_body`, `Mma` is the simdgroup-8x8x8
/// scoring in `attention_gqa_mma_body`. Both are reachable in-process through
/// [`AttentionKernels::encode_online_attention_qk`], which is what
/// `attention_qk_mma_matches_scalar` compares; the environment variable only
/// picks which one the production entry point forwards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionQk {
    Scalar,
    Mma,
}

/// True for the spellings that mean "on"; anything else is off, so a typo can
/// never silently select a path the caller did not ask for.
fn qk_env_truthy(raw: &str) -> bool {
    matches!(raw.trim(), "1" | "on" | "true" | "yes")
}

/// The process-wide prefill QK selection, read once (same `OnceLock` idiom as
/// [`decode_single_pass_max_keys`]: it is consulted once per layer per prompt,
/// and `getenv` takes a lock in some libc builds).
///
/// * `KAPLUMBA_SCALAR_ATTENTION=1|on|true|yes` -> [`AttentionQk::Scalar`],
///   always wins: this is the brief's A/B spelling for forcing the baseline.
/// * else `KAPLUMBA_MMA_ATTENTION=1|on|true|yes` -> [`AttentionQk::Mma`].
/// * else the default, **`Scalar`**.
///
/// The default stays `Scalar` deliberately: `attention_single_pass_bit_exact`
/// asserts the prefill GQA output equals the single-pass decode output in 0 of
/// N fp16 elements, and the MMA's reassociated dot product cannot satisfy that.
/// Flipping the default is a coordinator decision recorded in
/// `spec/prefill-120tps/notes_T1_T2.md`, not something this knob does on its
/// own. Decode never consults this: it goes through
/// `encode_online_attention_split` / the single-pass path, neither of which
/// calls `encode_online_attention`.
pub fn attention_qk() -> AttentionQk {
    static CACHE: std::sync::OnceLock<AttentionQk> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        if std::env::var("KAPLUMBA_SCALAR_ATTENTION")
            .map(|raw| qk_env_truthy(&raw))
            .unwrap_or(false)
        {
            return AttentionQk::Scalar;
        }
        if std::env::var("KAPLUMBA_MMA_ATTENTION")
            .map(|raw| qk_env_truthy(&raw))
            .unwrap_or(false)
        {
            return AttentionQk::Mma;
        }
        AttentionQk::Scalar
    })
}

/// Threadgroups of `per_group` threads needed to cover `threads`, at least one.
/// The kernels that index the grid themselves early-return on the surplus.
fn threads_for(threads: usize, per_group: usize) -> u64 {
    threads.div_ceil(per_group).max(1) as u64
}

impl AttentionKernels {
    pub fn new() -> Result<Self, KernelError> {
        let device = Device::system_default().ok_or(KernelError::DeviceUnavailable)?;
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(SOURCE, &options)
            .map_err(KernelError::Shader)?;
        let mut extra = |name: &str| -> Result<ComputePipelineState, KernelError> {
            // Through the audit, like every other family: a pipeline built here
            // is a pipeline that a later `--metrics`-style timed loop must not
            // be the first to reach. See `crate::pipeline_audit`.
            crate::pipeline_audit::build(&device, &library, name)
        };
        let online_attention_pipeline = extra("online_attention")?;
        let online_attention_256_pipeline = extra("online_attention_256")?;
        let online_attention_512_pipeline = extra("online_attention_512")?;
        let attention_dim_256_pipeline = extra("online_attention_dim_256")?;
        let attention_dim_512_pipeline = extra("online_attention_dim_512")?;
        let attention_tg2_256_pipeline = extra("online_attention_tg2_256")?;
        let attention_tg2_512_pipeline = extra("online_attention_tg2_512")?;
        let mut attention_gqa_pipelines = Vec::with_capacity(GQA_PIPELINES);
        let mut attention_gqa_mma_pipelines = Vec::with_capacity(GQA_PIPELINES);
        let mut decode_split_pipelines = Vec::with_capacity(GQA_PIPELINES);
        for head_dim in GQA_HEAD_DIMS {
            for nsg in GQA_RATIOS {
                attention_gqa_pipelines.push(extra(&gqa_pipeline_name(head_dim, nsg))?);
                // T1: built unconditionally so the selection is an env-var
                // choice between two already-compiled pipelines, never a
                // compile inside a measured dispatch.
                attention_gqa_mma_pipelines
                    .push(extra(&gqa_mma_pipeline_name(head_dim, nsg))?);
                decode_split_pipelines.push(extra(&decode_split_pipeline_name(head_dim, nsg))?);
            }
        }
        let attention_gqa_pipelines: [ComputePipelineState; GQA_PIPELINES] =
            attention_gqa_pipelines.try_into().map_err(|_| {
                KernelError::Pipeline("unexpected transposed attention pipeline count".to_string())
            })?;
        let attention_gqa_mma_pipelines: [ComputePipelineState; GQA_PIPELINES] =
            attention_gqa_mma_pipelines.try_into().map_err(|_| {
                KernelError::Pipeline("unexpected MMA attention pipeline count".to_string())
            })?;
        let decode_split_pipelines: [ComputePipelineState; GQA_PIPELINES] =
            decode_split_pipelines.try_into().map_err(|_| {
                KernelError::Pipeline("unexpected decode split pipeline count".to_string())
            })?;
        let mut decode_single_pipelines: [Option<ComputePipelineState>; GQA_PIPELINES] =
            [const { None }; GQA_PIPELINES];
        for head_dim in GQA_HEAD_DIMS {
            for nsg in GQA_RATIOS {
                let Some(name) = decode_single_pass_name(head_dim, nsg) else {
                    continue;
                };
                // A missing single-pass kernel must not take the whole library
                // down: the fallback below drops the dispatch to split+combine,
                // which is the shipping path. So this is the one `extra` call
                // whose failure is not propagated.
                if let Some(index) = gqa_index(head_dim, nsg) {
                    decode_single_pipelines[index] =
                        crate::pipeline_audit::build(&device, &library, &name).ok();
                }
            }
        }
        let decode_combine_256_pipeline = extra("attention_decode_combine_256")?;
        let decode_combine_512_pipeline = extra("attention_decode_combine_512")?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            _library: library,
            online_attention_pipeline,
            online_attention_256_pipeline,
            online_attention_512_pipeline,
            attention_dim_256_pipeline,
            attention_dim_512_pipeline,
            attention_tg2_256_pipeline,
            attention_tg2_512_pipeline,
            attention_gqa_pipelines,
            attention_gqa_mma_pipelines,
            decode_split_pipelines,
            decode_single_pipelines,
            decode_combine_256_pipeline,
            decode_combine_512_pipeline,
        })
    }

    pub fn online_attention(&self, request: AttentionRequest<'_>) -> Result<Vec<u16>, KernelError> {
        validate_request(&request)?;
        let AttentionRequest {
            queries,
            keys,
            values,
            query_positions,
            key_positions,
            q_heads,
            kv_heads,
            head_dim,
            scale,
            window,
        } = request;
        let key_count = key_positions.len();
        let output_elements = checked_product(
            query_positions.len(),
            checked_product(q_heads, head_dim, KvAttentionError::ShapeOverflow)?,
            KvAttentionError::ShapeOverflow,
        )?;
        let query_buffer = self.make_u16_buffer(queries)?;
        let key_buffer = self.make_u16_buffer(keys)?;
        let value_buffer = self.make_u16_buffer(values)?;
        let query_position_buffer = self.make_u32_buffer(query_positions)?;
        let key_position_buffer = self.make_u32_buffer(key_positions)?;
        let output = self.device.new_buffer(
            (output_elements * size_of::<u16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params = AttentionParams {
            queries: to_u32(query_positions.len(), "attention query count")?,
            q_heads: to_u32(q_heads, "attention query head count")?,
            kv_heads: to_u32(kv_heads, "attention KV head count")?,
            head_dim: to_u32(head_dim, "attention head dimension")?,
            key_count: to_u32(key_count, "attention key count")?,
            scale,
            window: to_u32(window, "attention window")?,
            splits: 0,
            keys_per_split: 0,
        };
        // Resolve the pipeline before opening an encoder. Returning early from
        // inside `set_compute_pipeline_state` used to drop a live compute
        // encoder, which Metal reports as
        // "Command encoder released without endEncoding". The generic kernel is
        // the documented fallback for any other head_dim up to MAX_HEAD_DIM.
        let pipeline = match head_dim {
            256 => &self.online_attention_256_pipeline,
            512 => &self.online_attention_512_pipeline,
            _ => &self.online_attention_pipeline,
        };
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&query_buffer), 0);
        encoder.set_buffer(1, Some(&key_buffer), 0);
        encoder.set_buffer(2, Some(&value_buffer), 0);
        encoder.set_buffer(3, Some(&query_position_buffer), 0);
        encoder.set_buffer(4, Some(&key_position_buffer), 0);
        encoder.set_buffer(5, Some(&output), 0);
        encoder.set_bytes(
            6,
            size_of::<AttentionParams>() as u64,
            &params as *const AttentionParams as *const c_void,
        );
        encoder.dispatch_threads(
            MTLSize::new((query_positions.len() * q_heads) as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(KernelError::CommandFailed);
        }
        read_u16_buffer(&output)
    }

    /// Encode the split-KV decode kernel and its combine pass into a caller-owned
    /// command buffer, reading keys/values straight out of the GPU-resident cache.
    /// Single query, any number of keys.
    ///
    /// Two dispatches, both fixed-size:
    ///
    /// 1. `attention_decode_split_{head_dim,ratio}` on `splits * kv_heads`
    ///    workgroups of `32 * (q_heads / kv_heads)` threads. Each owns a
    ///    contiguous `ceil(key_count / splits)` slice of the live window and
    ///    runs the *prefill* online softmax over it, writing an unnormalised
    ///    `(m, l, O)` partial of `head_dim + 2` floats. No score matrix: the
    ///    only scratch is the double-buffered weight tile in threadgroup memory
    ///    (`2 * ratio * 32` floats), the same as prefill.
    /// 2. `attention_decode_combine_{head_dim}` on `q_heads * head_dim` threads,
    ///    merging the partials with `exp(m_s - M)` rescaling and no threadgroup
    ///    memory at all.
    ///
    /// `partials` must hold `splits * q_heads * (head_dim + 2)` floats, and
    /// `splits <= ATTENTION_SPLITS_MAX`, so a buffer sized by
    /// [`decode_partial_elements`] serves every prompt length forever. The check
    /// below is what proves the caller did that: the old buffer was sized from
    /// the cache capacity captured at construction and silently ran out at 1215
    /// keys of context.
    pub fn encode_online_attention_split(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        request: AttentionEncodeRequest<'_>,
        partials: &Buffer,
    ) -> Result<(), KernelError> {
        let AttentionEncodeRequest {
            queries_count,
            q_heads,
            kv_heads,
            head_dim,
            key_count,
            ..
        } = request;
        if queries_count != 1
            || q_heads == 0
            || kv_heads == 0
            || head_dim == 0
            || key_count == 0
            || q_heads % kv_heads != 0
        {
            return Err(KernelError::InvalidInput(
                "split-key attention is for a single query with valid dimensions".to_string(),
            ));
        }
        let groups = q_heads / kv_heads;
        // No decode specialisation for this shape (an unusual head_dim or GQA
        // ratio). The prefill kernels are correct for a single query and need no
        // scratch, so fall back to them rather than refuse to decode.
        let Some(index) = gqa_index(head_dim, groups) else {
            return self.encode_online_attention(encoder, request);
        };
        // Single-pass arm. Off by default; see
        // [`decode_single_pass_max_keys`]. Checked before the partials size
        // guard because this path does not read the partials buffer at all, and
        // it is the only reason the buffer would be oversized for the shape.
        if key_count <= decode_single_pass_max_keys().unwrap_or(0) {
            if self.decode_single_pipelines[index].is_some() {
                return self.encode_decode_single_pass(encoder, request, index);
            }
        }
        let splits = decode_splits(q_heads, kv_heads, key_count);
        self.encode_decode_split_with_splits(encoder, request, partials, index, splits)
    }

    /// The split+combine pair with an explicit `splits`.
    ///
    /// Split out of [`Self::encode_online_attention_split`] so the split count
    /// is a parameter rather than a side effect of [`decode_splits`]. Production
    /// passes exactly what `decode_splits` returned, so the dispatch is
    /// unchanged; what this buys is a test that can pin `splits == 1` and
    /// compare it against the single-pass kernel on the same inputs, which is
    /// the only way to settle the bit-exactness question on the hardware.
    ///
    /// `index` must be the resolved `gqa_index(head_dim, q_heads / kv_heads)`.
    pub fn encode_decode_split_with_splits(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        request: AttentionEncodeRequest<'_>,
        partials: &Buffer,
        index: usize,
        splits: usize,
    ) -> Result<(), KernelError> {
        let AttentionEncodeRequest {
            q_heads,
            kv_heads,
            head_dim,
            key_count,
            ..
        } = request;
        if splits == 0 || splits > ATTENTION_SPLITS_MAX {
            return Err(KernelError::InvalidInput(format!(
                "attention split count {splits} is outside 1..={ATTENTION_SPLITS_MAX}"
            )));
        }
        let groups = q_heads / kv_heads;
        let keys_per_split = key_count.div_ceil(splits);
        // Repointed at the new buffer: the same "host and kernel must agree"
        // guard as before, but the requirement is now a function of the layer's
        // head count alone and never of how many keys are live.
        let required = splits
            .checked_mul(q_heads)
            .and_then(|rows| rows.checked_mul(head_dim + 2))
            .and_then(|count| count.checked_mul(size_of::<f32>()))
            .ok_or_else(|| {
                KernelError::InvalidInput("attention partial shape overflows".to_string())
            })?;
        if partials.length() < required as u64 {
            return Err(KernelError::InvalidInput(format!(
                "attention partial buffer is {} bytes, needs {required} for {splits} splits x {q_heads} heads x {head_dim} dims",
                partials.length()
            )));
        }
        let params = AttentionParams {
            queries: 1,
            q_heads: to_u32(q_heads, "query heads")?,
            kv_heads: to_u32(kv_heads, "KV heads")?,
            head_dim: to_u32(head_dim, "head dimension")?,
            key_count: to_u32(key_count, "key count")?,
            scale: request.scale,
            window: to_u32(request.window, "window")?,
            splits: to_u32(splits, "split count")?,
            keys_per_split: to_u32(keys_per_split, "keys per split")?,
        };
        let bytes = params_bytes(&params);
        let kv_groups = to_u32(kv_heads, "kv head count")?;
        encoder.dispatch_groups(
            &self.decode_split_pipelines[index],
            &[
                Some(request.queries),
                Some(request.keys),
                Some(request.values),
                Some(request.query_positions),
                Some(request.key_positions),
                Some(partials),
            ],
            bytes,
            (splits as u64) * u64::from(kv_groups),
            (32 * groups) as u64,
        )?;
        let combine = if head_dim == 256 {
            &self.decode_combine_256_pipeline
        } else {
            &self.decode_combine_512_pipeline
        };
        encoder.dispatch_1d(
            combine,
            &[Some(partials), Some(request.output)],
            bytes,
            (q_heads * head_dim) as u64,
            64,
        )
    }

    /// One dispatch, no partials, no combine: a single threadgroup per
    /// (query, kv head) scans the whole live KV window and writes the
    /// normalised output row.
    ///
    /// # Why this is bit-exact with split+combine
    ///
    /// Only for `splits == 1`, and that is not a coincidence -- it is the only
    /// case where the two agree at all. With `splits == 1` the split kernel's
    /// key range is `[0, key_count)`, which is the loop this kernel runs, in the
    /// same tile order, with the same lane->key mapping, the same
    /// `simd_max`/`simd_sum` reductions and the same in-order PV fold. So
    /// `(m, s, O)` are the same *numbers*, not merely the same mathematics --
    /// this is literally `attention_gqa_body<HEAD_DIM, NSG, false>`, the same
    /// template the prefill entry points instantiate, so there is no second copy
    /// of the loop that could drift.
    ///
    /// `attention_decode_combine` then reduces that one partial, which collapses
    /// to `factor = exp(m - m)`, `numerator = O`, `denominator = s` and
    /// `value = denominator > 0 ? numerator / denominator : 0`. Two things had to
    /// be checked rather than assumed, and both are asserted by
    /// `tests/attention_single_pass_bit_exact.rs`:
    ///
    ///   * `exp(0.0f) == 1.0f` exactly on this device, so both factors are
    ///     exactly one and the running sums are exactly `O` and `s`. If it were
    ///     not, the collapse would be inexact and the exactness test would fail
    ///     -- which is the point of running it rather than reasoning about it.
    ///   * the combine's `acc / s` and this kernel's `acc * (1/s)` round to the
    ///     same fp16. They are different f32 operations, and the difference was
    ///     measured at zero over 294912 elements rather than assumed away -- see
    ///     `the_reciprocal_multiply_is_absorbed_by_the_fp16_store` and the
    ///     `DIVIDE` note in `attention.metal`.
    ///
    /// For `splits > 1` no single-pass kernel can match it: the combine's
    /// `sum_i exp(m_i - M) * O_i` interleaves the rescale across splits in a
    /// different association order from the split body's per-tile
    /// `s = s * ms + simd_sum(w)`.
    /// `split_above_one_is_not_bit_exact_but_fp16_absorbs_most_of_it` measures
    /// that deviation rather than asserting it.
    ///
    /// `index` is the resolved `gqa_index(head_dim, groups)`, already checked by
    /// the caller. Public so a test can dispatch it directly rather than through
    /// the environment-variable gate, which is cached per process and would
    /// otherwise make "compare both paths in one run" impossible.
    pub fn encode_decode_single_pass(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        request: AttentionEncodeRequest<'_>,
        index: usize,
    ) -> Result<(), KernelError> {
        let pipeline = self
            .decode_single_pipelines
            .get(index)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                KernelError::InvalidInput(format!(
                    "no single-pass decode kernel for gqa index {index} (head_dim {}, ratio {})",
                    request.head_dim,
                    request.q_heads / request.kv_heads.max(1)
                ))
            })?;
        let AttentionEncodeRequest {
            queries_count,
            q_heads,
            kv_heads,
            head_dim,
            key_count,
            ..
        } = request;
        // Same shape contract as the split path. Repeating it rather than
        // trusting the caller: this is the path that would silently write a
        // normalised row over a buffer the caller sized for something else.
        if queries_count != 1
            || q_heads == 0
            || kv_heads == 0
            || head_dim == 0
            || key_count == 0
            || q_heads % kv_heads != 0
        {
            return Err(KernelError::InvalidInput(
                "single-pass attention is for a single query with valid dimensions".to_string(),
            ));
        }
        let groups = q_heads / kv_heads;
        // `splits == 0` is what selects the `[0, key_count)` range inside the
        // shared body. `keys_per_split` is unused on this path and is left zero
        // so a stale value cannot widen the range.
        let params = AttentionParams {
            queries: 1,
            q_heads: to_u32(q_heads, "query heads")?,
            kv_heads: to_u32(kv_heads, "KV heads")?,
            head_dim: to_u32(head_dim, "head dimension")?,
            key_count: to_u32(key_count, "key count")?,
            scale: request.scale,
            window: to_u32(request.window, "window")?,
            splits: 0,
            keys_per_split: 0,
        };
        debug_assert_eq!(
            gqa_index(head_dim, groups),
            Some(index),
            "single-pass dispatch was given a pipeline for a different shape"
        );
        encoder.dispatch_groups(
            pipeline,
            &[
                Some(request.queries),
                Some(request.keys),
                Some(request.values),
                Some(request.query_positions),
                Some(request.key_positions),
                Some(request.output),
            ],
            params_bytes(&params),
            kv_heads as u64,
            (32 * groups) as u64,
        )
    }
    /// Prefill dispatch with the QK implementation passed in explicitly (T1,
    /// `spec/prefill-120tps`): byte-for-byte the production body of
    /// [`AttentionKernels::encode_online_attention`] below, minus the ablation
    /// gate, plus an [`AttentionQk`] parameter. The explicit parameter is what
    /// lets `attention_qk_mma_matches_scalar` run both paths inside one process
    /// without fighting the `OnceLock` env cache. Grid geometry is identical
    /// for both arms -- only the pipeline differs.
    pub fn encode_online_attention_qk(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        request: AttentionEncodeRequest<'_>,
        qk: AttentionQk,
    ) -> Result<(), KernelError> {
        let AttentionEncodeRequest {
            queries,
            keys,
            values,
            query_positions,
            key_positions,
            output,
            queries_count,
            q_heads,
            kv_heads,
            head_dim,
            key_count,
            scale,
            window,
        } = request;
        if queries_count == 0
            || q_heads == 0
            || kv_heads == 0
            || head_dim == 0
            || key_count == 0
            || q_heads % kv_heads != 0
        {
            return Err(KernelError::InvalidInput(
                "attention dimensions must be non-zero and q_heads divisible by kv_heads"
                    .to_string(),
            ));
        }
        if head_dim > MAX_HEAD_DIM {
            return Err(KernelError::InvalidInput(format!(
                "attention head dimension {head_dim} exceeds {MAX_HEAD_DIM}"
            )));
        }
        let params = AttentionParams {
            queries: to_u32(queries_count, "attention query count")?,
            q_heads: to_u32(q_heads, "attention query head count")?,
            kv_heads: to_u32(kv_heads, "attention KV head count")?,
            head_dim: to_u32(head_dim, "attention head dimension")?,
            key_count: to_u32(key_count, "attention key count")?,
            scale,
            window: to_u32(window, "attention window")?,
            splits: 0,
            keys_per_split: 0,
        };
        // Prefill: one threadgroup per (query, KV HEAD), holding one simdgroup
        // per query head that shares that KV head. `groups` query heads read the
        // same K and V out of one threadgroup's L1, so the cache is re-read
        // `q_heads / groups` = 1 times instead of `q_heads` times -- on Gxma,
        // 2x fewer kv reads on the 50 sliding layers and 8x on the 10 full ones.
        // A `groups` with no specialisation falls back to the 256-thread kernel,
        // which is dimension-partitioned and makes no GQA assumption.
        let groups = q_heads / kv_heads;
        // The generic kernel indexes a flat `queries * q_heads` thread range, so
        // it needs its own group count; both specialised kernels index the grid
        // themselves and discard the surplus threads on an early return.
        let (pipeline, grid_groups, group_threads) = match gqa_index(head_dim, groups) {
            Some(index) => (
                match qk {
                    AttentionQk::Scalar => &self.attention_gqa_pipelines[index],
                    AttentionQk::Mma => &self.attention_gqa_mma_pipelines[index],
                },
                (queries_count * kv_heads) as u64,
                (32 * groups) as u64,
            ),
            None => match head_dim {
                256 => (
                    &self.attention_tg2_256_pipeline,
                    (queries_count * q_heads) as u64,
                    256,
                ),
                512 => (
                    &self.attention_tg2_512_pipeline,
                    (queries_count * q_heads) as u64,
                    256,
                ),
                _ => (
                    &self.online_attention_pipeline,
                    threads_for(queries_count * q_heads, 64),
                    64,
                ),
            },
        };
        encoder.dispatch_groups(
            pipeline,
            &[
                Some(queries),
                Some(keys),
                Some(values),
                Some(query_positions),
                Some(key_positions),
                Some(output),
            ],
            params_bytes(&params),
            grid_groups,
            group_threads,
        )
    }

    /// Production prefill entry point: the ablation gate, then the QK
    /// implementation the environment selected (see [`attention_qk`]).
    ///
    /// Decode does not come through here -- it dispatches through
    /// `encode_online_attention_split` -- so `KAPLUMBA_ABLATE=attention`
    /// remains a prefill-only switch, exactly what
    /// `probe_budget_reconcile::the_attention_ablation_is_a_no_op_on_decode`
    /// asserts.
    pub fn encode_online_attention(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        request: AttentionEncodeRequest<'_>,
    ) -> Result<(), KernelError> {
        if crate::ablated("attention") {
            return Ok(());
        }
        self.encode_online_attention_qk(encoder, request, attention_qk())
    }

    pub fn working_set_bytes(&self, request: &AttentionRequest<'_>) -> usize {
        request.q_heads * request.head_dim * size_of::<f32>()
    }

    fn make_u16_buffer(&self, values: &[u16]) -> Result<Buffer, KernelError> {
        if values.is_empty() {
            return Err(KernelError::InvalidInput(
                "attention u16 input cannot be empty".to_string(),
            ));
        }
        Ok(self.device.new_buffer_with_data(
            values.as_ptr() as *const c_void,
            size_of_val(values) as u64,
            MTLResourceOptions::StorageModeShared,
        ))
    }

    fn make_u32_buffer(&self, values: &[u32]) -> Result<Buffer, KernelError> {
        if values.is_empty() {
            return Err(KernelError::InvalidInput(
                "attention position input cannot be empty".to_string(),
            ));
        }
        Ok(self.device.new_buffer_with_data(
            values.as_ptr() as *const c_void,
            size_of_val(values) as u64,
            MTLResourceOptions::StorageModeShared,
        ))
    }
}

fn params_bytes<T>(params: &T) -> &[u8] {
    let size = std::mem::size_of_val(params);
    unsafe { std::slice::from_raw_parts(params as *const T as *const u8, size) }
}

fn validate_request(request: &AttentionRequest<'_>) -> Result<(), KvAttentionError> {
    if request.query_positions.is_empty()
        || request.key_positions.is_empty()
        || request.q_heads == 0
        || request.kv_heads == 0
        || request.head_dim == 0
        || request.head_dim > MAX_HEAD_DIM
        || !request.q_heads.is_multiple_of(request.kv_heads)
    {
        return Err(KvAttentionError::InvalidDimensions);
    }
    if !request.scale.is_finite() {
        return Err(KvAttentionError::InvalidScale);
    }
    let query_elements = checked_product(
        request.query_positions.len(),
        checked_product(
            request.q_heads,
            request.head_dim,
            KvAttentionError::ShapeOverflow,
        )?,
        KvAttentionError::ShapeOverflow,
    )?;
    let key_elements = checked_product(
        request.key_positions.len(),
        checked_product(
            request.kv_heads,
            request.head_dim,
            KvAttentionError::ShapeOverflow,
        )?,
        KvAttentionError::ShapeOverflow,
    )?;
    if request.queries.len() != query_elements
        || request.keys.len() != key_elements
        || request.values.len() != key_elements
    {
        return Err(KvAttentionError::InvalidBuffer);
    }
    if request
        .query_positions
        .windows(2)
        .any(|positions| positions[0] >= positions[1])
    {
        return Err(KvAttentionError::InvalidQueryPositions);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvAttentionError {
    InvalidDimensions,
    InvalidScale,
    ShapeOverflow,
    InvalidBuffer,
    InvalidQueryPositions,
}

impl From<KvAttentionError> for KernelError {
    fn from(error: KvAttentionError) -> Self {
        Self::InvalidInput(error.to_string())
    }
}

impl std::fmt::Display for KvAttentionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::InvalidDimensions => "attention dimensions are invalid",
            Self::InvalidScale => "attention scale is not finite",
            Self::ShapeOverflow => "attention shape overflows",
            Self::InvalidBuffer => "attention buffer length does not match shape",
            Self::InvalidQueryPositions => "query positions must be strictly increasing",
        };
        formatter.write_str(message)
    }
}

fn checked_product(
    left: usize,
    right: usize,
    error: KvAttentionError,
) -> Result<usize, KvAttentionError> {
    left.checked_mul(right).ok_or(error)
}

fn to_u32(value: usize, name: &str) -> Result<u32, KernelError> {
    u32::try_from(value).map_err(|_| KernelError::InvalidInput(format!("{name} is too large")))
}

fn read_u16_buffer(buffer: &Buffer) -> Result<Vec<u16>, KernelError> {
    let length = buffer.length() as usize;
    if !length.is_multiple_of(size_of::<u16>()) {
        return Err(KernelError::CommandFailed);
    }
    let pointer = buffer.contents() as *const u16;
    if pointer.is_null() {
        return Err(KernelError::CommandFailed);
    }
    Ok(unsafe { std::slice::from_raw_parts(pointer, length / size_of::<u16>()) }.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gxma's two layer shapes, from `config.json`: 50 sliding layers at
    /// `32 q / 16 kv / head_dim 256 / window 1024` and 10 full-attention layers
    /// at `32 q / 4 kv / head_dim 512`. `config.json` is the only source for
    /// these numbers, so they are written out rather than derived.
    const SLIDING_Q: usize = 32;
    const SLIDING_KV: usize = 16;
    const FULL_Q: usize = 32;
    const FULL_KV: usize = 4;

    /// The split count the **one-wave workgroup** rule picks, bypassing the
    /// environment so a test cannot be perturbed by a `KAPLUMBA_DECODE_SPLITS` or
    /// a `KAPLUMBA_DECODE_SPLIT_RULE` in the ambient shell.
    fn chosen(q_heads: usize, kv_heads: usize, key_count: usize) -> usize {
        decode_splits_with(
            q_heads,
            kv_heads,
            key_count,
            None,
            DecodeSplitRule::WorkgroupsPerCore,
        )
    }

    /// The split count the **default** (thread-target) rule picks. Every test
    /// above that quotes a split count is about the workgroup rule, so the
    /// default has to be pinned separately -- see
    /// `the_default_is_the_thread_target_and_the_one_wave_rule_is_opt_in`.
    fn default_chosen(q_heads: usize, kv_heads: usize, key_count: usize) -> usize {
        decode_splits_with(
            q_heads,
            kv_heads,
            key_count,
            None,
            DecodeSplitRule::ThreadTarget,
        )
    }

    /// The rule that shipped before the workgroups-per-core refactor, kept
    /// here so "unchanged for the sliding shape" is a check against the old
    /// arithmetic and not against a number someone typed twice.
    fn legacy_thread_target(q_heads: usize, key_count: usize) -> usize {
        let wanted = ATTENTION_DECODE_THREADS_TARGET / (32 * q_heads.max(1));
        let affordable = key_count.div_ceil(ATTENTION_MIN_KEYS_PER_SPLIT);
        wanted.clamp(1, ATTENTION_SPLITS_MAX).min(affordable).max(1)
    }

    /// **The default must not move.** The one-wave rule is 1.34x faster on the
    /// full-attention layers and it is *not* the default, because it changes the
    /// token stream: the combine pass sums the per-split partials in a different
    /// order and floating-point addition is not associative. So the shipped
    /// default has to evaluate to the old arithmetic for every shape, and this
    /// is where that is asserted rather than intended.
    ///
    /// The reference is `legacy_thread_target`, i.e. the pre-change code's
    /// expression, not a transcribed table.
    #[test]
    fn the_default_is_the_thread_target_and_the_one_wave_rule_is_opt_in() {
        for (q, kv) in [
            (SLIDING_Q, SLIDING_KV),
            (FULL_Q, FULL_KV),
            (40, 5),
            (48, 6),
            (32, 32),
        ] {
            for key_count in [1usize, 64, 65, 512, 1024, 2048, 4096, 8192] {
                assert_eq!(
                    default_chosen(q, kv, key_count),
                    legacy_thread_target(q, key_count),
                    "{q}q/{kv}kv @{key_count}: the DEFAULT rule changed, which changes \
                     the model's output and was not authorised",
                );
            }
        }
        // And the two rules genuinely differ where they are meant to, so this
        // test cannot pass by both being the same number.
        assert_eq!(default_chosen(FULL_Q, FULL_KV, 8192), 16);
        assert_eq!(chosen(FULL_Q, FULL_KV, 8192), 13);
    }

    /// `KAPLUMBA_DECODE_SPLIT_RULE` parsing. Total by construction, and an
    /// unrecognised value is the *default* rather than an error: this is read on
    /// the decode hot path, where a panic takes the process down and where a
    /// typo must not silently change the numerics.
    #[test]
    fn the_split_rule_selector_is_parsed_defensively() {
        // Case-sensitive, like `parse_single_pass_limit` beside it: an
        // environment variable's spelling is part of its interface, and a
        // case-insensitive match would be one more thing to keep in sync.
        for on in [
            "workgroups",
            "waves",
            "one_wave",
            "1",
            "on",
            "true",
            "yes",
            "auto",
            " workgroups ",
        ] {
            assert_eq!(
                parse_split_rule(on),
                DecodeSplitRule::WorkgroupsPerCore,
                "{on:?} should select the one-wave rule",
            );
        }
        for off in [
            "",
            "0",
            "off",
            "false",
            "no",
            "threads",
            "legacy",
            "abc",
            "2",
            "0x1",
            "workgroup",
            "-1",
            "1.5",
            "WORKGROUPS",
            "Workgroups",
        ] {
            assert_eq!(
                parse_split_rule(off),
                DecodeSplitRule::ThreadTarget,
                "{off:?} must fall back to the shipped default",
            );
        }
    }

    /// **The non-regression proof for 83% of the layers.** The sliding shape's
    /// one-wave budget is 288 workgroups, which for 16 kv heads is 18 splits --
    /// above the cap, so the cap decides and the old and new arithmetic must
    /// agree at every key count, not merely at the ones that were measured.
    #[test]
    fn sliding_layers_are_bit_identical_in_dispatch_to_the_legacy_rule() {
        for key_count in [
            1usize, 63, 64, 65, 127, 128, 129, 255, 256, 511, 512, 1023, 1024, 2048, 4096, 8192,
            65536,
        ] {
            assert_eq!(
                chosen(SLIDING_Q, SLIDING_KV, key_count),
                legacy_thread_target(SLIDING_Q, key_count),
                "sliding @{key_count} keys: the rule changed the split count, and \
                 50 of 60 layers depend on it not doing so",
            );
        }
    }

    /// The sliding layers' split count at the key counts a report quotes, plus
    /// the workgroup count those splits produce. At a full 1024-key window,
    /// 16 splits x 16 kv heads = 256 workgroups of 64 threads, which is inside
    /// the 288-workgroup one-wave budget. At 512 keys the *keys-per-split
    /// floor* decides (512/64 = 8), not the wave budget, so only 128
    /// workgroups are dispatched.
    #[test]
    fn sliding_layers_dispatch_one_wave_at_every_context_length() {
        for (key_count, splits) in [(512usize, 8usize), (1024, 16)] {
            assert_eq!(
                chosen(SLIDING_Q, SLIDING_KV, key_count),
                splits,
                "sliding @{key_count}"
            );
            let workgroups = splits * SLIDING_KV;
            assert!(
                workgroups <= decode_one_wave_workgroups(64).expect("64-thread budget"),
                "sliding @{key_count}: {workgroups} workgroups exceeds one wave",
            );
        }
    }

    /// The full-attention layers: `kv_heads 4`, so 16 splits is 64 workgroups of
    /// 256 threads where only 54 fit in one wave -- a second wave for 10 of
    /// them, which measured 1.17-1.34x *slower* than 13 splits for the identical
    /// byte count. The budget puts the grid at 52.
    #[test]
    fn full_layers_fit_one_wave_where_sixteen_splits_did_not() {
        let budget = decode_one_wave_workgroups(256).expect("256-thread budget");
        assert_eq!(budget, 54, "the one-wave budget moved; re-run the sweep");
        for key_count in [1024usize, 2048, 4096, 8192] {
            let splits = chosen(FULL_Q, FULL_KV, key_count);
            let workgroups = splits * FULL_KV;
            assert!(
                workgroups <= budget,
                "full @{key_count}: {workgroups} workgroups is more than one wave",
            );
            assert_ne!(
                splits, 16,
                "full @{key_count}: 16 splits is the whole point of the change",
            );
            assert!(
                workgroups + FULL_KV > budget,
                "full @{key_count}: {splits} splits leaves the wave underfull; \
                 the budget rounds down and this says it rounded down too far",
            );
        }
    }

    /// **The rule is a rule, not a fit to one number.** Three shapes with the
    /// same 256-thread threadgroup and different `kv_heads` were swept
    /// independently, and `budget / kv_heads` has to land on each one's measured
    /// optimum without being told what it was:
    ///
    /// | shape | `kv_heads` | measured best | grid | rule says |
    /// |---|---|---|---|---|
    /// | 32q/4kv/d512 | 4 | 13 splits | 52 | `54 / 4 = 13` |
    /// | 40q/5kv/d512 | 5 | 10 splits | 50 | `54 / 5 = 10` |
    /// | 48q/6kv/d512 | 6 | 9 splits | 54 | `54 / 6 = 9` |
    ///
    /// A budget tuned to make the first row work (say 60, from `20 cores x 3`)
    /// would say 15, 12 and 10 for these, and 15 and 12 are both *measured worse
    /// than the 16 that ships today*. This is the test that would have caught
    /// that.
    #[test]
    fn the_budget_reproduces_the_measured_optimum_of_every_shape_it_was_measured_on() {
        for (q_heads, kv_heads, measured_best_splits) in
            [(32usize, 4usize, 13usize), (40, 5, 10), (48, 6, 9)]
        {
            assert_eq!(
                32 * (q_heads / kv_heads),
                256,
                "{q_heads}q/{kv_heads}kv is not a 256-thread shape",
            );
            assert_eq!(
                chosen(q_heads, kv_heads, 4096),
                measured_best_splits,
                "{q_heads}q/{kv_heads}kv: the rule does not reproduce the measured \
                 optimum, so it is a fit to the 4-kv-head number and nothing else",
            );
        }
    }

    /// The full-attention layers at short context are still bounded by the
    /// keys-per-split floor, not by the wave budget: at 512 keys the shape can
    /// only afford 8 splits however many cores there are.
    #[test]
    fn the_keys_per_split_floor_still_bounds_short_context() {
        for key_count in [1usize, 63, 64, 65, 128, 256, 511, 512] {
            let affordable = key_count.div_ceil(ATTENTION_MIN_KEYS_PER_SPLIT);
            let splits = chosen(FULL_Q, FULL_KV, key_count);
            assert!(
                splits <= affordable.max(1),
                "full @{key_count}: {splits} splits exceeds the {affordable} the \
                 key count can afford",
            );
        }
        assert_eq!(chosen(FULL_Q, FULL_KV, 512), 8);
    }

    /// The full table the report quotes: split count, workgroups, workgroups
    /// per core, for both shapes at every key count. If a constant here moves,
    /// this fails and the note has to be re-measured rather than re-typed.
    #[test]
    fn the_reported_split_table() {
        let expected = [
            // (shape, keys, splits, workgroups)
            ("sliding", 512, 8, 128),
            ("sliding", 1024, 16, 256),
            ("full", 512, 8, 32),
            ("full", 1024, 13, 52),
            ("full", 2048, 13, 52),
            ("full", 4096, 13, 52),
            ("full", 8192, 13, 52),
        ];
        for (name, key_count, splits, workgroups) in expected {
            let (q, kv) = if name == "sliding" {
                (SLIDING_Q, SLIDING_KV)
            } else {
                (FULL_Q, FULL_KV)
            };
            assert_eq!(
                chosen(q, kv, key_count),
                splits,
                "{name} @{key_count}: split count moved",
            );
            assert_eq!(splits * kv, workgroups, "{name} @{key_count}: workgroups");
        }
    }

    /// Everything this returns has to fit the fixed-size partials buffer, for
    /// *any* shape a future config could present -- not just Gxma's two.
    #[test]
    fn the_result_is_always_inside_the_fixed_scratch() {
        for q_heads in [1usize, 2, 4, 8, 16, 32, 40, 64, 128] {
            for kv_heads in [1usize, 2, 4, 8, 16, 32] {
                if q_heads % kv_heads != 0 {
                    continue;
                }
                for key_count in [0usize, 1, 63, 64, 65, 512, 1024, 8192, 1 << 20] {
                    let splits = chosen(q_heads, kv_heads, key_count);
                    assert!(
                        (1..=ATTENTION_SPLITS_MAX).contains(&splits),
                        "{q_heads}q/{kv_heads}kv @{key_count}: {splits} splits is \
                         outside 1..={ATTENTION_SPLITS_MAX}",
                    );
                    // The buffer `decode_partial_elements` sizes, and the guard
                    // in `encode_decode_split_with_splits` enforces.
                    let floats = splits * q_heads * (256 + 2);
                    assert!(
                        floats <= decode_partial_elements(q_heads, 256),
                        "{q_heads}q/{kv_heads}kv @{key_count}: {floats} floats \
                         exceed the scratch",
                    );
                }
            }
        }
    }

    /// Degenerate inputs must not panic or divide by zero. `kv_heads == 0` is
    /// unreachable from the encode path (it rejects the request first) but the
    /// arithmetic is on the hot path and a panic there is a process death. With
    /// `kv_heads == 0` the result is whatever the legacy thread target says --
    /// finite, in range, and irrelevant because such a request never gets this
    /// far.
    #[test]
    fn degenerate_head_counts_do_not_panic() {
        assert_eq!(chosen(0, 0, 8192), 1);
        assert_eq!(chosen(0, 4, 8192), 1);
        assert_eq!(chosen(32, 4, 0), 1);
        assert_eq!(chosen(32, 0, 8192), 16);
        assert!((1..=ATTENTION_SPLITS_MAX).contains(&chosen(32, 0, 8192)));
    }

    /// A threadgroup width with no measured budget keeps the old arithmetic
    /// rather than guessing one, **even with the one-wave rule selected**.
    /// `kv_heads == q_heads` gives a 32-thread threadgroup, which is not in the
    /// table. Note that 64q/8kv is *also* a 256-thread threadgroup and therefore
    /// measured, which is why it is absent.
    #[test]
    fn an_unmeasured_threadgroup_width_keeps_the_legacy_thread_target() {
        assert_eq!(decode_one_wave_workgroups(32), None);
        assert_eq!(decode_one_wave_workgroups(128), None);
        assert_eq!(decode_one_wave_workgroups(512), None);
        for (q, kv) in [(32usize, 32usize), (8, 1), (4, 1), (16, 2), (64, 64)] {
            for key_count in [64usize, 512, 1024, 8192] {
                assert_eq!(
                    chosen(q, kv, key_count),
                    legacy_thread_target(q, key_count),
                    "{q}q/{kv}kv @{key_count}: unmeasured shape changed under the \
                     one-wave rule",
                );
            }
        }
    }

    /// The pinned sweep, which is how the constants were measured. Parsing is
    /// tested as a function rather than through the environment, which Rust 2024
    /// makes `unsafe` and which every other test in this process shares.
    #[test]
    fn the_pinned_split_value_is_parsed_defensively() {
        // `+4` and `4\n` are accepted because `usize::from_str` accepts a
        // leading `+` and the value is trimmed. Both are in range, so pinning
        // them is harmless; what must never happen is a value that silently
        // becomes the *heuristic*'s number by accident, which the range check
        // rules out.
        for good in ["1", "16", " 8 ", "4\n", "+4"] {
            assert_eq!(
                parse_pinned_splits(good),
                good.trim().parse::<usize>().ok(),
                "{good:?}",
            );
        }
        for bad in [
            "",
            "0",
            "17",
            "-1",
            "abc",
            "1.5",
            "99999999999999999999",
            "0x4",
            "4 4",
        ] {
            assert_eq!(parse_pinned_splits(bad), None, "{bad:?} must not pin");
        }
        // And a pinned value wins over the heuristic, in both directions, so
        // the sweep can compare 16 (worse) as well as the new value.
        assert_eq!(
            decode_splits_with(
                FULL_Q,
                FULL_KV,
                8192,
                Some(16),
                DecodeSplitRule::ThreadTarget
            ),
            16
        );
        assert_eq!(
            decode_splits_with(
                FULL_Q,
                FULL_KV,
                8192,
                Some(4),
                DecodeSplitRule::ThreadTarget
            ),
            4
        );
        assert_eq!(
            decode_splits_with(FULL_Q, FULL_KV, 64, Some(16), DecodeSplitRule::ThreadTarget),
            16
        );
    }
}
