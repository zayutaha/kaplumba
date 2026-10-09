use std::ffi::c_void;
use std::mem::{size_of, size_of_val};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    Library, MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

use crate::encoder::TensorRef;
use crate::exl3::{Exl3Codebook, Exl3Rate, KernelError};
use crate::pipeline_audit;

const SOURCE: &str = include_str!("linear.metal");

/// Repacked column-major / fused / split-K GEMV kernels.
///
/// Kept out of `linear.metal` because that file is partly generated:
/// `scripts/gen_exl3_gemv.py` rewrites everything between
/// `linear_gemv_exl3_k1(` and `kernel void linear_gemv(`, which is where these
/// kernels used to sit. Concatenated onto `SOURCE` so they still share one
/// `decode_state` and one compilation rather than becoming a second library.
const REPACKED_SOURCE: &str = include_str!("linear_repacked.metal");

/// `spec/decode-20tok-crux3`: the decode-algebra arms. Three instantiations of
/// one macro, differing only in the body of the weight extract, dispatched only
/// by `examples/crux3_decode_algebra.rs`. Nothing in this crate selects them, and
/// arms B and C produce deliberately wrong values on purpose — see the file's
/// header. Kept out of `linear_repacked.metal` because a transcription error in a
/// hand-copied kernel body is exactly the failure this probe must not have, and
/// `scripts/crux3_armA_equiv.py` proves arm A against the production source.
const CRUX3_SOURCE: &str = include_str!("crux3_decode_algebra.metal");

fn combined_source() -> String {
    let mut combined =
        String::with_capacity(SOURCE.len() + REPACKED_SOURCE.len() + CRUX3_SOURCE.len() + 2);
    combined.push_str(SOURCE);
    combined.push('\n');
    combined.push_str(REPACKED_SOURCE);
    combined.push('\n');
    combined.push_str(CRUX3_SOURCE);
    combined
}

/// Length of the decode-matvec pipeline table: index 0 is unused, then one slot
/// per `(codebook, aligned, bits)` triple -- 3 x 2 x 8 = 48, plus the unused
/// index 0. See [`LinearKernels::exl3_gemv_index`] for the layout.
const EXL3_GEMV_STATES: usize = 1 + 3 * 2 * 8;

/// Cached "is this switch set?" lookups.
///
/// These used to be live `std::env::var_os` calls on the hot path. `var_os` takes
/// the environment lock and linear-scans `environ`, and decode read several of
/// them once per matvec: seven matvecs x sixty layers is 420 calls per generated
/// token, ~840 including the element and model layers. A switch is set before the
/// process starts or not at all, so reading it once is behaviour-preserving; no
/// test mutates `env::set_var` mid-process.
fn env_flag(name: &str) -> bool {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, bool>>> =
        std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    // A poisoned lock only means some other thread panicked during a lookup; the
    // values in the map are still correct.
    let mut cache = cache.lock().unwrap_or_else(|error| error.into_inner());
    if let Some(value) = cache.get(name) {
        return *value;
    }
    let value = std::env::var_os(name).is_some();
    cache.insert(name.to_string(), value);
    value
}

/// GEMM-MMA prefill variant ids, see [`force_gemm_mma_variant`].
///
/// The env-flag arm reads `KAPLUMBA_MMA_4SG` / `KAPLUMBA_REGPIPE` /
/// `KAPLUMBA_REGPIPE_K32` through `env_flag`, which MEMOISES on first read --
/// by design (see [`env_flag`]). A test process therefore cannot flip variants
/// by setting variables mid-run: the first dispatch caches the answer. These
/// ids plus the atomic override below exist so a test can A/B every variant in
/// ONE process without touching the environment at all.
pub const GEMM_MMA_VARIANT_AUTO: u8 = 0;
/// `linear_gemm_mma`, the shipped 128x64 / 256-thread tile.
pub const GEMM_MMA_VARIANT_128X64: u8 = 1;
/// `linear_gemm_mma_4sg`, the 128x32 / 4-simdgroup research tile.
pub const GEMM_MMA_VARIANT_4SG: u8 = 2;
/// `linear_gemm_mma_regpipe`.
pub const GEMM_MMA_VARIANT_REGPIPE: u8 = 3;
/// `linear_gemm_mma_regpipe_k32`.
pub const GEMM_MMA_VARIANT_REGPIPE_K32: u8 = 4;

static GEMM_MMA_VARIANT_FORCED: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(GEMM_MMA_VARIANT_AUTO);

/// Force the GEMM-MMA prefill variant for subsequent [`LinearKernels::encode_gemm`]
/// calls in this process, overriding the env flags. `GEMM_MMA_VARIANT_AUTO`
/// restores env selection.
///
/// Test hook: the env flags above are process-cached, and one test binary
/// compares variants inside a single `#[test]`. Variants 2, 3 and 4 share the
/// production eligibility gate (`perm_mode != 1 && rate == Integer(3) &&
/// codebook == Mcg`) -- forcing them elsewhere falls back to 128X64 rather than
/// running a kernel on a shape its host gate would never send it.
pub fn force_gemm_mma_variant(variant: u8) {
    GEMM_MMA_VARIANT_FORCED.store(variant, std::sync::atomic::Ordering::Relaxed);
}

/// See [`force_gemm_mma_variant`]: back to env-flag selection.
pub fn clear_gemm_mma_variant() {
    GEMM_MMA_VARIANT_FORCED.store(GEMM_MMA_VARIANT_AUTO, std::sync::atomic::Ordering::Relaxed);
}

/// [`last_gemm_mma_variant`] reports this when no MMA prefill dispatch has
/// happened yet in this process (or the last prefill fell through to the
/// scalar tiled kernel, which does not touch the counter).
pub const GEMM_MMA_VARIANT_NONE: u8 = u8::MAX;

static GEMM_MMA_LAST_DISPATCHED: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(GEMM_MMA_VARIANT_NONE);

/// The variant id `encode_gemm` last ACTUALLY dispatched (after the
/// eligibility fallback, so a forced regpipe on an ineligible shape reports
/// `GEMM_MMA_VARIANT_128X64`, which is what ran).
///
/// Test hook, and a load-bearing one: every T4 variant computes bit-identical
/// output to `linear_gemm_mma`, so output comparison alone cannot tell "the
/// 4sg kernel ran and agreed" from "the dispatch never left the baseline" --
/// the vacuous pass that shipped `8f1a863` (AGENTS.md §3, Q11). A test that
/// asserts outputs must assert this next to them.
pub fn last_gemm_mma_variant() -> u8 {
    GEMM_MMA_LAST_DISPATCHED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Output tile-columns per threadgroup for the decode matvec: the NR0 axis.
///
/// The kernel reads a fixed three ring words per lane whatever the rate is, so a
/// lane spends 12 B of load traffic on `8*KA` bits of codes. At 3 bpw -- 410 of
/// the 411 tensors in this checkpoint -- that leaves the loop short of distinct
/// bytes in flight, and the measured 3 bpw asymptote is 168.6 GB/s against a
/// 240 GB/s ceiling. Giving a thread NR0 output tile-columns puts NR0
/// independent streams in flight without changing which bytes are read, and the
/// accumulator and reduction are unchanged, so the result is bit-exact.
///
/// `KAPLUMBA_DECODE_NR0` in {1, 2, 4, 8}; anything else, including unset, is 1.
/// Unset must stay 1: the default is what the reference prompts were verified
/// against, and a default that silently changes the decode kernel would make
/// every "the five reference prompts are exact" claim in this repository a claim
/// about whichever machine ran it. Read once, like every other switch here.
fn decode_nr0() -> u32 {
    static CACHE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        match std::env::var("KAPLUMBA_DECODE_NR0")
            .ok()
            .and_then(|raw| raw.trim().parse::<u32>().ok())
        {
            Some(value) if NR0_VARIANTS.contains(&value) => value,
            _ => 1,
        }
    })
}

/// The NR0 values a kernel name exists for, excluding 1 (which is the unsuffixed
/// `linear_gemv_exl3_kN_mcg`). Must match `NR0_VALUES` in
/// `scripts/gen_exl3_gemv.py`; a test asserts every name here resolves, which is
/// what catches the two drifting apart.
pub const NR0_VARIANTS: [u32; 3] = [2, 4, 8];
/// `KAPLUMBA_GEMV_MAX_GROUPS`, read once. `u64::MAX` (no cap) when unset.
fn gemv_max_groups() -> u64 {
    static CAP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("KAPLUMBA_GEMV_MAX_GROUPS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(u64::MAX)
    })
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FinalizeNormScaleParams {
    pub width: u32,
    pub blocks: u32,
    pub eps: f32,
    pub scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PostAttnPreFfnParams {
    pub width: u32,
    pub blocks: u32,
    pub eps: f32,
}

/// T3: the cross-layer fusion's parameters. Byte-identical in layout to
/// [`FinalizeNormScaleParams`], and that is deliberate — the layer-L half of the
/// fused kernel is the `_sk2` finalize, so reusing the struct makes it harder to
/// change one field in the fused kernel and not the other.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CrossLayerParams {
    pub width: u32,
    pub blocks: u32,
    pub eps: f32,
    pub scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LinearParams {
    in_features: u32,
    out_features: u32,
    tile_columns: u32,
    packed_words: u32,
    rate: u32,
    codebook: u32,
    queries: u32,
    perm_mode: u32,
}

#[derive(Clone, Copy)]
pub struct LinearRequest<'a> {
    pub packed: &'a [u16],
    pub input: &'a [u16],
    pub input_signs: &'a [u16],
    pub output_signs: &'a [u16],
    pub in_features: usize,
    pub out_features: usize,
    pub rate: Exl3Rate,
    pub codebook: Exl3Codebook,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PrefillRowParams {
    in_features: u32,
    out_features: u32,
    queries: u32,
    perm_mode: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearPrefillStats {
    pub query_chunks: usize,
    pub decoded_tiles: usize,
    pub weight_applications: usize,
    pub dispatch_groups: usize,
}

#[derive(Clone, Copy)]
pub struct PrefillRequest<'a> {
    pub packed: &'a [u16],
    pub input: &'a [u16],
    pub input_signs: &'a [u16],
    pub output_signs: &'a [u16],
    pub queries: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub rate: Exl3Rate,
    pub codebook: Exl3Codebook,
}

#[derive(Clone, Copy)]
pub struct LinearBufferRequest<'a> {
    pub packed: &'a Buffer,
    /// Byte offset of the tensor inside `packed` (non-zero for zero-copy weights).
    pub packed_offset: u64,
    pub input: &'a [u16],
    pub input_signs: &'a [u16],
    pub output_signs: &'a [u16],
    pub in_features: usize,
    pub out_features: usize,
    pub rate: Exl3Rate,
    pub codebook: Exl3Codebook,
}

/// Fully GPU-resident single-query (decode) linear request.
///
/// Every buffer is supplied by the caller so a decoder layer can be encoded
/// into one command buffer with no allocation and no host round trip.
pub struct LinearEncodeRequest<'a> {
    pub packed: &'a Buffer,
    /// Byte offset of the tensor inside `packed`.
    ///
    /// Zero-copy weights are bound over a whole shard mapping, so the tensor
    /// starts at a non-zero offset that is not Metal-aligned.
    pub packed_offset: u64,
    /// Tile permutation selector; see `permutation_index` in linear.metal.
    pub perm_mode: u32,
    pub input: &'a Buffer,
    pub input_signs: &'a Buffer,
    pub output_signs: &'a Buffer,
    pub transformed: &'a Buffer,
    pub raw: &'a Buffer,
    pub output: &'a Buffer,
    pub in_features: usize,
    pub out_features: usize,
    pub rate: Exl3Rate,
    pub codebook: Exl3Codebook,
    /// Set when `transformed` already holds this request's input transform, so
    /// the dispatch can be hoisted out and shared by matvecs that read the same
    /// activation. q/k/v and gate/up each share one input, which turns 7
    /// transforms per layer into 4.
    pub pretransformed: bool,
}

/// Fully GPU-resident multi-query (prefill) linear request.
pub struct PrefillEncodeRequest<'a> {
    pub packed: &'a Buffer,
    pub packed_offset: u64,
    pub perm_mode: u32,
    pub input: &'a Buffer,
    pub input_signs: &'a Buffer,
    pub output_signs: &'a Buffer,
    pub transformed: &'a Buffer,
    pub raw: &'a Buffer,
    pub output: &'a Buffer,
    pub queries: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub rate: Exl3Rate,
    pub codebook: Exl3Codebook,
}

/// Every `linear_gemv_ishizuki_*` function in the library, sorted.
///
/// Sorted so the pipeline build order — and therefore any error a missing kernel
/// raises — is the same on every run. `functionNames` comes back from the
/// Objective-C runtime in whatever order the compiler emitted, which is not
/// stable across source edits, and an A/B that changes which kernel it measured
/// when a header was touched is not an A/B.
fn ishizuki_kernel_names(library: &Library) -> Vec<String> {
    let mut names: Vec<String> = library
        .function_names()
        .into_iter()
        .filter(|name| name.starts_with("linear_gemv_ishizuki_"))
        .collect();
    names.sort();
    names
}

/// Names of the kernels in `linear_repacked.metal` that this module indexes
/// separately from the rest of the ishizuki family.
///
/// **This currently matches nothing, and that is a fact about the source rather
/// than about the filter.** The file's only `linear_gemv_repacked_` occurrence
/// is a comment; every kernel it actually defines is named
/// `linear_gemv_ishizuki_k3_*` (`_colmajor`, `_colmajor_simd`, `_fused2`,
/// `_fused2_sk2`, `_fused3`, `_fused3_sk2`, `_colmajor_stream`, `_colmajor_sk4`,
/// `_colmajor_contig4`, `_colmajor_contig4_sk2`, `_colmajor_contig4_sk4`,
/// `_colmajor_u4`), so `ishizuki_kernel_names` above already collects and
/// builds all of them.
///
/// The map is left in place because it is a dispatch input (see
/// [`Self::encode_gemv`]'s `trace_gemv`, which chains it), and removing it is a
/// behaviour change to a tracing path that is not this task's business. But the
/// honest reading is: the repacked family is reached through
/// `ishizuki_gemv_by_name`, and a name lookup in `repacked_gemv_by_name` always
/// returns `None`.
fn repacked_kernel_names(library: &Library) -> Vec<String> {
    let mut names: Vec<String> = library
        .function_names()
        .into_iter()
        .filter(|name| name.starts_with("linear_gemv_repacked_"))
        .collect();
    names.sort();
    names
}

pub struct LinearKernels {
    device: Device,
    queue: CommandQueue,
    _library: Library,
    input_pipeline: ComputePipelineState,
    gemv_pipeline: ComputePipelineState,
    finalize_pipeline: ComputePipelineState,
    pub finalize_simd_pipeline: ComputePipelineState,
    pub finalize_simd_sk2_pipeline: ComputePipelineState,
    pub finalize_simd_sk4_pipeline: ComputePipelineState,
    finalize_simd3_pipeline: ComputePipelineState,
    finalize_simd3_sk2_pipeline: ComputePipelineState,
    finalize_gated_gelu_pipeline: ComputePipelineState,
    finalize_gated_gelu_sk2_pipeline: ComputePipelineState,
    finalize_gated_gelu_and_down_input_pipeline: ComputePipelineState,
    finalize_gated_gelu_and_down_input_sk2_pipeline: ComputePipelineState,
    pub finalize_and_post_ffn_norm_scale_pipeline: ComputePipelineState,
    pub finalize_and_post_ffn_norm_scale_sk2_pipeline: ComputePipelineState,
    pub finalize_and_post_ffn_norm_scale_sk4_pipeline: ComputePipelineState,
    pub post_attn_pre_ffn_transform_pipeline: ComputePipelineState,
    pub finalize_o_proj_and_pre_ffn_transform_pipeline: ComputePipelineState,
    pub finalize_o_proj_and_pre_ffn_transform_sk2_pipeline: ComputePipelineState,
    pub finalize_o_proj_and_pre_ffn_transform_sk4_pipeline: ComputePipelineState,
    pub input_norm_and_qkv_transform_pipeline: ComputePipelineState,
    /// T3: layer L's finalize+post_ffn_norm+residual fused with layer L+1's
    /// input_layernorm+q_input_transform. One dispatch, one reduction tree per
    /// norm, both reproduced exactly from the kernels they replace.
    cross_layer_pipeline: ComputePipelineState,
    input_rows_pipeline: ComputePipelineState,
    gemv_tiled_pipeline: ComputePipelineState,
    /// Tensor-core prefill GEMM (`linear_gemm_mma`): dequantize the EXL3 weights
    /// of a 64-wide k chunk to threadgroup halfs, then run 128 queries x 64
    /// outputs through 256 threads with sixteen named accumulator chains per
    /// simdgroup. Needs `in_features` a multiple of 64 and `out_features` a
    /// multiple of 64; the engine's are multiples of 128.
    gemm_mma_pipeline: ComputePipelineState,
    /// RESEARCH VARIANT of `gemm_mma_pipeline` (`linear_gemm_mma_regpipe`):
    /// same tile, same accumulators, same bit-identical MCG decode, but the
    /// dequantised weights go to PER-THREAD REGISTERS for the next k chunk while
    /// the current chunk's MMAs run, then get published through the same
    /// `bstage`. One barrier per chunk instead of two, `bstage` unchanged at
    /// 9216 B. Only valid for `rate == Integer(3) && codebook == Mcg`, the
    /// folded-MCG arm; every other rate keeps `gemm_mma_pipeline`.
    gemm_mma_regpipe_pipeline: ComputePipelineState,
    /// RESEARCH VARIANT for T4 (`linear_gemm_mma_4sg`): the same KCHUNK=64 and
    /// the same sixteen accumulators per thread, but a 128q x 32n tile on FOUR
    /// simdgroups instead of 128q x 64n on eight -- 4 row x 4 column
    /// accumulators, 8 fragment loads per 16 MMAs (0.5) instead of 10 (0.625),
    /// and `bstage` at 5120 B instead of 9216 B. Bit-identical output to
    /// `gemm_mma_pipeline`; selected by `KAPLUMBA_MMA_4SG` or the test hook
    /// [`force_gemm_mma_variant`].
    gemm_mma_4sg_pipeline: ComputePipelineState,
    /// RESEARCH VARIANT for T4 (`linear_gemm_mma_regpipe_k32`): `regpipe` with
    /// KCHUNK=32, so `bstage` drops to 4608 B and the prefetch stream runs
    /// twice as often. KCHUNK=32 measured -2% against the ORIGINAL kernel;
    /// this re-measures it behind regpipe, which moved the dequant into the
    /// MMA basic block. Selected by `KAPLUMBA_REGPIPE_K32` or the test hook.
    gemm_mma_regpipe_k32_pipeline: ComputePipelineState,
    mrhs_repacked_m4: ComputePipelineState,
    mrhs_repacked_m8: ComputePipelineState,
    mrhs_repacked_m16: ComputePipelineState,
    mrhs_repacked_m32: ComputePipelineState,
    mrhs_exl3_m4: ComputePipelineState,
    mrhs_exl3_m8: ComputePipelineState,
    mrhs_exl3_m16: ComputePipelineState,
    mrhs_exl3_m32: ComputePipelineState,
    finalize_rows_pipeline: ComputePipelineState,
    zero_raw_pipeline: ComputePipelineState,
    gemv_reduce_sk2_pipeline: Option<ComputePipelineState>,
    gemv_reduce_sk4_pipeline: Option<ComputePipelineState>,
    /// `linear_gemv_fast` specialised on (bits, MCG codebook) for integer rates.
    fast_gemv_pipelines: Vec<Option<ComputePipelineState>>,
    /// Lane-per-thread EXL3 matvec, one per integer rate so `KA` folds to a
    /// constant; see `linear_gemv_exl3_k*` in linear.metal. Indexed by
    /// [codebook][aligned][bits]; see [`Self::exl3_gemv_index`].
    exl3_gemv_pipelines: Vec<Option<ComputePipelineState>>,
    /// The same pipelines keyed by kernel name, so a test can dispatch a named
    /// kernel and A/B two of them inside one process. Shares every
    /// `MTLComputePipelineState` with `exl3_gemv_pipelines`; the map holds
    /// clones, not rebuilds, so this costs no extra pipeline compilation.
    exl3_gemv_by_name: std::collections::HashMap<String, ComputePipelineState>,
    /// The kernel name each slot of `exl3_gemv_pipelines` was built from, in the
    /// same order. Recorded so a test can assert the INDEX selects the kernel it
    /// was built with; see [`Self::exl3_gemv_kernel_name`].
    exl3_gemv_slot_names: Vec<String>,
    /// Verbatim ishizuki `exl3_matvec`, for A/B only.
    ishizuki_gemv_pipelines: Vec<Option<ComputePipelineState>>,
    /// Every `linear_gemv_ishizuki_*` kernel, by name.
    ///
    /// This family is the one that actually runs: `encode_gemv` tries it *first*,
    /// and 410 of this checkpoint's 411 tensors are 4-byte aligned, so all 60
    /// layers' projections dispatch here. That makes two things a prerequisite
    /// rather than a nicety, and both need to name a kernel in this family:
    ///
    ///   * a bit-exactness **reference** (`tests/decode_matvec_golden.rs`), which
    ///     until now covered only `_mcg` -- i.e. it could not detect a change to
    ///     the kernel that carries 91% of a decode token's bytes;
    ///   * an A/B of a **variant** of the live kernel, which has to be
    ///     name-addressable for the same reason the EXL3 r-variants are.
    ///
    /// A `HashMap` rather than extending the indexed `Vec`, because these
    /// kernels are selected by *rate*, not by a `(codebook, aligned, bits)` slot,
    /// and the variants under test have no slot.
    ishizuki_gemv_by_name: std::collections::HashMap<String, ComputePipelineState>,
    repacked_gemv_by_name: std::collections::HashMap<String, ComputePipelineState>,
}

impl LinearKernels {
    /// The bit rate a `linear_gemv_ishizuki_k<N>[...]` name dispatches for.
    ///
    /// Parsed from the name so the rate table and the name map cannot disagree —
    /// the same reason `nr0_of_kernel_name` exists for the EXL3 r-variants, and
    /// for the same reason: the alternative is an env var read at dispatch time,
    /// which is how a benchmark ends up measuring a kernel the engine would not
    /// choose.
    ///
    /// A suffixed variant (`_k3_w2`) reports the same rate as `_k3`. That is what
    /// makes a variant substitutable: it must be dispatchable wherever the
    /// original is, with only the window arithmetic changed.
    pub fn ishizuki_rate_of_name(name: &str) -> Option<u8> {
        let rest = name.strip_prefix("linear_gemv_ishizuki_k")?;
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            return None;
        }
        let bits: u8 = digits.parse().ok()?;
        // 1..=8, and the rate must be the *whole* numeric prefix: `_k34` is not a
        // thing, and silently reading it as 3 would be a naming trap.
        (1..=8).contains(&bits).then_some(bits)
    }

    /// The process-wide instance, built on first use.
    ///
    /// Every `DecoderLayer` used to call [`Self::new`], so a 60-layer model ran
    /// 61 `new_library_with_source` calls on the same `include_str!`'d source and
    /// rebuilt the same ~70 pipeline states 61 times over -- a large one-off load
    /// cost, and a reason every layer's pipeline object was a *distinct*
    /// `MTLComputePipelineState` for identical code. No method on this type takes
    /// `&mut self`, so one shared instance is safe for every layer.
    pub fn shared() -> Result<&'static Self, KernelError> {
        static INSTANCE: std::sync::OnceLock<Result<LinearKernels, KernelError>> =
            std::sync::OnceLock::new();
        match INSTANCE.get_or_init(Self::new) {
            Ok(kernels) => Ok(kernels),
            // A `&'static Self` cannot be produced from an `Err`, and
            // `KernelError` is not `Clone`, so re-derive it.
            Err(error) => Err(crate::elements::clone_kernel_error(error)),
        }
    }

    pub fn new() -> Result<Self, KernelError> {
        let device = Device::system_default().ok_or(KernelError::DeviceUnavailable)?;
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(&combined_source(), &options)
            .map_err(KernelError::Shader)?;
        // Every pipeline below goes through `pipeline_audit::build`, never
        // `new_compute_pipeline_state_with_function` directly. The audit is
        // what makes "this family compiles once, at load" a measurement rather
        // than a reading of the code, and it is only worth what its coverage
        // is -- so `pipeline_audit`'s own test fails if a construction site
        // here ever bypasses it.
        //
        // Which means the eager build below is load-bearing, not tidiness: a
        // kernel reached for the first time inside a timed loop charges Metal's
        // shader compilation to that loop's number.
        let input_pipeline = pipeline_audit::build(&device, &library, "linear_input_transform")?;
        let mut ishizuki_gemv_pipelines = Vec::with_capacity(9);
        let mut ishizuki_gemv_by_name = std::collections::HashMap::new();
        ishizuki_gemv_pipelines.push(None);
        // Every `linear_gemv_ishizuki_*` in the library, not just the eight the
        // rate table dispatches. The extras are variants kept for A/B and for
        // the golden reference; a name that is absent fails at `shared()`, which
        // is where a typo should die rather than at a dispatch 20 minutes into a
        // benchmark.
        //
        // This loop also covers the repacked file: `combined_source()` splices
        // `linear_repacked.metal` into the same library and every kernel there
        // is named `linear_gemv_ishizuki_k3_*`, so the `_colmajor*`, `_fused*`,
        // `_contig4*` and `_u4` variants are built here rather than by
        // `repacked_kernel_names` -- which matches `linear_gemv_repacked_` and so
        // finds nothing at all. See `repacked_gemv_by_name` below.
        for name in ishizuki_kernel_names(&library) {
            let pipeline = pipeline_audit::build(&device, &library, &name)?;
            if let Some(bits) = Self::ishizuki_rate_of_name(&name) {
                // The first (unsuffixed) kernel for a rate owns the dispatch slot.
                if ishizuki_gemv_pipelines.len() <= bits as usize {
                    ishizuki_gemv_pipelines.resize(bits as usize + 1, None);
                }
                if ishizuki_gemv_pipelines[bits as usize].is_none() {
                    ishizuki_gemv_pipelines[bits as usize] = Some(pipeline.clone());
                }
            }
            ishizuki_gemv_by_name.insert(name, pipeline);
        }

        let mut repacked_gemv_by_name = std::collections::HashMap::new();
        for name in repacked_kernel_names(&library) {
            let pipeline = pipeline_audit::build(&device, &library, &name)?;
            repacked_gemv_by_name.insert(name, pipeline);
        }

        let mut exl3_gemv_pipelines = Vec::with_capacity(EXL3_GEMV_STATES);
        exl3_gemv_pipelines.push(None); // index 0 unused
        let mut exl3_gemv_by_name = std::collections::HashMap::new();
        let mut exl3_gemv_slot_names: Vec<String> = vec![String::new()]; // index 0
        // Index by [codebook][aligned][bits]; see `exl3_gemv_index`. The MCG
        // family is emitted with the codebook folded to a compile-time constant
        // (`_mcg` suffix), which is the only family the engine can dispatch; the
        // plain and mul1 families keep the runtime-codebook form.
        for codebook in [Exl3Codebook::Mcg, Exl3Codebook::Plain, Exl3Codebook::Mul1] {
            let family = if codebook == Exl3Codebook::Mcg {
                "_mcg"
            } else {
                ""
            };
            for aligned in [false, true] {
                for bits in 1..=8_u32 {
                    let suffix = if aligned { "" } else { "_u" };
                    let name = format!("linear_gemv_exl3_k{bits}{suffix}{family}");
                    let pipeline = pipeline_audit::build(&device, &library, &name)?;
                    exl3_gemv_by_name.insert(name.clone(), pipeline.clone());
                    exl3_gemv_slot_names.push(name);
                    exl3_gemv_pipelines.push(Some(pipeline));
                }
            }
        }
        debug_assert_eq!(exl3_gemv_pipelines.len(), EXL3_GEMV_STATES);
        debug_assert_eq!(exl3_gemv_slot_names.len(), EXL3_GEMV_STATES);
        // The NR0 family (`_mcg_r2` / `_r4` / `_r8`) is addressable by name but
        // deliberately NOT in `exl3_gemv_pipelines`: that table is indexed by
        // [codebook][aligned][bits] and `exl3_gemv_index` is what the engine
        // dispatches through, so adding a second kernel per slot would make the
        // index ambiguous. These are A/B-only, reached through
        // `encode_gemv_named` and `KAPLUMBA_DECODE_NR0`, and every one of them
        // is built here so that a name which does not compile is a startup error
        // rather than a silent `None` at dispatch time.
        //
        // `Option` because a build that drops the generated family (an older
        // `linear.metal`, a stale checkout) must still start: the engine's
        // default is NR0 = 1, which is the unsuffixed kernel and is always
        // present.
        //
        // The `is_ok()` on the pipeline is deliberate and is the one place the
        // audit can miss a build: `build` records *after* a successful
        // construction, so an NR0 kernel that fails to compile leaves no trace.
        // That is acceptable precisely because the fallback is the unsuffixed
        // kernel, which the table above did record.
        for bits in 1..=8_u32 {
            for nr0 in NR0_VARIANTS {
                let name = format!("linear_gemv_exl3_k{bits}_mcg_r{nr0}");
                // `get_function` first, so a build without the generated family
                // is a quiet skip rather than a `KernelError::Shader`.
                if library.get_function(&name, None).is_ok()
                    && let Ok(pipeline) = pipeline_audit::build(&device, &library, &name)
                {
                    exl3_gemv_by_name.insert(name, pipeline);
                }
            }
        }

        let gemv_pipeline = pipeline_audit::build(&device, &library, "linear_gemv")?;
        let finalize_pipeline = pipeline_audit::build(&device, &library, "linear_finalize")?;
        let finalize_simd_pipeline =
            pipeline_audit::build(&device, &library, "linear_finalize_simd")?;
        let finalize_simd_sk2_pipeline =
            pipeline_audit::build(&device, &library, "linear_finalize_simd_sk2")?;
        let finalize_simd_sk4_pipeline =
            pipeline_audit::build(&device, &library, "linear_finalize_simd_sk4")?;
        let finalize_simd3_pipeline =
            pipeline_audit::build(&device, &library, "linear_finalize_simd3")?;
        let finalize_gated_gelu_pipeline =
            pipeline_audit::build(&device, &library, "linear_finalize_gated_gelu_simd")?;
        let finalize_gated_gelu_sk2_pipeline =
            pipeline_audit::build(&device, &library, "linear_finalize_gated_gelu_simd_sk2")?;
        let finalize_simd3_sk2_pipeline =
            pipeline_audit::build(&device, &library, "linear_finalize_simd3_sk2")?;
        let finalize_gated_gelu_and_down_input_pipeline = pipeline_audit::build(
            &device,
            &library,
            "linear_finalize_gated_gelu_and_down_input_transform",
        )?;
        let finalize_gated_gelu_and_down_input_sk2_pipeline = pipeline_audit::build(
            &device,
            &library,
            "linear_finalize_gated_gelu_and_down_input_transform_sk2",
        )?;
        let finalize_and_post_ffn_norm_scale_pipeline = pipeline_audit::build(
            &device,
            &library,
            "linear_finalize_and_post_ffn_norm_scale",
        )?;
        let finalize_and_post_ffn_norm_scale_sk2_pipeline = pipeline_audit::build(
            &device,
            &library,
            "linear_finalize_and_post_ffn_norm_scale_sk2",
        )?;
        let finalize_and_post_ffn_norm_scale_sk4_pipeline = pipeline_audit::build(
            &device,
            &library,
            "linear_finalize_and_post_ffn_norm_scale_sk4",
        )?;
        let post_attn_pre_ffn_transform_pipeline = pipeline_audit::build(
            &device,
            &library,
            "post_attn_pre_ffn_transform",
        )?;
        let finalize_o_proj_and_pre_ffn_transform_pipeline = pipeline_audit::build(
            &device,
            &library,
            "linear_finalize_o_proj_and_pre_ffn_transform",
        )?;
        let finalize_o_proj_and_pre_ffn_transform_sk2_pipeline = pipeline_audit::build(
            &device,
            &library,
            "linear_finalize_o_proj_and_pre_ffn_transform_sk2",
        )?;
        let finalize_o_proj_and_pre_ffn_transform_sk4_pipeline = pipeline_audit::build(
            &device,
            &library,
            "linear_finalize_o_proj_and_pre_ffn_transform_sk4",
        )?;
        let input_norm_and_qkv_transform_pipeline = pipeline_audit::build(
            &device,
            &library,
            "input_norm_and_qkv_transform",
        )?;
        let cross_layer_pipeline = pipeline_audit::build(
            &device,
            &library,
            "finalize_post_ffn_and_next_input_norm_transform",
        )?;
        let input_rows_pipeline = pipeline_audit::build(&device, &library, "linear_input_rows")?;
        let gemv_tiled_pipeline = pipeline_audit::build(&device, &library, "linear_gemv_tiled")?;
        let gemm_mma_pipeline = pipeline_audit::build(&device, &library, "linear_gemm_mma")?;
        let gemm_mma_regpipe_pipeline =
            pipeline_audit::build(&device, &library, "linear_gemm_mma_regpipe")?;
        let gemm_mma_4sg_pipeline =
            pipeline_audit::build(&device, &library, "linear_gemm_mma_4sg")?;
        let gemm_mma_regpipe_k32_pipeline =
            pipeline_audit::build(&device, &library, "linear_gemm_mma_regpipe_k32")?;
        let mrhs_repacked_m4 = pipeline_audit::build(
            &device,
            &library,
            "linear_gemv_mrhs_ishizuki_k3_colmajor_contig4_m4",
        )?;
        let mrhs_repacked_m8 = pipeline_audit::build(
            &device,
            &library,
            "linear_gemv_mrhs_ishizuki_k3_colmajor_contig4_m8",
        )?;
        let mrhs_repacked_m16 = pipeline_audit::build(
            &device,
            &library,
            "linear_gemv_mrhs_ishizuki_k3_colmajor_contig4_m16",
        )?;
        let mrhs_repacked_m32 = pipeline_audit::build(
            &device,
            &library,
            "linear_gemv_mrhs_ishizuki_k3_colmajor_contig4_m32",
        )?;
        let mrhs_exl3_m4 =
            pipeline_audit::build(&device, &library, "linear_gemv_mrhs_exl3_k3_mcg_m4")?;
        let mrhs_exl3_m8 =
            pipeline_audit::build(&device, &library, "linear_gemv_mrhs_exl3_k3_mcg_m8")?;
        let mrhs_exl3_m16 =
            pipeline_audit::build(&device, &library, "linear_gemv_mrhs_exl3_k3_mcg_m16")?;
        let mrhs_exl3_m32 =
            pipeline_audit::build(&device, &library, "linear_gemv_mrhs_exl3_k3_mcg_m32")?;
        let finalize_rows_pipeline =
            pipeline_audit::build(&device, &library, "linear_finalize_rows")?;
        let zero_raw_pipeline = pipeline_audit::build(&device, &library, "linear_zero_raw")?;
        // `Option`, so a build without the standalone split-K reduce kernels
        // still starts. Nothing in the engine dispatches them -- the fused
        // `linear_finalize_simd_sk{2,4}` reduce in the same dispatch -- so this
        // is a bench/test-only pair.
        let gemv_reduce_sk2_pipeline =
            pipeline_audit::build(&device, &library, "gemv_reduce_sk2").ok();
        let gemv_reduce_sk4_pipeline =
            pipeline_audit::build(&device, &library, "gemv_reduce_sk4").ok();
        // Specialised decode pipelines, one per integer bit rate.
        let mut fast_gemv_pipelines = Vec::with_capacity(8);
        for bits in 1..=8_u32 {
            let entry = format!("linear_gemv_fast{bits}");
            fast_gemv_pipelines.push(pipeline_audit::build(&device, &library, &entry).ok());
        }
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            _library: library,
            input_pipeline,
            exl3_gemv_pipelines,
            exl3_gemv_by_name,
            exl3_gemv_slot_names,
            ishizuki_gemv_pipelines,
            ishizuki_gemv_by_name,
            repacked_gemv_by_name,
            gemv_pipeline,
            finalize_pipeline,
            finalize_simd_pipeline,
            finalize_simd_sk2_pipeline,
            finalize_simd_sk4_pipeline,
            finalize_simd3_pipeline,
            finalize_simd3_sk2_pipeline,
            finalize_gated_gelu_pipeline,
            finalize_gated_gelu_sk2_pipeline,
            finalize_gated_gelu_and_down_input_pipeline,
            finalize_gated_gelu_and_down_input_sk2_pipeline,
            finalize_and_post_ffn_norm_scale_pipeline,
            finalize_and_post_ffn_norm_scale_sk2_pipeline,
            finalize_and_post_ffn_norm_scale_sk4_pipeline,
            post_attn_pre_ffn_transform_pipeline,
            finalize_o_proj_and_pre_ffn_transform_pipeline,
            finalize_o_proj_and_pre_ffn_transform_sk2_pipeline,
            finalize_o_proj_and_pre_ffn_transform_sk4_pipeline,
            input_norm_and_qkv_transform_pipeline,
            cross_layer_pipeline,
            input_rows_pipeline,
            gemv_tiled_pipeline,
            gemm_mma_pipeline,
            gemm_mma_regpipe_pipeline,
            gemm_mma_4sg_pipeline,
            gemm_mma_regpipe_k32_pipeline,
            mrhs_repacked_m4,
            mrhs_repacked_m8,
            mrhs_repacked_m16,
            mrhs_repacked_m32,
            mrhs_exl3_m4,
            mrhs_exl3_m8,
            mrhs_exl3_m16,
            mrhs_exl3_m32,
            finalize_rows_pipeline,
            zero_raw_pipeline,
            gemv_reduce_sk2_pipeline,
            gemv_reduce_sk4_pipeline,
            fast_gemv_pipelines,
        })
    }

    pub fn gemv(&self, request: LinearRequest<'_>) -> Result<Vec<u16>, KernelError> {
        validate_dimensions(&request)?;
        let LinearRequest {
            packed,
            input,
            input_signs,
            output_signs,
            in_features,
            out_features,
            rate,
            codebook,
        } = request;
        let input_buffer = self.make_buffer(input)?;
        let input_sign_buffer = self.make_buffer(input_signs)?;
        let output_sign_buffer = self.make_buffer(output_signs)?;
        let transformed = self.device.new_buffer(
            size_of_val(input) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let raw = self.device.new_buffer(
            (out_features * size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output = self.device.new_buffer(
            size_of_val(output_signs) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let packed_buffer = self.make_buffer(packed)?;
        let input_blocks = u32::try_from(in_features / 128)
            .map_err(|_| KernelError::InvalidInput("input dimension is too large".to_string()))?;
        let output_blocks = u32::try_from(out_features / 128)
            .map_err(|_| KernelError::InvalidInput("output dimension is too large".to_string()))?;
        let params = LinearParams {
            in_features: u32::try_from(in_features).map_err(|_| {
                KernelError::InvalidInput("input dimension is too large".to_string())
            })?,
            out_features: u32::try_from(out_features).map_err(|_| {
                KernelError::InvalidInput("output dimension is too large".to_string())
            })?,
            tile_columns: u32::try_from(out_features / 16).map_err(|_| {
                KernelError::InvalidInput("output dimension is too large".to_string())
            })?,
            packed_words: u32::try_from(rate.packed_words())
                .map_err(|_| KernelError::InvalidInput("packed shape is too large".to_string()))?,
            rate: rate.encoded(),
            codebook: codebook.encoded(),
            queries: 1,
            perm_mode: 0u32,
        };
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode_input_transform_blocks(
            encoder,
            &[
                Some(&input_buffer),
                Some(&input_sign_buffer),
                Some(&transformed),
            ],
            input_blocks,
        );
        self.encode(
            encoder,
            &self.gemv_pipeline,
            &[Some(&packed_buffer), Some(&transformed), Some(&raw)],
            params_bytes(&params),
            out_features as u64,
        );
        self.encode(
            encoder,
            &self.finalize_pipeline,
            &[Some(&raw), Some(&output_sign_buffer), Some(&output)],
            &output_blocks.to_le_bytes(),
            u64::from(output_blocks),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(KernelError::CommandFailed);
        }
        read_u16_buffer(&output)
    }

    pub fn gemv_buffer(&self, request: LinearBufferRequest<'_>) -> Result<Vec<u16>, KernelError> {
        self.gemv_buffer_impl(request, None)
    }

    /// [`Self::gemv_buffer`] with the decode kernel chosen by NAME instead of by
    /// `(codebook, aligned, bits)`.
    ///
    /// This exists so a test can A/B two kernels inside ONE process. The
    /// environment switches cannot do that: `env_flag` memoises on first read and
    /// documents that no test mutates the environment mid-process, so flipping
    /// `KAPLUMBA_LEGACY_GEMV` halfway through a test would make the two halves
    /// measure different kernels and report the difference as if it were a
    /// property of the code. A named dispatch carries no hidden state.
    ///
    /// `name` must be one of the generated `linear_gemv_exl3_*` kernels and must
    /// be consistent with the request's `rate` and `codebook`; the caller owns
    /// that pairing. Nothing in the engine calls this.
    pub fn gemv_buffer_named(
        &self,
        name: &str,
        request: LinearBufferRequest<'_>,
    ) -> Result<Vec<u16>, KernelError> {
        self.gemv_buffer_impl(request, Some(name))
    }

    fn gemv_buffer_impl(
        &self,
        request: LinearBufferRequest<'_>,
        named: Option<&str>,
    ) -> Result<Vec<u16>, KernelError> {
        let LinearBufferRequest {
            packed,
            packed_offset,
            input,
            input_signs,
            output_signs,
            in_features,
            out_features,
            rate,
            codebook,
        } = request;
        validate_buffer_shape(
            packed.length().saturating_sub(packed_offset),
            packed_offset,
            input.len(),
            input_signs.len(),
            output_signs.len(),
            in_features,
            out_features,
            rate,
            codebook,
        )?;
        let input_buffer = self.make_buffer(input)?;
        let input_sign_buffer = self.make_buffer(input_signs)?;
        let output_sign_buffer = self.make_buffer(output_signs)?;
        let transformed = self.device.new_buffer(
            size_of_val(input) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let raw = self.device.new_buffer(
            (out_features * size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output = self.device.new_buffer(
            size_of_val(output_signs) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let input_blocks = to_u32(in_features / 128, "input dimension")?;
        let output_blocks = to_u32(out_features / 128, "output dimension")?;
        let params = LinearParams {
            in_features: to_u32(in_features, "input dimension")?,
            out_features: to_u32(out_features, "output dimension")?,
            tile_columns: to_u32(out_features / 16, "tile column count")?,
            packed_words: to_u32(rate.packed_words(), "packed shape")?,
            rate: rate.encoded(),
            codebook: codebook.encoded(),
            queries: 1,
            perm_mode: 0u32,
        };
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode_input_transform_blocks(
            encoder,
            &[
                Some(&input_buffer),
                Some(&input_sign_buffer),
                Some(&transformed),
            ],
            input_blocks,
        );
        // Integer rates go to the lane-per-thread matvec: one window setup per
        // eight codes, 256-thread groups, and enough threads to fill the GPU.
        // Grid is (out_features / 16) column-tiles x 256 threads.
        if let Some(name) = named {
            let pipeline = self
                .exl3_gemv_pipeline_by_name(name)
                .ok_or_else(|| KernelError::InvalidInput(format!("no decode kernel {name}")))?;
            self.encode_offset_256(
                encoder,
                pipeline,
                &[
                    TensorRef::new(packed, packed_offset),
                    TensorRef::new(&transformed, 0),
                    TensorRef::new(&raw, 0),
                ],
                params_bytes(&params),
                (out_features / 16 * 256) as u64,
            );
        } else if let Some(pipeline) = self.decode_gemv_pipeline(rate, packed_offset, codebook) {
            // `decode_gemv_pipeline`, not `exl3_gemv_pipeline`, so this path
            // honours `KAPLUMBA_DECODE_NR0` exactly as `encode_gemv` does. The
            // two entry points dispatch the same kernel for the same tensor, and
            // an NR0 knob that moved only one of them would give two different
            // answers for the same projection -- which is how a sweep comes to
            // measure the wrong kernel while the engine keeps passing.
            self.encode_offset_256(
                encoder,
                pipeline,
                &[
                    TensorRef::new(packed, packed_offset),
                    TensorRef::new(&transformed, 0),
                    TensorRef::new(&raw, 0),
                ],
                params_bytes(&params),
                (out_features / 16 / decode_nr0().max(1) as usize * 256) as u64,
            );
        } else {
            self.encode_offset(
                encoder,
                self.fast_gemv_pipeline(rate, codebook)
                    .unwrap_or(&self.gemv_pipeline),
                &[
                    TensorRef::new(packed, packed_offset),
                    TensorRef::new(&transformed, 0),
                    TensorRef::new(&raw, 0),
                ],
                params_bytes(&params),
                out_features as u64,
            );
        }
        self.encode(
            encoder,
            &self.finalize_pipeline,
            &[Some(&raw), Some(&output_sign_buffer), Some(&output)],
            &output_blocks.to_le_bytes(),
            u64::from(output_blocks),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(KernelError::CommandFailed);
        }
        read_u16_buffer(&output)
    }

    pub fn gemm_chunk_buffer(
        &self,
        request: LinearBufferRequest<'_>,
    ) -> Result<Vec<u16>, KernelError> {
        self.gemm_chunk_buffer_with_stats(request)
            .map(|(output, _)| output)
    }

    pub fn gemm_chunk_buffer_with_stats(
        &self,
        request: LinearBufferRequest<'_>,
    ) -> Result<(Vec<u16>, LinearPrefillStats), KernelError> {
        let LinearBufferRequest {
            packed,
            packed_offset,
            input,
            input_signs,
            output_signs,
            in_features,
            out_features,
            rate,
            codebook,
        } = request;
        let tokens = input.len() / in_features.max(1);
        validate_buffer_shape(
            packed.length().saturating_sub(packed_offset),
            packed_offset,
            input.len(),
            input_signs.len(),
            output_signs.len(),
            in_features,
            out_features,
            rate,
            codebook,
        )?;
        let input_buffer = self.make_buffer(input)?;
        let input_sign_buffer = self.make_buffer(input_signs)?;
        let output_sign_buffer = self.make_buffer(output_signs)?;
        let input_elements = input.len();
        let output_elements = checked_product(tokens, out_features, "buffer output")?;
        let transformed = self.device.new_buffer(
            size_of_val(input) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let raw = self.device.new_buffer(
            (output_elements * size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        zero_f32_buffer(&raw, output_elements)?;
        let output = self.device.new_buffer(
            (output_elements * size_of::<u16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let row_params = PrefillRowParams {
            in_features: to_u32(in_features, "input dimension")?,
            out_features: to_u32(out_features, "output dimension")?,
            queries: to_u32(tokens, "query count")?,
            perm_mode: 0u32,
        };
        let linear_params = LinearParams {
            in_features: to_u32(in_features, "input dimension")?,
            out_features: to_u32(out_features, "output dimension")?,
            tile_columns: to_u32(out_features / 16, "tile column count")?,
            packed_words: to_u32(rate.packed_words(), "packed shape")?,
            rate: rate.encoded(),
            codebook: codebook.encoded(),
            queries: to_u32(tokens, "query count")?,
            perm_mode: 0u32,
        };
        let stats = prefill_stats(tokens, in_features, out_features)?;
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode(
            encoder,
            &self.input_rows_pipeline,
            &[
                Some(&input_buffer),
                Some(&input_sign_buffer),
                Some(&transformed),
            ],
            params_bytes(&row_params),
            (input_elements / 128) as u64,
        );
        self.encode_groups(
            encoder,
            &self.gemv_tiled_pipeline,
            &[Some(packed), Some(&transformed), Some(&raw)],
            params_bytes(&linear_params),
            (out_features / 16) as u64,
            tokens.div_ceil(16) as u64,
        );
        self.encode(
            encoder,
            &self.finalize_rows_pipeline,
            &[Some(&raw), Some(&output_sign_buffer), Some(&output)],
            params_bytes(&row_params),
            (output_elements / 128) as u64,
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(KernelError::CommandFailed);
        }
        Ok((read_u16_buffer(&output)?, stats))
    }

    pub fn gemm_chunk(&self, request: PrefillRequest<'_>) -> Result<Vec<u16>, KernelError> {
        self.gemm_chunk_with_stats(request)
            .map(|(output, _)| output)
    }

    pub fn gemm_chunk_with_stats(
        &self,
        request: PrefillRequest<'_>,
    ) -> Result<(Vec<u16>, LinearPrefillStats), KernelError> {
        validate_prefill_dimensions(&request)?;
        let PrefillRequest {
            packed,
            input,
            input_signs,
            output_signs,
            queries,
            in_features,
            out_features,
            rate,
            codebook,
        } = request;
        let input_buffer = self.make_buffer(input)?;
        let input_sign_buffer = self.make_buffer(input_signs)?;
        let output_sign_buffer = self.make_buffer(output_signs)?;
        let packed_buffer = self.make_buffer(packed)?;
        let input_elements = checked_product(queries, in_features, "input shape")?;
        let output_elements = checked_product(queries, out_features, "output shape")?;
        let transformed = self.device.new_buffer(
            (input_elements * size_of::<u16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let raw = self.device.new_buffer(
            (output_elements * size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        zero_f32_buffer(&raw, output_elements)?;
        let output = self.device.new_buffer(
            (output_elements * size_of::<u16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let row_params = PrefillRowParams {
            in_features: to_u32(in_features, "input dimension")?,
            out_features: to_u32(out_features, "output dimension")?,
            queries: to_u32(queries, "query count")?,
            perm_mode: 0u32,
        };
        let linear_params = LinearParams {
            in_features: to_u32(in_features, "input dimension")?,
            out_features: to_u32(out_features, "output dimension")?,
            tile_columns: to_u32(out_features / 16, "tile column count")?,
            packed_words: to_u32(rate.packed_words(), "packed shape")?,
            rate: rate.encoded(),
            codebook: codebook.encoded(),
            queries: to_u32(queries, "query count")?,
            perm_mode: 0u32,
        };
        let query_chunks = queries.div_ceil(16);
        let tile_count = checked_product(in_features / 16, out_features / 16, "tile count")?;
        let decoded_tiles = checked_product(tile_count, query_chunks, "decoded tile count")?;
        let full_query_chunks = queries / 16;
        let final_query_count = queries % 16;
        let valid_queries = full_query_chunks * 16 + usize::from(final_query_count != 0);
        let weight_applications = checked_product(
            tile_count,
            checked_product(valid_queries, 256, "weight application count")?,
            "weight application count",
        )?;
        let stats = LinearPrefillStats {
            query_chunks,
            decoded_tiles,
            weight_applications,
            dispatch_groups: checked_product(out_features / 16, query_chunks, "dispatch count")?,
        };
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode(
            encoder,
            &self.input_rows_pipeline,
            &[
                Some(&input_buffer),
                Some(&input_sign_buffer),
                Some(&transformed),
            ],
            params_bytes(&row_params),
            (queries * in_features / 128) as u64,
        );
        self.encode_groups(
            encoder,
            &self.gemv_tiled_pipeline,
            &[Some(&packed_buffer), Some(&transformed), Some(&raw)],
            params_bytes(&linear_params),
            (out_features / 16) as u64,
            queries.div_ceil(16) as u64,
        );
        self.encode(
            encoder,
            &self.finalize_rows_pipeline,
            &[Some(&raw), Some(&output_sign_buffer), Some(&output)],
            params_bytes(&row_params),
            (queries * out_features / 128) as u64,
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(KernelError::CommandFailed);
        }
        Ok((read_u16_buffer(&output)?, stats))
    }

    /// Pick the specialised decode kernel when the rate and codebook allow it.
    /// Lane-per-thread matvec specialised on rate and on whether `packed` is
    /// 4-byte aligned at `packed_offset`.
    ///
    /// Arena-resident tensors always are (offset 0 of a page-aligned buffer).
    /// Mapped checkpoint tensors are not -- `lm_head.trellis` starts at a file
    /// offset of % 4 == 2 -- and reading those as `uint *` is what made every
    /// earlier zero-copy build decode to noise.
    /// Dispatch the A/B ishizuki kernel, which needs K and N in addition to the
    /// usual three buffers. They ride in the params slot, which the encoder binds
    /// immediately after the tensor bindings.
    #[allow(clippy::too_many_arguments)]
    /// Dispatch a named `linear_gemv_ishizuki_*` kernel.
    ///
    /// Separate from [`Self::encode_ishizuki_gemv`] only in where the pipeline
    /// comes from — everything after that is shared, deliberately, so a variant
    /// cannot differ from the original in its grid, its threadgroup size, or its
    /// bound buffers. A variant that changed the launch shape would not be
    /// measuring the window arithmetic, it would be measuring two changes at
    /// once, and the A/B would be unattributable.
    ///
    /// The rate is checked against the name so a test cannot pass `_k3`'s data
    /// to `_k6`: the two disagree about `KA`, and the result would be garbage
    /// rather than wrong-looking.
    pub fn encode_ishizuki_gemv_named(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        name: &str,
        request: &LinearEncodeRequest<'_>,
    ) -> Result<(), KernelError> {
        // The input transform first, exactly as `encode_gemv` does it. Omitting
        // this was a real bug: the kernel then reads a `transformed` buffer that
        // was never written, produces all zeros, and looks like a kernel that
        // does not work rather than a dispatch that skipped a step. Found by
        // `tests/decode_matvec_golden.rs`, whose all-zero guard is what turned
        // "the digest is empty" into a diagnosable message.
        if !request.pretransformed {
            let input_blocks = to_u32(request.in_features / 128, "input dimension")?;
            encoder.dispatch_1d(
                &self.input_pipeline,
                &[
                    Some(request.input),
                    Some(request.input_signs),
                    Some(request.transformed),
                ],
                &input_blocks.to_le_bytes(),
                u64::from(input_blocks) * 32,
                32,
            )?;
        }
        let pipeline = self.ishizuki_gemv_pipeline_by_name(name).ok_or_else(|| {
            KernelError::InvalidInput(format!("no ishizuki decode kernel {name}"))
        })?;
        let Exl3Rate::Integer(bits) = request.rate else {
            return Err(KernelError::InvalidInput(
                "the ishizuki family only dispatches integer rates".to_string(),
            ));
        };
        match Self::ishizuki_rate_of_name(name) {
            Some(named) if named == bits => {}
            Some(named) => {
                return Err(KernelError::InvalidInput(format!(
                    "{name} decodes {named} bpw but the request is {bits} bpw"
                )));
            }
            None => {
                return Err(KernelError::InvalidInput(format!(
                    "{name} does not name a bit rate"
                )));
            }
        }
        let is_2col_sk4 = name.ends_with("_2col_sk4");
        let is_2col_sk2 = name.ends_with("_2col_sk2");
        let is_1col_sk2 = name.ends_with("_1col_sk2");
        let is_stream = name.ends_with("_stream");
        let is_sk2 = name.ends_with("_sk2") && !is_2col_sk2 && !is_1col_sk2;
        let is_sk4 = name.ends_with("_sk4") && !is_2col_sk4;
        let is_2col = name.ends_with("_2col");
        let is_1col = name.ends_with("_1col");
        if is_2col_sk4 {
            let pairs = ((request.out_features / 16) / 2) as u64;
            let groups = pairs * 4;
            let shape_values = [
                to_u32(request.in_features, "input dimension")?,
                to_u32(request.out_features, "output dimension")?,
            ];
            encoder.dispatch_1d_offset(
                pipeline,
                &[
                    TensorRef::new(request.packed, request.packed_offset),
                    TensorRef::new(request.transformed, 0),
                    TensorRef::new(request.raw, 0),
                ],
                params_bytes(&shape_values),
                groups * 256,
                256,
            )?;
            let output_blocks = to_u32(request.out_features / 128, "output dimension")?;
            return encoder.dispatch_1d(
                &self.finalize_simd_sk4_pipeline,
                &[
                    Some(request.raw),
                    Some(request.output_signs),
                    Some(request.output),
                ],
                &output_blocks.to_le_bytes(),
                u64::from(output_blocks) * 32,
                32,
            );
        } else if is_2col_sk2 {
            let pairs = ((request.out_features / 16) / 2) as u64;
            let groups = pairs * 2;
            let shape_values = [
                to_u32(request.in_features, "input dimension")?,
                to_u32(request.out_features, "output dimension")?,
            ];
            encoder.dispatch_1d_offset(
                pipeline,
                &[
                    TensorRef::new(request.packed, request.packed_offset),
                    TensorRef::new(request.transformed, 0),
                    TensorRef::new(request.raw, 0),
                ],
                params_bytes(&shape_values),
                groups * 256,
                256,
            )?;
            let output_blocks = to_u32(request.out_features / 128, "output dimension")?;
            return encoder.dispatch_1d(
                &self.finalize_simd_sk2_pipeline,
                &[
                    Some(request.raw),
                    Some(request.output_signs),
                    Some(request.output),
                ],
                &output_blocks.to_le_bytes(),
                u64::from(output_blocks) * 32,
                32,
            );
        } else if is_stream {
            let tiles = (request.out_features / 16) as u64;
            let groups = (tiles + 7) / 8;
            let shape_values = [
                to_u32(request.in_features, "input dimension")?,
                to_u32(request.out_features, "output dimension")?,
            ];
            encoder.dispatch_1d_offset(
                pipeline,
                &[
                    TensorRef::new(request.packed, request.packed_offset),
                    TensorRef::new(request.transformed, 0),
                    TensorRef::new(request.raw, 0),
                ],
                params_bytes(&shape_values),
                groups * 256,
                256,
            )?;
        } else if is_sk2 {
            let tiles = (request.out_features / 16) as u64;
            let groups = tiles * 2;
            let shape_values = [
                to_u32(request.in_features, "input dimension")?,
                to_u32(request.out_features, "output dimension")?,
            ];
            encoder.dispatch_1d_offset(
                pipeline,
                &[
                    TensorRef::new(request.packed, request.packed_offset),
                    TensorRef::new(request.transformed, 0),
                    TensorRef::new(request.raw, 0),
                ],
                params_bytes(&shape_values),
                groups * 256,
                256,
            )?;
            let output_blocks = to_u32(request.out_features / 128, "output dimension")?;
            return encoder.dispatch_1d(
                &self.finalize_simd_sk2_pipeline,
                &[
                    Some(request.raw),
                    Some(request.output_signs),
                    Some(request.output),
                ],
                &output_blocks.to_le_bytes(),
                u64::from(output_blocks) * 32,
                32,
            );
        } else if is_sk4 {
            let tiles = (request.out_features / 16) as u64;
            let groups = tiles * 4;
            let shape_values = [
                to_u32(request.in_features, "input dimension")?,
                to_u32(request.out_features, "output dimension")?,
            ];
            encoder.dispatch_1d_offset(
                pipeline,
                &[
                    TensorRef::new(request.packed, request.packed_offset),
                    TensorRef::new(request.transformed, 0),
                    TensorRef::new(request.raw, 0),
                ],
                params_bytes(&shape_values),
                groups * 256,
                256,
            )?;
            let output_blocks = to_u32(request.out_features / 128, "output dimension")?;
            return encoder.dispatch_1d(
                &self.finalize_simd_sk4_pipeline,
                &[
                    Some(request.raw),
                    Some(request.output_signs),
                    Some(request.output),
                ],
                &output_blocks.to_le_bytes(),
                u64::from(output_blocks) * 32,
                32,
            );
        } else if is_1col_sk2 {
            let tiles = (request.out_features / 16) as u64;
            let groups = tiles * 2;
            let shape_values = [
                to_u32(request.in_features, "input dimension")?,
                to_u32(request.out_features, "output dimension")?,
            ];
            encoder.dispatch_1d_offset(
                pipeline,
                &[
                    TensorRef::new(request.packed, request.packed_offset),
                    TensorRef::new(request.transformed, 0),
                    TensorRef::new(request.raw, 0),
                ],
                params_bytes(&shape_values),
                groups * 256,
                256,
            )?;
            let output_blocks = to_u32(request.out_features / 128, "output dimension")?;
            return encoder.dispatch_1d(
                &self.finalize_simd_sk2_pipeline,
                &[
                    Some(request.raw),
                    Some(request.output_signs),
                    Some(request.output),
                ],
                &output_blocks.to_le_bytes(),
                u64::from(output_blocks) * 32,
                32,
            );
        } else if is_1col {
            let tiles = (request.out_features / 16) as u64;
            let groups = tiles.min(gemv_max_groups()).max(1);
            let shape_values = [
                to_u32(request.in_features, "input dimension")?,
                to_u32(request.out_features, "output dimension")?,
            ];
            encoder.dispatch_1d_offset(
                pipeline,
                &[
                    TensorRef::new(request.packed, request.packed_offset),
                    TensorRef::new(request.transformed, 0),
                    TensorRef::new(request.raw, 0),
                ],
                params_bytes(&shape_values),
                groups * 256,
                256,
            )?;
        } else if is_2col {
            let pairs = (request.out_features / 32) as u64;
            let groups = pairs.min(gemv_max_groups()).max(1);
            let shape_values = [
                to_u32(request.in_features, "input dimension")?,
                to_u32(request.out_features, "output dimension")?,
            ];
            encoder.dispatch_1d_offset(
                pipeline,
                &[
                    TensorRef::new(request.packed, request.packed_offset),
                    TensorRef::new(request.transformed, 0),
                    TensorRef::new(request.raw, 0),
                ],
                params_bytes(&shape_values),
                groups * 256,
                256,
            )?;
        } else {
            self.encode_ishizuki_gemv(
                encoder,
                pipeline,
                request.packed,
                request.packed_offset,
                request.transformed,
                request.raw,
                request.in_features,
                request.out_features,
            )?;
        }
        let output_blocks = to_u32(request.out_features / 128, "output dimension")?;
        encoder.dispatch_1d(
            &self.finalize_simd_pipeline,
            &[
                Some(request.raw),
                Some(request.output_signs),
                Some(request.output),
            ],
            &output_blocks.to_le_bytes(),
            u64::from(output_blocks) * 32,
            32,
        )
    }

    /// The pipeline for a named `linear_gemv_ishizuki_*` kernel, if it exists.
    pub fn ishizuki_gemv_pipeline_by_name(&self, name: &str) -> Option<&ComputePipelineState> {
        self.ishizuki_gemv_by_name.get(name)
    }

    /// Every `linear_gemv_ishizuki_*` kernel name, sorted.
    ///
    /// Public so a test can assert the family is complete and so an A/B harness
    /// can enumerate variants rather than hard-coding names that then drift.
    pub fn ishizuki_gemv_kernel_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.ishizuki_gemv_by_name.keys().cloned().collect();
        names.sort();
        names
    }

    fn encode_ishizuki_gemv(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        pipeline: &ComputePipelineState,
        packed: &Buffer,
        packed_offset: u64,
        transformed: &Buffer,
        raw: &Buffer,
        in_features: usize,
        out_features: usize,
    ) -> Result<(), KernelError> {
        // Same two u32s, now in the params slot: see the `shape` binding note on
        // `linear_gemv_ishizuki_k*`. Numerically inert.
        let shape_values = [
            to_u32(in_features, "input dimension")?,
            to_u32(out_features, "output dimension")?,
        ];
        // One threadgroup per 16 outputs, capped. At N=21504 that is 1344
        // threadgroups on 20 cores -- about 1.4 waves -- and a trailing partial
        // wave costs more than the extra parallelism buys. Capping the grid and
        // letting each threadgroup stride over several output tiles turns the
        // tail into an exact multiple. lm_head, which launches 16384
        // threadgroups, reaches 168.7 GB/s on this same kernel.
        let tiles = (out_features / 16) as u64;
        let groups = tiles.min(gemv_max_groups()).max(1);
        encoder.dispatch_1d_offset(
            pipeline,
            &[
                TensorRef::new(packed, packed_offset),
                TensorRef::new(transformed, 0),
                TensorRef::new(raw, 0),
            ],
            params_bytes(&shape_values),
            groups * 256,
            256,
        )
    }

    /// The ishizuki kernel `encode_gemv` would dispatch for this tensor, if any.
    ///
    /// Exists because the dispatch order in `encode_gemv` is *ishizuki first,
    /// EXL3 second*, and the two are easy to confuse: the whole `_mcg`
    /// specialisation work -- the codebook folded to a compile-time constant, the
    /// 8.1 -> 9.2 tok/s step -- only pays if the EXL3 arm is the one that runs.
    /// `KAPLUMBA_OWN_GEMV` flips the order, and which arm is live decides what any
    /// EXL3 measurement is a measurement *of*. There was no way to ask before
    /// this, which is a gap a reader of the notes could easily fall into.
    pub fn ishizuki_gemv_kernel_name(&self, rate: Exl3Rate, packed_offset: u64) -> Option<String> {
        self.ishizuki_gemv_pipeline(rate, packed_offset, 0)
            .map(|_| "ishizuki".to_string())
    }

    fn ishizuki_gemv_pipeline(
        &self,
        rate: Exl3Rate,
        packed_offset: u64,
        perm_mode: u32,
    ) -> Option<&ComputePipelineState> {
        if env_flag("KAPLUMBA_OWN_GEMV") {
            return None; // A/B against the pre-port kernel
        }
        let Exl3Rate::Integer(bits) = rate else {
            return None;
        };
        if bits < 1 || bits > 8 || packed_offset % 4 != 0 {
            return None;
        }
        // The 2-word window where it is provably equivalent, unless asked for the
        // 3-word form by name.
        //
        // Selected by looking the `_w2` name up in the map, so the dispatch
        // decision and the kernel that exists cannot disagree: a rate with no
        // `_w2` kernel simply falls through to the 3-word slot, which is why 6 bpw
        // (`lm_head`) is unaffected rather than broken. The alternative — a
        // parallel `bool` per rate — is a second list to keep in step with the
        // generator, and would fail open (dispatching a kernel that does not
        // exist) rather than closed.
        //
        // `KAPLUMBA_ISHIZUKI_3WORD=1` restores the original. It exists because the
        // variant is bit-identical by construction *and* verified on the hardware,
        // so there is no correctness reason to want the old form — only an
        // attribution reason, when re-measuring a number that was taken before
        // this landed.
        if perm_mode == 1 || env_flag("KAPLUMBA_REPACKED_GEMV") {
            if let Some(pipeline) = self
                .ishizuki_gemv_by_name
                .get(&format!("linear_gemv_ishizuki_k{bits}_colmajor_contig4"))
                .or_else(|| {
                    self.ishizuki_gemv_by_name
                        .get(&format!("linear_gemv_ishizuki_k{bits}_colmajor"))
                })
            {
                return Some(pipeline);
            }
        }
        if !env_flag("KAPLUMBA_ISHIZUKI_3WORD") {
            if let Some(pipeline) = self
                .ishizuki_gemv_by_name
                .get(&format!("linear_gemv_ishizuki_k{bits}_w2"))
            {
                return Some(pipeline);
            }
        }
        self.ishizuki_gemv_pipelines[bits as usize].as_ref()
    }

    /// Slot of a `(codebook, aligned, bits)` triple in `exl3_gemv_pipelines`.
    ///
    /// The decode matvec is specialised on three axes -- integer rate, buffer
    /// alignment, and codebook -- because all three fold a comparison away from
    /// the innermost loop. The codebook axis is new here: it was a runtime
    /// argument branching eight times per lane, which cost 61.7% of the kernel
    /// (62.4 -> 163.0 GB/s on the real 5376x5376 shape, bit-exact).
    ///
    /// Layout must stay in step with the construction loop in [`Self::new`]:
    /// `1 + codebook_family * 16 + aligned * 8 + (bits - 1)`, family order MCG,
    /// plain, mul1. Index 0 is unused, so the table is [`EXL3_GEMV_STATES`] long.
    ///
    /// Slot of a `(codebook, aligned, bits)` triple in `exl3_gemv_pipelines`.
    ///
    /// Public so a test can pin the mapping. It is public because getting it
    /// wrong is silent and catastrophic: an off-by-one here selects the kernel
    /// for the NEXT bit rate, so every 3-bit decoder linear in the model runs
    /// the 4-bit kernel. The kernel still runs, still returns plausible-looking
    /// numbers, and the whole engine is simply wrong. That happened while this
    /// change was being written -- `1 + family*16 + aligned*8 + bits` instead of
    /// `... + (bits - 1)` -- and no test caught it, because the bit-exactness
    /// test dispatches by NAME and never goes through this function. Hence
    /// `exl3_gemv_index_matches_the_generated_kernel_names`.
    pub fn exl3_gemv_index(
        rate: Exl3Rate,
        packed_offset: u64,
        codebook: Exl3Codebook,
    ) -> Option<usize> {
        let Exl3Rate::Integer(bits) = rate else {
            return None;
        };
        if bits == 0 || bits > 8 {
            return None;
        }
        let family = match codebook {
            Exl3Codebook::Mcg => 0usize,
            Exl3Codebook::Plain => 1,
            Exl3Codebook::Mul1 => 2,
        };
        let aligned = usize::from(packed_offset % 4 == 0);
        // `bits` is 1-based, the table is 0-based, so subtract one. Getting this
        // wrong selects the kernel for the NEXT bit rate -- which for the 3-bit
        // decoder linears means running the 4-bit kernel, and every output is
        // wrong while the kernel still runs and still looks fast.
        Some(1 + family * 16 + aligned * 8 + (bits as usize - 1))
    }

    fn exl3_gemv_pipeline(
        &self,
        rate: Exl3Rate,
        packed_offset: u64,
        codebook: Exl3Codebook,
    ) -> Option<&ComputePipelineState> {
        if env_flag("KAPLUMBA_LEGACY_GEMV") {
            return None; // A/B switch against the pre-lane-per-thread kernels
        }
        let index = Self::exl3_gemv_index(rate, packed_offset, codebook)?;
        self.exl3_gemv_pipelines.get(index)?.as_ref()
    }

    /// Dispatch a specific decode-matvec pipeline by name, bypassing the
    /// `(codebook, aligned, bits)` selection above.
    ///
    /// This exists so a test can A/B two kernels inside ONE process. The
    /// environment switches cannot do that: `env_flag` memoises on first read
    /// and documents that no test mutates the environment mid-process, because
    /// a cached switch flipped halfway through a test would silently make the
    /// two halves measure different kernels and report a difference in the code.
    ///
    /// The name must be one of the generated `linear_gemv_exl3_*` kernels; the
    /// caller is responsible for it matching the `params` it encodes. Nothing in
    /// the engine calls this.
    /// The NR0 a named kernel was generated with, read off its `_r{n}` suffix.
    ///
    /// `encode_gemv_named` takes the kernel name rather than consulting
    /// `KAPLUMBA_DECODE_NR0`, so it has to recover the axis from the name or the
    /// grid it dispatches will not match the kernel it dispatches *to* -- an
    /// r4 kernel with an r1-sized grid leaves three quarters of the output
    /// unwritten, and returns a buffer that is right for three quarters of the
    /// columns and zero for the rest. That is a plausible-looking wrong answer,
    /// so the grid is derived from the name rather than from the environment.
    pub fn nr0_of_kernel_name(name: &str) -> u32 {
        match name.rsplit_once("_r") {
            Some((_, suffix)) => suffix.parse::<u32>().unwrap_or(1),
            None => 1,
        }
    }

    pub fn encode_gemv_named(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        name: &str,
        request: &LinearEncodeRequest<'_>,
    ) -> Result<(), KernelError> {
        // The ishizuki family is checked first, and it is not an EXL3 family: it
        // takes `shape` in a params slot rather than a `LinearParams` struct, and
        // it ignores the codebook. Dispatching it through the EXL3 path would
        // build a `LinearParams` and read it as two u32s, which happens to work
        // only by accident of field order.
        if name.starts_with("linear_gemv_ishizuki_") {
            return self.encode_ishizuki_gemv_named(encoder, name, request);
        }
        // Dimension shape only. `validate_encode` also requires the MCG codebook,
        // which is the right gate for the engine but not for a test that needs to
        // dispatch the plain and mul1 families to show they stay distinct.
        if request.in_features == 0
            || request.out_features == 0
            || !request.in_features.is_multiple_of(128)
            || !request.out_features.is_multiple_of(128)
            || !request.out_features.is_multiple_of(16)
        {
            return Err(KernelError::InvalidInput(
                "linear dimensions must be non-zero multiples of 128 (output of 16)".to_string(),
            ));
        }
        let in_features = request.in_features;
        let out_features = request.out_features;
        let pipeline = self
            .exl3_gemv_pipeline_by_name(name)
            .ok_or_else(|| KernelError::InvalidInput(format!("no decode kernel {name}")))?;
        let params = LinearParams {
            in_features: to_u32(in_features, "input dimension")?,
            out_features: to_u32(out_features, "output dimension")?,
            tile_columns: to_u32(out_features / 16, "tile column count")?,
            packed_words: to_u32(request.rate.packed_words(), "packed shape")?,
            rate: request.rate.encoded(),
            codebook: request.codebook.encoded(),
            queries: 1,
            perm_mode: request.perm_mode,
        };
        if !request.pretransformed {
            let input_blocks = to_u32(in_features / 128, "input dimension")?;
            encoder.dispatch_1d(
                &self.input_pipeline,
                &[
                    Some(request.input),
                    Some(request.input_signs),
                    Some(request.transformed),
                ],
                &input_blocks.to_le_bytes(),
                u64::from(input_blocks),
                64,
            )?;
        }
        let nr0 = Self::nr0_of_kernel_name(name).max(1) as usize;
        let tiles = out_features / 16;
        if !tiles.is_multiple_of(nr0) {
            return Err(KernelError::InvalidInput(format!(
                "out_features/16 = {tiles} is not a multiple of {name}'s NR0 ({nr0})"
            )));
        }
        encoder.dispatch_1d_offset(
            pipeline,
            &[
                TensorRef::new(request.packed, request.packed_offset),
                TensorRef::new(request.transformed, 0),
                TensorRef::new(request.raw, 0),
            ],
            params_bytes(&params),
            (tiles / nr0 * 256) as u64,
            256,
        )?;
        let output_blocks = to_u32(out_features / 128, "output dimension")?;
        encoder.dispatch_1d(
            &self.finalize_pipeline,
            &[
                Some(request.raw),
                Some(request.output_signs),
                Some(request.output),
            ],
            &output_blocks.to_le_bytes(),
            u64::from(output_blocks),
            64,
        )
    }

    /// Look up a decode-matvec pipeline by kernel name.
    ///
    /// Public so a test can assert that every `(codebook, aligned, bits)` slot
    /// the loader can produce resolves to a kernel that is actually in the
    /// library. A missing one is not an error at dispatch: `encode_gemv` falls
    /// through to the next family, so a gap in this table would show up as
    /// finite, plausible, wrong output from a checkpoint whose rate happens to
    /// land on it. See `crates/engine/tests/rate4_8_kernel_coverage.rs`.
    pub fn exl3_gemv_pipeline_by_name(&self, name: &str) -> Option<&ComputePipelineState> {
        self.exl3_gemv_by_name.get(name)
    }

    /// Test/diagnostic-only: which kernel the EXL3 arm would name for this
    /// tensor, as a string.
    ///
    /// `probe_forward` prints this next to a head failure, because "which arm
    /// ran" is the whole question when two checkpoints differ only in
    /// `packed_offset`'s alignment: `ishizuki_gemv_pipeline` refuses anything not
    /// 4-byte aligned, so the same weights take different families depending on
    /// where the tensor happened to land in the file.
    pub fn exl3_gemv_kernel_name_for_test(
        &self,
        rate: Exl3Rate,
        packed_offset: u64,
        codebook: Exl3Codebook,
    ) -> Option<&str> {
        self.exl3_gemv_kernel_name(rate, packed_offset, codebook)
    }

    /// Test/diagnostic-only: the ishizuki kernel `encode_gemv` would select, or
    /// `None` if this tensor takes the EXL3 arm instead.
    pub fn ishizuki_name_for_test(
        &self,
        rate: Exl3Rate,
        packed_offset: u64,
    ) -> Option<&'static str> {
        // `ishizuki_gemv_pipeline` returns a pipeline object and a name cannot be
        // recovered from it, but the *rate* arm is what matters here: a non-4-byte
        // offset, or a non-`Integer(1..=8)` rate, is the only way this returns
        // `None` for a checkpoint the loader accepted.
        if self
            .ishizuki_gemv_pipeline(rate, packed_offset, 0)
            .is_some()
        {
            Some("ishizuki")
        } else {
            None
        }
    }

    /// The kernel name the live decode path dispatches to for
    /// `(rate, codebook)`, honouring `KAPLUMBA_DECODE_NR0`.
    ///
    /// Separate from [`Self::exl3_gemv_kernel_name`] on purpose: that one
    /// answers "what does the index table hold", which is a property of the
    /// table and is what a test pins. This one answers "what will actually
    /// run", which is the table *plus* the environment. Conflating them is how a
    /// sweep knob ends up invisible to the test that is supposed to be checking
    /// the thing the sweep is sweeping.
    pub fn decode_gemv_kernel_name(
        &self,
        rate: Exl3Rate,
        packed_offset: u64,
        codebook: Exl3Codebook,
    ) -> Option<String> {
        let base = self.exl3_gemv_kernel_name(rate, packed_offset, codebook)?;
        let nr0 = decode_nr0();
        if nr0 == 1 {
            return Some(base.to_string());
        }
        // Only the ALIGNED MCG family has r-variants; anything else keeps NR0=1
        // rather than failing, because a sweep knob must not be able to make the
        // engine unrunnable.
        if !base.ends_with("_mcg") || base.contains("_u") {
            return Some(base.to_string());
        }
        let stem = base.trim_end_matches("_mcg");
        Some(format!("{stem}_mcg_r{nr0}"))
    }

    /// [`Self::decode_gemv_kernel_name`], resolved to a pipeline, or the
    /// unsuffixed kernel if the r-variant is absent from this build.
    fn decode_gemv_pipeline(
        &self,
        rate: Exl3Rate,
        packed_offset: u64,
        codebook: Exl3Codebook,
    ) -> Option<&ComputePipelineState> {
        let name = self.decode_gemv_kernel_name(rate, packed_offset, codebook)?;
        self.exl3_gemv_pipeline_by_name(&name)
            .or_else(|| self.exl3_gemv_pipeline(rate, packed_offset, codebook))
    }

    /// The kernel name a `(codebook, aligned, bits)` triple dispatches to, or
    /// `None` if the triple is not one the table covers.
    ///
    /// This exists so a test can pin `exl3_gemv_index` against the table the
    /// construction loop actually built, rather than against a second copy of
    /// the layout written in the test. The bug it guards against is silent and
    /// total: an off-by-one in the index makes every 3-bit decoder linear in the
    /// model run the 4-bit kernel, and the engine keeps producing finite
    /// numbers. See [`Self::exl3_gemv_index`].
    ///
    /// The name is recorded at construction time and read back out of the slot,
    /// rather than recovered by matching pipeline objects: a
    /// `ComputePipelineState` clone is a distinct Rust value, so pointer
    /// identity does not survive the clone that populates the name map, and
    /// matching on it silently returns `None` for every slot.
    pub fn exl3_gemv_kernel_name(
        &self,
        rate: Exl3Rate,
        packed_offset: u64,
        codebook: Exl3Codebook,
    ) -> Option<&str> {
        let index = Self::exl3_gemv_index(rate, packed_offset, codebook)?;
        self.exl3_gemv_slot_names.get(index).map(String::as_str)
    }

    fn fast_gemv_pipeline(
        &self,
        rate: Exl3Rate,
        codebook: Exl3Codebook,
    ) -> Option<&ComputePipelineState> {
        let Exl3Rate::Integer(bits) = rate else {
            return None;
        };
        if codebook != Exl3Codebook::Mcg {
            return None;
        }
        self.fast_gemv_pipelines
            .get(usize::from(bits).saturating_sub(1))
            .and_then(Option::as_ref)
    }

    /// Encode a single-query (decode) linear entirely from GPU buffers.
    ///
    /// Nothing is committed or read back here: the caller owns the command
    /// buffer, so a full 60-layer forward can be encoded as one batch and
    /// synchronised once.
    /// Dispatch only the input transform, so several matvecs that read the same
    /// activation can share it instead of each recomputing it.
    ///
    /// `blocks` is `in_features / 128`. The dispatch is `blocks * 32` threads in
    /// groups of exactly 32 -- one simdgroup per 128-element block. Both halves of
    /// that matter: the kernel's butterfly is `simd_shuffle_xor`-based, which is
    /// undefined when part of a simdgroup is inactive, so the threadgroup must not
    /// straddle a block boundary; and one thread per block is 42 threads at
    /// in_features = 5376, which cannot fill 20 GPU cores while each keeps 128
    /// floats live.
    pub fn encode_input_transform(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        input: &Buffer,
        input_signs: &Buffer,
        transformed: &Buffer,
        in_features: usize,
    ) -> Result<(), KernelError> {
        if crate::ablated("input_transform") {
            return Ok(());
        }
        let input_blocks = to_u32(in_features / 128, "input dimension")?;
        encoder.dispatch_1d(
            &self.input_pipeline,
            &[Some(input), Some(input_signs), Some(transformed)],
            &input_blocks.to_le_bytes(),
            u64::from(input_blocks) * 32,
            32,
        )
    }

    /// The same dispatch over a raw encoder, for the buffer-returning entry
    /// points, which do not go through `MetalEncoder` at all.
    fn encode_input_transform_blocks(
        &self,
        encoder: &ComputeCommandEncoderRef,
        bindings: &[Option<&Buffer>],
        blocks: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.input_pipeline);
        for (index, binding) in bindings.iter().enumerate() {
            encoder.set_buffer(index as u64, binding.map(|value| &**value), 0);
        }
        encoder.set_bytes(
            bindings.len() as u64,
            size_of::<u32>() as u64,
            (&blocks as *const u32) as *const c_void,
        );
        encoder.dispatch_threads(
            MTLSize::new(u64::from(blocks) * 32, 1, 1),
            MTLSize::new(32, 1, 1),
        );
    }

    /// Record one dispatch under `KAPLUMBA_TRACE_GEMV`, once per distinct name.
    ///
    /// The dispatch order in [`Self::encode_gemv`] is **ishizuki first, EXL3
    /// second**, decided by `packed_offset % 4` and `KAPLUMBA_OWN_GEMV`. Every
    /// kernel in this file therefore has one question asked about it before any of
    /// its numbers mean anything: *is this the one that runs?* For the shipped
    /// checkpoint the answer was that 410 of 411 tensors are 4-byte aligned, so
    /// they take the **ishizuki** arm and the whole `_mcg` family -- the codebook
    /// specialisation, and the NR0 axis -- only ever runs for `lm_head`, whose
    /// trellis starts at `% 4 == 2`.
    ///
    /// That is a claim about which kernel runs, not a measurement, so it needs a
    /// way to be checked rather than a way to be believed. This prints the answer
    /// for a real run.
    fn trace_gemv(name: &str) {
        static SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> =
            std::sync::OnceLock::new();
        static ONCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let on = *ONCE.get_or_init(|| std::env::var_os("KAPLUMBA_TRACE_GEMV").is_some());
        if !on {
            return;
        }
        let seen = SEEN.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()));
        let mut seen = seen.lock().unwrap_or_else(|error| error.into_inner());
        if seen.insert(name.to_string()) {
            eprintln!("GEMV-DISPATCH {name}");
        }
    }

    pub fn encode_gemv(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        request: LinearEncodeRequest<'_>,
    ) -> Result<(), KernelError> {
        let LinearEncodeRequest {
            packed,
            packed_offset,
            perm_mode,
            input,
            input_signs,
            output_signs,
            transformed,
            raw,
            output,
            in_features,
            out_features,
            rate,
            codebook,
            pretransformed,
        } = request;
        validate_encode(in_features, out_features, rate, codebook)?;
        let input_blocks = to_u32(in_features / 128, "input dimension")?;
        let output_blocks = to_u32(out_features / 128, "output dimension")?;
        if !pretransformed {
            encoder.dispatch_1d(
                &self.input_pipeline,
                &[Some(input), Some(input_signs), Some(transformed)],
                &input_blocks.to_le_bytes(),
                u64::from(input_blocks) * 32,
                32,
            )?;
        }
        // Integer rates go to the lane-per-thread matvec: one window setup per
        // eight codes, 256-thread groups, and enough threads to fill the GPU.
        // This is the real decode path; `gemv_buffer` routes the same way.
        let use_sk = env_flag("KAPLUMBA_USE_SK")
            && (perm_mode == 1 || env_flag("KAPLUMBA_REPACKED_GEMV"))
            && matches!(rate, Exl3Rate::Integer(3));

        if use_sk {
            if in_features >= 16384 {
                if let Some(pipeline) = self
                    .ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_colmajor_contig4_sk4")
                {
                    Self::trace_gemv("linear_gemv_ishizuki_k3_colmajor_contig4_sk4");
                    let tiles = (out_features / 16) as u64;
                    let groups = tiles * 4;
                    let shape_values = [
                        to_u32(in_features, "input dimension")?,
                        to_u32(out_features, "output dimension")?,
                    ];
                    encoder.dispatch_1d_offset(
                        pipeline,
                        &[
                            TensorRef::new(packed, packed_offset),
                            TensorRef::new(transformed, 0),
                            TensorRef::new(raw, 0),
                        ],
                        params_bytes(&shape_values),
                        groups * 256,
                        256,
                    )?;
                    return encoder.dispatch_1d(
                        &self.finalize_simd_sk4_pipeline,
                        &[Some(raw), Some(output_signs), Some(output)],
                        &output_blocks.to_le_bytes(),
                        u64::from(output_blocks) * 32,
                        32,
                    );
                }
            }
            if let Some(pipeline) =
                self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_colmajor_contig4_sk2")
            {
                Self::trace_gemv("linear_gemv_ishizuki_k3_colmajor_contig4_sk2");
                let tiles = (out_features / 16) as u64;
                let groups = tiles * 2;
                let shape_values = [
                    to_u32(in_features, "input dimension")?,
                    to_u32(out_features, "output dimension")?,
                ];
                encoder.dispatch_1d_offset(
                    pipeline,
                    &[
                        TensorRef::new(packed, packed_offset),
                        TensorRef::new(transformed, 0),
                        TensorRef::new(raw, 0),
                    ],
                    params_bytes(&shape_values),
                    groups * 256,
                    256,
                )?;
                return encoder.dispatch_1d(
                    &self.finalize_simd_sk2_pipeline,
                    &[Some(raw), Some(output_signs), Some(output)],
                    &output_blocks.to_le_bytes(),
                    u64::from(output_blocks) * 32,
                    32,
                );
            }
        }

        self.encode_gemv_raw(
            encoder,
            packed,
            packed_offset,
            perm_mode,
            rate,
            codebook,
            in_features,
            out_features,
            transformed,
            raw,
        )?;
        encoder.dispatch_1d(
            &self.finalize_simd_pipeline,
            &[Some(raw), Some(output_signs), Some(output)],
            &output_blocks.to_le_bytes(),
            u64::from(output_blocks) * 32,
            32,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_gemv_raw(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        packed: &Buffer,
        packed_offset: u64,
        perm_mode: u32,
        rate: Exl3Rate,
        codebook: Exl3Codebook,
        in_features: usize,
        out_features: usize,
        transformed: &Buffer,
        raw: &Buffer,
    ) -> Result<(), KernelError> {
        let params = LinearParams {
            in_features: to_u32(in_features, "input dimension")?,
            out_features: to_u32(out_features, "output dimension")?,
            tile_columns: to_u32(out_features / 16, "tile column count")?,
            packed_words: to_u32(rate.packed_words(), "packed shape")?,
            rate: rate.encoded(),
            codebook: codebook.encoded(),
            queries: 1,
            perm_mode,
        };
        if perm_mode == 1 && out_features % 64 == 0 && env_flag("KAPLUMBA_4COL") {
            if let Exl3Rate::Integer(bits) = rate {
                let name = format!("linear_gemv_ishizuki_k{bits}_colmajor_4col");
                if let Some(pipeline) = self.ishizuki_gemv_by_name.get(&name) {
                    Self::trace_gemv(&name);
                    let shape_values = [
                        to_u32(in_features, "input dimension")?,
                        to_u32(out_features, "output dimension")?,
                    ];
                    let quads = (out_features / 64) as u64;
                    let groups = quads.min(gemv_max_groups()).max(1);
                    return encoder.dispatch_1d_offset(
                        pipeline,
                        &[
                            TensorRef::new(packed, packed_offset),
                            TensorRef::new(transformed, 0),
                            TensorRef::new(raw, 0),
                        ],
                        params_bytes(&shape_values),
                        groups * 256,
                        256,
                    );
                }
            }
        }
        if perm_mode == 1 && out_features % 32 == 0 && !env_flag("KAPLUMBA_NO_2COL") {
            if let Exl3Rate::Integer(bits) = rate {
                let name = format!("linear_gemv_ishizuki_k{bits}_colmajor_2col");
                if let Some(pipeline) = self.ishizuki_gemv_by_name.get(&name) {
                    Self::trace_gemv(&name);
                    let shape_values = [
                        to_u32(in_features, "input dimension")?,
                        to_u32(out_features, "output dimension")?,
                    ];
                    let pairs = (out_features / 32) as u64;
                    let groups = pairs.min(gemv_max_groups()).max(1);
                    return encoder.dispatch_1d_offset(
                        pipeline,
                        &[
                            TensorRef::new(packed, packed_offset),
                            TensorRef::new(transformed, 0),
                            TensorRef::new(raw, 0),
                        ],
                        params_bytes(&shape_values),
                        groups * 256,
                        256,
                    );
                }
            }
        }

        if let Some(pipeline) = self.ishizuki_gemv_pipeline(rate, packed_offset, perm_mode) {
            // Trace the name that was *actually selected*, not one reconstructed
            // from the rate: the trace exists to answer "which kernel ran", and a
            // reconstructed name would keep answering the old question correctly
            // while the answer changed.
            Self::trace_gemv(
                self.repacked_gemv_by_name
                    .iter()
                    .chain(self.ishizuki_gemv_by_name.iter())
                    .find(|(_, candidate)| std::ptr::eq(*candidate, pipeline))
                    .map(|(name, _)| name.as_str())
                    .unwrap_or("linear_gemv_ishizuki_k<unknown>"),
            );
            self.encode_ishizuki_gemv(
                encoder,
                pipeline,
                packed,
                packed_offset,
                transformed,
                raw,
                in_features,
                out_features,
            )?;
        } else if let Some(pipeline) = self.decode_gemv_pipeline(rate, packed_offset, codebook) {
            if let Some(name) = self.decode_gemv_kernel_name(rate, packed_offset, codebook) {
                Self::trace_gemv(&name);
            }
            // One threadgroup per NR0 output tile-columns, so the grid shrinks by
            // NR0. The kernel derives its own `out_base = tn * 16 * NR0`, so
            // every column is still written exactly once and in the same place;
            // only the mapping from threadgroup index to columns changes.
            let tiles = out_features / 16;
            let nr0 = decode_nr0().max(1) as usize;
            if !tiles.is_multiple_of(nr0) {
                return Err(KernelError::InvalidInput(format!(
                    "out_features/16 = {tiles} is not a multiple of the NR0 axis \
                     ({nr0}); pick a shape whose output tiles divide evenly"
                )));
            }
            encoder.dispatch_1d_offset(
                pipeline,
                &[
                    TensorRef::new(packed, packed_offset),
                    TensorRef::new(transformed, 0),
                    TensorRef::new(raw, 0),
                ],
                params_bytes(&params),
                (tiles / nr0 * 256) as u64,
                256,
            )?;
        } else {
            let gemv_pipeline = self
                .fast_gemv_pipeline(rate, codebook)
                .unwrap_or(&self.gemv_pipeline);
            encoder.dispatch_1d_offset(
                gemv_pipeline,
                &[
                    TensorRef::new(packed, packed_offset),
                    TensorRef::new(transformed, 0),
                    TensorRef::new(raw, 0),
                ],
                params_bytes(&params),
                out_features as u64,
                64,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_gemv_raw_sk2(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        packed: &Buffer,
        packed_offset: u64,
        perm_mode: u32,
        rate: Exl3Rate,
        in_features: usize,
        out_features: usize,
        transformed: &Buffer,
        raw: &Buffer,
    ) -> Result<(), KernelError> {
        if perm_mode == 1 && out_features % 32 == 0 {
            if let Exl3Rate::Integer(bits) = rate {
                let use_contig4 = env_flag("KAPLUMBA_SK2_CONTIG4");
                // `_1col_sk2` is the default. One output column per threadgroup
                // instead of two doubles the grid -- 672 threadgroups against
                // 336 for `o_proj`/`down_proj` at N = 5376 -- and halves the
                // per-threadgroup reduction. Measured on `Gxma-repacked` at
                // 16,536-token context, `--chat --temperature 0`, same prompt,
                // median of 23: **76.35 ms -> 75.57 ms**, with all 90 greedy
                // token ids byte-identical. Opt out with `KAPLUMBA_NO_SK2_1COL`.
                let use_1col = !env_flag("KAPLUMBA_NO_SK2_1COL");
                let (name, groups) = if use_contig4 {
                    (
                        format!("linear_gemv_ishizuki_k{bits}_colmajor_contig4_sk2"),
                        ((out_features / 16) as u64) * 2,
                    )
                } else if use_1col {
                    (
                        format!("linear_gemv_ishizuki_k{bits}_colmajor_1col_sk2"),
                        ((out_features / 16) as u64) * 2,
                    )
                } else {
                    (
                        format!("linear_gemv_ishizuki_k{bits}_colmajor_2col_sk2"),
                        (((out_features / 16) / 2) as u64) * 2,
                    )
                };
                if let Some(pipeline) = self.ishizuki_gemv_by_name.get(&name) {
                    Self::trace_gemv(&name);
                    let shape_values = [
                        to_u32(in_features, "input dimension")?,
                        to_u32(out_features, "output dimension")?,
                    ];
                    return encoder.dispatch_1d_offset(
                        pipeline,
                        &[
                            TensorRef::new(packed, packed_offset),
                            TensorRef::new(transformed, 0),
                            TensorRef::new(raw, 0),
                        ],
                        params_bytes(&shape_values),
                        groups * 256,
                        256,
                    );
                }
            }
        }
        Err(KernelError::InvalidInput(
            "missing sk2 pipeline for gemv_raw_sk2".to_string(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_gemv_raw_sk4(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        packed: &Buffer,
        packed_offset: u64,
        perm_mode: u32,
        rate: Exl3Rate,
        in_features: usize,
        out_features: usize,
        transformed: &Buffer,
        raw: &Buffer,
    ) -> Result<(), KernelError> {
        if perm_mode == 1 && out_features % 32 == 0 {
            if let Exl3Rate::Integer(bits) = rate {
                let name = format!("linear_gemv_ishizuki_k{bits}_colmajor_2col_sk4");
                if let Some(pipeline) = self.ishizuki_gemv_by_name.get(&name) {
                    Self::trace_gemv(&name);
                    let shape_values = [
                        to_u32(in_features, "input dimension")?,
                        to_u32(out_features, "output dimension")?,
                    ];
                    let pairs = ((out_features / 16) / 2) as u64;
                    let groups = pairs * 4;
                    return encoder.dispatch_1d_offset(
                        pipeline,
                        &[
                            TensorRef::new(packed, packed_offset),
                            TensorRef::new(transformed, 0),
                            TensorRef::new(raw, 0),
                        ],
                        params_bytes(&shape_values),
                        groups * 256,
                        256,
                    );
                }
            }
        }
        Err(KernelError::InvalidInput(
            "missing sk4 pipeline for gemv_raw_sk4".to_string(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_gemv_fused2(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        packed_a: &Buffer,
        offset_a: u64,
        out_features_a: usize,
        output_signs_a: &Buffer,
        output_a: &Buffer,
        packed_b: &Buffer,
        offset_b: u64,
        out_features_b: usize,
        output_signs_b: &Buffer,
        output_b: &Buffer,
        input: &Buffer,
        input_signs: &Buffer,
        transformed: &Buffer,
        raw_a: &Buffer,
        raw_b: &Buffer,
        in_features: usize,
        pretransformed: bool,
    ) -> Result<(), KernelError> {
        if !pretransformed {
            let input_blocks = to_u32(in_features / 128, "input dimension")?;
            encoder.dispatch_1d(
                &self.input_pipeline,
                &[Some(input), Some(input_signs), Some(transformed)],
                &input_blocks.to_le_bytes(),
                u64::from(input_blocks) * 32,
                32,
            )?;
        }
        let tiles_a = (out_features_a / 16) as u64;
        let tiles_b = (out_features_b / 16) as u64;
        let shape = [
            to_u32(in_features, "in_features")?,
            to_u32(out_features_a, "out_features_a")?,
            to_u32(out_features_b, "out_features_b")?,
        ];
        // The fused QKV GEMVs split K by default. `fused2_sk2` / `fused3_sk2`
        // launch `(tiles_a + tiles_b[+ tiles_c]) * 2` threadgroups against the
        // non-SK 2col variants' half that, which on a 20-core M4 Pro is the
        // difference between one wave and two. Measured on `Gxma-repacked` at
        // 16,536-token context, `--chat --temperature 0`, stacked on
        // `_1col_sk2`, median of 23: **75.57 ms -> 74.62 ms**, again with all
        // 90 greedy token ids byte-identical -- the f32 reassociation is
        // masked by the fp16 rounding in `linear_finalize_simd_sk2`, which is
        // what the sampler reads, which is the same property
        // `fused_bit_exact` asserts. `KAPLUMBA_USE_SK` keeps its old meaning
        // as an independent opt-in; opt out with `KAPLUMBA_NO_QKV_SK2`.
        if env_flag("KAPLUMBA_USE_SK") || !env_flag("KAPLUMBA_NO_QKV_SK2") {
            if let Some(pipeline) =
                self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_fused2_sk2")
            {
                // Traced like every sibling: without this the now-default fused2
                // SK arm was invisible to `KAPLUMBA_TRACE_GEMV`, which only made
                // `fused3_sk2` show and made the split look like it never ran.
                Self::trace_gemv("linear_gemv_ishizuki_k3_fused2_sk2");
                let groups = (tiles_a + tiles_b) * 2;
                encoder.dispatch_1d_offset(
                    pipeline,
                    &[
                        TensorRef::new(packed_a, offset_a),
                        TensorRef::new(packed_b, offset_b),
                        TensorRef::new(transformed, 0),
                        TensorRef::new(raw_a, 0),
                        TensorRef::new(raw_b, 0),
                    ],
                    params_bytes(&shape),
                    groups * 256,
                    256,
                )?;
                let blocks_a = to_u32(out_features_a / 128, "output dimension a")?;
                encoder.dispatch_1d(
                    &self.finalize_simd_sk2_pipeline,
                    &[Some(raw_a), Some(output_signs_a), Some(output_a)],
                    &blocks_a.to_le_bytes(),
                    u64::from(blocks_a) * 32,
                    32,
                )?;
                let blocks_b = to_u32(out_features_b / 128, "output dimension b")?;
                return encoder.dispatch_1d(
                    &self.finalize_simd_sk2_pipeline,
                    &[Some(raw_b), Some(output_signs_b), Some(output_b)],
                    &blocks_b.to_le_bytes(),
                    u64::from(blocks_b) * 32,
                    32,
                );
            }
        }
        let (pipeline, groups) = if out_features_a == out_features_b && !env_flag("KAPLUMBA_NO_FUSED2_CO") {
            if out_features_a % 32 == 0 && env_flag("KAPLUMBA_FUSED2_CO4") {
                if let Some(pipe) = self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_fused2_co4") {
                    (pipe, tiles_a / 2)
                } else if let Some(pipe) = self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_fused2_co") {
                    (pipe, tiles_a)
                } else {
                    (
                        self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_fused2")
                            .ok_or_else(|| {
                                KernelError::InvalidInput(
                                    "missing linear_gemv_ishizuki_k3_fused2".to_string(),
                                )
                            })?,
                        tiles_a + tiles_b,
                    )
                }
            } else if let Some(pipe) = self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_fused2_co") {
                (pipe, tiles_a)
            } else {
                (
                    self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_fused2")
                        .ok_or_else(|| {
                            KernelError::InvalidInput(
                                "missing linear_gemv_ishizuki_k3_fused2".to_string(),
                            )
                        })?,
                    tiles_a + tiles_b,
                )
            }
        } else if out_features_a % 32 == 0 && out_features_b % 32 == 0 && !env_flag("KAPLUMBA_NO_FUSED2_2COL") {
            if let Some(pipe) = self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_fused2_2col") {
                Self::trace_gemv("linear_gemv_ishizuki_k3_fused2_2col");
                (pipe, (tiles_a + tiles_b) / 2)
            } else {
                (
                    self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_fused2")
                        .ok_or_else(|| {
                            KernelError::InvalidInput(
                                "missing linear_gemv_ishizuki_k3_fused2".to_string(),
                            )
                        })?,
                    tiles_a + tiles_b,
                )
            }
        } else {
            (
                self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_fused2")
                    .ok_or_else(|| {
                        KernelError::InvalidInput(
                            "missing linear_gemv_ishizuki_k3_fused2".to_string(),
                        )
                    })?,
                tiles_a + tiles_b,
            )
        };
        encoder.dispatch_1d_offset(
            pipeline,
            &[
                TensorRef::new(packed_a, offset_a),
                TensorRef::new(packed_b, offset_b),
                TensorRef::new(transformed, 0),
                TensorRef::new(raw_a, 0),
                TensorRef::new(raw_b, 0),
            ],
            params_bytes(&shape),
            groups * 256,
            256,
        )?;
        let blocks_a = to_u32(out_features_a / 128, "output dimension a")?;
        encoder.dispatch_1d(
            &self.finalize_simd_pipeline,
            &[Some(raw_a), Some(output_signs_a), Some(output_a)],
            &blocks_a.to_le_bytes(),
            u64::from(blocks_a) * 32,
            32,
        )?;
        let blocks_b = to_u32(out_features_b / 128, "output dimension b")?;
        encoder.dispatch_1d(
            &self.finalize_simd_pipeline,
            &[Some(raw_b), Some(output_signs_b), Some(output_b)],
            &blocks_b.to_le_bytes(),
            u64::from(blocks_b) * 32,
            32,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_gemv_fused2_mlp(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        packed_gate: &Buffer,
        offset_gate: u64,
        out_features_gate: usize,
        gate_out_signs: &Buffer,
        packed_up: &Buffer,
        offset_up: u64,
        out_features_up: usize,
        up_out_signs: &Buffer,
        down_input_signs: Option<&Buffer>,
        activated_or_transformed: &Buffer,
        input: &Buffer,
        input_signs: &Buffer,
        transformed: &Buffer,
        raw_gate: &Buffer,
        raw_up: &Buffer,
        in_features: usize,
        pretransformed: bool,
    ) -> Result<(), KernelError> {
        if !pretransformed {
            let input_blocks = to_u32(in_features / 128, "input dimension")?;
            encoder.dispatch_1d(
                &self.input_pipeline,
                &[Some(input), Some(input_signs), Some(transformed)],
                &input_blocks.to_le_bytes(),
                u64::from(input_blocks) * 32,
                32,
            )?;
        }
        let tiles_a = (out_features_gate / 16) as u64;
        let tiles_b = (out_features_up / 16) as u64;
        let shape = [
            to_u32(in_features, "in_features")?,
            to_u32(out_features_gate, "out_features_gate")?,
            to_u32(out_features_up, "out_features_up")?,
        ];
        if env_flag("KAPLUMBA_USE_SK") {
            if let Some(pipeline) =
                self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_fused2_sk2")
            {
                Self::trace_gemv("linear_gemv_ishizuki_k3_fused2_sk2 (mlp)");
                let groups = (tiles_a + tiles_b) * 2;
                encoder.dispatch_1d_offset(
                    pipeline,
                    &[
                        TensorRef::new(packed_gate, offset_gate),
                        TensorRef::new(packed_up, offset_up),
                        TensorRef::new(transformed, 0),
                        TensorRef::new(raw_gate, 0),
                        TensorRef::new(raw_up, 0),
                    ],
                    params_bytes(&shape),
                    groups * 256,
                    256,
                )?;
                let blocks = to_u32(out_features_gate / 128, "features dimension")?;
                if let Some(down_signs) = down_input_signs {
                    return encoder.dispatch_1d(
                        &self.finalize_gated_gelu_and_down_input_sk2_pipeline,
                        &[
                            Some(raw_gate),
                            Some(gate_out_signs),
                            Some(raw_up),
                            Some(up_out_signs),
                            Some(down_signs),
                            Some(activated_or_transformed),
                        ],
                        &blocks.to_le_bytes(),
                        u64::from(blocks) * 32,
                        32,
                    );
                } else {
                    return encoder.dispatch_1d(
                        &self.finalize_gated_gelu_sk2_pipeline,
                        &[
                            Some(raw_gate),
                            Some(gate_out_signs),
                            Some(raw_up),
                            Some(up_out_signs),
                            Some(activated_or_transformed),
                        ],
                        &blocks.to_le_bytes(),
                        u64::from(blocks) * 32,
                        32,
                    );
                }
            }
        }
        let (pipeline, groups) = if env_flag("KAPLUMBA_FUSED2_4COL")
            && out_features_gate % 64 == 0
            && out_features_up % 64 == 0
            && self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_fused2_4col").is_some()
        {
            let pipe = self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_fused2_4col").unwrap();
            Self::trace_gemv("linear_gemv_ishizuki_k3_fused2_4col (mlp)");
            (pipe, (tiles_a + tiles_b) / 4)
        } else if out_features_gate == out_features_up && !env_flag("KAPLUMBA_NO_FUSED2_CO") {
            if out_features_gate % 32 == 0 && env_flag("KAPLUMBA_FUSED2_CO4") {
                if let Some(pipe) = self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_fused2_co4") {
                    Self::trace_gemv("linear_gemv_ishizuki_k3_fused2_co4 (mlp)");
                    (pipe, tiles_a / 2)
                } else if let Some(pipe) = self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_fused2_co") {
                    Self::trace_gemv("linear_gemv_ishizuki_k3_fused2_co (mlp)");
                    (pipe, tiles_a)
                } else {
                    (
                        self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_fused2")
                            .ok_or_else(|| {
                                KernelError::InvalidInput(
                                    "missing linear_gemv_ishizuki_k3_fused2".to_string(),
                                )
                            })?,
                        tiles_a + tiles_b,
                    )
                }
            } else if let Some(pipe) = self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_fused2_co") {
                Self::trace_gemv("linear_gemv_ishizuki_k3_fused2_co (mlp)");
                (pipe, tiles_a)
            } else {
                (
                    self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_fused2")
                        .ok_or_else(|| {
                            KernelError::InvalidInput(
                                "missing linear_gemv_ishizuki_k3_fused2".to_string(),
                            )
                        })?,
                    tiles_a + tiles_b,
                )
            }
        } else if !env_flag("KAPLUMBA_NO_FUSED2_2COL")
            && out_features_gate % 32 == 0
            && out_features_up % 32 == 0
            && self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_fused2_2col").is_some()
        {
            let pipe = self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_fused2_2col").unwrap();
            Self::trace_gemv("linear_gemv_ishizuki_k3_fused2_2col (mlp)");
            (pipe, (tiles_a + tiles_b) / 2)
        } else {
            (
                self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_fused2")
                    .ok_or_else(|| {
                        KernelError::InvalidInput(
                            "missing linear_gemv_ishizuki_k3_fused2".to_string(),
                        )
                    })?,
                tiles_a + tiles_b,
            )
        };
        encoder.dispatch_1d_offset(
            pipeline,
            &[
                TensorRef::new(packed_gate, offset_gate),
                TensorRef::new(packed_up, offset_up),
                TensorRef::new(transformed, 0),
                TensorRef::new(raw_gate, 0),
                TensorRef::new(raw_up, 0),
            ],
            params_bytes(&shape),
            groups * 256,
            256,
        )?;
        let blocks = to_u32(out_features_gate / 128, "features dimension")?;
        if let Some(down_signs) = down_input_signs {
            encoder.dispatch_1d(
                &self.finalize_gated_gelu_and_down_input_pipeline,
                &[
                    Some(raw_gate),
                    Some(gate_out_signs),
                    Some(raw_up),
                    Some(up_out_signs),
                    Some(down_signs),
                    Some(activated_or_transformed),
                ],
                &blocks.to_le_bytes(),
                u64::from(blocks) * 32,
                32,
            )
        } else {
            encoder.dispatch_1d(
                &self.finalize_gated_gelu_pipeline,
                &[
                    Some(raw_gate),
                    Some(gate_out_signs),
                    Some(raw_up),
                    Some(up_out_signs),
                    Some(activated_or_transformed),
                ],
                &blocks.to_le_bytes(),
                u64::from(blocks) * 32,
                32,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_gemv_fused3(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        packed_a: &Buffer,
        offset_a: u64,
        out_features_a: usize,
        output_signs_a: &Buffer,
        output_a: &Buffer,
        packed_b: &Buffer,
        offset_b: u64,
        out_features_b: usize,
        output_signs_b: &Buffer,
        output_b: &Buffer,
        packed_c: &Buffer,
        offset_c: u64,
        out_features_c: usize,
        output_signs_c: &Buffer,
        output_c: &Buffer,
        input: &Buffer,
        input_signs: &Buffer,
        transformed: &Buffer,
        raw_a: &Buffer,
        raw_b: &Buffer,
        raw_c: &Buffer,
        in_features: usize,
        pretransformed: bool,
    ) -> Result<(), KernelError> {
        if !pretransformed {
            let input_blocks = to_u32(in_features / 128, "input dimension")?;
            encoder.dispatch_1d(
                &self.input_pipeline,
                &[Some(input), Some(input_signs), Some(transformed)],
                &input_blocks.to_le_bytes(),
                u64::from(input_blocks) * 32,
                32,
            )?;
        }
        let tiles_a = (out_features_a / 16) as u64;
        let tiles_b = (out_features_b / 16) as u64;
        let tiles_c = (out_features_c / 16) as u64;
        let shape = [
            to_u32(in_features, "in_features")?,
            to_u32(out_features_a, "out_features_a")?,
            to_u32(out_features_b, "out_features_b")?,
            to_u32(out_features_c, "out_features_c")?,
        ];
        // The fused QKV GEMVs split K by default. `fused2_sk2` / `fused3_sk2`
        // launch `(tiles_a + tiles_b[+ tiles_c]) * 2` threadgroups against the
        // non-SK 2col variants' half that, which on a 20-core M4 Pro is the
        // difference between one wave and two. Measured on `Gxma-repacked` at
        // 16,536-token context, `--chat --temperature 0`, stacked on
        // `_1col_sk2`, median of 23: **75.57 ms -> 74.62 ms**, again with all
        // 90 greedy token ids byte-identical -- the f32 reassociation is
        // masked by the fp16 rounding in `linear_finalize_simd_sk2`, which is
        // what the sampler reads, which is the same property
        // `fused_bit_exact` asserts. `KAPLUMBA_USE_SK` keeps its old meaning
        // as an independent opt-in; opt out with `KAPLUMBA_NO_QKV_SK2`.
        if env_flag("KAPLUMBA_USE_SK") || !env_flag("KAPLUMBA_NO_QKV_SK2") {
            if let Some(pipeline) =
                self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_fused3_sk2")
            {
                Self::trace_gemv("linear_gemv_ishizuki_k3_fused3_sk2");
                let groups = (tiles_a + tiles_b + tiles_c) * 2;
                encoder.dispatch_1d_offset(
                    pipeline,
                    &[
                        TensorRef::new(packed_a, offset_a),
                        TensorRef::new(packed_b, offset_b),
                        TensorRef::new(packed_c, offset_c),
                        TensorRef::new(transformed, 0),
                        TensorRef::new(raw_a, 0),
                        TensorRef::new(raw_b, 0),
                        TensorRef::new(raw_c, 0),
                    ],
                    params_bytes(&shape),
                    groups * 256,
                    256,
                )?;
                let blocks_a = to_u32(out_features_a / 128, "output dimension a")?;
                let blocks_b = to_u32(out_features_b / 128, "output dimension b")?;
                let blocks_c = to_u32(out_features_c / 128, "output dimension c")?;
                let blocks_abc = [blocks_a, blocks_b, blocks_c];
                let total_blocks = (blocks_a + blocks_b + blocks_c) as u64;
                return encoder.dispatch_1d(
                    &self.finalize_simd3_sk2_pipeline,
                    &[
                        Some(raw_a),
                        Some(output_signs_a),
                        Some(output_a),
                        Some(raw_b),
                        Some(output_signs_b),
                        Some(output_b),
                        Some(raw_c),
                        Some(output_signs_c),
                        Some(output_c),
                    ],
                    params_bytes(&blocks_abc),
                    total_blocks * 32,
                    32,
                );
            }
        }
        let (pipeline, groups) = if out_features_a % 32 == 0
            && out_features_b % 32 == 0
            && out_features_c % 32 == 0
            && !env_flag("KAPLUMBA_NO_FUSED3_2COL")
        {
            if let Some(pipe) = self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_fused3_2col") {
                Self::trace_gemv("linear_gemv_ishizuki_k3_fused3_2col");
                (pipe, (tiles_a + tiles_b + tiles_c) / 2)
            } else {
                (
                    self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_fused3")
                        .ok_or_else(|| {
                            KernelError::InvalidInput("missing linear_gemv_ishizuki_k3_fused3".to_string())
                        })?,
                    tiles_a + tiles_b + tiles_c,
                )
            }
        } else {
            (
                self.ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_fused3")
                    .ok_or_else(|| {
                        KernelError::InvalidInput("missing linear_gemv_ishizuki_k3_fused3".to_string())
                    })?,
                tiles_a + tiles_b + tiles_c,
            )
        };
        encoder.dispatch_1d_offset(
            pipeline,
            &[
                TensorRef::new(packed_a, offset_a),
                TensorRef::new(packed_b, offset_b),
                TensorRef::new(packed_c, offset_c),
                TensorRef::new(transformed, 0),
                TensorRef::new(raw_a, 0),
                TensorRef::new(raw_b, 0),
                TensorRef::new(raw_c, 0),
            ],
            params_bytes(&shape),
            groups * 256,
            256,
        )?;
        let blocks_a = to_u32(out_features_a / 128, "output dimension a")?;
        let blocks_b = to_u32(out_features_b / 128, "output dimension b")?;
        let blocks_c = to_u32(out_features_c / 128, "output dimension c")?;
        let blocks_abc = [blocks_a, blocks_b, blocks_c];
        let total_blocks = (blocks_a + blocks_b + blocks_c) as u64;
        encoder.dispatch_1d(
            &self.finalize_simd3_pipeline,
            &[
                Some(raw_a),
                Some(output_signs_a),
                Some(output_a),
                Some(raw_b),
                Some(output_signs_b),
                Some(output_b),
                Some(raw_c),
                Some(output_signs_c),
                Some(output_c),
            ],
            params_bytes(&blocks_abc),
            total_blocks * 32,
            32,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_finalize_and_post_ffn_norm_scale(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        raw: &Buffer,
        output_signs: &Buffer,
        weight: &Buffer,
        residual: &Buffer,
        output: &Buffer,
        width: usize,
        eps: f32,
        scale: f32,
    ) -> Result<(), KernelError> {
        let blocks = to_u32(width / 128, "finalize blocks")?;
        let params = FinalizeNormScaleParams {
            width: to_u32(width, "finalize width")?,
            blocks,
            eps,
            scale,
        };
        if crate::ablated("finalize_post_ffn") {
            return Ok(());
        }
        encoder.dispatch_groups(
            &self.finalize_and_post_ffn_norm_scale_pipeline,
            &[
                Some(raw),
                Some(output_signs),
                Some(weight),
                Some(residual),
                Some(output),
            ],
            params_bytes(&params),
            1,
            256,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_finalize_and_post_ffn_norm_scale_sk2(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        raw: &Buffer,
        output_signs: &Buffer,
        weight: &Buffer,
        residual: &Buffer,
        output: &Buffer,
        width: usize,
        eps: f32,
        scale: f32,
    ) -> Result<(), KernelError> {
        let blocks = to_u32(width / 128, "finalize blocks")?;
        let params = FinalizeNormScaleParams {
            width: to_u32(width, "finalize width")?,
            blocks,
            eps,
            scale,
        };
        if crate::ablated("finalize_post_ffn") {
            return Ok(());
        }
        encoder.dispatch_groups(
            &self.finalize_and_post_ffn_norm_scale_sk2_pipeline,
            &[
                Some(raw),
                Some(output_signs),
                Some(weight),
                Some(residual),
                Some(output),
            ],
            params_bytes(&params),
            1,
            1024,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_finalize_and_post_ffn_norm_scale_sk4(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        raw: &Buffer,
        output_signs: &Buffer,
        weight: &Buffer,
        residual: &Buffer,
        output: &Buffer,
        width: usize,
        eps: f32,
        scale: f32,
    ) -> Result<(), KernelError> {
        let blocks = to_u32(width / 128, "finalize blocks")?;
        let params = FinalizeNormScaleParams {
            width: to_u32(width, "finalize width")?,
            blocks,
            eps,
            scale,
        };
        if crate::ablated("finalize_post_ffn") {
            return Ok(());
        }
        encoder.dispatch_groups(
            &self.finalize_and_post_ffn_norm_scale_sk4_pipeline,
            &[
                Some(raw),
                Some(output_signs),
                Some(weight),
                Some(residual),
                Some(output),
            ],
            params_bytes(&params),
            1,
            1024,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_post_attn_pre_ffn_transform(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        proj: &Buffer,
        post_attn_norm: &Buffer,
        residual: &Buffer,
        hidden_b: &Buffer,
        pre_ffn_norm: &Buffer,
        gate_input_signs: &Buffer,
        transformed: &Buffer,
        width: usize,
        eps: f32,
    ) -> Result<(), KernelError> {
        let blocks = to_u32(width / 128, "post_attn blocks")?;
        let params = PostAttnPreFfnParams {
            width: to_u32(width, "post_attn width")?,
            blocks,
            eps,
        };
        if crate::ablated("post_attn_pre_ffn") {
            return Ok(());
        }
        encoder.dispatch_groups(
            &self.post_attn_pre_ffn_transform_pipeline,
            &[
                Some(proj),
                Some(post_attn_norm),
                Some(residual),
                Some(hidden_b),
                Some(pre_ffn_norm),
                Some(gate_input_signs),
                Some(transformed),
            ],
            params_bytes(&params),
            1,
            256,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_finalize_o_proj_and_pre_ffn_transform(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        raw: &Buffer,
        o_out_signs: &Buffer,
        post_attn_norm: &Buffer,
        residual: &Buffer,
        hidden_b: &Buffer,
        pre_ffn_norm: &Buffer,
        gate_input_signs: &Buffer,
        transformed: &Buffer,
        width: usize,
        eps: f32,
    ) -> Result<(), KernelError> {
        let blocks = to_u32(width / 128, "post_attn blocks")?;
        let params = PostAttnPreFfnParams {
            width: to_u32(width, "post_attn width")?,
            blocks,
            eps,
        };
        if crate::ablated("finalize_o_proj") {
            return Ok(());
        }
        encoder.dispatch_groups(
            &self.finalize_o_proj_and_pre_ffn_transform_pipeline,
            &[
                Some(raw),
                Some(o_out_signs),
                Some(post_attn_norm),
                Some(residual),
                Some(hidden_b),
                Some(pre_ffn_norm),
                Some(gate_input_signs),
                Some(transformed),
            ],
            params_bytes(&params),
            1,
            256,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_finalize_o_proj_and_pre_ffn_transform_sk2(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        raw: &Buffer,
        o_out_signs: &Buffer,
        post_attn_norm: &Buffer,
        residual: &Buffer,
        hidden_b: &Buffer,
        pre_ffn_norm: &Buffer,
        gate_input_signs: &Buffer,
        transformed: &Buffer,
        width: usize,
        eps: f32,
    ) -> Result<(), KernelError> {
        let blocks = to_u32(width / 128, "post_attn blocks")?;
        let params = PostAttnPreFfnParams {
            width: to_u32(width, "post_attn width")?,
            blocks,
            eps,
        };
        if crate::ablated("finalize_o_proj") {
            return Ok(());
        }
        encoder.dispatch_groups(
            &self.finalize_o_proj_and_pre_ffn_transform_sk2_pipeline,
            &[
                Some(raw),
                Some(o_out_signs),
                Some(post_attn_norm),
                Some(residual),
                Some(hidden_b),
                Some(pre_ffn_norm),
                Some(gate_input_signs),
                Some(transformed),
            ],
            params_bytes(&params),
            1,
            1024,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_finalize_o_proj_and_pre_ffn_transform_sk4(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        raw: &Buffer,
        o_out_signs: &Buffer,
        post_attn_norm: &Buffer,
        residual: &Buffer,
        hidden_b: &Buffer,
        pre_ffn_norm: &Buffer,
        gate_input_signs: &Buffer,
        transformed: &Buffer,
        width: usize,
        eps: f32,
    ) -> Result<(), KernelError> {
        let blocks = to_u32(width / 128, "post_attn blocks")?;
        let params = PostAttnPreFfnParams {
            width: to_u32(width, "post_attn width")?,
            blocks,
            eps,
        };
        if crate::ablated("finalize_o_proj") {
            return Ok(());
        }
        encoder.dispatch_groups(
            &self.finalize_o_proj_and_pre_ffn_transform_sk4_pipeline,
            &[
                Some(raw),
                Some(o_out_signs),
                Some(post_attn_norm),
                Some(residual),
                Some(hidden_b),
                Some(pre_ffn_norm),
                Some(gate_input_signs),
                Some(transformed),
            ],
            params_bytes(&params),
            1,
            1024,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_input_norm_and_qkv_transform(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        input: &Buffer,
        input_norm: &Buffer,
        q_input_signs: &Buffer,
        normed: &Buffer,
        transformed: &Buffer,
        width: usize,
        eps: f32,
    ) -> Result<(), KernelError> {
        let blocks = to_u32(width / 128, "input_norm blocks")?;
        let params = PostAttnPreFfnParams {
            width: to_u32(width, "input_norm width")?,
            blocks,
            eps,
        };
        if crate::ablated("input_norm_qkv") {
            return Ok(());
        }
        encoder.dispatch_groups(
            &self.input_norm_and_qkv_transform_pipeline,
            &[
                Some(input),
                Some(input_norm),
                Some(q_input_signs),
                Some(normed),
                Some(transformed),
            ],
            params_bytes(&params),
            1,
            256,
        )
    }

    /// T3: layer L's finalize + post_ffn_norm + residual add, fused with
    /// layer L+1's input_layernorm + q input-sign transform.
    ///
    /// `output` is still written. The residual stream is read by this layer's
    /// own `o_proj` finalize on the way in and by the lm_head path at the very
    /// end, so eliminating the store would not eliminate the data; what the
    /// fusion removes is the round trip through it and one dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_cross_layer_finalize_and_norm(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        raw: &Buffer,
        output_signs: &Buffer,
        post_ffn_norm: &Buffer,
        residual: &Buffer,
        output: &Buffer,
        next_input_norm: &Buffer,
        next_q_signs: &Buffer,
        normed: &Buffer,
        transformed: &Buffer,
        width: usize,
        eps: f32,
        scale: f32,
    ) -> Result<(), KernelError> {
        if crate::ablated("finalize_post_ffn") {
            return Ok(());
        }
        let params = CrossLayerParams {
            width: to_u32(width, "cross-layer width")?,
            blocks: to_u32(width / 128, "cross-layer blocks")?,
            eps,
            scale,
        };
        // 1024 threads, because the layer-L half is the `_sk2` finalize and its
        // 1024-thread reduction tree is what makes it bit-identical to the
        // 256-thread original.
        encoder.dispatch_groups(
            &self.cross_layer_pipeline,
            &[
                Some(raw),
                Some(output_signs),
                Some(post_ffn_norm),
                Some(residual),
                Some(output),
                Some(next_input_norm),
                Some(next_q_signs),
                Some(normed),
                Some(transformed),
            ],
            params_bytes(&params),
            1,
            1024,
        )
    }

    /// Encode a multi-query (prefill) linear entirely from GPU buffers.
    pub fn encode_gemm(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        request: PrefillEncodeRequest<'_>,
    ) -> Result<(), KernelError> {
        let PrefillEncodeRequest {
            packed,
            packed_offset,
            perm_mode,
            input,
            input_signs,
            output_signs,
            transformed,
            raw,
            output,
            queries,
            in_features,
            out_features,
            rate,
            codebook,
        } = request;
        if queries == 0 {
            return Err(KernelError::InvalidInput(
                "prefill query count must be non-zero".to_string(),
            ));
        }
        validate_encode(in_features, out_features, rate, codebook)?;
        let input_elements = checked_product(queries, in_features, "prefill input")?;
        let output_elements = checked_product(queries, out_features, "prefill output")?;
        let row_params = PrefillRowParams {
            in_features: to_u32(in_features, "input dimension")?,
            out_features: to_u32(out_features, "output dimension")?,
            queries: to_u32(queries, "query count")?,
            perm_mode,
        };
        let linear_params = LinearParams {
            in_features: to_u32(in_features, "input dimension")?,
            out_features: to_u32(out_features, "output dimension")?,
            tile_columns: to_u32(out_features / 16, "tile column count")?,
            packed_words: to_u32(rate.packed_words(), "packed shape")?,
            rate: rate.encoded(),
            codebook: codebook.encoded(),
            queries: to_u32(queries, "query count")?,
            perm_mode,
        };
        // Tensor-core path: the mma kernel reads the signed+transformed input
        // and STORES (not accumulates) every output block as f32, so it skips
        // the raw zeroing the tiled kernel needs. Output signs are still
        // applied by the shared finalize below, exactly as the other paths do.
        //
        // DISABLED by default (8f1a863 shipped it enabled and it is numerically
        // wrong in situ: the CLI emits garbage on every reference prompt, even
        // though `linear_metal` passes 6/6 against a CPU reference, so the bug
        // is in the integration — raw buffer type / the input_rows transform /
        // the finalize contract — not in the standalone kernel. Its arithmetic
        // speed is real (10.2 -> 22.6 tok/s, ~9.4 TFLOPS) and worth fixing, but
        // a fast wrong answer is worth nothing. Set
        // KAPLUMBA_DISABLE_MMA_PREFILL=1 to A/B it while debugging.
        // Enabled by default: the bstage overflow that made this path emit
        // garbage (8f1a863) is fixed, and `mma_prefill_gemm_matches_cpu_reference`
        // now actually covers it. Set KAPLUMBA_DISABLE_MMA_PREFILL=1 to fall
        // back to the scalar tiled kernel for A/B.
        if !env_flag("KAPLUMBA_DISABLE_MMA_PREFILL")
            && in_features % 64 == 0
            && self.gemm_mma_pipeline.thread_execution_width() == 32
        {
            // `linear_gemm_mma` is a 128-query x 64-output tile on 256 threads
            // (see its header). It needs both dimensions to be multiples of 64
            // and the device to allow 256 threads per threadgroup; otherwise the
            // scalar tiled kernel below runs instead. `validate_encode` demands
            // multiples of 128, so in practice the guard only ever fires on a
            // hypothetical small device.
            //
            // T4 adds two research tiles behind their own switches:
            //   KAPLUMBA_MMA_4SG       -> linear_gemm_mma_4sg (128q x 32n, 128
            //                             threads, grid x = out/32)
            //   KAPLUMBA_REGPIPE_K32   -> linear_gemm_mma_regpipe_k32
            //   KAPLUMBA_MMA_128X64    -> force the original tile (A/B control
            //                             for the T4-measured regpipe default)
            // The default (no flag) is now REGPIPE: T4 measured it +4.7% GEMM
            // at 614 tokens (6035 -> 5753 ms, 5.85 -> 6.13 TFLOPS) and it is
            // bit-identical to linear_gemm_mma, so the eligible MCG-3bpw arm
            // ships it. Everything the research gate excludes still falls back
            // to linear_gemm_mma below.
            // Both share regpipe's MCG-3bpw host gate (4sg carries only the
            // folded-MCG arm, see its header). Selection is funnelled
            // through one `variant` so a test can force any of the four with
            // `force_gemm_mma_variant` (env_flag memoises, tests cannot flip
            // it mid-process).
            let forced = GEMM_MMA_VARIANT_FORCED.load(std::sync::atomic::Ordering::Relaxed);
            let mut variant = if forced != GEMM_MMA_VARIANT_AUTO {
                forced
            } else if env_flag("KAPLUMBA_MMA_4SG") {
                GEMM_MMA_VARIANT_4SG
            } else if env_flag("KAPLUMBA_REGPIPE_K32") {
                GEMM_MMA_VARIANT_REGPIPE_K32
            } else if env_flag("KAPLUMBA_REGPIPE") {
                GEMM_MMA_VARIANT_REGPIPE
            } else if env_flag("KAPLUMBA_MMA_128X64") {
                GEMM_MMA_VARIANT_128X64
            } else {
                // T4 default: regpipe for eligible MCG-3bpw checkpoints.
                GEMM_MMA_VARIANT_REGPIPE
            };
            // The three research tiles are gated on the folded-MCG arm,
            // exactly as regpipe always was (4sg carries that arm alone -- no
            // `fractional_window` / `params.codebook`, see its header): standard
            // linear_gemm_mma is verified across all layouts, so anything the
            // gate excludes falls back to it (never to a disabled path).
            let research_eligible =
                perm_mode != 1 && rate == Exl3Rate::Integer(3) && codebook == Exl3Codebook::Mcg;
            if (variant == GEMM_MMA_VARIANT_REGPIPE
                || variant == GEMM_MMA_VARIANT_REGPIPE_K32
                || variant == GEMM_MMA_VARIANT_4SG)
                && !research_eligible
            {
                variant = GEMM_MMA_VARIANT_128X64;
            }
            let max_threads = self.gemm_mma_pipeline.max_total_threads_per_threadgroup();
            let selected: Option<(u8, &ComputePipelineState, u64, u64)> = match variant {
                GEMM_MMA_VARIANT_4SG if out_features % 32 == 0 && max_threads >= 128 => Some((
                    GEMM_MMA_VARIANT_4SG,
                    &self.gemm_mma_4sg_pipeline,
                    (out_features / 32) as u64,
                    128,
                )),
                GEMM_MMA_VARIANT_REGPIPE if out_features % 64 == 0 && max_threads >= 256 => Some((
                    GEMM_MMA_VARIANT_REGPIPE,
                    &self.gemm_mma_regpipe_pipeline,
                    (out_features / 64) as u64,
                    256,
                )),
                GEMM_MMA_VARIANT_REGPIPE_K32 if out_features % 64 == 0 && max_threads >= 256 => {
                    Some((
                        GEMM_MMA_VARIANT_REGPIPE_K32,
                        &self.gemm_mma_regpipe_k32_pipeline,
                        (out_features / 64) as u64,
                        256,
                    ))
                }
                _ if out_features % 64 == 0 && max_threads >= 256 => Some((
                    GEMM_MMA_VARIANT_128X64,
                    &self.gemm_mma_pipeline,
                    (out_features / 64) as u64,
                    256,
                )),
                _ => None,
            };
            if let Some((dispatched, gemm, grid_x, threads)) = selected {
                GEMM_MMA_LAST_DISPATCHED.store(dispatched, std::sync::atomic::Ordering::Relaxed);
                encoder.dispatch_1d(
                    &self.input_rows_pipeline,
                    &[Some(input), Some(input_signs), Some(transformed)],
                    params_bytes(&row_params),
                    (input_elements / 128) as u64,
                    64,
                )?;
                encoder.dispatch_2d(
                    gemm,
                    &[
                        TensorRef::new(packed, packed_offset),
                        TensorRef::new(transformed, 0),
                        TensorRef::new(raw, 0),
                    ],
                    params_bytes(&linear_params),
                    grid_x,
                    queries.div_ceil(128) as u64,
                    threads,
                    1,
                )?;
                return encoder.dispatch_1d(
                    &self.finalize_rows_pipeline,
                    &[Some(raw), Some(output_signs), Some(output)],
                    params_bytes(&row_params),
                    (output_elements / 128) as u64,
                    64,
                );
            }
        }
        encoder.dispatch_1d(
            &self.input_rows_pipeline,
            &[Some(input), Some(input_signs), Some(transformed)],
            params_bytes(&row_params),
            (input_elements / 128) as u64,
            64,
        )?;
        encoder.dispatch_2d(
            &self.gemv_tiled_pipeline,
            &[
                TensorRef::new(packed, packed_offset),
                TensorRef::new(transformed, 0),
                TensorRef::new(raw, 0),
            ],
            params_bytes(&linear_params),
            (out_features / 16) as u64,
            queries.div_ceil(16) as u64,
            16,
            16,
        )?;
        encoder.dispatch_1d(
            &self.finalize_rows_pipeline,
            &[Some(raw), Some(output_signs), Some(output)],
            params_bytes(&row_params),
            (output_elements / 128) as u64,
            64,
        )
    }

    pub fn mrhs_repacked_pipeline(&self, queries: usize) -> &ComputePipelineState {
        if queries <= 4 {
            &self.mrhs_repacked_m4
        } else if queries <= 8 {
            &self.mrhs_repacked_m8
        } else if queries <= 16 {
            &self.mrhs_repacked_m16
        } else {
            &self.mrhs_repacked_m32
        }
    }

    pub fn mrhs_exl3_pipeline(&self, queries: usize) -> &ComputePipelineState {
        if queries <= 4 {
            &self.mrhs_exl3_m4
        } else if queries <= 8 {
            &self.mrhs_exl3_m8
        } else if queries <= 16 {
            &self.mrhs_exl3_m16
        } else {
            &self.mrhs_exl3_m32
        }
    }

    /// Dispatches high-bandwidth Multi-RHS GEMV for short token batches (queries <= 32).
    ///
    /// Streams model weights from DRAM at full bandwidth (~185 GB/s) while keeping weights
    /// in registers to compute across all query rows with zero inner-loop threadgroup barriers.
    /// Falls back to tensor-core `encode_gemm` when queries > 32 or when MRHS is disabled.
    pub fn encode_gemv_mrhs(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        request: PrefillEncodeRequest<'_>,
    ) -> Result<(), KernelError> {
        let PrefillEncodeRequest {
            packed,
            packed_offset,
            perm_mode,
            input,
            input_signs,
            output_signs,
            transformed,
            raw,
            output,
            queries,
            in_features,
            out_features,
            rate,
            codebook,
        } = request;

        validate_encode(in_features, out_features, rate, codebook)?;

        if queries == 0 {
            return Ok(());
        }

        // Fallback to MMA if MRHS is disabled, queries > 32, rate is not 3 bpw, or codebook is not MCG
        if env_flag("KAPLUMBA_DISABLE_MRHS")
            || queries > 32
            || rate != Exl3Rate::Integer(3)
            || codebook != Exl3Codebook::Mcg
        {
            return self.encode_gemm(
                encoder,
                PrefillEncodeRequest {
                    packed,
                    packed_offset,
                    perm_mode,
                    input,
                    input_signs,
                    output_signs,
                    transformed,
                    raw,
                    output,
                    queries,
                    in_features,
                    out_features,
                    rate,
                    codebook,
                },
            );
        }

        let row_params = PrefillRowParams {
            in_features: to_u32(in_features, "input dimension")?,
            out_features: to_u32(out_features, "output dimension")?,
            queries: to_u32(queries, "query count")?,
            perm_mode,
        };
        let input_elements = checked_product(queries, in_features, "input element count")?;
        let output_elements = checked_product(queries, out_features, "output element count")?;

        // 1. Transform all query rows of activations
        encoder.dispatch_1d(
            &self.input_rows_pipeline,
            &[Some(input), Some(input_signs), Some(transformed)],
            params_bytes(&row_params),
            (input_elements / 128) as u64,
            64,
        )?;

        // 2. Dispatch MRHS GEMV kernel
        let tiles = (out_features / 16) as u64;
        let groups = tiles.min(gemv_max_groups()).max(1);

        if perm_mode == 1 {
            let shape_values = [
                to_u32(in_features, "input dimension")?,
                to_u32(out_features, "output dimension")?,
                to_u32(queries, "query count")?,
            ];
            let pipeline = self.mrhs_repacked_pipeline(queries);
            encoder.dispatch_1d_offset(
                pipeline,
                &[
                    TensorRef::new(packed, packed_offset),
                    TensorRef::new(transformed, 0),
                    TensorRef::new(raw, 0),
                ],
                params_bytes(&shape_values),
                groups * 256,
                256,
            )?;
        } else {
            let linear_params = LinearParams {
                in_features: to_u32(in_features, "input dimension")?,
                out_features: to_u32(out_features, "output dimension")?,
                tile_columns: to_u32(out_features / 16, "tile column count")?,
                packed_words: to_u32(rate.packed_words(), "packed shape")?,
                rate: rate.encoded(),
                codebook: codebook.encoded(),
                queries: to_u32(queries, "query count")?,
                perm_mode,
            };
            let pipeline = self.mrhs_exl3_pipeline(queries);
            encoder.dispatch_1d_offset(
                pipeline,
                &[
                    TensorRef::new(packed, packed_offset),
                    TensorRef::new(transformed, 0),
                    TensorRef::new(raw, 0),
                ],
                params_bytes(&linear_params),
                groups * 256,
                256,
            )?;
        }

        // 3. Finalize all query rows
        encoder.dispatch_1d(
            &self.finalize_rows_pipeline,
            &[Some(raw), Some(output_signs), Some(output)],
            params_bytes(&row_params),
            (output_elements / 128) as u64,
            64,
        )
    }

    /// Convenience wrapper for unit testing MRHS prefill on host-provided slices.
    pub fn gemv_mrhs(
        &self,
        request: PrefillRequest<'_>,
        perm_mode: u32,
    ) -> Result<Vec<u16>, KernelError> {
        validate_prefill_dimensions(&request)?;
        let PrefillRequest {
            packed,
            input,
            input_signs,
            output_signs,
            queries,
            in_features,
            out_features,
            rate,
            codebook,
        } = request;
        let input_buffer = self.make_buffer(input)?;
        let input_sign_buffer = self.make_buffer(input_signs)?;
        let output_sign_buffer = self.make_buffer(output_signs)?;
        let packed_buffer = self.make_buffer(packed)?;
        let input_elements = checked_product(queries, in_features, "input shape")?;
        let output_elements = checked_product(queries, out_features, "output shape")?;
        let transformed = self.device.new_buffer(
            (input_elements * size_of::<u16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let raw = self.device.new_buffer(
            (output_elements * size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        zero_f32_buffer(&raw, output_elements)?;
        let output = self.device.new_buffer(
            (output_elements * size_of::<u16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let mut encoder = crate::encoder::MetalEncoder::new(&self.queue)?;
        self.encode_gemv_mrhs(
            &mut encoder,
            PrefillEncodeRequest {
                packed: &packed_buffer,
                packed_offset: 0,
                perm_mode,
                input: &input_buffer,
                input_signs: &input_sign_buffer,
                output_signs: &output_sign_buffer,
                transformed: &transformed,
                raw: &raw,
                output: &output,
                queries,
                in_features,
                out_features,
                rate,
                codebook,
            },
        )?;
        encoder.commit_and_wait()?;
        read_u16_buffer(&output)
    }


    fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        bindings: &[Option<&Buffer>],
        params: &[u8],
        threads: u64,
    ) {
        encoder.set_compute_pipeline_state(pipeline);
        for (index, binding) in bindings.iter().enumerate() {
            encoder.set_buffer(index as u64, binding.map(|value| &**value), 0);
        }
        encoder.set_bytes(
            bindings.len() as u64,
            params.len() as u64,
            params.as_ptr() as *const c_void,
        );
        encoder.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(64, 1, 1));
    }

    /// Like [`Self::encode_offset`] but with 256-thread groups.
    ///
    /// `linear_gemv_exl3` needs 256 (8 simdgroups) because it reduces across
    /// simdgroups through threadgroup memory; the older kernels use 64.
    fn encode_offset_256(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        bindings: &[TensorRef<'_>],
        params: &[u8],
        threads: u64,
    ) {
        encoder.set_compute_pipeline_state(pipeline);
        for (index, binding) in bindings.iter().enumerate() {
            encoder.set_buffer(index as u64, Some(&**binding.buffer), binding.offset);
        }
        encoder.set_bytes(
            bindings.len() as u64,
            params.len() as u64,
            params.as_ptr() as *const c_void,
        );
        encoder.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(256, 1, 1));
    }

    fn encode_offset(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        bindings: &[TensorRef<'_>],
        params: &[u8],
        threads: u64,
    ) {
        encoder.set_compute_pipeline_state(pipeline);
        for (index, binding) in bindings.iter().enumerate() {
            encoder.set_buffer(index as u64, Some(&**binding.buffer), binding.offset);
        }
        encoder.set_bytes(
            bindings.len() as u64,
            params.len() as u64,
            params.as_ptr() as *const c_void,
        );
        encoder.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(64, 1, 1));
    }

    fn encode_groups(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        bindings: &[Option<&Buffer>],
        params: &[u8],
        groups_x: u64,
        groups_y: u64,
    ) {
        encoder.set_compute_pipeline_state(pipeline);
        for (index, binding) in bindings.iter().enumerate() {
            encoder.set_buffer(index as u64, binding.map(|value| &**value), 0);
        }
        encoder.set_bytes(
            bindings.len() as u64,
            params.len() as u64,
            params.as_ptr() as *const c_void,
        );
        encoder
            .dispatch_thread_groups(MTLSize::new(groups_x, groups_y, 1), MTLSize::new(16, 16, 1));
    }

    fn make_buffer(&self, values: &[u16]) -> Result<Buffer, KernelError> {
        if values.is_empty() {
            return Err(KernelError::InvalidInput(
                "buffer cannot be empty".to_string(),
            ));
        }
        Ok(self.device.new_buffer_with_data(
            values.as_ptr() as *const c_void,
            size_of_val(values) as u64,
            MTLResourceOptions::StorageModeShared,
        ))
    }
}

/// Shape checks shared by the GPU-resident encode paths.
fn validate_encode(
    in_features: usize,
    out_features: usize,
    rate: Exl3Rate,
    codebook: Exl3Codebook,
) -> Result<(), KernelError> {
    if in_features == 0
        || out_features == 0
        || !in_features.is_multiple_of(128)
        || !out_features.is_multiple_of(128)
        || !out_features.is_multiple_of(16)
        || codebook != Exl3Codebook::Mcg
    {
        return Err(KernelError::InvalidInput(
            "linear dimensions must be non-zero multiples of 128 (output of 16) for MCG"
                .to_string(),
        ));
    }
    let _ = rate;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_buffer_shape(
    packed_bytes: u64,
    _packed_offset: u64,
    input_len: usize,
    input_sign_len: usize,
    output_sign_len: usize,
    in_features: usize,
    out_features: usize,
    rate: Exl3Rate,
    codebook: Exl3Codebook,
) -> Result<(), KernelError> {
    if in_features == 0
        || out_features == 0
        || !in_features.is_multiple_of(128)
        || !out_features.is_multiple_of(128)
        || input_len == 0
        || !input_len.is_multiple_of(in_features)
        || input_sign_len != in_features
        || output_sign_len != out_features
        || matches!(rate, Exl3Rate::Fractional(_)) && codebook != Exl3Codebook::Mul1
    {
        return Err(KernelError::InvalidInput(
            "buffer-backed linear shape is invalid".to_string(),
        ));
    }
    let expected = (in_features / 16)
        .checked_mul(out_features / 16)
        .and_then(|value| value.checked_mul(rate.packed_words()))
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| KernelError::InvalidInput("packed shape overflows".to_string()))?;
    if packed_bytes != expected as u64 {
        return Err(KernelError::InvalidInput(format!(
            "packed buffer length {packed_bytes} does not match {expected}"
        )));
    }
    Ok(())
}

fn prefill_stats(
    tokens: usize,
    in_features: usize,
    out_features: usize,
) -> Result<LinearPrefillStats, KernelError> {
    let query_chunks = tokens.div_ceil(16);
    let tile_count = checked_product(in_features / 16, out_features / 16, "tile count")?;
    let decoded_tiles = checked_product(tile_count, query_chunks, "decoded tile count")?;
    let full_query_chunks = tokens / 16;
    let final_query_count = tokens % 16;
    let valid_queries = full_query_chunks * 16 + usize::from(final_query_count != 0);
    let weight_applications = checked_product(
        tile_count,
        checked_product(valid_queries, 256, "weight application count")?,
        "weight application count",
    )?;
    Ok(LinearPrefillStats {
        query_chunks,
        decoded_tiles,
        weight_applications,
        dispatch_groups: checked_product(out_features / 16, query_chunks, "dispatch count")?,
    })
}

fn validate_dimensions(request: &LinearRequest<'_>) -> Result<(), KernelError> {
    let request = *request;
    let LinearRequest {
        packed,
        input,
        input_signs,
        output_signs,
        in_features,
        out_features,
        rate,
        codebook,
    } = request;
    if in_features == 0
        || out_features == 0
        || !in_features.is_multiple_of(128)
        || !out_features.is_multiple_of(128)
    {
        return Err(KernelError::InvalidInput(
            "linear dimensions must be non-zero multiples of 128".to_string(),
        ));
    }
    if input.len() != in_features || input_signs.len() != in_features {
        return Err(KernelError::InvalidInput(
            "input and input sign lengths do not match in_features".to_string(),
        ));
    }
    if output_signs.len() != out_features {
        return Err(KernelError::InvalidInput(
            "output sign length does not match out_features".to_string(),
        ));
    }
    if matches!(rate, Exl3Rate::Fractional(_)) && codebook != Exl3Codebook::Mul1 {
        return Err(KernelError::InvalidInput(
            "fractional EXL3 requires mul1".to_string(),
        ));
    }
    let expected = (in_features / 16)
        .checked_mul(out_features / 16)
        .and_then(|value| value.checked_mul(rate.packed_words()))
        .ok_or_else(|| KernelError::InvalidInput("packed shape overflows".to_string()))?;
    if packed.len() != expected {
        return Err(KernelError::InvalidInput(format!(
            "packed length {} does not match {expected}",
            packed.len()
        )));
    }
    Ok(())
}

fn validate_prefill_dimensions(request: &PrefillRequest<'_>) -> Result<(), KernelError> {
    let request = *request;
    if request.queries == 0
        || request.in_features == 0
        || request.out_features == 0
        || !request.in_features.is_multiple_of(128)
        || !request.out_features.is_multiple_of(128)
    {
        return Err(KernelError::InvalidInput(
            "prefill dimensions must be non-zero multiples of 128".to_string(),
        ));
    }
    let input_len = checked_product(request.queries, request.in_features, "input shape")?;
    checked_product(request.queries, request.out_features, "output shape")?;
    if request.input.len() != input_len
        || request.input_signs.len() != request.in_features
        || request.output_signs.len() != request.out_features
    {
        return Err(KernelError::InvalidInput(
            "prefill input or sign length does not match dimensions".to_string(),
        ));
    }
    if matches!(request.rate, Exl3Rate::Fractional(_)) && request.codebook != Exl3Codebook::Mul1 {
        return Err(KernelError::InvalidInput(
            "fractional EXL3 requires mul1".to_string(),
        ));
    }
    let expected = (request.in_features / 16)
        .checked_mul(request.out_features / 16)
        .and_then(|value| value.checked_mul(request.rate.packed_words()))
        .ok_or_else(|| KernelError::InvalidInput("packed shape overflows".to_string()))?;
    if request.packed.len() != expected {
        return Err(KernelError::InvalidInput(format!(
            "packed length {} does not match {expected}",
            request.packed.len()
        )));
    }
    Ok(())
}

fn checked_product(left: usize, right: usize, name: &str) -> Result<usize, KernelError> {
    left.checked_mul(right)
        .ok_or_else(|| KernelError::InvalidInput(format!("{name} overflows")))
}

fn to_u32(value: usize, name: &str) -> Result<u32, KernelError> {
    u32::try_from(value).map_err(|_| KernelError::InvalidInput(format!("{name} is too large")))
}

fn zero_f32_buffer(buffer: &Buffer, count: usize) -> Result<(), KernelError> {
    let length = buffer.length() as usize;
    if !length.is_multiple_of(size_of::<f32>()) || count > length / size_of::<f32>() {
        return Err(KernelError::CommandFailed);
    }
    let pointer = buffer.contents() as *mut f32;
    if pointer.is_null() {
        return Err(KernelError::CommandFailed);
    }
    unsafe { std::ptr::write_bytes(pointer, 0, count) };
    Ok(())
}

fn params_bytes<T>(params: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(params as *const T as *const u8, size_of::<T>()) }
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
