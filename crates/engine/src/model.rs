use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use metal::Buffer;
use thiserror::Error;

use kaplumba_kernels::attention::{
    AttentionEncodeRequest, AttentionKernels, AttentionRequest, decode_partial_elements,
};
use kaplumba_kernels::elements::{ElementKernels, RopeBufferRequest, RopeRequest};
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate, KernelError};
use kaplumba_kernels::linear::{
    LinearBufferRequest, LinearEncodeRequest, LinearKernels, LinearRequest, PrefillEncodeRequest,
    PrefillRequest,
};

use crate::exl3::unpack_signs;
use crate::gpu_cache::{GpuCacheError, GpuKvCache};
use crate::kv_cache::{FullKvCache, KvCache, KvCacheError, SlidingKvCache};
use crate::metal::{DEFAULT_RESIDENT_LIMIT_BYTES, MetalContext, MetalError};
use crate::safetensors::inspect_model_dir;
use crate::weights::{TextWeights, WeightError};

/// Rows of prompt fed through one pass of the network.
///
/// Caps the activation scratch independently of prompt length, per AGENTS.md §8.
/// 1024 rows: measured, not chosen for tidiness. The breakdown at 1024 rows:
///
/// | buffer | bytes | scales with chunk |
/// |---|---|---|
/// | `raw` (f32, `chunk * inter`) | 88.08 MB | yes |
/// | `transformed` (u16, `chunk * inter`) | 44.04 MB | yes |
/// | `gate` + `up` + `activated` (u16, `chunk * inter`) | 132.12 MB | yes |
/// | `q` (u16, `chunk * q_max`, q_max = 16384) | 33.55 MB | yes |
/// | `k` + `v` (u16, `chunk * kv_max`, kv_max = 4096) | 16.78 MB | yes |
/// | `hidden_a/b/normed/attended/proj` (u16, `chunk * hidden`) | 55.05 MB | yes |
/// | `attn_partials` (f32, `16 splits * q_heads * (head_dim + 2)`) | 1.05 MB | no |
/// | `logits` + `capped` + fixed | 1.20 MB | no |
/// | **total** | **~372 MB** | |
///
/// That is 4x the 93 MB it was at 256 rows, and the standing budget is 12 GiB
/// of a 24 GiB box, so it has to be paid for out of measured RSS. It is:
/// interleaved A/B on the real checkpoint, `--max-new-tokens 1 --metrics`,
/// `peak_rss_bytes` median of 5 at 614 tokens and 5 at 7814:
///
/// | chunk | 614 tok/s | rss @614 | 7814 tok/s | rss @7814 |
/// |---|---:|---:|---:|---:|
/// | 256 | 70.62 (69.75-70.93) | 7.85 GiB | 69.34 (69.08-69.42) | 8.15 GiB |
/// | 512 | 71.97 (71.52-72.21) | 7.90 GiB | -- | -- |
/// | **1024** | **77.79 (77.55-78.23)** | **8.01 GiB** | **74.57 (74.30-74.59)** | **7.91 GiB** |
///
/// +160 MB at 614 and nothing measurable at 7814 -- the extra scratch displaces
/// file cache, not the 15.5 GB of weights. Budget headroom is unchanged.
///
/// ### Why the chunk size is what it is
///
/// The premise for raising it was that a 614-token prompt's last 102-token
/// chunk is only 40% of a 512-row tile. **That premise is false, and the
/// measurement says so.** `linear_gemm_mma` dispatches
/// `ceil(queries / 128)` threadgroups along y, so the number of 128-query tiles
/// is a property of the TOKEN count, not of the chunk count: 614 tokens is
/// `ceil(614/128) = 5` tiles whether they arrive as 256+256+102, as 512+102, or
/// as one 614-row chunk. Measured per-chunk GEMM own_ms at 614 tokens, from the
/// `KAPLUMBA_GPU_PROFILE=1` per-chunk table:
///
/// | chunk | tiles/dispatch | ms per 128-row tile | GEMM total |
/// |---|---:|---:|---:|
/// | 128 | 1 | 1354.0 | 6783.5 |
/// | 256 | 2, 2, 1 | 1271.7 | 6429.6 |
/// | 512 | 4, 1 | 1222.8 | 6235.9 |
/// | 1024 | 5 | 1204.1 | 6020.7 |
///
/// All four issue the same 5 tiles and waste the same 26 of 640 rows (4.06%);
/// what improves monotonically is the ms *per tile* as the grid grows, because a
/// small grid does not fill 20 GPU cores. So the loss was never the partial
/// tail -- that part is irreducible without sub-128-row tiles, which the 8x8 MMA
/// shape cannot use -- it was that splitting 5 tiles across 3 dispatches cost
/// 6.4% of GEMM versus issuing them as one. 256 -> 1024 is 4x the scratch for
/// 10.2% at 614 tokens and 7.5% at 7814.
///
/// 2048 was not adopted: it needs 743 MB of scratch for no measured gain once
/// the grid is already 1344 threadgroups, and it pushes a 7814-token prompt to
/// four chunks rather than eight. The knob stays at 1024, the largest power of
/// two that still covers a 614-token prompt in ONE dispatch.
///
/// The `128`-row entry is the control that proves the story: it issues the same
/// 5 tiles in 5 dispatches of 1 and is the slowest of all four.
///
/// ### What is NOT in this table
///
/// Per-element accumulation order. A chunk boundary splits QUERY rows, never K,
/// so every output element is still produced by exactly one threadgroup running
/// the identical K loop: outputs are bit-identical at 256 and at 1024 rows, and
/// `linear_metal`'s 2.1603336e-3 worst error is untouched. Only the dispatch
/// grid changes.
///
/// | buffer | bytes | scales with chunk |
/// |---|---|---|
/// | `raw` (f32, `chunk * inter`) | 22.02 MB | yes |
/// | `transformed` (u16, `chunk * inter`) | 11.01 MB | yes |
/// | `gate` + `up` + `activated` (u16, `chunk * inter`) | 33.03 MB | yes |
/// | `q` (u16, `chunk * q_max`, q_max = 16384) | 8.39 MB | yes |
/// | `k` + `v` (u16, `chunk * kv_max`, kv_max = 4096) | 4.19 MB | yes |
/// | `hidden_a/b/normed/attended/proj` (u16, `chunk * hidden`) | 13.76 MB | yes |
/// | `attn_partials` (f32, `16 splits * q_heads * (head_dim + 2)`) | 1.05 MB | no |
/// | `logits` + `capped` + fixed | 1.20 MB | no |
/// | **total** | **~90 MB** | |
///
/// This was 64 rows until the attention scratch was right-sized. It had been
/// allocated `q_max * 1024` = `q_heads * head_dim * 1024` floats, but the old
/// split-key buffer was indexed by *head*, so the real requirement was
/// `q_heads * key_count`. That made it 67.11 MB, which alone pushed a 256-row
/// chunk to 160.7 MB and forced the chunk down to 64. Right-sizing it bought
/// back the 4x larger chunk (fewer, fatter dispatches) *and* trimmed the
/// minimum footprint — and then sizing it from a construction-time cache
/// capacity instead of the live key count broke decode outright past 1024 keys.
/// It is now split-KV flash decoding: `ATTENTION_SPLITS_MAX * q_heads *
/// (head_dim + 2)` floats, fixed, with no dependence on context length.
pub const PREFILL_CHUNK: usize = 1024;

/// MEASUREMENT OVERRIDE: `KAPLUMBA_PREFILL_CHUNK=N` replaces [`PREFILL_CHUNK`]
/// for this process so one binary can A/B chunk sizes without a rebuild per
/// value. Falls back to the constant unless the variable names a power of two
/// of at least 32. Not a product setting — nothing reads it in the shipped
/// path, and it can push `scratch_max_tokens()` past [`PREFILL_CHUNK`], which
/// the doc comment there forbids. It exists so the next agent does not have to
/// rebuild five times to re-measure the table above.
pub fn prefill_chunk() -> usize {
    static CACHE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        match std::env::var("KAPLUMBA_PREFILL_CHUNK")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
        {
            Some(value) if value >= 32 && value.is_power_of_two() => value,
            _ => PREFILL_CHUNK,
        }
    })
}

/// Decode command buffers created, and decode steps run, since the process
/// started.
///
/// A process-wide counter rather than a field on [`TextModel`], because the only
/// thing that wants it is the `KAPLUMBA_COUNT_BUFFERS` diagnostic, and adding a
/// field to `TextModel` for it would put a diagnostic-only member in the type
/// every layer and forward path carries.
///
/// Relaxed ordering: a diagnostic tally with no synchronisation to carry, whose
/// only reader is a human after the run has finished.
fn decode_counters() -> (&'static AtomicUsize, &'static AtomicUsize) {
    static BUFFERS: AtomicUsize = AtomicUsize::new(0);
    static STEPS: AtomicUsize = AtomicUsize::new(0);
    (&BUFFERS, &STEPS)
}

/// Command buffers per decode step over the process's lifetime, if
/// `KAPLUMBA_COUNT_BUFFERS` is set.
pub fn report_buffers_per_step() {
    if std::env::var_os("KAPLUMBA_COUNT_BUFFERS").is_none() {
        return;
    }
    let (buffers, steps) = decode_counters();
    let buffers = buffers.load(Ordering::Relaxed);
    let steps = steps.load(Ordering::Relaxed);
    eprintln!(
        "BUFFERS total={buffers} steps={steps} per_step={:.2}",
        buffers as f64 / steps.max(1) as f64
    );
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error(transparent)]
    Kernel(#[from] KernelError),
    #[error(transparent)]
    Cache(#[from] KvCacheError),
    #[error(transparent)]
    GpuCache(#[from] GpuCacheError),
    #[error("invalid layer configuration: {0}")]
    InvalidConfig(String),
    /// Another process holds the GPU. Not a fault of this run: the caller should
    /// treat it as "not now" and retry later, which is why it is a distinct
    /// variant rather than an `InvalidConfig` string. Two instances on this
    /// 24 GiB box swap and collapse to ~1.4 tok/s, so this is load-bearing
    /// protection, not a nicety.
    #[error("{0}")]
    GpuBusy(String),
    #[error("invalid layer weights: {0}")]
    InvalidWeights(String),
    #[error("invalid layer input: {0}")]
    InvalidInput(String),
    #[error("checkpoint inspection failed: {0}")]
    Checkpoint(String),
    #[error(transparent)]
    Weights(#[from] WeightError),
    #[error(transparent)]
    Metal(#[from] MetalError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeMode {
    Default,
    Proportional,
}

/// Name and RMS of one intermediate activation, used by layer diagnostics.
pub type StepMagnitude = (&'static str, f64);

#[derive(Debug, Clone, Copy)]
pub struct LayerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub max_sequence_len: usize,
    pub full_attention: bool,
    pub sliding_window: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_mode: RopeMode,
    pub rope_partial_factor: f32,
}

impl LayerConfig {
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.q_heads == 0
            || self.kv_heads == 0
            || self.head_dim == 0
            || self.head_dim > 512
            || self.max_sequence_len == 0
            || !self.q_heads.is_multiple_of(self.kv_heads)
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
        {
            return Err(ModelError::InvalidConfig(
                "layer dimensions and numeric parameters are invalid".to_string(),
            ));
        }
        if !self.full_attention && self.sliding_window == 0 {
            return Err(ModelError::InvalidConfig(
                "sliding layers require a non-zero window".to_string(),
            ));
        }
        if self.rope_mode == RopeMode::Proportional
            && (!self.rope_partial_factor.is_finite()
                || self.rope_partial_factor <= 0.0
                || self.rope_partial_factor > 1.0)
        {
            return Err(ModelError::InvalidConfig(
                "proportional RoPE factor must be in (0, 1]".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct QuantizedLinear {
    pub in_features: usize,
    pub out_features: usize,
    pub rate: Exl3Rate,
    pub codebook: Exl3Codebook,
    pub packed: Vec<u16>,
    pub metal_buffer: Option<Buffer>,
    /// Byte offset of the packed tensor inside `metal_buffer`.
    ///
    /// Zero-copy weights share one buffer per shard, so the tensor does not
    /// start at offset zero.
    pub metal_offset: u64,
    /// Packed element count. A zero-copy buffer is a page-aligned window around
    /// the tensor, so its length cannot be used to infer this.
    pub packed_len: Option<usize>,
    /// Tile permutation selector. 0 is the row-major default implied by the
    /// reference packer; other values are the candidate fragment shuffles that
    /// the coherence probe searches.
    pub permutation_mode: u32,
    pub input_signs: Vec<u16>,
    pub output_signs: Vec<u16>,
    /// GPU-resident copies of the sign vectors, uploaded once so the hot path
    /// never re-uploads them per call.
    pub input_signs_buffer: Option<Buffer>,
    pub output_signs_buffer: Option<Buffer>,
}

impl QuantizedLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        rate: Exl3Rate,
        codebook: Exl3Codebook,
        packed: Vec<u16>,
        input_signs: Vec<u16>,
        output_signs: Vec<u16>,
    ) -> Result<Self, ModelError> {
        let linear = Self {
            in_features,
            out_features,
            rate,
            codebook,
            packed,
            metal_buffer: None,
            metal_offset: 0,
            packed_len: None,
            permutation_mode: 0,
            input_signs,
            output_signs,
            input_signs_buffer: None,
            output_signs_buffer: None,
        };
        linear.validate()?;
        Ok(linear)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_metal_buffer(
        in_features: usize,
        out_features: usize,
        rate: Exl3Rate,
        codebook: Exl3Codebook,
        metal_buffer: Buffer,
        metal_offset: u64,
        packed_len: Option<usize>,
        input_signs: Vec<u16>,
        output_signs: Vec<u16>,
    ) -> Result<Self, ModelError> {
        let linear = Self {
            in_features,
            out_features,
            rate,
            codebook,
            packed: Vec::new(),
            metal_buffer: Some(metal_buffer),
            metal_offset,
            packed_len,
            permutation_mode: 0,
            input_signs,
            output_signs,
            input_signs_buffer: None,
            output_signs_buffer: None,
        };
        linear.validate()?;
        Ok(linear)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if self.in_features == 0
            || self.out_features == 0
            || !self.in_features.is_multiple_of(128)
            || !self.out_features.is_multiple_of(128)
        {
            return Err(ModelError::InvalidWeights(
                "linear dimensions must be non-zero multiples of 128".to_string(),
            ));
        }
        if matches!(self.rate, Exl3Rate::Fractional(_)) && self.codebook != Exl3Codebook::Mul1 {
            return Err(ModelError::InvalidWeights(
                "fractional EXL3 requires mul1".to_string(),
            ));
        }
        if let Exl3Rate::Integer(bits) = self.rate
            && !(1..=8).contains(&bits)
        {
            return Err(ModelError::InvalidWeights(format!(
                "unsupported integer EXL3 rate {bits}"
            )));
        }
        if let Exl3Rate::Fractional(bits) = self.rate
            && !(1..=3).contains(&bits)
        {
            return Err(ModelError::InvalidWeights(format!(
                "unsupported fractional EXL3 rate {bits}.5"
            )));
        }
        let expected = self
            .in_features
            .checked_div(16)
            .and_then(|rows| {
                self.out_features
                    .checked_div(16)
                    .and_then(|cols| rows.checked_mul(cols))
            })
            .and_then(|tiles| tiles.checked_mul(self.rate.packed_words()))
            .ok_or_else(|| ModelError::InvalidWeights("linear shape overflows".to_string()))?;
        let packed_len = match (self.packed_len, &self.metal_buffer) {
            (Some(len), _) => len,
            (None, Some(buffer)) => usize::try_from(buffer.length() / 2).map_err(|_| {
                ModelError::InvalidWeights("packed buffer is too large".to_string())
            })?,
            (None, None) => self.packed.len(),
        };
        if packed_len != expected
            || self.input_signs.len() != self.in_features
            || self.output_signs.len() != self.out_features
        {
            return Err(ModelError::InvalidWeights(format!(
                "linear buffers do not match {}x{} shape",
                self.in_features, self.out_features
            )));
        }
        Ok(())
    }
}

/// Upload a linear's sign vectors to the GPU once.
fn upload_sign_buffers(linear: &mut QuantizedLinear) -> Result<(), ModelError> {
    if linear.input_signs.is_empty() || linear.output_signs.is_empty() {
        return Ok(());
    }
    linear.input_signs_buffer = Some(u16_device_buffer(&linear.input_signs)?);
    linear.output_signs_buffer = Some(u16_device_buffer(&linear.output_signs)?);
    Ok(())
}

fn system_device() -> Result<metal::Device, ModelError> {
    metal::Device::system_default()
        .ok_or_else(|| ModelError::Checkpoint("Metal system device is unavailable".to_string()))
}

fn u16_device_buffer(values: &[u16]) -> Result<Buffer, ModelError> {
    if values.is_empty() {
        return Err(ModelError::InvalidWeights(
            "device buffer source cannot be empty".to_string(),
        ));
    }
    let device = system_device()?;
    Ok(device.new_buffer_with_data(
        values.as_ptr() as *const std::ffi::c_void,
        std::mem::size_of_val(values) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ))
}

/// Read-only per-layer vectors uploaded to the GPU once at load time.
struct LayerGpuWeights {
    q_norm: Buffer,
    k_norm: Buffer,
    input_norm: Buffer,
    post_attention_norm: Buffer,
    pre_ffn_norm: Buffer,
    post_ffn_norm: Buffer,
}

#[derive(Debug, Clone)]
pub struct DecoderLayerWeights {
    pub q_proj: QuantizedLinear,
    pub k_proj: QuantizedLinear,
    pub v_proj: Option<QuantizedLinear>,
    pub o_proj: QuantizedLinear,
    pub gate_proj: QuantizedLinear,
    pub up_proj: QuantizedLinear,
    pub down_proj: QuantizedLinear,
    pub q_norm: Vec<u16>,
    pub k_norm: Vec<u16>,
    pub input_norm: Vec<u16>,
    pub post_attention_norm: Vec<u16>,
    pub pre_ffn_norm: Vec<u16>,
    pub post_ffn_norm: Vec<u16>,
    pub layer_scalar: f32,
}

impl DecoderLayerWeights {
    pub fn validate(&self, config: &LayerConfig) -> Result<(), ModelError> {
        self.q_proj.validate()?;
        self.k_proj.validate()?;
        if let Some(v_proj) = &self.v_proj {
            v_proj.validate()?;
        }
        if config.full_attention && self.v_proj.is_some() {
            return Err(ModelError::InvalidWeights(
                "full attention layers must reuse the raw K projection for V".to_string(),
            ));
        }
        if !config.full_attention && self.v_proj.is_none() {
            return Err(ModelError::InvalidWeights(
                "sliding attention layers require a V projection".to_string(),
            ));
        }
        self.o_proj.validate()?;
        self.gate_proj.validate()?;
        self.up_proj.validate()?;
        self.down_proj.validate()?;
        if self.q_proj.in_features != config.hidden_size
            || self.q_proj.out_features != config.q_heads * config.head_dim
            || self.k_proj.in_features != config.hidden_size
            || self.k_proj.out_features != config.kv_heads * config.head_dim
            || self.o_proj.in_features != config.q_heads * config.head_dim
            || self.o_proj.out_features != config.hidden_size
            || self.gate_proj.in_features != config.hidden_size
            || self.gate_proj.out_features != config.intermediate_size
            || self.up_proj.in_features != config.hidden_size
            || self.up_proj.out_features != config.intermediate_size
            || self.down_proj.in_features != config.intermediate_size
            || self.down_proj.out_features != config.hidden_size
        {
            return Err(ModelError::InvalidWeights(
                "projection shapes do not match layer configuration".to_string(),
            ));
        }
        if self.q_norm.len() != config.head_dim
            || self.k_norm.len() != config.head_dim
            || self.input_norm.len() != config.hidden_size
            || self.post_attention_norm.len() != config.hidden_size
            || self.pre_ffn_norm.len() != config.hidden_size
            || self.post_ffn_norm.len() != config.hidden_size
            || !self.layer_scalar.is_finite()
        {
            return Err(ModelError::InvalidWeights(
                "normalization weights or layer scalar are invalid".to_string(),
            ));
        }
        if let Some(v_proj) = &self.v_proj
            && (v_proj.in_features != config.hidden_size
                || v_proj.out_features != config.kv_heads * config.head_dim)
        {
            return Err(ModelError::InvalidWeights(
                "V projection shape does not match layer configuration".to_string(),
            ));
        }
        Ok(())
    }
}

/// Per-operation wall-clock accumulator for the phase profiler.
///
/// The laps measure CPU-side encode cost around each dispatch, NOT GPU
/// execution: dispatches into one command buffer run asynchronously, so a lap
/// is only the host time to submit the work. Relative encode-cost shares can
/// still be useful (a 1000-dispatch phase has proportionally more submit
/// overhead), but GPU time attribution must come from the flush-boundary
/// phases, from [`GpuProfile`] (the real one), or from the attribution env
/// switches (KAPLUMBA_LINEARS_ONLY, KAPLUMBA_SKIP_ATTENTION), which move real
/// GPU work.
///
/// Lap call sites live at the same boundaries [`GpuProfile`] stamps, so the two
/// tables line up row for row and the difference between them is the
/// submit-versus-execute gap. Under the phase profiler the first lap of a layer
/// also absorbs that layer's `flush_and_wait`, so it reads far larger than its
/// neighbours; that is a host-timing artefact, not a hotspot.
pub struct OpTimer<'t> {
    map: &'t mut std::collections::BTreeMap<&'static str, f64>,
    last: std::time::Instant,
    on: bool,
}

impl<'t> OpTimer<'t> {
    fn new(map: &'t mut std::collections::BTreeMap<&'static str, f64>, on: bool) -> Self {
        Self {
            map,
            last: std::time::Instant::now(),
            on,
        }
    }

    pub fn lap(&mut self, name: &'static str) {
        if self.on {
            let now = std::time::Instant::now();
            *self.map.entry(name).or_default() += now.duration_since(self.last).as_secs_f64() * 1e6;
            self.last = now;
        }
    }
}

/// Median of the plausible samples, or `None` if the GPU clock gave us nothing
/// usable.
///
/// `GPUStartTime` / `GPUEndTime` are documented as undefined before completion,
/// and a command buffer with no dispatches can legitimately read back as exactly
/// zero (start == end). So a floor is only reported when at least one sample came
/// back in a believable range.
fn median_floor_ms(samples: Vec<f64>) -> Option<f64> {
    let mut samples: Vec<f64> = samples
        .into_iter()
        .filter(|ms| *ms > 0.0 && *ms < 1.0)
        .collect();
    if samples.is_empty() {
        return None;
    }
    samples.sort_by(f64::total_cmp);
    Some(samples[samples.len() / 2])
}

/// Stamped command buffers waiting to be turned into [`GpuStamp`]s, and the
/// cadence at which they are drained.
///
/// `commit_stamped` deliberately does *not* apply `MAX_IN_FLIGHT` back-pressure
/// (it is a diagnostic path), but `new_command_buffer` still blocks once too many
/// buffers are outstanding on a queue — the encoder module measured ~175 ms of
/// blocking past ~64 outstanding. A 60-layer forward has 15 boundaries per layer,
/// so 900 retained buffers per chunk would stall the host mid-forward. Draining
/// every 15 boundaries (one layer's worth) keeps the window at a quarter of that
/// and costs at most one wait per layer.
///
/// The wait is also the profiler's main distortion: it serialises host encoding
/// against GPU execution, so the *gaps* between buffers are an artefact of the
/// profiler, not of the model. That is why [`GpuStamp::gap_ms`] exists and why the
/// table reports own-time and gap-time as separate columns. Attribution
/// (own_ms per kernel) is unaffected; total wall time is not.
const DRAIN_EVERY: usize = 15;

/// The process-wide [`AttentionKernels`] instance.
///
/// `AttentionKernels::new()` recompiles `attention.metal` and rebuilds every
/// pipeline, and all sixty `DecoderLayer`s used to call it, so a load ran the
/// same shader compilation sixty-one times. This lives here rather than in
/// `attention.rs` because the accessor belongs next to the type it caches;
/// `ElementKernels::shared` and `LinearKernels::shared` are the same change in
/// the crates that own those types. The type has no `&mut self` method.
fn shared_attention_kernels() -> Result<&'static AttentionKernels, ModelError> {
    static INSTANCE: std::sync::OnceLock<Result<AttentionKernels, KernelError>> =
        std::sync::OnceLock::new();
    match INSTANCE.get_or_init(AttentionKernels::new) {
        Ok(kernels) => Ok(kernels),
        // `get_or_init` borrows the cell, so a stored `KernelError` cannot be
        // moved out of it and it is not `Clone`. Construction failed at startup,
        // so this path is not hot; the message is what a caller needs.
        Err(error) => Err(ModelError::Kernel(KernelError::Shader(error.to_string()))),
    }
}

/// Cached "is this attribution switch set?" lookup.
///
/// These used to be live `std::env::var_os` calls evaluated inside the forward
/// pass -- `KAPLUMBA_LINEARS_ONLY` and `KAPLUMBA_SKIP_ATTENTION` once per layer
/// each, so ~120 environment-lock acquisitions and `environ` scans per generated
/// token. A switch is set before the process starts or not at all, so reading it
/// once is behaviour-preserving.
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

/// Cached `usize` switch, for the measurement aids that take a count. Read once
/// for the same reason as [`env_flag`], and parsed once so a malformed value
/// cannot be re-parsed per layer.
fn env_usize(name: &str) -> usize {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, usize>>> =
        std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut cache = cache.lock().unwrap_or_else(|error| error.into_inner());
    if let Some(value) = cache.get(name) {
        return *value;
    }
    let value = std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(0);
    cache.insert(name.to_string(), value);
    value
}

/// [`env_usize`], clamped into a `u32` for a shader parameter.
fn env_u32(name: &str) -> Option<u32> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, u32>>> =
        std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut cache = cache.lock().unwrap_or_else(|error| error.into_inner());
    if let Some(value) = cache.get(name) {
        return Some(*value);
    }
    let value = std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<u32>().ok());
    if let Some(value) = value {
        cache.insert(name.to_string(), value);
    }
    value
}

/// Per-kernel GPU time, accumulated across every chunk of one forward pass.
///
/// This is the real profiler. `KAPLUMBA_PROFILE_PHASES` (and [`OpTimer`]) measure
/// host encode time plus flush boundaries, which is why the old numbers were
/// useless; this one reads `GPUStartTime` / `GPUEndTime` off every command buffer
/// the forward stamps, so every number is GPU-clock time.
///
/// One instance spans the whole `forward_buffer_into`, so a 614-token prefill
/// (one 614-row chunk, since [`PREFILL_CHUNK`] is 1024) produces one table.
pub struct GpuProfile {
    /// `KAPLUMBA_GPU_PROFILE=1`. When false every method is a no-op and
    /// [`Self::stamp`] returns without touching the encoder, so the non-profiling
    /// path is unchanged.
    enabled: bool,
    /// kernel name -> (invocations, total own ms), summed over all layers and
    /// chunks. The key is the part of a stamped label after the `/`.
    by_kernel: std::collections::BTreeMap<String, (u64, f64)>,
    /// `layer07` -> (invocations, total own ms), for the per-layer view.
    by_layer: std::collections::BTreeMap<String, (u64, f64)>,
    /// MEASUREMENT AID: `"<rows>:<kernel>"` -> (invocations, total own ms), so a
    /// multi-chunk prefill can be split per chunk. `rows` is the row count of the
    /// chunk in flight when the stamp was ingested, which is the only place the
    /// chunk length is known without threading it through every label.
    by_chunk: std::collections::BTreeMap<String, (u64, f64)>,
    /// Row count of the chunk currently being encoded.
    chunk_rows: usize,
    /// Stamped buffers committed since the last drain.
    pending: usize,
    boundaries: u64,
    stamps_ingested: u64,
    chunks: u64,
    tokens: usize,
    layers: usize,
    /// Sum of every buffer's own GPU time: the cost of the instrumented work.
    total_own_ms: f64,
    /// Sum of the idle intervals between consecutive buffers. Under this profiler
    /// that is submission latency plus the drain waits, i.e. mostly overhead.
    total_gap_ms: f64,
    first_start_s: Option<f64>,
    last_end_s: Option<f64>,
    /// Median own_ms of a committed buffer with no dispatches in it: the floor
    /// every row in the table sits on.
    empty_commit_ms: Option<f64>,
    /// Median own_ms of a buffer holding exactly one trivial dispatch: the floor
    /// for a one-kernel row.
    one_dispatch_ms: Option<f64>,
}

impl GpuProfile {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            by_kernel: std::collections::BTreeMap::new(),
            by_layer: std::collections::BTreeMap::new(),
            by_chunk: std::collections::BTreeMap::new(),
            chunk_rows: 0,
            pending: 0,
            boundaries: 0,
            stamps_ingested: 0,
            chunks: 0,
            tokens: 0,
            layers: 0,
            total_own_ms: 0.0,
            total_gap_ms: 0.0,
            first_start_s: None,
            last_end_s: None,
            empty_commit_ms: None,
            one_dispatch_ms: None,
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Per-kernel own time for this forward, as `(name, invocations, own_ms)`,
    /// in name order.
    ///
    /// Exists for `KAPLUMBA_DECODE_PROFILE=kernel`, which accumulates a decode
    /// step's table into one process-wide report instead of printing a table per
    /// generated token. The per-kernel detail is private, so without this the
    /// accumulator could only be handed a total and the mode would be
    /// indistinguishable from `step`.
    pub fn kernel_rows(&self) -> Vec<(String, u64, f64)> {
        self.by_kernel
            .iter()
            .map(|(name, (count, own))| (name.clone(), *count, *own))
            .collect()
    }

    /// `first stamped start`, `last stamped end` on the GPU timeline, if any
    /// stamp came back. The span between them is the forward's duration with
    /// the gaps a per-boundary stamp introduces left *inside* it, so it is
    /// reported rather than used as a stopwatch.
    pub fn gpu_span_s(&self) -> Option<(f64, f64)> {
        self.first_start_s.zip(self.last_end_s)
    }

    /// Sum of every stamped buffer's own GPU time for this forward.
    pub fn own_ms(&self) -> f64 {
        self.total_own_ms
    }

    /// End the current command buffer under `label` and continue on a fresh one,
    /// so the label's GPU duration can be read back later.
    ///
    /// The boundary is the distortion: work that a single command buffer would
    /// have overlapped across the boundary is now split into two buffers, and a
    /// Metal command queue executes them one after the other. That is why the
    /// table reports own-time per boundary (still correct) and treats the
    /// inter-boundary gap as profiler overhead.
    pub fn stamp(&mut self, encoder: &mut MetalEncoder, label: &str) -> Result<(), ModelError> {
        if !self.enabled {
            return Ok(());
        }
        encoder.commit_stamped(label).map_err(ModelError::from)?;
        self.pending += 1;
        self.boundaries += 1;
        if self.pending >= DRAIN_EVERY {
            self.drain(encoder)?;
        }
        Ok(())
    }

    /// Fold the retained stamps into the totals. Public so the table can be unit
    /// tested without a model; see `tests/gpu_profile_table.rs`.
    ///
    /// A disabled profile absorbs nothing, so the non-profiling path cannot
    /// accumulate by accident even if a caller gets this far.
    pub fn ingest(&mut self, stamps: &[kaplumba_kernels::encoder::GpuStamp]) {
        if !self.enabled {
            return;
        }
        self.stamps_ingested += stamps.len() as u64;
        for stamp in stamps {
            // `layer07/q_proj` splits into a per-layer row and a per-kernel row;
            // anything else (`chunk/embedding`, `head/gemv`) is per-chunk work and
            // keeps its qualified name, so the two never collide in the table.
            let (layer, kernel) = match stamp.label.split_once('/') {
                Some((layer, kernel)) if layer.starts_with("layer") => {
                    (Some(layer.to_string()), kernel.to_string())
                }
                _ => (None, stamp.label.clone()),
            };
            let kernel_name = kernel.clone();
            let kernel_slot = self.by_kernel.entry(kernel).or_default();
            kernel_slot.0 += 1;
            kernel_slot.1 += stamp.own_ms;
            if let Some(layer) = layer {
                let layer_slot = self.by_layer.entry(layer).or_default();
                layer_slot.0 += 1;
                layer_slot.1 += stamp.own_ms;
                let chunk_slot = self
                    .by_chunk
                    .entry(format!("{}r/{}", self.chunk_rows, kernel_name))
                    .or_default();
                chunk_slot.0 += 1;
                chunk_slot.1 += stamp.own_ms;
            }
            self.total_own_ms += stamp.own_ms;
            self.total_gap_ms += stamp.gap_ms;
            self.first_start_s = Some(
                self.first_start_s
                    .map_or(stamp.start_s, |s: f64| s.min(stamp.start_s)),
            );
            self.last_end_s = Some(
                self.last_end_s
                    .map_or(stamp.end_s, |e: f64| e.max(stamp.end_s)),
            );
        }
    }

    /// Wait for the retained buffers and fold them in. This is the back-pressure
    /// that keeps the queue from blocking, and the serialisation that inflates
    /// wall time; see [`DRAIN_EVERY`].
    pub fn drain(&mut self, encoder: &mut MetalEncoder) -> Result<(), ModelError> {
        if !self.enabled || self.pending == 0 {
            return Ok(());
        }
        let stamps = encoder.take_gpu_stamps();
        self.pending = 0;
        self.ingest(&stamps);
        Ok(())
    }

    /// Record the measured per-boundary floors. The first calibration wins, so a
    /// later chunk cannot move a floor a reader has already seen quoted.
    pub fn set_calibration(&mut self, empty_commit_ms: Option<f64>, one_dispatch_ms: Option<f64>) {
        if self.empty_commit_ms.is_none() {
            self.empty_commit_ms = empty_commit_ms;
        }
        if self.one_dispatch_ms.is_none() {
            self.one_dispatch_ms = one_dispatch_ms;
        }
    }

    /// Fold a calibration batch: the `cal/*` stamps become the measured floor,
    /// and anything else in the batch is real work that was still pending, so it
    /// is ingested rather than discarded.
    ///
    /// This is why the classifier looks at the label rather than at the batch's
    /// contents by position: calibration runs at the top of a chunk, and the
    /// previous chunk's tail boundaries can still be retained. Taking them here
    /// and throwing them into the calibration sample pool silently lost them from
    /// the table -- 10 of 2718 stamps in the first working run, and every one of
    /// the previous chunk's `head/*` and `chunk/*` rows with it.
    pub fn ingest_calibration(
        &mut self,
        stamps: &[kaplumba_kernels::encoder::GpuStamp],
    ) -> (Vec<f64>, Vec<f64>) {
        let mut empty = Vec::new();
        let mut one = Vec::new();
        let mut real = Vec::new();
        for stamp in stamps {
            match stamp.label.strip_prefix("cal/") {
                Some("empty") => empty.push(stamp.own_ms),
                Some(_) => one.push(stamp.own_ms),
                None => real.push(stamp.clone()),
            }
        }
        self.ingest(&real);
        // The batch emptied the encoder's retained list, so `pending` is exactly
        // what was taken; re-derive it instead of leaving a stale count that would
        // desynchronise the drain cadence for the rest of the forward.
        self.pending = real.len();
        (empty, one)
    }

    /// The report. Kernel table sorted by total descending, then the layer
    /// subtotal, then the overhead and the gap/own split.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "\n=== GPU PROFILE (KAPLUMBA_GPU_PROFILE=1) ===\n\
             tokens={} chunks={} layers={} boundaries={} stamps_ingested={}\n\
             sum of per-kernel GPU own_ms = {:.3} ms   inter-boundary gap = {:.3} ms   \
             wall span = {:.3} ms\n",
            self.tokens,
            self.chunks,
            self.layers,
            self.boundaries,
            self.stamps_ingested,
            self.total_own_ms,
            self.total_gap_ms,
            self.last_end_s
                .zip(self.first_start_s)
                .map_or(0.0, |(end, start)| (end - start) * 1e3),
        ));
        match (self.empty_commit_ms, self.one_dispatch_ms) {
            (None, None) => {
                out.push_str("overhead: unavailable (GPU clock returned no plausible sample)\n")
            }
            (empty, one) => {
                let ms = |value: Option<f64>| match value {
                    Some(value) => format!("{value:.4} ms"),
                    None => "unavailable".to_string(),
                };
                out.push_str(&format!(
                    "overhead floor: empty commit {} | one trivial dispatch {} \
                     (discount rows below this)\n",
                    ms(empty),
                    ms(one)
                ));
            }
        }
        // A profiler that silently drops boundaries is worse than no profiler, so
        // say so. `boundaries > 0` because a profile fed directly through
        // `ingest` (tests) never committed anything to compare against.
        if self.boundaries > 0 && self.stamps_ingested != self.boundaries {
            out.push_str(&format!(
                "WARNING: {} of {} stamped boundaries were not read back; this table under-counts\n",
                self.boundaries.saturating_sub(self.stamps_ingested),
                self.boundaries
            ));
        }
        out.push_str(&format!(
            "  {:<26} {:>7} {:>11} {:>11}\n",
            "kernel", "count", "total_ms", "mean_ms"
        ));
        let mut rows: Vec<(&String, &u64, &f64)> = self
            .by_kernel
            .iter()
            .map(|(name, (count, total))| (name, count, total))
            .collect();
        rows.sort_by(|a, b| b.2.total_cmp(a.2).then_with(|| a.0.cmp(b.0)));
        for (name, count, total) in rows {
            out.push_str(&format!(
                "  {name:<26} {count:>7} {total:>11.3} {:>11.4}\n",
                total / *count as f64
            ));
        }
        // Per-layer subtotal: which layers are expensive regardless of which
        // kernel inside them is.
        let mut layer_rows: Vec<(&String, &u64, &f64)> = self
            .by_layer
            .iter()
            .map(|(name, (count, total))| (name, count, total))
            .collect();
        layer_rows.sort_by(|a, b| b.2.total_cmp(a.2).then_with(|| a.0.cmp(b.0)));
        if !layer_rows.is_empty() {
            let layer_total: f64 = layer_rows.iter().map(|row| row.2).sum();
            let fastest = layer_rows.last().map_or(0.0, |row| *row.2);
            out.push_str(&format!(
                "  per-layer: {} layers, sum {:.3} ms, slowest {} {:.3} ms, fastest {} {:.3} ms\n",
                layer_rows.len(),
                layer_total,
                layer_rows.first().map_or("", |row| row.0.as_str()),
                layer_rows.first().map_or(0.0, |row| *row.2),
                layer_rows.last().map_or("", |row| row.0.as_str()),
                fastest,
            ));
            out.push_str("  per-layer totals (slowest first): ");
            for (name, _, total) in &layer_rows {
                out.push_str(&format!("{name}={total:.1} "));
            }
            out.push('\n');
        }
        if !self.by_chunk.is_empty() {
            out.push_str("  MEASUREMENT per-chunk (rowcount/kernel, count, total_ms):\n");
            for (key, (count, total)) in &self.by_chunk {
                out.push_str(&format!("    {key:<28} {count:>6} {total:>11.3}\n"));
            }
        }
        out
    }
}

pub struct DecoderLayer {
    pub config: LayerConfig,
    pub weights: DecoderLayerWeights,
    /// Host mirror of the cache, created on demand for the legacy slice path.
    ///
    /// The GPU path keeps its authoritative cache in `gpu_cache`, so holding a
    /// second full copy here would add ~1.5 GB of duplicated KV at 8k context
    /// and can push the real checkpoint past the Metal allocation ceiling.
    cache: Option<KvCache>,
    // Shared, process-wide kernel sets rather than per-layer copies. Every
    // `DecoderLayer` used to construct its own, which meant 61
    // `new_library_with_source` calls on the same `include_str!`'d source and 61
    // distinct `MTLComputePipelineState` objects for identical code -- a large
    // one-off load cost, and a blocker on any future binding dedup. The kernel
    // types have no `&mut self` method, so sharing is sound.
    elements: &'static ElementKernels,
    linear: &'static LinearKernels,
    attention: &'static AttentionKernels,
    /// GPU-resident mirror of the host cache, used by the buffer forward path.
    pub gpu_cache: GpuKvCache,
    gpu_weights: LayerGpuWeights,
}

impl DecoderLayer {
    pub fn new(config: LayerConfig, weights: DecoderLayerWeights) -> Result<Self, ModelError> {
        config.validate()?;
        weights.validate(&config)?;
        let device = system_device()?;
        // Full-attention caches start at a small capacity and double on demand
        // up to `max_sequence_len`; allocating the worst case up front was most
        // of the KV footprint for conversations that only need a few hundred
        // positions.
        const INITIAL_FULL_CACHE: usize = 512;
        let gpu_cache = if config.full_attention {
            GpuKvCache::new(
                &device,
                config.max_sequence_len.min(INITIAL_FULL_CACHE),
                config.kv_heads,
                config.head_dim,
            )?
        } else {
            GpuKvCache::new(
                &device,
                config.sliding_window,
                config.kv_heads,
                config.head_dim,
            )?
        };
        let gpu_weights = LayerGpuWeights {
            q_norm: u16_device_buffer(&weights.q_norm)?,
            k_norm: u16_device_buffer(&weights.k_norm)?,
            input_norm: u16_device_buffer(&weights.input_norm)?,
            post_attention_norm: u16_device_buffer(&weights.post_attention_norm)?,
            pre_ffn_norm: u16_device_buffer(&weights.pre_ffn_norm)?,
            post_ffn_norm: u16_device_buffer(&weights.post_ffn_norm)?,
        };
        let mut weights = weights;
        upload_sign_buffers(&mut weights.q_proj)?;
        upload_sign_buffers(&mut weights.k_proj)?;
        if let Some(v_proj) = &mut weights.v_proj {
            upload_sign_buffers(v_proj)?;
        }
        upload_sign_buffers(&mut weights.o_proj)?;
        upload_sign_buffers(&mut weights.gate_proj)?;
        upload_sign_buffers(&mut weights.up_proj)?;
        upload_sign_buffers(&mut weights.down_proj)?;
        Ok(Self {
            elements: ElementKernels::shared()?,
            linear: LinearKernels::shared()?,
            attention: shared_attention_kernels()?,
            config,
            weights,
            cache: None,
            gpu_cache,
            gpu_weights,
        })
    }

    pub fn forward(&mut self, input: &[u16], positions: &[u32]) -> Result<Vec<u16>, ModelError> {
        self.host_cache()?;
        self.validate_input(input, positions)?;
        let tokens = positions.len();
        let normalized = self.elements.rms_norm(
            input,
            &self.weights.input_norm,
            tokens,
            self.config.hidden_size,
            self.config.rms_norm_eps,
        )?;
        let q = self.run_linear(&self.weights.q_proj, &normalized, tokens)?;
        let k = self.run_linear(&self.weights.k_proj, &normalized, tokens)?;
        let raw_v = if let Some(v_proj) = &self.weights.v_proj {
            self.run_linear(v_proj, &normalized, tokens)?
        } else {
            k.clone()
        };
        let q = self.elements.rms_norm(
            &q,
            &self.weights.q_norm,
            tokens * self.config.q_heads,
            self.config.head_dim,
            self.config.rms_norm_eps,
        )?;
        let k = self.elements.rms_norm(
            &k,
            &self.weights.k_norm,
            tokens * self.config.kv_heads,
            self.config.head_dim,
            self.config.rms_norm_eps,
        )?;
        let v = self.elements.rms_norm_unweighted(
            &raw_v,
            tokens * self.config.kv_heads,
            self.config.head_dim,
            self.config.rms_norm_eps,
        )?;
        let offset = positions[0] as usize;
        let q = self.elements.rope(RopeRequest {
            input: &q,
            tokens,
            heads: self.config.q_heads,
            head_dim: self.config.head_dim,
            offset,
            theta: self.config.rope_theta,
            partial_factor: self.config.rope_partial_factor,
            proportional: self.config.rope_mode == RopeMode::Proportional,
        })?;
        let k = self.elements.rope(RopeRequest {
            input: &k,
            tokens,
            heads: self.config.kv_heads,
            head_dim: self.config.head_dim,
            offset,
            theta: self.config.rope_theta,
            partial_factor: self.config.rope_partial_factor,
            proportional: self.config.rope_mode == RopeMode::Proportional,
        })?;
        self.append_cache(&k, &v, positions)?;
        let attended = self.run_attention(&q, positions)?;
        let attended = self.run_linear(&self.weights.o_proj, &attended, tokens)?;
        let attended = self.elements.rms_norm(
            &attended,
            &self.weights.post_attention_norm,
            tokens,
            self.config.hidden_size,
            self.config.rms_norm_eps,
        )?;
        let mut output = self.elements.residual_add(input, &attended)?;
        let normalized = self.elements.rms_norm(
            &output,
            &self.weights.pre_ffn_norm,
            tokens,
            self.config.hidden_size,
            self.config.rms_norm_eps,
        )?;
        let gate = self.run_linear(&self.weights.gate_proj, &normalized, tokens)?;
        let up = self.run_linear(&self.weights.up_proj, &normalized, tokens)?;
        let activated = self.elements.gated_gelu(&gate, &up)?;
        let mlp = self.run_linear(&self.weights.down_proj, &activated, tokens)?;
        let mlp = self.elements.rms_norm(
            &mlp,
            &self.weights.post_ffn_norm,
            tokens,
            self.config.hidden_size,
            self.config.rms_norm_eps,
        )?;
        output = self.elements.residual_add(&output, &mlp)?;
        Ok(self
            .elements
            .layer_scale(&output, self.weights.layer_scalar)?)
    }

    /// Encode one decoder layer entirely on the GPU.
    ///
    /// Nothing here commits or reads back: the caller owns the command buffer
    /// and the scratch buffers, so a whole forward can be encoded as a single
    /// batch and synchronised once instead of ~21 times per layer.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    /// Dispatches the hidden-width RMS norm, or skips it under
    /// `KAPLUMBA_SKIP_HIDDEN_NORMS`.
    ///
    /// Skipping makes the output wrong but keeps it finite, so the KV cache still
    /// fills and the forward still completes. That is what makes in-situ
    /// attribution possible: `KAPLUMBA_LINEARS_ONLY` could not produce a decode
    /// step at all, because with attention skipped the logits come out empty.
    #[allow(clippy::too_many_arguments)]
    fn hidden_norm(
        &self,
        encoder: &mut MetalEncoder,
        weight: &Buffer,
        input: &Buffer,
        output: &Buffer,
        rows: usize,
        eps: f32,
    ) -> Result<(), ModelError> {
        if env_flag("KAPLUMBA_SKIP_HIDDEN_NORMS") {
            return Ok(());
        }
        self.elements
            .encode_rms_norm(
                encoder,
                input,
                weight,
                output,
                rows,
                self.config.hidden_size,
                eps,
            )
            .map_err(ModelError::from)
    }

    pub fn forward_buffer(
        &mut self,
        index: usize,
        input: &Buffer,
        output: &Buffer,
        tokens: usize,
        scratch: &ForwardScratch,
        encoder: &mut MetalEncoder,
        timer: &mut OpTimer,
        profile: &mut GpuProfile,
    ) -> Result<(), ModelError> {
        // The label prefix makes the per-layer view work: `ingest` splits
        // `layer07/q_proj` into a per-layer row and a per-kernel row, so one
        // table answers both "which kernel" and "which layer".
        let prefix = || format!("layer{index:02}/");
        // Attribution switch: skip every non-linear op so the GEMV share of a
        // decode step can be measured directly. Output is meaningless with this
        // on; it exists only to answer "where does the 793 ms go".
        if env_flag("KAPLUMBA_LINEARS_ONLY") {
            self.encode_linear(
                encoder,
                &self.weights.q_proj,
                input,
                &scratch.q,
                tokens,
                scratch,
                false,
            )?;
            self.encode_linear(
                encoder,
                &self.weights.k_proj,
                input,
                &scratch.k,
                tokens,
                scratch,
                false,
            )?;
            if let Some(v) = &self.weights.v_proj {
                self.encode_linear(encoder, v, input, &scratch.v, tokens, scratch, false)?;
            }
            self.encode_linear(
                encoder,
                &self.weights.o_proj,
                &scratch.q,
                output,
                tokens,
                scratch,
                false,
            )?;
            let gate = &scratch.gate;
            let up = &scratch.up;
            self.encode_linear(
                encoder,
                &self.weights.gate_proj,
                input,
                gate,
                tokens,
                scratch,
                false,
            )?;
            self.encode_linear(
                encoder,
                &self.weights.up_proj,
                input,
                up,
                tokens,
                scratch,
                false,
            )?;
            self.encode_linear(
                encoder,
                &self.weights.down_proj,
                up,
                output,
                tokens,
                scratch,
                false,
            )?;
            // Stamp the bypassed layers too, or their GPU time would silently
            // vanish from the totals and the table would under-report.
            profile.stamp(encoder, &format!("{}linears_only", prefix()))?;
            return Ok(());
        }
        let config = &self.config;
        let hidden = config.hidden_size;
        let inter = config.intermediate_size;
        let head_dim = config.head_dim;
        let kv_dim = config.kv_heads * head_dim;
        let eps = config.rms_norm_eps;
        let elements = tokens * hidden;
        let kv_elements = tokens * kv_dim;
        let inter_elements = tokens * inter;
        let full = config.full_attention;
        let offset = self.gpu_cache.next_position();
        // `encode_linear` routes on exactly this predicate, so one flag says
        // whether the shared input transform below is worth issuing.
        let decode = tokens == 1;

        self.hidden_norm(
            encoder,
            &self.gpu_weights.input_norm,
            input,
            &scratch.normed,
            tokens,
            eps,
        )?;

        // q, k and v all read `scratch.normed`, so transform it once and let the
        // three matvecs share it. That was 3 of the 7 input transforms a layer
        // issued, each a tiny dependent dispatch in front of a large one.
        //
        // Decode only. The prefill matmul runs `linear_input_rows` first, which
        // rewrites all `queries * in_features` elements of `transformed` from the
        // input and its signs (linear.metal, `linear_input_rows`), so whatever
        // this left there is overwritten before anything reads it: 2 dispatches
        // per layer per chunk of pure waste. `PrefillEncodeRequest` has no
        // `pretransformed` field precisely because the prefill never reuses it.
        if decode {
            self.encode_input_transform(encoder, &self.weights.q_proj, &scratch.normed, scratch)?;
        }
        // Boundary 1 of 15. The input norm -- and, in decode, the shared
        // transform it precedes -- is elementwise over `tokens * hidden` and
        // cheaper than the boundary that separates them, so they share one
        // stamp: splitting them would buy a number below the measured overhead
        // floor for 60 extra command buffers per chunk. The same reasoning folds
        // the trailing norm/residual pairs below.
        profile.stamp(encoder, &format!("{}norm_transform", prefix()))?;
        timer.lap("norm_transform");
        if decode && self.weights.q_proj.permutation_mode == 1 && self.weights.v_proj.is_some() {
            let v_proj = self.weights.v_proj.as_ref().unwrap();
            let packed_q = self.weights.q_proj.metal_buffer.as_ref().ok_or_else(|| {
                ModelError::InvalidWeights("q_proj has no GPU-resident weights".to_string())
            })?;
            let packed_k = self.weights.k_proj.metal_buffer.as_ref().ok_or_else(|| {
                ModelError::InvalidWeights("k_proj has no GPU-resident weights".to_string())
            })?;
            let packed_v = v_proj.metal_buffer.as_ref().ok_or_else(|| {
                ModelError::InvalidWeights("v_proj has no GPU-resident weights".to_string())
            })?;
            let q_out_signs = self.weights.q_proj.output_signs_buffer.as_ref().ok_or_else(|| {
                ModelError::InvalidWeights("q_proj output signs are not resident".to_string())
            })?;
            let k_out_signs = self.weights.k_proj.output_signs_buffer.as_ref().ok_or_else(|| {
                ModelError::InvalidWeights("k_proj output signs are not resident".to_string())
            })?;
            let v_out_signs = v_proj.output_signs_buffer.as_ref().ok_or_else(|| {
                ModelError::InvalidWeights("v_proj output signs are not resident".to_string())
            })?;
            let input_signs = self.weights.q_proj.input_signs_buffer.as_ref().ok_or_else(|| {
                ModelError::InvalidWeights("q_proj input signs are not resident".to_string())
            })?;
            self.linear
                .encode_gemv_fused3(
                    encoder,
                    packed_q,
                    self.weights.q_proj.metal_offset,
                    self.weights.q_proj.out_features,
                    q_out_signs,
                    &scratch.q,
                    packed_k,
                    self.weights.k_proj.metal_offset,
                    self.weights.k_proj.out_features,
                    k_out_signs,
                    &scratch.k,
                    packed_v,
                    v_proj.metal_offset,
                    v_proj.out_features,
                    v_out_signs,
                    &scratch.v,
                    &scratch.normed,
                    input_signs,
                    &scratch.transformed,
                    &scratch.raw,
                    &scratch.raw_b,
                    &scratch.raw_c,
                    self.weights.q_proj.in_features,
                    true,
                )
                .map_err(ModelError::Kernel)?;
            profile.stamp(encoder, &format!("{}qkv_proj", prefix()))?;
            timer.lap("qkv_proj");
        } else if decode && self.weights.q_proj.permutation_mode == 1 {
            let packed_q = self.weights.q_proj.metal_buffer.as_ref().ok_or_else(|| {
                ModelError::InvalidWeights("q_proj has no GPU-resident weights".to_string())
            })?;
            let packed_k = self.weights.k_proj.metal_buffer.as_ref().ok_or_else(|| {
                ModelError::InvalidWeights("k_proj has no GPU-resident weights".to_string())
            })?;
            let q_out_signs = self.weights.q_proj.output_signs_buffer.as_ref().ok_or_else(|| {
                ModelError::InvalidWeights("q_proj output signs are not resident".to_string())
            })?;
            let k_out_signs = self.weights.k_proj.output_signs_buffer.as_ref().ok_or_else(|| {
                ModelError::InvalidWeights("k_proj output signs are not resident".to_string())
            })?;
            let input_signs = self.weights.q_proj.input_signs_buffer.as_ref().ok_or_else(|| {
                ModelError::InvalidWeights("q_proj input signs are not resident".to_string())
            })?;
            self.linear
                .encode_gemv_fused2(
                    encoder,
                    packed_q,
                    self.weights.q_proj.metal_offset,
                    self.weights.q_proj.out_features,
                    q_out_signs,
                    &scratch.q,
                    packed_k,
                    self.weights.k_proj.metal_offset,
                    self.weights.k_proj.out_features,
                    k_out_signs,
                    &scratch.k,
                    &scratch.normed,
                    input_signs,
                    &scratch.transformed,
                    &scratch.raw,
                    &scratch.raw_b,
                    self.weights.q_proj.in_features,
                    true,
                )
                .map_err(ModelError::Kernel)?;
            self.elements
                .encode_copy(encoder, &scratch.k, &scratch.v, kv_elements)?;
            profile.stamp(encoder, &format!("{}qk_proj", prefix()))?;
            timer.lap("qk_proj");
        } else {
            self.encode_linear(
                encoder,
                &self.weights.q_proj,
                &scratch.normed,
                &scratch.q,
                tokens,
                scratch,
                decode,
            )?;
            profile.stamp(encoder, &format!("{}q_proj", prefix()))?;
            timer.lap("q_proj");
            self.encode_linear(
                encoder,
                &self.weights.k_proj,
                &scratch.normed,
                &scratch.k,
                tokens,
                scratch,
                decode,
            )?;
            profile.stamp(encoder, &format!("{}k_proj", prefix()))?;
            timer.lap("k_proj");
            if let Some(v_proj) = &self.weights.v_proj {
                self.encode_linear(
                    encoder,
                    v_proj,
                    &scratch.normed,
                    &scratch.v,
                    tokens,
                    scratch,
                    true,
                )?;
            } else {
                // Full attention reuses the raw K projection as V, before K is
                // normalised and rotated in place.
                self.elements
                    .encode_copy(encoder, &scratch.k, &scratch.v, kv_elements)?;
            }
            profile.stamp(encoder, &format!("{}v_proj", prefix()))?;
            timer.lap("v_proj");
        }

        self.elements.encode_rms_norm(
            encoder,
            &scratch.q,
            &self.gpu_weights.q_norm,
            &scratch.q,
            tokens * config.q_heads,
            head_dim,
            eps,
        )?;
        self.elements.encode_rms_norm(
            encoder,
            &scratch.k,
            &self.gpu_weights.k_norm,
            &scratch.k,
            tokens * config.kv_heads,
            head_dim,
            eps,
        )?;
        self.elements.encode_rms_norm_unweighted(
            encoder,
            &scratch.v,
            &scratch.v,
            tokens * config.kv_heads,
            head_dim,
            eps,
        )?;

        self.elements.encode_rope(
            encoder,
            &scratch.q,
            &scratch.q,
            RopeBufferRequest {
                tokens,
                heads: config.q_heads,
                head_dim,
                offset,
                theta: config.rope_theta,
                partial_factor: config.rope_partial_factor,
                proportional: config.rope_mode == RopeMode::Proportional,
            },
        )?;
        self.elements.encode_rope(
            encoder,
            &scratch.k,
            &scratch.k,
            RopeBufferRequest {
                tokens,
                heads: config.kv_heads,
                head_dim,
                offset,
                theta: config.rope_theta,
                partial_factor: config.rope_partial_factor,
                proportional: config.rope_mode == RopeMode::Proportional,
            },
        )?;
        // Boundary 5: three head-width norms and two ropes, five dispatches, all
        // elementwise over `tokens * head_dim * heads`. Too small to be worth a
        // boundary each.
        profile.stamp(encoder, &format!("{}kv_norm_rope", prefix()))?;
        timer.lap("kv_norm_rope");

        // Full-attention caches start small and grow on demand (see
        // `GpuKvCache::ensure_capacity`); the 10 full layers at 8192 positions
        // were most of the 1.4 GiB KV footprint. Sliding caches ring at a fixed
        // window and are left alone.
        if full {
            let device = system_device()?;
            self.gpu_cache
                .ensure_capacity(&device, offset + tokens, config.max_sequence_len)?;
        }
        let capacity = self.gpu_cache.capacity();
        let slot = self.gpu_cache.reserve(offset, tokens, full)?;
        self.elements.encode_cache_append(
            encoder,
            &scratch.k,
            &scratch.v,
            self.gpu_cache.keys(),
            self.gpu_cache.values(),
            self.gpu_cache.positions(),
            tokens,
            kv_dim,
            capacity,
            slot,
            offset,
        )?;
        self.gpu_cache.advance(tokens);
        profile.stamp(encoder, &format!("{}cache_append", prefix()))?;
        timer.lap("cache_append");

        let key_count = self.gpu_cache.key_count(full);
        let window = if full { 0 } else { config.sliding_window };
        // Attribution: skip the attention read. Output goes wrong but the KV
        // cache still fills, so the forward completes and the remaining GPU time
        // is attributable.
        if env_flag("KAPLUMBA_SKIP_ATTENTION") {
            let key_count = self.gpu_cache.key_count(full);
            let window = if full { 0 } else { config.sliding_window };
            self.attention.encode_online_attention(
                encoder,
                AttentionEncodeRequest {
                    queries: &scratch.q,
                    keys: self.gpu_cache.keys(),
                    values: self.gpu_cache.values(),
                    query_positions: &scratch.query_positions,
                    key_positions: self.gpu_cache.positions(),
                    output: &scratch.q,
                    queries_count: tokens,
                    q_heads: config.q_heads,
                    kv_heads: config.kv_heads,
                    head_dim,
                    key_count,
                    scale: 1.0 / (head_dim as f32).sqrt(),
                    window,
                },
            )?;
            profile.stamp(encoder, &format!("{}skip_attention_tail", prefix()))?;
            return Ok(());
        }
        // Split-key for the decode shape (one query). Prefill has many queries and
        // already fills the machine with `queries * q_heads` threads, so it keeps
        // the serial kernel.
        let attention_request = AttentionEncodeRequest {
            queries: &scratch.q,
            keys: self.gpu_cache.keys(),
            values: self.gpu_cache.values(),
            query_positions: &scratch.query_positions,
            key_positions: self.gpu_cache.positions(),
            output: &scratch.attended,
            queries_count: tokens,
            q_heads: config.q_heads,
            kv_heads: config.kv_heads,
            head_dim,
            key_count,
            scale: 1.0,
            window,
        };
        if tokens == 1 {
            self.attention.encode_online_attention_split(
                encoder,
                attention_request,
                &scratch.attn_partials,
            )?;
        } else {
            self.attention
                .encode_online_attention(encoder, attention_request)?;
        }
        profile.stamp(encoder, &format!("{}attention", prefix()))?;
        timer.lap("attention");

        self.encode_linear(
            encoder,
            &self.weights.o_proj,
            &scratch.attended,
            &scratch.proj,
            tokens,
            scratch,
            false,
        )?;
        profile.stamp(encoder, &format!("{}o_proj", prefix()))?;
        timer.lap("o_proj");
        if decode {
            self.elements.encode_rms_norm_residual(
                encoder,
                &scratch.proj,
                &self.gpu_weights.post_attention_norm,
                input,
                &scratch.hidden_b,
                tokens,
                self.config.hidden_size,
                eps,
            )?;
        } else {
            self.hidden_norm(
                encoder,
                &self.gpu_weights.post_attention_norm,
                &scratch.proj,
                &scratch.normed,
                tokens,
                eps,
            )?;
            self.elements.encode_residual_add(
                encoder,
                input,
                &scratch.normed,
                &scratch.hidden_b,
                elements,
            )?;
        }
        profile.stamp(encoder, &format!("{}post_attn_norm_res", prefix()))?;
        timer.lap("post_attn_norm_res");
        self.hidden_norm(
            encoder,
            &self.gpu_weights.pre_ffn_norm,
            &scratch.hidden_b,
            &scratch.normed,
            tokens,
            eps,
        )?;

        // gate and up read the same activation too -- decode only, for the
        // reason given above the q/k/v transform.
        if decode {
            self.encode_input_transform(
                encoder,
                &self.weights.gate_proj,
                &scratch.normed,
                scratch,
            )?;
        }
        profile.stamp(encoder, &format!("{}pre_ffn_norm_transform", prefix()))?;
        timer.lap("pre_ffn_norm_transform");

        if decode && self.weights.gate_proj.permutation_mode == 1 {
            let packed_gate = self.weights.gate_proj.metal_buffer.as_ref().ok_or_else(|| {
                ModelError::InvalidWeights("gate_proj has no GPU-resident weights".to_string())
            })?;
            let packed_up = self.weights.up_proj.metal_buffer.as_ref().ok_or_else(|| {
                ModelError::InvalidWeights("up_proj has no GPU-resident weights".to_string())
            })?;
            let gate_out_signs = self.weights.gate_proj.output_signs_buffer.as_ref().ok_or_else(|| {
                ModelError::InvalidWeights("gate_proj output signs are not resident".to_string())
            })?;
            let up_out_signs = self.weights.up_proj.output_signs_buffer.as_ref().ok_or_else(|| {
                ModelError::InvalidWeights("up_proj output signs are not resident".to_string())
            })?;
            let input_signs = self.weights.gate_proj.input_signs_buffer.as_ref().ok_or_else(|| {
                ModelError::InvalidWeights("gate_proj input signs are not resident".to_string())
            })?;
            let down_input_signs = self.weights.down_proj.input_signs_buffer.as_ref().ok_or_else(|| {
                ModelError::InvalidWeights("down_proj input signs are not resident".to_string())
            })?;
            self.linear
                .encode_gemv_fused2_mlp(
                    encoder,
                    packed_gate,
                    self.weights.gate_proj.metal_offset,
                    self.weights.gate_proj.out_features,
                    gate_out_signs,
                    packed_up,
                    self.weights.up_proj.metal_offset,
                    self.weights.up_proj.out_features,
                    up_out_signs,
                    Some(down_input_signs),
                    &scratch.transformed,
                    &scratch.normed,
                    input_signs,
                    &scratch.transformed,
                    &scratch.raw,
                    &scratch.raw_b,
                    self.weights.gate_proj.in_features,
                    true,
                )
                .map_err(ModelError::Kernel)?;
            profile.stamp(encoder, &format!("{}gate_up_gelu", prefix()))?;
            timer.lap("gate_up_gelu");
            self.encode_linear(
                encoder,
                &self.weights.down_proj,
                &scratch.transformed,
                &scratch.proj,
                tokens,
                scratch,
                true,
            )?;
            profile.stamp(encoder, &format!("{}down_proj", prefix()))?;
            timer.lap("down_proj");
        } else {
            self.encode_linear(
                encoder,
                &self.weights.gate_proj,
                &scratch.normed,
                &scratch.gate,
                tokens,
                scratch,
                decode,
            )?;
            profile.stamp(encoder, &format!("{}gate_proj", prefix()))?;
            timer.lap("gate_proj");
            self.encode_linear(
                encoder,
                &self.weights.up_proj,
                &scratch.normed,
                &scratch.up,
                tokens,
                scratch,
                decode,
            )?;
            profile.stamp(encoder, &format!("{}up_proj", prefix()))?;
            timer.lap("up_proj");
            self.elements.encode_gated_gelu(
                encoder,
                &scratch.gate,
                &scratch.up,
                &scratch.activated,
                inter_elements,
            )?;
            profile.stamp(encoder, &format!("{}gated_gelu", prefix()))?;
            timer.lap("gated_gelu");
            self.encode_linear(
                encoder,
                &self.weights.down_proj,
                &scratch.activated,
                &scratch.proj,
                tokens,
                scratch,
                false,
            )?;
            profile.stamp(encoder, &format!("{}down_proj", prefix()))?;
            timer.lap("down_proj");
        }
        if decode {
            self.elements.encode_rms_norm_residual_layer_scale(
                encoder,
                &scratch.proj,
                &self.gpu_weights.post_ffn_norm,
                &scratch.hidden_b,
                output,
                tokens,
                self.config.hidden_size,
                eps,
                self.weights.layer_scalar,
            )?;
        } else {
            self.hidden_norm(
                encoder,
                &self.gpu_weights.post_ffn_norm,
                &scratch.proj,
                &scratch.proj,
                tokens,
                eps,
            )?;
            self.elements.encode_residual_add(
                encoder,
                &scratch.hidden_b,
                &scratch.proj,
                output,
                elements,
            )?;
            self.elements.encode_layer_scale(
                encoder,
                output,
                output,
                elements,
                self.weights.layer_scalar,
            )?;
        }
        profile.stamp(encoder, &format!("{}post_ffn_out", prefix()))?;
        timer.lap("post_ffn_out");
        Ok(())
    }

    /// Dispatch just the input transform for `linear`, so the matvecs that share
    /// this activation can pass `pretransformed: true`.
    fn encode_input_transform(
        &self,
        encoder: &mut MetalEncoder,
        linear: &QuantizedLinear,
        input: &Buffer,
        scratch: &ForwardScratch,
    ) -> Result<(), ModelError> {
        let input_signs = linear.input_signs_buffer.as_ref().ok_or_else(|| {
            ModelError::InvalidWeights("linear input signs are not resident".to_string())
        })?;
        self.linear
            .encode_input_transform(
                encoder,
                input,
                input_signs,
                &scratch.transformed,
                linear.in_features,
            )
            .map_err(ModelError::Kernel)
    }

    fn encode_linear(
        &self,
        encoder: &mut MetalEncoder,
        linear: &QuantizedLinear,
        input: &Buffer,
        output: &Buffer,
        tokens: usize,
        scratch: &ForwardScratch,
        pretransformed: bool,
    ) -> Result<(), ModelError> {
        let packed = linear.metal_buffer.as_ref().ok_or_else(|| {
            ModelError::InvalidWeights("linear has no GPU-resident weights".to_string())
        })?;
        let input_signs = linear.input_signs_buffer.as_ref().ok_or_else(|| {
            ModelError::InvalidWeights("linear input signs are not resident".to_string())
        })?;
        let output_signs = linear.output_signs_buffer.as_ref().ok_or_else(|| {
            ModelError::InvalidWeights("linear output signs are not resident".to_string())
        })?;
        if tokens == 1 {
            self.linear.encode_gemv(
                encoder,
                LinearEncodeRequest {
                    packed,
                    packed_offset: linear.metal_offset,
                    perm_mode: linear.permutation_mode,
                    input,
                    input_signs,
                    output_signs,
                    transformed: &scratch.transformed,
                    raw: &scratch.raw,
                    output,
                    in_features: linear.in_features,
                    out_features: linear.out_features,
                    rate: linear.rate,
                    codebook: linear.codebook,
                    pretransformed,
                },
            )?;
        } else {
            self.linear.encode_gemm(
                encoder,
                PrefillEncodeRequest {
                    packed,
                    packed_offset: linear.metal_offset,
                    perm_mode: linear.permutation_mode,
                    input,
                    input_signs,
                    output_signs,
                    transformed: &scratch.transformed,
                    raw: &scratch.raw,
                    output,
                    queries: tokens,
                    in_features: linear.in_features,
                    out_features: linear.out_features,
                    rate: linear.rate,
                    codebook: linear.codebook,
                },
            )?;
        }
        Ok(())
    }

    /// Create the host cache on first use by the legacy slice path.
    fn host_cache(&mut self) -> Result<&mut KvCache, ModelError> {
        if self.cache.is_none() {
            self.cache = Some(if self.config.full_attention {
                KvCache::Full(FullKvCache::new(
                    self.config.max_sequence_len,
                    self.config.kv_heads,
                    self.config.head_dim,
                )?)
            } else {
                KvCache::Sliding(SlidingKvCache::with_capacity(
                    self.config.sliding_window,
                    self.config.kv_heads,
                    self.config.head_dim,
                )?)
            });
        }
        Ok(self.cache.as_mut().expect("host cache was just created"))
    }

    /// Diagnostic: run one layer and report the magnitude after every step.
    pub fn forward_traced(
        &mut self,
        input: &[u16],
        positions: &[u32],
    ) -> Result<(Vec<u16>, Vec<StepMagnitude>), ModelError> {
        fn rms(values: &[u16]) -> f64 {
            let sum: f64 = values
                .iter()
                .map(|v| {
                    let v = half::f16::from_bits(*v).to_f32() as f64;
                    v * v
                })
                .sum();
            (sum / values.len().max(1) as f64).sqrt()
        }
        let mut trace: Vec<StepMagnitude> = Vec::new();
        self.host_cache()?;
        self.validate_input(input, positions)?;
        let tokens = positions.len();
        trace.push(("input", rms(input)));
        let normalized = self.elements.rms_norm(
            input,
            &self.weights.input_norm,
            tokens,
            self.config.hidden_size,
            self.config.rms_norm_eps,
        )?;
        trace.push(("input_norm", rms(&normalized)));
        let q = self.run_linear(&self.weights.q_proj, &normalized, tokens)?;
        trace.push(("q_proj", rms(&q)));
        let k = self.run_linear(&self.weights.k_proj, &normalized, tokens)?;
        trace.push(("k_proj", rms(&k)));
        let raw_v = if let Some(v_proj) = &self.weights.v_proj {
            self.run_linear(v_proj, &normalized, tokens)?
        } else {
            k.clone()
        };
        trace.push(("raw_v", rms(&raw_v)));
        let q = self.elements.rms_norm(
            &q,
            &self.weights.q_norm,
            tokens * self.config.q_heads,
            self.config.head_dim,
            self.config.rms_norm_eps,
        )?;
        trace.push(("q_norm", rms(&q)));
        let k = self.elements.rms_norm(
            &k,
            &self.weights.k_norm,
            tokens * self.config.kv_heads,
            self.config.head_dim,
            self.config.rms_norm_eps,
        )?;
        trace.push(("k_norm", rms(&k)));
        let v = self.elements.rms_norm_unweighted(
            &raw_v,
            tokens * self.config.kv_heads,
            self.config.head_dim,
            self.config.rms_norm_eps,
        )?;
        trace.push(("v_norm", rms(&v)));
        let offset = positions[0] as usize;
        let q = self.elements.rope(RopeRequest {
            input: &q,
            tokens,
            heads: self.config.q_heads,
            head_dim: self.config.head_dim,
            offset,
            theta: self.config.rope_theta,
            partial_factor: self.config.rope_partial_factor,
            proportional: self.config.rope_mode == RopeMode::Proportional,
        })?;
        let k = self.elements.rope(RopeRequest {
            input: &k,
            tokens,
            heads: self.config.kv_heads,
            head_dim: self.config.head_dim,
            offset,
            theta: self.config.rope_theta,
            partial_factor: self.config.rope_partial_factor,
            proportional: self.config.rope_mode == RopeMode::Proportional,
        })?;
        trace.push(("rope", rms(&q)));
        self.append_cache(&k, &v, positions)?;
        let attended = self.run_attention(&q, positions)?;
        trace.push(("attention", rms(&attended)));
        let attended = self.run_linear(&self.weights.o_proj, &attended, tokens)?;
        trace.push(("o_proj", rms(&attended)));
        let attended = self.elements.rms_norm(
            &attended,
            &self.weights.post_attention_norm,
            tokens,
            self.config.hidden_size,
            self.config.rms_norm_eps,
        )?;
        trace.push(("post_attn_norm", rms(&attended)));
        let mut output = self.elements.residual_add(input, &attended)?;
        trace.push(("residual1", rms(&output)));
        let normalized = self.elements.rms_norm(
            &output,
            &self.weights.pre_ffn_norm,
            tokens,
            self.config.hidden_size,
            self.config.rms_norm_eps,
        )?;
        let gate = self.run_linear(&self.weights.gate_proj, &normalized, tokens)?;
        let up = self.run_linear(&self.weights.up_proj, &normalized, tokens)?;
        trace.push(("gate_proj", rms(&gate)));
        let activated = self.elements.gated_gelu(&gate, &up)?;
        trace.push(("gated_gelu", rms(&activated)));
        let mlp = self.run_linear(&self.weights.down_proj, &activated, tokens)?;
        trace.push(("down_proj", rms(&mlp)));
        let mlp = self.elements.rms_norm(
            &mlp,
            &self.weights.post_ffn_norm,
            tokens,
            self.config.hidden_size,
            self.config.rms_norm_eps,
        )?;
        trace.push(("post_ffn_norm", rms(&mlp)));
        output = self.elements.residual_add(&output, &mlp)?;
        trace.push(("residual2", rms(&output)));
        let output = self
            .elements
            .layer_scale(&output, self.weights.layer_scalar)?;
        trace.push(("layer_scale", rms(&output)));
        Ok((output, trace))
    }

    pub fn next_position(&self) -> usize {
        self.cache
            .as_ref()
            .map_or_else(|| self.gpu_cache.next_position(), KvCache::next_position)
    }

    pub fn cache_bytes(&self) -> usize {
        self.cache
            .as_ref()
            .map_or_else(|| self.gpu_cache.resident_bytes(), KvCache::resident_bytes)
    }

    pub fn reset_cache(&mut self) -> Result<(), ModelError> {
        self.gpu_cache.reset();
        if let Some(cache) = self.cache.take() {
            // Rebuilding is the cheapest way to drop the stored tensors.
            let replaced = if self.config.full_attention {
                KvCache::Full(FullKvCache::new(
                    self.config.max_sequence_len,
                    self.config.kv_heads,
                    self.config.head_dim,
                )?)
            } else {
                KvCache::Sliding(SlidingKvCache::with_capacity(
                    self.config.sliding_window,
                    self.config.kv_heads,
                    self.config.head_dim,
                )?)
            };
            drop(cache);
            self.cache = Some(replaced);
        }
        Ok(())
    }

    fn validate_input(&self, input: &[u16], positions: &[u32]) -> Result<(), ModelError> {
        if input.is_empty()
            || positions.is_empty()
            || input.len() != positions.len() * self.config.hidden_size
        {
            return Err(ModelError::InvalidInput(
                "layer input and position lengths do not match".to_string(),
            ));
        }
        if positions[0] as usize != self.next_position() {
            return Err(ModelError::InvalidInput(
                "layer positions must begin at the cache position".to_string(),
            ));
        }
        if positions
            .windows(2)
            .any(|window| window[1] != window[0] + 1)
        {
            return Err(ModelError::InvalidInput(
                "layer positions must be contiguous".to_string(),
            ));
        }
        Ok(())
    }

    fn run_linear(
        &self,
        linear: &QuantizedLinear,
        input: &[u16],
        tokens: usize,
    ) -> Result<Vec<u16>, ModelError> {
        if let Some(packed) = &linear.metal_buffer {
            let request = LinearBufferRequest {
                packed,
                packed_offset: linear.metal_offset,
                input,
                input_signs: &linear.input_signs,
                output_signs: &linear.output_signs,
                in_features: linear.in_features,
                out_features: linear.out_features,
                rate: linear.rate,
                codebook: linear.codebook,
            };
            return if tokens == 1 {
                Ok(self.linear.gemv_buffer(request)?)
            } else {
                Ok(self.linear.gemm_chunk_buffer(request)?)
            };
        }
        if tokens == 1 {
            Ok(self.linear.gemv(LinearRequest {
                packed: &linear.packed,
                input,
                input_signs: &linear.input_signs,
                output_signs: &linear.output_signs,
                in_features: linear.in_features,
                out_features: linear.out_features,
                rate: linear.rate,
                codebook: linear.codebook,
            })?)
        } else {
            Ok(self.linear.gemm_chunk(PrefillRequest {
                packed: &linear.packed,
                input,
                input_signs: &linear.input_signs,
                output_signs: &linear.output_signs,
                queries: tokens,
                in_features: linear.in_features,
                out_features: linear.out_features,
                rate: linear.rate,
                codebook: linear.codebook,
            })?)
        }
    }

    fn append_cache(
        &mut self,
        keys: &[u16],
        values: &[u16],
        positions: &[u32],
    ) -> Result<(), ModelError> {
        let position = positions[0] as usize;
        let cache = self.host_cache()?;
        match cache {
            KvCache::Full(cache) => {
                cache.append_chunk(position, keys, values)?;
            }
            KvCache::Sliding(cache) => {
                cache.append_chunk(position, keys, values)?;
            }
        }
        Ok(())
    }

    fn run_attention(
        &mut self,
        queries: &[u16],
        positions: &[u32],
    ) -> Result<Vec<u16>, ModelError> {
        // Ensure the host cache exists, then borrow the fields we need
        // disjointly so the kernel call can see both at once.
        self.host_cache()?;
        let DecoderLayer {
            attention,
            config,
            cache,
            ..
        } = self;
        let cache = cache
            .as_ref()
            .ok_or_else(|| ModelError::InvalidConfig("host cache missing".to_string()))?;
        let q_heads = config.q_heads;
        let kv_heads = config.kv_heads;
        let head_dim = config.head_dim;
        let full_attention = config.full_attention;
        let sliding_window = config.sliding_window;
        let window = if full_attention { 0 } else { sliding_window };
        match cache {
            KvCache::Full(cache) => {
                let count = cache.active_key_count();
                let stream_len = cache.stream_len();
                Ok(attention.online_attention(AttentionRequest {
                    queries,
                    keys: &cache.key_storage()[..count * stream_len],
                    values: &cache.value_storage()[..count * stream_len],
                    query_positions: positions,
                    key_positions: cache.key_positions(),
                    q_heads,
                    kv_heads,
                    head_dim,
                    scale: 1.0,
                    window,
                })?)
            }
            KvCache::Sliding(cache) => Ok(attention.online_attention(AttentionRequest {
                queries,
                keys: cache.key_storage(),
                values: cache.value_storage(),
                query_positions: positions,
                key_positions: cache.key_positions(),
                q_heads,
                kv_heads,
                head_dim,
                scale: 1.0,
                window,
            })?),
        }
    }
}

/// Fixed-size activation scratch reused by every layer of every forward.
///
/// Sizes depend only on the configured chunk length, never on the context
/// length, so steady-state decoding allocates nothing and the working set stays
/// constant as the conversation grows.
pub struct ForwardScratch {
    pub max_tokens: usize,
    pub hidden_a: Buffer,
    pub hidden_b: Buffer,
    pub normed: Buffer,
    pub q: Buffer,
    pub k: Buffer,
    pub v: Buffer,
    pub attended: Buffer,
    pub proj: Buffer,
    pub gate: Buffer,
    pub up: Buffer,
    pub activated: Buffer,
    /// FP32 accumulator shared by every linear in the batch.
    pub raw: Buffer,
    pub raw_b: Buffer,
    pub raw_c: Buffer,
    /// FP16 input-transform staging, shared by every linear in the batch.
    pub transformed: Buffer,
    pub query_positions: Buffer,
    pub token_ids: Buffer,
    pub last_hidden: Buffer,
    pub logits: Buffer,
    pub capped: Buffer,
    /// Split-KV decode scratch: one unnormalised `(m, l, O)` partial per
    /// (split, query head), so `splits * q_heads * (head_dim + 2)` floats. Sized
    /// once from the layer shapes, never from a key count.
    pub attn_partials: Buffer,
    // There is deliberately no greedy-pick scratch here. The device-side pick
    // it fed (`ElementKernels::encode_greedy_pick`, whose `greedy_key` packs
    // `0xFFFF - index` into 16 bits) is wrong for this model's 262,144-entry
    // vocabulary: every index at or past 65,536 underflows that subtraction, the
    // packed key stops being ordered, and the "winner" is a low token id rather
    // than the largest logit. Measured on the real checkpoint: the host's argmax
    // over the same `capped` buffer is index 9,259 at 28.5, while the device's
    // reduction returns key 0xFFFFFFFF, i.e. index 0. Greedy decode therefore
    // takes the argmax on the host, off `capped`, which is the only path that has
    // ever produced correct text.
}

impl ForwardScratch {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_tokens: usize,
        hidden: usize,
        q_max: usize,
        kv_max: usize,
        inter: usize,
        vocab: usize,
        attn_partials: usize,
    ) -> Result<Self, ModelError> {
        if max_tokens == 0 || hidden == 0 || vocab == 0 {
            return Err(ModelError::InvalidConfig(
                "forward scratch dimensions must be non-zero".to_string(),
            ));
        }
        let device = system_device()?;
        let options = metal::MTLResourceOptions::StorageModeShared;
        let f32_elements = |rows: usize, width: usize| -> Result<Buffer, ModelError> {
            let bytes = rows
                .checked_mul(width)
                .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| ModelError::InvalidConfig("scratch shape overflows".to_string()))?;
            Ok(device.new_buffer(bytes as u64, options))
        };
        let u16_elements = |rows: usize, width: usize| -> Result<Buffer, ModelError> {
            let bytes = rows
                .checked_mul(width)
                .and_then(|count| count.checked_mul(std::mem::size_of::<u16>()))
                .ok_or_else(|| ModelError::InvalidConfig("scratch shape overflows".to_string()))?;
            Ok(device.new_buffer(bytes as u64, options))
        };
        // The vocabulary head writes one f32 accumulator per output feature, so
        // the shared accumulator must cover a full single-row head pass.
        let raw_elements = (max_tokens * inter).max(vocab);
        let raw = device.new_buffer((raw_elements * std::mem::size_of::<f32>()) as u64, options);
        let raw_b = device.new_buffer((raw_elements * std::mem::size_of::<f32>()) as u64, options);
        let raw_c = device.new_buffer((raw_elements * std::mem::size_of::<f32>()) as u64, options);
        let query_positions =
            device.new_buffer((max_tokens * std::mem::size_of::<u32>()) as u64, options);
        let token_ids =
            device.new_buffer((max_tokens * std::mem::size_of::<u32>()) as u64, options);
        Ok(Self {
            max_tokens,
            hidden_a: u16_elements(max_tokens, hidden)?,
            hidden_b: u16_elements(max_tokens, hidden)?,
            normed: u16_elements(max_tokens, hidden)?,
            q: u16_elements(max_tokens, q_max)?,
            k: u16_elements(max_tokens, kv_max)?,
            v: u16_elements(max_tokens, kv_max)?,
            attended: u16_elements(max_tokens, hidden)?,
            proj: u16_elements(max_tokens, hidden)?,
            gate: u16_elements(max_tokens, inter)?,
            up: u16_elements(max_tokens, inter)?,
            activated: u16_elements(max_tokens, inter)?,
            raw,
            raw_b,
            raw_c,
            transformed: u16_elements(max_tokens, inter)?,
            query_positions,
            token_ids,
            last_hidden: u16_elements(1, hidden)?,
            logits: u16_elements(1, vocab)?,
            capped: u16_elements(1, vocab)?,
            // One (m, l, O[head_dim]) partial per (KV split, query head) for the
            // widest layer shape. This replaces a `q_heads * key_count` fp32
            // score matrix, which was O(L^2), sized from the cache capacity seen
            // at construction, and hard-failed decode at 1215 keys of context:
            // `attention score buffer is 131072 bytes, needs 155520`. The
            // replacement is fixed-size and independent of context length, per
            // AGENTS.md §8; 1.05 MB at 16 splits x 32 heads x 512 dims.
            attn_partials: f32_elements(1, attn_partials)?,
        })
    }

    pub fn resident_bytes(&self) -> u64 {
        [
            &self.hidden_a,
            &self.hidden_b,
            &self.normed,
            &self.q,
            &self.k,
            &self.v,
            &self.attended,
            &self.proj,
            &self.gate,
            &self.up,
            &self.activated,
            &self.raw,
            &self.transformed,
            &self.query_positions,
            &self.token_ids,
            &self.last_hidden,
            &self.logits,
            &self.capped,
            &self.attn_partials,
        ]
        .iter()
        .map(|buffer| buffer.length())
        .sum()
    }
}

pub struct TextModel {
    embedding: Vec<u16>,
    embedding_buffer: Option<Buffer>,
    /// Byte offset of the embedding rows within `embedding_buffer`.
    embedding_offset: u64,
    /// `embedding_buffer` holds raw BF16 exactly as stored in the checkpoint, so
    /// the embedding kernel must widen it. True when the table is offloaded.
    embedding_is_bf16: bool,
    vocab_size: usize,
    hidden_size: usize,
    layers: Vec<DecoderLayer>,
    final_norm: Vec<u16>,
    lm_head: QuantizedLinear,
    final_softcap: f32,
    _checkpoint_weights: Option<TextWeights>,
    elements: &'static ElementKernels,
    linear: &'static LinearKernels,
    /// Queue used for the single command buffer that carries a whole forward.
    forward_queue: metal::CommandQueue,
    final_norm_buffer: Buffer,
    scratch: ForwardScratch,
    scratch_tokens: usize,
    /// MEASUREMENT AID: destination for the injected `spin_burn` dispatch, and
    /// the thread count it is dispatched with. Allocated lazily, and only when
    /// `KAPLUMBA_DECODE_INJECT` asks for it, so a normal process does not carry
    /// it. Sized by nothing but [`Self::SPIN_THREADS`], so it does not grow with
    /// context length (AGENTS.md §8).
    spin_scratch: Option<Buffer>,
}

impl TextModel {
    /// Threads the injected `spin_burn` dispatch runs. Wide enough to fill the
    /// GPU's cores several times over, so the loop's cost is set by `iters` and
    /// the clock rather than by occupancy.
    const SPIN_THREADS: u64 = 4096;
    pub fn new(
        embedding: Vec<u16>,
        layers: Vec<DecoderLayer>,
        final_norm: Vec<u16>,
        lm_head: QuantizedLinear,
        final_softcap: f32,
    ) -> Result<Self, ModelError> {
        if embedding.is_empty()
            || layers.is_empty()
            || !final_softcap.is_finite()
            || final_softcap <= 0.0
        {
            return Err(ModelError::InvalidConfig(
                "text model components must be non-empty and softcap must be positive".to_string(),
            ));
        }
        let hidden_size = layers[0].config.hidden_size;
        let vocab_size = embedding
            .len()
            .checked_div(hidden_size)
            .ok_or_else(|| ModelError::InvalidConfig("embedding shape is invalid".to_string()))?;
        if vocab_size == 0
            || embedding.len() != vocab_size * hidden_size
            || final_norm.len() != hidden_size
            || lm_head.in_features != hidden_size
            || lm_head.out_features != vocab_size
        {
            return Err(ModelError::InvalidConfig(
                "text model dimensions do not match embedding, layers, and head".to_string(),
            ));
        }
        if layers
            .iter()
            .any(|layer| layer.config.hidden_size != hidden_size)
        {
            return Err(ModelError::InvalidConfig(
                "all decoder layers must use the same hidden size".to_string(),
            ));
        }
        lm_head.validate()?;
        Self::finish_construction(
            embedding,
            None,
            0,
            false,
            vocab_size,
            hidden_size,
            layers,
            final_norm,
            lm_head,
            final_softcap,
            None,
        )
    }

    /// Shared tail of both constructors: uploads the read-only vectors the
    /// buffer path needs and allocates the fixed-size activation scratch.
    #[allow(clippy::too_many_arguments)]
    fn finish_construction(
        embedding: Vec<u16>,
        embedding_buffer: Option<Buffer>,
        embedding_offset: u64,
        embedding_is_bf16: bool,
        vocab_size: usize,
        hidden_size: usize,
        layers: Vec<DecoderLayer>,
        final_norm: Vec<u16>,
        mut lm_head: QuantizedLinear,
        final_softcap: f32,
        checkpoint_weights: Option<TextWeights>,
    ) -> Result<Self, ModelError> {
        let device = system_device()?;
        upload_sign_buffers(&mut lm_head)?;
        let final_norm_buffer = u16_device_buffer(&final_norm)?;
        let mut q_max = 0;
        let mut kv_max = 0;
        let mut inter = 0;
        let mut attn_partials = 1;
        for layer in &layers {
            q_max = q_max.max(layer.config.q_heads * layer.config.head_dim);
            kv_max = kv_max.max(layer.config.kv_heads * layer.config.head_dim);
            inter = inter.max(layer.config.intermediate_size);
            attn_partials = attn_partials.max(decode_partial_elements(
                layer.config.q_heads,
                layer.config.head_dim,
            ));
        }
        let scratch = ForwardScratch::new(
            1,
            hidden_size,
            q_max,
            kv_max,
            inter,
            vocab_size,
            attn_partials,
        )?;
        Ok(Self {
            embedding,
            embedding_buffer,
            embedding_offset,
            embedding_is_bf16,
            vocab_size,
            hidden_size,
            layers,
            final_norm,
            lm_head,
            final_softcap,
            _checkpoint_weights: checkpoint_weights,
            elements: ElementKernels::shared()?,
            linear: LinearKernels::shared()?,
            forward_queue: device.new_command_queue(),
            final_norm_buffer,
            scratch,
            scratch_tokens: 1,
            spin_scratch: None,
        })
    }

    /// Convenience wrapper for callers that pass an exactly-sized buffer.
    pub fn from_metal_parts(
        embedding_buffer: Buffer,
        layers: Vec<DecoderLayer>,
        final_norm: Vec<u16>,
        lm_head: QuantizedLinear,
        final_softcap: f32,
        checkpoint_weights: Option<TextWeights>,
    ) -> Result<Self, ModelError> {
        let bytes = embedding_buffer.length();
        Self::from_metal_parts_with_len(
            embedding_buffer,
            bytes,
            0,
            false,
            layers,
            final_norm,
            lm_head,
            final_softcap,
            checkpoint_weights,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_metal_parts_with_len(
        embedding_buffer: Buffer,
        embedding_bytes: u64,
        embedding_offset: u64,
        embedding_is_bf16: bool,
        layers: Vec<DecoderLayer>,
        final_norm: Vec<u16>,
        lm_head: QuantizedLinear,
        final_softcap: f32,
        checkpoint_weights: Option<TextWeights>,
    ) -> Result<Self, ModelError> {
        if layers.is_empty() || !final_softcap.is_finite() || final_softcap <= 0.0 {
            return Err(ModelError::InvalidConfig(
                "metal text model components are invalid".to_string(),
            ));
        }
        let hidden_size = layers[0].config.hidden_size;
        // Use the caller's length, not the buffer's: an offloaded embedding is a
        // page-aligned window over the shard mapping and is therefore longer than
        // the tensor, which would inflate the derived vocabulary size.
        let embedding_bytes = usize::try_from(embedding_bytes)
            .map_err(|_| ModelError::InvalidConfig("embedding buffer is too large".to_string()))?;
        let embedding_elements = embedding_bytes
            .checked_div(2)
            .ok_or_else(|| ModelError::InvalidConfig("embedding buffer is not FP16".to_string()))?;
        let vocab_size = embedding_elements
            .checked_div(hidden_size)
            .ok_or_else(|| ModelError::InvalidConfig("embedding shape is invalid".to_string()))?;
        if vocab_size == 0
            || embedding_elements != vocab_size * hidden_size
            || final_norm.len() != hidden_size
            || lm_head.in_features != hidden_size
            || lm_head.out_features != vocab_size
        {
            return Err(ModelError::InvalidConfig(
                "metal text model dimensions do not match".to_string(),
            ));
        }
        lm_head.validate()?;
        Self::finish_construction(
            Vec::new(),
            Some(embedding_buffer),
            embedding_offset,
            embedding_is_bf16,
            vocab_size,
            hidden_size,
            layers,
            final_norm,
            lm_head,
            final_softcap,
            checkpoint_weights,
        )
    }

    pub fn load_from_dir(model_dir: &Path) -> Result<Self, ModelError> {
        // Serialise GPU access before anything is mapped. The model needs
        // 11.5-14 GiB; a second concurrent instance pushes the machine into swap
        // and invalidates every measurement taken alongside it. A second process
        // waits here rather than failing, so a fleet of agents queues up.
        crate::gpu_lock::acquire(&format!("load {}", model_dir.display()))
            .map_err(|error| ModelError::GpuBusy(error.to_string()))?;
        let metadata = inspect_model_dir(model_dir)
            .map_err(|error| ModelError::Checkpoint(error.to_string()))?;
        metadata
            .config
            .validate_current_checkpoint()
            .map_err(|error| ModelError::Checkpoint(error.to_string()))?;
        let context = MetalContext::new()?;
        let weights = TextWeights::load_dedicated(
            model_dir,
            &metadata.selection,
            context,
            DEFAULT_RESIDENT_LIMIT_BYTES,
        )?;
        // The embedding is offloaded when the loader mapped it instead of copying
        // it, in which case the view is a page-aligned window and its length
        // exceeds the tensor, so the real byte length is passed separately.
        const EMBED_NAME: &str = "model.language_model.embed_tokens.weight";
        let embedding_bytes = weights
            .tensors()
            .get(EMBED_NAME)
            .map_or(0, |tensor| tensor.end - tensor.start);
        // `no_copy_view` page-aligns downward, so the tensor's rows start at a
        // non-zero offset inside the returned window. That offset has to reach
        // both the GPU kernel and the CPU reader, or every embedding row is read
        // from the wrong place.
        // Prefer a raw BF16 view so the 2.82 GB table stays mapped rather than
        // copied; the kernel widens each element as it reads it.
        let (embedding_buffer, embedding_offset, embedding_is_bf16) =
            match weights.raw_zero_copy_tensor_buffer(EMBED_NAME)? {
                Some((buffer, offset)) => (buffer, offset, true),
                None => (weights.tensor_buffer(EMBED_NAME)?, 0, false),
            };
        let mut layers = Vec::with_capacity(metadata.config.num_hidden_layers);
        for index in 0..metadata.config.num_hidden_layers {
            let full = metadata.config.layer_types[index].is_full();
            let head_dim = if full {
                metadata.config.global_head_dim
            } else {
                metadata.config.head_dim
            };
            let kv_heads = if full {
                metadata.config.num_global_key_value_heads
            } else {
                metadata.config.num_key_value_heads
            };
            let layer_config = LayerConfig {
                hidden_size: metadata.config.hidden_size,
                intermediate_size: metadata.config.intermediate_size,
                q_heads: metadata.config.num_attention_heads,
                kv_heads,
                head_dim,
                max_sequence_len: 8192,
                full_attention: full,
                sliding_window: if full {
                    0
                } else {
                    metadata.config.sliding_window
                },
                rms_norm_eps: metadata.config.rms_norm_eps,
                rope_theta: if full {
                    metadata.config.rope.full.theta
                } else {
                    metadata.config.rope.sliding.theta
                },
                rope_mode: if full {
                    RopeMode::Proportional
                } else {
                    RopeMode::Default
                },
                rope_partial_factor: metadata
                    .config
                    .rope
                    .full
                    .partial_rotary_factor
                    .unwrap_or(1.0),
            };
            let prefix = format!("model.language_model.layers.{index}");
            let layer_weights = DecoderLayerWeights {
                q_proj: load_quantized_linear(
                    &weights,
                    &format!("{prefix}.self_attn.q_proj"),
                    metadata.config.hidden_size,
                    metadata.config.num_attention_heads * head_dim,
                    Exl3Rate::Integer(3),
                    Exl3Codebook::Mcg,
                )?,
                k_proj: load_quantized_linear(
                    &weights,
                    &format!("{prefix}.self_attn.k_proj"),
                    metadata.config.hidden_size,
                    kv_heads * head_dim,
                    Exl3Rate::Integer(3),
                    Exl3Codebook::Mcg,
                )?,
                v_proj: if full {
                    None
                } else {
                    Some(load_quantized_linear(
                        &weights,
                        &format!("{prefix}.self_attn.v_proj"),
                        metadata.config.hidden_size,
                        kv_heads * head_dim,
                        Exl3Rate::Integer(3),
                        Exl3Codebook::Mcg,
                    )?)
                },
                o_proj: load_quantized_linear(
                    &weights,
                    &format!("{prefix}.self_attn.o_proj"),
                    metadata.config.num_attention_heads * head_dim,
                    metadata.config.hidden_size,
                    Exl3Rate::Integer(3),
                    Exl3Codebook::Mcg,
                )?,
                gate_proj: load_quantized_linear(
                    &weights,
                    &format!("{prefix}.mlp.gate_proj"),
                    metadata.config.hidden_size,
                    metadata.config.intermediate_size,
                    Exl3Rate::Integer(3),
                    Exl3Codebook::Mcg,
                )?,
                up_proj: load_quantized_linear(
                    &weights,
                    &format!("{prefix}.mlp.up_proj"),
                    metadata.config.hidden_size,
                    metadata.config.intermediate_size,
                    Exl3Rate::Integer(3),
                    Exl3Codebook::Mcg,
                )?,
                down_proj: load_quantized_linear(
                    &weights,
                    &format!("{prefix}.mlp.down_proj"),
                    metadata.config.intermediate_size,
                    metadata.config.hidden_size,
                    Exl3Rate::Integer(3),
                    Exl3Codebook::Mcg,
                )?,
                q_norm: weights.read_f16(&format!("{prefix}.self_attn.q_norm.weight"))?,
                k_norm: weights.read_f16(&format!("{prefix}.self_attn.k_norm.weight"))?,
                input_norm: weights.read_f16(&format!("{prefix}.input_layernorm.weight"))?,
                post_attention_norm: weights
                    .read_f16(&format!("{prefix}.post_attention_layernorm.weight"))?,
                pre_ffn_norm: weights
                    .read_f16(&format!("{prefix}.pre_feedforward_layernorm.weight"))?,
                post_ffn_norm: weights
                    .read_f16(&format!("{prefix}.post_feedforward_layernorm.weight"))?,
                layer_scalar: half::f16::from_bits(
                    weights.read_f16(&format!("{prefix}.layer_scalar"))?[0],
                )
                .to_f32(),
            };
            if env_flag("KAPLUMBA_LOAD_TRACE") {
                eprintln!("layer {index} building...");
            }
            layers.push(DecoderLayer::new(layer_config, layer_weights)?);
            if env_flag("KAPLUMBA_LOAD_TRACE") {
                eprintln!("layer {index} done");
            }
        }
        let final_norm = weights.read_f16("model.language_model.norm.weight")?;
        let lm_head = load_quantized_linear(
            &weights,
            "lm_head",
            metadata.config.hidden_size,
            metadata.config.vocab_size,
            Exl3Rate::Integer(6),
            Exl3Codebook::Mcg,
        )?;
        let mut model = Self::from_metal_parts_with_len(
            embedding_buffer,
            embedding_bytes,
            embedding_offset,
            embedding_is_bf16,
            layers,
            final_norm,
            lm_head,
            metadata.config.final_logit_softcapping,
            Some(weights),
        )?;
        if metadata.quantization.layout.as_deref() == Some("colmajor") {
            for layer in &mut model.layers {
                for linear in [
                    Some(&mut layer.weights.q_proj),
                    Some(&mut layer.weights.k_proj),
                    layer.weights.v_proj.as_mut(),
                    Some(&mut layer.weights.o_proj),
                    Some(&mut layer.weights.gate_proj),
                    Some(&mut layer.weights.up_proj),
                    Some(&mut layer.weights.down_proj),
                ]
                .into_iter()
                .flatten()
                {
                    linear.permutation_mode = 1;
                }
            }
        }
        model.validate_current_graph()?;
        model.warmup_decode()?;
        Ok(model)
    }

    pub fn warmup_decode(&mut self) -> Result<(), ModelError> {
        let _ = self.forward_buffer(&[1], &[0])?;
        self.reset_caches()?;
        Ok(())
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn validate_current_graph(&self) -> Result<(), ModelError> {
        if self.layers.len() != 60
            || self.hidden_size != 5376
            || self.vocab_size != 262144
            || self.lm_head.rate != Exl3Rate::Integer(6)
            || self.lm_head.codebook != Exl3Codebook::Mcg
        {
            return Err(ModelError::InvalidConfig(
                "model does not match the confirmed Gemma 4 text graph".to_string(),
            ));
        }
        for (index, layer) in self.layers.iter().enumerate() {
            let expected_full = index % 6 == 5;
            let expected_kv_heads = if expected_full { 4 } else { 16 };
            let expected_head_dim = if expected_full { 512 } else { 256 };
            if layer.config.full_attention != expected_full
                || layer.config.hidden_size != self.hidden_size
                || layer.config.intermediate_size != 21504
                || layer.config.q_heads != 32
                || layer.config.kv_heads != expected_kv_heads
                || layer.config.head_dim != expected_head_dim
                || (!expected_full && layer.config.sliding_window != 1024)
            {
                return Err(ModelError::InvalidConfig(format!(
                    "layer {index} does not match the confirmed Gemma 4 graph"
                )));
            }
        }
        Ok(())
    }

    /// Run a full forward pass entirely on the GPU.
    ///
    /// The whole 60-layer pass, the vocabulary head, and the logit softcap are
    /// encoded into a single command buffer and synchronised once; only the final
    /// softcapped logits are read back. This removes the ~1,260 per-token
    /// command-buffer barriers that dominated decode latency.
    ///
    /// Prefill only needs logits for the position that is actually sampled, so
    /// the 262,144-wide vocabulary head runs for the last row alone.
    /// The queue the forward runs on, so a caller can queue several steps.
    pub fn forward_queue(&self) -> metal::CommandQueue {
        self.forward_queue.clone()
    }

    /// Encode one decode step at `position` on the device, taking its input
    /// token from the host.
    ///
    /// `token` is the previous step's sample, staged into `scratch.token_ids[0]`,
    /// which is what `embedding_scaled` reads for a single-token step
    /// (`crates/kernels/src/elements.metal`).
    ///
    /// # The caller must hold a barrier across the next call
    ///
    /// This stages `token_ids[0]` and `query_positions[0]` into the *shared*
    /// scratch from the host, and every layer reuses those buffers. So it is
    /// only correct when nothing that reads them is still in flight: commit this
    /// step's encoder and wait before starting the next one. A caller that
    /// queues several steps and waits once at the end races those writes against
    /// the previous step's kernels, and which way the race lands is a function of
    /// how far the GPU got while the host encoded — that is, of scheduling. The
    /// same applies to `GpuCache::ensure_capacity`, which copies the live KV
    /// prefix on the host from inside a forward.
    ///
    /// There is no device-side token hand-off to overlap the steps with, and
    /// there cannot be one at present: the pick's only host-visible output entry
    /// point, `ElementKernels::encode_greedy_pick`, refuses a destination index
    /// of 0, which is the index the next step's embedding reads. See
    /// [`ForwardScratch`] for why the pick cannot be used at all on this
    /// vocabulary.
    pub fn forward_buffer_into_step(
        &mut self,
        encoder: &mut MetalEncoder,
        position: u32,
        token: u32,
    ) -> Result<(), ModelError> {
        if token as usize >= self.vocab_size {
            return Err(ModelError::InvalidInput(format!(
                "token id {token} is outside the vocabulary"
            )));
        }
        // NOTE: deliberately *not* bracketed by `decode_profile::begin_step` /
        // `close_step`. `generate.rs` owns that bracket for the real decode loop
        // -- it has to, because it is the only caller that knows the host wall
        // time of the step -- and a bracket here as well would stamp four
        // boundaries and record two steps, one of them a slice of the other.
        self.forward_buffer_into(&[token], &[position], encoder)?;
        Ok(())
    }

    /// The softcapped logits row of the last forward, as the host sees it.
    ///
    /// This is a read of `scratch.capped`, and `capped` is written by the
    /// forward's last kernels, so it is only meaningful once the encoder that
    /// ran the forward has completed. Reading it earlier returns whatever the
    /// previous forward left there: a freshly allocated buffer reads as all
    /// zeros, which an argmax turns into token 0.
    pub fn last_logits(&self) -> Vec<u16> {
        let logits = self.scratch.capped.contents() as *const u16;
        unsafe { std::slice::from_raw_parts(logits, self.vocab_size) }.to_vec()
    }

    pub fn forward_buffer_into_vec(
        &mut self,
        token_ids: &[u32],
        positions: &[u32],
        out: &mut Vec<u16>,
    ) -> Result<(), ModelError> {
        // Validate before creating the encoder: an early return would drop a
        // command encoder that was never ended, which Metal asserts on.
        if token_ids.len() != positions.len() || token_ids.is_empty() {
            return Err(ModelError::InvalidInput(
                "model token and position counts must match and be non-empty".to_string(),
            ));
        }
        let queue = self.forward_queue.clone();
        let mut encoder = MetalEncoder::new(&queue)?;
        // `KAPLUMBA_DECODE_PROFILE=step` brackets a single-token forward with
        // two stamped boundaries so its GPU-timeline duration can be read. Only
        // a one-token forward is a decode step; a prefill is timed by its caller
        // and must not be able to enter a decode total.
        let decode_step = token_ids.len() == 1;
        if decode_step {
            encoder.set_max_dispatches(2500);
            crate::decode_profile::begin_step(&mut encoder, |encoder| {
                self.calibration_dispatch(encoder)
            })?;
        }
        // On any error the encoder is left un-ended, which Metal asserts on at
        // dealloc, so abandon it explicitly before propagating.
        if let Err(error) = self.forward_buffer_into(token_ids, positions, &mut encoder) {
            encoder.abandon();
            return Err(error);
        }
        // Read the tally BEFORE `commit_and_wait`, which consumes the encoder.
        if decode_step && std::env::var_os("KAPLUMBA_COUNT_BUFFERS").is_some() {
            let (buffers, steps) = decode_counters();
            buffers.fetch_add(encoder.command_buffers_created(), Ordering::Relaxed);
            steps.fetch_add(1, Ordering::Relaxed);
        }
        let started = std::time::Instant::now();
        if decode_step {
            crate::decode_profile::close_step(&mut encoder)?;
            crate::decode_profile::end_step(&mut encoder, started.elapsed().as_secs_f64() * 1e3)?;
        }
        encoder.commit_and_wait()?;
        out.resize(self.vocab_size, 0);
        let logits = self.scratch.capped.contents() as *const u16;
        unsafe {
            std::ptr::copy_nonoverlapping(logits, out.as_mut_ptr(), self.vocab_size);
        }
        Ok(())
    }

    /// One forward step, committed and waited, returning the logits.
    pub fn forward_buffer(
        &mut self,
        token_ids: &[u32],
        positions: &[u32],
    ) -> Result<Vec<u16>, ModelError> {
        let mut out = Vec::with_capacity(self.vocab_size);
        self.forward_buffer_into_vec(token_ids, positions, &mut out)?;
        Ok(out)
    }

    /// Encode one forward into a caller-supplied encoder, without committing.
    ///
    /// The encoder is committed and waited by the caller, which must hold that
    /// barrier across the next call: `forward_chunk` stages `token_ids` and
    /// `query_positions` into the shared scratch from the host. See
    /// [`Self::forward_buffer_into_step`].
    /// Feed `token_ids` through the model, splitting long prompts into bounded
    /// chunks so the activation scratch is sized by the chunk and not by the
    /// prompt length.
    ///
    /// AGENTS.md §8 requires the scratch to be explicitly owned, fixed-size and
    /// independent of context length. A single-shot pass over an 8192-token
    /// prompt made `ForwardScratch` about 2.5 GiB -- `raw` alone is
    /// `tokens * inter * 4` bytes (672 MiB at 8192 tokens, `inter = 21504`).
    /// Chunking keeps the whole scratch under the 100 MB budget.
    ///
    /// Chunking is semantically inert: positions are absolute, the KV cache
    /// accumulates across chunks (`GPUKeyValueCache::advance` runs during each
    /// chunk's encode, so `next_position` is correct for the next chunk), and
    /// every row attends to exactly the keys it would have seen in one pass. The
    /// LM head already runs on the last row only, so only the final chunk's
    /// logits are returned.
    ///
    /// Under `KAPLUMBA_GPU_PROFILE=1` this is also the scope of one profile
    /// report: the [`GpuProfile`] is created here, spans every chunk, and prints
    /// one table when the last chunk is done. A 614-token prompt is one
    /// chunk and one table; an 8192-token prompt is eight and one table.
    /// Returns nothing: the softcapped logits row is read separately, through
    /// [`Self::last_logits`], once the caller's encoder has completed.
    pub fn forward_buffer_into(
        &mut self,
        token_ids: &[u32],
        positions: &[u32],
        encoder: &mut MetalEncoder,
    ) -> Result<(), ModelError> {
        if token_ids.len() != positions.len() || token_ids.is_empty() {
            return Err(ModelError::InvalidInput(
                "model token and position counts must match and be non-empty".to_string(),
            ));
        }
        // Prefill (many tokens) is profiled whenever the switch is on. A decode
        // step is one token, and stamping it would print a whole table per
        // generated token, so it takes the second switch: with that unset, a
        // decode step does no profiling work at all rather than paying for
        // measurements nobody reads.
        //
        // `KAPLUMBA_DECODE_PROFILE=kernel` is the third way in, and it exists
        // because the two switches above cannot measure a decode step: with
        // only the first set, every table a decode run prints is the PREFILL's,
        // whose own_ms sum is a function of the prompt and so is flat in
        // `--max-new-tokens`. This mode stamps the step and hands the table to
        // the decode accumulator instead of printing it per token.
        let decode_kernel = token_ids.len() == 1
            && crate::decode_profile::mode() == crate::decode_profile::Mode::Kernel;
        let profile = decode_kernel
            || (env_flag("KAPLUMBA_GPU_PROFILE")
                && (token_ids.len() > 1 || env_flag("KAPLUMBA_GPU_PROFILE_DECODE")));
        let mut gpu = GpuProfile::new(profile);
        gpu.tokens = token_ids.len();
        gpu.layers = self.layers.len();
        let chunk = prefill_chunk();
        let result: Result<(), ModelError> = if token_ids.len() <= chunk {
            gpu.chunks += 1;
            gpu.chunk_rows = token_ids.len();
            self.forward_chunk(token_ids, positions, encoder, &mut gpu, true)
        } else {
            // Only the last chunk's logits survive this loop, so only the last
            // chunk may run the vocabulary head. `peek` is the whole test: the
            // chunk iterator is deterministic, so "no next chunk" is exactly
            // "last", and it costs no second pass over the token ids.
            let mut chunks = token_ids
                .chunks(chunk)
                .zip(positions.chunks(chunk))
                .peekable();
            let mut outcome = Ok(());
            while let Some((chunk_ids, chunk_positions)) = chunks.next() {
                gpu.chunks += 1;
                gpu.chunk_rows = chunk_ids.len();
                let last = chunks.peek().is_none();
                if let Err(error) =
                    self.forward_chunk(chunk_ids, chunk_positions, encoder, &mut gpu, last)
                {
                    // Report what ran before the failure rather than throwing
                    // the measurements away; then propagate.
                    outcome = Err(error);
                    break;
                }
            }
            outcome
        };
        if profile {
            // Drain the tail before printing: the last boundaries are still
            // retained, and a partial report would under-count.
            gpu.drain(encoder)?;
            if decode_kernel {
                // One decode step's kernels, accumulated across every step into
                // a single report. See `decode_profile`.
                let span = gpu.gpu_span_s().map(|(start, end)| (end - start) * 1e3);
                crate::decode_profile::record_kernel_step(span, gpu.own_ms(), gpu.kernel_rows());
            } else {
                self.report_gpu_profile(&gpu);
            }
        }
        result
    }

    /// One chunk of the forward pass. `token_ids` must be at most
    /// [`PREFILL_CHUNK`] long; every buffer it touches is sized from that.
    ///
    /// `run_head` is false for every chunk but the last of a multi-chunk prefill.
    /// Those chunks' logits are dropped by [`Self::forward_buffer_into`], so the
    /// row lift, the 262,144-wide vocabulary head and the greedy pick are dead
    /// work for them: the head alone is 262,144 x 5,376 weights at 6 bpw =
    /// 1.06 GB read per invocation, and a three-chunk prefill was paying it
    /// three times to keep one row.
    ///
    /// Nothing downstream of the head can notice. `forward_buffer_into` returns
    /// the last chunk's logits and no other chunk's, and the next chunk
    /// overwrites `scratch.token_ids` from the host before the first dispatch
    /// that reads it — which is only safe because the caller holds a barrier
    /// between chunks, i.e. because the prefill below is committed and waited
    /// before the first decode step. A single-chunk prefill
    /// -- every decode step, and any prompt up to [`PREFILL_CHUNK`] tokens --
    /// takes the `true` branch and is bit-for-bit unchanged.
    fn forward_chunk(
        &mut self,
        token_ids: &[u32],
        positions: &[u32],
        encoder: &mut MetalEncoder,
        gpu: &mut GpuProfile,
        run_head: bool,
    ) -> Result<(), ModelError> {
        if token_ids.len() != positions.len() || token_ids.is_empty() {
            return Err(ModelError::InvalidInput(
                "model token and position counts must match and be non-empty".to_string(),
            ));
        }
        let tokens = token_ids.len();
        // Chunks resume at an absolute position past the cache cursor when
        // their cache append is skipped by an attribution switch
        // (KAPLUMBA_LINEARS_ONLY / KAPLUMBA_SKIP_ATTENTION); any other gap is
        // still an error.
        if positions[0] as usize != self.gpu_next_position()
            && !(env_flag("KAPLUMBA_LINEARS_ONLY") || env_flag("KAPLUMBA_SKIP_ATTENTION"))
        {
            return Err(ModelError::InvalidInput(
                "positions must begin at the current cache position".to_string(),
            ));
        }
        if positions.windows(2).any(|w| w[1] != w[0] + 1) {
            return Err(ModelError::InvalidInput(
                "positions must be contiguous".to_string(),
            ));
        }
        if let Some(bad) = token_ids.iter().find(|id| **id as usize >= self.vocab_size) {
            return Err(ModelError::InvalidInput(format!(
                "token id {bad} is outside the vocabulary"
            )));
        }
        self.ensure_scratch(tokens)?;
        if gpu.enabled() {
            // Price the boundary before the real work, on the same queue, so the
            // table can say what a sub-microsecond row is actually made of.
            self.calibrate_gpu_profile(encoder, gpu)?;
        }

        // Shared-memory staging for the two small host-side arrays.
        //
        // These are HOST writes into SHARED memory, so every command buffer
        // already committed must have finished reading them. With the rolling
        // scheme up to MAX_IN_FLIGHT are outstanding, and skipping this made
        // greedy output depend on scheduling: six runs of one 614-token prompt
        // produced two different answers. The wait costs nothing measurable --
        // the previous chunk's scratch is reused by this one regardless, so the
        // GPU had to reach that point before this chunk could start anyway.
        encoder.flush_and_wait()?;
        unsafe {
            let position_ptr = self.scratch.query_positions.contents() as *mut u32;
            std::ptr::copy_nonoverlapping(positions.as_ptr(), position_ptr, tokens);
            let token_ptr = self.scratch.token_ids.contents() as *mut u32;
            std::ptr::copy_nonoverlapping(token_ids.as_ptr(), token_ptr, tokens);
        }

        // KAPLUMBA_PROFILE_PHASES=1 commits and waits at each phase boundary
        // instead of batching the whole forward into one command buffer. The
        // absolute numbers are slightly pessimistic because it removes the
        // natural overlap between phases, but the distribution is what we need to
        // find out where a decode step actually goes.
        //
        // This is the *host* profiler: `phase` measures the CPU, and the flushes
        // make the host wait for the GPU so the number is "GPU time plus
        // submission latency". It is demoted below the GPU-clock table
        // (KAPLUMBA_GPU_PROFILE) for exactly that reason, and kept because the
        // flush boundaries are the only way to see the coarse split without
        // paying for a command buffer per op.
        let profile = env_flag("KAPLUMBA_PROFILE_PHASES");
        macro_rules! gpu_flush {
            () => {
                if profile {
                    encoder.flush_and_wait().map_err(ModelError::from)?;
                }
            };
        }
        let mut phase = std::time::Instant::now();
        let mut embed_ms = 0.0_f64;
        let mut layer_ms = 0.0_f64;
        let mut head_ms = 0.0_f64;
        macro_rules! mark {
            ($acc:ident) => {
                if profile {
                    $acc += phase.elapsed().as_secs_f64() * 1e3;
                    phase = std::time::Instant::now();
                }
            };
        }
        let hidden = self.hidden_size;
        let vocab = self.vocab_size;
        let scale = (hidden as f32).sqrt();

        // MEASUREMENT AID: a deliberately slow dispatch inside the chunk, so a
        // timing instrument can be shown to resolve a cost injected into the
        // region it claims to measure. `KAPLUMBA_DECODE_INJECT=<dispatches>`
        // with `KAPLUMBA_DECODE_INJECT_ITERS=<n>` (default 4096 threads x
        // 200000 iterations per dispatch). The decode accumulator records the
        // injection in its report, and the host wall clock gives the injected
        // cost without reference to the instrument, so the two can be compared.
        let inject = if tokens == 1 {
            env_usize("KAPLUMBA_DECODE_INJECT")
        } else {
            0
        };
        if inject > 0 {
            self.ensure_spin_scratch()?;
            let scratch = self
                .spin_scratch
                .as_ref()
                .ok_or_else(|| ModelError::InvalidConfig("spin scratch is absent".to_string()))?;
            self.elements.encode_spin_burn(
                encoder,
                scratch,
                inject,
                env_u32("KAPLUMBA_DECODE_INJECT_ITERS").unwrap_or(200_000),
                Self::SPIN_THREADS,
            )?;
        }

        // Layer 0 consumes the embedding, then layers ping-pong between the two
        // hidden buffers. Input/output never alias within a layer.
        let embedding_buffer = self
            .embedding_buffer
            .as_ref()
            .ok_or_else(|| ModelError::InvalidConfig("model has no GPU embedding".to_string()))?;
        self.elements.encode_embedding_scaled(
            encoder,
            &self.scratch.token_ids,
            embedding_buffer,
            &self.scratch.hidden_a,
            tokens,
            hidden,
            vocab,
            scale,
            self.embedding_offset,
            self.embedding_is_bf16,
        )?;
        gpu_flush!();
        gpu.stamp(encoder, "chunk/embedding")?;

        if profile {
            mark!(embed_ms);
        }
        // Host encode cost per op, summed over every layer. Printed *below* the
        // GPU-clock table and labelled as submit cost: a dispatch handed to an
        // asynchronous queue has not executed when the encode call returns, so
        // this measures the CPU, not the GPU. Useful only as a share.
        let mut op_us: std::collections::BTreeMap<&'static str, f64> =
            std::collections::BTreeMap::new();
        let mut timer = OpTimer::new(&mut op_us, profile);
        let layers = self.layers.len();
        for index in 0..layers {
            let (input, output) = if index % 2 == 0 {
                (&self.scratch.hidden_a, &self.scratch.hidden_b)
            } else {
                (&self.scratch.hidden_b, &self.scratch.hidden_a)
            };
            self.layers[index].forward_buffer(
                index,
                input,
                output,
                tokens,
                &self.scratch,
                encoder,
                &mut timer,
                gpu,
            )?;
            if profile {
                // Per-layer host-flush cost. `phase` is read but not reset here:
                // it is the start of this layer, and resetting per layer would
                // leave the loop's last assignment dead. `mark!(layer_ms)` below
                // closes the whole layer region once.
                gpu_flush!();
                let now = std::time::Instant::now();
                eprintln!(
                    "  LAYER {index:>2} {:>8.2} ms",
                    now.duration_since(phase).as_secs_f64() * 1e3
                );
            }
        }
        // Close the layer region so it is counted in the phase total. Without
        // this the sum omits every layer -- the defect that made the old
        // `PHASE` line useless.
        if profile {
            mark!(layer_ms);
        }
        let encode_done = std::time::Instant::now();
        if profile {
            let mut total_us = 0.0;
            for (name, us) in &op_us {
                eprintln!(
                    "  ENCODE {name:<22} {:>9.2} ms  (host submit, not GPU)",
                    us / 1e3
                );
                total_us += *us;
            }
            eprintln!(
                "  ENCODE (all layers' host encode-submit cost {:.2} ms; the first lap of each \
                  layer also absorbs that layer's flush wait, so it is not comparable to the \
                  others. For GPU-clock per-kernel time use KAPLUMBA_GPU_PROFILE=1.)",
                encode_done.duration_since(phase).as_secs_f64() * 1e3 + total_us / 1e3
            );
        }
        gpu_flush!();
        // Layer k writes into hidden_b when k is even and hidden_a when k is odd,
        // so the final hidden state lives in hidden_a for an even layer count.
        let final_hidden = if layers.is_multiple_of(2) {
            &self.scratch.hidden_a
        } else {
            &self.scratch.hidden_b
        };

        if !run_head {
            // The head block is skipped whole, not partially: `last_hidden` keeps
            // whatever the calibration self-copy left in it and `capped` is never
            // written, so the caller must not read a buffer this chunk never
            // filled. See `run_head` above.
            return Ok(());
        }

        // Only the sampled position needs logits, so lift the final row out and
        // run the 262,144-wide vocabulary head for that row alone.
        self.elements.encode_copy_last_row(
            encoder,
            final_hidden,
            &self.scratch.last_hidden,
            tokens,
            hidden,
        )?;
        gpu.stamp(encoder, "chunk/copy_last_row")?;
        // `encode_lm_head` stamps its own sub-steps (head/final_norm, head/gemv,
        // head/softcap).
        self.encode_lm_head(encoder, gpu)?;
        gpu_flush!();
        // No greedy pick is encoded here. It used to be, into
        // `scratch.token_ids[tokens]` — one element past the end of that buffer
        // whenever a chunk filled it, into the very buffer the next step's
        // embedding reads, while the decode loop read back a
        // `scratch.greedy_token` that no kernel ever wrote. It is also wrong on
        // its own terms: `greedy_key` packs `0xFFFF - index` into 16 bits, which
        // underflows for every index at or past 65,536 and so for most of this
        // model's 262,144-entry vocabulary, making the packed key unordered and
        // the winner an arbitrary low token id. Measured on the real checkpoint,
        // the host argmax over this same `capped` buffer was index 9,259 at
        // 28.5 while the device reduction returned 0xFFFFFFFF, i.e. index 0 —
        // which is why the decode loop that used it emitted `max_new_tokens`
        // copies of token 0 for every prompt. The argmax is taken on the host
        // below instead.
        mark!(head_ms);
        if profile {
            let total = embed_ms + layer_ms + head_ms;
            eprintln!(
                "PHASE tokens={tokens} embed={embed_ms:.2}ms layers={layer_ms:.2}ms head={head_ms:.2}ms total={total:.2}ms"
            );
            eprintln!(
                "  (host flush-boundary phases, NOT GPU-clock time: this is CPU encode plus \
                  submission latency, and each boundary is a serialisation point. For per-kernel \
                  GPU time use KAPLUMBA_GPU_PROFILE=1.)"
            );
        }
        // The logits are deliberately *not* returned: `capped` is written by the
        // kernels encoded just above, so a host read here is a read-before-write
        // that returns the previous forward's row (all zeros on a fresh buffer,
        // which an argmax turns into token 0). A caller reads it through
        // `last_logits` after its barrier instead.
        let _ = vocab;
        Ok(())
    }

    /// One trivial dispatch for the `cal/one` calibration sample: a one-element
    /// copy of the row buffer onto itself, which is the cheapest real kernel
    /// available without a scratch buffer of its own. Source and destination are
    /// the same address, so the calibration cannot perturb any data the forward
    /// goes on to read.
    pub(crate) fn calibration_dispatch(
        &self,
        encoder: &mut MetalEncoder,
    ) -> Result<(), ModelError> {
        self.elements
            .encode_copy(
                encoder,
                &self.scratch.last_hidden,
                &self.scratch.last_hidden,
                1,
            )
            .map_err(ModelError::from)
    }

    /// Price the boundary itself, so a reader can discount sub-microsecond rows:
    /// a committed buffer with nothing in it, and one holding a single trivial
    /// dispatch. Both are stamped and read back exactly like a real boundary, so
    /// these are measured floors, not estimates.
    ///
    /// Runs on the caller's encoder and therefore on the same queue, ahead of the
    /// real work, so it is part of the same prefill the caller then times. The
    /// previous chunk's tail boundaries may still be retained at this point;
    /// [`GpuProfile::ingest_calibration`] keeps them.
    fn calibrate_gpu_profile(
        &self,
        encoder: &mut MetalEncoder,
        profile: &mut GpuProfile,
    ) -> Result<(), ModelError> {
        const SAMPLES: usize = 32;
        /// Well under the ~64-outstanding point where `new_command_buffer`
        /// blocks: nothing waits until a batch is committed.
        const BATCH: usize = 8;
        let mut empty = Vec::with_capacity(SAMPLES);
        let mut one = Vec::with_capacity(SAMPLES);
        for index in 0..SAMPLES {
            // Nothing encoded: pure command-buffer overhead. Stamping an empty
            // encoder is legal (end_encoding on an encoder with no dispatches),
            // and that is exactly the boundary cost we want to price.
            encoder
                .commit_stamped("cal/empty")
                .map_err(ModelError::from)?;
            self.calibration_dispatch(encoder)?;
            encoder
                .commit_stamped("cal/one")
                .map_err(ModelError::from)?;
            if (index + 1) % BATCH == 0 {
                let stamps = encoder.take_gpu_stamps();
                let (batch_empty, batch_one) = profile.ingest_calibration(&stamps);
                empty.extend(batch_empty);
                one.extend(batch_one);
            }
        }
        profile.set_calibration(median_floor_ms(empty), median_floor_ms(one));
        Ok(())
    }

    /// Print the accumulated GPU profile. One call covers every chunk of the
    /// forward, so a 614-token prefill is one table.
    fn report_gpu_profile(&self, profile: &GpuProfile) {
        eprint!("{}", profile.render());
    }

    fn encode_lm_head(
        &self,
        encoder: &mut MetalEncoder,
        profile: &mut GpuProfile,
    ) -> Result<(), ModelError> {
        let hidden = self.hidden_size;
        let vocab = self.vocab_size;
        // In-place: each thread owns one row, so normalising last_hidden onto
        // itself is safe and avoids a second buffer.
        self.elements.encode_rms_norm(
            encoder,
            &self.scratch.last_hidden,
            &self.final_norm_buffer,
            &self.scratch.last_hidden,
            1,
            hidden,
            self.layers[0].config.rms_norm_eps,
        )?;
        profile.stamp(encoder, "head/final_norm")?;
        let packed = self.lm_head.metal_buffer.as_ref().ok_or_else(|| {
            ModelError::InvalidWeights("lm head has no GPU-resident weights".to_string())
        })?;
        let input_signs =
            self.lm_head.input_signs_buffer.as_ref().ok_or_else(|| {
                ModelError::InvalidWeights("lm head signs not resident".to_string())
            })?;
        let output_signs =
            self.lm_head.output_signs_buffer.as_ref().ok_or_else(|| {
                ModelError::InvalidWeights("lm head signs not resident".to_string())
            })?;
        self.linear.encode_gemv(
            encoder,
            LinearEncodeRequest {
                packed,
                packed_offset: self.lm_head.metal_offset,
                perm_mode: self.lm_head.permutation_mode,
                input: &self.scratch.last_hidden,
                input_signs,
                output_signs,
                transformed: &self.scratch.transformed,
                raw: &self.scratch.raw,
                output: &self.scratch.logits,
                in_features: self.lm_head.in_features,
                out_features: vocab,
                rate: self.lm_head.rate,
                codebook: self.lm_head.codebook,
                pretransformed: false,
            },
        )?;
        // The 262,144-wide vocabulary projection is the single largest kernel in
        // the chunk after the layer GEMMs, so it gets its own boundary rather
        // than hiding inside the head's phase total.
        profile.stamp(encoder, "head/gemv")?;
        self.elements.encode_softcap(
            encoder,
            &self.scratch.logits,
            &self.scratch.capped,
            vocab,
            self.final_softcap,
        )?;
        profile.stamp(encoder, "head/softcap")?;
        Ok(())
    }

    /// Grow the scratch if a larger chunk is requested; steady state is a no-op.
    fn ensure_scratch(&mut self, tokens: usize) -> Result<(), ModelError> {
        if tokens <= self.scratch.max_tokens {
            self.scratch_tokens = tokens;
            return Ok(());
        }
        let mut q_max = 0;
        let mut kv_max = 0;
        let mut inter = 0;
        let mut attn_partials = 1;
        for layer in &self.layers {
            q_max = q_max.max(layer.config.q_heads * layer.config.head_dim);
            kv_max = kv_max.max(layer.config.kv_heads * layer.config.head_dim);
            inter = inter.max(layer.config.intermediate_size);
            attn_partials = attn_partials.max(decode_partial_elements(
                layer.config.q_heads,
                layer.config.head_dim,
            ));
        }
        self.scratch = ForwardScratch::new(
            tokens,
            self.hidden_size,
            q_max,
            kv_max,
            inter,
            self.vocab_size,
            attn_partials,
        )?;
        self.scratch_tokens = tokens;
        Ok(())
    }

    /// Allocate the `spin_burn` destination on first use.
    ///
    /// 16 KiB: `SPIN_THREADS` floats. Fixed size, independent of the model and
    /// of the context length, which is what AGENTS.md §8 requires of any buffer
    /// this engine holds. Private storage: the kernel only writes it and nothing
    /// ever reads it, so there is no reason to make it host-visible.
    fn ensure_spin_scratch(&mut self) -> Result<(), ModelError> {
        if self.spin_scratch.is_none() {
            let device = system_device()?;
            self.spin_scratch = Some(device.new_buffer(
                Self::SPIN_THREADS * std::mem::size_of::<f32>() as u64,
                metal::MTLResourceOptions::StorageModePrivate,
            ));
        }
        Ok(())
    }

    /// Bytes this engine has deliberately allocated and keeps resident.
    ///
    /// This is the figure the 18 GB budget is about: checkpoint weights, KV
    /// caches, activation scratch, and embedding. It is independent of what the
    /// kernel's `phys_footprint` happens to charge at any instant.
    pub fn allocated_bytes(&self) -> u64 {
        let weights = self
            ._checkpoint_weights
            .as_ref()
            .map_or(0, TextWeights::resident_bytes);
        let kv: usize = self
            .layers
            .iter()
            .map(|layer| layer.gpu_cache.resident_bytes())
            .sum();
        weights + kv as u64 + self.scratch.resident_bytes()
    }

    /// Per-component breakdown of [`Self::allocated_bytes`].
    pub fn allocation_breakdown(&self) -> (u64, u64, u64) {
        let weights = self
            ._checkpoint_weights
            .as_ref()
            .map_or(0, TextWeights::resident_bytes);
        let kv: u64 = self
            .layers
            .iter()
            .map(|layer| layer.gpu_cache.resident_bytes() as u64)
            .sum();
        (weights, kv, self.scratch.resident_bytes())
    }

    pub fn scratch_resident_bytes(&self) -> u64 {
        self.scratch.resident_bytes()
    }

    /// Rows the activation scratch is currently sized for.
    ///
    /// Must never exceed [`PREFILL_CHUNK`]: AGENTS.md §8 requires the scratch to
    /// be fixed-size and independent of context length, so this is an assertion
    /// surface for "did a long prompt sneak past the chunking".
    pub fn scratch_max_tokens(&self) -> usize {
        self.scratch.max_tokens
    }

    pub fn gpu_next_position(&self) -> usize {
        self.layers
            .first()
            .map_or(0, |layer| layer.gpu_cache.next_position())
    }

    pub fn reset_gpu_caches(&mut self) {
        for layer in &mut self.layers {
            layer.gpu_cache.reset();
        }
    }

    /// Raw embedding rows for `token_ids`, without the scale factor.
    pub fn embedding(&self, token_ids: &[u32]) -> Vec<u16> {
        let mut output = vec![0_u16; token_ids.len() * self.hidden_size];
        let source_ptr: *const u16 = if let Some(buffer) = &self.embedding_buffer {
            unsafe { (buffer.contents() as *const u16).add(self.embedding_offset as usize / 2) }
        } else {
            self.embedding.as_ptr()
        };
        for (row, token) in token_ids.iter().enumerate() {
            let token = *token as usize;
            if token >= self.vocab_size {
                continue;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    source_ptr.add(token * self.hidden_size),
                    output.as_mut_ptr().add(row * self.hidden_size),
                    self.hidden_size,
                );
            }
        }
        output
    }

    /// Diagnostic: trace one layer's internal magnitudes.
    pub fn trace_layer(
        &mut self,
        index: usize,
        hidden: &[u16],
        positions: &[u32],
    ) -> Vec<StepMagnitude> {
        self.layers[index]
            .forward_traced(hidden, positions)
            .expect("traced layer forward")
            .1
    }

    /// Run a single layer on the host path. Diagnostic only.
    pub fn forward_layer(&mut self, index: usize, hidden: &[u16], positions: &[u32]) -> Vec<u16> {
        self.layers[index].forward(hidden, positions).unwrap()
    }

    /// Select the EXL3 tile permutation used by every projection.
    pub fn set_permutation_mode(&mut self, mode: u32) {
        self.lm_head.permutation_mode = mode;
        for layer in &mut self.layers {
            for linear in [
                Some(&mut layer.weights.q_proj),
                Some(&mut layer.weights.k_proj),
                layer.weights.v_proj.as_mut(),
                Some(&mut layer.weights.o_proj),
                Some(&mut layer.weights.gate_proj),
                Some(&mut layer.weights.up_proj),
                Some(&mut layer.weights.down_proj),
            ]
            .into_iter()
            .flatten()
            {
                linear.permutation_mode = mode;
            }
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn next_position(&self) -> usize {
        self.layers.first().map_or(0, DecoderLayer::next_position)
    }

    pub fn reset_caches(&mut self) -> Result<(), ModelError> {
        for layer in &mut self.layers {
            layer.reset_cache()?;
        }
        Ok(())
    }

    pub fn cache_bytes(&self) -> usize {
        self.layers.iter().map(DecoderLayer::cache_bytes).sum()
    }

    pub fn forward(
        &mut self,
        token_ids: &[u32],
        positions: &[u32],
    ) -> Result<Vec<u16>, ModelError> {
        if token_ids.len() != positions.len() || token_ids.is_empty() {
            return Err(ModelError::InvalidInput(
                "model token and position counts must match and be non-empty".to_string(),
            ));
        }
        let mut hidden = if let Some(embedding) = &self.embedding_buffer {
            self.elements.embedding_scaled_buffer(
                token_ids,
                embedding,
                self.hidden_size,
                self.vocab_size,
                (self.hidden_size as f32).sqrt(),
            )?
        } else {
            self.elements.embedding_scaled(
                token_ids,
                &self.embedding,
                self.hidden_size,
                self.vocab_size,
                (self.hidden_size as f32).sqrt(),
            )?
        };
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, positions)?;
        }
        let tokens = positions.len();
        hidden = self.elements.rms_norm(
            &hidden,
            &self.final_norm,
            tokens,
            self.hidden_size,
            self.layers[0].config.rms_norm_eps,
        )?;
        let logits = if let Some(packed) = &self.lm_head.metal_buffer {
            let request = LinearBufferRequest {
                packed,
                packed_offset: self.lm_head.metal_offset,
                input: &hidden,
                input_signs: &self.lm_head.input_signs,
                output_signs: &self.lm_head.output_signs,
                in_features: self.lm_head.in_features,
                out_features: self.lm_head.out_features,
                rate: self.lm_head.rate,
                codebook: self.lm_head.codebook,
            };
            if tokens == 1 {
                self.linear.gemv_buffer(request)?
            } else {
                self.linear.gemm_chunk_buffer(request)?
            }
        } else if tokens == 1 {
            self.linear.gemv(LinearRequest {
                packed: &self.lm_head.packed,
                input: &hidden,
                input_signs: &self.lm_head.input_signs,
                output_signs: &self.lm_head.output_signs,
                in_features: self.lm_head.in_features,
                out_features: self.lm_head.out_features,
                rate: self.lm_head.rate,
                codebook: self.lm_head.codebook,
            })?
        } else {
            self.linear.gemm_chunk(PrefillRequest {
                packed: &self.lm_head.packed,
                input: &hidden,
                input_signs: &self.lm_head.input_signs,
                output_signs: &self.lm_head.output_signs,
                queries: tokens,
                in_features: self.lm_head.in_features,
                out_features: self.lm_head.out_features,
                rate: self.lm_head.rate,
                codebook: self.lm_head.codebook,
            })?
        };
        Ok(self.elements.softcap(&logits, self.final_softcap)?)
    }
}

fn load_quantized_linear(
    weights: &TextWeights,
    name: &str,
    in_features: usize,
    out_features: usize,
    rate: Exl3Rate,
    codebook: Exl3Codebook,
) -> Result<QuantizedLinear, ModelError> {
    let trellis_name = format!("{name}.trellis");
    // Use the resident arena copy when the loader materialised one, and only fall
    // back to the shard mapping when it was told to skip the copy.
    let copied = weights
        .tensors()
        .get(&trellis_name)
        .is_some_and(|tensor| !tensor.spans.is_empty());
    let (packed, packed_offset) = if copied {
        (weights.tensor_buffer(&trellis_name)?, 0)
    } else {
        weights
            .zero_copy_tensor_buffer(&trellis_name)?
            .ok_or_else(|| ModelError::InvalidWeights(format!("{trellis_name} is not resident")))?
    };
    let packed_len = weights
        .tensors()
        .get(&trellis_name)
        .map(|tensor| (tensor.source_len / 2) as usize);
    if env_flag("KAPLUMBA_DEBUG_OFFSETS") {
        eprintln!(
            "OFFSET {name:<52} packed_offset={packed_offset:>12} aligned4={} len={packed_len:?}",
            packed_offset % 4 == 0
        );
    }
    let input_signs = read_sign_component(weights, name, "suh", "su", in_features)?;
    let output_signs = read_sign_component(weights, name, "svh", "sv", out_features)?;
    QuantizedLinear::from_metal_buffer(
        in_features,
        out_features,
        rate,
        codebook,
        packed,
        packed_offset,
        packed_len,
        input_signs,
        output_signs,
    )
}

fn read_sign_component(
    weights: &TextWeights,
    name: &str,
    half_suffix: &str,
    packed_suffix: &str,
    expected: usize,
) -> Result<Vec<u16>, ModelError> {
    if let Ok(values) = weights.read_f16(&format!("{name}.{half_suffix}")) {
        if values.len() != expected {
            return Err(ModelError::InvalidWeights(format!(
                "{name}.{half_suffix} has {} values, expected {expected}",
                values.len()
            )));
        }
        return Ok(values);
    }
    let packed = weights.read_f16(&format!("{name}.{packed_suffix}"))?;
    let values = unpack_signs(&packed)
        .into_iter()
        .map(|value| half::f16::from_f32(value).to_bits())
        .collect::<Vec<_>>();
    if values.len() != expected {
        return Err(ModelError::InvalidWeights(format!(
            "{name}.{packed_suffix} expands to {} values, expected {expected}",
            values.len()
        )));
    }
    Ok(values)
}
