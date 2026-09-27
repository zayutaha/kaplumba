use std::path::Path;

use metal::Buffer;
use thiserror::Error;

use kaplumba_kernels::attention::{AttentionEncodeRequest, AttentionKernels, AttentionRequest};
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

pub struct DecoderLayer {
    pub config: LayerConfig,
    pub weights: DecoderLayerWeights,
    /// Host mirror of the cache, created on demand for the legacy slice path.
    ///
    /// The GPU path keeps its authoritative cache in `gpu_cache`, so holding a
    /// second full copy here would add ~1.5 GB of duplicated KV at 8k context
    /// and can push the real checkpoint past the Metal allocation ceiling.
    cache: Option<KvCache>,
    elements: ElementKernels,
    linear: LinearKernels,
    attention: AttentionKernels,
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
            elements: ElementKernels::new()?,
            linear: LinearKernels::new()?,
            attention: AttentionKernels::new()?,
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
    pub fn forward_buffer(
        &mut self,
        input: &Buffer,
        output: &Buffer,
        tokens: usize,
        scratch: &ForwardScratch,
        encoder: &mut MetalEncoder,
    ) -> Result<(), ModelError> {
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

        self.elements.encode_rms_norm(
            encoder,
            input,
            &self.gpu_weights.input_norm,
            &scratch.normed,
            tokens,
            hidden,
            eps,
        )?;

        self.encode_linear(
            encoder,
            &self.weights.q_proj,
            &scratch.normed,
            &scratch.q,
            tokens,
            scratch,
        )?;
        self.encode_linear(
            encoder,
            &self.weights.k_proj,
            &scratch.normed,
            &scratch.k,
            tokens,
            scratch,
        )?;
        if let Some(v_proj) = &self.weights.v_proj {
            self.encode_linear(
                encoder,
                v_proj,
                &scratch.normed,
                &scratch.v,
                tokens,
                scratch,
            )?;
        } else {
            // Full attention reuses the raw K projection as V, before K is
            // normalised and rotated in place.
            self.elements
                .encode_copy(encoder, &scratch.k, &scratch.v, kv_elements)?;
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

        // Full-attention caches start small and grow on demand (see
        // `GpuKvCache::ensure_capacity`); the 10 full layers at 8192 positions
        // were most of the 1.4 GiB KV footprint. Sliding caches ring at a fixed
        // window and are left alone.
        if full {
            let device = system_device()?;
            self.gpu_cache.ensure_capacity(
                &device,
                offset + tokens,
                config.max_sequence_len,
            )?;
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
                output: &scratch.attended,
                queries_count: tokens,
                q_heads: config.q_heads,
                kv_heads: config.kv_heads,
                head_dim,
                key_count,
                scale: 1.0,
                window,
            },
        )?;

        self.encode_linear(
            encoder,
            &self.weights.o_proj,
            &scratch.attended,
            &scratch.proj,
            tokens,
            scratch,
        )?;
        self.elements.encode_rms_norm(
            encoder,
            &scratch.proj,
            &self.gpu_weights.post_attention_norm,
            &scratch.normed,
            tokens,
            hidden,
            eps,
        )?;
        self.elements.encode_residual_add(
            encoder,
            input,
            &scratch.normed,
            &scratch.hidden_b,
            elements,
        )?;
        self.elements.encode_rms_norm(
            encoder,
            &scratch.hidden_b,
            &self.gpu_weights.pre_ffn_norm,
            &scratch.normed,
            tokens,
            hidden,
            eps,
        )?;

        self.encode_linear(
            encoder,
            &self.weights.gate_proj,
            &scratch.normed,
            &scratch.gate,
            tokens,
            scratch,
        )?;
        self.encode_linear(
            encoder,
            &self.weights.up_proj,
            &scratch.normed,
            &scratch.up,
            tokens,
            scratch,
        )?;
        self.elements.encode_gated_gelu(
            encoder,
            &scratch.gate,
            &scratch.up,
            &scratch.activated,
            inter_elements,
        )?;
        self.encode_linear(
            encoder,
            &self.weights.down_proj,
            &scratch.activated,
            &scratch.proj,
            tokens,
            scratch,
        )?;
        self.elements.encode_rms_norm(
            encoder,
            &scratch.proj,
            &self.gpu_weights.post_ffn_norm,
            &scratch.proj,
            tokens,
            hidden,
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
        Ok(())
    }

    fn encode_linear(
        &self,
        encoder: &mut MetalEncoder,
        linear: &QuantizedLinear,
        input: &Buffer,
        output: &Buffer,
        tokens: usize,
        scratch: &ForwardScratch,
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
            input, &self.weights.input_norm, tokens, self.config.hidden_size, self.config.rms_norm_eps,
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
        let q = self.elements.rms_norm(&q, &self.weights.q_norm, tokens * self.config.q_heads, self.config.head_dim, self.config.rms_norm_eps)?;
        trace.push(("q_norm", rms(&q)));
        let k = self.elements.rms_norm(&k, &self.weights.k_norm, tokens * self.config.kv_heads, self.config.head_dim, self.config.rms_norm_eps)?;
        trace.push(("k_norm", rms(&k)));
        let v = self.elements.rms_norm_unweighted(&raw_v, tokens * self.config.kv_heads, self.config.head_dim, self.config.rms_norm_eps)?;
        trace.push(("v_norm", rms(&v)));
        let offset = positions[0] as usize;
        let q = self.elements.rope(RopeRequest { input: &q, tokens, heads: self.config.q_heads, head_dim: self.config.head_dim, offset, theta: self.config.rope_theta, partial_factor: self.config.rope_partial_factor, proportional: self.config.rope_mode == RopeMode::Proportional })?;
        let k = self.elements.rope(RopeRequest { input: &k, tokens, heads: self.config.kv_heads, head_dim: self.config.head_dim, offset, theta: self.config.rope_theta, partial_factor: self.config.rope_partial_factor, proportional: self.config.rope_mode == RopeMode::Proportional })?;
        trace.push(("rope", rms(&q)));
        self.append_cache(&k, &v, positions)?;
        let attended = self.run_attention(&q, positions)?;
        trace.push(("attention", rms(&attended)));
        let attended = self.run_linear(&self.weights.o_proj, &attended, tokens)?;
        trace.push(("o_proj", rms(&attended)));
        let attended = self.elements.rms_norm(&attended, &self.weights.post_attention_norm, tokens, self.config.hidden_size, self.config.rms_norm_eps)?;
        trace.push(("post_attn_norm", rms(&attended)));
        let mut output = self.elements.residual_add(input, &attended)?;
        trace.push(("residual1", rms(&output)));
        let normalized = self.elements.rms_norm(&output, &self.weights.pre_ffn_norm, tokens, self.config.hidden_size, self.config.rms_norm_eps)?;
        let gate = self.run_linear(&self.weights.gate_proj, &normalized, tokens)?;
        let up = self.run_linear(&self.weights.up_proj, &normalized, tokens)?;
        trace.push(("gate_proj", rms(&gate)));
        let activated = self.elements.gated_gelu(&gate, &up)?;
        trace.push(("gated_gelu", rms(&activated)));
        let mlp = self.run_linear(&self.weights.down_proj, &activated, tokens)?;
        trace.push(("down_proj", rms(&mlp)));
        let mlp = self.elements.rms_norm(&mlp, &self.weights.post_ffn_norm, tokens, self.config.hidden_size, self.config.rms_norm_eps)?;
        trace.push(("post_ffn_norm", rms(&mlp)));
        output = self.elements.residual_add(&output, &mlp)?;
        trace.push(("residual2", rms(&output)));
        let output = self.elements.layer_scale(&output, self.weights.layer_scalar)?;
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
    /// FP16 input-transform staging, shared by every linear in the batch.
    pub transformed: Buffer,
    pub query_positions: Buffer,
    pub token_ids: Buffer,
    pub last_hidden: Buffer,
    pub logits: Buffer,
    pub capped: Buffer,
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
    ) -> Result<Self, ModelError> {
        if max_tokens == 0 || hidden == 0 || vocab == 0 {
            return Err(ModelError::InvalidConfig(
                "forward scratch dimensions must be non-zero".to_string(),
            ));
        }
        let device = system_device()?;
        let options = metal::MTLResourceOptions::StorageModeShared;
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
            transformed: u16_elements(max_tokens, inter)?,
            query_positions,
            token_ids,
            last_hidden: u16_elements(1, hidden)?,
            logits: u16_elements(1, vocab)?,
            capped: u16_elements(1, vocab)?,
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
        ]
        .iter()
        .map(|buffer| buffer.length())
        .sum()
    }
}

pub struct TextModel {
    embedding: Vec<u16>,
    embedding_buffer: Option<Buffer>,
    vocab_size: usize,
    hidden_size: usize,
    layers: Vec<DecoderLayer>,
    final_norm: Vec<u16>,
    lm_head: QuantizedLinear,
    final_softcap: f32,
    _checkpoint_weights: Option<TextWeights>,
    elements: ElementKernels,
    linear: LinearKernels,
    /// Queue used for the single command buffer that carries a whole forward.
    forward_queue: metal::CommandQueue,
    final_norm_buffer: Buffer,
    scratch: ForwardScratch,
    scratch_tokens: usize,
}

impl TextModel {
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
        for layer in &layers {
            q_max = q_max.max(layer.config.q_heads * layer.config.head_dim);
            kv_max = kv_max.max(layer.config.kv_heads * layer.config.head_dim);
            inter = inter.max(layer.config.intermediate_size);
        }
        let scratch = ForwardScratch::new(1, hidden_size, q_max, kv_max, inter, vocab_size)?;
        Ok(Self {
            embedding,
            embedding_buffer,
            vocab_size,
            hidden_size,
            layers,
            final_norm,
            lm_head,
            final_softcap,
            _checkpoint_weights: checkpoint_weights,
            elements: ElementKernels::new()?,
            linear: LinearKernels::new()?,
            forward_queue: device.new_command_queue(),
            final_norm_buffer,
            scratch,
            scratch_tokens: 1,
        })
    }

    pub fn from_metal_parts(
        embedding_buffer: Buffer,
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
        let embedding_bytes = usize::try_from(embedding_buffer.length())
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
        let embedding_buffer = weights.tensor_buffer("model.language_model.embed_tokens.weight")?;
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
            if std::env::var_os("KAPLUMBA_LOAD_TRACE").is_some() {
                eprintln!("layer {index} building...");
            }
            layers.push(DecoderLayer::new(layer_config, layer_weights)?);
            if std::env::var_os("KAPLUMBA_LOAD_TRACE").is_some() {
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
        let model = Self::from_metal_parts(
            embedding_buffer,
            layers,
            final_norm,
            lm_head,
            metadata.config.final_logit_softcapping,
            Some(weights),
        )?;
        model.validate_current_graph()?;
        Ok(model)
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
    pub fn forward_buffer(
        &mut self,
        token_ids: &[u32],
        positions: &[u32],
    ) -> Result<Vec<u16>, ModelError> {
        if token_ids.len() != positions.len() || token_ids.is_empty() {
            return Err(ModelError::InvalidInput(
                "model token and position counts must match and be non-empty".to_string(),
            ));
        }
        let tokens = token_ids.len();
        if positions[0] as usize != self.gpu_next_position() {
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

        // Shared-memory staging for the two small host-side arrays.
        unsafe {
            let position_ptr = self.scratch.query_positions.contents() as *mut u32;
            std::ptr::copy_nonoverlapping(positions.as_ptr(), position_ptr, tokens);
            let token_ptr = self.scratch.token_ids.contents() as *mut u32;
            std::ptr::copy_nonoverlapping(token_ids.as_ptr(), token_ptr, tokens);
        }

        let mut encoder = MetalEncoder::new(&self.forward_queue)?;
        let hidden = self.hidden_size;
        let vocab = self.vocab_size;
        let scale = (hidden as f32).sqrt();

        // Layer 0 consumes the embedding, then layers ping-pong between the two
        // hidden buffers. Input/output never alias within a layer.
        let embedding_buffer = self
            .embedding_buffer
            .as_ref()
            .ok_or_else(|| ModelError::InvalidConfig("model has no GPU embedding".to_string()))?;
        self.elements.encode_embedding_scaled(
            &mut encoder,
            &self.scratch.token_ids,
            embedding_buffer,
            &self.scratch.hidden_a,
            tokens,
            hidden,
            vocab,
            scale,
        )?;

        let layers = self.layers.len();
        for index in 0..layers {
            let (input, output) = if index % 2 == 0 {
                (&self.scratch.hidden_a, &self.scratch.hidden_b)
            } else {
                (&self.scratch.hidden_b, &self.scratch.hidden_a)
            };
            self.layers[index].forward_buffer(
                input,
                output,
                tokens,
                &self.scratch,
                &mut encoder,
            )?;
        }
        // Layer k writes into hidden_b when k is even and hidden_a when k is odd,
        // so the final hidden state lives in hidden_a for an even layer count.
        let final_hidden = if layers.is_multiple_of(2) {
            &self.scratch.hidden_a
        } else {
            &self.scratch.hidden_b
        };

        // Only the sampled position needs logits, so lift the final row out and
        // run the 262,144-wide vocabulary head for that row alone.
        self.elements.encode_copy_last_row(
            &mut encoder,
            final_hidden,
            &self.scratch.last_hidden,
            tokens,
            hidden,
        )?;
        self.encode_lm_head(&mut encoder)?;
        encoder.commit_and_wait()?;

        let logits = self.scratch.capped.contents() as *const u16;
        Ok(unsafe { std::slice::from_raw_parts(logits, vocab) }.to_vec())
    }

    fn encode_lm_head(&self, encoder: &mut MetalEncoder) -> Result<(), ModelError> {
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
            },
        )?;
        self.elements.encode_softcap(
            encoder,
            &self.scratch.logits,
            &self.scratch.capped,
            vocab,
            self.final_softcap,
        )?;
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
        for layer in &self.layers {
            q_max = q_max.max(layer.config.q_heads * layer.config.head_dim);
            kv_max = kv_max.max(layer.config.kv_heads * layer.config.head_dim);
            inter = inter.max(layer.config.intermediate_size);
        }
        self.scratch = ForwardScratch::new(
            tokens,
            self.hidden_size,
            q_max,
            kv_max,
            inter,
            self.vocab_size,
        )?;
        self.scratch_tokens = tokens;
        Ok(())
    }

    /// Bytes this engine has deliberately allocated and keeps resident.
    ///
    /// This is the figure the 18 GB budget is about: checkpoint weights, KV
    /// caches, activation scratch, and embedding. It is independent of what the
    /// kernel's `phys_footprint` happens to charge at any instant.
    pub fn allocated_bytes(&self) -> u64 {
        let weights = self._checkpoint_weights.as_ref().map_or(0, TextWeights::resident_bytes);
        let kv: usize = self.layers.iter().map(|layer| layer.gpu_cache.resident_bytes()).sum();
        weights + kv as u64 + self.scratch.resident_bytes()
    }

    /// Per-component breakdown of [`Self::allocated_bytes`].
    pub fn allocation_breakdown(&self) -> (u64, u64, u64) {
        let weights = self._checkpoint_weights.as_ref().map_or(0, TextWeights::resident_bytes);
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
            buffer.contents() as *const u16
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
    pub fn forward_layer(
        &mut self,
        index: usize,
        hidden: &[u16],
        positions: &[u32],
    ) -> Vec<u16> {
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
            .ok_or_else(|| {
                ModelError::InvalidWeights(format!("{trellis_name} is not resident"))
            })?
    };
    let packed_len = weights
        .tensors()
        .get(&trellis_name)
        .map(|tensor| (tensor.source_len / 2) as usize);
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
