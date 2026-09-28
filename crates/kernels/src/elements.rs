use std::ffi::c_void;
use std::mem::{size_of, size_of_val};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library, MTLResourceOptions,
};

pub use crate::exl3::KernelError;

const SOURCE: &str = include_str!("elements.metal");

#[repr(C)]
#[derive(Clone, Copy)]
struct EmbeddingParams {
    tokens: u32,
    hidden: u32,
    vocab: u32,
    scale: f32,
    /// Byte offset of the embedding rows inside the bound buffer. Non-zero when
    /// the table is offloaded, since the view is a page-aligned window over the
    /// shard mapping and the rows do not begin at its first byte.
    offset: u32,
    /// Table is BF16 in the checkpoint and must be widened in the kernel.
    bf16: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NormParams {
    rows: u32,
    width: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RopeParams {
    tokens: u32,
    heads: u32,
    head_dim: u32,
    offset: u32,
    theta: f32,
    partial_factor: f32,
    proportional: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ScalarParams {
    count: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SoftcapParams {
    count: u32,
    softcap: f32,
}

#[derive(Clone, Copy)]
pub struct RopeRequest<'a> {
    pub input: &'a [u16],
    pub tokens: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub offset: usize,
    pub theta: f32,
    pub partial_factor: f32,
    pub proportional: bool,
}

/// Shape/config for a GPU-resident RoPE dispatch (input/output supplied as buffers).
#[repr(C)]
#[derive(Clone, Copy)]
struct CacheAppendParams {
    tokens: u32,
    stream_len: u32,
    capacity: u32,
    slot_start: u32,
    position_start: u32,
}

#[derive(Clone, Copy)]
pub struct RopeBufferRequest {
    pub tokens: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub offset: usize,
    pub theta: f32,
    pub partial_factor: f32,
    pub proportional: bool,
}

pub struct ElementKernels {
    device: Device,
    queue: CommandQueue,
    _library: Library,
    embedding_pipeline: ComputePipelineState,
    rms_norm_weighted_pipeline: ComputePipelineState,
    rms_norm_unweighted_pipeline: ComputePipelineState,
    /// Threadgroup-per-row variants, used when `rows` is small. See the kernels
    /// for why the per-row variants collapse at `rows == 1`.
    rms_norm_weighted_tg_pipeline: ComputePipelineState,
    rms_norm_unweighted_tg_pipeline: ComputePipelineState,
    /// Greedy pick, so the decode loop can stay on the GPU between steps.
    greedy_pick_pipeline: ComputePipelineState,
    greedy_token_pipeline: ComputePipelineState,
    residual_pipeline: ComputePipelineState,
    layer_scale_pipeline: ComputePipelineState,
    gelu_pipeline: ComputePipelineState,
    gated_gelu_pipeline: ComputePipelineState,
    rope_pipeline: ComputePipelineState,
    softcap_pipeline: ComputePipelineState,
    cache_append_pipeline: ComputePipelineState,
    embedding_buffer_pipeline: ComputePipelineState,
    copy_last_row_pipeline: ComputePipelineState,
    copy_buffer_pipeline: ComputePipelineState,
}

impl ElementKernels {
    /// Row count below which the threadgroup-per-row norm is used.
    ///
    /// The per-row kernel gives one thread to each row, so the entire machine
    /// sees only `rows` threads. That saturates the GPU somewhere in the
    /// thousands of rows, not at 32. Measured at width 5376
    /// (`examples/norm_sweep.rs`, 256 rows):
    ///
    /// | kernel | time |
    /// |---|---:|
    /// | per-row (256 threads) | 0.7595 ms |
    /// | threadgroup-per-row, G=256 | 0.0253 ms |
    ///
    /// — 30x, because 256 threads is one or two threadgroups on a 20-core GPU
    /// while threadgroup-per-row launches 256 x 256. The old threshold of 32
    /// therefore sent the *prefill* path (PREFILL_CHUNK = 256 rows) down the
    /// 256-thread kernel, which is the worst of both. Kept as a constant so
    /// `KAPLUMBA_FORCE_PER_ROW_NORMS` can still A/B the two.
    const THREADGROUP_NORM_MAX_ROWS: usize = 8192;
    /// Escape hatch: force the original per-row norms regardless of row count.
    pub fn force_per_row_norms() -> bool {
        std::env::var_os("KAPLUMBA_FORCE_PER_ROW_NORMS").is_some()
    }
    /// Threads per group for the threadgroup-per-row norms.
    const NORM_GROUP_THREADS: u64 = 256;

    pub fn new() -> Result<Self, KernelError> {
        let device = Device::system_default().ok_or(KernelError::DeviceUnavailable)?;
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(SOURCE, &options)
            .map_err(KernelError::Shader)?;
        let embedding_pipeline = pipeline(&device, &library, "embedding_scaled")?;
        let rms_norm_weighted_pipeline = pipeline(&device, &library, "rms_norm_weighted")?;
        let rms_norm_unweighted_pipeline = pipeline(&device, &library, "rms_norm_unweighted")?;
        let rms_norm_weighted_tg_pipeline =
            pipeline(&device, &library, "rms_norm_weighted_tg")?;
        let rms_norm_unweighted_tg_pipeline =
            pipeline(&device, &library, "rms_norm_unweighted_tg")?;
        let greedy_pick_pipeline = pipeline(&device, &library, "greedy_pick")?;
        let greedy_token_pipeline = pipeline(&device, &library, "greedy_token")?;
        let residual_pipeline = pipeline(&device, &library, "residual_add")?;
        let layer_scale_pipeline = pipeline(&device, &library, "layer_scale")?;
        let gelu_pipeline = pipeline(&device, &library, "gelu_tanh")?;
        let gated_gelu_pipeline = pipeline(&device, &library, "gated_gelu")?;
        let rope_pipeline = pipeline(&device, &library, "rope")?;
        let softcap_pipeline = pipeline(&device, &library, "softcap")?;
        let cache_append_pipeline = pipeline(&device, &library, "cache_append")?;
        let embedding_buffer_pipeline = pipeline(&device, &library, "embedding_scaled_buffer")?;
        let copy_last_row_pipeline = pipeline(&device, &library, "copy_last_row")?;
        let copy_buffer_pipeline = pipeline(&device, &library, "copy_buffer")?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            _library: library,
            embedding_pipeline,
            rms_norm_weighted_pipeline,
            rms_norm_unweighted_pipeline,
            rms_norm_weighted_tg_pipeline,
            rms_norm_unweighted_tg_pipeline,
            greedy_pick_pipeline,
            greedy_token_pipeline,
            residual_pipeline,
            layer_scale_pipeline,
            gelu_pipeline,
            gated_gelu_pipeline,
            rope_pipeline,
            softcap_pipeline,
            cache_append_pipeline,
            embedding_buffer_pipeline,
            copy_last_row_pipeline,
            copy_buffer_pipeline,
        })
    }

    pub fn embedding_scaled(
        &self,
        tokens: &[u32],
        embedding: &[u16],
        hidden_size: usize,
        vocab_size: usize,
        scale: f32,
    ) -> Result<Vec<u16>, KernelError> {
        if tokens.is_empty() || hidden_size == 0 || vocab_size == 0 {
            return Err(KernelError::InvalidInput(
                "embedding dimensions and token count must be non-zero".to_string(),
            ));
        }
        if !scale.is_finite() {
            return Err(KernelError::InvalidInput(
                "embedding scale must be finite".to_string(),
            ));
        }
        let expected = checked_product(hidden_size, vocab_size, "embedding shape")?;
        if embedding.len() != expected {
            return Err(KernelError::InvalidInput(format!(
                "embedding length {} does not match {expected}",
                embedding.len()
            )));
        }
        if tokens.iter().any(|token| *token as usize >= vocab_size) {
            return Err(KernelError::InvalidInput(
                "embedding token id is outside the vocabulary".to_string(),
            ));
        }
        let output_elements = checked_product(tokens.len(), hidden_size, "embedding output")?;
        let token_buffer = self.make_u32_buffer(tokens)?;
        let embedding_buffer = self.make_u16_buffer(embedding)?;
        let output = self.make_output(output_elements)?;
        let params = EmbeddingParams {
            tokens: to_u32(tokens.len(), "token count")?,
            hidden: to_u32(hidden_size, "hidden size")?,
            vocab: to_u32(vocab_size, "vocabulary size")?,
            scale,
            offset: 0,
            bf16: 0,
        };
        self.run(
            &self.embedding_pipeline,
            &[Some(&token_buffer), Some(&embedding_buffer), Some(&output)],
            params_bytes(&params),
            output_elements as u64,
        )?;
        read_u16_buffer(&output)
    }

    pub fn embedding_scaled_buffer(
        &self,
        tokens: &[u32],
        embedding: &Buffer,
        hidden_size: usize,
        vocab_size: usize,
        scale: f32,
    ) -> Result<Vec<u16>, KernelError> {
        if tokens.is_empty() || hidden_size == 0 || vocab_size == 0 || !scale.is_finite() {
            return Err(KernelError::InvalidInput(
                "embedding buffer dimensions are invalid".to_string(),
            ));
        }
        let expected = checked_product(hidden_size, vocab_size, "embedding shape")?
            .checked_mul(2)
            .ok_or_else(|| KernelError::InvalidInput("embedding shape overflows".to_string()))?;
        if embedding.length() != expected as u64 {
            return Err(KernelError::InvalidInput(
                "embedding buffer length does not match dimensions".to_string(),
            ));
        }
        if tokens.iter().any(|token| *token as usize >= vocab_size) {
            return Err(KernelError::InvalidInput(
                "embedding token id is outside the vocabulary".to_string(),
            ));
        }
        let output_elements = checked_product(tokens.len(), hidden_size, "embedding output")?;
        let token_buffer = self.make_u32_buffer(tokens)?;
        let output = self.make_output(output_elements)?;
        let params = EmbeddingParams {
            tokens: to_u32(tokens.len(), "token count")?,
            hidden: to_u32(hidden_size, "hidden size")?,
            vocab: to_u32(vocab_size, "vocabulary size")?,
            scale,
            offset: 0,
            bf16: 0,
        };
        self.run(
            &self.embedding_pipeline,
            &[Some(&token_buffer), Some(embedding), Some(&output)],
            params_bytes(&params),
            output_elements as u64,
        )?;
        read_u16_buffer(&output)
    }

    pub fn rms_norm(
        &self,
        input: &[u16],
        weight: &[u16],
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<Vec<u16>, KernelError> {
        validate_norm(input.len(), Some(weight.len()), rows, width, eps)?;
        let input_buffer = self.make_u16_buffer(input)?;
        let weight_buffer = self.make_u16_buffer(weight)?;
        let output = self.make_output(checked_product(rows, width, "norm output")?)?;
        let params = NormParams {
            rows: to_u32(rows, "row count")?,
            width: to_u32(width, "norm width")?,
            eps,
        };
        self.run(
            &self.rms_norm_weighted_pipeline,
            &[Some(&input_buffer), Some(&weight_buffer), Some(&output)],
            params_bytes(&params),
            rows as u64,
        )?;
        read_u16_buffer(&output)
    }

    /// Force the threadgroup-per-row norm, regardless of the dispatch threshold.
    ///
    /// Exists so tests can compare the two paths directly; production code goes
    /// through [`Self::encode_rms_norm`], which chooses between them.
    pub fn rms_norm_threadgroup(
        &self,
        input: &[u16],
        weight: &[u16],
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<Vec<u16>, KernelError> {
        validate_norm(input.len(), Some(weight.len()), rows, width, eps)?;
        let input_buffer = self.make_u16_buffer(input)?;
        let weight_buffer = self.make_u16_buffer(weight)?;
        let output = self.make_output(checked_product(rows, width, "norm output")?)?;
        let params = NormParams {
            rows: to_u32(rows, "row count")?,
            width: to_u32(width, "norm width")?,
            eps,
        };
        self.run_groups(
            &self.rms_norm_weighted_tg_pipeline,
            &[Some(&input_buffer), Some(&weight_buffer), Some(&output)],
            params_bytes(&params),
            rows as u64,
            Self::NORM_GROUP_THREADS,
        )?;
        read_u16_buffer(&output)
    }

    pub fn rms_norm_unweighted(
        &self,
        input: &[u16],
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<Vec<u16>, KernelError> {
        validate_norm(input.len(), None, rows, width, eps)?;
        let input_buffer = self.make_u16_buffer(input)?;
        let output = self.make_output(checked_product(rows, width, "norm output")?)?;
        let params = NormParams {
            rows: to_u32(rows, "row count")?,
            width: to_u32(width, "norm width")?,
            eps,
        };
        self.run(
            &self.rms_norm_unweighted_pipeline,
            &[Some(&input_buffer), Some(&output)],
            params_bytes(&params),
            rows as u64,
        )?;
        read_u16_buffer(&output)
    }

    pub fn residual_add(&self, left: &[u16], right: &[u16]) -> Result<Vec<u16>, KernelError> {
        if left.is_empty() || left.len() != right.len() {
            return Err(KernelError::InvalidInput(
                "residual inputs must be non-empty and equal length".to_string(),
            ));
        }
        let left_buffer = self.make_u16_buffer(left)?;
        let right_buffer = self.make_u16_buffer(right)?;
        let output = self.make_output(left.len())?;
        let count = to_u32(left.len(), "residual length")?;
        self.run(
            &self.residual_pipeline,
            &[Some(&left_buffer), Some(&right_buffer), Some(&output)],
            &count.to_le_bytes(),
            left.len() as u64,
        )?;
        read_u16_buffer(&output)
    }

    pub fn layer_scale(&self, input: &[u16], scale: f32) -> Result<Vec<u16>, KernelError> {
        if input.is_empty() || !scale.is_finite() {
            return Err(KernelError::InvalidInput(
                "layer scale input must be non-empty and scale finite".to_string(),
            ));
        }
        let input_buffer = self.make_u16_buffer(input)?;
        let output = self.make_output(input.len())?;
        let params = ScalarParams {
            count: to_u32(input.len(), "layer scale length")?,
            scale,
        };
        self.run(
            &self.layer_scale_pipeline,
            &[Some(&input_buffer), Some(&output)],
            params_bytes(&params),
            input.len() as u64,
        )?;
        read_u16_buffer(&output)
    }

    pub fn gelu_tanh(&self, input: &[u16]) -> Result<Vec<u16>, KernelError> {
        if input.is_empty() {
            return Err(KernelError::InvalidInput(
                "GELU input must be non-empty".to_string(),
            ));
        }
        let input_buffer = self.make_u16_buffer(input)?;
        let output = self.make_output(input.len())?;
        let count = to_u32(input.len(), "GELU length")?;
        self.run(
            &self.gelu_pipeline,
            &[Some(&input_buffer), Some(&output)],
            &count.to_le_bytes(),
            input.len() as u64,
        )?;
        read_u16_buffer(&output)
    }

    pub fn gated_gelu(&self, gate: &[u16], up: &[u16]) -> Result<Vec<u16>, KernelError> {
        if gate.is_empty() || gate.len() != up.len() {
            return Err(KernelError::InvalidInput(
                "gated GELU inputs must be non-empty and equal length".to_string(),
            ));
        }
        let gate_buffer = self.make_u16_buffer(gate)?;
        let up_buffer = self.make_u16_buffer(up)?;
        let output = self.make_output(gate.len())?;
        let count = to_u32(gate.len(), "gated GELU length")?;
        self.run(
            &self.gated_gelu_pipeline,
            &[Some(&gate_buffer), Some(&up_buffer), Some(&output)],
            &count.to_le_bytes(),
            gate.len() as u64,
        )?;
        read_u16_buffer(&output)
    }

    pub fn rope(&self, request: RopeRequest<'_>) -> Result<Vec<u16>, KernelError> {
        let RopeRequest {
            input,
            tokens,
            heads,
            head_dim,
            offset,
            theta,
            partial_factor,
            proportional,
        } = request;
        if tokens == 0 || heads == 0 || head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(KernelError::InvalidInput(
                "RoPE dimensions must be non-zero and head_dim must be even".to_string(),
            ));
        }
        if !theta.is_finite() || theta <= 0.0 {
            return Err(KernelError::InvalidInput(
                "RoPE theta must be finite and positive".to_string(),
            ));
        }
        if proportional
            && (!partial_factor.is_finite() || partial_factor <= 0.0 || partial_factor > 1.0)
        {
            return Err(KernelError::InvalidInput(
                "proportional RoPE factor must be in (0, 1]".to_string(),
            ));
        }
        let expected = checked_product(
            tokens,
            checked_product(heads, head_dim, "RoPE head shape")?,
            "RoPE input shape",
        )?;
        if input.len() != expected {
            return Err(KernelError::InvalidInput(format!(
                "RoPE input length {} does not match {expected}",
                input.len()
            )));
        }
        let input_buffer = self.make_u16_buffer(input)?;
        let output = self.make_output(input.len())?;
        let params = RopeParams {
            tokens: to_u32(tokens, "RoPE token count")?,
            heads: to_u32(heads, "RoPE head count")?,
            head_dim: to_u32(head_dim, "RoPE head dimension")?,
            offset: to_u32(offset, "RoPE offset")?,
            theta,
            partial_factor,
            proportional: u32::from(proportional),
        };
        self.run(
            &self.rope_pipeline,
            &[Some(&input_buffer), Some(&output)],
            params_bytes(&params),
            input.len() as u64,
        )?;
        read_u16_buffer(&output)
    }

    pub fn softcap(&self, input: &[u16], softcap: f32) -> Result<Vec<u16>, KernelError> {
        if input.is_empty() || !softcap.is_finite() || softcap <= 0.0 {
            return Err(KernelError::InvalidInput(
                "softcap input must be non-empty and cap finite and positive".to_string(),
            ));
        }
        let input_buffer = self.make_u16_buffer(input)?;
        let output = self.make_output(input.len())?;
        let params = SoftcapParams {
            count: to_u32(input.len(), "softcap length")?,
            softcap,
        };
        self.run(
            &self.softcap_pipeline,
            &[Some(&input_buffer), Some(&output)],
            params_bytes(&params),
            input.len() as u64,
        )?;
        read_u16_buffer(&output)
    }

    // ---- GPU-resident encode-only variants (T14b) ----
    // These encode a dispatch into a caller-owned `MetalEncoder` without
    // committing or reading back, so a whole decoder layer (or forward) can be
    // encoded into a single command buffer and sync once. The caller owns and
    // reuses the input/output buffers (ping-pong scratch) and is responsible for
    // buffer sizing; the scalar parameters are validated here for parity with
    // the slice-based paths.

    #[allow(clippy::too_many_arguments)]
    pub fn encode_rms_norm(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        input: &Buffer,
        weight: &Buffer,
        output: &Buffer,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<(), KernelError> {
        if rows == 0 || width == 0 || !eps.is_finite() || eps <= 0.0 {
            return Err(KernelError::InvalidInput(
                "RMSNorm dimensions must be non-zero and eps finite and positive".to_string(),
            ));
        }
        let params = NormParams {
            rows: to_u32(rows, "row count")?,
            width: to_u32(width, "norm width")?,
            eps,
        };
        // One thread per row only makes sense when `rows` alone saturates the
        // GPU. It does not: at prefill's 256 rows the whole machine gets 256
        // threads, and the threadgroup-per-row kernel is ~30x faster (see
        // THREADGROUP_NORM_MAX_ROWS and examples/norm_sweep.rs). This is
        // numerically identical work — same sum of squares, same rsqrt — only
        // the reduction tree differs.
        if rows < Self::THREADGROUP_NORM_MAX_ROWS && !Self::force_per_row_norms() {
            return encoder.dispatch_groups(
                &self.rms_norm_weighted_tg_pipeline,
                &[Some(input), Some(weight), Some(output)],
                params_bytes(&params),
                rows as u64,
                Self::NORM_GROUP_THREADS,
            );
        }
        encoder.dispatch_1d(
            &self.rms_norm_weighted_pipeline,
            &[Some(input), Some(weight), Some(output)],
            params_bytes(&params),
            rows as u64,
            64,
        )
    }

    /// Encode a greedy pick: reduce the capped logits to the argmax token and
    /// write it into `out_tokens[row]`.
    ///
    /// The reduction is one unsigned max over packed (value, index) keys, so it
    /// needs no second pass and no comparison branching.
    pub fn encode_greedy_pick(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        logits: &Buffer,
        best: &Buffer,
        out_tokens: &Buffer,
        row: usize,
    ) -> Result<(), KernelError> {
        if row == 0 {
            return Err(KernelError::InvalidInput(
                "greedy pick row must be non-zero".to_string(),
            ));
        }
        let vocab = to_u32((logits.length() / 2) as usize, "logit count")?;
        // One simdgroup per 32 logits, capped so a 262k vocabulary does not ask
        // for 8,192 groups; each thread strides from there.
        let groups = ((vocab / 32) as u64).min(4096);
        encoder.dispatch_groups(
            &self.greedy_pick_pipeline,
            &[Some(logits), Some(best)],
            &vocab.to_le_bytes(),
            groups,
            32,
        )?;
        encoder.dispatch_1d(
            &self.greedy_token_pipeline,
            &[Some(best), Some(out_tokens)],
            &row.to_le_bytes(),
            1,
            1,
        )
    }

    pub fn encode_rms_norm_unweighted(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        input: &Buffer,
        output: &Buffer,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<(), KernelError> {
        if rows == 0 || width == 0 || !eps.is_finite() || eps <= 0.0 {
            return Err(KernelError::InvalidInput(
                "RMSNorm dimensions must be non-zero and eps finite and positive".to_string(),
            ));
        }
        let params = NormParams {
            rows: to_u32(rows, "row count")?,
            width: to_u32(width, "norm width")?,
            eps,
        };
        if rows < Self::THREADGROUP_NORM_MAX_ROWS && !Self::force_per_row_norms() {
            return encoder.dispatch_groups(
                &self.rms_norm_unweighted_tg_pipeline,
                &[Some(input), Some(output)],
                params_bytes(&params),
                rows as u64,
                Self::NORM_GROUP_THREADS,
            );
        }
        encoder.dispatch_1d(
            &self.rms_norm_unweighted_pipeline,
            &[Some(input), Some(output)],
            params_bytes(&params),
            rows as u64,
            64,
        )
    }

    pub fn encode_residual_add(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        left: &Buffer,
        right: &Buffer,
        output: &Buffer,
        count: usize,
    ) -> Result<(), KernelError> {
        if count == 0 {
            return Err(KernelError::InvalidInput(
                "residual count must be non-zero".to_string(),
            ));
        }
        let params = to_u32(count, "residual length")?;
        encoder.dispatch_1d(
            &self.residual_pipeline,
            &[Some(left), Some(right), Some(output)],
            &params.to_le_bytes(),
            count as u64,
            64,
        )
    }

    pub fn encode_layer_scale(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        input: &Buffer,
        output: &Buffer,
        count: usize,
        scale: f32,
    ) -> Result<(), KernelError> {
        if count == 0 || !scale.is_finite() {
            return Err(KernelError::InvalidInput(
                "layer scale count must be non-zero and scale finite".to_string(),
            ));
        }
        let params = ScalarParams {
            count: to_u32(count, "layer scale length")?,
            scale,
        };
        encoder.dispatch_1d(
            &self.layer_scale_pipeline,
            &[Some(input), Some(output)],
            params_bytes(&params),
            count as u64,
            64,
        )
    }

    pub fn encode_gated_gelu(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        gate: &Buffer,
        up: &Buffer,
        output: &Buffer,
        count: usize,
    ) -> Result<(), KernelError> {
        if count == 0 {
            return Err(KernelError::InvalidInput(
                "gated GELU count must be non-zero".to_string(),
            ));
        }
        let params = to_u32(count, "gated GELU length")?;
        encoder.dispatch_1d(
            &self.gated_gelu_pipeline,
            &[Some(gate), Some(up), Some(output)],
            &params.to_le_bytes(),
            count as u64,
            64,
        )
    }

    pub fn encode_rope(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        input: &Buffer,
        output: &Buffer,
        request: RopeBufferRequest,
    ) -> Result<(), KernelError> {
        let RopeBufferRequest {
            tokens,
            heads,
            head_dim,
            offset,
            theta,
            partial_factor,
            proportional,
        } = request;
        if tokens == 0 || heads == 0 || head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(KernelError::InvalidInput(
                "RoPE dimensions must be non-zero and head_dim must be even".to_string(),
            ));
        }
        if !theta.is_finite() || theta <= 0.0 {
            return Err(KernelError::InvalidInput(
                "RoPE theta must be finite and positive".to_string(),
            ));
        }
        if proportional
            && (!partial_factor.is_finite() || partial_factor <= 0.0 || partial_factor > 1.0)
        {
            return Err(KernelError::InvalidInput(
                "proportional RoPE factor must be in (0, 1]".to_string(),
            ));
        }
        let params = RopeParams {
            tokens: to_u32(tokens, "RoPE token count")?,
            heads: to_u32(heads, "RoPE head count")?,
            head_dim: to_u32(head_dim, "RoPE head dimension")?,
            offset: to_u32(offset, "RoPE offset")?,
            theta,
            partial_factor,
            proportional: u32::from(proportional),
        };
        let threads = checked_product(
            tokens,
            checked_product(heads, head_dim, "RoPE head shape")?,
            "RoPE element count",
        )?;
        encoder.dispatch_1d(
            &self.rope_pipeline,
            &[Some(input), Some(output)],
            params_bytes(&params),
            threads as u64,
            64,
        )
    }

    pub fn encode_softcap(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        input: &Buffer,
        output: &Buffer,
        count: usize,
        softcap: f32,
    ) -> Result<(), KernelError> {
        if count == 0 || !softcap.is_finite() || softcap <= 0.0 {
            return Err(KernelError::InvalidInput(
                "softcap count must be non-zero and cap finite and positive".to_string(),
            ));
        }
        let params = SoftcapParams {
            count: to_u32(count, "softcap length")?,
            softcap,
        };
        encoder.dispatch_1d(
            &self.softcap_pipeline,
            &[Some(input), Some(output)],
            params_bytes(&params),
            count as u64,
            64,
        )
    }

    /// Encode a scatter of one chunk of keys/values into a GPU-resident cache.
    ///
    /// The cache slots are written in place on the device so a whole decoder
    /// layer, and in turn a whole forward, can run inside a single command
    /// buffer with no host round trip.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn encode_cache_append(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        keys: &Buffer,
        values: &Buffer,
        key_cache: &Buffer,
        value_cache: &Buffer,
        position_cache: &Buffer,
        tokens: usize,
        stream_len: usize,
        capacity: usize,
        slot_start: usize,
        position_start: usize,
    ) -> Result<(), KernelError> {
        if tokens == 0 || stream_len == 0 || capacity == 0 {
            return Err(KernelError::InvalidInput(
                "cache append dimensions must be non-zero".to_string(),
            ));
        }
        let params = CacheAppendParams {
            tokens: to_u32(tokens, "cache token count")?,
            stream_len: to_u32(stream_len, "cache stream length")?,
            capacity: to_u32(capacity, "cache capacity")?,
            slot_start: to_u32(slot_start % capacity, "cache slot start")?,
            position_start: to_u32(position_start, "cache position start")?,
        };
        let threads = checked_product(tokens, stream_len, "cache append element count")?;
        encoder.dispatch_1d(
            &self.cache_append_pipeline,
            &[
                Some(keys),
                Some(values),
                Some(key_cache),
                Some(value_cache),
                Some(position_cache),
            ],
            params_bytes(&params),
            threads as u64,
            64,
        )
    }

    /// Encode a scaled embedding lookup entirely on device.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_embedding_scaled(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        token_ids: &Buffer,
        embedding: &Buffer,
        output: &Buffer,
        tokens: usize,
        hidden: usize,
        vocab: usize,
        scale: f32,
        embedding_offset: u64,
        embedding_is_bf16: bool,
    ) -> Result<(), KernelError> {
        if tokens == 0 || hidden == 0 || vocab == 0 || !scale.is_finite() {
            return Err(KernelError::InvalidInput(
                "embedding dimensions must be non-zero and scale finite".to_string(),
            ));
        }
        let params = EmbeddingParams {
            tokens: to_u32(tokens, "embedding token count")?,
            hidden: to_u32(hidden, "embedding hidden size")?,
            vocab: to_u32(vocab, "embedding vocabulary")?,
            scale,
            offset: to_u32(
                usize::try_from(embedding_offset).unwrap_or(u32::MAX as usize),
                "embedding offset",
            )?,
            bf16: u32::from(embedding_is_bf16),
        };
        let threads = checked_product(tokens, hidden, "embedding element count")?;
        encoder.dispatch_1d(
            &self.embedding_buffer_pipeline,
            &[Some(token_ids), Some(embedding), Some(output)],
            params_bytes(&params),
            threads as u64,
            64,
        )
    }

    /// Encode a copy of the final row of a [rows, width] activation.
    pub fn encode_copy_last_row(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        input: &Buffer,
        output: &Buffer,
        rows: usize,
        width: usize,
    ) -> Result<(), KernelError> {
        if rows == 0 || width == 0 || width > u32::MAX as usize || rows > u32::MAX as usize {
            return Err(KernelError::InvalidInput(
                "copy_last_row dimensions must be non-zero and addressable".to_string(),
            ));
        }
        let shape = ((rows - 1) as u32) | ((width as u32) << 16);
        encoder.dispatch_1d(
            &self.copy_last_row_pipeline,
            &[Some(input), Some(output)],
            &shape.to_le_bytes(),
            width as u64,
            64,
        )
    }

    /// Encode a device-to-device copy of `count` FP16 elements.
    pub fn encode_copy(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        input: &Buffer,
        output: &Buffer,
        count: usize,
    ) -> Result<(), KernelError> {
        if count == 0 {
            return Err(KernelError::InvalidInput(
                "copy count must be non-zero".to_string(),
            ));
        }
        encoder.dispatch_1d(
            &self.copy_buffer_pipeline,
            &[Some(input), Some(output)],
            &to_u32(count, "copy count")?.to_le_bytes(),
            count as u64,
            64,
        )
    }

    fn make_output(&self, elements: usize) -> Result<Buffer, KernelError> {
        if elements == 0 {
            return Err(KernelError::InvalidInput(
                "output cannot be empty".to_string(),
            ));
        }
        Ok(self.device.new_buffer(
            (elements * size_of::<u16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        ))
    }

    fn make_u16_buffer(&self, values: &[u16]) -> Result<Buffer, KernelError> {
        if values.is_empty() {
            return Err(KernelError::InvalidInput(
                "u16 input cannot be empty".to_string(),
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
                "u32 input cannot be empty".to_string(),
            ));
        }
        Ok(self.device.new_buffer_with_data(
            values.as_ptr() as *const c_void,
            size_of_val(values) as u64,
            MTLResourceOptions::StorageModeShared,
        ))
    }

    fn run(
        &self,
        pipeline: &ComputePipelineState,
        bindings: &[Option<&Buffer>],
        params: &[u8],
        threads: u64,
    ) -> Result<(), KernelError> {
        if threads == 0 {
            return Err(KernelError::InvalidInput(
                "kernel thread count must be non-zero".to_string(),
            ));
        }
        let mut encoder = crate::encoder::MetalEncoder::new(&self.queue)?;
        encoder.dispatch_1d(pipeline, bindings, params, threads, 64)?;
        encoder.commit_and_wait()
    }

    /// Like [`Self::run`] but dispatching threadgroups rather than a 1D grid.
    fn run_groups(
        &self,
        pipeline: &ComputePipelineState,
        bindings: &[Option<&Buffer>],
        params: &[u8],
        groups: u64,
        threads_per_group: u64,
    ) -> Result<(), KernelError> {
        if groups == 0 {
            return Err(KernelError::InvalidInput(
                "kernel group count must be non-zero".to_string(),
            ));
        }
        let mut encoder = crate::encoder::MetalEncoder::new(&self.queue)?;
        encoder.dispatch_groups(pipeline, bindings, params, groups, threads_per_group)?;
        encoder.commit_and_wait()
    }
}

fn pipeline(
    device: &Device,
    library: &Library,
    name: &str,
) -> Result<ComputePipelineState, KernelError> {
    let function = library
        .get_function(name, None)
        .map_err(KernelError::Shader)?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(KernelError::Pipeline)
}

fn validate_norm(
    input_len: usize,
    weight_len: Option<usize>,
    rows: usize,
    width: usize,
    eps: f32,
) -> Result<(), KernelError> {
    if rows == 0 || width == 0 || !eps.is_finite() || eps <= 0.0 {
        return Err(KernelError::InvalidInput(
            "RMSNorm dimensions must be non-zero and eps finite and positive".to_string(),
        ));
    }
    if input_len != checked_product(rows, width, "RMSNorm input")?
        || weight_len.is_some_and(|length| length != width)
    {
        return Err(KernelError::InvalidInput(
            "RMSNorm input or weight length does not match dimensions".to_string(),
        ));
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
