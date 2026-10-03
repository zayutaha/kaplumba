use std::ffi::c_void;
use std::mem::{size_of, size_of_val};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library, MTLResourceOptions,
};

pub use crate::exl3::KernelError;

/// Copy a [`KernelError`] that lives behind a `&'static` reference.
///
/// `KernelError` is not `Clone` (it holds a `String`), so the shared-instance
/// cache below cannot store a failure and hand it back by reference. This keeps
/// the variant and its message.
pub(crate) fn clone_kernel_error(error: &KernelError) -> KernelError {
    match error {
        KernelError::DeviceUnavailable => KernelError::DeviceUnavailable,
        KernelError::Shader(message) => KernelError::Shader(message.clone()),
        KernelError::Pipeline(message) => KernelError::Pipeline(message.clone()),
        KernelError::InvalidInput(message) => KernelError::InvalidInput(message.clone()),
        KernelError::CommandFailed => KernelError::CommandFailed,
    }
}
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
struct NormScaleParams {
    rows: u32,
    width: u32,
    eps: f32,
    scale: f32,
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

/// Params for `norm_rope_qkv`, which does Q/K weighted-norm + RoPE and V
/// unweighted-norm in a single dispatch.
///
/// Field order and types mirror the Metal struct exactly; it is a
/// `constant`-address-space `NormRopeQkvParams` and the two must agree byte for
/// byte. There is no token field: this is the single-token decode form, and the
/// one caller that could pass more (prefill) does not.
#[repr(C)]
#[derive(Clone, Copy)]
struct NormRopeQkvParams {
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    offset: u32,
    eps: f32,
    theta: f32,
    partial_factor: f32,
    proportional: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NormRopeQkvCacheParams {
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    offset: u32,
    eps: f32,
    theta: f32,
    partial_factor: f32,
    proportional: u32,
    slot: u32,
    stream_len: u32,
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

/// Threads the `rope` kernel needs: one per (token, head, rotation pair) over
/// the rotated span, plus one per (token, head, column) over the non-rotated
/// tail.
///
/// The kernel is one thread per PAIR, not per column. Both call sites must use
/// this; when `rope()` dispatched `input.len()` and `encode_rope` dispatched
/// something else, they disagreed and the proportional path failed.
pub fn rope_threads(
    tokens: usize,
    heads: usize,
    head_dim: usize,
    partial_factor: f32,
    proportional: bool,
) -> Result<usize, KernelError> {
    let half_width = head_dim / 2;
    let rotated_pairs = if proportional {
        ((partial_factor * head_dim as f32 / 2.0) as usize).min(half_width)
    } else {
        half_width
    };
    let rotated = checked_product(
        tokens,
        checked_product(heads, rotated_pairs, "RoPE rotated pair count")?,
        "RoPE rotated thread count",
    )?;
    let tail = if rotated_pairs < half_width {
        checked_product(
            tokens,
            checked_product(heads, half_width - rotated_pairs, "RoPE tail column count")?,
            "RoPE tail thread count",
        )?
    } else {
        0
    };
    rotated
        .checked_add(tail)
        .ok_or_else(|| KernelError::InvalidInput("RoPE thread count overflowed".to_string()))
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
    rms_norm_residual_tg_pipeline: ComputePipelineState,
    rms_norm_residual_layer_scale_tg_pipeline: ComputePipelineState,
    /// Greedy pick, so the decode loop can stay on the GPU between steps.
    greedy_pick_pipeline: ComputePipelineState,
    greedy_token_pipeline: ComputePipelineState,
    residual_pipeline: ComputePipelineState,
    layer_scale_pipeline: ComputePipelineState,
    gelu_pipeline: ComputePipelineState,
    gated_gelu_pipeline: ComputePipelineState,
    rope_pipeline: ComputePipelineState,
    /// Q/K norm+RoPE and V norm in one dispatch, replacing the five-dispatch
    /// sequence. Decode only: it assumes one token, so `offset` is the position.
    norm_rope_qkv_pipeline: ComputePipelineState,
    norm_rope_qkv_cache_pipeline: ComputePipelineState,
    softcap_pipeline: ComputePipelineState,
    cache_append_pipeline: ComputePipelineState,
    embedding_buffer_pipeline: ComputePipelineState,
    copy_last_row_pipeline: ComputePipelineState,
    copy_buffer_pipeline: ComputePipelineState,
    /// Device-side argmax pipelines with 16-bit slot partitioning.
    argmax_clear_pipeline: ComputePipelineState,
    argmax_slots_pipeline: ComputePipelineState,
    argmax_keys_to_token_pipeline: ComputePipelineState,
    /// Device-side sampler. Two dispatches per step; see `elements.metal` for
    /// the sizing and for the `top_k` ceiling the buffers impose.
    ///
    /// Built unconditionally, so the greedy path pays two extra shader compiles
    /// at load. Measured on `Gxma/Gxma`: the whole load is ~10 s, and shader
    /// compilation is cached by the driver between runs, so this is not
    /// measurable in `load_ms`; `KAPLUMBA_PIPELINE_AUDIT=1` is the way to check
    /// that none of it lands inside a decode step.
    sample_topk_pipeline: ComputePipelineState,
    sample_draw_pipeline: ComputePipelineState,
    /// MEASUREMENT AID. See [`Self::encode_spin_burn`].
    spin_burn_pipeline: ComputePipelineState,
}

impl ElementKernels {
    /// The process-wide instance, built on first use.
    ///
    /// Every `DecoderLayer` used to call [`Self::new`], so a 60-layer model ran
    /// 61 `new_library_with_source` calls on the same `include_str!`'d source and
    /// rebuilt the same pipeline states 61 times over. That is a large one-off
    /// load cost, and it also made every layer's pipeline object a *distinct*
    /// `MTLComputePipelineState` for identical code, which blocks any future
    /// binding deduplication. The pipelines are immutable after construction (no
    /// `&mut self` method exists on this type), so one shared instance is safe.
    pub fn shared() -> Result<&'static Self, KernelError> {
        static INSTANCE: std::sync::OnceLock<Result<ElementKernels, KernelError>> =
            std::sync::OnceLock::new();
        match INSTANCE.get_or_init(Self::new) {
            Ok(kernels) => Ok(kernels),
            // A `&'static Self` cannot be produced from an `Err`, and
            // `KernelError` is not `Clone`, so re-derive it.
            Err(error) => Err(clone_kernel_error(&error)),
        }
    }

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
    /// therefore sent the *prefill* path (PREFILL_CHUNK = 1024 rows) down the
    /// 256-thread kernel, which is the worst of both. Kept as a constant so
    /// `KAPLUMBA_FORCE_PER_ROW_NORMS` can still A/B the two.
    const THREADGROUP_NORM_MAX_ROWS: usize = 8192;
    /// Escape hatch: force the original per-row norms regardless of row count.
    ///
    /// Cached: this used to be a live `std::env::var_os`, and every norm dispatch
    /// consults it -- five per decoder layer, so ~300 environment-lock
    /// acquisitions and `environ` scans per generated token. A switch is set
    /// before the process starts or not at all, so once is behaviour-preserving.
    pub fn force_per_row_norms() -> bool {
        static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *FLAG.get_or_init(|| std::env::var_os("KAPLUMBA_FORCE_PER_ROW_NORMS").is_some())
    }
    /// Threads per group for the threadgroup-per-row norms.
    const NORM_GROUP_THREADS: u64 = 256;
    /// Threads per group for both sampler dispatches.
    ///
    /// 256, not the 1024 `elements.metal`'s design note suggested. The sampler's
    /// work is a fixed number of compare-exchange pairs and a fixed number of
    /// barriers either way -- more threads means fewer pairs per thread and
    /// *identical* barrier count, so the only thing 1024 buys is occupancy that
    /// 4096 shared bytes-per-pair makes impossible anyway (32 KiB of shared
    /// memory admits one 1024-thread group per core, 256 admits four).
    const SAMPLE_GROUP_THREADS: u64 = 256;
    /// Largest `head_dim` `norm_rope_qkv` accepts.
    ///
    /// Bounded by the kernel's `threadgroup float normed_shared[512]`, which has
    /// to hold a whole normalised head because the rotation reads the partner
    /// column that a different lane wrote. 512 is the model's full-attention head
    /// dim, so nothing in this model is excluded by the check; the bound exists so
    /// a larger head fails loudly at the call instead of scribbling past the
    /// threadgroup array.
    const NORM_STAGING_MAX: usize = 512;

    pub fn new() -> Result<Self, KernelError> {
        let device = Device::system_default().ok_or(KernelError::DeviceUnavailable)?;
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(SOURCE, &options)
            .map_err(KernelError::Shader)?;
        let embedding_pipeline = pipeline(&device, &library, "embedding_scaled")?;
        let rms_norm_weighted_pipeline = pipeline(&device, &library, "rms_norm_weighted")?;
        let rms_norm_unweighted_pipeline = pipeline(&device, &library, "rms_norm_unweighted")?;
        let rms_norm_weighted_tg_pipeline = pipeline(&device, &library, "rms_norm_weighted_tg")?;
        let rms_norm_unweighted_tg_pipeline =
            pipeline(&device, &library, "rms_norm_unweighted_tg")?;
        let rms_norm_residual_tg_pipeline = pipeline(&device, &library, "rms_norm_residual_tg")?;
        let rms_norm_residual_layer_scale_tg_pipeline =
            pipeline(&device, &library, "rms_norm_residual_layer_scale_tg")?;
        let greedy_pick_pipeline = pipeline(&device, &library, "greedy_pick")?;
        let greedy_token_pipeline = pipeline(&device, &library, "greedy_token")?;
        let residual_pipeline = pipeline(&device, &library, "residual_add")?;
        let layer_scale_pipeline = pipeline(&device, &library, "layer_scale")?;
        let gelu_pipeline = pipeline(&device, &library, "gelu_tanh")?;
        let gated_gelu_pipeline = pipeline(&device, &library, "gated_gelu")?;
        let rope_pipeline = pipeline(&device, &library, "rope")?;
        let norm_rope_qkv_pipeline = pipeline(&device, &library, "norm_rope_qkv")?;
        let norm_rope_qkv_cache_pipeline = pipeline(&device, &library, "norm_rope_qkv_cache")?;
        let softcap_pipeline = pipeline(&device, &library, "softcap")?;
        let cache_append_pipeline = pipeline(&device, &library, "cache_append")?;
        let embedding_buffer_pipeline = pipeline(&device, &library, "embedding_scaled_buffer")?;
        let copy_last_row_pipeline = pipeline(&device, &library, "copy_last_row")?;
        let copy_buffer_pipeline = pipeline(&device, &library, "copy_buffer")?;
        let argmax_clear_pipeline = pipeline(&device, &library, "argmax_clear")?;
        let argmax_slots_pipeline = pipeline(&device, &library, "argmax_slots")?;
        let argmax_keys_to_token_pipeline = pipeline(&device, &library, "argmax_keys_to_token")?;
        let sample_topk_pipeline = pipeline(&device, &library, "sample_topk_blocks")?;
        let sample_draw_pipeline = pipeline(&device, &library, "sample_draw")?;
        let spin_burn_pipeline = pipeline(&device, &library, "spin_burn")?;
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
            rms_norm_residual_tg_pipeline,
            rms_norm_residual_layer_scale_tg_pipeline,
            greedy_pick_pipeline,
            greedy_token_pipeline,
            residual_pipeline,
            layer_scale_pipeline,
            gelu_pipeline,
            gated_gelu_pipeline,
            rope_pipeline,
            norm_rope_qkv_pipeline,
            norm_rope_qkv_cache_pipeline,
            softcap_pipeline,
            cache_append_pipeline,
            embedding_buffer_pipeline,
            copy_last_row_pipeline,
            copy_buffer_pipeline,
            argmax_clear_pipeline,
            argmax_slots_pipeline,
            argmax_keys_to_token_pipeline,
            sample_topk_pipeline,
            sample_draw_pipeline,
            spin_burn_pipeline,
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
            rope_threads(tokens, heads, head_dim, partial_factor, proportional)? as u64,
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

    #[allow(clippy::too_many_arguments)]
    pub fn encode_rms_norm_residual(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        input: &Buffer,
        weight: &Buffer,
        residual: &Buffer,
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
        encoder.dispatch_groups(
            &self.rms_norm_residual_tg_pipeline,
            &[Some(input), Some(weight), Some(residual), Some(output)],
            params_bytes(&params),
            rows as u64,
            Self::NORM_GROUP_THREADS,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_rms_norm_residual_layer_scale(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        input: &Buffer,
        weight: &Buffer,
        residual: &Buffer,
        output: &Buffer,
        rows: usize,
        width: usize,
        eps: f32,
        scale: f32,
    ) -> Result<(), KernelError> {
        if rows == 0 || width == 0 || !eps.is_finite() || eps <= 0.0 {
            return Err(KernelError::InvalidInput(
                "RMSNorm dimensions must be non-zero and eps finite and positive".to_string(),
            ));
        }
        let params = NormScaleParams {
            rows: to_u32(rows, "row count")?,
            width: to_u32(width, "norm width")?,
            eps,
            scale,
        };
        encoder.dispatch_groups(
            &self.rms_norm_residual_layer_scale_tg_pipeline,
            &[Some(input), Some(weight), Some(residual), Some(output)],
            params_bytes(&params),
            rows as u64,
            Self::NORM_GROUP_THREADS,
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

    /// Encode device-side argmax reduction over logits into `best` (one u32 per 65536 elements).
    pub fn encode_argmax(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        logits: &Buffer,
        best: &Buffer,
        vocab: usize,
    ) -> Result<(), KernelError> {
        if crate::ablated("head_argmax") {
            return Ok(());
        }
        let slots = vocab.div_ceil(65536);
        encoder.dispatch_1d(
            &self.argmax_clear_pipeline,
            &[Some(best)],
            &(slots as u32).to_le_bytes(),
            32,
            32,
        )?;
        let per_slot = 128u64;
        let groups = per_slot * slots as u64;
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct ArgmaxParams {
            count: u32,
            per_slot: u32,
        }
        let params = ArgmaxParams {
            count: to_u32(vocab, "vocab size")?,
            per_slot: per_slot as u32,
        };
        encoder.dispatch_groups(
            &self.argmax_slots_pipeline,
            &[Some(logits), Some(best)],
            params_bytes(&params),
            groups,
            256,
        )
    }

    /// Reduce device-side argmax keys to a single token ID on GPU into `token_out`,
    /// and optionally into `host_token_out[step_idx]`.
    pub fn encode_argmax_to_token(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        keys: &Buffer,
        token_out: Option<&Buffer>,
        host_token_out: Option<&Buffer>,
        step_idx: u32,
    ) -> Result<(), KernelError> {
        encoder.dispatch_1d(
            &self.argmax_keys_to_token_pipeline,
            &[Some(keys), token_out, host_token_out],
            &step_idx.to_le_bytes(),
            1,
            1,
        )
    }

    /// The dispatch geometry the device sampler needs for `vocab` and `top_k`,
    /// or the reason it cannot serve them.
    ///
    /// Public because the same numbers appear in three places that must agree:
    /// this function, the host-side validation in
    /// `kaplumba_engine::sampler::SamplerParams::validate`, and the buffers in
    /// `ForwardScratch`. `spec/device-sampler` R3/R7.
    ///
    /// The shape is fixed by one inequality. `blocks * k_padded == candidates`
    /// and `per_block * blocks >= count`, with both `candidates` and `per_block`
    /// staged in shared memory, so
    ///
    /// ```text
    /// count * k_padded <= SAMPLE_STAGE * SAMPLE_CANDIDATES
    /// ```
    ///
    /// At the shipped 262,144-wide vocabulary that is `k_padded <= 64`, which
    /// is exactly the shipped `--top-k` default. Wider `top_k` is rejected, not
    /// approximated.
    pub fn sample_geometry(
        vocab: usize,
        top_k: usize,
    ) -> Result<(u32, u32, u32, u32), KernelError> {
        const SAMPLE_CANDIDATES: usize = 4096;
        const SAMPLE_STAGE: usize = 4096;
        const SAMPLE_TOP_MAX: usize = 256;
        let bad = |why: &str| KernelError::InvalidInput(format!("device sampler: {why}"));
        if vocab == 0 || top_k == 0 {
            return Err(bad("vocab and top_k must be non-zero"));
        }
        if vocab < SAMPLE_CANDIDATES {
            return Err(bad(&format!(
                "vocabulary {vocab} is below the {SAMPLE_CANDIDATES} elements the \
                 candidate staging needs"
            )));
        }
        // Reject an over-cap `top_k` BEFORE padding it. The `next_pow2` loop is
        // bounded by the cap for a reason: `k_padded <<= 1` overflows to 0, and
        // `0 < top_k` is then true forever, so an unbounded `next_pow2` is an
        // infinite loop rather than a wrong answer — a large `top_k` would hang
        // the process instead of returning the error the caller is waiting for.
        // Bounding the loop alone is NOT sufficient: it would stop at the cap and
        // let `top_k = 1000` through as `k_padded = 256`, sampling from a
        // narrower candidate set than the config asked for. So the cap is
        // checked on the input, and the loop is bounded defensively.
        if top_k > SAMPLE_TOP_MAX {
            return Err(bad(&format!(
                "top_k {top_k} is above the {SAMPLE_TOP_MAX} the staging arrays accept"
            )));
        }
        let mut k_padded = 1usize;
        while k_padded < top_k {
            k_padded <<= 1;
        }
        if k_padded > SAMPLE_TOP_MAX || SAMPLE_CANDIDATES % k_padded != 0 {
            return Err(bad(&format!(
                "top_k {top_k} pads to {k_padded}, which is not a divisor of \
                 {SAMPLE_CANDIDATES} within the {SAMPLE_TOP_MAX} cap"
            )));
        }
        let blocks = SAMPLE_CANDIDATES / k_padded;
        let per_block = vocab.div_ceil(blocks);
        if per_block > SAMPLE_STAGE {
            return Err(bad(&format!(
                "top_k {top_k} at vocab {vocab} needs {per_block} elements staged \
                 per block, over the {SAMPLE_STAGE} limit"
            )));
        }
        if per_block < k_padded {
            return Err(bad(&format!(
                "top_k {top_k} at vocab {vocab} stages only {per_block} elements \
                 per block, so some emitted candidates would be padding"
            )));
        }
        Ok((
            to_u32(vocab, "vocab size")?,
            to_u32(k_padded, "padded top_k")?,
            to_u32(blocks, "sampler blocks")?,
            to_u32(per_block, "sampler block span")?,
        ))
    }

    /// D1 of the device sampler: per-block top-k into `candidates`.
    ///
    /// The `vocab` / `top_k` pair must have been accepted by
    /// [`Self::sample_geometry`]; it is re-checked here because that is the only
    /// place a mismatch becomes a wrong answer rather than an error.
    pub fn encode_sample_topk_blocks(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        logits: &Buffer,
        candidates: &Buffer,
        vocab: usize,
        top_k: usize,
    ) -> Result<(), KernelError> {
        if crate::ablated("sample_topk") {
            return Ok(());
        }
        let (count, k_padded, blocks, per_block) = Self::sample_geometry(vocab, top_k)?;
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct SampleTopKParams {
            count: u32,
            k_padded: u32,
            blocks: u32,
            per_block: u32,
        }
        let params = SampleTopKParams {
            count,
            k_padded,
            blocks,
            per_block,
        };
        encoder.dispatch_groups(
            &self.sample_topk_pipeline,
            &[Some(logits), Some(candidates)],
            params_bytes(&params),
            blocks as u64,
            Self::SAMPLE_GROUP_THREADS,
        )
    }

    /// D2 of the device sampler: global top-k, softmax, top-p, inverse-CDF, and
    /// the draw, writing the token into both staging buffers.
    ///
    /// `rng` is a one-element `ulong` buffer, seeded host-side once per turn.
    /// It is advanced here and nowhere else, and `sample_draw` runs exactly once
    /// per decode step.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_sample_draw(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        candidates: &Buffer,
        rng: &Buffer,
        token_out: Option<&Buffer>,
        host_token_out: Option<&Buffer>,
        ranked_out: Option<&Buffer>,
        vocab: usize,
        top_k: usize,
        temperature: f32,
        top_p: f32,
        step_idx: u32,
    ) -> Result<(), KernelError> {
        if crate::ablated("sample_draw") {
            return Ok(());
        }
        let (_, k_padded, blocks, per_block) = Self::sample_geometry(vocab, top_k)?;
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(KernelError::InvalidInput(
                "device sampler temperature must be finite and positive".to_string(),
            ));
        }
        if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            return Err(KernelError::InvalidInput(
                "device sampler top-p must be in (0, 1]".to_string(),
            ));
        }
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct SampleDrawParams {
            count: u32,
            top_k: u32,
            k_padded: u32,
            candidates: u32,
            per_block: u32,
            step_idx: u32,
            temperature: f32,
            top_p: f32,
        }
        let params = SampleDrawParams {
            count: to_u32(vocab, "vocab size")?,
            top_k: to_u32(top_k, "top_k")?,
            k_padded,
            candidates: to_u32(
                blocks as usize * k_padded as usize,
                "sampler candidate count",
            )?,
            per_block,
            step_idx,
            temperature,
            top_p,
        };
        encoder.dispatch_groups(
            &self.sample_draw_pipeline,
            &[
                Some(candidates),
                None,
                Some(rng),
                token_out,
                host_token_out,
                ranked_out,
            ],
            params_bytes(&params),
            1,
            Self::SAMPLE_GROUP_THREADS,
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
        // One thread per (token, head, rotation pair) for the rotated span, plus
        // one per (token, head, column) for the non-rotated tail. The kernel
        // derives the same split from `params`, so this only has to total the
        // two, and both dispatch sites go through `rope_threads` so they cannot
        // disagree -- which is exactly how the `rope()` helper and
        // `encode_rope` drifted apart once already.
        encoder.dispatch_1d(
            &self.rope_pipeline,
            &[Some(input), Some(output)],
            params_bytes(&params),
            rope_threads(tokens, heads, head_dim, partial_factor, proportional)? as u64,
            64,
        )
    }

    /// Encode the fused Q/K norm+RoPE and V norm, replacing
    /// `encode_rms_norm` x2 + `encode_rms_norm_unweighted` + `encode_rope` x2.
    ///
    /// Bit-identical to that sequence, not merely close: the kernel reproduces
    /// the fp16 rounding the separate norms write and the separate `rope` reads
    /// back, and reduces the sum of squares in the same order. See the comment on
    /// `norm_rope_qkv` in `elements.metal` -- the `float(half(...))` and the
    /// strided accumulation are both load-bearing, and removing either makes the
    /// output differ.
    ///
    /// `head_dim` is 256 on sliding layers and 512 on full-attention layers; both
    /// are handled by the same kernel through its strided loops.
    ///
    /// Dispatch is one threadgroup per head, `NORM_GROUP_THREADS` wide. That is
    /// a hard requirement, not a preference: the reduction tree inside is
    /// wired to 256 lanes, and a different width would sum the head in a
    /// different order and stop matching the reference norms.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_norm_rope_qkv(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        q_weight: &Buffer,
        k_weight: &Buffer,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        offset: usize,
        theta: f32,
        partial_factor: f32,
        proportional: bool,
        eps: f32,
    ) -> Result<(), KernelError> {
        if q_heads == 0
            || kv_heads == 0
            || head_dim == 0
            || !head_dim.is_multiple_of(2)
            || head_dim > Self::NORM_STAGING_MAX
        {
            return Err(KernelError::InvalidInput(format!(
                "norm_rope_qkv needs non-zero head counts, an even head_dim, and \
                 head_dim <= {} (the kernel's threadgroup staging array)",
                Self::NORM_STAGING_MAX
            )));
        }
        if !theta.is_finite() || theta <= 0.0 || !eps.is_finite() || eps <= 0.0 {
            return Err(KernelError::InvalidInput(
                "norm_rope_qkv theta and eps must be finite and positive".to_string(),
            ));
        }
        if proportional
            && (!partial_factor.is_finite() || partial_factor <= 0.0 || partial_factor > 1.0)
        {
            return Err(KernelError::InvalidInput(
                "proportional RoPE factor must be in (0, 1]".to_string(),
            ));
        }
        let params = NormRopeQkvParams {
            q_heads: to_u32(q_heads, "norm_rope_qkv q head count")?,
            kv_heads: to_u32(kv_heads, "norm_rope_qkv kv head count")?,
            head_dim: to_u32(head_dim, "norm_rope_qkv head dimension")?,
            offset: to_u32(offset, "norm_rope_qkv offset")?,
            eps,
            theta,
            partial_factor,
            proportional: u32::from(proportional),
        };
        // q_heads + 2 * kv_heads: one group per Q head, per K head and per V head.
        // `checked_product`/`checked_add` are the overflow-reporting helpers the
        // rest of this file uses for every other shape computation.
        let kv_and_v = checked_product(kv_heads, 2, "norm_rope_qkv K and V head count")?;
        let groups = checked_add(q_heads, kv_and_v, "norm_rope_qkv threadgroup count")?;
        encoder.dispatch_groups(
            &self.norm_rope_qkv_pipeline,
            &[Some(q), Some(k), Some(v), Some(q_weight), Some(k_weight)],
            params_bytes(&params),
            groups as u64,
            Self::NORM_GROUP_THREADS,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_norm_rope_qkv_cache(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        q: &Buffer,
        raw_k: &Buffer,
        raw_v: &Buffer,
        key_cache: &Buffer,
        value_cache: &Buffer,
        position_cache: &Buffer,
        q_weight: &Buffer,
        k_weight: &Buffer,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        offset: usize,
        theta: f32,
        partial_factor: f32,
        proportional: bool,
        eps: f32,
        slot: usize,
        stream_len: usize,
    ) -> Result<(), KernelError> {
        if q_heads == 0
            || kv_heads == 0
            || head_dim == 0
            || !head_dim.is_multiple_of(2)
            || head_dim > Self::NORM_STAGING_MAX
        {
            return Err(KernelError::InvalidInput(format!(
                "norm_rope_qkv_cache needs non-zero head counts, an even head_dim, and \
                 head_dim <= {} (the kernel's threadgroup staging array)",
                Self::NORM_STAGING_MAX
            )));
        }
        if !theta.is_finite() || theta <= 0.0 || !eps.is_finite() || eps <= 0.0 {
            return Err(KernelError::InvalidInput(
                "norm_rope_qkv_cache theta and eps must be finite and positive".to_string(),
            ));
        }
        if proportional
            && (!partial_factor.is_finite() || partial_factor <= 0.0 || partial_factor > 1.0)
        {
            return Err(KernelError::InvalidInput(
                "proportional RoPE factor must be in (0, 1]".to_string(),
            ));
        }
        let params = NormRopeQkvCacheParams {
            q_heads: to_u32(q_heads, "norm_rope_qkv_cache q head count")?,
            kv_heads: to_u32(kv_heads, "norm_rope_qkv_cache kv head count")?,
            head_dim: to_u32(head_dim, "norm_rope_qkv_cache head dimension")?,
            offset: to_u32(offset, "norm_rope_qkv_cache offset")?,
            eps,
            theta,
            partial_factor,
            proportional: u32::from(proportional),
            slot: to_u32(slot, "norm_rope_qkv_cache slot")?,
            stream_len: to_u32(stream_len, "norm_rope_qkv_cache stream_len")?,
        };
        if crate::ablated("kv_norm_rope") {
            return Ok(());
        }
        let kv_and_v = checked_product(kv_heads, 2, "norm_rope_qkv_cache K and V head count")?;
        let groups = checked_add(q_heads, kv_and_v, "norm_rope_qkv_cache threadgroup count")?;
        encoder.dispatch_groups(
            &self.norm_rope_qkv_cache_pipeline,
            &[
                Some(q),
                Some(raw_k),
                Some(raw_v),
                Some(key_cache),
                Some(value_cache),
                Some(position_cache),
                Some(q_weight),
                Some(k_weight),
            ],
            params_bytes(&params),
            groups as u64,
            Self::NORM_GROUP_THREADS,
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
        if crate::ablated("head_softcap") {
            return Ok(());
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

    /// Encode `dispatches` copies of a deliberately slow kernel, so a timing
    /// instrument can be shown to resolve a cost injected into the region it
    /// claims to measure.
    ///
    /// This exists because a profiler that cannot fail has no coverage claim. The
    /// kernel is a dependent FMA loop with no memory traffic, so its cost is set
    /// by `iters` and the machine's clock alone; comparing the instrument's
    /// number against the host wall clock therefore gives a real cross-check
    /// rather than a tautology.
    ///
    /// `out` must hold at least `threads` floats. The binding list is
    /// `&[Some(out), None]` on purpose: `dispatch_1d` packs the trailing
    /// `params` bytes at index `bindings.len()`, and the kernel reads its trip
    /// count from `buffer(2)`, so the list must have two entries. See the
    /// `spin_burn` comment in `elements.metal` for why the wrong index fails
    /// silently rather than loudly.
    pub fn encode_spin_burn(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        out: &Buffer,
        dispatches: usize,
        iters: u32,
        threads: u64,
    ) -> Result<(), KernelError> {
        if dispatches == 0 || threads == 0 || threads > u32::MAX as u64 {
            return Err(KernelError::InvalidInput(
                "spin_burn dispatch and thread counts must be non-zero".to_string(),
            ));
        }
        for _ in 0..dispatches {
            encoder.dispatch_1d(
                &self.spin_burn_pipeline,
                &[Some(out), None],
                &iters.to_le_bytes(),
                threads,
                64,
            )?;
        }
        Ok(())
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

/// Build one compute pipeline, through the audit.
///
/// See [`crate::pipeline_audit`]: the point is not the build, it is that the
/// build is *counted*, so "the element kernels compile once, at load" is a
/// measurement and not a reading of this file.
fn pipeline(
    device: &Device,
    library: &Library,
    name: &str,
) -> Result<ComputePipelineState, KernelError> {
    crate::pipeline_audit::build(device, library, name)
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

fn checked_add(left: usize, right: usize, name: &str) -> Result<usize, KernelError> {
    left.checked_add(right)
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

/// Decode winning token index from the 16-bit packed slot keys.
///
/// Each key stores the 16-bit float value order in the high 16 bits, and
/// `0xFFFF - in_slot_index` in the low 16 bits.
pub fn decode_argmax_token(keys: &[u32]) -> u32 {
    let mut best_slot = 0usize;
    let mut best = keys[0];
    for slot in 1..keys.len() {
        if (keys[slot] >> 16) > (best >> 16) {
            best = keys[slot];
            best_slot = slot;
        }
    }
    let in_slot = 0xFFFFu32.wrapping_sub(best & 0xFFFF);
    (best_slot as u32) * 65536 + in_slot
}
