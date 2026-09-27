use std::ffi::c_void;
use std::mem::{size_of, size_of_val};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

pub use crate::exl3::KernelError;

const SOURCE: &str = include_str!("attention.metal");
const MAX_HEAD_DIM: usize = 512;

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
    scores_pipeline: ComputePipelineState,
    row_reduce_pipeline: ComputePipelineState,
    combine_pipeline: ComputePipelineState,
}

impl AttentionKernels {
    pub fn new() -> Result<Self, KernelError> {
        let device = Device::system_default().ok_or(KernelError::DeviceUnavailable)?;
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(SOURCE, &options)
            .map_err(KernelError::Shader)?;
        let function = library
            .get_function("online_attention", None)
            .map_err(KernelError::Shader)?;
        let online_attention_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(KernelError::Pipeline)?;
        let mut extra = |name: &str| -> Result<ComputePipelineState, KernelError> {
            let f = library.get_function(name, None).map_err(KernelError::Shader)?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(KernelError::Pipeline)
        };
        let scores_pipeline = extra("attention_scores")?;
        let row_reduce_pipeline = extra("attention_row_reduce")?;
        let combine_pipeline = extra("attention_combine")?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            _library: library,
            online_attention_pipeline,
            scores_pipeline,
            row_reduce_pipeline,
            combine_pipeline,
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
        };
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.online_attention_pipeline);
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

    /// Encode the online-softmax attention kernel into a caller-owned command
    /// buffer, reading keys/values straight out of the GPU-resident cache.
    /// Split-key attention for the decode shape: one query, few tokens.
    ///
    /// The serial kernel dispatches `queries * q_heads` threads, so decode runs
    /// on 32 threads while streaming the whole KV cache. Measured 153 ms per
    /// token in situ against 132 us for the same shape on an L2-resident buffer,
    /// so the serialisation dominates. This runs three passes with
    /// `q_heads * key_count` and `q_heads * head_dim` threads instead, and reads
    /// the KV data once rather than once per thread.
    ///
    /// `scores` must hold `q_heads * key_count` floats.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_online_attention_split(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        request: AttentionEncodeRequest<'_>,
        scores: &Buffer,
        row_max: &Buffer,
        row_sum: &Buffer,
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
        let params = AttentionParams {
            queries: 1,
            q_heads: to_u32(q_heads, "query heads")?,
            kv_heads: to_u32(kv_heads, "KV heads")?,
            head_dim: to_u32(head_dim, "head dimension")?,
            key_count: to_u32(key_count, "key count")?,
            scale,
            window: to_u32(window, "window")?,
        };
        let bytes = params_bytes(&params);

        encoder.dispatch_1d(
            &self.scores_pipeline,
            &[
                Some(queries),
                Some(keys),
                Some(query_positions),
                Some(key_positions),
                Some(scores),
            ],
            bytes,
            (q_heads * key_count) as u64,
            64,
        )?;
        encoder.dispatch_1d(
            &self.row_reduce_pipeline,
            &[Some(scores), Some(row_max), Some(row_sum)],
            bytes,
            q_heads as u64,
            64,
        )?;
        encoder.dispatch_1d(
            &self.combine_pipeline,
            &[Some(values), Some(scores), Some(row_max), Some(row_sum), Some(output)],
            bytes,
            (q_heads * head_dim) as u64,
            64,
        )
    }

    pub fn encode_online_attention(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        request: AttentionEncodeRequest<'_>,
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
        };
        encoder.dispatch_1d(
            &self.online_attention_pipeline,
            &[
                Some(queries),
                Some(keys),
                Some(values),
                Some(query_positions),
                Some(key_positions),
                Some(output),
            ],
            params_bytes(&params),
            (queries_count * q_heads) as u64,
            64,
        )
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
