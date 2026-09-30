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
const ATTENTION_DECODE_THREADS_TARGET: usize = 16_384;

/// Floats the decode partial buffer holds for a layer: two softmax statistics
/// and one unnormalised accumulator per (split, query head).
pub fn decode_partial_elements(q_heads: usize, head_dim: usize) -> usize {
    let rows = ATTENTION_SPLITS_MAX.saturating_mul(q_heads);
    rows.saturating_mul(head_dim)
        .saturating_add(rows.saturating_mul(2))
}

/// How many ways to cut `key_count` keys for the decode path.
///
/// Aiming at a fixed *thread* count (not a fixed workgroup count) matters
/// because a threadgroup is `32 * (q_heads / kv_heads)` threads, so a layer with
/// a small GQA ratio would otherwise get a quarter of the threads at the same
/// split count. The result is always in `1 ..= ATTENTION_SPLITS_MAX`, which is
/// what lets the scratch stay fixed-size.
pub fn decode_splits(q_heads: usize, key_count: usize) -> usize {
    if q_heads == 0 || key_count == 0 {
        return 1;
    }
    // MEASUREMENT KNOB. `KAPLUMBA_DECODE_SPLITS=N` pins the split count so the
    // choice can be swept without a rebuild, which matters because the optimum
    // is not the thread target: an isolated harness measured the full-attention
    // layers (4 kv heads) at 8k running 1.91x faster at 12 splits than at 16,
    // reproduced twice at 3 significant figures. The shipped thread target gives
    // 16 for both layer shapes, and that is suspicious rather than established:
    // it fixes the *thread* count at 16384 while the *workgroup* count then
    // differs 4x between shapes (16 kv heads x 16 = 256 workgroups of 64
    // threads, versus 4 x 16 = 64 workgroups of 256 threads), and 64 workgroups
    // on 20 cores quantises badly. That is a hypothesis, not a measurement, so
    // the default stays at the thread target until this is swept end to end.
    if let Some(pinned) = std::env::var("KAPLUMBA_DECODE_SPLITS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| (1..=ATTENTION_SPLITS_MAX).contains(v))
    {
        return pinned;
    }
    let wanted = ATTENTION_DECODE_THREADS_TARGET / (32 * q_heads.max(1));
    let affordable = key_count.div_ceil(ATTENTION_MIN_KEYS_PER_SPLIT);
    wanted.clamp(1, ATTENTION_SPLITS_MAX).min(affordable).max(1)
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
    /// The same body run over one split of the KV axis (flash-decoding).
    decode_split_pipelines: [ComputePipelineState; GQA_PIPELINES],
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

fn decode_split_pipeline_name(head_dim: usize, nsg: usize) -> String {
    format!("attention_decode_split_{head_dim}_{nsg}")
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
            let f = library
                .get_function(name, None)
                .map_err(KernelError::Shader)?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(KernelError::Pipeline)
        };
        let online_attention_pipeline = extra("online_attention")?;
        let online_attention_256_pipeline = extra("online_attention_256")?;
        let online_attention_512_pipeline = extra("online_attention_512")?;
        let attention_dim_256_pipeline = extra("online_attention_dim_256")?;
        let attention_dim_512_pipeline = extra("online_attention_dim_512")?;
        let attention_tg2_256_pipeline = extra("online_attention_tg2_256")?;
        let attention_tg2_512_pipeline = extra("online_attention_tg2_512")?;
        let mut attention_gqa_pipelines = Vec::with_capacity(GQA_PIPELINES);
        let mut decode_split_pipelines = Vec::with_capacity(GQA_PIPELINES);
        for head_dim in GQA_HEAD_DIMS {
            for nsg in GQA_RATIOS {
                attention_gqa_pipelines.push(extra(&gqa_pipeline_name(head_dim, nsg))?);
                decode_split_pipelines.push(extra(&decode_split_pipeline_name(head_dim, nsg))?);
            }
        }
        let attention_gqa_pipelines: [ComputePipelineState; GQA_PIPELINES] =
            attention_gqa_pipelines.try_into().map_err(|_| {
                KernelError::Pipeline("unexpected transposed attention pipeline count".to_string())
            })?;
        let decode_split_pipelines: [ComputePipelineState; GQA_PIPELINES] =
            decode_split_pipelines.try_into().map_err(|_| {
                KernelError::Pipeline("unexpected decode split pipeline count".to_string())
            })?;
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
            decode_split_pipelines,
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
        let splits = decode_splits(q_heads, key_count);
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
                &self.attention_gqa_pipelines[index],
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
