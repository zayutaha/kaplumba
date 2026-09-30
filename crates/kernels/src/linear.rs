use std::ffi::c_void;
use std::mem::{size_of, size_of_val};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    Library, MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

use crate::encoder::TensorRef;
use crate::exl3::{Exl3Codebook, Exl3Rate, KernelError};

const SOURCE: &str = include_str!("linear.metal");

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

pub struct LinearKernels {
    device: Device,
    queue: CommandQueue,
    _library: Library,
    input_pipeline: ComputePipelineState,
    gemv_pipeline: ComputePipelineState,
    finalize_pipeline: ComputePipelineState,
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
    finalize_rows_pipeline: ComputePipelineState,
    zero_raw_pipeline: ComputePipelineState,
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
}

impl LinearKernels {
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
            .new_library_with_source(SOURCE, &options)
            .map_err(KernelError::Shader)?;
        let input_function = library
            .get_function("linear_input_transform", None)
            .map_err(KernelError::Shader)?;
        let gemv_function = library
            .get_function("linear_gemv", None)
            .map_err(KernelError::Shader)?;
        let finalize_function = library
            .get_function("linear_finalize", None)
            .map_err(KernelError::Shader)?;
        let input_rows_function = library
            .get_function("linear_input_rows", None)
            .map_err(KernelError::Shader)?;
        let gemv_tiled_function = library
            .get_function("linear_gemv_tiled", None)
            .map_err(KernelError::Shader)?;
        let finalize_rows_function = library
            .get_function("linear_finalize_rows", None)
            .map_err(KernelError::Shader)?;
        let input_pipeline = device
            .new_compute_pipeline_state_with_function(&input_function)
            .map_err(KernelError::Pipeline)?;
        let mut ishizuki_gemv_pipelines = Vec::with_capacity(9);
        ishizuki_gemv_pipelines.push(None);
        for bits in 1..=8_u32 {
            let name = format!("linear_gemv_ishizuki_k{bits}");
            let function = library
                .get_function(&name, None)
                .map_err(KernelError::Pipeline)?;
            ishizuki_gemv_pipelines.push(Some(
                device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(KernelError::Pipeline)?,
            ));
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
                    let function = library
                        .get_function(&name, None)
                        .map_err(KernelError::Pipeline)?;
                    let pipeline = device
                        .new_compute_pipeline_state_with_function(&function)
                        .map_err(KernelError::Pipeline)?;
                    exl3_gemv_by_name.insert(name.clone(), pipeline.clone());
                    exl3_gemv_slot_names.push(name);
                    exl3_gemv_pipelines.push(Some(pipeline));
                }
            }
        }
        debug_assert_eq!(exl3_gemv_pipelines.len(), EXL3_GEMV_STATES);
        debug_assert_eq!(exl3_gemv_slot_names.len(), EXL3_GEMV_STATES);

        let gemv_pipeline = device
            .new_compute_pipeline_state_with_function(&gemv_function)
            .map_err(KernelError::Pipeline)?;
        let finalize_pipeline = device
            .new_compute_pipeline_state_with_function(&finalize_function)
            .map_err(KernelError::Pipeline)?;
        let input_rows_pipeline = device
            .new_compute_pipeline_state_with_function(&input_rows_function)
            .map_err(KernelError::Pipeline)?;
        let gemv_tiled_pipeline = device
            .new_compute_pipeline_state_with_function(&gemv_tiled_function)
            .map_err(KernelError::Pipeline)?;
        let gemm_mma_function = library
            .get_function("linear_gemm_mma", None)
            .map_err(KernelError::Shader)?;
        let gemm_mma_pipeline = device
            .new_compute_pipeline_state_with_function(&gemm_mma_function)
            .map_err(KernelError::Pipeline)?;
        let gemm_mma_regpipe_function = library
            .get_function("linear_gemm_mma_regpipe", None)
            .map_err(KernelError::Shader)?;
        let gemm_mma_regpipe_pipeline = device
            .new_compute_pipeline_state_with_function(&gemm_mma_regpipe_function)
            .map_err(KernelError::Pipeline)?;
        let finalize_rows_pipeline = device
            .new_compute_pipeline_state_with_function(&finalize_rows_function)
            .map_err(KernelError::Pipeline)?;
        let zero_raw_function = library
            .get_function("linear_zero_raw", None)
            .map_err(KernelError::Shader)?;
        let zero_raw_pipeline = device
            .new_compute_pipeline_state_with_function(&zero_raw_function)
            .map_err(KernelError::Pipeline)?;
        // Specialised decode pipelines, one per integer bit rate.
        let mut fast_gemv_pipelines = Vec::with_capacity(8);
        for bits in 1..=8_u32 {
            let entry = format!("linear_gemv_fast{bits}");
            let specialised = match library.get_function(&entry, None) {
                Ok(function) => device
                    .new_compute_pipeline_state_with_function(&function)
                    .ok(),
                Err(_) => None,
            };
            fast_gemv_pipelines.push(specialised);
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
            gemv_pipeline,
            finalize_pipeline,
            input_rows_pipeline,
            gemv_tiled_pipeline,
            gemm_mma_pipeline,
            gemm_mma_regpipe_pipeline,
            finalize_rows_pipeline,
            zero_raw_pipeline,
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
        } else if let Some(pipeline) = self.exl3_gemv_pipeline(rate, packed_offset, codebook) {
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

    fn ishizuki_gemv_pipeline(
        &self,
        rate: Exl3Rate,
        packed_offset: u64,
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
    pub fn encode_gemv_named(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        name: &str,
        request: &LinearEncodeRequest<'_>,
    ) -> Result<(), KernelError> {
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
        encoder.dispatch_1d_offset(
            pipeline,
            &[
                TensorRef::new(request.packed, request.packed_offset),
                TensorRef::new(request.transformed, 0),
                TensorRef::new(request.raw, 0),
            ],
            params_bytes(&params),
            (out_features / 16 * 256) as u64,
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

    fn exl3_gemv_pipeline_by_name(&self, name: &str) -> Option<&ComputePipelineState> {
        self.exl3_gemv_by_name.get(name)
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
        if let Some(pipeline) = self.ishizuki_gemv_pipeline(rate, packed_offset) {
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
        } else if let Some(pipeline) = self.exl3_gemv_pipeline(rate, packed_offset, codebook) {
            encoder.dispatch_1d_offset(
                pipeline,
                &[
                    TensorRef::new(packed, packed_offset),
                    TensorRef::new(transformed, 0),
                    TensorRef::new(raw, 0),
                ],
                params_bytes(&params),
                (out_features / 16 * 256) as u64,
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
        encoder.dispatch_1d(
            &self.finalize_pipeline,
            &[Some(raw), Some(output_signs), Some(output)],
            &output_blocks.to_le_bytes(),
            u64::from(output_blocks),
            64,
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
            if out_features % 64 == 0
                && self.gemm_mma_pipeline.max_total_threads_per_threadgroup() >= 256
            {
                // The register-pipelined variant is the folded-MCG case only;
                // everything else dispatches the original kernel below.
                let regpipe = !env_flag("KAPLUMBA_NO_REGPIPE")
                    && rate == Exl3Rate::Integer(3)
                    && codebook == Exl3Codebook::Mcg;
                let gemm = if regpipe {
                    &self.gemm_mma_regpipe_pipeline
                } else {
                    &self.gemm_mma_pipeline
                };
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
                    (out_features / 64) as u64,
                    queries.div_ceil(128) as u64,
                    256,
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
    packed_offset: u64,
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
