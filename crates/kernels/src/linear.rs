use std::ffi::c_void;
use std::mem::{size_of, size_of_val};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    Library, MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

use crate::encoder::TensorRef;
use crate::exl3::{Exl3Codebook, Exl3Rate, KernelError};

const SOURCE: &str = include_str!("linear.metal");

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
    finalize_rows_pipeline: ComputePipelineState,
    zero_raw_pipeline: ComputePipelineState,
    /// `linear_gemv_fast` specialised on (bits, MCG codebook) for integer rates.
    fast_gemv_pipelines: Vec<Option<ComputePipelineState>>,
}

impl LinearKernels {
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
            gemv_pipeline,
            finalize_pipeline,
            input_rows_pipeline,
            gemv_tiled_pipeline,
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
        self.encode(
            encoder,
            &self.input_pipeline,
            &[
                Some(&input_buffer),
                Some(&input_sign_buffer),
                Some(&transformed),
            ],
            &input_blocks.to_le_bytes(),
            u64::from(input_blocks),
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
        self.encode(
            encoder,
            &self.input_pipeline,
            &[
                Some(&input_buffer),
                Some(&input_sign_buffer),
                Some(&transformed),
            ],
            &input_blocks.to_le_bytes(),
            u64::from(input_blocks),
        );
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
        encoder.dispatch_1d(
            &self.input_pipeline,
            &[Some(input), Some(input_signs), Some(transformed)],
            &input_blocks.to_le_bytes(),
            u64::from(input_blocks),
            64,
        )?;
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
        // The tiled kernel accumulates across query tiles, so start clean.
        encoder.dispatch_1d(
            &self.zero_raw_pipeline,
            &[Some(raw)],
            &[],
            output_elements as u64,
            64,
        )?;
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
