use std::ffi::c_void;
use std::mem::{size_of, size_of_val};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    Library, MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

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

pub struct LinearKernels {
    device: Device,
    queue: CommandQueue,
    _library: Library,
    input_pipeline: ComputePipelineState,
    gemv_pipeline: ComputePipelineState,
    finalize_pipeline: ComputePipelineState,
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
        let input_pipeline = device
            .new_compute_pipeline_state_with_function(&input_function)
            .map_err(KernelError::Pipeline)?;
        let gemv_pipeline = device
            .new_compute_pipeline_state_with_function(&gemv_function)
            .map_err(KernelError::Pipeline)?;
        let finalize_pipeline = device
            .new_compute_pipeline_state_with_function(&finalize_function)
            .map_err(KernelError::Pipeline)?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            _library: library,
            input_pipeline,
            gemv_pipeline,
            finalize_pipeline,
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
