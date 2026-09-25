use std::ffi::c_void;
use std::mem::{size_of, size_of_val};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};
use thiserror::Error;

const SOURCE: &str = include_str!("exl3.metal");

#[derive(Debug, Error)]
pub enum KernelError {
    #[error("Metal system device is unavailable")]
    DeviceUnavailable,
    #[error("Metal shader compilation failed: {0}")]
    Shader(String),
    #[error("Metal pipeline creation failed: {0}")]
    Pipeline(String),
    #[error("invalid kernel input: {0}")]
    InvalidInput(String),
    #[error("Metal command buffer failed")]
    CommandFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exl3Rate {
    Integer(u8),
    Fractional(u8),
}

impl Exl3Rate {
    pub fn packed_words(self) -> usize {
        match self {
            Self::Integer(bits) => 16 * bits as usize,
            Self::Fractional(bits) => 16 * bits as usize + 8,
        }
    }

    pub(crate) fn encoded(self) -> u32 {
        match self {
            Self::Integer(bits) => u32::from(bits),
            Self::Fractional(bits) => 16 + u32::from(bits),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exl3Codebook {
    Plain,
    Mcg,
    Mul1,
}

impl Exl3Codebook {
    pub(crate) fn encoded(self) -> u32 {
        match self {
            Self::Plain => 0,
            Self::Mcg => 1,
            Self::Mul1 => 2,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CodecParams {
    rate: u32,
    codebook: u32,
}

pub struct Exl3Kernels {
    device: Device,
    queue: CommandQueue,
    _library: Library,
    decode_pipeline: ComputePipelineState,
    hadamard_pipeline: ComputePipelineState,
}

impl Exl3Kernels {
    pub fn new() -> Result<Self, KernelError> {
        let device = Device::system_default().ok_or(KernelError::DeviceUnavailable)?;
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(SOURCE, &options)
            .map_err(KernelError::Shader)?;
        let decode_function = library
            .get_function("exl3_decode_tile", None)
            .map_err(KernelError::Shader)?;
        let hadamard_function = library
            .get_function("hadamard128_vectors", None)
            .map_err(KernelError::Shader)?;
        let decode_pipeline = device
            .new_compute_pipeline_state_with_function(&decode_function)
            .map_err(KernelError::Pipeline)?;
        let hadamard_pipeline = device
            .new_compute_pipeline_state_with_function(&hadamard_function)
            .map_err(KernelError::Pipeline)?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            _library: library,
            decode_pipeline,
            hadamard_pipeline,
        })
    }

    pub fn decode_tile(
        &self,
        packed: &[u16],
        rate: Exl3Rate,
        codebook: Exl3Codebook,
    ) -> Result<[f32; 256], KernelError> {
        match rate {
            Exl3Rate::Integer(bits) if !(1..=8).contains(&bits) => {
                return Err(KernelError::InvalidInput(format!(
                    "integer EXL3 rate {bits} is unsupported"
                )));
            }
            Exl3Rate::Fractional(bits) if !(1..=3).contains(&bits) => {
                return Err(KernelError::InvalidInput(format!(
                    "fractional EXL3 rate {bits}.5 is unsupported"
                )));
            }
            _ => {}
        }
        if packed.len() != rate.packed_words() {
            return Err(KernelError::InvalidInput(format!(
                "packed length {} does not match {}",
                packed.len(),
                rate.packed_words()
            )));
        }
        if matches!(rate, Exl3Rate::Fractional(_)) && codebook != Exl3Codebook::Mul1 {
            return Err(KernelError::InvalidInput(
                "fractional EXL3 requires mul1".to_string(),
            ));
        }
        let input = self.make_input_buffer(packed)?;
        let output = self.device.new_buffer(
            256 * size_of::<f32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params = CodecParams {
            rate: rate.encoded(),
            codebook: codebook.encoded(),
        };
        self.dispatch(&self.decode_pipeline, &input, &output, &params, 256)?;
        read_f32_buffer(&output).map(|values| {
            let mut result = [0.0; 256];
            result.copy_from_slice(&values);
            result
        })
    }

    pub fn hadamard_128(&self, input: &[u16]) -> Result<Vec<u16>, KernelError> {
        if input.is_empty() || !input.len().is_multiple_of(128) {
            return Err(KernelError::InvalidInput(
                "Hadamard input length must be a non-zero multiple of 128".to_string(),
            ));
        }
        let input_buffer = self.make_input_buffer(input)?;
        let output = self.device.new_buffer(
            size_of_val(input) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let vector_count = (input.len() / 128) as u32;
        self.dispatch(
            &self.hadamard_pipeline,
            &input_buffer,
            &output,
            &vector_count,
            input.len() as u64,
        )?;
        read_u16_buffer(&output)
    }

    fn make_input_buffer(&self, values: &[u16]) -> Result<Buffer, KernelError> {
        if values.is_empty() {
            return Err(KernelError::InvalidInput(
                "input cannot be empty".to_string(),
            ));
        }
        Ok(self.device.new_buffer_with_data(
            values.as_ptr() as *const c_void,
            size_of_val(values) as u64,
            MTLResourceOptions::StorageModeShared,
        ))
    }

    fn dispatch(
        &self,
        pipeline: &ComputePipelineState,
        input: &Buffer,
        output: &Buffer,
        params: &impl AsRawParams,
        threads: u64,
    ) -> Result<(), KernelError> {
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(2, params.byte_len(), params.pointer());
        encoder.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(64, 1, 1));
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(KernelError::CommandFailed);
        }
        Ok(())
    }
}

trait AsRawParams {
    fn byte_len(&self) -> u64;
    fn pointer(&self) -> *const c_void;
}

impl AsRawParams for CodecParams {
    fn byte_len(&self) -> u64 {
        size_of::<Self>() as u64
    }

    fn pointer(&self) -> *const c_void {
        self as *const Self as *const c_void
    }
}

impl AsRawParams for u32 {
    fn byte_len(&self) -> u64 {
        size_of::<Self>() as u64
    }

    fn pointer(&self) -> *const c_void {
        self as *const Self as *const c_void
    }
}

fn read_f32_buffer(buffer: &Buffer) -> Result<Vec<f32>, KernelError> {
    let length = buffer.length() as usize;
    if !length.is_multiple_of(size_of::<f32>()) {
        return Err(KernelError::CommandFailed);
    }
    let pointer = buffer.contents() as *const f32;
    if pointer.is_null() {
        return Err(KernelError::CommandFailed);
    }
    Ok(unsafe { std::slice::from_raw_parts(pointer, length / size_of::<f32>()) }.to_vec())
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
