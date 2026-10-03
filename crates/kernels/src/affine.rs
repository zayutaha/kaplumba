//! spec/mtp-40tps T5b: reference GEMV over MLX affine-4bit packed weights.
//!
//! See `affine.metal` for the math. Activations are FP16 (the engine converts
//! the checkpoint's BF16 to FP16 at load), so scales/biases must likewise be
//! supplied as FP16. Accumulation is FP32, matching MLX's bf16 mm.

use std::ffi::c_void;
use std::mem::{size_of, size_of_val};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library, MTLResourceOptions,
};

pub use crate::exl3::KernelError;

use crate::elements::clone_kernel_error;

const SOURCE: &str = include_str!("affine.metal");

pub struct AffineKernels {
    device: Device,
    queue: CommandQueue,
    _library: Library,
    gemv_pipeline: ComputePipelineState,
}

impl AffineKernels {
    pub fn shared() -> Result<&'static Self, KernelError> {
        static INSTANCE: std::sync::OnceLock<Result<AffineKernels, KernelError>> =
            std::sync::OnceLock::new();
        match INSTANCE.get_or_init(Self::new) {
            Ok(kernels) => Ok(kernels),
            Err(error) => Err(clone_kernel_error(error)),
        }
    }

    pub fn new() -> Result<Self, KernelError> {
        let device = Device::system_default().ok_or(KernelError::DeviceUnavailable)?;
        let options = CompileOptions::new();
        let library = match device.new_library_with_source(SOURCE, &options) {
            Ok(library) => library,
            Err(e) => return Err(KernelError::Shader(e.to_string())),
        };
        let gemv_pipeline = crate::pipeline_audit::build(&device, &library, "affine4_gemv")?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            _library: library,
            gemv_pipeline,
        })
    }

    fn buffer(&self, byte_len: u64) -> Buffer {
        self.device
            .new_buffer(byte_len, MTLResourceOptions::StorageModeShared)
    }

    /// y = W_dequant(x) with W packed [n, k/8] u32, scales/biases [n, k/group] fp16.
    pub fn gemv(
        &self,
        x: &[u16],
        weight: &[u32],
        scales: &[u16],
        biases: &[u16],
        n_rows: usize,
        k_dim: usize,
        group_size: usize,
    ) -> Result<Vec<u16>, KernelError> {
        if k_dim % 8 != 0 || k_dim % group_size != 0 || x.len() != k_dim {
            return Err(KernelError::InvalidInput(
                "k_dim must be a multiple of 8 and group_size, x must span k_dim".to_string(),
            ));
        }
        if weight.len() != n_rows * (k_dim / 8)
            || scales.len() != n_rows * (k_dim / group_size)
            || biases.len() != n_rows * (k_dim / group_size)
        {
            return Err(KernelError::InvalidInput(
                "weight/scales/biases length mismatch".to_string(),
            ));
        }
        let wb = self.buffer(std::mem::size_of_val(weight) as u64);
        let sb = self.buffer(std::mem::size_of_val(scales) as u64);
        let bb = self.buffer(std::mem::size_of_val(biases) as u64);
        let xb = self.buffer(std::mem::size_of_val(x) as u64);
        let yb = self.buffer((n_rows * 2) as u64);
        unsafe {
            std::ptr::copy_nonoverlapping(weight.as_ptr(), wb.contents() as *mut u32, weight.len());
            std::ptr::copy_nonoverlapping(scales.as_ptr(), sb.contents() as *mut u16, scales.len());
            std::ptr::copy_nonoverlapping(biases.as_ptr(), bb.contents() as *mut u16, biases.len());
            std::ptr::copy_nonoverlapping(x.as_ptr(), xb.contents() as *mut u16, x.len());
        }
        self.gemv_buffers(&xb, &wb, &sb, &bb, &yb, n_rows, k_dim, group_size)?;
        let src = yb.contents() as *const u16;
        let mut out = vec![0u16; n_rows];
        unsafe {
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n_rows);
        }
        Ok(out)
    }

    /// Same math as [`Self::gemv`] on caller-owned Metal buffers, so a loaded
    /// checkpoint does not copy weights per call.
    pub fn gemv_buffers(
        &self,
        x: &Buffer,
        w: &Buffer,
        s: &Buffer,
        b: &Buffer,
        y: &Buffer,
        n_rows: usize,
        k_dim: usize,
        group_size: usize,
    ) -> Result<(), KernelError> {
        if k_dim % 8 != 0 || k_dim % group_size != 0 {
            return Err(KernelError::InvalidInput(
                "k_dim must be a multiple of 8 and group_size".to_string(),
            ));
        }
        let n_bytes = (n_rows as u32).to_le_bytes();
        let k_bytes = (k_dim as u32).to_le_bytes();
        let g_bytes = (group_size as u32).to_le_bytes();

        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gemv_pipeline);
        encoder.set_buffer(0, Some(w), 0);
        encoder.set_buffer(1, Some(s), 0);
        encoder.set_buffer(2, Some(b), 0);
        encoder.set_buffer(3, Some(x), 0);
        encoder.set_buffer(4, Some(y), 0);
        encoder.set_bytes(5, size_of::<u32>() as u64, n_bytes.as_ptr() as *const c_void);
        encoder.set_bytes(6, size_of::<u32>() as u64, k_bytes.as_ptr() as *const c_void);
        encoder.set_bytes(7, size_of::<u32>() as u64, g_bytes.as_ptr() as *const c_void);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(n_rows as u64, 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    pub fn alloc_u16(&self, count: usize) -> Result<Buffer, KernelError> {
        Ok(self.device.new_buffer(
            (count * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        ))
    }
}
