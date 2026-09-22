//! Metal device/queue context and minimal GPU ops for T1.0 smoke tests.

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

#[derive(Debug, thiserror::Error)]
pub enum MetalError {
    #[error("no Metal device available (not macOS / no GPU?)")]
    NoDevice,
    #[error("failed to create pipeline: {0}")]
    Pipeline(String),
    #[error("failed to create library: {0}")]
    Library(String),
    #[error("command buffer did not complete successfully")]
    CommandFailed,
}

/// Owns the default Metal device and a command queue for the engine process.
pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
}

impl MetalContext {
    pub fn new() -> Result<Self, MetalError> {
        let device = Device::system_default().ok_or(MetalError::NoDevice)?;
        let queue = device.new_command_queue();
        Ok(Self { device, queue })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// Shared-mode buffer of `len` bytes (host-visible; UMA on Apple Silicon).
    pub fn new_shared_buffer(&self, len: usize) -> Buffer {
        self.device
            .new_buffer(len as u64, MTLResourceOptions::StorageModeShared)
    }

    /// GPU fill of a u32 buffer via compute shader (smoke: prove encode+commit+wait).
    pub fn fill_u32(&self, buf: &Buffer, value: u32) -> Result<(), MetalError> {
        let src = self.device.new_buffer_with_data(
            value.to_le_bytes().as_ptr() as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let pipeline = fill_pipeline(&self.device)?;
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(buf), 0);
        enc.set_buffer(1, Some(&src), 0);
        let elems = (buf.length() / 4).min(1 << 20).max(1);
        let width = pipeline.thread_execution_width().min(elems);
        enc.dispatch_threads(MTLSize::new(elems, 1, 1), MTLSize::new(width, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        if cb.status() != MTLCommandBufferStatus::Completed {
            return Err(MetalError::CommandFailed);
        }
        Ok(())
    }
}

const FILL_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void fill_u32(device uint *out [[buffer(0)]],
                     constant uint &v [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {
    out[id] = v;
}
"#;

fn fill_pipeline(device: &Device) -> Result<ComputePipelineState, MetalError> {
    let opts = CompileOptions::new();
    let lib = device
        .new_library_with_source(FILL_MSL, &opts)
        .map_err(|e| MetalError::Library(e))?;
    let func = lib
        .get_function("fill_u32", None)
        .map_err(|e| MetalError::Pipeline(format!("get_function fill_u32: {e}")))?;
    device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| MetalError::Pipeline(format!("{e}")))
}

/// Allocate a shared u32 buffer, GPU-fill with `value`, wait, return host copy.
pub fn clear_buffer_u32(
    ctx: &MetalContext,
    elems: usize,
    value: u32,
) -> Result<Vec<u32>, MetalError> {
    let buf = ctx.new_shared_buffer(elems * 4);
    ctx.fill_u32(&buf, value)?;
    let bytes = buf.contents();
    let mut out = vec![0u32; elems];
    unsafe {
        std::ptr::copy_nonoverlapping(bytes as *const u8, out.as_mut_ptr() as *mut u8, elems * 4);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_fill_buffer_on_gpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::NoDevice) => {
                eprintln!("skipping: no Metal device");
                return;
            }
            Err(e) => panic!("metal init failed: {e}"),
        };
        let n = 4096;
        let got = clear_buffer_u32(&ctx, n, 0xDEAD_BEEF).expect("fill");
        assert_eq!(got.len(), n);
        assert!(
            got.iter().all(|&x| x == 0xDEAD_BEEF),
            "buffer not fully filled"
        );
    }

    #[test]
    fn metal_context_creates() {
        match MetalContext::new() {
            Ok(_) => {}
            Err(MetalError::NoDevice) => {
                eprintln!("no device; context test is environment-dependent")
            }
            Err(e) => panic!("unexpected metal error: {e}"),
        }
    }
}
