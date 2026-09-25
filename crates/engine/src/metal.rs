use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};
use thiserror::Error;

pub const DEFAULT_RESIDENT_LIMIT_BYTES: u64 = 18_000_000_000;
pub const DEFAULT_SEGMENT_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum MetalError {
    #[error("Metal system device is unavailable")]
    DeviceUnavailable,
    #[error("invalid Metal buffer size {0}")]
    InvalidBufferSize(u64),
    #[error("Metal buffer allocation returned {actual} bytes for requested {requested}")]
    AllocationSizeMismatch { requested: u64, actual: u64 },
    #[error("Metal shared buffer has no accessible contents")]
    ContentsUnavailable,
}

pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    max_buffer_length: u64,
}

impl MetalContext {
    pub fn new() -> Result<Self, MetalError> {
        let device = Device::system_default().ok_or(MetalError::DeviceUnavailable)?;
        let queue = device.new_command_queue();
        let max_buffer_length = device.max_buffer_length();
        if max_buffer_length == 0 {
            return Err(MetalError::InvalidBufferSize(0));
        }
        Ok(Self {
            device,
            queue,
            max_buffer_length,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    pub fn max_buffer_length(&self) -> u64 {
        self.max_buffer_length
    }

    pub fn allocate_shared(&self, length: u64) -> Result<Buffer, MetalError> {
        if length == 0 || length > self.max_buffer_length {
            return Err(MetalError::InvalidBufferSize(length));
        }
        let buffer = self
            .device
            .new_buffer(length, MTLResourceOptions::StorageModeShared);
        let actual = buffer.length();
        if actual != length {
            return Err(MetalError::AllocationSizeMismatch {
                requested: length,
                actual,
            });
        }
        Ok(buffer)
    }
}

pub fn shared_contents(buffer: &Buffer) -> Result<*mut std::ffi::c_void, MetalError> {
    let contents = buffer.contents();
    if contents.is_null() {
        Err(MetalError::ContentsUnavailable)
    } else {
        Ok(contents)
    }
}
