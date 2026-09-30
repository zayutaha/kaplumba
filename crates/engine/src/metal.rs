use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};
use thiserror::Error;

pub const DEFAULT_RESIDENT_LIMIT_BYTES: u64 = 18_000_000_000;

/// The resident-weight ceiling, overridable with `KAPLUMBA_RESIDENT_LIMIT`.
///
/// This exists because the ceiling is a *policy* number -- how much of a 24 GiB
/// box the loader is willing to fill -- and testing a policy needs to move it.
/// The 4 bpw / 8 bpw-head checkpoint needs 18,046,374,518 bytes resident, which is
/// 46 MB over the default, and there is no other way to ask "does the failure go
/// away when the whole model is in the arena" without editing a constant and
/// rebuilding.
///
/// Read once, at first use, and the override is rejected rather than clamped: a
/// typo'd value that silently becomes the default is worse than a load that
/// refuses to start.
pub fn resident_limit_bytes() -> u64 {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| match std::env::var("KAPLUMBA_RESIDENT_LIMIT") {
        Ok(raw) => match raw.trim().parse::<u64>() {
            Ok(value) if value >= 1_000_000_000 => value,
            Ok(value) => panic!(
                "KAPLUMBA_RESIDENT_LIMIT is {value} bytes, which is below the 1 GB floor; \
                 a value that small cannot hold a model and is almost certainly a unit \
                 mistake (did you mean MB or GB?)"
            ),
            Err(_) => panic!("KAPLUMBA_RESIDENT_LIMIT is {raw:?}, which is not a byte count"),
        },
        Err(_) => DEFAULT_RESIDENT_LIMIT_BYTES,
    })
}
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
