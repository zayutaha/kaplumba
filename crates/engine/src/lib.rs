//! Kaplumba engine core: pure Rust + raw Metal (no MLX, no Python).
//!
//! T1.0: workspace + Metal device smoke. T1.1: EXL3 artifact loader.

pub mod load;
pub mod metal_ctx;

pub use load::{inspect_dir, load_exl3, DType, Exl3Model, LoadError, ModelConfig, TensorInfo};
pub use metal_ctx::{clear_buffer_u32, MetalContext, MetalError};
