//! Kaplumba engine core: pure Rust + raw Metal (no MLX, no Python).
//!
//! T1.0: workspace + Metal device smoke. Loader/kernels land in T1.1+.

pub mod metal_ctx;

pub use metal_ctx::{clear_buffer_u32, MetalContext, MetalError};
