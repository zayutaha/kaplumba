//! Kaplumba engine core: pure Rust + raw Metal (no MLX, no Python).
//!
//! T1.0: workspace + Metal device smoke. T1.1: EXL3 loader. T1.2: tokenizer.

pub mod load;
pub mod metal_ctx;
pub mod tokenizer;

pub use load::{inspect_dir, load_exl3, DType, Exl3Model, LoadError, ModelConfig, TensorInfo};
pub use metal_ctx::{clear_buffer_u32, MetalContext, MetalError};
pub use tokenizer::{GemmaSpecialIds, GemmaTokenizer, TokenizerError};
