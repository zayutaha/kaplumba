//! Kaplumba engine core: pure Rust + raw Metal (no MLX, no Python).
//!
//! T1.0: workspace + Metal device smoke. T1.1: EXL3 loader. T1.2: tokenizer.
//! T1.3a: EXL3 CPU decode primitives (trellis/codebook/perm/signs/hadamard).

pub mod exl3;
pub mod load;
pub mod metal_ctx;
pub mod tokenizer;

pub use load::{DType, Exl3Model, LoadError, ModelConfig, TensorInfo, inspect_dir, load_exl3};
pub use metal_ctx::{MetalContext, MetalError, clear_buffer_u32};
pub use tokenizer::{GemmaSpecialIds, GemmaTokenizer, TokenizerError};
