//! Kaplumba engine core: pure Rust + raw Metal (no MLX, no Python).
//!
//! T1.0: workspace + Metal device smoke. T1.1: EXL3 loader. T1.2: tokenizer.
//! T1.3a: EXL3 CPU decode primitives (trellis/codebook/perm/signs/hadamard).
//! T1.3b: reconstruct + streaming linear forward.
//! T1.3c: Gemma 4 text forward (naive CPU golden path) + greedy decode.

pub mod exl3;
pub mod gemma4;
pub mod load;
pub mod metal_ctx;
pub mod tokenizer;

pub use gemma4::{
    Gemma4Model, GemmaCfg, GemmaError, LayerCfg, LayerKind, RawTensor, RopeCfg, TensorSource,
};
pub use load::{
    DType, Exl3Model, LoadError, ModelConfig, RopeParams, TensorInfo, inspect_dir, load_exl3,
};
pub use metal_ctx::{MetalContext, MetalError, clear_buffer_u32};
pub use tokenizer::{GemmaSpecialIds, GemmaTokenizer, TokenizerError};
