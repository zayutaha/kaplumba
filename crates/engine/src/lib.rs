//! Kaplumba engine core: pure Rust + raw Metal (no MLX, no Python).
//!
//! T1.0: workspace + Metal device smoke. T1.1: EXL3 loader. T1.2: tokenizer.
//! T1.3a: EXL3 CPU decode primitives (trellis/codebook/perm/signs/hadamard).
//! T1.3b: reconstruct + streaming linear forward.
//! T1.3c: Gemma 4 text forward (naive CPU golden path) + greedy decode.
//! T1.4: streaming generation loop + host sampler (bench/CLI plumbing).

pub mod exl3;
pub mod gemma4;
pub mod generate;
pub mod load;
pub mod metal_ctx;
pub mod sampler;
#[cfg(test)]
pub(crate) mod testutil;
pub mod tokenizer;

pub use gemma4::{
    Gemma4Model, GemmaCfg, GemmaError, LayerCfg, LayerKind, RawTensor, RopeCfg, TensorSource,
};
pub use generate::{FinishReason, GenParams, GenStats, generate};
pub use load::{
    DType, Exl3Model, LoadError, ModelConfig, RopeParams, TensorInfo, inspect_dir, load_exl3,
};
pub use metal_ctx::{MetalContext, MetalError, clear_buffer_u32};
pub use sampler::{Pcg32, SampleParams, sample};
pub use tokenizer::{GemmaSpecialIds, GemmaTokenizer, TokenizerError};
