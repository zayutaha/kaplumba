//! Raw Metal kernels for the kaplumba engine.
//!
//! T2.1 provides the EXL3 decode-GEMV and the two 128-point Hadamard passes
//! that make up one `linear_forward_reconstruct` layer
//! (`had_in` → `gemv` → `had_out`). MSL is compiled at runtime from the
//! embedded sources; the host wrappers own the pipelines, the per-`k` pair
//! tables and reusable shared-memory scratch.
//!
//! T3.5/T3.7 add the attention pass ([`Exl3Attention`]): a flash-style online
//! softmax whose entire working set is registers, so prefill memory stays
//! bounded by the KV cache rather than by an O(L²) score matrix.
//!
//! The kernels are numerically matched to the CPU reference path: the Hadamard
//! passes are bit-exact, the GEMV matches within f32 accumulation-order
//! tolerance, and attention within a couple of f16 ulps (all verified against
//! the engine's CPU reference in tests).
//!
//! Not here yet (by design, see `spec/rust-metal-engine`): the prefill GEMM
//! (T3.2) and any sampler work.

pub mod attn;
pub mod exl3;
pub mod fuse;

pub use attn::{ATTN_ROWS, AttnParams, Exl3Attention};
pub use exl3::{DeviceSpan, Exl3Gemv, Exl3Shape, KernelError};
pub use fuse::Fuse;
