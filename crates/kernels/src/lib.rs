//! Raw Metal kernels for the kaplumba engine.
//!
//! T2.1 provides the EXL3 decode-GEMV and the two 128-point Hadamard passes
//! that make up one `linear_forward_reconstruct` layer
//! (`had_in` → `gemv` → `had_out`). MSL is compiled at runtime from
//! [`exl3::MSL_SOURCE`]-equivalent embedded source; the host wrapper owns the
//! pipelines, the per-`k` pair tables and reusable shared-memory scratch.
//!
//! The kernels are numerically matched to the CPU reference path: the Hadamard
//! passes are bit-exact, the GEMV matches within f32 accumulation-order
//! tolerance (verified against `linear_forward_reconstruct` in the engine's
//! tests).
//!
//! Not here yet (by design, see `spec/rust-metal-engine`): the prefill GEMM
//! (T3.2), KV-cache attention (T4), and any sampler work.

pub mod exl3;

pub use exl3::{DeviceSpan, Exl3Gemv, Exl3Shape, KernelError};
