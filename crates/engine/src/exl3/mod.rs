//! EXL3 CPU decode path (T1.3a/b): trellis unpack, codebook, perm, signs,
//! Hadamard, reconstruct, naive linear forward. Fresh ports of the reference
//! algorithm (not vendored code).

pub mod codebook;
pub mod decode;
pub mod forward;
pub mod hadamard;
pub mod perm;
pub mod reconstruct;
pub mod signs;
pub mod trellis;

pub use codebook::{CodebookMode, decode_3inst};
pub use decode::{decode_codewords, decode_packed_tile};
pub use forward::{Exl3Linear, linear_forward_reconstruct};
pub use hadamard::{HAD_DIM, HAD_SCALE, had_r_128, hadamard_128};
pub use perm::{kernel_order_to_row_major, tensor_core_perm, tensor_core_perm_inverse};
pub use reconstruct::{
    TrellisView, reconstruct_inner, reconstruct_inner_slab, reconstruct_public_weights,
    trellis_bytes_as_u16,
};
pub use signs::{unpack_sign_bitfield, unpack_signs_or_pass};
pub use trellis::{pack_trellis_tile, unpack_trellis_tile};
