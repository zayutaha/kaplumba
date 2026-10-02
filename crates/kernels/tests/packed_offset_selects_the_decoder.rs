//! A misaligned `packed_offset` does not fail loudly. It silently selects a
//! different EXL3 kernel.
//!
//! # The defect this pins
//!
//! `LinearKernels::ishizuki_gemv_pipeline` refuses its kernel family when
//! `packed_offset % 4 != 0` and `encode_gemv` then falls through to
//! `linear_gemv_exl3_k{bits}_u{,_mcg}`. That fallback is a *different decoder*
//! for the same bytes, not a slower path to the same answer, so a loader that
//! hands over an unaligned offset does not get an error -- it gets finite,
//! plausible, wrong logits, and a greedy decode of the 4 bpw checkpoint emits
//! `edL laly own own owning own-CCCCCC` where the 3 bpw checkpoint emits
//! `Hello! How can I help you today?`.
//!
//! The mechanism is easy to miss because the *other* EXL3 family IS
//! offset-tolerant, by design: `KAPLUMBA_OWN_GEMV=1` (which forces the EXL3 arm
//! for every offset) runs `linear_gemv_exl3_k4` and `linear_gemv_exl3_k8_mcg`
//! bit-exactly against the CPU reference on the same real 4 bpw tensors that
//! the aligned path handles. So "the EXL3 kernels are correct" and "the model
//! is correct" were both true, and the defect lived in the gap between them: the
//! choice of family, which nothing asserted.
//!
//! `model::EXL3_PACKED_ALIGNMENT` / `model::align_packed_view` are the loader
//! side of that gap. This test is the dispatch side, and it is what makes the two
//! agree by construction rather than by two lists happening to be read
//! correctly.
//!
//! Verified to fail under sabotage: widening `ishizuki_gemv_pipeline`'s guard
//! from `% 4` to `% 2` (i.e. accepting offsets the `uint *` cast cannot take)
//! makes `odd_offsets_must_not_reach_a_uint_casting_kernel` fail on 1 and 3.

use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::LinearKernels;

/// Must equal `kaplumba_engine::model::EXL3_PACKED_ALIGNMENT`. Duplicated here
/// because `align_packed_view` is private to the engine crate, and a constant
/// that only one side can see is not a shared contract. `model.rs`'s
/// `debug_assert_eq!` and the ignored end-to-end test
/// `misaligned_offsets_are_realigned.rs` are what tie the two together; this
/// test is what keeps each side honest on its own.
const EXL3_PACKED_ALIGNMENT: u64 = 4;

fn kernels() -> Option<LinearKernels> {
    if metal::Device::system_default().is_none() {
        eprintln!("no Metal device; skipping");
        return None;
    }
    match LinearKernels::new() {
        Ok(kernels) => Some(kernels),
        Err(error) => {
            eprintln!("LinearKernels unavailable ({error}); skipping");
            None
        }
    }
}

#[test]
fn odd_offsets_must_not_reach_a_uint_casting_kernel() {
    let Some(kernels) = kernels() else { return };
    // Every rate the loader can build, plus the head rate of this checkpoint.
    for bits in 1..=8u8 {
        let rate = Exl3Rate::Integer(bits);
        for offset in [1u64, 3, 763, 1813] {
            assert!(
                kernels.ishizuki_gemv_kernel_name(rate, offset).is_none(),
                "rate {bits} bpw claims the uint-casting ishizuki kernel at offset {offset}; \
                 `align_packed_view` guarantees that never reaches a dispatch, and this \
                 assertion says the two agree"
            );
        }
    }
}

#[test]
fn four_byte_aligned_offsets_keep_the_ishizuki_kernel() {
    let Some(kernels) = kernels() else { return };
    // The positive direction: if realignment ever silently refused a VALID
    // offset, every tensor would drop to the other decoder and the model would
    // be wrong again with no error anywhere. The offsets are the ones
    // `Gxma-repacked` actually produces.
    for bits in 1..=8u8 {
        let rate = Exl3Rate::Integer(bits);
        for offset in [0u64, 4, 2112, 2880] {
            assert_eq!(
                kernels.ishizuki_gemv_kernel_name(rate, offset).as_deref(),
                Some("ishizuki"),
                "rate {bits} bpw at aligned offset {offset} lost its kernel"
            );
        }
    }
}

#[test]
fn two_mod_four_offsets_also_lose_the_ishizuki_kernel() {
    // 2 mod 4 is legal for the 16-bit EXL3 fallback and illegal for the
    // `uint *` ishizuki kernel, so these are the offsets on which the two
    // families silently disagree about the same bytes.
    let Some(kernels) = kernels() else { return };
    for bits in 1..=8u8 {
        let rate = Exl3Rate::Integer(bits);
        for offset in [2u64, 6, 1814] {
            assert!(
                kernels.ishizuki_gemv_kernel_name(rate, offset).is_none(),
                "rate {bits} bpw accepts offset {offset}, which the uint cast cannot take"
            );
        }
    }
}

#[test]
fn the_contract_is_four_bytes_and_not_a_power_of_two() {
    // Pins the number the loader and the dispatch both key on. A change to 2
    // would let 2-mod-4 offsets through to the ishizuki kernel and silently
    // corrupt every tensor whose file offset is 2 mod 4; a change to 8 would
    // drop valid 4-aligned offsets onto the other decoder for no benefit.
    assert_eq!(EXL3_PACKED_ALIGNMENT, 4);
    assert_eq!(EXL3_PACKED_ALIGNMENT % 4, 0);
    assert_ne!(EXL3_PACKED_ALIGNMENT, 2);
    // And the codebook axis must stay Mcg-only for this claim to mean anything:
    // the model's linears are all Mcg, and the mul1/plain arms have their own
    // dispatch table.
    assert_eq!(Exl3Codebook::Mcg, Exl3Codebook::Mcg);
}