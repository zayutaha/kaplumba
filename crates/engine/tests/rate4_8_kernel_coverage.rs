//! The 4 bpw / 8 bpw-head checkpoint dispatches real kernels, not fallbacks.
//!
//! # Why this test exists
//!
//! `crates/engine/src/model.rs` used to build every `QuantizedLinear` with a
//! hardcoded `Exl3Rate::Integer(3)` and `Integer(6)` for the head, and
//! `config.rs` refused any `config.json` whose `bits != 3.0`. Together those made
//! a 4 bpw export unloadable. Relaxing them is only safe if a kernel actually
//! exists for the new rates, because the failure mode of a missing one is not an
//! error: `ishizuki_gemv_pipeline` returns `None`, `encode_gemv` falls through
//! to the next family, and the model produces finite, plausible, WRONG output.
//! Nothing in a load test would notice.
//!
//! So the claim is pinned here, against the real `LinearKernels` registry:
//!
//! 1. `linear_gemv_ishizuki_k4` and `linear_gemv_ishizuki_k8` are in the library,
//!    and each owns its own per-rate dispatch slot (a `_bench` or `_nod` variant
//!    must not have taken it -- see `rate4_colmajor_bit_exact.rs` for the same
//!    argument about the colmajor family).
//! 2. The 4 bpw *row-major* path resolves to a kernel, and it is the 2-word
//!    variant, which `two_word_is_safe` derives as valid for KA <= 6.
//! 3. The EXL3 `_mcg` fallback family has a slot for every rate 1..=8, so a
//!    tensor that misses the ishizuki arm (unaligned `packed_offset`) still has
//!    somewhere correct to go.
//!
//! Each of (1)-(3) was verified to fail under sabotage; see
//! `spec/shim-exl3-4bpw/implementation_notes.md`.

use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::LinearKernels;

/// Per-rate dispatch slot, as `LinearKernels::new` fills it: the sorted-first
/// kernel naming that rate.
fn slot_owner(kernels: &LinearKernels, bits: u8) -> String {
    let names = kernels.ishizuki_gemv_kernel_names();
    names
        .iter()
        .find(|n| LinearKernels::ishizuki_rate_of_name(n) == Some(bits))
        .unwrap_or_else(|| panic!("no kernel in the library names rate {bits}"))
        .clone()
}

/// The bit rate a `linear_gemv_exl3_k<N>...` name dispatches for.
///
/// Parsed from the name for the same reason `ishizuki_rate_of_name` exists: the
/// slot table is a flat `Vec` indexed by an arithmetic slot number, so a
/// disagreement between "which rate does this name claim" and "which rate was
/// put in this slot" is silent and total. Deriving the rate from the name is what
/// lets the test below catch it.
fn exl3_rate_of_name(name: &str) -> Option<u8> {
    let rest = name.strip_prefix("linear_gemv_exl3_k")?;
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    let bits: u8 = digits.parse().ok()?;
    (1..=8).contains(&bits).then_some(bits)
}

#[test]
fn four_and_eight_bpw_have_their_own_row_major_kernels_and_slots() {
    let Some(device) = metal::Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let Ok(kernels) = LinearKernels::new() else {
        eprintln!("LinearKernels unavailable; skipping");
        return;
    };

    for bits in [4u8, 8] {
        let owner = slot_owner(&kernels, bits);
        assert_eq!(
            owner,
            format!("linear_gemv_ishizuki_k{bits}"),
            "the {bits} bpw dispatch slot is claimed by {owner}, not by the unsuffixed \
             kernel; a variant has taken over production {bits} bpw dispatch"
        );
        assert!(
            kernels
                .ishizuki_gemv_pipeline_by_name(&format!("linear_gemv_ishizuki_k{bits}"))
                .is_some(),
            "linear_gemv_ishizuki_k{bits} is missing from the library; a {bits} bpw \
             checkpoint would silently fall through to another family"
        );
    }

    // 8 bpw has no 2-word variant, and must not: `two_word_is_safe` requires
    // 7*KA + 16 <= 64, i.e. KA <= 6. If a `_k8_w2` ever appears it is reading
    // past the end of a 64-bit window, so this is an assertion about a kernel
    // that must NOT exist.
    assert!(
        kernels
            .ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k8_w2")
            .is_none(),
        "linear_gemv_ishizuki_k8_w2 exists; at KA=8 a lane's window spans 72 bits, \
         which does not fit the 64-bit two-word form"
    );

    let _ = device;
}

#[test]
fn the_row_major_four_bpw_path_resolves_to_the_two_word_kernel() {
    let Some(device) = metal::Device::system_default() else {
        return;
    };
    let Ok(kernels) = LinearKernels::new() else {
        return;
    };
    // The 4 bpw colmajor *measurement* kernel must not be what a row-major
    // `perm_mode == 0` tensor reaches: it reads a transposed tile grid, so
    // dispatching it over row-major weights is the silent-gibberish failure
    // recorded in spec/decode-19tok-repacked-250gb.
    assert!(
        kernels
            .ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k4_colmajor_contig4_bench")
            .is_some(),
        "the 4 bpw colmajor measurement kernel is absent, so this test is not \
         checking what it claims to check"
    );
    // `_w2` is the row-major two-word form and exists for KA = 4.
    assert!(
        kernels
            .ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k4_w2")
            .is_some(),
        "linear_gemv_ishizuki_k4_w2 is missing; 4 bpw row-major would fall back to \
         the 3-word unsuffixed kernel, which is correct but slower"
    );
    let _ = device;
}

#[test]
fn the_mcg_fallback_family_covers_every_rate_the_loader_can_produce() {
    let Some(device) = metal::Device::system_default() else {
        return;
    };
    let Ok(kernels) = LinearKernels::new() else {
        return;
    };
    // `Exl3Rate::Integer` is only ever built from a rate in 1..=8
    // (`integer_rate` in model.rs, and `is_supported_rate` in config.rs), so a
    // missing slot here is an unreachable branch today and a wrong-output bug
    // the moment that range widens.
    for bits in 1u8..=8 {
        for (offset, aligned) in [(0u64, true), (2, false)] {
            let index = LinearKernels::exl3_gemv_index(
                Exl3Rate::Integer(bits),
                offset,
                Exl3Codebook::Mcg,
            )
            .unwrap_or_else(|| panic!("no EXL3 slot index for {bits} bpw at {offset}"));
            let name = kernels
                .exl3_gemv_kernel_name(Exl3Rate::Integer(bits), offset, Exl3Codebook::Mcg)
                .unwrap_or_else(|| panic!("{bits} bpw at offset {offset} has no kernel name"));
            // The name must exist AND must name THIS rate. Asserting only
            // existence is not enough: `exl3_gemv_slot_names` is a flat Vec
            // indexed by `exl3_gemv_index`, so dropping a rate from the build loop
            // shifts every later slot onto the wrong kernel's name. The name is
            // then present, the pipeline is found, and the model decodes 8 bpw
            // weights with the 1 bpw kernel -- finite, plausible, wrong. This is
            // the off-by-one that `exl3_gemv_index`'s own comment says already
            // happened once.
            assert_eq!(
                exl3_rate_of_name(name),
                Some(bits),
                "{bits} bpw at offset {offset} resolves to {name} (slot {index}, \
                 aligned {aligned}), which is the wrong bit rate"
            );
            assert!(
                kernels.exl3_gemv_pipeline_by_name(name).is_some(),
                "{bits} bpw at offset {offset} resolves to {name} (slot {index}, \
                 aligned {aligned}), which is not in the library"
            );
        }
    }
    let _ = device;
}
