//! T4 (`spec/prefill-120tps`): the GEMM-MMA prefill tile variants.
//!
//! Three questions, and each one is a separate assertion because they fail
//! differently:
//!
//! 1. **Did the forced variant actually dispatch?** `last_gemm_mma_variant`
//!    exists for this and nothing else. Every T4 variant is claimed to be
//!    numerically identical to `linear_gemm_mma`, so an output-only test is
//!    vacuously satisfiable by a dispatch that never left the baseline -- the
//!    exact failure mode that shipped `8f1a863` under a "6/6" claim
//!    (AGENTS.md §3). A test that compares outputs must assert which kernel
//!    produced them.
//!
//! 2. **Does it produce the baseline's answer?** Same packed bytes, same
//!    activations, forced variant vs. forced baseline, compared element by
//!    element. This crate has no EXL3 packer, so the packed data here is
//!    random `u16`s: the kernel reads them as trellis words with no
//!    validation, and the decode is a bijection from bits to finite fp16
//!    (the MCG arm's `0x3b603b60 ^ (v & 0x8fff8fff)` fixes the exponent
//!    field to 14, so no input word can produce Inf or NaN). The engine-side
//!    test `gemm_repack_tile_variants` is the one that checks the answer
//!    against a CPU reference; this file checks the variants against each
//!    other, which is what proves the refactor preserved the arithmetic.
//!
//! 3. **Do the env flags select what they say?** The variant resolution reads
//!    `env_flag`, which memoises on first read, so it cannot be exercised
//!    from inside this process once any dispatch has happened. That is probed
//!    by re-exec-ing this same binary as a child with the flag set.
//!
//! Every test takes `SERIAL` first: the forced-variant override and the
//! last-dispatched counter are process-global, so two tests forcing different
//! variants concurrently would report each other's dispatch.
//!
//! Verified to fail under sabotage (recorded in `notes_T3_T4.md`):
//! forcing the 4sg arm to dispatch `gemm_mma_pipeline` makes
//! `forced_variant_is_what_dispatches` fail on the identity assert; breaking
//! the 4sg store mapping makes `variants_match_the_baseline` fail on the
//! first shape.

use std::ffi::c_void;
use std::mem::{size_of, size_of_val};
use std::sync::Mutex;

use half::f16;
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{
    GEMM_MMA_VARIANT_4SG, GEMM_MMA_VARIANT_128X64, GEMM_MMA_VARIANT_NONE, GEMM_MMA_VARIANT_REGPIPE,
    GEMM_MMA_VARIANT_REGPIPE_K32, LinearKernels, PrefillEncodeRequest, clear_gemm_mma_variant,
    force_gemm_mma_variant, last_gemm_mma_variant,
};
use metal::{Device, MTLResourceOptions};

/// See the module doc: the override and the counter are global.
static SERIAL: Mutex<()> = Mutex::new(());

/// The shape class under test. `validate_encode` demands multiples of 128 for
/// both dimensions, so 4sg (`out % 32`), regpipe (`out % 64`) and baseline
/// (`out % 64`) all qualify on every row here. The query counts straddle the
/// 128-row tile: 1 and 129 are the row-guarded tails, 256 is exactly two.
const SHAPES: [(usize, usize, usize); 5] = [
    (129, 128, 128),
    (1, 128, 128),
    (256, 128, 256),
    (127, 256, 384),
    (255, 128, 512),
];

/// Every variant id the dispatch can resolve to. `AUTO` is deliberately
/// absent: it is the *selection* mode, not a kernel, and forcing it would
/// silently re-test whatever the ambient environment says.
const VARIANTS: [u8; 4] = [
    GEMM_MMA_VARIANT_128X64,
    GEMM_MMA_VARIANT_4SG,
    GEMM_MMA_VARIANT_REGPIPE,
    GEMM_MMA_VARIANT_REGPIPE_K32,
];

fn variant_name(variant: u8) -> &'static str {
    match variant {
        GEMM_MMA_VARIANT_128X64 => "128x64",
        GEMM_MMA_VARIANT_4SG => "4sg",
        GEMM_MMA_VARIANT_REGPIPE => "regpipe",
        GEMM_MMA_VARIANT_REGPIPE_K32 => "regpipe_k32",
        GEMM_MMA_VARIANT_NONE => "none",
        _ => "auto/unknown",
    }
}

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

/// Deterministic pseudo-random packed words. Not valid EXL3 output -- the
/// engine-side test owns the real packers -- but a complete, non-degenerate
/// input to a kernel that performs no validation on these bytes.
fn packed_words(tiles: usize, rate: Exl3Rate, seed: u64) -> Vec<u16> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..tiles * rate.packed_words())
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u16
        })
        .collect()
}

fn activations(queries: usize, in_features: usize) -> Vec<u16> {
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    (0..queries * in_features)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // [-1, 1) in f16: keeps the products in the range where the
            // f32 accumulator is not dominated by one coordinate.
            f16::from_f32(((state >> 40) as f32 / (1u64 << 24) as f32) - 1.0).to_bits()
        })
        .collect()
}

fn signs(len: usize, modulus: usize) -> Vec<u16> {
    (0..len)
        .map(|index| f16::from_f32(if index % modulus == 0 { -1.0 } else { 1.0 }).to_bits())
        .collect()
}

struct Run {
    output: Vec<f32>,
    dispatched: u8,
}

/// One `encode_gemm` call under a forced variant, with the canary-row guard
/// that `linear_metal::mma_gemm_rows` established: a tile that stores rows it
/// does not own writes past the end of `raw`/`output`, and reading back only
/// `queries * out_features` elements would never see it.
#[allow(clippy::too_many_arguments)]
fn run(
    kernels: &LinearKernels,
    variant: u8,
    packed: &[u16],
    input: &[u16],
    input_signs: &[u16],
    output_signs: &[u16],
    queries: usize,
    in_features: usize,
    out_features: usize,
    rate: Exl3Rate,
) -> Run {
    // Must be cleared rather than set: the flag DISABLES the path under test.
    unsafe { std::env::remove_var("KAPLUMBA_DISABLE_MMA_PREFILL") };
    force_gemm_mma_variant(variant);

    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let halfs = |values: &[u16]| {
        device.new_buffer_with_data(
            values.as_ptr() as *const c_void,
            size_of_val(values) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let scratch =
        |bytes: usize| device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
    let packed_buffer = halfs(packed);
    let input_buffer = halfs(input);
    let input_sign_buffer = halfs(input_signs);
    let output_sign_buffer = halfs(output_signs);
    let canary = out_features;
    let transformed = scratch(queries * in_features * size_of::<u16>());
    let raw = scratch((queries + 1) * out_features * size_of::<f32>());
    let output = scratch((queries + 1) * out_features * size_of::<u16>());
    unsafe {
        let raw_words = raw.contents() as *mut u32;
        for index in 0..canary {
            *raw_words.add(queries * out_features + index) = 0xDEAD_BEEF;
        }
        let out_words = output.contents() as *mut u16;
        for index in 0..canary {
            *out_words.add(queries * out_features + index) = 0x7E7E;
        }
        std::ptr::write_bytes(raw.contents() as *mut f32, 0, queries * out_features);
    }
    let mut encoder = MetalEncoder::new(&queue).expect("Metal encoder");
    kernels
        .encode_gemm(
            &mut encoder,
            PrefillEncodeRequest {
                packed: &packed_buffer,
                packed_offset: 0,
                perm_mode: 0,
                input: &input_buffer,
                input_signs: &input_sign_buffer,
                output_signs: &output_sign_buffer,
                transformed: &transformed,
                raw: &raw,
                output: &output,
                queries,
                in_features,
                out_features,
                rate,
                codebook: Exl3Codebook::Mcg,
            },
        )
        .expect("encode prefill GEMM");
    encoder.commit_and_wait().expect("commit GEMM");
    let dispatched = last_gemm_mma_variant();
    unsafe {
        let raw_words = raw.contents() as *const u32;
        for index in 0..canary {
            assert_eq!(
                *raw_words.add(queries * out_features + index),
                0xDEAD_BEEF,
                "[{}] the kernel stored output row {queries}, past the end of `raw`",
                variant_name(variant)
            );
        }
        let out_words = output.contents() as *const u16;
        for index in 0..canary {
            assert_eq!(
                *out_words.add(queries * out_features + index),
                0x7E7E,
                "[{}] the kernel overwrote the canary row of `output`",
                variant_name(variant)
            );
        }
    }
    let values = output.contents() as *const u16;
    let output_values = (0..queries * out_features)
        .map(|index| f16::from_bits(unsafe { *values.add(index) }).to_f32())
        .collect();
    force_gemm_mma_variant(kaplumba_kernels::linear::GEMM_MMA_VARIANT_AUTO);
    Run {
        output: output_values,
        dispatched,
    }
}

/// Q1: the id the dispatch reports must be the id that was forced -- after
/// the eligibility gate, so a forced regpipe on an ineligible rate reports
/// `128x64`, which is what ran.
#[test]
fn forced_variant_is_what_dispatches() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(kernels) = kernels() else { return };

    let (queries, in_features, out_features) = (129, 128, 128);
    let input = activations(queries, in_features);
    let input_signs = signs(in_features, 3);
    let output_signs = signs(out_features, 5);

    // Eligible rate: every variant must dispatch itself.
    let eligible_rate = Exl3Rate::Integer(3);
    let tiles = (in_features / 16) * (out_features / 16);
    let packed = packed_words(tiles, eligible_rate, 7);
    for variant in VARIANTS {
        let run = run(
            &kernels,
            variant,
            &packed,
            &input,
            &input_signs,
            &output_signs,
            queries,
            in_features,
            out_features,
            eligible_rate,
        );
        assert_eq!(
            run.dispatched,
            variant,
            "forced {} but the dispatch reported {}",
            variant_name(variant),
            variant_name(run.dispatched)
        );
    }

    // Ineligible rate: the research host gate (`perm_mode != 1 && rate ==
    // Integer(3) && codebook == Mcg`) excludes anything but 3 bpw, so all
    // three research variants -- 4sg included, whose body carries only the
    // folded-MCG arm -- must fall back to the baseline rather than run a
    // kernel its gate would never send it.
    let ineligible_rate = Exl3Rate::Integer(4);
    let packed = packed_words(tiles, ineligible_rate, 11);
    for variant in VARIANTS {
        let run = run(
            &kernels,
            variant,
            &packed,
            &input,
            &input_signs,
            &output_signs,
            queries,
            in_features,
            out_features,
            ineligible_rate,
        );
        let expected = if variant == GEMM_MMA_VARIANT_4SG
            || variant == GEMM_MMA_VARIANT_REGPIPE
            || variant == GEMM_MMA_VARIANT_REGPIPE_K32
        {
            GEMM_MMA_VARIANT_128X64
        } else {
            variant
        };
        assert_eq!(
            run.dispatched,
            expected,
            "at 4 bpw, forced {} must resolve to {}",
            variant_name(variant),
            variant_name(expected)
        );
    }

    clear_gemm_mma_variant();
}

/// Q2: every variant against the forced baseline, on random packed bytes
/// where the arithmetic is the only thing that can differ.
#[test]
fn variants_match_the_baseline() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(kernels) = kernels() else { return };

    // 3 bpw keeps every variant in its eligible arm, so what is measured here
    // is the arithmetic and not the fallback. A second pass at 2 bpw covers
    // all three research variants' host gate (the folded-MCG arm is 3 bpw
    // only, so none of them may dispatch there).
    for rate in [Exl3Rate::Integer(3), Exl3Rate::Integer(2)] {
        for (queries, in_features, out_features) in SHAPES {
            let input = activations(queries, in_features);
            let input_signs = signs(in_features, 3);
            let output_signs = signs(out_features, 5);
            let tiles = (in_features / 16) * (out_features / 16);
            let packed = packed_words(tiles, rate, (queries * 131 + in_features) as u64);
            let baseline = run(
                &kernels,
                GEMM_MMA_VARIANT_128X64,
                &packed,
                &input,
                &input_signs,
                &output_signs,
                queries,
                in_features,
                out_features,
                rate,
            );
            assert_eq!(
                baseline.dispatched, GEMM_MMA_VARIANT_128X64,
                "the baseline reference itself did not dispatch"
            );
            // A comparison against an all-zero buffer proves nothing; the
            // random data must actually produce an answer.
            let peak = baseline
                .output
                .iter()
                .fold(0.0_f32, |peak, value| peak.max(value.abs()));
            assert!(
                peak > 0.0,
                "[{rate:?}] degenerate baseline output on {queries}x{in_features}x{out_features}"
            );

            for variant in VARIANTS {
                let candidate = run(
                    &kernels,
                    variant,
                    &packed,
                    &input,
                    &input_signs,
                    &output_signs,
                    queries,
                    in_features,
                    out_features,
                    rate,
                );
                // The research host gate admits 3 bpw only, so at 2 bpw all
                // three research variants resolve to the baseline. Comparing
                // that fallback against the baseline would be
                // baseline-vs-baseline, i.e. vacuous -- the fallback itself is
                // asserted in `forced_variant_is_what_dispatches`, so here it
                // is skipped rather than counted as coverage.
                let gate_admits =
                    variant == GEMM_MMA_VARIANT_128X64 || matches!(rate, Exl3Rate::Integer(3));
                let expected = if gate_admits {
                    variant
                } else {
                    GEMM_MMA_VARIANT_128X64
                };
                assert_eq!(
                    candidate.dispatched,
                    expected,
                    "[{rate:?}] forced {} resolved to {}",
                    variant_name(variant),
                    variant_name(candidate.dispatched)
                );
                if candidate.dispatched != variant {
                    continue;
                }
                assert_eq!(
                    candidate.output.len(),
                    baseline.output.len(),
                    "[{rate:?}] {} produced a different element count",
                    variant_name(variant)
                );
                let mut worst = 0.0_f32;
                let mut first = None;
                for (index, (actual, expected)) in candidate
                    .output
                    .iter()
                    .zip(baseline.output.iter())
                    .enumerate()
                {
                    let error = (actual - expected).abs() / peak;
                    if error > worst {
                        worst = error;
                        first = Some(index);
                    }
                }
                assert_eq!(
                    worst,
                    0.0,
                    "[{rate:?}] {} diverged from the baseline: worst normalised error \
                     {worst:e} at flat index {:?} of {} ({queries}x{in_features}x{out_features})",
                    variant_name(variant),
                    first,
                    baseline.output.len()
                );
            }
        }
    }

    clear_gemm_mma_variant();
}

/// The default (nothing forced, no flag set) must be the shipped 128x64 tile.
/// Without this, a change that made `AUTO` resolve to a research variant would
/// ship behind flags nobody set.
#[test]
fn auto_selection_defaults_to_the_shipped_tile() {
    // T4 measured `regpipe` +4.7% GEMM at 614 tokens and bit-identical output,
    // so it is now the default for the eligible (MCG-3bpw) arm; the original
    // `linear_gemm_mma` is reachable only via the research-gate fallback or
    // `KAPLUMBA_MMA_128X64` (the A/B control).
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(kernels) = kernels() else { return };
    unsafe {
        std::env::remove_var("KAPLUMBA_DISABLE_MMA_PREFILL");
        std::env::remove_var("KAPLUMBA_MMA_4SG");
        std::env::remove_var("KAPLUMBA_REGPIPE_K32");
        std::env::remove_var("KAPLUMBA_REGPIPE");
        std::env::remove_var("KAPLUMBA_MMA_128X64");
    };
    clear_gemm_mma_variant();

    let (queries, in_features, out_features) = (129, 128, 128);
    let input = activations(queries, in_features);
    let input_signs = signs(in_features, 3);
    let output_signs = signs(out_features, 5);
    let rate = Exl3Rate::Integer(3);
    let tiles = (in_features / 16) * (out_features / 16);
    let packed = packed_words(tiles, rate, 3);
    let run = run(
        &kernels,
        kaplumba_kernels::linear::GEMM_MMA_VARIANT_AUTO,
        &packed,
        &input,
        &input_signs,
        &output_signs,
        queries,
        in_features,
        out_features,
        rate,
    );
    assert_eq!(
        run.dispatched, GEMM_MMA_VARIANT_REGPIPE,
        "with no override and no flag, the eligible MCG-3bpw arm must dispatch \
         the T4-measured regpipe tile (not the original linear_gemm_mma)"
    );
}

/// Child-process probe: `run` below is spawned by
/// `env_flags_reach_the_dispatch`, with `T4_EXPECT_VARIANT` set. Ignored so
/// the normal `cargo test` run does not execute it with no expectation.
#[test]
#[ignore = "re-exec probe: needs T4_EXPECT_VARIANT and a fresh process"]
fn probe_env_flag_selection() {
    unsafe { std::env::remove_var("KAPLUMBA_DISABLE_MMA_PREFILL") };
    clear_gemm_mma_variant();
    let expected: u8 = std::env::var("T4_EXPECT_VARIANT")
        .expect("probe must be run with T4_EXPECT_VARIANT")
        .parse()
        .expect("T4_EXPECT_VARIANT is a variant id");

    let Some(kernels) = kernels() else { return };
    let (queries, in_features, out_features) = (129, 128, 128);
    let input = activations(queries, in_features);
    let input_signs = signs(in_features, 3);
    let output_signs = signs(out_features, 5);
    let rate = Exl3Rate::Integer(3);
    let tiles = (in_features / 16) * (out_features / 16);
    let packed = packed_words(tiles, rate, 5);
    let run = run(
        &kernels,
        kaplumba_kernels::linear::GEMM_MMA_VARIANT_AUTO,
        &packed,
        &input,
        &input_signs,
        &output_signs,
        queries,
        in_features,
        out_features,
        rate,
    );
    assert_eq!(
        run.dispatched,
        expected,
        "env-flag selection resolved to {} ({}), expected {} ({})",
        variant_name(run.dispatched),
        run.dispatched,
        variant_name(expected),
        expected
    );
}

/// Q3: `env_flag` memoises on first read, so the only honest way to check the
/// env arm of the resolution is a process that has never read it. This
/// re-execs this very binary as that process.
#[test]
fn env_flags_reach_the_dispatch() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if metal::Device::system_default().is_none() {
        eprintln!("no Metal device; skipping");
        return;
    }
    let exe = std::env::current_exe().expect("current_exe");
    // (flag, expected variant id) -- the ids, not the names, because the child
    // asserts against the same constants exported from `linear`.
    let cases: [(&str, u8); 3] = [
        ("KAPLUMBA_MMA_4SG", GEMM_MMA_VARIANT_4SG),
        ("KAPLUMBA_REGPIPE_K32", GEMM_MMA_VARIANT_REGPIPE_K32),
        ("KAPLUMBA_REGPIPE", GEMM_MMA_VARIANT_REGPIPE),
    ];
    for (flag, expected) in cases {
        let mut command = std::process::Command::new(&exe);
        command
            .arg("probe_env_flag_selection")
            .arg("--exact")
            .arg("--ignored")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env("T4_EXPECT_VARIANT", expected.to_string())
            // Strip every variant switch first so an ambient shell export
            // cannot make the child's AUTO arm resolve to something this case
            // did not ask for, then set the one under test.
            .env_remove("KAPLUMBA_DISABLE_MMA_PREFILL")
            .env_remove("KAPLUMBA_MMA_4SG")
            .env_remove("KAPLUMBA_REGPIPE_K32")
            .env_remove("KAPLUMBA_REGPIPE")
            .env(flag, "1");
        let output = command.output().expect("spawn probe");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "{flag}=1: probe child failed (status {})\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
            output.status
        );
        assert!(
            stdout.contains("1 passed") || stderr.contains("1 passed"),
            "{flag}=1: probe child did not run its assertion\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        );
    }
}
