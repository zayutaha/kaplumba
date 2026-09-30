//! The decode matvec's output must not move when `linear.metal` is edited.
//!
//! `tests/decode_nr0.rs` compares NR0 variants *against each other*. That is the
//! wrong question on its own: a change that re-associates the accumulator in
//! every kernel at once moves every variant together, so all of them still agree
//! with each other and the test passes while the model outputs different numbers.
//! Verified, not assumed -- re-ordering the four-lane butterfly so that
//! `shuffle_xor 1` comes after `shuffle_xor 2` instead of before is a *double*
//! accumulation, so it changes almost every output, 79325 of them across this
//! harness's cases, and `decode_nr0.rs` passed it without complaint because its
//! baseline was rebuilt from the same edited source.
//!
//! So the check that matters is against a fixed reference: the output the
//! **committed** kernel produces. This test holds that reference as raw bits.
//!
//! # What the reference covers
//!
//! See [`common::matvec_golden::CASES`]: 3 bpw -- 410 of this checkpoint's 411
//! tensors -- at five shapes, and 6 bpw, `lm_head`'s rate, at one. The other six
//! rates are unreachable from this checkpoint, so referencing them would pin
//! numbers nobody could act on.
//!
//! The 8 bpw kernel is the one place the explicit `fma` chain changed the output,
//! by one fp16 ulp in 2 of 21504 outputs; see `scripts/gen_exl3_gemv.py`. 8 bpw is
//! not dispatchable here, and 6 bpw -- which is, and where a 1-ulp logit change
//! can flip a token -- is bit-exact against the pre-NR0 kernel.
//!
//! # Regenerating
//!
//! Only when a `linear.metal` change is *meant* to move these bits, which under
//! this project's constraints means essentially never:
//!
//! ```text
//! KAPLUMBA_ACCEPT_MATVEC_GOLDEN=1 cargo test --release -p kaplumba-engine \
//!     --test decode_matvec_golden -- --nocapture
//! ```
//!
//! The env var makes the test print the new digest instead of failing, so the diff
//! is reviewed rather than rubber-stamped.

#[path = "common/matvec_golden.rs"]
pub mod matvec_golden;

use half::f16;
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{LinearEncodeRequest, LinearKernels};
use matvec_golden::{CASES, compute, parse, render};

const GOLDEN: &str = include_str!("data/decode_matvec_golden.txt");

/// Kernels digested per shape. Pinned as a constant so the count assertion
/// catches a change to `kernels_for` — including one that *adds* a kernel, which
/// is exactly what happens when a window variant is introduced, and which would
/// otherwise silently go unreferenced.
const KERNELS_PER_CASE: usize = 2;

/// The 2-word window variant must be **bit-identical** to the kernel it replaces.
///
/// This is a different claim from the golden above, and both are needed. The
/// golden asks "did the shipped kernel's output change?". This asks "is the
/// variant the same function as the original?" — which is what licenses shipping
/// it at all, since decode's bit-exactness is a project constraint.
///
/// The claim rests on a derivation (`two_word_is_safe` in
/// `scripts/gen_exl3_gemv.py`): the kernel extracts A0 bits `[48-7*KA, 63]`, the
/// third word contributes only to A0 bits `[0, sh)`, and at every emitted rate the
/// lowest used bit is at or above the largest `sh` that occurs. A derivation is
/// exactly the kind of thing that is wrong in a way that still compiles, so it is
/// checked against the hardware at every emitted rate and at shapes chosen to
/// exercise the tail loop and the wrap.
///
/// # Why `A0` is digested and not the final output
///
/// The comparison is on the raw fp32 the kernel writes, not the fp16 the caller
/// would see. A 1-ulp fp32 difference rounds away in fp16 for almost every
/// output, so an fp16 comparison would pass a genuinely wrong variant on most
/// columns — the same rounding-too-early blind spot the golden avoids.
#[test]
fn the_two_word_window_is_bit_identical_to_the_three_word_original() {
    if std::env::var_os("KAPLUMBA_ACCEPT_MATVEC_GOLDEN").is_some() {
        return;
    }
    let Some(device) = metal::Device::system_default() else {
        return;
    };
    let kernels = LinearKernels::shared().expect("kernels");
    let queue = device.new_command_queue();

    // Every emitted rate, so the set cannot silently shrink. `TWO_WORD_SAFE_RATES`
    // is a derivation; this is the check that the derivation still matches the
    // kernels that exist.
    let mut emitted: Vec<u8> = kernels
        .ishizuki_gemv_kernel_names()
        .iter()
        .filter_map(|name| name.strip_suffix("_w2").map(|_| name))
        .filter_map(|name| LinearKernels::ishizuki_rate_of_name(name))
        .collect();
    emitted.sort_unstable();
    emitted.dedup();
    assert!(
        !emitted.is_empty(),
        "no _w2 kernels found; the two-word family is not being generated, so this \
         test would pass on an empty set"
    );
    assert_eq!(
        emitted,
        vec![1, 2, 3, 4],
        "the generated two-word rates changed. `TWO_WORD_SAFE_RATES` in \
         scripts/gen_exl3_gemv.py derives these; if the set moved, that derivation \
         changed and this test's rate list must move with it."
    );

    let mut compared = 0usize;
    let mut mismatches: Vec<String> = Vec::new();
    // (in, out) shapes. 5376x21504 is `gate_proj`; 21504x5376 is `down_proj`, whose
    // k loop is 16x longer and which therefore exercises the tail loop's
    // remainder; 128x1024 has one k tile per simdgroup so the unrolled loop runs
    // zero times and ONLY the tail loop executes; 1024x1024 makes the unrolled
    // loop run and the tail loop empty. Between them both loops are covered.
    for (in_f, out_f) in [(128usize, 1024usize), (1024, 1024), (5376, 5376)] {
        for bits in 1u8..=4 {
            let original = format!("linear_gemv_ishizuki_k{bits}");
            let variant = format!("linear_gemv_ishizuki_k{bits}_w2");
            if kernels.ishizuki_gemv_pipeline_by_name(&original).is_none()
                || kernels.ishizuki_gemv_pipeline_by_name(&variant).is_none()
            {
                continue;
            }
            let (a, b) = run_pair(
                &device, &kernels, &queue, in_f, out_f, bits, &original, &variant,
            );
            compared += 1;
            if a != b {
                let bad: Vec<usize> = a
                    .iter()
                    .zip(b.iter())
                    .enumerate()
                    .filter(|(_, (x, y))| x != y)
                    .map(|(i, _)| i)
                    .collect();
                mismatches.push(format!(
                    "{bits}bpw {in_f}x{out_f}: {variant} differs from {original} in \
                     {}/{} outputs, first at column {}{}",
                    bad.len(),
                    a.len(),
                    bad[0],
                    if bad.len() == 1 {
                        format!(" ({:08x} vs {:08x})", b[bad[0]], a[bad[0]])
                    } else {
                        String::new()
                    }
                ));
            }
        }
    }
    assert!(
        compared >= 8,
        "only {compared} (rate, shape) pairs were compared; the emitted set has 4 \
         rates, so fewer than 8 means a shape or a kernel name stopped resolving and \
         the loop is skipping work"
    );
    assert!(
        mismatches.is_empty(),
        "the two-word window is not bit-identical to the original.\n\n{}\n\nThe \
         derivation says the third word is never read at these rates. If it is \
         wrong, do NOT ship the variant -- and do not \"fix\" it by adding the word \
         back without understanding why, because that would mean the derivation has \
         a hole that would silently reopen for a future rate.",
        mismatches.join("\n")
    );
}

type RunPair = (Vec<u32>, Vec<u32>);

#[allow(clippy::too_many_arguments)]
fn run_pair(
    device: &metal::Device,
    kernels: &LinearKernels,
    queue: &metal::CommandQueue,
    in_f: usize,
    out_f: usize,
    bits: u8,
    left: &str,
    right: &str,
) -> RunPair {
    let tiles = (in_f / 16) * (out_f / 16);
    let mut packed: Vec<u16> = Vec::new();
    let mut states = vec![0u16; 256];
    for tile in 0..tiles {
        for (code, slot) in states.iter_mut().enumerate() {
            *slot = matvec_golden::states_for(tile, code, bits);
        }
        packed.extend(
            kaplumba_engine::exl3::pack_integer_states(&states, bits)
                .unwrap()
                .0,
        );
    }
    let packed_b = device.new_buffer_with_data(
        packed.as_ptr() as *const std::ffi::c_void,
        (packed.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input: Vec<u16> = (0..in_f)
        .map(|i| {
            let x = i as f32;
            half::f16::from_f32((x * 0.0173).sin() + (x * 0.0031).cos() + 0.25).to_bits()
        })
        .collect();
    let input_b = device.new_buffer_with_data(
        input.as_ptr() as *const std::ffi::c_void,
        (in_f * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let is: Vec<u16> = (0..in_f)
        .map(|i| half::f16::from_f32(if i % 3 == 0 { 0.0021 } else { -0.0017 }).to_bits())
        .collect();
    let is_b = device.new_buffer_with_data(
        is.as_ptr() as *const std::ffi::c_void,
        (in_f * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let os: Vec<u16> = (0..out_f)
        .map(|i| half::f16::from_f32(if i % 5 == 0 { -0.0031 } else { 0.0023 }).to_bits())
        .collect();
    let os_b = device.new_buffer_with_data(
        os.as_ptr() as *const std::ffi::c_void,
        (out_f * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let trans = device.new_buffer(
        (in_f * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let raw = device.new_buffer(
        (out_f * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let out = device.new_buffer(
        (out_f * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let run = |name: &str| -> Vec<u32> {
        unsafe {
            std::ptr::write_bytes(raw.contents() as *mut f32, 0, out_f);
        }
        let mut encoder = MetalEncoder::new(queue).unwrap();
        kernels
            .encode_gemv_named(
                &mut encoder,
                name,
                &LinearEncodeRequest {
                    packed: &packed_b,
                    packed_offset: 0,
                    perm_mode: 0,
                    input: &input_b,
                    input_signs: &is_b,
                    output_signs: &os_b,
                    transformed: &trans,
                    raw: &raw,
                    output: &out,
                    in_features: in_f,
                    out_features: out_f,
                    rate: Exl3Rate::Integer(bits),
                    codebook: Exl3Codebook::Mcg,
                    pretransformed: false,
                },
            )
            .unwrap();
        encoder.commit_and_wait().unwrap();
        let ptr = raw.contents() as *const u32;
        (0..out_f)
            .map(|i| unsafe { std::ptr::read(ptr.add(i)) })
            .collect()
    };
    (run(left), run(right))
}

#[test]
fn the_decode_matvec_matches_its_committed_output() {
    let reference = parse(GOLDEN);
    // Two kernels per case, not one: the ishizuki family is what the engine
    // dispatches for 91% of a decode token's bytes, and it is the family a
    // change to the window arithmetic would touch. A count assertion of
    // `CASES.len()` would have let the ishizuki half be dropped without
    // failing.
    assert_eq!(
        reference.len(),
        CASES.len() * KERNELS_PER_CASE,
        "the golden file holds {} cases, this test runs {} ({} shapes x {} kernels); \
         `CASES` or `kernels_for` changed without the reference being regenerated",
        reference.len(),
        CASES.len() * KERNELS_PER_CASE,
        CASES.len(),
        KERNELS_PER_CASE,
    );

    if std::env::var_os("KAPLUMBA_ACCEPT_MATVEC_GOLDEN").is_some() {
        let cases = compute();
        print!("{}", render(&cases));
        eprintln!(
            "\nKAPLUMBA_ACCEPT_MATVEC_GOLDEN was set. The digest above is what belongs in \
             crates/engine/tests/data/decode_matvec_golden.txt. Review it before committing: \
             this is the model's arithmetic, and a 1-ulp change in a logit can flip a token."
        );
        return;
    }

    let got = compute();
    let failures: Vec<String> = got
        .iter()
        .zip(reference.iter())
        .filter_map(|(case, (want_header, want))| {
            if &case.header != want_header {
                return Some(format!(
                    "case order changed: this run produced '{}', the reference has '{want_header}'",
                    case.header
                ));
            }
            if case.values.len() != want.len() {
                return Some(format!(
                    "{}: {} outputs now, the reference has {}",
                    case.header,
                    case.values.len(),
                    want.len()
                ));
            }
            let bad: Vec<usize> = case
                .values
                .iter()
                .zip(want.iter())
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(i, _)| i)
                .collect();
            if bad.is_empty() {
                return None;
            }
            let sample: Vec<String> = bad
                .iter()
                .take(4)
                .map(|i| {
                    // The word width follows the family: ishizuki is digested as
                    // fp32, `_mcg` as fp16. Guessing wrong here would report a
                    // nonsense delta on a real difference, which is worse than
                    // reporting none.
                    let (now, before) = if matvec_golden::writes_raw(&case.header) {
                        (f32::from_bits(case.values[*i]), f32::from_bits(want[*i]))
                    } else {
                        (
                            f16::from_bits(case.values[*i] as u16).to_f32(),
                            f16::from_bits(want[*i] as u16).to_f32(),
                        )
                    };
                    format!(
                        "col {i}: {:08x} now vs {:08x} in the reference \
                         ({now:.9} vs {before:.9}, delta {:+e})",
                        case.values[*i],
                        want[*i],
                        now - before,
                    )
                })
                .collect();
            Some(format!(
                "{}: {}/{} outputs differ\n    {}",
                case.header,
                bad.len(),
                case.values.len(),
                sample.join("\n    ")
            ))
        })
        .collect();

    assert!(
        failures.is_empty(),
        "the decode matvec's output moved. Decode is bit-exact by project constraint (AGENTS.md \
         §8) and a 1-ulp change in a logit can flip a generated token, so this is a quality \
         change and not a refactor.\n\n{}\n\nIf the change was genuinely intended, regenerate \
         with KAPLUMBA_ACCEPT_MATVEC_GOLDEN=1 and review the diff.\n\nThe likeliest cause is FMA \
         contraction. The accumulator is an explicit `fma` chain precisely so the compiler \
         cannot re-associate it -- see the comment in `scripts/gen_exl3_gemv.py` -- so a \
         re-association here means that chain was edited, reformatted past the generator's \
         normalisation, or bypassed by an `_mcg` kernel that was hand-written.",
        failures.join("\n\n")
    );
}
