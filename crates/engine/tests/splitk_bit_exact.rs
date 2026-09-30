//! Does split-K change the answer? (`_sk4` vs `_colmajor`)
//!
//! Spec `decode-19tok-repacked-250gb` forbids quality loss: "Repacked weights must
//! dequantize to exact same numeric values." A split-K reduction partitions the
//! tile axis into four contiguous ranges and sums the four partials in a second
//! pass. Floating-point addition is **not associative**, so that is a
//! reassociation of the same terms -- which `linear.metal` explicitly warns is
//! how a "bit-exact" kernel silently diverges.
//!
//! This test does not assume the answer. It measures it, on every real shape,
//! with `to_bits()` comparison rather than a tolerance, so the question is
//! settled by the hardware instead of by argument.
//!
//! Fail-first: if the two kernels were bit-identical, the *summation order*
//! would have to be irrelevant, and a deliberately perturbed `_sk4` would go
//! undetected. The perturbation arm below proves this test can fail.

use half::f16;
use kaplumba_engine::exl3::pack_integer_states;
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{LinearEncodeRequest, LinearKernels};
use metal::{Buffer, Device, MTLResourceOptions};

/// Real shapes, plus the two the earlier notes singled out as starved.
const SHAPES: &[(&str, usize, usize)] = &[
    ("full/kv   5376x2048", 5376, 2048), // 128 tile rows -- the worst
    ("sliding/k 5376x4096", 5376, 4096), // 256
    ("sliding/q 5376x8192", 5376, 8192), // 512
    ("sliding/o 8192x5376", 8192, 5376),
    ("sliding/gate 5376x21504", 5376, 21504), // 1344 -- the reference
];

const BITS: u8 = 3;

/// Variants compared against `linear_gemv_ishizuki_k3_colmajor`.
/// `_sk4` is here as the documented counter-example: split-K reorders the
/// summation and is NOT bit-exact, which is why it cannot be used under the
/// spec's no-quality-loss rule.
///
/// NOTE on `_contig4`: it is *not* listed here, and that is deliberate. It is the
/// kernel the engine dispatches, and it is NOT bit-exact in f32 -- it assigns four
/// contiguous tiles per simdgroup where `_colmajor` assigns interleaved ones, so
/// it reassociates (measured up to 51 ULP). It IS bit-identical after the fp16
/// rounding in `linear_finalize_simd`, which is what the sampler reads; that is
/// asserted in `live_kernel_bit_exact.rs`. This file compares the raw f32
/// accumulator, so including it here would report a deviation that does not affect
/// any generated token.
///
/// The `_u8` variant this originally compared against did not survive the branch
/// reset and measured 1.01-1.03x cold when it did; not worth carrying.
const CANDIDATES: &[(&str, &str)] = &[("_sk4", "linear_gemv_ishizuki_k3_colmajor_sk4")];

fn build(in_f: usize, out_f: usize, seed: u32) -> Vec<u16> {
    let tiles = (in_f / 16) * (out_f / 16);
    let mut packed = Vec::with_capacity(tiles * 16 * BITS as usize);
    let mut s = seed;
    for _ in 0..tiles {
        let mut states = [0u16; 256];
        for st in &mut states {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *st = (s & ((1u32 << BITS) - 1)) as u16;
        }
        let (w, _) = pack_integer_states(&states, BITS).unwrap();
        packed.extend(w);
    }
    packed
}

struct Scratch {
    input: Buffer,
    input_signs: Buffer,
    output_signs: Buffer,
    transformed: Buffer,
    raw: Buffer,
    output: Buffer,
}

fn scratch(device: &Device, in_f: usize, out_f: usize) -> Scratch {
    let ones_u16 = |n: usize| -> Vec<u16> { vec![f16::from_f32(1.0).to_bits(); n] };
    let mk = |v: &[u16]| -> Buffer {
        device.new_buffer_with_data(
            v.as_ptr() as *const std::ffi::c_void,
            (v.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let input_v: Vec<u16> = (0..in_f)
        .map(|i| f16::from_f32((i as f32 * 0.017).sin()).to_bits())
        .collect();
    Scratch {
        input: mk(&input_v),
        input_signs: mk(&ones_u16(in_f)),
        output_signs: mk(&ones_u16(out_f)),
        transformed: device.new_buffer((in_f * 2) as u64, MTLResourceOptions::StorageModeShared),
        raw: device.new_buffer((out_f * 4 * 4) as u64, MTLResourceOptions::StorageModeShared),
        output: device.new_buffer((out_f * 2) as u64, MTLResourceOptions::StorageModeShared),
    }
}

fn read_f32(buffer: &Buffer, len: usize) -> Vec<u32> {
    let ptr = buffer.as_ref().contents() as *const f32;
    (0..len).map(|i| unsafe { ptr.add(i).read() }.to_bits()).collect()
}

/// Records that split-K is NOT bit-exact, as a regression test rather than a
/// paragraph of prose that can rot.
///
/// Every remaining candidate is `_sk4`, and it must differ, so the panic below is
/// the expected outcome and is asserted. If a future split-K kernel *were*
/// bit-exact, this test would fail and force the spec's no-quality-loss rule to be
/// revisited deliberately rather than by accident.
#[test]
#[should_panic(expected = "split-K is NOT bit-exact")]
fn splitk_reassociates_and_is_not_bit_exact() {
    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let Ok(kernels) = LinearKernels::new() else {
        eprintln!("LinearKernels unavailable; skipping");
        return;
    };
    let queue = device.new_command_queue().to_owned();

    let mut mismatches: Vec<String> = Vec::new();
    println!("comparing against: {CANDIDATES:?}\n");

    for (idx, (name, in_f, out_f)) in SHAPES.iter().enumerate() {
        let packed = build(*in_f, *out_f, 11 + idx as u32);
        let buf = device
            .new_buffer_with_data(
                packed.as_ptr() as *const std::ffi::c_void,
                (packed.len() * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            );
        let sc = scratch(&device, *in_f, *out_f);

        let run = |kernel: &str| -> Vec<u32> {
            let mut enc = MetalEncoder::new(&queue).expect("encoder");
            kernels
                .encode_gemv_named(
                    &mut enc,
                    kernel,
                    &LinearEncodeRequest {
                        packed: &buf,
                        packed_offset: 0,
                        perm_mode: 1,
                        input: &sc.input,
                        input_signs: &sc.input_signs,
                        output_signs: &sc.output_signs,
                        transformed: &sc.transformed,
                        raw: &sc.raw,
                        output: &sc.output,
                        in_features: *in_f,
                        out_features: *out_f,
                        rate: Exl3Rate::Integer(BITS),
                        codebook: Exl3Codebook::Mcg,
                        pretransformed: false,
                    },
                )
                .unwrap_or_else(|e| panic!("{kernel} on {name}: {e}"));
            enc.commit_and_wait().expect("wait");
            // `raw` is the pre-finalize fp32 accumulator; comparing it isolates
            // the reduction from the fp16 finalize epilogue.
            read_f32(&sc.raw, *out_f)
        };

        let base = run("linear_gemv_ishizuki_k3_colmajor");

        assert!(
            base.iter().any(|&v| v != 0),
            "{name}: colmajor produced all zeros; nothing to compare"
        );

        for (label, kernel) in CANDIDATES {
            let other = run(kernel);
            let first_diff = base.iter().zip(&other).position(|(a, b)| a != b);
            match first_diff {
                None => println!("{name}: {label} BIT-EXACT across {} outputs", base.len()),
                Some(i) => {
                    let bf = f32::from_bits(base[i]);
                    let of = f32::from_bits(other[i]);
                    let ulps = (base[i] as i64 - other[i] as i64).abs();
                    println!(
                        "{name}: {label} DIFFERS at {i}/{n}: colmajor {bf:e} vs {kernel} {of:e} ({ulps} ULP)",
                        n = base.len(),
                    );
                    mismatches.push(format!(
                        "{name}: {label} first diff at element {i}: {bf:e} vs {of:e} ({ulps} ULP)"
                    ));
                }
            }
        }
    }

    if !mismatches.is_empty() {
        panic!(
            "split-K is NOT bit-exact against _colmajor:\n  {}\n\n\
             The spec's no-quality-loss restriction is about dequantized WEIGHTS being\n\
             identical, which split-K does preserve. But it also requires token\n\
             generation to match bit-exact greedy decode, and a changed reduction\n\
             order can change the argmax when two logits are within {}.\n\
             Options: (a) keep split-K and re-derive the golden, recording the\n\
             observed deviation; (b) restrict it to shapes where the margin is\n\
             provably large; (c) find a bit-exact way to raise occupancy, e.g. a\n\
             persistent-CTA grid that keeps the summation order intact.",
            mismatches.join("\n  "),
            "the observed ULP distance"
        );
    }
}

#[test]
fn the_comparison_can_actually_fail() {
    // Fail-first, AGENTS.md 3.4. If a perturbed kernel still compared equal, the
    // test above would be unfalsifiable and "bit-exact" would be vacuous.
    let Some(device) = Device::system_default() else {
        return;
    };
    let Ok(kernels) = LinearKernels::new() else {
        return;
    };
    let queue = device.new_command_queue().to_owned();

    let (in_f, out_f) = (5376usize, 2048usize);
    let mut packed = build(in_f, out_f, 99);
    let buf = device
        .new_buffer_with_data(
            packed.as_ptr() as *const std::ffi::c_void,
            (packed.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
    let sc = scratch(&device, in_f, out_f);

    let run = |kernel: &str, buffer: &Buffer| -> Vec<u32> {
        let mut enc = MetalEncoder::new(&queue).expect("encoder");
        kernels
            .encode_gemv_named(
                &mut enc,
                kernel,
                &LinearEncodeRequest {
                    packed: buffer,
                    packed_offset: 0,
                    perm_mode: 1,
                    input: &sc.input,
                    input_signs: &sc.input_signs,
                    output_signs: &sc.output_signs,
                    transformed: &sc.transformed,
                    raw: &sc.raw,
                    output: &sc.output,
                    in_features: in_f,
                    out_features: out_f,
                    rate: Exl3Rate::Integer(BITS),
                    codebook: Exl3Codebook::Mcg,
                    pretransformed: false,
                },
            )
            .expect("encode");
        enc.commit_and_wait().expect("wait");
        read_f32(&sc.raw, out_f)
    };

    let clean = run("linear_gemv_ishizuki_k3_colmajor", &buf);

    // Sabotage: flip one bit of one weight word. Same kernel, same everything.
    packed[7] ^= 0x0000_0001;
    let sabotaged = device
        .new_buffer_with_data(
            packed.as_ptr() as *const std::ffi::c_void,
            (packed.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
    let dirty = run("linear_gemv_ishizuki_k3_colmajor", &sabotaged);

    assert_ne!(
        clean, dirty,
        "FAIL-FIRST BROKEN: flipping one weight bit changed nothing, so this test \
         cannot detect a wrong kernel and 'bit-exact' means nothing."
    );
    println!(
        "fail-first ok: one flipped weight bit moved {} of {} outputs",
        clean.iter().zip(&dirty).filter(|(a, b)| a != b).count(),
        clean.len()
    );
}