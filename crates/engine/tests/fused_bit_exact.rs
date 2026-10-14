//! Does the fused GEMV's threadgroup partition actually put each tensor in the
//! right place?
//!
//! `encode_gemv_fused2` and `encode_gemv_fused3` merge several projections into one
//! dispatch by partitioning the grid on threadgroup index: `fused2` sends
//! `tgid < tiles_a` to tensor A and `tgid >= tiles_a` to B; `fused3` is the
//! three-way version.
//!
//! **A wrong partition is the most dangerous bug in this file.** It does not crash,
//! does not produce zeros, and does not look obviously wrong -- every output is
//! written exactly once with plausible magnitudes, just the wrong ones. It can only
//! be caught by comparing against something that cannot make the same mistake.
//!
//! ## Why unequal shapes, and why this comparison
//!
//! Comparing the fused kernels against each other would not be enough: two kernels
//! that share a partition bug produce self-consistently wrong output. So each fused
//! call is compared against **independent `encode_gemv` calls**, one per tensor,
//! which take a completely different code path and cannot reproduce a partition
//! error. Every case uses three different shapes, so a swapped or off-by-one
//! partition routes threadgroups at the wrong weights and the comparison fails.
//!
//! `gate_proj` (5376x21504) and `up_proj` (5376x21504) genuinely are the same shape in
//! this model, so the equal-shape case is included too -- as a positive control,
//! to show the test can pass rather than only fail.
//!
//! ## Fail-first
//!
//! `the_fused_comparison_can_fail` flips one weight bit and requires the fused
//! output to change. Without it, a comparison against an independent path could
//! pass trivially if both were reading the same wrong buffer.

use half::f16;
use kaplumba_engine::exl3::pack_integer_states;
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{LinearEncodeRequest, LinearKernels};
use metal::{Buffer, Device, MTLResourceOptions};

const BITS: u8 = 3;

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
        let (w, _) = pack_integer_states(&states, BITS).expect("pack");
        packed.extend(w);
    }
    packed
}

fn upload(device: &Device, v: &[u16]) -> Buffer {
    device.new_buffer_with_data(
        v.as_ptr() as *const std::ffi::c_void,
        (v.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn ones(device: &Device, n: usize) -> Buffer {
    upload(device, &vec![f16::from_f32(1.0).to_bits(); n])
}

struct Scratch {
    input: Buffer,
    input_signs: Buffer,
    transformed: Buffer,
    raw: Buffer,
    out: Buffer,
}

fn scratch(device: &Device, in_f: usize, out: usize) -> Scratch {
    let input_v: Vec<u16> = (0..in_f)
        .map(|i| f16::from_f32((i as f32 * 0.013).sin()).to_bits())
        .collect();
    Scratch {
        input: upload(device, &input_v),
        input_signs: ones(device, in_f),
        transformed: device.new_buffer((in_f * 2) as u64, MTLResourceOptions::StorageModeShared),
        raw: device.new_buffer((out * 4 * 4) as u64, MTLResourceOptions::StorageModeShared),
        out: device.new_buffer((out * 2) as u64, MTLResourceOptions::StorageModeShared),
    }
}

fn read_u16(buffer: &Buffer, len: usize) -> Vec<u16> {
    let ptr = buffer.as_ref().contents() as *const u16;
    (0..len).map(|i| unsafe { ptr.add(i).read() }).collect()
}

fn nonempty(v: &[u16], what: &str) {
    assert!(
        v.iter().any(|&x| x != 0),
        "{what} is all zeros; the kernel computed nothing to compare"
    );
}

/// One projection, encoded on its own through the plain `encode_gemv` path.
fn encode_alone(
    kernels: &LinearKernels,
    queue: &metal::CommandQueue,
    device: &Device,
    packed: &[u16],
    in_f: usize,
    out_f: usize,
) -> Vec<u16> {
    let buf = upload(device, packed);
    let sc = scratch(device, in_f, out_f);
    let mut e = MetalEncoder::new(queue).expect("encoder");
    kernels
        .encode_gemv(
            &mut e,
            LinearEncodeRequest {
                packed: &buf,
                packed_offset: 0,
                perm_mode: 1,
                input: &sc.input,
                input_signs: &sc.input_signs,
                output_signs: &ones(device, out_f),
                transformed: &sc.transformed,
                raw: &sc.raw,
                output: &sc.out,
                in_features: in_f,
                out_features: out_f,
                rate: Exl3Rate::Integer(BITS),
                codebook: Exl3Codebook::Mcg,
                pretransformed: false,
            },
        )
        .expect("encode_gemv");
    e.commit_and_wait().expect("wait");
    read_u16(&sc.out, out_f)
}

#[test]
fn fused2_matches_two_independent_encodes() {
    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let Ok(kernels) = LinearKernels::new() else {
        eprintln!("LinearKernels unavailable; skipping");
        return;
    };
    let queue = device.new_command_queue().to_owned();

    // in_f, out_a, out_b, label. Deliberately unequal except the first.
    let cases: &[(usize, usize, usize, &str)] = &[
        (5376, 21504, 21504, "gate/up: EQUAL shapes (the real model) - positive control"),
        (5376, 21504, 8192, "unequal A=21504 B=8192"),
        (5376, 8192, 21504, "unequal A=8192 B=21504 (partition reversed)"),
        (5376, 4096, 32768, "unequal, B 4x wider than A"),
    ];

    for (idx, (in_f, out_a, out_b, label)) in cases.iter().enumerate() {
        let pa = build(*in_f, *out_a, 31 + idx as u32 * 7);
        let pb = build(*in_f, *out_b, 97 + idx as u32 * 13);

        // Reference: each tensor encoded entirely on its own.
        let ref_a = encode_alone(&kernels, &queue, &device, &pa, *in_f, *out_a);
        let ref_b = encode_alone(&kernels, &queue, &device, &pb, *in_f, *out_b);
        nonempty(&ref_a, &format!("fused2 {label} A"));
        nonempty(&ref_b, &format!("fused2 {label} B"));

        // Fused: both in one dispatch.
        let ba = upload(&device, &pa);
        let bb = upload(&device, &pb);
        let sc_a = scratch(&device, *in_f, *out_a);
        let sc_b = scratch(&device, *in_f, *out_b);
        let mut e = MetalEncoder::new(&queue).expect("encoder");
        kernels
            .encode_gemv_fused2(
                &mut e,
                &ba,
                0,
                *out_a,
                &ones(&device, *out_a),
                &sc_a.out,
                &bb,
                0,
                *out_b,
                &ones(&device, *out_b),
                &sc_b.out,
                &sc_a.input,
                &sc_a.input_signs,
                &sc_a.transformed,
                &sc_a.raw,
                &sc_b.raw,
                *in_f,
                false,
                0,
            )
            .unwrap_or_else(|e| panic!("fused2 {label}: {e}"));
        e.commit_and_wait().expect("wait");

        let got_a = read_u16(&sc_a.out, *out_a);
        let got_b = read_u16(&sc_b.out, *out_b);
        assert_eq!(
            ref_a, got_a,
            "fused2 {label}: output A differs from an independent encode; the \
             threadgroup partition is wrong"
        );
        assert_eq!(
            ref_b, got_b,
            "fused2 {label}: output B differs from an independent encode; the \
             threadgroup partition is wrong"
        );
        println!("  fused2 {label}: A and B both match independent encodes");
    }
}

#[test]
fn fused3_matches_three_independent_encodes() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let Ok(kernels) = LinearKernels::new() else {
        return;
    };
    let queue = device.new_command_queue().to_owned();

    let cases: &[(usize, usize, usize, usize, &str)] = &[
        (5376, 8192, 4096, 4096, "sliding q/k/v (real: k and v are the same shape)"),
        (5376, 16384, 1024, 1024, "full q/kv (real)"),
        (5376, 2048, 8192, 4096, "3-way all different"),
        (5376, 32768, 2048, 1024, "3-way, A 8x C"),
        (5376, 1024, 2048, 32768, "3-way, C 8x A"),
    ];

    for (idx, (in_f, a, b, c, label)) in cases.iter().enumerate() {
        let pa = build(*in_f, *a, 41 + idx as u32 * 11);
        let pb = build(*in_f, *b, 53 + idx as u32 * 17);
        let pc = build(*in_f, *c, 67 + idx as u32 * 19);

        let ref_a = encode_alone(&kernels, &queue, &device, &pa, *in_f, *a);
        let ref_b = encode_alone(&kernels, &queue, &device, &pb, *in_f, *b);
        let ref_c = encode_alone(&kernels, &queue, &device, &pc, *in_f, *c);
        nonempty(&ref_a, &format!("fused3 {label} A"));
        nonempty(&ref_b, &format!("fused3 {label} B"));
        nonempty(&ref_c, &format!("fused3 {label} C"));

        let ba = upload(&device, &pa);
        let bb = upload(&device, &pb);
        let bc = upload(&device, &pc);
        let sa = scratch(&device, *in_f, *a);
        let sb = scratch(&device, *in_f, *b);
        let sc = scratch(&device, *in_f, *c);
        let mut e = MetalEncoder::new(&queue).expect("encoder");
        kernels
            .encode_gemv_fused3(
                &mut e,
                &ba, 0, *a, &ones(&device, *a), &sa.out,
                &bb, 0, *b, &ones(&device, *b), &sb.out,
                &bc, 0, *c, &ones(&device, *c), &sc.out,
                &sa.input, &sa.input_signs, &sa.transformed,
                &sa.raw, &sb.raw, &sc.raw,
                *in_f,
                false,
                0,
            )
            .unwrap_or_else(|e| panic!("fused3 {label}: {e}"));
        e.commit_and_wait().expect("wait");

        for (which, r, g, n) in [
            ("A", ref_a.clone(), read_u16(&sa.out, *a), *a),
            ("B", ref_b.clone(), read_u16(&sb.out, *b), *b),
            ("C", ref_c.clone(), read_u16(&sc.out, *c), *c),
        ] {
            assert_eq!(
                r, g,
                "fused3 {label}: output {which} ({n} elements) differs from an \
                 independent encode; the threadgroup partition is wrong"
            );
        }
        println!("  fused3 {label}: A, B and C all match independent encodes");
    }
}

/// Fail-first, AGENTS.md 3.4.
#[test]
fn the_fused_comparison_can_fail() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let Ok(kernels) = LinearKernels::new() else {
        return;
    };
    let queue = device.new_command_queue().to_owned();

    let in_f = 5376usize;
    let (out_a, out_b) = (21504usize, 8192usize); // deliberately unequal
    let good = build(in_f, out_a, 5);
    let sc = scratch(&device, in_f, out_a);

    let run = |packed: &[u16]| -> Vec<u16> {
        let buf = upload(&device, packed);
        let mut e = MetalEncoder::new(&queue).expect("encoder");
        kernels
            .encode_gemv_fused2(
                &mut e,
                &buf, 0, out_a, &ones(&device, out_a), &sc.out,
                &buf, 0, out_b, &ones(&device, out_b), &upload(&device, &vec![0u16; out_b]),
                &sc.input, &sc.input_signs, &sc.transformed,
                &sc.raw, &device.new_buffer((out_b * 16) as u64, MTLResourceOptions::StorageModeShared),
                in_f, false, 0,
            )
            .expect("encode");
        e.commit_and_wait().expect("wait");
        read_u16(&sc.out, out_a)
    };

    let clean = run(&good);
    let mut dirty = good.clone();
    // Perturb inside tile 0 so the change reaches a real output element.
    dirty[3] ^= 0x0000_0001;
    let sabotaged = run(&dirty);

    assert!(
        clean.iter().any(|&x| x != 0),
        "FAIL-FIRST BROKEN: clean output is all zeros"
    );
    assert_ne!(
        clean, sabotaged,
        "FAIL-FIRST BROKEN: flipping one weight bit changed no output, so this \
         test cannot detect a wrong kernel and 'bit-exact' means nothing."
    );
    println!(
        "  fail-first ok: one flipped weight bit moved {} of {} outputs",
        clean.iter().zip(&sabotaged).filter(|(a, b)| a != b).count(),
        clean.len()
    );
}

#[test]
fn fused2_matches_dual_independent_encodes_with_stride_x() {
    let device = metal::Device::system_default().expect("no metal device");
    let queue = device.new_command_queue();
    let kernels = LinearKernels::new().expect("kernels");

    let in_f = 5376usize;
    let (out_a, out_b) = (21504usize, 21504usize);
    let pa = build(in_f, out_a, 1);
    let pb = build(in_f, out_b, 2);

    // Two distinct input vectors: A gets 1.0, B gets 2.0
    let input_a_vals = vec![f16::from_f32(1.0).to_bits(); in_f];
    let input_b_vals = vec![f16::from_f32(2.0).to_bits(); in_f];
    let buf_input_a = upload(&device, &input_a_vals);
    let buf_input_b = upload(&device, &input_b_vals);

    let signs_a = ones(&device, in_f);
    let signs_b = ones(&device, in_f);
    let out_signs_a = ones(&device, out_a);
    let out_signs_b = ones(&device, out_b);

    // Reference A: single linear with input A
    let sc_a = scratch(&device, in_f, out_a);
    let mut ea = MetalEncoder::new(&queue).expect("encoder");
    let ba = upload(&device, &pa);
    kernels
        .encode_gemv(
            &mut ea,
            LinearEncodeRequest {
                packed: &ba,
                packed_offset: 0,
                perm_mode: 1,
                input: &buf_input_a,
                input_signs: &signs_a,
                output_signs: &out_signs_a,
                transformed: &sc_a.transformed,
                raw: &sc_a.raw,
                output: &sc_a.out,
                in_features: in_f,
                out_features: out_a,
                rate: Exl3Rate::Integer(BITS),
                codebook: Exl3Codebook::Mcg,
                pretransformed: false,
            },
        )
        .expect("encode a");
    ea.commit_and_wait().expect("wait");
    let ref_a = read_u16(&sc_a.out, out_a);

    // Reference B: single linear with input B
    let sc_b = scratch(&device, in_f, out_b);
    let mut eb = MetalEncoder::new(&queue).expect("encoder");
    let bb = upload(&device, &pb);
    kernels
        .encode_gemv(
            &mut eb,
            LinearEncodeRequest {
                packed: &bb,
                packed_offset: 0,
                perm_mode: 1,
                input: &buf_input_b,
                input_signs: &signs_b,
                output_signs: &out_signs_b,
                transformed: &sc_b.transformed,
                raw: &sc_b.raw,
                output: &sc_b.out,
                in_features: in_f,
                out_features: out_b,
                rate: Exl3Rate::Integer(BITS),
                codebook: Exl3Codebook::Mcg,
                pretransformed: false,
            },
        )
        .expect("encode b");
    eb.commit_and_wait().expect("wait");
    let ref_b = read_u16(&sc_b.out, out_b);

    // Fused dual-input: stage transformed inputs sequentially at offset 0 and in_f * 2 bytes
    let mut combined_transformed = Vec::with_capacity(in_f * 2);
    // pretransform A: input_a * signs_a
    combined_transformed.extend_from_slice(&read_u16(&sc_a.transformed, in_f));
    // pretransform B: input_b * signs_b
    combined_transformed.extend_from_slice(&read_u16(&sc_b.transformed, in_f));
    let buf_combined_transformed = upload(&device, &combined_transformed);

    let fused_sc_a = scratch(&device, in_f, out_a);
    let fused_sc_b = scratch(&device, in_f, out_b);
    let mut ef = MetalEncoder::new(&queue).expect("encoder");
    kernels
        .encode_gemv_fused2(
            &mut ef,
            &ba, 0, out_a, &out_signs_a, &fused_sc_a.out,
            &bb, 0, out_b, &out_signs_b, &fused_sc_b.out,
            &buf_input_a, &signs_a, &buf_combined_transformed,
            &fused_sc_a.raw, &fused_sc_b.raw,
            in_f,
            true, // pretransformed
            in_f as u32, // stride_x
        )
        .expect("encode fused2 with stride_x");
    ef.commit_and_wait().expect("wait");

    let got_a = read_u16(&fused_sc_a.out, out_a);
    let got_b = read_u16(&fused_sc_b.out, out_b);

    assert_eq!(got_a, ref_a, "dual-input fused2 output A did not match reference");
    assert_eq!(got_b, ref_b, "dual-input fused2 output B did not match reference");
}