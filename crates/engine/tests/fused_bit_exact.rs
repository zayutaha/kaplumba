//! Is the 8x-unroll fused GEMV bit-exact against its 4x original?
//!
//! `linear_gemv_ishizuki_k3_fused2_u8` and `_fused3_u8` are the biggest single
//! lever in the decode step: `gate_up_proj` is 36.8% of a step and `qkv_proj`
//! 17.1% by `KAPLUMBA_DECODE_PROFILE=kernel`. The previous round of `_u8` work
//! missed them because `encode_gemv_fused2` / `_fused3` look their pipeline up by
//! hardcoded name, so the measured end-to-end gain was +1.2% against a predicted
//! +24%.
//!
//! ## Why these shapes are chosen the way they are
//!
//! The fused kernels split the grid by threadgroup index: `fused2` sends
//! `tgid < tiles_a` to tensor A and `tgid >= tiles_a` to B; `fused3` is the
//! three-way version. **A wrong partition does not crash, does not produce
//! zeros, and does not look obviously wrong** -- it writes every output exactly
//! once with plausible magnitudes, just the wrong ones. That is the failure mode
//! a `to_bits()` comparison against a *sibling kernel on the same data* cannot
//! see if both shapes are equal, because a self-consistent partition bug produces
//! self-consistent output on both.
//!
//! So every case uses **three different shapes** (A != B != C, and all three
//! distinct tile-row counts). Then a swapped or off-by-one partition sends a
//! threadgroup at tensor A's weight data with tensor B's output pointer, and the
//! two kernels cannot agree by accident.
//!
//! `gate_proj` (5376x21504) and `up_proj` (5376x21504) are genuinely the same
//! shape in this model, so the equal-shape case is included too -- as a positive
//! control, to show the test *can* pass, not just fail.
//!
//! ## Fail-first
//!
//! `the_comparison_can_fail` flips one weight bit and asserts the fused output
//! changes. Without it, a test that compared the fused kernels only against each
//! other could pass trivially if both were reading the wrong buffer.

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
    output_signs_a: Buffer,
    output_signs_b: Buffer,
    output_signs_c: Buffer,
    transformed: Buffer,
    raw_a: Buffer,
    raw_b: Buffer,
    raw_c: Buffer,
    out_a: Buffer,
    out_b: Buffer,
    out_c: Buffer,
}

fn scratch(device: &Device, in_f: usize, out: &[usize]) -> Scratch {
    let mut raw = |n: usize| {
        device.new_buffer((n * 4 * 4) as u64, MTLResourceOptions::StorageModeShared)
    };
    let mut fin = |n: usize| {
        device.new_buffer((n * 2) as u64, MTLResourceOptions::StorageModeShared)
    };
    let input_v: Vec<u16> = (0..in_f)
        .map(|i| f16::from_f32((i as f32 * 0.013).sin()).to_bits())
        .collect();
    Scratch {
        input: upload(device, &input_v),
        input_signs: ones(device, in_f),
        output_signs_a: ones(device, out[0]),
        output_signs_b: ones(device, out[1]),
        output_signs_c: ones(device, *out.get(2).unwrap_or(&out[1])),
        transformed: fin(in_f),
        raw_a: raw(out[0]),
        raw_b: raw(out[1]),
        raw_c: raw(*out.get(2).unwrap_or(&out[1])),
        out_a: fin(out[0]),
        out_b: fin(out[1]),
        out_c: fin(*out.get(2).unwrap_or(&out[1])),
    }
}

fn read_f32(buffer: &Buffer, len: usize) -> Vec<u32> {
    let ptr = buffer.as_ref().contents() as *const f32;
    (0..len).map(|i| unsafe { ptr.add(i).read() }.to_bits()).collect()
}

fn nonempty(v: &[u32], what: &str) {
    assert!(
        v.iter().any(|&x| x != 0),
        "{what} is all zeros; the kernel computed nothing to compare"
    );
}

/// Compare one fused pair, 4x against 8x, on both outputs.
#[test]
fn fused2_u8_is_bit_exact_against_fused2() {
    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let Ok(kernels) = LinearKernels::new() else {
        eprintln!("LinearKernels unavailable; skipping");
        return;
    };
    let queue = device.new_command_queue().to_owned();

    // in_f, out_a, out_b, label
    let cases: &[(usize, usize, usize, &str)] = &[
        (5376, 21504, 21504, "gate_up: EQUAL shapes (the real model) - positive control"),
        (5376, 21504, 8192, "unequal A=21504 B=8192"),
        (5376, 8192, 21504, "unequal A=8192 B=21504 (partition reversed)"),
        (5376, 4096, 32768, "unequal, B 4x wider than A"),
        (5376, 32768, 4096, "unequal, A 4x wider than B"),
    ];

    for (idx, (in_f, out_a, out_b, label)) in cases.iter().enumerate() {
        let pa = upload(&device, &build(*in_f, *out_a, 31 + idx as u32 * 7));
        let pb = upload(&device, &build(*in_f, *out_b, 97 + idx as u32 * 13));
        let sc = scratch(&device, *in_f, &[*out_a, *out_b]);

        let run = |kernel: &str| -> (Vec<u32>, Vec<u32>) {
            let mut enc = MetalEncoder::new(&queue).expect("encoder");
            let shape_vals = [*in_f as u32, *out_a as u32, *out_b as u32];
            kernels
                .encode_gemv_fused2(
                    &mut enc,
                    &pa,
                    0,
                    *out_a,
                    &sc.output_signs_a,
                    &sc.out_a,
                    &pb,
                    0,
                    *out_b,
                    &sc.output_signs_b,
                    &sc.out_b,
                    &sc.input,
                    &sc.input_signs,
                    &sc.transformed,
                    &sc.raw_a,
                    &sc.raw_b,
                    *in_f,
                    false,
                    Some(kernel),
                )
                .unwrap_or_else(|e| panic!("{kernel} {label}: {e}"));
            enc.commit_and_wait().expect("wait");
            let _ = shape_vals;
            (read_f32(&sc.raw_a, *out_a), read_f32(&sc.raw_b, *out_b))
        };

        let name_4x = "linear_gemv_ishizuki_k3_fused2";
        let name_8x = "linear_gemv_ishizuki_k3_fused2_u8";
        let (a4, b4) = run(name_4x);
        let (a8, b8) = run(name_8x);
        nonempty(&a4, &format!("fused2 {label} A"));
        nonempty(&b4, &format!("fused2 {label} B"));

        for (which, x, y, n) in [("A", &a4, &a8, *out_a), ("B", &b4, &b8, *out_b)] {
            match x.iter().zip(y).position(|(p, q)| p != q) {
                None => println!("  fused2 {label}: output {which} BIT-EXACT across {n}"),
                Some(i) => {
                    let d = (x[i] as i64 - y[i] as i64).abs();
                    panic!(
                        "fused2 {label}: output {which} DIFFERS at {i}/{n}: \
                         4x {:e} vs 8x {:e} ({d} ULP)",
                        f32::from_bits(x[i]),
                        f32::from_bits(y[i])
                    );
                }
            }
        }
    }
}

#[test]
fn fused3_u8_is_bit_exact_against_fused3() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let Ok(kernels) = LinearKernels::new() else {
        return;
    };
    let queue = device.new_command_queue().to_owned();

    let cases: &[(usize, usize, usize, usize, &str)] = &[
        (5376, 8192, 4096, 4096, "sliding q/k/v (real)"),
        (5376, 16384, 1024, 1024, "full q/kv (real)"),
        (5376, 2048, 8192, 4096, "3-way all different"),
        (5376, 32768, 2048, 1024, "3-way, A 16x C"),
        (5376, 1024, 2048, 32768, "3-way, C 16x A"),
    ];

    for (idx, (in_f, a, b, c, label)) in cases.iter().enumerate() {
        let pa = upload(&device, &build(*in_f, *a, 41 + idx as u32 * 11));
        let pb = upload(&device, &build(*in_f, *b, 53 + idx as u32 * 17));
        let pc = upload(&device, &build(*in_f, *c, 67 + idx as u32 * 19));
        let sc = scratch(&device, *in_f, &[*a, *b, *c]);

        let run = |kernel: &str| -> (Vec<u32>, Vec<u32>, Vec<u32>) {
            let mut enc = MetalEncoder::new(&queue).expect("encoder");
            kernels
                .encode_gemv_fused3(
                    &mut enc,
                    &pa, 0, *a, &sc.output_signs_a, &sc.out_a,
                    &pb, 0, *b, &sc.output_signs_b, &sc.out_b,
                    &pc, 0, *c, &sc.output_signs_c, &sc.out_c,
                    &sc.input, &sc.input_signs, &sc.transformed,
                    &sc.raw_a, &sc.raw_b, &sc.raw_c,
                    *in_f,
                    false,
                    Some(kernel),
                )
                .unwrap_or_else(|e| panic!("{kernel} {label}: {e}"));
            enc.commit_and_wait().expect("wait");
            (read_f32(&sc.raw_a, *a), read_f32(&sc.raw_b, *b), read_f32(&sc.raw_c, *c))
        };

        let (a4, b4, c4) = run("linear_gemv_ishizuki_k3_fused3");
        let (a8, b8, c8) = run("linear_gemv_ishizuki_k3_fused3_u8");
        nonempty(&a4, &format!("fused3 {label} A"));
        nonempty(&b4, &format!("fused3 {label} B"));
        nonempty(&c4, &format!("fused3 {label} C"));

        for (which, x, y, n) in [
            ("A", &a4, &a8, *a),
            ("B", &b4, &b8, *b),
            ("C", &c4, &c8, *c),
        ] {
            match x.iter().zip(y).position(|(p, q)| p != q) {
                None => println!("  fused3 {label}: output {which} BIT-EXACT across {n}"),
                Some(i) => {
                    let d = (x[i] as i64 - y[i] as i64).abs();
                    panic!(
                        "fused3 {label}: output {which} DIFFERS at {i}/{n}: \
                         4x {:e} vs 8x {:e} ({d} ULP)",
                        f32::from_bits(x[i]),
                        f32::from_bits(y[i])
                    );
                }
            }
        }
    }
}

/// Fail-first, AGENTS.md 3.4.
///
/// Flips one weight bit and requires the fused output to change. Without this,
/// a test comparing the fused kernels only against each other could pass
/// trivially if both were reading the same wrong buffer -- which is precisely the
/// partition bug this file exists to catch.
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
    let sc = scratch(&device, in_f, &[out_a, out_b]);

    let mut dirty = good.clone();
    // Perturb inside tile 0 so the change reaches a real output element.
    dirty[3] ^= 0x0000_0001;

    let run = |packed: &[u16], kernel: &str| -> (Vec<u32>, Vec<u32>) {
        let buf = upload(&device, packed);
        let mut enc = MetalEncoder::new(&queue).expect("encoder");
        kernels
            .encode_gemv_fused2(
                &mut enc, &buf, 0, out_a, &sc.output_signs_a, &sc.out_a,
                &buf, 0, out_b, &sc.output_signs_b, &sc.out_b,
                &sc.input, &sc.input_signs, &sc.transformed,
                &sc.raw_a, &sc.raw_b, in_f, false, Some(kernel),
            )
            .expect("encode");
        enc.commit_and_wait().expect("wait");
        (read_f32(&sc.raw_a, out_a), read_f32(&sc.raw_b, out_b))
    };

    for kernel in [
        "linear_gemv_ishizuki_k3_fused2",
        "linear_gemv_ishizuki_k3_fused2_u8",
    ] {
        let (ca, cb) = run(&good, kernel);
        let (da, db) = run(&dirty, kernel);
        let changed = ca.iter().zip(&da).filter(|(x, y)| x != y).count()
            + cb.iter().zip(&db).filter(|(x, y)| x != y).count();
        assert_ne!(
            ca, da,
            "FAIL-FIRST BROKEN for {kernel}: flipping one weight bit changed no output \
             in A, so this test cannot detect a wrong kernel"
        );
        assert!(
            changed > 0,
            "FAIL-FIRST BROKEN for {kernel}: no output changed at all"
        );
        println!("  fail-first ok for {kernel}: {changed} outputs moved");
    }
}