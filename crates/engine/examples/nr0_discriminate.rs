//! Probe: which (shape, rate) combinations expose the FMA-contraction
//! re-association between NR0=1 and NR0=8 at 8 bpw?
//!
//! Motivation: `tests/decode_nr0.rs` found `linear_gemv_exl3_k8_mcg_r8` differing
//! from `linear_gemv_exl3_k8_mcg` in 2 of 21504 outputs at 5376x21504. That shape
//! is the most expensive one in the suite to *build* (115 M codes to generate and
//! pack), so the suite cannot afford it at every rate. This finds the cheapest
//! shape that still catches the difference, so the suite can keep the coverage
//! and drop the cost.
//!
//! Not a test: a probe whose output is a table for a human to read. It asserts
//! only that it found at least one shape that discriminates, because a probe
//! that finds nothing would otherwise look like a clean bill of health.

use half::f16;
use kaplumba_engine::exl3::pack_integer_states;
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{LinearEncodeRequest, LinearKernels};
use metal::{Device, MTLResourceOptions};

fn states_for(tile: usize, code: usize, rate: u8) -> u16 {
    let mut state = 0x1234_5678_u32
        ^ (tile as u32).wrapping_mul(0x9E37_79B9)
        ^ (code as u32).wrapping_mul(0x85EB_CA6B)
        ^ ((rate as u32) << 24);
    for _ in 0..3 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        state ^= state >> 13;
    }
    let usable = 1usize << rate.min(8);
    let value = if usable > 6 {
        3 + (state as usize % (usable - 5))
    } else {
        state as usize % usable
    };
    value as u16
}

fn main() {
    let device = Device::system_default().expect("Metal device");
    let kernels = LinearKernels::shared().expect("kernels");
    let queue = device.new_command_queue();

    let shapes: &[(usize, usize)] = &[
        (128, 1024),
        (1024, 1024),
        (5376, 5376),
        (128, 21504),
        (1024, 21504),
        (5376, 4096),
        (5376, 21504),
        (21504, 5376),
    ];

    println!(
        "{:<16}{:>4}{:>7}{:>7}{:>7}{:>7}{:>10}{:>10}",
        "shape", "bpw", "build_s", "r2", "r4", "r8", "codes_M", "verdict"
    );
    let mut discriminates = 0;
    for rate in [3u8, 6, 8] {
        for &(in_f, out_f) in shapes {
            if !((out_f / 16) % 8 == 0) {
                continue;
            }
            let started = std::time::Instant::now();
            let tiles = (in_f / 16) * (out_f / 16);
            let mut packed = Vec::new();
            let mut states = vec![0u16; 256];
            for tile in 0..tiles {
                for (code, slot) in states.iter_mut().enumerate() {
                    *slot = states_for(tile, code, rate);
                }
                packed.extend(pack_integer_states(&states, rate).unwrap().0);
            }
            let build_s = started.elapsed().as_secs_f64();
            let codes_m = (in_f * out_f) as f64 / 1e6;

            let packed_b = device.new_buffer_with_data(
                packed.as_ptr() as *const std::ffi::c_void,
                (packed.len() * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let input: Vec<u16> = (0..in_f)
                .map(|i| {
                    let x = i as f32;
                    f16::from_f32((x * 0.0173).sin() + (x * 0.0031).cos() + 0.25).to_bits()
                })
                .collect();
            let input_b = device.new_buffer_with_data(
                input.as_ptr() as *const std::ffi::c_void,
                (in_f * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let signs: Vec<u16> = (0..in_f)
                .map(|i| f16::from_f32(if i % 3 == 0 { 0.0021 } else { -0.0017 }).to_bits())
                .collect();
            let is_b = device.new_buffer_with_data(
                signs.as_ptr() as *const std::ffi::c_void,
                (in_f * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let os: Vec<u16> = (0..out_f)
                .map(|i| f16::from_f32(if i % 5 == 0 { -0.0031 } else { 0.0023 }).to_bits())
                .collect();
            let os_b = device.new_buffer_with_data(
                os.as_ptr() as *const std::ffi::c_void,
                (out_f * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let trans = device.new_buffer((in_f * 2) as u64, MTLResourceOptions::StorageModeShared);
            let raw = device.new_buffer((out_f * 4) as u64, MTLResourceOptions::StorageModeShared);
            let out = device.new_buffer((out_f * 2) as u64, MTLResourceOptions::StorageModeShared);

            let run = |name: &str| -> Vec<u16> {
                let mut encoder = MetalEncoder::new(&queue).unwrap();
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
                            rate: Exl3Rate::Integer(rate),
                            codebook: Exl3Codebook::Mcg,
                            pretransformed: false,
                        },
                    )
                    .unwrap();
                encoder.commit_and_wait().unwrap();
                let ptr = out.contents() as *const u16;
                unsafe { std::slice::from_raw_parts(ptr, out_f) }.to_vec()
            };

            let base = run(&format!("linear_gemv_exl3_k{rate}_mcg"));
            let counts: Vec<usize> = [2u32, 4, 8]
                .iter()
                .map(|nr0| {
                    let got = run(&format!("linear_gemv_exl3_k{rate}_mcg_r{nr0}"));
                    base.iter().zip(got.iter()).filter(|(a, b)| a != b).count()
                })
                .collect();
            let differs = counts.iter().any(|c| *c > 0);
            if differs {
                discriminates += 1;
            }
            println!(
                "{:<16}{:>4}{:>7.2}{:>7}{:>7}{:>7}{:>10.1}{:>10}",
                format!("{in_f}x{out_f}"),
                rate,
                build_s,
                counts[0],
                counts[1],
                counts[2],
                codes_m,
                if differs { "DIFFERS" } else { "identical" }
            );
        }
    }
    println!(
        "\n{discriminates} combinations discriminate; the cheapest of those is the one \
         the suite should keep"
    );
    assert!(
        discriminates > 0,
        "no shape discriminated, so this probe cannot tell a re-association from an \
         identical kernel and its verdict column is meaningless"
    );
}
