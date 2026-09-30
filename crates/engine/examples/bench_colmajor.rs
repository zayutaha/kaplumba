//! Achieved-bandwidth benchmark for Column-Major (repacked) GEMV layout.

use half::f16;
use kaplumba_engine::exl3::{pack_integer_states, reorder_tensor_to_colmajor};
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::exl3::Exl3Codebook;
use kaplumba_kernels::exl3::Exl3Rate;
use kaplumba_kernels::linear::{LinearEncodeRequest, LinearKernels};
use metal::{Device, MTLResourceOptions};
use std::time::Instant;

fn build(in_f: usize, out_f: usize, bits: u8, seed: u32) -> Vec<u16> {
    let tile_cols = out_f / 16;
    let tiles = (in_f / 16) * tile_cols;
    let mut packed = Vec::with_capacity(tiles * 16 * bits as usize);
    let mut s = seed;
    for _ in 0..tiles {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let states: Vec<u16> = (0..256)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s & ((1u32 << bits) - 1)) as u16
            })
            .collect();
        let (w, _) = pack_integer_states(&states, bits).unwrap();
        packed.extend(w);
    }
    packed
}

fn main() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let kernels = LinearKernels::new().expect("kernels");
    let bits = 3u8;

    println!("=== BENCHMARK: Column-Major Repacked GEMV ===");
    for (name, in_f, out_f) in [
        ("q_proj   5376x5376", 5376usize, 5376usize),
        ("up_proj  5376x21504", 5376, 21504),
        ("down     21504x5376", 21504, 5376),
        ("gate_up  5376x43008", 5376, 43008),
        ("lm_head  5376x262144", 5376, 262144),
    ] {
        let packed_orig = build(in_f, out_f, bits, 7);
        let packed = reorder_tensor_to_colmajor(&packed_orig, in_f, out_f).expect("colmajor");
        let weight_bytes = (packed.len() * 2) as f64;
        let target_bytes = 3_000_000_000u64; // ~3 GB working set
        let copies = ((target_bytes / (packed.len() as u64 * 2)).max(2)) as usize;
        let mut big: Vec<u16> = Vec::with_capacity(packed.len() * copies);
        for c in 0..copies {
            for v in packed.iter() {
                big.push(v ^ (c as u16).wrapping_mul(0x9E37));
            }
        }
        let buf = device.new_buffer_with_data(
            big.as_ptr() as *const std::ffi::c_void,
            (big.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked,
        );
        let stride = (packed.len() * 2) as u64;
        println!(
            "{name}: working set {:.2} GB over {copies} copies ({:.1} MB each)",
            big.len() as f64 * 2.0 / 1e9,
            packed.len() as f64 * 2.0 / 1e6
        );
        let input: Vec<u16> = (0..in_f)
            .map(|i| f16::from_f32((i as f32 * 0.01).sin()).to_bits())
            .collect();
        let input_b = device.new_buffer_with_data(
            input.as_ptr() as *const _,
            (in_f * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let is: Vec<u16> = vec![0; (in_f + 127) / 128 * 8];
        let is_b = device.new_buffer_with_data(
            is.as_ptr() as *const _,
            (is.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let os: Vec<u16> = vec![0; out_f / 8];
        let os_b = device.new_buffer_with_data(
            os.as_ptr() as *const _,
            (os.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let trans = device.new_buffer((in_f * 2) as u64, MTLResourceOptions::StorageModeShared);
        let raw = device.new_buffer((out_f * 16) as u64, MTLResourceOptions::StorageModeShared);
        let outb = device.new_buffer((out_f * 2) as u64, MTLResourceOptions::StorageModeShared);

        let iters = 20;
        for (k_name, label) in [
            ("linear_gemv_ishizuki_k3_colmajor", "baseline colmajor"),
            ("linear_gemv_ishizuki_k3_colmajor_stream", "stream colmajor"),
            ("linear_gemv_ishizuki_k3_colmajor_sk4", "split-k 4 colmajor"),
            ("linear_gemv_ishizuki_k3_colmajor_u4", "u4 vector colmajor"),
        ] {
            // Warmup
            for _ in 0..2 {
                let mut e = MetalEncoder::new(&queue).unwrap();
                kernels
                    .encode_gemv_named(
                        &mut e,
                        k_name,
                        &LinearEncodeRequest {
                            packed: &buf,
                            packed_offset: 0,
                            perm_mode: 0,
                            input: &input_b,
                            input_signs: &is_b,
                            output_signs: &os_b,
                            transformed: &trans,
                            raw: &raw,
                            output: &outb,
                            in_features: in_f,
                            out_features: out_f,
                            rate: Exl3Rate::Integer(bits),
                            codebook: Exl3Codebook::Mcg,
                            pretransformed: false,
                        },
                    )
                    .unwrap();
                e.commit_and_wait().unwrap();
            }

            let mut best = f64::MAX;
            for _ in 0..5 {
                let mut enc = MetalEncoder::new(&queue).unwrap();
                let t = Instant::now();
                for w in 0..iters as u64 {
                    kernels
                        .encode_gemv_named(
                            &mut enc,
                            k_name,
                            &LinearEncodeRequest {
                                packed: &buf,
                                packed_offset: (w as usize % copies) as u64 * stride,
                                perm_mode: 0,
                                input: &input_b,
                                input_signs: &is_b,
                                output_signs: &os_b,
                                transformed: &trans,
                                raw: &raw,
                                output: &outb,
                                in_features: in_f,
                                out_features: out_f,
                                rate: Exl3Rate::Integer(bits),
                                codebook: Exl3Codebook::Mcg,
                                pretransformed: false,
                            },
                        )
                        .unwrap();
                }
                enc.commit_and_wait().unwrap();
                best = best.min(t.elapsed().as_secs_f64() / iters as f64);
            }
            let secs = best;
            let gbps = weight_bytes / secs / 1e9;
            let weights = (in_f * out_f) as f64;
            println!(
                "  [{label:<15}] {:>7.2} ms  {:>6.1} GB/s  {:>5.2} e9 weights/s  ({:.1}% of 273 GB/s)",
                secs * 1e3,
                gbps,
                weights / secs / 1e9,
                gbps / 273.0 * 100.0
            );
        }
    }
}
