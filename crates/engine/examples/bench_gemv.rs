//! Achieved-bandwidth probe for the EXL3 decode GEMV.
//!
//! Answers one question: is decode memory-bound or compute-bound? Compare
//! achieved GB/s against the ~273 GB/s the M4 Pro can do. If achieved is far
//! below peak, the cost is in code extraction / codebook, not in moving bytes,
//! and raising the rate ceiling is pointless.

use half::f16;
use kaplumba_kernels::exl3::Exl3Codebook;
use kaplumba_kernels::exl3::Exl3Rate;
use kaplumba_kernels::linear::{LinearEncodeRequest, LinearKernels};
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_engine::exl3::pack_integer_states;
use metal::{Device, MTLResourceOptions};
use std::time::Instant;

fn build(in_f: usize, out_f: usize, bits: u8, seed: u32) -> Vec<u16> {
    let tile_rows = 16;
    let tile_cols = out_f / 16;
    let tiles = (in_f / 16) * tile_cols;
    let mut packed = Vec::with_capacity(tiles * 16 * bits as usize);
    let mut s = seed;
    for _ in 0..tiles {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let states: Vec<u16> = (0..256)
            .map(|_| { s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223); (s & ((1u32 << bits) - 1)) as u16 })
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
    // Real Gemma 4 shapes.
    // Global warmup: the first shape measured eats the GPU clock ramp, which made
    // q_proj vary by 70% between runs and produced bogus before/after comparisons.
    {
        let in_f = 5376usize;
        let out_f = 5376usize;
        let packed = build(in_f, out_f, bits, 3);
        let buf = device.new_buffer_with_data(
            packed.as_ptr() as *const std::ffi::c_void,
            (packed.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared);
        let input: Vec<u16> = (0..in_f).map(|i| f16::from_f32((i as f32 * 0.01).sin()).to_bits()).collect();
        let input_b = device.new_buffer_with_data(input.as_ptr() as *const std::ffi::c_void, (in_f * 2) as u64, MTLResourceOptions::StorageModeShared);
        let is: Vec<u16> = vec![0; (in_f + 127) / 128 * 8];
        let is_b = device.new_buffer_with_data(is.as_ptr() as *const std::ffi::c_void, (is.len() * 2) as u64, MTLResourceOptions::StorageModeShared);
        let os: Vec<u16> = vec![0; out_f / 8];
        let os_b = device.new_buffer_with_data(os.as_ptr() as *const std::ffi::c_void, (os.len() * 2) as u64, MTLResourceOptions::StorageModeShared);
        let trans = device.new_buffer((in_f * 2) as u64, MTLResourceOptions::StorageModeShared);
        let raw = device.new_buffer((out_f * 2) as u64, MTLResourceOptions::StorageModeShared);
        let outb = device.new_buffer((out_f * 2) as u64, MTLResourceOptions::StorageModeShared);
        for _ in 0..40 {
            let mut e = MetalEncoder::new(&queue).unwrap();
            kernels.encode_gemv(&mut e, LinearEncodeRequest {
                packed: &buf, packed_offset: 0, perm_mode: 0,
                input: &input_b, input_signs: &is_b, output_signs: &os_b,
                transformed: &trans, raw: &raw, output: &outb,
                in_features: in_f, out_features: out_f,
                rate: Exl3Rate::Integer(bits), codebook: Exl3Codebook::Mcg,
            pretransformed: false,
        }).unwrap();
            e.commit_and_wait().unwrap();
        }
        println!("(warmed)");
    }

    for (name, in_f, out_f) in [
        ("q_proj   5376x5376", 5376usize, 5376usize),
        ("up_proj  5376x21504", 5376, 21504),
        ("down     21504x5376", 21504, 5376),
        ("lm_head  5376x262144", 5376, 262144),
    ] {
        // Cold measurement. A single copy of a 10-43 MB projection is L2-resident
        // on this part, so timing one in isolation measures cache bandwidth, not
        // DRAM -- which is exactly what the model does *not* get, because 60
        // layers stream 11 GB per token and evict everything. Allocate enough
        // copies to exceed L2 and rotate the offset so every pass is cold.
        let packed = build(in_f, out_f, bits, 7);
        let weight_bytes = (packed.len() * 2) as f64;
        let target_bytes = 3_000_000_000u64; // ~3 GB working set
        let copies = ((target_bytes / (packed.len() as u64 * 2)).max(2)) as usize;
        let mut big: Vec<u16> = Vec::with_capacity(packed.len() * copies);
        for c in 0..copies {
            for (i, v) in packed.iter().enumerate() {
                big.push(v ^ (c as u16).wrapping_mul(0x9E37));
            }
        }
        let buf = device.new_buffer_with_data(
            big.as_ptr() as *const std::ffi::c_void,
            (big.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let stride = (packed.len() * 2) as u64;
        println!(
            "{name}: working set {:.2} GB over {copies} copies ({:.1} MB each)",
            big.len() as f64 * 2.0 / 1e9,
            packed.len() as f64 * 2.0 / 1e6
        );
        let input: Vec<u16> = (0..in_f).map(|i| f16::from_f32((i as f32 * 0.01).sin()).to_bits()).collect();
        let input_b = device.new_buffer_with_data(input.as_ptr() as *const _, (in_f * 2) as u64, MTLResourceOptions::StorageModeShared);
        let is: Vec<u16> = vec![0; (in_f + 127) / 128 * 8];
        let is_b = device.new_buffer_with_data(is.as_ptr() as *const _, (is.len() * 2) as u64, MTLResourceOptions::StorageModeShared);
        let os: Vec<u16> = vec![0; out_f / 8];
        let os_b = device.new_buffer_with_data(os.as_ptr() as *const _, (os.len() * 2) as u64, MTLResourceOptions::StorageModeShared);
        let trans = device.new_buffer((in_f * 2) as u64, MTLResourceOptions::StorageModeShared);
        let raw = device.new_buffer((out_f * 2) as u64, MTLResourceOptions::StorageModeShared);
        let outb = device.new_buffer((out_f * 2) as u64, MTLResourceOptions::StorageModeShared);

        let iters = 20;
        // warmup
        for _ in 0..3 {
            let mut e = MetalEncoder::new(&queue).unwrap();
            kernels.encode_gemv(&mut e, LinearEncodeRequest {
                packed: &buf, packed_offset: 0, perm_mode: 0,
                input: &input_b, input_signs: &is_b, output_signs: &os_b,
                transformed: &trans, raw: &raw, output: &outb,
                in_features: in_f, out_features: out_f,
                rate: Exl3Rate::Integer(bits), codebook: Exl3Codebook::Mcg,
            pretransformed: false,
        }).unwrap();
            e.commit_and_wait().unwrap();
        }
        // Best of several batches: the minimum is the least contaminated by
        // scheduling noise and clock ramping.
        let mut best = f64::MAX;
        for _ in 0..5 {
            let mut enc = MetalEncoder::new(&queue).unwrap();
            let t = Instant::now();
            for w in 0..iters as u64 {
                kernels.encode_gemv(&mut enc, LinearEncodeRequest {
                    packed: &buf, packed_offset: (w as usize % copies) as u64 * stride, perm_mode: 0,
                    input: &input_b, input_signs: &is_b, output_signs: &os_b,
                    transformed: &trans, raw: &raw, output: &outb,
                    in_features: in_f, out_features: out_f,
                    rate: Exl3Rate::Integer(bits), codebook: Exl3Codebook::Mcg,
            pretransformed: false,
        }).unwrap();
            }
            enc.commit_and_wait().unwrap();
            best = best.min(t.elapsed().as_secs_f64() / iters as f64);
        }
        let secs = best;
        let gbps = weight_bytes / secs / 1e9;
        let weights = (in_f * out_f) as f64;
        println!(
            "{name}: {:>7.2} ms  {:>6.1} GB/s  {:>5.2} e9 weights/s  ({:.1}% of 273 GB/s)",
            secs * 1e3, gbps, weights / secs / 1e9, gbps / 273.0 * 100.0
        );
    }
}
