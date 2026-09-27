//! Is the in-situ matvec slowdown interleaving, or the shapes themselves?
//!
//! In the model the matvec achieves ~17 GB/s. bench_gemv measures the same
//! kernels one shape at a time and gets 40-64 GB/s. This runs the seven real
//! projections of one layer back to back in a single command buffer, which is
//! exactly the order and adjacency the model uses, and compares:
//!
//!   per-shape   : each shape timed alone, as bench_gemv does
//!   interleaved  : the same seven, in model order, in one encoder
//!
//! If interleaved is also fast, the loss comes from mixing in element kernels.
//! If interleaved is slow, the shapes themselves are the problem.

use half::f16;
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::exl3::Exl3Codebook;
use kaplumba_kernels::exl3::Exl3Rate;
use kaplumba_kernels::linear::{LinearEncodeRequest, LinearKernels};
use metal::{Buffer, Device, MTLResourceOptions};
use std::time::Instant;

const HIDDEN: usize = 5376;
const INTER: usize = 21504;
const KV_SLIDING: usize = 4096;   // 16 kv heads * 256
const Q_SLIDING: usize = 8192;    // 32 q heads * 256
const KV_FULL: usize = 2048;      // 4 kv heads * 512
const Q_FULL: usize = 16384;     // 32 q heads * 512

struct Proj {
    name: &'static str,
    packed: Buffer,
    input: Buffer,
    signs_in: Buffer,
    signs_out: Buffer,
    transformed: Buffer,
    raw: Buffer,
    output: Buffer,
    in_f: usize,
    out_f: usize,
    bytes: u64,
}

fn build(device: &Device, name: &'static str, in_f: usize, out_f: usize, seed: u32) -> Proj {
    let tiles = (in_f / 16) * (out_f / 16);
    let words = tiles * 24; // KA = 3 -> 24 words per tile
    let mut v: Vec<u16> = Vec::with_capacity(words * 2);
    let mut s = seed;
    for _ in 0..words * 2 {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        v.push((s & 0xffff) as u16);
    }
    let bytes = (v.len() * 2) as u64;
    let u16buf = |n: usize| device.new_buffer((n * 2) as u64, MTLResourceOptions::StorageModeShared);
    let x: Vec<u16> = (0..in_f).map(|i| f16::from_f32((i as f32 * 0.01).sin()).to_bits()).collect();
    let si = (in_f + 127) / 128 * 8;
    let so = out_f / 8;
    Proj {
        name,
        packed: device.new_buffer_with_data(
            v.as_ptr() as *const std::ffi::c_void, bytes, MTLResourceOptions::StorageModeShared),
        input: device.new_buffer_with_data(
            x.as_ptr() as *const std::ffi::c_void, (in_f * 2) as u64, MTLResourceOptions::StorageModeShared),
        signs_in: u16buf(si),
        signs_out: u16buf(so),
        transformed: u16buf(in_f),
        raw: device.new_buffer((out_f * 4) as u64, MTLResourceOptions::StorageModeShared),
        output: u16buf(out_f),
        in_f, out_f, bytes,
    }
}

fn fire(kernels: &LinearKernels, e: &mut MetalEncoder, p: &Proj) {
    kernels.encode_gemv(e, LinearEncodeRequest {
        packed: &p.packed, packed_offset: 0, perm_mode: 0,
        input: &p.input, input_signs: &p.signs_in, output_signs: &p.signs_out,
        transformed: &p.transformed, raw: &p.raw, output: &p.output,
        in_features: p.in_f, out_features: p.out_f,
        rate: Exl3Rate::Integer(3), codebook: Exl3Codebook::Mcg,
        pretransformed: false,
    }).unwrap();
}

fn time<F: Fn(&mut MetalEncoder)>(queue: &metal::CommandQueue, iters: u64, f: F) -> f64 {
    for _ in 0..3 { let mut e = MetalEncoder::new(queue).unwrap(); f(&mut e); e.commit_and_wait().unwrap(); }
    let mut best = f64::MAX;
    for _ in 0..6 {
        let mut e = MetalEncoder::new(queue).unwrap();
        let t = Instant::now();
        for _ in 0..iters { f(&mut e); }
        e.commit_and_wait().unwrap();
        best = best.min(t.elapsed().as_secs_f64() / iters as f64);
    }
    best
}

/// The lm_head is the same kernel at a very different shape: 5376 -> 262144,
/// 536 MB, 4.7% of the bytes but a fixed 8.5 ms of the token in earlier notes.
/// Measuring it directly is the only way to know whether that is real.
fn bench_head(device: &Device, kernels: &LinearKernels, queue: &metal::CommandQueue) {
    const VOCAB: usize = 262144;
    let head = build(device, "lm_head", HIDDEN, VOCAB, 909);
    let s = time(queue, 10, |e| fire(kernels, e, &head));
    println!(
        "\nlm_head: {} MB, {:.3} ms, {:.1} GB/s ({:.0}% of 235 GB/s streaming)",
        head.bytes as f64 / 1e6, s * 1e3, head.bytes as f64 / s / 1e9,
        head.bytes as f64 / s / 235e9 * 100.0
    );
    println!("   ideal at 125 GB/s (what a layer matvec achieves): {:.3} ms", head.bytes as f64 / 125e9 * 1e3);
}

fn main() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let kernels = LinearKernels::new().expect("kernels");

    bench_head(&device, &kernels, &queue);
    for (label, kv, q) in [("sliding layer", KV_SLIDING, Q_SLIDING), ("full layer", KV_FULL, Q_FULL)] {
        let projs = vec![
            build(&device, "q_proj",   HIDDEN, HIDDEN, 7),
            build(&device, "k_proj",   HIDDEN, kv,     101),
            build(&device, "v_proj",   HIDDEN, kv,     211),
            build(&device, "o_proj",   q,      HIDDEN, 307),
            build(&device, "gate_proj",HIDDEN, INTER,  401),
            build(&device, "up_proj",  HIDDEN, INTER,  503),
            build(&device, "down_proj",INTER, HIDDEN,  601),
        ];
        let total: u64 = projs.iter().map(|p| p.bytes).sum();

        println!("{label}: 7 projections, {:.1} MB total", total as f64 / 1e6);
        let mut per_shape = 0.0;
        for p in &projs {
            let s = time(&queue, 20, |e| fire(&kernels, e, p));
            per_shape += s;
            println!("   {:<10} {:>7.3} ms  {:>6.1} GB/s", p.name, s * 1e3, p.bytes as f64 / s / 1e9);
        }
        let combined = time(&queue, 10, |e| { for p in &projs { fire(&kernels, e, p); } });
        println!(
            "   {:<10} {:>7.3} ms  {:>6.1} GB/s   (sum of parts was {:.3} ms)",
            "ALL 7", combined * 1e3, total as f64 / combined / 1e9, per_shape * 1e3
        );
        println!("   => model in situ achieves ~17 GB/s\n");
    }
}
