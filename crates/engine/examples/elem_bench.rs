//! What does one non-GEMV op actually cost on the GPU?
//!
//! A decode step spends ~785 ms in the layer stack, but the linear GEMVs should
//! only account for ~220 ms of it. This measures the element and attention
//! kernels that make up the rest, in isolation, at the shapes the real model
//! uses. There are roughly 23 such ops per layer across 60 layers, so even a
//! tenth of a millisecond each would be ~140 ms per token.

use kaplumba_kernels::elements::ElementKernels;
use kaplumba_kernels::encoder::MetalEncoder;
use metal::{Device, MTLResourceOptions};
use std::time::Instant;

const HIDDEN: usize = 5376;
const INTER: usize = 21504;

fn main() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let el = ElementKernels::new().expect("element kernels");
    let buf = |n: usize| device.new_buffer((n * 2) as u64, MTLResourceOptions::StorageModeShared);

    let cases: Vec<(&str, Box<dyn Fn(&mut MetalEncoder)>)> = vec![
        ("rms_norm 1x5376", Box::new(|e: &mut MetalEncoder| {
            static_names();
            let _ = e;
        })),
    ];
    let _ = cases;

    // Buffers are created once outside the timing loop.
    let input = buf(HIDDEN);
    let weight = buf(HIDDEN);
    let output = buf(HIDDEN);
    let big_in = buf(INTER);
    let big_out = buf(INTER);

    let mut time = |name: &str, iters: u32, mut f: Box<dyn Fn(&mut MetalEncoder)>| {
        for _ in 0..5 {
            let mut e = MetalEncoder::new(&queue).unwrap();
            f(&mut e);
            e.commit_and_wait().unwrap();
        }
        let mut best = f64::MAX;
        for _ in 0..10 {
            let mut e = MetalEncoder::new(&queue).unwrap();
            let t = Instant::now();
            for _ in 0..iters {
                f(&mut e);
            }
            e.commit_and_wait().unwrap();
            best = best.min(t.elapsed().as_secs_f64() / iters as f64);
        }
        println!("  {name:<26} {:>8.1} us", best * 1e6);
        best
    };

    time("rms_norm 1x5376", 100, Box::new(|e| {
        el.encode_rms_norm(e, &input, &weight, &output, 1, HIDDEN, 1e-6).unwrap();
    }));
    time("gated_gelu 21504", 100, Box::new(|e| {
        el.encode_gated_gelu(e, &big_in, &big_in, &big_out, INTER).unwrap();
    }));
    time("residual_add 1x5376", 100, Box::new(|e| {
        el.encode_residual_add(e, &input, &output, &output, 1).unwrap();
    }));
    time("rope 8x512", 100, Box::new(|e| {
        el.encode_rope(e, &input, &output, kaplumba_kernels::elements::RopeBufferRequest {
            tokens: 1, heads: 8, head_dim: 512, offset: 100,
            theta: 1_000_000.0, partial_factor: 1.0, proportional: false,
        }).unwrap();
    }));
    time("copy_last_row 1x5376", 100, Box::new(|e| {
        el.encode_copy_last_row(e, &input, &output, 1, HIDDEN).unwrap();
    }));
    println!("\n  23 such ops x 60 layers = 1380 per token");
}

fn static_names() {}
