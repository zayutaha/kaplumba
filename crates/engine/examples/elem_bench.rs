//! What does one non-GEMV op actually cost on the GPU?
//!
//! A decode step spends ~785 ms in the layer stack, but the linear GEMVs should
//! only account for ~220 ms of it. This measures the element and attention
//! kernels that make up the rest, in isolation, at the shapes the real model
//! uses. There are roughly 23 such ops per layer across 60 layers, so even a
//! tenth of a millisecond each would be ~140 ms per token.

use kaplumba_kernels::attention::{AttentionEncodeRequest, AttentionKernels};
use kaplumba_kernels::elements::ElementKernels;
use kaplumba_kernels::encoder::MetalEncoder;
use metal::{Device, MTLResourceOptions};
use std::sync::Arc;
use std::time::Instant;

const HIDDEN: usize = 5376;
const INTER: usize = 21504;

fn main() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let el = ElementKernels::new().expect("element kernels");
    let at: Arc<AttentionKernels> = Arc::new(AttentionKernels::new().expect("attention kernels"));
    let at_attn = at.clone();
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

    let mut time = move |name: &str, iters: u32, mut f: Box<dyn Fn(&mut MetalEncoder)>| {
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
    // T1 / Q3: attention at the two real decode shapes. 50 of 60 layers are
    // sliding (16 kv heads, head_dim 256, 1024-slot ring); 10 are full
    // (4 kv heads, head_dim 512, all cached keys).
    {
        let q = 1usize;
        for (label, keys, qh, kvh, hd) in [
            ("sliding  keys=1024 q32/kv16 d256", 1024usize, 32usize, 16usize, 256usize),
            ("full     keys=200  q32/kv4  d512", 200usize, 32usize, 4usize, 512usize),
            ("full     keys=2000 q32/kv4  d512", 2000, 32, 4, 512),
        ] {
            let qb = device.new_buffer((q * qh * hd * 2) as u64, MTLResourceOptions::StorageModeShared);
            let kb = device.new_buffer((keys * kvh * hd * 2) as u64, MTLResourceOptions::StorageModeShared);
            let vb = kb.clone();
            let qp = device.new_buffer(64, MTLResourceOptions::StorageModeShared);
            let kp = device.new_buffer((keys * 4) as u64, MTLResourceOptions::StorageModeShared);
            let ob = device.new_buffer((q * qh * hd * 2) as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                let p = kp.contents() as *mut u32;
                for i in 0..keys { *p.add(i) = i as u32; }
            }
            let handle = at_attn.clone();
            let us = time(&format!("attention {label}"), 50, Box::new(move |e| {
                handle.encode_online_attention(e, AttentionEncodeRequest {
                    queries: &qb, keys: &kb, values: &vb,
                    query_positions: &qp, key_positions: &kp, output: &ob,
                    queries_count: q, q_heads: qh, kv_heads: kvh, head_dim: hd,
                    key_count: keys, scale: 0.044194173824159216,
                    window: if kvh == 16 { 1024 } else { 0 },
                }).unwrap();
            }));
            let _ = us;
        }
    }
    println!("\n  60 attention calls per token");
}

fn static_names() {}
