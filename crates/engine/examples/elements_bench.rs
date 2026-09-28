//! Where do the non-linear kernels go in situ?
//!
//! Attribution said: linears-only prefill wall 9.3 s, full prefill 27 s,
//! attention 0.4 s -> ~17 s in the element kernels. Standalone they should be
//! microseconds. This runs each at real prefill shapes (256-token chunk) and
//! decode shapes (1 token), alone and interleaved with the mma GEMM, to see
//! whether the in-situ slowdown is in the kernels themselves or in mixing.

use half::f16;
use kaplumba_kernels::elements::ElementKernels;
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::linear::{LinearEncodeRequest, LinearKernels};
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use metal::{Device, MTLResourceOptions};
use std::time::Instant;

const HIDDEN: usize = 5376;
const INTER: usize = 21504;
const Q_SLIDING: usize = 8192; // 32 q heads * 256
const KV_SLIDING: usize = 1024; // 4 kv heads * 256

fn bench(
    label: &str,
    queue: &metal::CommandQueue,
    iters: usize,
    f: &mut dyn FnMut(&mut MetalEncoder),
) -> f64 {
    // warmup
    for _ in 0..3 {
        let mut e = MetalEncoder::new(queue).unwrap();
        f(&mut e);
        e.commit_and_wait().unwrap();
    }
    let mut best = f64::MAX;
    for _ in 0..5 {
        let mut e = MetalEncoder::new(queue).unwrap();
        let t = Instant::now();
        for _ in 0..iters {
            f(&mut e);
        }
        e.commit_and_wait().unwrap();
        let dt = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        best = best.min(dt);
    }
    println!("{label:<44} {best:>9.4} ms");
    best
}

fn main() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let elements = ElementKernels::new().unwrap();
    let linear = LinearKernels::new().unwrap();

    let mk = |bytes: u64| device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
    let tokens = 256usize;

    // Prefill-shape buffers.
    let x = mk((tokens * HIDDEN * 2) as u64);
    let w = mk((HIDDEN * 2) as u64);
    let y = mk((tokens * HIDDEN * 2) as u64);
    let gate = mk((tokens * INTER * 2) as u64);
    let up = mk((tokens * INTER * 2) as u64);
    let act = mk((tokens * INTER * 2) as u64);
    let q = mk((tokens * Q_SLIDING * 2) as u64);
    let k = mk((tokens * KV_SLIDING * 2) as u64);
    let v = mk((tokens * KV_SLIDING * 2) as u64);
    let qpos = mk((tokens * 4) as u64);
    let kpos = mk((4096 * 4) as u64);
    let cache = mk((4096 * KV_SLIDING * 2) as u64);

    // Decode-shape buffers (1 token).
    let dx = mk((HIDDEN * 2) as u64);
    let dy = mk((HIDDEN * 2) as u64);
    let dgate = mk((INTER * 2) as u64);
    let dup = mk((INTER * 2) as u64);
    let dact = mk((INTER * 2) as u64);

    // One mma GEMM at a real shape (gate_proj: 5376 -> 21504), 256 queries.
    let tiles = (HIDDEN / 16) * (INTER / 16);
    let words = tiles * 24; // KA=3
    let mut packed_v: Vec<u16> = Vec::with_capacity(words * 2);
    let mut s = 123u32;
    for _ in 0..words * 2 {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        packed_v.push((s & 0xffff) as u16);
    }
    let packed = device.new_buffer_with_data(
        packed_v.as_ptr() as *const std::ffi::c_void,
        (packed_v.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let signs_in = mk(((HIDDEN + 127) / 128 * 8 * 2) as u64);
    let signs_out = mk(((INTER + 127) / 128 * 8 * 2) as u64);
    let transformed = mk((tokens * HIDDEN * 2) as u64);
    let gemm_raw = mk((tokens * INTER * 4) as u64);
    let gemm_out = mk((tokens * INTER * 2) as u64);
    let gemm = |e: &mut MetalEncoder| {
        linear
            .encode_gemm(
                e,
                kaplumba_kernels::linear::PrefillEncodeRequest {
                    packed: &packed,
                    packed_offset: 0,
                    perm_mode: 0,
                    input: &x,
                    input_signs: &signs_in,
                    output_signs: &signs_out,
                    transformed: &transformed,
                    raw: &gemm_raw,
                    output: &gemm_out,
                    queries: tokens,
                    in_features: HIDDEN,
                    out_features: INTER,
                    rate: Exl3Rate::Integer(3),
                    codebook: Exl3Codebook::Mcg,
                },
            )
            .unwrap();
    };

    println!("== prefill shapes, 256 tokens ==");
    let t_norm = bench(
        "rms_norm 256x5376 (one dispatch)",
        &queue,
        200,
        &mut |e| {
            elements
                .encode_rms_norm(e, &x, &w, &y, tokens, HIDDEN, 1e-5)
                .unwrap();
        },
    );
    let t_rope = bench(
        "rope 256tok x 32 heads x 256 dim",
        &queue,
        200,
        &mut |e| {
            elements
                .encode_rope(
                    e,
                    &q,
                    &q,
                    kaplumba_kernels::elements::RopeBufferRequest {
                        tokens,
                        heads: 32,
                        head_dim: 256,
                        offset: 0,
                        theta: 1_000_000.0,
                        partial_factor: 1.0,
                        proportional: false,
                    },
                )
                .unwrap();
        },
    );
    let t_gelu = bench(
        "gated_gelu 256x21504",
        &queue,
        200,
        &mut |e| {
            elements
                .encode_gated_gelu(e, &gate, &up, &act, tokens * INTER)
                .unwrap();
        },
    );
    let t_res = bench(
        "residual_add 256x5376",
        &queue,
        200,
        &mut |e| {
            elements.encode_residual_add(e, &x, &y, &y, tokens * HIDDEN).unwrap();
        },
    );
    let t_gemm = bench("mma GEMM 256x5376->21504", &queue, 50, &mut |e| gemm(e));

    // The in-situ mix: 8 norms + 2 rope + 1 gelu + 2 residual per layer, with
    // 7 GEMMs. One "layer" of element kernels, then one GEMM, repeat.
    let mixed = bench(
        "MIX: 13 element kernels + 1 GEMM (1 layer-like)",
        &queue,
        20,
        &mut |e| {
            for _ in 0..8 {
                elements.encode_rms_norm(e, &x, &w, &y, tokens, HIDDEN, 1e-5).unwrap();
            }
            for _ in 0..2 {
                elements
                    .encode_rope(
                        e,
                        &q,
                        &q,
                        kaplumba_kernels::elements::RopeBufferRequest {
                            tokens,
                            heads: 32,
                            head_dim: 256,
                            offset: 0,
                            theta: 1_000_000.0,
                            partial_factor: 1.0,
                            proportional: false,
                        },
                    )
                    .unwrap();
            }
            elements.encode_gated_gelu(e, &gate, &up, &act, tokens * INTER).unwrap();
            for _ in 0..2 {
                elements.encode_residual_add(e, &x, &y, &y, tokens * HIDDEN).unwrap();
            }
            gemm(e);
        },
    );
    let elem_sum = 8.0 * t_norm + 2.0 * t_rope + t_gelu + 2.0 * t_res;
    println!(
        "   sum of parts: elems {:.3} + gemm {:.3} = {:.3} vs mixed {:.3} ({:.2}x)",
        elem_sum,
        t_gemm,
        elem_sum + t_gemm,
        mixed,
        mixed / (elem_sum + t_gemm)
    );

    println!("== decode shapes, 1 token ==");
    bench("rms_norm 1x5376", &queue, 500, &mut |e| {
        elements.encode_rms_norm(e, &dx, &w, &dy, 1, HIDDEN, 1e-5).unwrap();
    });
    bench("gated_gelu 1x21504", &queue, 500, &mut |e| {
        elements.encode_gated_gelu(e, &dgate, &dup, &dact, INTER).unwrap();
    });
    bench("residual_add 1x5376", &queue, 500, &mut |e| {
        elements.encode_residual_add(e, &dx, &dy, &dy, HIDDEN).unwrap();
    });
    let _ = (kpos, cache, k, v, qpos);
}
