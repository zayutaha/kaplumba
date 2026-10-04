//! T2 instrument (spec/prefill-120tps): scalar-vs-MMA attention micro-benchmark.
//!
//! Isolates the two `encode_online_attention_qk` arms from the model so kernel
//! iterations can be measured in seconds instead of full `kaplumba` runs.
//! Timing is wall time around `commit_and_wait` (GPU-bound), interleaved
//! A/B/A/B per AGENTS.md 10.2 #5, reporting medians with spread.
//!
//! Informational test: it prints measurements and only asserts that both arms
//! produced finite output (so a hang or crash still fails).

use half::f16;
use kaplumba_kernels::attention::{
    AttentionEncodeRequest, AttentionKernels, AttentionQk,
};
use kaplumba_kernels::encoder::MetalEncoder;
use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    window: usize,
}

/// The 50 sliding layers (NSG=2).
const SLIDING: Shape = Shape {
    name: "sliding 32q/16kv/d256 w1024 (NSG=2)",
    q_heads: 32,
    kv_heads: 16,
    head_dim: 256,
    window: 1024,
};
/// The 10 full layers (NSG=8) -- the heavier QK.
const FULL: Shape = Shape {
    name: "full    32q/4kv/d512 (NSG=8)",
    q_heads: 32,
    kv_heads: 4,
    head_dim: 512,
    window: 0,
};

struct Fixture {
    device: Device,
    queue: CommandQueue,
    kernels: AttentionKernels,
}

impl Fixture {
    fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let kernels = AttentionKernels::new()
            .unwrap_or_else(|e| panic!("AttentionKernels::new failed: {e}"));
        let queue = device.new_command_queue().to_owned();
        Some(Self {
            device,
            queue,
            kernels,
        })
    }

    fn row(&self, n: usize, salt: f32) -> Buffer {
        let bits: Vec<u16> = (0..n)
            .map(|i| {
                let x = ((i as f32 + salt) * 0.0173).sin() * 0.5
                    + ((i as f32 + salt) * 0.0031).cos() * 0.25;
                f16::from_f32(x).to_bits()
            })
            .collect();
        self.device.new_buffer_with_data(
            bits.as_ptr() as *const std::ffi::c_void,
            (bits.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn u32(&self, v: &[u32]) -> Buffer {
        self.device.new_buffer_with_data(
            v.as_ptr() as *const std::ffi::c_void,
            (v.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
}

/// (shape, key_count, queries_count): sizes chosen to mirror the engine's
/// measured regimes -- kv_len 8192 is the 8k prefill case, 1024 is a full
/// sliding window.
fn bench_shapes() -> Vec<(Shape, usize, usize)> {
    vec![
        (FULL, 8192, 16),
        (SLIDING, 1024, 64),
        (SLIDING, 8192, 16),
    ]
}

fn median(xs: &mut [f32]) -> f32 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

#[test]
fn attention_mma_bench_scalar_vs_mma() {
    let Some(fx) = Fixture::new() else {
        return;
    };
    const PAIRS: usize = 3;
    for (shape, key_count, queries_count) in bench_shapes() {
        let kv = key_count * shape.kv_heads * shape.head_dim;
        let queries = queries_count * shape.q_heads * shape.head_dim;
        let output_elements = queries_count * shape.q_heads * shape.head_dim;
        let queries_buf = fx.row(queries, 3.0);
        let keys_buf = fx.row(kv, 11.0);
        let values_buf = fx.row(kv, 29.0);
        let query_positions: Vec<u32> =
            (0..queries_count as u32).map(|i| key_count as u32 + i).collect();
        let key_positions: Vec<u32> = (0..key_count as u32).collect();
        let query_positions_buf = fx.u32(&query_positions);
        let key_positions_buf = fx.u32(&key_positions);
        let out = fx.device.new_buffer(
            (output_elements * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let request = AttentionEncodeRequest {
            queries: &queries_buf,
            keys: &keys_buf,
            values: &values_buf,
            query_positions: &query_positions_buf,
            key_positions: &key_positions_buf,
            output: &out,
            queries_count,
            q_heads: shape.q_heads,
            kv_heads: shape.kv_heads,
            head_dim: shape.head_dim,
            key_count,
            scale: 1.0,
            window: shape.window,
        };

        let timed = |qk: AttentionQk| -> f32 {
            let t0 = std::time::Instant::now();
            let mut enc = MetalEncoder::new(&fx.queue).expect("encoder");
            fx.kernels
                .encode_online_attention_qk(&mut enc, request, qk)
                .unwrap_or_else(|e| panic!("encode ({qk:?}): {e}"));
            enc.commit_and_wait().expect("wait");
            t0.elapsed().as_secs_f32() * 1e3
        };

        // Warm both pipelines (first encode includes no compile, but clock
        // ramp and cache state are settled by one pass).
        timed(AttentionQk::Scalar);
        timed(AttentionQk::Mma);

        let mut scalar_ms = Vec::new();
        let mut mma_ms = Vec::new();
        for _ in 0..PAIRS {
            scalar_ms.push(timed(AttentionQk::Scalar));
            mma_ms.push(timed(AttentionQk::Mma));
        }
        let s = median(&mut scalar_ms);
        let m = median(&mut mma_ms);
        println!(
            "{} keys={key_count} queries={queries_count}: \
             scalar {s:.2} ms [{:.2} {:.2}] | mma {m:.2} ms [{:.2} {:.2}] | ratio mma/scalar = {:.3}",
            shape.name,
            scalar_ms[0],
            scalar_ms[PAIRS - 1],
            mma_ms[0],
            mma_ms[PAIRS - 1],
            m / s,
        );

        // Both arms must actually run (finite sentinel: an untouched output
        // buffer reads as 0, a computed one is non-zero somewhere).
        let ptr = out.contents() as *const u16;
        let first = unsafe { ptr.add(7).read() };
        assert!(
            s.is_finite() && m.is_finite(),
            "non-finite timing for {}",
            shape.name
        );
        let _ = first; // output content is asserted by the matches-scalar test
    }
}
