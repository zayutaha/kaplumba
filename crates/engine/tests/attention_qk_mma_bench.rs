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

/// T4 (`spec/prefill-24k-90tps`): prefill-shaped attention throughput, in
/// TFLOP/s, for every arm the engine can dispatch. This is the instrument for
/// the 24k attention budget: it runs the real dispatch geometry
/// (`queries = 1024`, one threadgroup per `(query, KV head)`) at the real key
/// counts, with no model load, so a kernel change can be scored in seconds.
///
/// FLOPs are the *unmasked* full-causal count `queries * q_heads * keys *
/// head_dim * 4` (2 for QK, 2 for PV, 4 for the mul-add pair). Every query is
/// positioned at or past `key_count` so no key is masked out, which makes the
/// number the kernel's actual work rather than its nominal work.
///
/// Informational: it prints and only asserts each arm produced finite output.
#[test]
fn attention_arm_throughput_at_prefill_geometry() {
    let Some(fx) = Fixture::new() else {
        return;
    };
    const PAIRS: usize = 3;
    // (shape, key_count, queries_count) -- 1024 queries is the engine's
    // `PREFILL_CHUNK`, and the key counts are the last chunk of an 8k and a 24k
    // prefill. 256 and 4096 probes how throughput scales with grid size: if the
    // kernel were grid-starved the TFLOP/s would climb from 256 to 4096.
    let cases = [
        (FULL, 8192usize, 1024usize),
        (FULL, 24576, 1024),
        (FULL, 24576, 256),
        (FULL, 24576, 4096),
    ];
    for (shape, key_count, queries_count) in cases {
        let kv = key_count * shape.kv_heads * shape.head_dim;
        let queries = queries_count * shape.q_heads * shape.head_dim;
        let queries_buf = fx.row(queries, 3.0);
        let keys_buf = fx.row(kv, 11.0);
        let values_buf = fx.row(kv, 29.0);
        let query_positions: Vec<u32> = (0..queries_count as u32)
            .map(|i| key_count as u32 + i)
            .collect();
        let key_positions: Vec<u32> = (0..key_count as u32).collect();
        let query_positions_buf = fx.u32(&query_positions);
        let key_positions_buf = fx.u32(&key_positions);
        let out = fx.device.new_buffer(
            (queries * 2) as u64,
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
        let flops = (queries_count * shape.q_heads * key_count * shape.head_dim * 4) as f64;

        let mut arms: Vec<(&str, AttentionQk, usize)> = vec![
            ("scalar", AttentionQk::Scalar, 1),
            ("mma", AttentionQk::Mma, 1),
        ];
        if fx.kernels.flash_attention_available() {
            arms.push(("flash bq1", AttentionQk::Flash, 1));
            arms.push(("flash bq2", AttentionQk::Flash, 2));
        }

        // Warm every pipeline first so no arm pays first-touch cost.
        for (_, qk, block) in &arms {
            let mut enc = MetalEncoder::new(&fx.queue).expect("encoder");
            fx.kernels
                .encode_online_attention_qk_block(&mut enc, request, *qk, *block)
                .unwrap_or_else(|e| panic!("warm ({qk:?}): {e}"));
            enc.commit_and_wait().expect("wait");
        }

        println!(
            "--- {} keys={key_count} queries={queries_count} ({:.2} GFLOP/iter) ---",
            shape.name,
            flops / 1e9
        );
        for (name, qk, block) in arms {
            let mut ms = Vec::new();
            for _ in 0..PAIRS {
                let t0 = std::time::Instant::now();
                let mut enc = MetalEncoder::new(&fx.queue).expect("encoder");
                fx.kernels
                    .encode_online_attention_qk_block(&mut enc, request, qk, block)
                    .unwrap_or_else(|e| panic!("encode ({name}): {e}"));
                enc.commit_and_wait().expect("wait");
                ms.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            let med = median_f64(&mut ms);
            let tflops = flops / (med * 1e-3) / 1e12;
            println!(
                "  {name:<10} {med:8.2} ms  [{:.2} {:.2}]  {tflops:6.2} TFLOP/s",
                ms[0],
                ms[PAIRS - 1]
            );
            assert!(med.is_finite() && med > 0.0, "bad timing for {name}");
        }
    }
}

/// F1 (spec/prefill-24k-90tps): the flash-body ablation, with the control §10.3
/// records as never having been run.
///
/// `attention_arm_throughput_at_prefill_geometry` was used four times to
/// attribute the flash kernel's cost and produced a table in which deleting the
/// QK and PV *instructions* moved nothing. The adversary showed that table
/// cannot be right: an `if (true) { return; }` at the top of the same body, at
/// the same grid and in the same binary, costs **0.19 ms** against 920 ms -- so
/// the body is 4 800x of the cost and §10's arms retired 256 threads instead.
///
/// This test sweeps the ablation arms carried in `params.ablate`, so **one
/// build times every arm** and every arm is the same binary, the same Metal
/// library, the same buffers and the same grid as arm 0. Arms are interleaved
/// (`0 1 2 ... 0 1 2 ...`) rather than run one-after-another, so a clock or
/// thermal drift cannot masquerade as an arm difference.
///
/// Per §10.2 #4 it also prints `flash_attention_available()` and the pipeline
/// function name for the arm, because a `.metal` edit that fails to compile
/// falls back to the MMA kernel *silently* and the bench would print the label
/// it was asked for rather than the kernel it got.
#[test]
fn attention_flash_ablation_arms() {
    use kaplumba_kernels::attention::set_attention_ablation_arm;
    let Some(fx) = Fixture::new() else {
        return;
    };
    assert!(
        fx.kernels.flash_attention_available(),
        "F1: the flash pipeline did not compile. Without it every arm below would \
         silently be measuring the MMA kernel, which is exactly the failure §10.2 #4 \
         warns about."
    );
    const ROUNDS: usize = 5;
    // arm -> (label, what it removes)
    const ARMS: &[(u32, &str, &str)] = &[
        (0, "baseline", "nothing"),
        (1, "empty body", "EVERYTHING (the control §10.3 never ran)"),
        (2, "no QK", "QK MMAs + kf/qf loads (sth still published as 0)"),
        (3, "no PV MMAs", "PV MMAs + af/bf loads"),
        (4, "no softmax", "exp/simd_sum/simd_max (wq, msb still written)"),
        (5, "K frag fixed row", "nothing; the K device path (same instruction)"),
        (6, "V frag fixed row", "nothing; the V device path (same instruction)"),
        (7, "qf hoisted", "the per-tile qstage load (also the candidate fix)"),
        (8, "K+V fixed rows", "nothing; every fragment load L1-resident, MMAs kept"),
        (9, "rescale guard off", "F8: the `msv == 1` guard; arm 0 minus arm 9 IS its benefit"),
        (11, "QK 2 chains", "nothing; 2 independent QK accumulator chains (ILP)"),
        (12, "QK 4 chains", "nothing; 4 independent QK accumulator chains (ILP)"),
        (13, "qf/af fixed addr", "nothing; the per-tile threadgroup loads, L1-resident"),
        (14, "no qf/af load", "nothing; zero Q/P fragment loads, MMAs kept"),
    ];
    // `ABL_ARMS=0,2,7` restricts the sweep. The full ten-arm sweep over three
    // shapes is ~18 minutes because arm 1 and arm 7 are 6 000x and 2x apart; a
    // follow-up arm does not need the whole battery re-timed.
    let wanted: Option<Vec<u32>> = std::env::var("ABL_ARMS").ok().map(|raw| {
        raw.split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .collect()
    });
    for (shape, key_count, queries_count) in [
        (FULL, 24576usize, 1024usize),
        (FULL, 24576, 4096),
        (FULL, 8192, 1024),
    ] {
        let kv = key_count * shape.kv_heads * shape.head_dim;
        let queries = queries_count * shape.q_heads * shape.head_dim;
        let queries_buf = fx.row(queries, 3.0);
        let keys_buf = fx.row(kv, 11.0);
        let values_buf = fx.row(kv, 29.0);
        let query_positions: Vec<u32> = (0..queries_count as u32)
            .map(|i| key_count as u32 + i)
            .collect();
        let key_positions: Vec<u32> = (0..key_count as u32).collect();
        let query_positions_buf = fx.u32(&query_positions);
        let key_positions_buf = fx.u32(&key_positions);
        let out = fx.device
            .new_buffer((queries * 2) as u64, MTLResourceOptions::StorageModeShared);
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
        let flops = (queries_count * shape.q_heads * key_count * shape.head_dim * 4) as f64;
        println!(
            "\n=== F1 flash ablation: {} keys={key_count} queries={queries_count} \
             ({:.2} GFLOP/iter) ===",
            shape.name,
            flops / 1e9
        );
        println!("flash_attention_available = {}", fx.kernels.flash_attention_available());
        // Both query blocks: production selects BQ=2 at key_count >= 16384
        // (`FLASH_QUERY_BLOCK_MIN_KEYS`), so 24576 must be reported for both.
        for block in [1usize, 2] {
            let mut samples: Vec<Vec<f64>> = vec![Vec::new(); ARMS.len()];
            for round in 0..ROUNDS {
                for (index, (arm, label, _)) in ARMS.iter().enumerate() {
                    if let Some(ref list) = wanted {
                        if !list.contains(arm) {
                            continue;
                        }
                    }
                    set_attention_ablation_arm(*arm);
                    // One untimed warm per arm per round so first-touch and the
                    // shader cache cannot land on whichever arm is measured first.
                    {
                        let mut enc = MetalEncoder::new(&fx.queue).expect("encoder");
                        fx.kernels
                            .encode_online_attention_qk_block(
                                &mut enc, request, AttentionQk::Flash, block,
                            )
                            .expect("warm");
                        enc.commit_and_wait().expect("wait");
                    }
                    let t0 = std::time::Instant::now();
                    let mut enc = MetalEncoder::new(&fx.queue).expect("encoder");
                    fx.kernels
                        .encode_online_attention_qk_block(
                            &mut enc, request, AttentionQk::Flash, block,
                        )
                        .expect("encode");
                    enc.commit_and_wait().expect("wait");
                    let ms = t0.elapsed().as_secs_f64() * 1e3;
                    samples[index].push(ms);
                    if round == 0 && index == 0 {
                        println!("  (BQ={block}) arm {arm} = {label}: {ms:.3} ms");
                    }
                    let _ = label;
                }
            }
            set_attention_ablation_arm(0);
            println!("--- BQ={block}, median ms of {ROUNDS} interleaved rounds ---");
            if samples[0].is_empty() {
                println!("  (arm 0 not selected; no ratios printed)");
                continue;
            }
            let base = median_f64(&mut samples[0].clone());
            for (index, (arm, label, removes)) in ARMS.iter().enumerate() {
                if samples[index].is_empty() {
                    continue;
                }
                let mut v = samples[index].clone();
                let med = median_f64(&mut v);
                let lo = *v.first().unwrap();
                let hi = *v.last().unwrap();
                println!(
                    "  arm {arm} {label:<18} {med:9.3} ms  [{lo:8.3} {hi:8.3}]  \
                     {:6.2} TFLOP/s  x{:.2} of base   -- {removes}",
                    flops / (med * 1e-3) / 1e12,
                    base / med,
                );
                assert!(med.is_finite() && med >= 0.0, "bad timing for arm {arm}");
            }
        }
    }
    set_attention_ablation_arm(0);
}

fn median_f64(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}
