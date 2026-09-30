//! What does one decode attention layer cost, per layer shape, versus context?
//!
//! The engine reports a single `attention` bucket of ~2.3 ms/token summed over all
//! 60 layers, which says nothing about which layer shape is responsible or how
//! the cost moves with context. This drives the *shipped* decode path --
//! `AttentionKernels::encode_online_attention_split`, the exact two dispatches
//! `model.rs:1727` encodes for `tokens == 1` -- on buffers sized and filled the
//! way `GpuKvCache` fills them, and reports GPU-clocked own time per layer.
//!
//! It deliberately does NOT load the model: the shapes are the config, the byte
//! counts are arithmetic, and a 134 MiB buffer probe cannot collide with a
//! 12 GiB checkpoint.
//!
//! Instrument: `commit_stamped_and_wait` + `take_gpu_stamps`, i.e. the same
//! `GPUStartTime`/`GPUEndTime` own_ms `KAPLUMBA_GPU_PROFILE=1` reports, so the
//! numbers are comparable with the in-model profile.
//!
//! ## Protocol, and why each part is here
//!
//! Three things invalidated a first cut of this probe, and each fix is a
//! correction the project has already been bitten by (see AGENTS.md §5 and
//! `spec/decode-19tok-repacked-250gb/implementation_notes.md` §1):
//!
//! 1. **One encoder, not one per pass.** `commit_stamped` re-arms a command
//!    buffer and `Drop` releases it uncommitted, so a pass-per-encoder loop
//!    makes the queue block in `new_command_buffer`. Measured here as a 600 s
//!    hang; documented at `encoder.rs:657-668`.
//! 2. **Interleaved, not sequential.** A sequential sweep over context lengths
//!    measured the sliding shape at 1024 keys as 73.0 us in one run and
//!    203.6 us in another -- 2.8x, from the same binary, with the GPU free at
//!    both starts. Every variant is therefore measured in every round, and the
//!    first variant is re-measured at the end as a clock check.
//! 3. **The read window moves.** Both shapes' KV are allocated at 8192 keys
//!    (134 MiB each) and each pass reads a window at a different offset, so no
//!    pass is served from lines the previous pass left in L2. A fixed 16 MiB
//!    window measured 313 GB/s here, which is above this box's 273 GB/s spec
//!    peak and therefore obviously cache-assisted.
//!
//! Usage: `cargo run --release --example attn_bench` (via `scripts/gpuq try`).

use kaplumba_kernels::attention::{
    AttentionEncodeRequest, AttentionKernels, decode_partial_elements, decode_splits,
};
use kaplumba_kernels::encoder::MetalEncoder;
use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};

const Q_HEADS: usize = 32;
const SLIDING_KV_HEADS: usize = 16;
const SLIDING_HEAD_DIM: usize = 256;
const FULL_KV_HEADS: usize = 4;
const FULL_HEAD_DIM: usize = 512;
const WINDOW: usize = 1024;
/// Both shapes' KV are allocated this deep so the read window can move.
const KV_DEPTH: usize = 8192;
const TIMED_ROUNDS: usize = 9;
const WARMUP_ROUNDS: usize = 3;

struct Shape {
    name: &'static str,
    kv_heads: usize,
    head_dim: usize,
    window: usize,
}

struct Kv {
    keys: Buffer,
    values: Buffer,
    positions: Buffer,
}

/// Fault every page in and leave a pattern with no long runs, so the stream
/// cannot be served from a compressed run of zeros. `bw.rs:24-45` records what
/// happens if this is skipped: a never-written shared buffer reported 55 GB/s
/// and that number became the project's bandwidth denominator.
fn fill_u16(buffer: &Buffer, words: usize) {
    let ptr = buffer.contents() as *mut u8;
    let mut x: u32 = 0x9e37_79b9;
    for i in (0..words * 2).step_by(2048) {
        unsafe { std::ptr::write_unaligned(ptr.add(i) as *mut u32, x) };
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    }
    for page in (0..words * 2).step_by(4096) {
        unsafe { std::ptr::write_volatile(ptr.add(page), 0) };
    }
}

fn make_kv(device: &Device, shape: &Shape) -> Kv {
    let words = KV_DEPTH * shape.kv_heads * shape.head_dim;
    let keys = device.new_buffer((words * 2) as u64, MTLResourceOptions::StorageModeShared);
    let values = device.new_buffer((words * 2) as u64, MTLResourceOptions::StorageModeShared);
    let positions = device.new_buffer((KV_DEPTH * 4) as u64, MTLResourceOptions::StorageModeShared);
    fill_u16(&keys, words);
    fill_u16(&values, words);
    // `positions[i]` is rewritten per pass, so the window can be placed anywhere.
    let p = positions.contents() as *mut u32;
    for i in 0..KV_DEPTH {
        unsafe { *p.add(i) = i as u32 };
    }
    Kv {
        keys,
        values,
        positions,
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    v[v.len() / 2]
}

/// Byte layout per live key position: K and V, `kv_heads` heads of `head_dim`
/// fp16 each. `gpu_cache.rs:56-58` sizes the stream as `kv_heads * head_dim`,
/// and `keys`/`values` are separate equal-length buffers (`gpu_cache.rs:63-64`).
fn kv_bytes_per_key(shape: &Shape) -> usize {
    2 * shape.kv_heads * shape.head_dim * 2
}

/// Place the live window at `offset`: `positions[k] = offset + k` and the single
/// query at `offset + key_count - 1`, which is what makes every key in
/// `[0, key_count)` live for both the `k_abs <= q_abs` test and the sliding
/// window test.
fn place_window(kv: &Kv, qpos: &Buffer, offset: usize, key_count: usize) {
    let p = kv.positions.contents() as *mut u32;
    for k in 0..key_count {
        unsafe { *p.add(k) = (offset + k) as u32 };
    }
    unsafe { *(qpos.contents() as *mut u32) = (offset + key_count - 1) as u32 };
}

/// One `encode_online_attention_split` pair, GPU-clocked. Returns own_ms.
fn time_once(
    kernels: &AttentionKernels,
    encoder: &std::cell::RefCell<MetalEncoder<'_>>,
    shape: &Shape,
    kv: &Kv,
    q: &Buffer,
    qpos: &Buffer,
    out: &Buffer,
    partials: &Buffer,
    key_count: usize,
    offset: usize,
) -> f64 {
    place_window(kv, qpos, offset, key_count);
    let mut e = encoder.borrow_mut();
    kernels
        .encode_online_attention_split(
            &mut e,
            AttentionEncodeRequest {
                queries: q,
                keys: &kv.keys,
                values: &kv.values,
                query_positions: qpos,
                key_positions: &kv.positions,
                output: out,
                queries_count: 1,
                q_heads: Q_HEADS,
                kv_heads: shape.kv_heads,
                head_dim: shape.head_dim,
                key_count,
                scale: 1.0,
                window: shape.window,
            },
            partials,
        )
        .expect("encode split attention");
    e.commit_stamped_and_wait("attn").expect("commit");
    let stamps = e.take_gpu_stamps();
    assert_eq!(stamps.len(), 1, "one stamped buffer per pass");
    stamps[0].own_ms
}

/// Everything one measured variant owns, so both shapes' buffers can be
/// allocated once and shared across every variant and every round.
struct Rig<'q> {
    kv: Kv,
    q: Buffer,
    qpos: Buffer,
    out: Buffer,
    partials: Buffer,
    encoder: std::cell::RefCell<MetalEncoder<'q>>,
}

impl<'q> Rig<'q> {
    fn new(device: &Device, queue: &'q CommandQueue, shape: &Shape) -> Self {
        let q = device.new_buffer(
            (Q_HEADS * shape.head_dim * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out = device.new_buffer(
            (Q_HEADS * shape.head_dim * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let qpos = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        let partials = device.new_buffer(
            (decode_partial_elements(Q_HEADS, shape.head_dim) * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        fill_u16(&q, Q_HEADS * shape.head_dim);
        fill_u16(&out, Q_HEADS * shape.head_dim);
        let kv = make_kv(device, shape);
        let encoder = std::cell::RefCell::new(MetalEncoder::new(queue).expect("encoder"));
        Self {
            kv,
            q,
            qpos,
            out,
            partials,
            encoder,
        }
    }

    fn measure(
        &self,
        kernels: &AttentionKernels,
        shape: &Shape,
        key_count: usize,
        offset: usize,
    ) -> f64 {
        time_once(
            kernels,
            &self.encoder,
            shape,
            &self.kv,
            &self.q,
            &self.qpos,
            &self.out,
            &self.partials,
            key_count,
            offset,
        )
    }
}

struct Variant {
    shape_index: usize,
    key_count: usize,
    /// Offset rotation seed, so two variants never share a window.
    seed: usize,
}

fn main() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let kernels = AttentionKernels::new().expect("attention kernels");
    let split_env = std::env::var("KAPLUMBA_DECODE_SPLITS").ok();

    let shapes = [
        Shape {
            name: "sliding",
            kv_heads: SLIDING_KV_HEADS,
            head_dim: SLIDING_HEAD_DIM,
            window: WINDOW,
        },
        Shape {
            name: "full",
            kv_heads: FULL_KV_HEADS,
            head_dim: FULL_HEAD_DIM,
            window: 0,
        },
    ];

    // Occupancy constants, because the two layer types differ in THREADGROUP
    // MEMORY as well as in thread count, and the kernel's `qfbuf[NSG*HEAD_DIM]`
    // is what sets it (attention.metal:975).
    println!(
        "device: max threadgroup memory {} KiB, max threads/threadgroup {}",
        device.max_threadgroup_memory_length() / 1024,
        device.max_threads_per_threadgroup().width,
    );
    for shape in &shapes {
        let nsg = Q_HEADS / shape.kv_heads;
        let wbuf = 2 * nsg * 32 * 4;
        let qfbuf = nsg * shape.head_dim * 4;
        let full_layers = if shape.window == 0 { 10 } else { 50 };
        println!(
            "  {:<7} x{} layers: 32*{} = {} threads/workgroup, wbuf {} B + qfbuf {} B = {} B, \
             {} workgroups @ splits=16, {} keys/layer/token = {} MiB",
            shape.name,
            full_layers,
            nsg,
            32 * nsg,
            wbuf,
            qfbuf,
            wbuf + qfbuf,
            16 * shape.kv_heads,
            KV_DEPTH.min(if shape.window == 0 { KV_DEPTH } else { WINDOW }),
            WINDOW.min(KV_DEPTH) * kv_bytes_per_key(shape) / (1024 * 1024),
        );
    }

    let rigs: Vec<Rig<'_>> = shapes.iter().map(|s| Rig::new(&device, &queue, s)).collect();

    // ---------------------------------------------------------------- sweep
    let key_counts = [64usize, 128, 256, 512, 1024, 2048, 4096, 8192];
    let mut variants: Vec<Variant> = Vec::new();
    for (shape_index, shape) in shapes.iter().enumerate() {
        for &key_count in &key_counts {
            if key_count > WINDOW && shape.window != 0 {
                continue; // the sliding ring is 1024 deep, always
            }
            variants.push(Variant {
                shape_index,
                key_count,
                seed: variants.len() * 977,
            });
        }
    }

    // Warm the GPU: AGENTS.md records cold clocks up to 7x slow, and a cold
    // first measurement is how the first cut of this probe reported 2.8x spread
    // between runs.
    for _ in 0..300 {
        let _ = rigs[0].measure(&kernels, &shapes[0], 1024, 0);
    }

    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); variants.len()];
    let mut repeat: Vec<f64> = Vec::new();
    for round in 0..(WARMUP_ROUNDS + TIMED_ROUNDS) {
        for (index, variant) in variants.iter().enumerate() {
            let shape = &shapes[variant.shape_index];
            let span = KV_DEPTH - variant.key_count;
            let offset = if span == 0 {
                0
            } else {
                (round * 1013 + variant.seed) % (span + 1)
            };
            let ms = rigs[variant.shape_index].measure(
                &kernels,
                shape,
                variant.key_count,
                offset,
            );
            if round >= WARMUP_ROUNDS {
                samples[index].push(ms);
            }
        }
        // Clock check: the very first variant, once more, at the end.
        if round == WARMUP_ROUNDS + TIMED_ROUNDS - 1 {
            repeat.push(rigs[0].measure(&kernels, &shapes[0], 64, 4096));
        }
    }

    println!(
        "\n== interleaved sweep: {} variants, {} timed rounds each, window rotated every round",
        variants.len(),
        TIMED_ROUNDS
    );
    println!(
        "   {:<7} {:>6} {:>6} {:>8} {:>10} {:>10} {:>10} {:>8} {:>8} {:>8} {:>7} {:>7}",
        "shape",
        "keys",
        "splits",
        "groups",
        "median_us",
        "best_us",
        "worst_us",
        "spread",
        "GB/s",
        "kv_MB",
        "GFLOPS",
        "B/FLOP"
    );
    let mut first_variant: Vec<f64> = Vec::new();
    for (index, variant) in variants.iter().enumerate() {
        let shape = &shapes[variant.shape_index];
        let mut times = std::mem::take(&mut samples[index]);
        let med = median(&mut times);
        if index == 0 {
            first_variant = times.clone();
        }
        let best = *times.first().expect("timed");
        let worst = *times.last().expect("timed");
        let kv_bytes = (variant.key_count * kv_bytes_per_key(shape)) as f64;
        let splits = decode_splits(Q_HEADS, variant.key_count);
        // QK and PV each do one MAC per (query head, key, dimension), and a MAC
        // is 2 FLOP, so 4 * q_heads * keys * head_dim. GQA does not divide this
        // out: a kv head's K row is read once but every one of its
        // `q_heads / kv_heads` query heads runs its own dot product against it.
        let flops = 4.0 * Q_HEADS as f64 * variant.key_count as f64 * shape.head_dim as f64;
        println!(
            "   {:<7} {:>6} {:>6} {:>8} {:>10.1} {:>10.1} {:>10.1} {:>8.2} {:>8.1} {:>8.1} {:>7.2} {:>7.1}",
            shape.name,
            variant.key_count,
            splits,
            splits * shape.kv_heads,
            med * 1e3,
            best * 1e3,
            worst * 1e3,
            worst / best,
            kv_bytes / (med * 1e-3) / 1e9,
            kv_bytes / 1e6,
            flops / (med * 1e-3) / 1e9,
            kv_bytes / flops,
        );
    }
    if let Some(first) = repeat.first() {
        println!(
            "\n   clock check: sliding @64 was {:.1} us in the sweep, {:.1} us re-measured at the end",
            median(&mut first_variant) * 1e3,
            first * 1e3
        );
    }

    // ------------------------------------------------- split-count A/B
    if let Some(value) = &split_env {
        println!("\n(KAPLUMBA_DECODE_SPLITS={value} was set in the environment)");
    } else {
        for (shape_index, key_count) in [(1usize, 8192usize), (1, 1024), (0, 1024)] {
            ab_splits(&kernels, &rigs[shape_index], &shapes[shape_index], key_count, &[4, 8, 12, 16]);
        }
    }
}

/// Interleaved A/B over split counts at one shape and key count. The claim under
/// test is `attention.rs:52-59`: an isolated harness measured the full layers at
/// 8k running 1.91x faster at 12 splits than at 16, twice at 3 significant
/// figures, and the default was left alone because that is "a hypothesis, not a
/// measurement".
fn ab_splits(
    kernels: &AttentionKernels,
    rig: &Rig<'_>,
    shape: &Shape,
    key_count: usize,
    split_list: &[usize],
) {
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); split_list.len()];
    for round in 0..(WARMUP_ROUNDS + TIMED_ROUNDS) {
        for (index, &splits) in split_list.iter().enumerate() {
            unsafe {
                std::env::set_var("KAPLUMBA_DECODE_SPLITS", splits.to_string());
            }
            let span = KV_DEPTH - key_count;
            let offset = if span == 0 {
                0
            } else {
                (round * 1013 + index * 313) % (span + 1)
            };
            let ms = rig.measure(kernels, shape, key_count, offset);
            if round >= WARMUP_ROUNDS {
                samples[index].push(ms);
            }
        }
    }
    let kv_bytes = (key_count * kv_bytes_per_key(shape)) as f64;
    println!(
        "\n== interleaved A/B, {} shape, {key_count} keys, {TIMED_ROUNDS} rounds each",
        shape.name
    );
    println!(
        "   {:>7} {:>8} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "splits", "groups", "median_us", "best_us", "worst_us", "spread", "GB/s"
    );
    for (index, &splits) in split_list.iter().enumerate() {
        let mut times = std::mem::take(&mut samples[index]);
        let med = median(&mut times);
        let best = *times.first().expect("timed");
        let worst = *times.last().expect("timed");
        println!(
            "   {:>7} {:>8} {:>10.1} {:>10.1} {:>10.1} {:>8.2} {:>8.1}",
            splits,
            splits * shape.kv_heads,
            med * 1e3,
            best * 1e3,
            worst * 1e3,
            worst / best,
            kv_bytes / (med * 1e-3) / 1e9,
        );
    }
}
