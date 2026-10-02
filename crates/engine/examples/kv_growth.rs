//! What does the on-demand KV growth actually cost, and when does it happen?
//!
//! `GpuKvCache::ensure_capacity` (`gpu_cache.rs:203`) starts the 10
//! full-attention caches at 512 positions and steps them up by 1.25x at a time,
//! preserving contents with a host `std::ptr::copy_nonoverlapping`. That is a
//! synchronous CPU memcpy of the whole live KV prefix, on the critical path of a
//! token, at each rung of the ladder. This drives the real `GpuKvCache` --
//! the same type `DecoderLayer` holds -- with the model's real shape, walks a
//! synthetic 8k generation, and reports the capacity ladder, the copy bytes, and
//! the wall cost of every reallocation event.
//!
//! No model, no weights: `GpuKvCache` allocates its own Metal buffers, so the
//! footprint here is the KV ladder itself (1.41 GiB at 8192) and nothing else.
//!
//! It also prints the capacity the model actually reaches, because the ladder
//! is not the ladder a reader expects: the top rung clamps to the policy
//! context limit `MAX_POLICY_CAPACITY` = `DEFAULT_CONTEXT_LIMIT` = 32768
//! (`gpu_cache.rs`), not to the checkpoint's `max_position_embeddings` of
//! 262144.

use kaplumba_engine::gpu_cache::GpuKvCache;
use metal::Device;
use std::time::Instant;

/// 10 full_attention layers: 4 kv heads x head_dim 512 (`config.json`
/// `num_global_key_value_heads`, `global_head_dim`; `model.rs:2810-2819`).
const FULL_LAYERS: usize = 10;
const FULL_KV_HEADS: usize = 4;
const FULL_HEAD_DIM: usize = 512;
/// `model.rs:2826`.
const MAX_SEQUENCE_LEN: usize = 8192;
/// `model.rs:1073`.
const INITIAL_FULL_CACHE: usize = 512;
/// 50 sliding_attention layers: 16 kv heads x head_dim 256, `sliding_window` 1024.
const SLIDING_LAYERS: usize = 50;
const SLIDING_KV_HEADS: usize = 16;
const SLIDING_HEAD_DIM: usize = 256;
const SLIDING_WINDOW: usize = 1024;

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn main() {
    let device = Device::system_default().expect("Metal device");

    // One cache, walked 1..=8192 the way `model.rs:1655-1659` walks it: every
    // token of a 1-token decode chunk calls ensure_capacity(offset + 1).
    let mut full = GpuKvCache::new(
        &device,
        MAX_SEQUENCE_LEN.min(INITIAL_FULL_CACHE),
        FULL_KV_HEADS,
        FULL_HEAD_DIM,
    )
    .expect("full cache");
    let stream = full.stream_len();
    let element_bytes = stream * 2;
    println!(
        "full layer: stream_len {} ({} kv heads x {}), {} B per position for K and V",
        stream,
        FULL_KV_HEADS,
        FULL_HEAD_DIM,
        element_bytes * 2
    );
    println!(
        "resident_bytes at construction: {} MiB  ({} layers = {} MiB)",
        mib(full.resident_bytes()),
        FULL_LAYERS,
        mib(full.resident_bytes() * FULL_LAYERS)
    );

    println!("\nposition  capacity   grew?  copy_K+V_MiB/layer  all_10_MiB  wall_us");
    let mut grew = Vec::new();
    let mut samples: Vec<(usize, f64, usize)> = Vec::new();
    for position in 0..MAX_SEQUENCE_LEN {
        let before = full.capacity();
        let t = Instant::now();
        full.ensure_capacity(&device, position + 1, MAX_SEQUENCE_LEN)
            .expect("ensure_capacity");
        let dt = t.elapsed().as_secs_f64() * 1e6;
        let after = full.capacity();
        if after != before {
            // `gpu_cache.rs:191`: live = min(next_position, old capacity).
            let live = position.min(before);
            let copy = live * element_bytes * 2;
            println!(
                "{position:>8}  {after:>8}  {:>5}  {:>17.1}  {:>11.1}  {dt:>7.1}",
                "YES",
                mib(copy),
                mib(copy * FULL_LAYERS),
            );
            grew.push((position, after, copy * FULL_LAYERS));
        } else {
            samples.push((position, dt, after));
        }
        full.advance(1);
    }
    let noop: Vec<f64> = samples.iter().map(|(_, dt, _)| *dt).collect();
    let mut sorted = noop.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    println!(
        "\nno-op ensure_capacity at a fixed capacity: median {:.2} us, p95 {:.2} us, max {:.2} us \
         (n={})",
        sorted[sorted.len() / 2],
        sorted[sorted.len() * 95 / 100],
        sorted[sorted.len() - 1],
        sorted.len()
    );
    println!(
        "total reallocation events: {}; total copy across all 10 layers: {:.1} MiB",
        grew.len(),
        grew.iter().map(|(_, _, bytes)| *bytes).sum::<usize>() as f64 / (1024.0 * 1024.0)
    );
    println!(
        "resident_bytes at {} positions: {} MiB ({} layers = {} MiB = {:.3} GiB)",
        MAX_SEQUENCE_LEN,
        mib(full.resident_bytes()),
        FULL_LAYERS,
        mib(full.resident_bytes() * FULL_LAYERS),
        (full.resident_bytes() * FULL_LAYERS) as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // The cost is a set of spikes, not a slope: one per rung. Print them as a
    // share of a plausible 80 ms token so the product impact is legible. The 1.25x
    // ladder has more rungs than the doubling it replaced (19 against 6 to reach
    // 32768), but each copies less and the last one copies less still, so neither
    // the peak nor the total moves enough to matter -- see the arithmetic on
    // `growth_target` in `gpu_cache.rs`.
    println!("\nreallocation spike as a share of an 80 ms token:");
    for (position, capacity, bytes) in &grew {
        println!(
            "  position {position:>5} -> capacity {capacity:>5}: {:.1} MiB copied, \
             {:.2}% of one 80 ms token if the memcpy were entirely exposed",
            mib(*bytes),
            mib(*bytes) / 1_048_576.0 * 100.0
        );
    }

    // The sliding ring never grows: `model.rs:1655` gates ensure_capacity on
    // `full`, and `GpuKvCache::new` is handed `sliding_window` as the capacity.
    let sliding = GpuKvCache::new(
        &device,
        SLIDING_WINDOW,
        SLIDING_KV_HEADS,
        SLIDING_HEAD_DIM,
    )
    .expect("sliding cache");
    let key_count_at = |position: usize| {
        let mut cache = GpuKvCache::new(
            &device,
            SLIDING_WINDOW,
            SLIDING_KV_HEADS,
            SLIDING_HEAD_DIM,
        )
        .expect("sliding cache");
        cache.advance(position);
        cache.key_count(false)
    };
    println!(
        "\nsliding layer: capacity {} (== sliding_window), key_count at position 10 = {}, \
         at position 1024 = {}, at position 8000 = {}",
        sliding.capacity(),
        key_count_at(10),
        key_count_at(1024),
        key_count_at(8000),
    );
    println!(
        "  resident_bytes: {} MiB per layer, {} layers = {} MiB = {:.3} GiB, independent of context",
        mib(sliding.resident_bytes()),
        SLIDING_LAYERS,
        mib(sliding.resident_bytes() * SLIDING_LAYERS),
        (sliding.resident_bytes() * SLIDING_LAYERS) as f64 / (1024.0 * 1024.0 * 1024.0)
    );
}
