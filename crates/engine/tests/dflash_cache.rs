//! spec/dflash-correctness T6 (R13) and T8 (R6): the drafter's context cache.
//!
//! Three invariants, each of which is a way the cache can be subtly wrong:
//!
//! 1. **Chunk invariance (R6).** A 64-row context fed as one chunk and as four
//!    16-row chunks must produce byte-identical K/V. This is the whole reason
//!    chunked prefill streaming is safe: if it holds, the peak memory is bounded
//!    by the chunk and never by the prompt, and nothing downstream can tell the
//!    difference. If it fails, the taps were being fused across a row boundary
//!    (`fc` sees a different input) or positions were off by the chunk offset.
//!
//! 2. **Rewind equivalence (R13).** Dropping the last `k` rows must leave
//!    exactly the state a fresh cache would have after appending `n - k` rows.
//!    A rejected speculative tail is only cheap if the cache can be rewound
//!    rather than rebuilt.
//!
//! 3. **The ring keeps absolute positions.** Once the context passes
//!    `sliding_window`, layers 0..4 hold the trailing window and the last layer
//!    holds everything, and each layer must still report where its surviving
//!    rows really are. The mask is on position *distance*, so a ring that
//!    renumbers its rows from zero silently attends the wrong keys.
//!
//! Host-side weight reads plus GPU kernels; no target model, no `scripts/gpuq`.

use std::path::PathBuf;

use half::f16;
use kaplumba_engine::dflash::{DflashConfig, DflashKvCache, DflashModel, DflashWeights};

fn dir() -> PathBuf {
    std::env::var("KAPLUMBA_DFLASH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/zayaantaha/.kaplumba/models/Drafter"))
}

fn pseudo(seed: u64, count: usize, scale: f32) -> Vec<u16> {
    let _ = seed;
    (0..count)
        .map(|index| {
            let t = index as f32;
            let value = (core::f32::consts::TAU * (t / 64.0)).sin()
                + 0.7 * (core::f32::consts::TAU * (t / 256.0)).sin()
                + 0.4 * (core::f32::consts::TAU * (t / 997.0)).sin()
                + 0.15 * (core::f32::consts::TAU * (t / 4096.0)).sin();
            f16::from_f32(value * scale).to_bits()
        })
        .collect()
}

struct Harness {
    model: DflashModel,
    config: DflashConfig,
}

impl Harness {
    fn build() -> Option<Self> {
        let dir = dir();
        if !dir.join("model.safetensors").exists() {
            eprintln!("drafter checkpoint not present at {dir:?}; skipping");
            return None;
        }
        let device = metal::Device::system_default().expect("metal device");
        let weights = DflashWeights::load(&dir, &device).expect("drafter weights");
        let config = DflashConfig::load(&dir).expect("drafter config");
        let model = DflashModel::new(weights, config.clone()).expect("drafter model");
        Some(Self { model, config })
    }

    /// `n_taps` synthetic tap buffers covering `rows` rows.
    fn taps(&self, rows: usize) -> Vec<Vec<u16>> {
        (0..self.config.num_taps())
            .map(|tap| pseudo(0x7A9 ^ tap as u64, rows * self.config.hidden_size, 0.35))
            .collect()
    }
}

fn borrow(taps: &[Vec<u16>]) -> Vec<&[u16]> {
    taps.iter().map(|tap| tap.as_slice()).collect()
}

/// How many rows the cache holds for one layer, and which absolute position its
/// first row has.
fn shape(cache: &DflashKvCache) -> Vec<(usize, usize)> {
    (0..cache.num_layers())
        .map(|layer| (cache.rows(layer), cache.first_position(layer)))
        .collect()
}

#[test]
fn dflash_cache_is_chunk_invariant() {
    let Some(harness) = Harness::build() else {
        return;
    };
    const TOTAL: usize = 64;
    const CHUNK: usize = 16;

    let taps = harness.taps(TOTAL);

    // One 64-row call.
    let mut whole = DflashKvCache::new(&harness.config);
    harness
        .model
        .append_context(&borrow(&taps), TOTAL, TOTAL, &mut whole)
        .expect("whole append");

    // Four 16-row calls over the same rows.
    let mut chunked = DflashKvCache::new(&harness.config);
    let mut position = CHUNK;
    for start in (0..TOTAL).step_by(CHUNK) {
        let slice: Vec<Vec<u16>> = taps
            .iter()
            .map(|tap| {
                let width = harness.config.hidden_size;
                tap[start * width..(start + CHUNK) * width].to_vec()
            })
            .collect();
        harness
            .model
            .append_context(&borrow(&slice), CHUNK, position, &mut chunked)
            .expect("chunk append");
        position += CHUNK;
    }

    assert_eq!(
        shape(&whole),
        shape(&chunked),
        "chunked and whole appends disagree about rows/positions"
    );
    assert_eq!(whole.total_rows(), chunked.total_rows());
    for layer in 0..harness.config.num_layers {
        assert_eq!(
            whole.keys(layer),
            chunked.keys(layer),
            "layer {layer} keys differ between a whole and a chunked append"
        );
        assert_eq!(
            whole.values(layer),
            chunked.values(layer),
            "layer {layer} values differ between a whole and a chunked append"
        );
    }
}

#[test]
fn dflash_cache_rewind_equals_never_having_appended() {
    let Some(harness) = Harness::build() else {
        return;
    };
    const TOTAL: usize = 48;
    const KEEP: usize = 32;
    const DROPPED: usize = TOTAL - KEEP;

    let taps = harness.taps(TOTAL);
    let mut rewound = DflashKvCache::new(&harness.config);
    harness
        .model
        .append_context(&borrow(&taps), TOTAL, TOTAL, &mut rewound)
        .expect("append");
    harness
        .model
        .rewind_context(DROPPED, &mut rewound)
        .expect("rewind");

    // The same context, never over-appended.
    let mut fresh = DflashKvCache::new(&harness.config);
    let kept: Vec<Vec<u16>> = taps
        .iter()
        .map(|tap| {
            let width = harness.config.hidden_size;
            tap[..KEEP * width].to_vec()
        })
        .collect();
    harness
        .model
        .append_context(&borrow(&kept), KEEP, KEEP, &mut fresh)
        .expect("fresh append");

    assert_eq!(rewound.total_rows(), KEEP);
    assert_eq!(
        shape(&rewound),
        shape(&fresh),
        "a rewound cache does not look like one that never over-appended"
    );
    for layer in 0..harness.config.num_layers {
        assert_eq!(
            rewound.keys(layer),
            fresh.keys(layer),
            "layer {layer} keys differ after rewind"
        );
        assert_eq!(
            rewound.values(layer),
            fresh.values(layer),
            "layer {layer} values differ after rewind"
        );
    }
}

#[test]
fn dflash_cache_ring_keeps_absolute_positions() {
    let Some(harness) = Harness::build() else {
        return;
    };
    let window = harness.config.sliding_window;
    if window == 0 {
        eprintln!("drafter has no sliding window; nothing to wrap");
        return;
    }
    // Two windows' worth, so the ring wraps at least once.
    let total = window + window / 2;
    let taps = harness.taps(total);
    let mut cache = DflashKvCache::new(&harness.config);
    harness
        .model
        .append_context(&borrow(&taps), total, total, &mut cache)
        .expect("append");

    let last = harness.config.num_layers - 1;
    for layer in 0..harness.config.num_layers {
        let rows = cache.rows(layer);
        let first = cache.first_position(layer);
        assert_eq!(
            cache.keys(layer).len(),
            rows * cache.kv_width(),
            "layer {layer} keys are not `rows * kv_width`"
        );
        if layer == last {
            // The last layer attends everything, so nothing was evicted.
            assert_eq!(rows, total, "the last layer must not evict");
            assert_eq!(first, 0);
        } else {
            // A sliding layer keeps the trailing window and reports where the
            // surviving rows really are -- NOT 0.
            assert_eq!(rows, window, "layer {layer} should hold exactly the window");
            assert_eq!(
                first,
                total - window,
                "layer {layer} reports a first position that renumbers its rows"
            );
        }
    }

    // R6/R20: the sliding layers' footprint is bounded by the window, not by the
    // context. This is the property that keeps the cache off the memory budget's
    // critical path.
    for layer in 0..last {
        assert!(
            cache.keys(layer).len() <= window * cache.kv_width(),
            "layer {layer} grew past its window"
        );
    }
}
