//! spec/dflash-correctness T8 (R6): chunked prefill streaming.
//!
//! R6 is what lets the drafter's context be built without ever holding the whole
//! prompt: stream each prefill chunk's taps into the cache and release them, so
//! peak memory is bounded by the chunk and never by the prompt length (AGENTS.md
//! §8). The invariant that makes it safe is that **how you chunk the prefill must
//! not be observable in the resulting cache.**
//!
//! This is the end-to-end version of `dflash_cache`'s
//! `dflash_cache_is_chunk_invariant`: there, the taps were synthetic, so it
//! validated the cache arithmetic. Here the taps come from the real target
//! forward through `dflash_taps`' capture path, so it validates the whole chain —
//! tap capture, `fc`+`hidden_norm` fusion, `k_proj`/`v_proj`+`k_norm`+RoPE
//! projection, and the cache — against a different chunking of the same tokens.
//!
//! A failure here means one of:
//!   - `fc` was fed across a chunk boundary (it must see one row at a time), or
//!   - RoPE positions were taken from the chunk offset instead of the absolute
//!     position, or
//!   - the tap blit mis-selected its ping-pong source for one of the chunks.
//!
//! Ignored by default: it loads the 16 GB target **and** the 769 MB drafter.
//!
//! ```text
//! KAPLUMBA_TEST_MODEL=~/.kaplumba/models/Gxma-repacked \
//! KAPLUMBA_DFLASH_DIR=~/.kaplumba/models/Drafter \
//!   cargo test -p kaplumba-engine --release --test dflash_streaming -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use kaplumba_engine::dflash::{DflashConfig, DflashKvCache, DflashModel, DflashWeights};
use kaplumba_engine::model::TextModel;

/// 32 tokens: two 16-row chunks, which is the DFlash block size and so the only
/// chunking the round will actually use.
const TOKENS: [u32; 32] = [
    2, 105, 2364, 107, 9259, 106, 107, 105, 4368, 107, 100, 45518, 107, 101, 105, 2364, 107, 9259,
    106, 107, 105, 4368, 107, 100, 45518, 107, 101, 105, 2364, 107, 9259, 106,
];

fn model_path() -> String {
    std::env::var("KAPLUMBA_TEST_MODEL").expect("KAPLUMBA_TEST_MODEL is required")
}

fn drafter_dir() -> PathBuf {
    std::env::var("KAPLUMBA_DFLASH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/zayaantaha/.kaplumba/models/Drafter"))
}

struct Harness {
    target: TextModel,
    drafter: DflashModel,
    config: DflashConfig,
}

impl Harness {
    fn build() -> Self {
        let mut target = TextModel::load_from_dir(Path::new(&model_path())).expect("target model");
        let dir = drafter_dir();
        let device = metal::Device::system_default().expect("metal device");
        let weights = DflashWeights::load(&dir, &device).expect("drafter weights");
        let config = DflashConfig::load(&dir).expect("drafter config");
        let drafter = DflashModel::new(weights, config.clone()).expect("drafter model");
        // Tap exactly the layers the drafter was trained on, in the config's
        // order: `fc` expects that exact concatenation.
        target
            .enable_taps(&config.target_layer_ids)
            .expect("enable taps");
        target.reset_gpu_caches();
        Self {
            target,
            drafter,
            config,
        }
    }

    /// Forward `ids` starting at absolute `position`, returning the captured taps
    /// as borrowable slices of `rows * hidden_size` each.
    fn taps(&mut self, ids: &[u32], position: usize) -> Vec<Vec<u16>> {
        let positions: Vec<u32> = (0..ids.len() as u32).map(|i| position as u32 + i).collect();
        self.target
            .forward_buffer(ids, &positions)
            .expect("target forward");
        self.target.take_taps(ids.len()).expect("take taps")
    }
}

fn borrow(taps: &[Vec<u16>]) -> Vec<&[u16]> {
    taps.iter().map(|tap| tap.as_slice()).collect()
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; loads the 16 GB target and the drafter"]
fn dflash_streaming_is_chunk_invariant_end_to_end() {
    let dir = drafter_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("drafter checkpoint not present at {dir:?}; skipping");
        return;
    }
    let mut harness = Harness::build();
    let layers = harness.config.num_layers;
    let total = TOKENS.len();

    // --- path A: one 32-row chunk ----------------------------------------
    let whole_taps = harness.taps(&TOKENS, 0);
    let mut whole = DflashKvCache::new(&harness.config);
    harness
        .drafter
        .append_context(&borrow(&whole_taps), total, total, &mut whole)
        .expect("whole append");

    // --- path B: two 16-row chunks, streamed ------------------------------
    // A fresh target cache is required: positions must resume, and the taps for
    // the second chunk must be produced by a forward whose own attention has
    // seen the first chunk -- which is exactly what happens when you stream.
    let mut streamed_harness = Harness::build();
    let mut streamed = DflashKvCache::new(&streamed_harness.config);
    let half = total / 2;
    for (chunk, start) in [(0usize, 0usize), (1usize, half)] {
        let ids = &TOKENS[start..start + half];
        let taps = streamed_harness.taps(ids, start);
        streamed_harness
            .drafter
            .append_context(&borrow(&taps), half, start + half, &mut streamed)
            .expect("streamed append");
        eprintln!(
            "streamed chunk {chunk}: rows {half} at position {} -> cache holds {}",
            start + half,
            streamed.total_rows()
        );
    }

    assert_eq!(whole.total_rows(), streamed.total_rows());
    assert_eq!(whole.num_layers(), layers);
    // Layers 0..4 slide and the last does not, so the two must agree per layer
    // and each layer must agree with itself across the two chunkings.
    for layer in 0..layers {
        assert_eq!(
            whole.rows(layer),
            streamed.rows(layer),
            "layer {layer} holds a different row count under the two chunkings"
        );
        assert_eq!(
            whole.first_position(layer),
            streamed.first_position(layer),
            "layer {layer} reports a different first position under the two chunkings"
        );
    }

    // The actual claim, and the reason this cannot be validated with synthetic
    // taps: every projected key and value must be byte-identical.
    let mut worst_layer = 0usize;
    let mut worst_diff = 0usize;
    for layer in 0..layers {
        let keys_differ = whole
            .keys(layer)
            .iter()
            .zip(streamed.keys(layer).iter())
            .filter(|(a, b)| a != b)
            .count();
        let values_differ = whole
            .values(layer)
            .iter()
            .zip(streamed.values(layer).iter())
            .filter(|(a, b)| a != b)
            .count();
        eprintln!(
            "layer {layer}: keys {} elem, {keys_differ} differ; values {} elem, {values_differ} differ",
            whole.keys(layer).len(),
            whole.values(layer).len()
        );
        if keys_differ.max(values_differ) > worst_diff {
            worst_diff = keys_differ.max(values_differ);
            worst_layer = layer;
        }
    }
    assert_eq!(
        worst_diff, 0,
        "layer {worst_layer} differs on {worst_diff} elements between a whole-chunk \
         and a streamed prefill of the same tokens. Chunking the prefill changed the \
         drafter's context, so a streamed round would draft from a different history \
         than a whole-prompt one"
    );
}
