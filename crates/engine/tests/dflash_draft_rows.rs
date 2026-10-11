//! spec/dflash-correctness: which logits rows become the draft tokens.
//!
//! The block is `[anchor, mask x (block_size - 1)]` and `forward_block` puts
//! `anchor_embedding` on **row 0** (`dflash.rs`: `if row == 0 { anchor } else
//! { mask }`). The anchor predicts a token the target has *already* emitted, so
//! only rows `1..=draft_rows` are real drafts.
//!
//! `draft_block` selected `0..draft_rows` -- the anchor included and the last
//! masked row dropped -- while its own doc comment and `draft_tokens`' both said
//! the anchor is dropped. Nothing asserted the selection: `dflash_golden` only
//! exercises `forward_block`, and no test in the tree called `draft_block`.
//!
//! The reference is explicit. `dflash/model.py:249` slices the drafter's output
//! with `[:, 1 - verify_size :, :]` -- the last `verify_size - 1` rows -- and
//! `:268` fills `block_output_ids[:, 1:]` from those logits, so the drafts occupy
//! slots 1.. and the anchor's row is never predicted from.
//!
//! Ignored by default: it loads the 16 GB target (for the tied head) and the
//! drafter.
//!
//! **KNOWN BLOCKER -- this test does not yet reach its assertion.** It aborts in
//! `draft_block` with:
//!
//! ```text
//! Kernels("invalid kernel input: packed buffer length 2208470208 does not match 1056964608")
//! ```
//!
//! `1056964608` is exactly `262144 * 5376 * 6 / 8`, the target head's packed size
//! at 6 bpw, so the expected length is right and the *supplied* buffer is the
//! wrong object. `draft_logits` passes `head.metal_buffer` straight to the
//! drafter's own `linear` encoder, and nothing has ever called it: `draft_block`
//! and `draft_logits` have **no callers outside `dflash.rs`** (`grep -rn` over
//! `crates/` finds only doc-comment mentions). So the target-head-to-drafter
//! plumbing has never been exercised, and this mismatch is the first thing it
//! does. Resolving it likely means passing the packed *view* (offset/length) of
//! the head rather than its whole `metal_buffer`; that is a small change once
//! someone confirms which slice the encoder wants.
//!
//! The assertion below is what the test must reach once that is fixed. It is
//! left in place rather than deleted so the failure mode is visible and the
//! fix is immediately checkable.
//!
//! ```text
//! KAPLUMBA_TEST_MODEL=~/.kaplumba/models/Gxma-repacked \
//! KAPLUMBA_DFLASH_DIR=~/.kaplumba/models/Drafter \
//!   cargo test -p kaplumba-engine --release --test dflash_draft_rows -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use half::f16;
use kaplumba_engine::dflash::{DflashConfig, DflashKvCache, DflashModel, DflashWeights};
use kaplumba_engine::model::TextModel;

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

/// Greedy argmax of one f16 logits row. Deliberately re-implemented here rather
/// than imported: a test that calls the engine's own `argmax_row` cannot catch a
/// bug in it, and `argmax_row` is private anyway.
fn argmax_row(logits: &[u16], vocab: usize, row: usize) -> u32 {
    let base = row * vocab;
    let mut best_index = 0usize;
    let mut best_value = f16::from_bits(logits[base]).to_f32();
    for index in 1..vocab {
        let value = f16::from_bits(logits[base + index]).to_f32();
        if value > best_value {
            best_value = value;
            best_index = index;
        }
    }
    best_index as u32
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; loads the 16 GB target and the drafter"]
fn draft_tokens_are_the_masked_rows_not_the_anchor() {
    let dir = drafter_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("drafter checkpoint not present at {dir:?}; skipping");
        return;
    }
    let mut target =
        TextModel::load_from_dir(Path::new(&model_path())).expect("target model");
    let device = metal::Device::system_default().expect("metal device");
    let weights = DflashWeights::load(&dir, &device).expect("drafter weights");
    let config = DflashConfig::load(&dir).expect("drafter config");
    let mut drafter = DflashModel::new(weights, config.clone()).expect("drafter model");

    target
        .enable_taps(&config.target_layer_ids)
        .expect("enable taps");
    target.reset_gpu_caches();

    let total = TOKENS.len();
    let positions: Vec<u32> = (0..total as u32).collect();
    target
        .forward_buffer(&TOKENS, &positions)
        .expect("target forward");
    let taps = target.take_taps(total).expect("take taps");
    let borrowed: Vec<&[u16]> = taps.iter().map(|tap| tap.as_slice()).collect();

    let mut cache = DflashKvCache::new(&config);
    drafter
        .append_context(&borrowed, total, total, &mut cache)
        .expect("append context");

    // `draft_block` needs the mask token's raw embedding, which `append_context`
    // does not. z-lab takes it from the target's embedding table too
    // (`_raw_input_embeddings`), so it comes from the same place as the anchor.
    let mask = target.raw_embedding(&[config.mask_token_id]);
    drafter.set_mask_embedding(&mask).expect("mask embedding");

    let anchor = target.raw_embedding(&[TOKENS[total - 1]]);
    let head = target.head();
    let draft = drafter
        .draft_block(&anchor, total, &cache, head)
        .expect("draft block");

    let vocab = head.out_features;
    let block_size = config.block_size;
    let draft_rows = config.draft_tokens();
    assert_eq!(draft_rows, block_size - 1, "a block is one anchor plus masks");

    let logits = drafter
        .draft_logits(&draft.block.hidden, head, config.final_logit_softcapping, block_size)
        .expect("draft logits");
    assert_eq!(
        logits.len(),
        block_size * vocab,
        "the head must return one logits row per block row"
    );

    let masked: Vec<u32> = (1..=draft_rows)
        .map(|row| argmax_row(&logits, vocab, row))
        .collect();
    let with_anchor: Vec<u32> = (0..draft_rows)
        .map(|row| argmax_row(&logits, vocab, row))
        .collect();

    assert_eq!(
        draft.tokens.len(),
        draft_rows,
        "the drafter must return block_size - 1 draft tokens"
    );

    // Without this the test could pass vacuously: if the two row ranges happen to
    // agree on every position, `tokens == masked` says nothing about which range
    // was used. Failing here means the block's own logits cannot distinguish the
    // two candidate selections, which is a property of this checkpoint and
    // context, not a pass.
    assert_ne!(
        masked, with_anchor,
        "rows 1..=draft_rows and rows 0..draft_rows give identical argmaxes on every \
         position, so this test cannot tell the two selections apart on this input"
    );

    assert_eq!(
        draft.tokens, masked,
        "draft_block must return the argmax of rows 1..=draft_rows (the masked rows); \
         it returned the argmax of rows 0..draft_rows, which includes the anchor's row \
         -- a token the target has already emitted -- and drops the last masked row"
    );
}
