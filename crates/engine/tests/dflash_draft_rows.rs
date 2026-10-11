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
//! **EXPECTED FAIL -- and the failure is the point.** This test currently stops at
//! an assertion that records a real product bug, with the numbers measured rather
//! than guessed. Run it and read the message; it says:
//!
//! ```text
//! head: in=5376 out=262144 rate=Integer(6) packed_len=Some(528482304)
//!       host_packed_words=0 metal_buffer=Some(7529545728) metal_offset=5321075520
//! the head's metal buffer has 2208470208 B from offset 5321075520 but its trellis
//! is 1056964608 B, so 1151505600 B of trailing data follow the head in the same
//! buffer. `gemm_chunk_buffer` requires `length - offset == expected` ...
//! ```
//!
//! The head's `metal_buffer` is the **whole 7.53 GB shard**; its trellis is
//! `packed_len * 2 = 1056964608` B starting mid-buffer, and a further tensor
//! follows it. `validate_buffer_shape` (`linear.rs:4197`) demands
//! `length - offset == expected`, which holds only when the tensor is the buffer's
//! *tail*. `encode_gemv` -- the head's usual path, and the only one ever exercised
//! -- accepts the same layout, so the two validators disagree.
//!
//! Nothing caught this because `draft_block` and `draft_logits` have **no callers
//! outside `dflash.rs`** (`grep -rn` over `crates/` finds only doc-comment
//! mentions), so the target-head-to-drafter plumbing has never run once.
//!
//! Fixing it is what unblocks measuring `alpha` (see `tasks.md` D2/D3). Preferred
//! fix is an explicit packed length on `LinearBufferRequest`, which keeps the
//! equality check for every other caller; the alternative is bounding the region
//! with `packed_len` rather than the buffer's end.
//!
//! Once that lands, delete the assertion below -- the masked-rows assertions after
//! it are the test's real content and are the immediate check of the
//! `1..=draft_rows` fix.
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

    // `draft_logits` hands `head.metal_buffer` straight to the drafter's encoder,
    // whose validator wants a buffer that is exactly the packed trellis --
    // `(in/16) * (out/16) * packed_words * 2` bytes, 6 bpw for this checkpoint.
    // The model's own head buffer is a larger object and is refused:
    //
    //   packed buffer length 2208470208 does not match 1056964608
    //
    // Nothing has ever called `draft_logits`, so this was never noticed. Rebuild
    // a correctly sized device buffer from the host words to give the encoder the
    // view it expects; the numbers below are printed so the sizes are on record.
    let mut head = target.head().clone();
    let expected = (head.in_features / 16) * (head.out_features / 16);
    eprintln!(
        "head: in={} out={} rate={:?} packed_len={:?} host_packed_words={} \
         metal_buffer={:?} metal_offset={}",
        head.in_features,
        head.out_features,
        head.rate,
        head.packed_len,
        head.packed.len(),
        head.metal_buffer.as_ref().map(|buffer| buffer.length()),
        head.metal_offset,
    );
    // Root cause, recorded as an assertion so the numbers cannot drift out of
    // the test. The head's `metal_buffer` is the WHOLE shard; its trellis starts
    // at `metal_offset` and is `packed_len * 2` bytes, with a further tensor
    // after it. `gemm_chunk_buffer`'s validator
    // (`linear.rs:4197`) demands `length - offset == expected`, which only holds
    // when the tensor is the buffer's *tail* -- true for the dedicated buffers
    // the drafter's own linears get, false here. `encode_gemv`, the path the head
    // normally uses, accepts this layout, so this is an inconsistency between two
    // validators rather than a corrupt buffer.
    let buffer_len = head.metal_buffer.as_ref().map(|b| b.length()).unwrap_or(0);
    let region = buffer_len.saturating_sub(head.metal_offset);
    let trellis = head
        .packed_len
        .map(|words| (words * std::mem::size_of::<u16>()) as u64)
        .unwrap_or(0);
    assert!(
        trellis != 0,
        "the head reports no packed_len, so its trellis size is unknown"
    );
    assert!(
        region >= trellis,
        "the head's region ({region} B from offset {}) is smaller than its trellis \
         ({trellis} B), which would be a genuinely corrupt buffer rather than the \
         tail-vs-region issue",
        head.metal_offset
    );
    let offset = head.metal_offset;
    let trailing = region.saturating_sub(trellis);
    // D2, fixed: the head is NOT at the buffer's tail, which is why
    // `draft_logits` passes `packed_len` explicitly through
    // `gemm_chunk_buffer_sliced` rather than letting the buffer's end define the
    // tensor. Printed rather than asserted, because the *sizes* are properties of
    // the checkpoint while the *behaviour* -- that the logits come out right -- is
    // asserted below by the row selection and, indirectly, by this call
    // succeeding at all (it returned
    // `packed buffer length 2208470208 does not match 1056964608` before the fix).
    eprintln!(
        "head trellis {trellis} B at offset {offset} inside a {region} B region \
         ({trailing} B of trailing tensors); sliced validation used"
    );
    assert!(
        trellis > 0 && region >= trellis,
        "the head must report a non-empty trellis that its buffer region covers"
    );
    let _ = expected;
    let head = &head;
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
