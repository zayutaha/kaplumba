//! spec/dflash-correctness D3: measure `alpha`, the one number that decides the spec.
//!
//! Speculative decoding pays only if per-token acceptance is high enough. With
//! `gamma` draft tokens a round emits `sum_{j=0}^{gamma} alpha^j` tokens, so against
//! the measured 15.25 tok/s plain-decode baseline and the measured round times
//! (b16 0.6324 s, b32 0.7517 s):
//!
//! | alpha | t/s b16 | t/s b32 |
//! |---|---|---|
//! | 0.90 | 12.9 | 12.8 | <- slower than plain decode
//! | 0.94 | 16.6 | 19.1 |
//! | 0.977 | 21.4 | 30.4 |
//! | 1.00 | 25.3 | 42.6 |
//!
//! **break-even is 0.919 at b32 and 30 t/s needs 0.977.** For temperature > 0 the
//! verifier is speculative sampling, under which per-token acceptance is exactly
//! `alpha = sum_x min(p(x), q(x)) = 1 - TV(p, q)`.
//!
//! ## Why this prints a matrix and two tap conventions rather than one number
//!
//! z-lab's round places the block at positions `start .. start+block_size-1`, with
//! row 0 the anchor (the token at `start`, already emitted) and rows `1..` the
//! drafts, so **row `r` predicts position `start + r`**. Comparing row `r` against
//! the target's distribution for `start + r` is the diagonal.
//!
//! Three earlier versions of this probe were wrong in ways only the numbers show,
//! and each is worth recording:
//!
//! 1. It read `last_logits()` instead of `forward_buffer`'s return value.
//!    `last_logits` reads `scratch.capped`, which `encode_lm_head` writes only on
//!    the softcap path, so it can hold a previous forward's data.
//! 2. It fed the drafter a synthetic token-id soup. The positive control caught it:
//!    two *adjacent positions of the target itself* shared 4 of 64 top-64 tokens.
//! 3. It compared across a one-position shift, which shows up as an off-diagonal
//!    ridge; the matrix is printed so the shift is visible rather than assumed.
//!
//! It also measures both tap-layer conventions, because z-lab's
//! `extract_context_feature` applies `offset = 1` (`dflash/model.py:41`) while the
//! engine taps `config.target_layer_ids` verbatim.
//!
//! Ignored by default: loads the 16 GB target and the drafter.
//!
//! ```text
//! KAPLUMBA_TEST_MODEL=~/.kaplumba/models/Gxma-repacked \
//! KAPLUMBA_DFLASH_DIR=~/.kaplumba/models/Drafter \
//!   cargo test -p kaplumba-engine --release --test dflash_alpha_probe -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use half::f16;
use kaplumba_engine::dflash::{DflashConfig, DflashKvCache, DflashModel, DflashWeights};
use kaplumba_engine::model::TextModel;
use kaplumba_engine::tokenizer::GemmaTokenizer;

/// Real prose. A synthetic token-id soup drives the model into a high-entropy state
/// where consecutively predicted distributions barely overlap, which invalidates any
/// `alpha` measured there -- and the positive control proved it.
const PROMPT: &str = "Explain the theory of relativity in simple terms. Special \
relativity says the speed of light is the same for every observer, so time slows \
down for anything moving quickly and lengths contract along the direction of \
motion. General relativity extends this to gravity, describing it not as a force \
but as the curvature of spacetime produced by mass and energy.";

/// The requested sampler: temperature 1.0, top_k 64.
const TOP_K: usize = 64;

/// Target positions collected, and therefore how many drafter rows can be compared.
const COLLECT: usize = 6;

fn model_path() -> String {
    std::env::var("KAPLUMBA_TEST_MODEL").expect("KAPLUMBA_TEST_MODEL is required")
}

fn drafter_dir() -> PathBuf {
    std::env::var("KAPLUMBA_DFLASH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/zayaantaha/.kaplumba/models/Drafter"))
}

/// Softmax at temperature 1, truncated to the `k` largest entries and renormalised
/// -- what `_sampling_probs` does in `dflash/model.py`.
fn top_k_distribution(logits: &[u16], k: usize) -> Vec<(u32, f64)> {
    let values: Vec<f64> = logits
        .iter()
        .map(|bits| f16::from_bits(*bits).to_f32() as f64)
        .collect();
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut probs: Vec<f64> = values.iter().map(|v| (v - max).exp()).collect();
    let total: f64 = probs.iter().sum();
    if !(total > 0.0) {
        return Vec::new();
    }
    for prob in probs.iter_mut() {
        *prob /= total;
    }
    let mut order: Vec<u32> = (0..probs.len() as u32).collect();
    order.sort_by(|a, b| {
        probs[*b as usize]
            .partial_cmp(&probs[*a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    });
    order.truncate(k);
    let kept: f64 = order.iter().map(|token| probs[*token as usize]).sum();
    if !(kept > 0.0) {
        return Vec::new();
    }
    order
        .into_iter()
        .map(|token| (token, probs[token as usize] / kept))
        .collect()
}

/// `alpha = sum_x min(p(x), q(x))` over the union of the two supports.
fn acceptance(p: &[(u32, f64)], q: &[(u32, f64)]) -> f64 {
    let mut map: HashMap<u32, (f64, f64)> = HashMap::new();
    for (token, prob) in p {
        map.entry(*token).or_insert((0.0, 0.0)).0 = *prob;
    }
    for (token, prob) in q {
        map.entry(*token).or_insert((0.0, 0.0)).1 = *prob;
    }
    map.values().map(|(a, b)| a.min(*b)).sum()
}

fn argmax(logits: &[u16]) -> u32 {
    let mut best = 0usize;
    let mut best_value = f16::from_bits(logits[0]).to_f32();
    for (index, bits) in logits.iter().enumerate().skip(1) {
        let value = f16::from_bits(*bits).to_f32();
        if value > best_value {
            best_value = value;
            best = index;
        }
    }
    best as u32
}

fn top3(dist: &[(u32, f64)]) -> String {
    dist.iter()
        .take(3)
        .map(|(token, prob)| format!("{token}:{prob:.4}"))
        .collect::<Vec<_>>()
        .join(" ")
}

struct Case {
    mean_diagonal: f64,
    best_alpha: f64,
    best_row: usize,
    best_t: usize,
    drafts: Vec<u32>,
    top_target: String,
}

/// Prefill, build the drafter's context from `layers`, chain a few target steps, draft
/// one block, and return the acceptance numbers for the round's diagonal.
fn measure_case(
    target: &mut TextModel,
    drafter: &DflashModel,
    tokens: &[u32],
    layers: &[usize],
    config: &DflashConfig,
) -> Case {
    let total = tokens.len();
    target.reset_gpu_caches();
    target.enable_taps(layers).expect("enable taps");
    let positions: Vec<u32> = (0..total as u32).collect();

    // `forward_buffer` RETURNS the last row's logits -- the supported read.
    let mut logits = target
        .forward_buffer(tokens, &positions)
        .expect("target forward");
    let taps = target.take_taps(total).expect("take taps");
    let borrowed: Vec<&[u16]> = taps.iter().map(|tap| tap.as_slice()).collect();

    // DIAGNOSTIC (D4): are the taps distinct, and do the draft logits vary by row?
    let tap_sum: u64 = borrowed
        .iter()
        .map(|tap| tap.iter().map(|bits| *bits as u64).sum::<u64>())
        .sum();
    println!("  [d4] tap checksum {tap_sum}");
    let mut cache = DflashKvCache::new(config);
    drafter
        .append_context(&borrowed, total, total, &mut cache)
        .expect("append context");

    // `targets[t]` is the target's distribution for position `total + t`.
    let mut targets = vec![top_k_distribution(&logits, TOP_K)];
    for step in 0..COLLECT - 1 {
        let next = argmax(&logits);
        logits = target
            .forward_buffer(&[next], &[(total + step) as u32])
            .expect("chained forward");
        targets.push(top_k_distribution(&logits, TOP_K));
    }

    let head = target.head();
    let block_size = config.block_size;
    let draft_rows = config.draft_tokens();
    let anchor = target.raw_embedding(&[tokens[total - 1]]);
    let draft = drafter
        .draft_block(&anchor, total, &cache, head)
        .expect("draft block");
    let block_logits = drafter
        .draft_logits(
            &draft.block.hidden,
            head,
            config.final_logit_softcapping,
            block_size,
        )
        .expect("draft logits");
    let vocab = head.out_features;

    let dists: Vec<Vec<(u32, f64)>> = (0..block_size)
        .map(|row| top_k_distribution(&block_logits[row * vocab..(row + 1) * vocab], TOP_K))
        .collect();

    let mut best = (0usize, 0usize, -1.0_f64);
    for row in 0..block_size {
        for (t, target_dist) in targets.iter().enumerate() {
            let a = acceptance(target_dist, &dists[row]);
            if a > best.2 {
                best = (row, t, a);
            }
        }
    }
    // The round's alignment: row r is compared against the distribution for
    // position start + r, i.e. targets[r - 1].
    let diagonal: Vec<f64> = (1..=draft_rows)
        .filter(|row| *row <= targets.len())
        .map(|row| acceptance(&targets[row - 1], &dists[row]))
        .collect();
    let mean_diagonal = if diagonal.is_empty() {
        0.0
    } else {
        diagonal.iter().sum::<f64>() / diagonal.len() as f64
    };
    // DIAGNOSTIC (D4): how many distinct draft tokens, and what do the first few
    // logits of each row look like?
    let mut distinct = draft.tokens.clone();
    distinct.sort_unstable();
    distinct.dedup();
    println!("  [d4] {} distinct draft tokens of {}", distinct.len(), draft.tokens.len());
    for row in [0usize, 1, 2, 3, 15] {
        if row >= dists.len() {
            continue;
        }
        println!("  [d4] row {row} q top3: {}", top3(&dists[row]));
    }
    let ctx_first = cache.all_first_positions();
    println!("  [d4] cache rows per layer {:?}", ctx_first);
    Case {
        mean_diagonal,
        best_alpha: best.2,
        best_row: best.0,
        best_t: best.1,
        drafts: draft.tokens.clone(),
        top_target: top3(&targets[0]),
    }
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; loads the 16 GB target and the drafter"]
fn alpha_at_temperature_1_top_k_64() {
    let dir = drafter_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("drafter checkpoint not present at {dir:?}; skipping");
        return;
    }
    let mut target = TextModel::load_from_dir(Path::new(&model_path())).expect("target model");
    let device = metal::Device::system_default().expect("metal device");
    let weights = DflashWeights::load(&dir, &device).expect("drafter weights");
    let config = DflashConfig::load(&dir).expect("drafter config");
    let mut drafter = DflashModel::new(weights, config.clone()).expect("drafter model");

    let tokenizer = GemmaTokenizer::from_model_dir(Path::new(&model_path())).expect("tokenizer");
    let tokens = tokenizer.encode_prompt(PROMPT).expect("encode prompt");
    let mask = target.raw_embedding(&[config.mask_token_id]);
    drafter.set_mask_embedding(&mask).expect("mask embedding");

    let base_layers: Vec<usize> = config.target_layer_ids.clone();
    let shifted_layers: Vec<usize> = base_layers.iter().map(|layer| layer + 1).collect();

    println!();
    println!(
        "alpha probe: temperature 1.0, top_k {TOP_K}, prompt {} tokens, block_size {}, \
         draft_rows {}",
        tokens.len(),
        config.block_size,
        config.draft_tokens()
    );
    println!(
        "  self control (p0 vs itself) must be 1.0: {:.6}",
        {
            let positions: Vec<u32> = (0..tokens.len() as u32).collect();
            target.reset_gpu_caches();
            let logits = target.forward_buffer(&tokens, &positions).expect("control");
            let p = top_k_distribution(&logits, TOP_K);
            acceptance(&p, &p)
        }
    );

    let mut summary = Vec::new();
    for (label, layers) in [
        ("as-configured", base_layers.clone()),
        ("config+1 (z-lab offset=1)", shifted_layers.clone()),
    ] {
        let case = measure_case(&mut target, &drafter, &tokens, &layers, &config);
        println!();
        println!("=== tap layers {label}: {layers:?} ===");
        println!("  target p0 top3: {}", case.top_target);
        println!("  drafts: {:?}", case.drafts);
        println!(
            "  mean alpha on the round's diagonal: {:.4}; best {:.4} at row {} vs t={}",
            case.mean_diagonal, case.best_alpha, case.best_row, case.best_t
        );
        summary.push((label, layers, case));
    }

    println!();
    println!("=== SUMMARY (mean alpha on the round's diagonal) ===");
    for (label, layers, case) in &summary {
        println!(
            "  {label:<28} {layers:?}  mean {:.4}  best {:.4}",
            case.mean_diagonal, case.best_alpha
        );
    }
    println!();
    println!("DECISION RULE (measured round times, 15.25 tok/s baseline):");
    println!("  alpha < 0.919 -> slower than plain decode, DO NOT SHIP");
    println!("  alpha 0.919-0.977 -> a real speedup, but under 30 t/s at b32");
    println!("  alpha >= 0.977 -> 30 t/s at b32 is reachable");

    // The two cases tap different layers, so the taps differ, so the drafter's
    // context differs, so its drafts must differ. They are byte-identical, and that
    // is the root cause of `alpha == 0.0000` rather than a symptom of it: the
    // drafter's output does not depend on its context at all, so it cannot be
    // correlated with the target it was supposed to predict. `enable_taps` does
    // update the layer list (it sets `TapScratch { layers: layers.to_vec(), .. }`),
    // and T7's own test shows different layers produce different tap *contents*, so
    // the loss is downstream -- in `append_context`'s fusion/projection, the
    // context attention, or the positions the context rows are cached at.
    //
    // This assertion fails today. It is the check that should flip to green when the
    // context path is fixed, which is why it is here rather than in a comment.
    assert_ne!(
        summary[0].2.drafts, summary[1].2.drafts,
        "the drafter returned identical drafts from two different tap layer sets \
         ({:?} vs {:?}), so its context does not affect its predictions. alpha = \
         0.0000 on every row and every target position follows from that. This is a \
         broken context path, not a low-quality drafter, and it makes the 30 t/s \
         question unanswerable until it is fixed. Note that `dflash_golden` only \
         ever exercised `forward_block` with ZERO context rows, so this path has \
         never been validated against the reference.",
        summary[0].1, summary[1].1
    );
}
