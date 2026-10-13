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
use kaplumba_engine::exl3::{apply_hadamard_matrix, decode_tile, BitRate, Codebook, SignVector};
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

fn widen(values: &[u16]) -> Vec<f32> {
    values
        .iter()
        .map(|bits| f16::from_bits(*bits).to_f32())
        .collect()
}

/// D5: `max|v|`, RMS and a non-finite count. Printed, not asserted.
///
/// The whole point of the 30 t/s question turned out to be scale: a block whose
/// residual stream vanishes gives a uniform distribution, one that saturates
/// `final_logit_softcapping` gives a flat one, and both look identical from the
/// outside (two distinct tokens in fifteen rows).
fn stats(label: &str, values: &[f32]) -> f32 {
    let mut max = 0.0_f32;
    let mut sum = 0.0_f64;
    let mut bad = 0usize;
    for value in values {
        if !value.is_finite() {
            bad += 1;
        }
        max = max.max(value.abs());
        sum += (*value as f64) * (*value as f64);
    }
    let rms = if values.is_empty() {
        0.0
    } else {
        (sum / values.len() as f64).sqrt() as f32
    };
    println!(
        "  [d5] {label:<26} len {:>7} max|v| {:>12.5} rms {:>11.6} nonfinite {bad}",
        values.len(),
        max,
        rms
    );
    rms
}

/// The largest element of a batch of f16 logits, widened.
fn widened_max(values: &[u16]) -> f32 {
    values
        .iter()
        .map(|bits| f16::from_bits(*bits).to_f32())
        .fold(f32::MIN, f32::max)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Reconstruct one 128-column LM-head band on the host. Reconstructing the
/// full 5376×262144 fp32 matrix would need 5.6 GB, so this follows the EXL3
/// row-major tile order and its 128×128 Hadamard transforms for one band.
fn host_head_band(
    head: &kaplumba_engine::model::QuantizedLinear,
    input: &[u16],
    column_start: usize,
    softcap: f32,
) -> Vec<f32> {
    const BAND: usize = 128;
    assert_eq!(input.len(), head.in_features);
    assert_eq!(column_start % BAND, 0, "host bands align to Hadamard tiles");
    assert!(column_start + BAND <= head.out_features);
    let rate = match head.rate {
        kaplumba_kernels::exl3::Exl3Rate::Integer(bits) => BitRate::Integer(bits),
        other => panic!("unsupported target head rate {other:?}"),
    };
    assert_eq!(head.codebook, kaplumba_kernels::exl3::Exl3Codebook::Mcg);
    let rows = head.in_features;
    let tiles_n = head.out_features / 16;
    let tile_rows = rows / 16;
    let first_tile_n = column_start / 16;
    let tiles_in_band = BAND / 16;
    let words_per_tile = rate.packed_words();
    let expected_words = tile_rows * tiles_n * words_per_tile;
    assert_eq!(
        head.packed_len.unwrap_or(head.packed.len()),
        expected_words,
        "head packed length must cover the complete EXL3 matrix"
    );
    let packed_buffer = head
        .metal_buffer
        .as_ref()
        .expect("mapped target head buffer");
    let packed_bytes = expected_words * std::mem::size_of::<u16>();
    assert!(
        head.metal_offset + packed_bytes as u64 <= packed_buffer.length(),
        "head trellis region must fit in its mapped shard buffer"
    );
    let packed_base = unsafe {
        (packed_buffer.contents() as *const u8)
            .add(head.metal_offset as usize)
            .cast::<u16>()
    };

    let mut weights = vec![0.0_f32; rows * BAND];
    for tile_row in 0..tile_rows {
        for local_tile_n in 0..tiles_in_band {
            let tile_n = first_tile_n + local_tile_n;
            let packed_offset = (tile_row * tiles_n + tile_n) * words_per_tile;
            let tile = if head.packed.is_empty() {
                let packed_tile = unsafe {
                    std::slice::from_raw_parts(packed_base.add(packed_offset), words_per_tile)
                };
                decode_tile(packed_tile, rate, Codebook::Mcg)
            } else {
                decode_tile(
                    &head.packed[packed_offset..packed_offset + words_per_tile],
                    rate,
                    Codebook::Mcg,
                )
            }
            .expect("decode target head EXL3 tile");
            for row in 0..16 {
                for column in 0..16 {
                    let matrix_row = tile_row * 16 + row;
                    let matrix_column = local_tile_n * 16 + column;
                    weights[matrix_row * BAND + matrix_column] = tile[row * 16 + column];
                }
            }
        }
    }
    apply_hadamard_matrix(&mut weights, rows, BAND).expect("host head Hadamard");
    let input_signs = SignVector::from_half_bits(head.input_signs.clone())
        .decode(rows)
        .expect("decode target head input scales");
    let all_output_signs = SignVector::from_half_bits(head.output_signs.clone())
        .decode(head.out_features)
        .expect("decode target head output scales");
    let start = column_start;
    for row in 0..rows {
        for column in 0..BAND {
            weights[row * BAND + column] *= input_signs[row] * all_output_signs[start + column];
        }
    }

    let mut logits = vec![0.0_f32; BAND];
    for column in 0..BAND {
        let mut sum = 0.0_f32;
        for row in 0..rows {
            sum += f16::from_bits(input[row]).to_f32() * weights[row * BAND + column];
        }
        logits[column] = softcap * (sum / softcap).tanh();
    }
    logits
}

struct Case {
    mean_diagonal: f64,
    best_alpha: f64,
    best_row: usize,
    best_t: usize,
    drafts: Vec<u32>,
    top_target: String,
    /// The drafter's *capped* block logits, every row, kept so two cases can be
    /// compared at the logits level and not only at their argmaxes (T12).
    block_logits: Vec<u16>,
}

/// Prefill, build the drafter's context from `layers`, chain a few target steps, draft
/// one block, and return the acceptance numbers for the round's diagonal.
fn measure_case(
    target: &mut TextModel,
    drafter: &mut DflashModel,
    tokens: &[u32],
    layers: &[usize],
    config: &DflashConfig,
) -> Case {
    let total = tokens.len();

    // DIAGNOSTIC (D10), part 1. The oracle for the *head* path.
    //
    // `draft_logits` applies the target's head through the drafter's own GEMM
    // entry point (`gemm_chunk_buffer_sliced`: the packed-buffer path with the
    // head's 6-bit rate, its metal offset and its scale vectors). The target's
    // own logits go through `encode_lm_head`. If those two disagree on the same
    // input, the drafter's head is applied wrongly and nothing about the
    // drafter's hidden state can matter. This captures the target's final normed
    // row and the logits its own head made from it.
    //
    // `forward_with_hidden` is a one-token decode API. It writes one final
    // hidden row; passing a prompt-length batch only writes the first row, so
    // reading the final row returned zeros. Prefill everything before the last
    // three prompt tokens, then capture three real decode rows and target
    // logits. Wipe the cache before the tap prefill below.
    target.reset_gpu_caches();
    let oracle_start = total.saturating_sub(3);
    if oracle_start > 0 {
        let prefill_positions: Vec<u32> = (0..oracle_start as u32).collect();
        target
            .forward_buffer(&tokens[..oracle_start], &prefill_positions)
            .expect("head-oracle prefix prefill");
    }
    let mut head_oracle = Vec::with_capacity(total - oracle_start);
    for position in oracle_start..total {
        let (hidden, logits) = target
            .forward_with_hidden(&tokens[position..position + 1], &[position as u32])
            .unwrap_or_else(|error| panic!("[d10] oracle failed at position {position}: {error}"));
        assert_eq!(
            hidden.len(),
            config.hidden_size,
            "D10 expects one hidden row"
        );
        head_oracle.push((hidden, logits));
    }
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

    // D6: the target's own pre-softcap logits, the yardstick the drafter's have
    // to be measured against. Both models feed the *same* head, so a drafter
    // whose logits are 10x the target's saturates `final_logit_softcapping` and
    // its distribution flattens to a coin flip.
    stats("target p0 logits", &widen(&logits));

    // The anchor is the token at position `total` -- the one the target has just
    // predicted there -- NOT the prompt's last token at `total - 1`.
    //
    // The block places row 0 at position `total` (`draft_block(anchor, total, ..)`),
    // and z-lab's round is identical: `block_output_ids[0]` is
    // `output_ids[start]`, which was filled by sampling the prefill's logits
    // (`dflash/model.py:249`, `:264`). Feeding `tokens[total - 1]` instead tells
    // the drafter the sequence has the prompt's last token *twice* -- the most
    // recent token is the one a next-token model leans on hardest, so this is not
    // a rounding detail: it puts a wrong token in the position the drafter has to
    // extend, and the prediction that follows is not the continuation of the
    // prompt at all.
    let anchor_token = argmax(&logits);

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
    let anchor = target.raw_embedding(&[anchor_token]);
    let mask = target.raw_embedding(&[config.mask_token_id]);
    println!(
        "  [d9] anchor token {anchor_token} (the target's own argmax at position {total}); \
         prompt's last token was {}",
        tokens[total - 1]
    );
    let draft = drafter
        .draft_block(&anchor, total, &cache, head)
        .expect("draft block");

    // DIAGNOSTIC (D5): the real magnitudes, on the real checkpoints.
    //
    // The block collapses to two distinct tokens, which is either a flat
    // distribution (saturated softcap) or a uniform one (vanishing residual).
    // Both are visible here as numbers, and neither is visible from the drafts.
    // DIAGNOSTIC (D8): the block rows must differ.
    //
    // Rows 1..15 are the *same* input (the mask embedding) at 15 different
    // positions, so every difference between them is RoPE and the attention it
    // induces. A drafter that gives all fifteen the same argmax is not
    // predicting fifteen different tokens; it is predicting one and repeating
    // it. The cosine against row 0 says whether the rows are distinguishable at
    // all, which separates "the block lost its positions" from "the head
    // flattens a correct block".
    {
        let hidden = draft.block.hidden.as_slice();
        let width = config.hidden_size;
        let rows_here = hidden.len() / width;
        let row_rms = |row: usize| -> f32 {
            let slice = &hidden[row * width..(row + 1) * width];
            (slice
                .iter()
                .map(|bits| {
                    let value = f16::from_bits(*bits).to_f32();
                    (value as f64) * (value as f64)
                })
                .sum::<f64>()
                / width as f64)
                .sqrt() as f32
        };
        let base = &hidden[0..width];
        let base_norm: f64 = base
            .iter()
            .map(|bits| {
                let value = f16::from_bits(*bits).to_f32() as f64;
                value * value
            })
            .sum::<f64>()
            .sqrt();
        let mut cosine = Vec::with_capacity(rows_here);
        for row in 0..rows_here {
            let slice = &hidden[row * width..(row + 1) * width];
            let dot: f64 = slice
                .iter()
                .zip(base.iter())
                .map(|(a, b)| {
                    f16::from_bits(*a).to_f32() as f64 * f16::from_bits(*b).to_f32() as f64
                })
                .sum();
            let norm: f64 = slice
                .iter()
                .map(|bits| {
                    let value = f16::from_bits(*bits).to_f32() as f64;
                    value * value
                })
                .sum::<f64>()
                .sqrt();
            cosine.push(dot / (norm * base_norm).max(1e-12));
        }
        println!(
            "  [d8] row RMS {:?}",
            (0..rows_here)
                .map(|row| format!("{:.2}", row_rms(row)))
                .collect::<Vec<_>>()
        );
        println!(
            "  [d8] cosine to row 0 {:?}",
            cosine
                .iter()
                .take(6)
                .map(|value| format!("{value:.4}"))
                .collect::<Vec<_>>()
        );
    }

    // DIAGNOSTIC (D10): the head path, against an oracle.
    //
    // `draft_logits` applies the *target's* head through the drafter's own GEMM
    // entry point (`gemm_chunk_buffer_sliced`, the packed-buffer path with the
    // head's 6-bit rate, metal offset and scale vectors). The target's own
    // logits go through `encode_lm_head` instead. If the two disagree on the
    // *same* input, then the drafter's head is being applied wrongly and no
    // Compare every vocabulary entry, not selected columns or a head-only
    // self-control. Both routes use the same head and softcap, leaving target
    // GEMV versus DFlash sliced-GEMM arithmetic as the difference. A 1.0
    // max-logit bound is 1/30 of the configured softcap; greedy output must
    // match exactly.
    for (index, (row, target_logits)) in head_oracle.iter().enumerate() {
        let rms = stats(&format!("target final-norm row {index}"), &widen(row));
        assert!(rms > 0.01, "D10 oracle row {index} is unexpectedly zero");
        let target_f32 = widen(target_logits);
        stats(&format!("target logits oracle {index}"), &target_f32);
        let replayed = drafter
            .draft_logits(row, head, config.final_logit_softcapping, 1)
            .expect("D10 replay target head through DFlash GEMM");
        let replayed_f32 = widen(&replayed);
        let diff = max_abs_diff(&replayed_f32, &target_f32);
        assert!(
            diff <= 1.0,
            "D10 row {index}: target-head replay max logit error {diff:.4} exceeds 1.0"
        );
        assert_eq!(
            argmax(&replayed),
            argmax(target_logits),
            "D10 row {index}: target-head replay changed greedy token"
        );

        let head = target.head();
        for column_start in [0, 128, 65_536, 131_072, head.out_features - 128] {
            let host = host_head_band(head, row, column_start, config.final_logit_softcapping);
            let end = column_start + host.len();
            let target_error = max_abs_diff(&host, &target_f32[column_start..end]);
            let dflash_error = max_abs_diff(&host, &replayed_f32[column_start..end]);
            assert!(
                target_error <= 0.05,
                "T7 row {index}, columns {column_start}..{end}: host vs target GEMV error {target_error:.4} exceeds 0.05"
            );
            assert!(
                dflash_error <= 0.05,
                "T7 row {index}, columns {column_start}..{end}: host vs DFlash GEMM error {dflash_error:.4} exceeds 0.05"
            );
            println!(
                "  [t7] row {index}, columns {column_start}..{end}: host-target {target_error:.4}, host-DFlash {dflash_error:.4}"
            );
        }
        println!(
            "  [d10] row {index}: full-vocabulary head replay max diff {diff:.4}; greedy token {} matches",
            argmax(target_logits)
        );
    }

    println!("  [d5] --- real magnitudes ---");
    let anchor_rms = stats("anchor (raw embed row)", &widen(&anchor));
    let mask_rms = stats("mask (raw embed row)", &widen(&mask));
    for (slot, tap) in borrowed.iter().enumerate() {
        stats(
            &format!("tap {slot} (layer {})", config.target_layer_ids[slot]),
            &widen(tap),
        );
    }
    if let Ok(fused) = drafter.fuse_context(&borrowed, total) {
        let fused_rms = stats("fused ctx (fc+hidden_norm)", &widen(&fused));
        println!(
            "  [d5] fused/anchor RMS ratio {:.4}, fused/mask RMS ratio {:.4}",
            fused_rms / anchor_rms.max(1e-9),
            fused_rms / mask_rms.max(1e-9)
        );
        let positions: Vec<u32> = (0..total as u32).collect();
        if let Ok((k, v)) = drafter.project_context_kv(0, &fused, total, &positions, total) {
            stats("layer0 ctx K", &widen(&k));
            stats("layer0 ctx V", &widen(&v));
        }
    }
    let block_hidden_rms = stats("block hidden (norm out)", &widen(&draft.block.hidden));
    stats(
        "layer0 block K (gemm out)",
        &widen(&draft.block.block_keys[0]),
    );
    stats(
        "layer0 block V (gemm out)",
        &widen(&draft.block.block_values[0]),
    );

    // The real-path version of `dflash_context_rows_are_visible_in_the_block_output`:
    // same anchor, same positions, a context built from the same taps in a
    // different order. If the block output does not move, the context is not
    // reaching the predictions, and no kernel change can fix that.
    {
        let mut reversed: Vec<Vec<u16>> = taps.clone();
        reversed.reverse();
        let reversed_refs: Vec<&[u16]> = reversed.iter().map(|tap| tap.as_slice()).collect();
        let mut other = DflashKvCache::new(config);
        if drafter
            .append_context(&reversed_refs, total, total, &mut other)
            .is_ok()
        {
            let first = other.all_first_positions();
            if let Ok(block) = drafter.forward_block(&kaplumba_engine::dflash::BlockContext {
                anchor_embedding: &anchor,
                ctx_keys: other.all_keys(),
                ctx_values: other.all_values(),
                ctx_first: &first,
                position: total,
            }) {
                let moved = max_abs_diff(&widen(&draft.block.hidden), &widen(&block.hidden));
                println!(
                    "  [d5] context reordered: block hidden moves {moved:.5} ({:.4} of its own RMS \
                     {block_hidden_rms:.5})",
                    moved / block_hidden_rms.max(1e-9)
                );
            }
        }
    }

    let block_logits = drafter
        .draft_logits(
            &draft.block.hidden,
            head,
            config.final_logit_softcapping,
            block_size,
        )
        .expect("draft logits");
    let vocab = head.out_features;
    let logits_f32 = widen(&block_logits);
    let logit_rms = stats("draft logits (softcapped)", &logits_f32);
    // The softcap is monotone, so the pre-cap logit is recoverable:
    // `x = 30 * atanh(capped / 30)`. This is the number that matters --
    // `tanh(x/30)*30` is ~linear below |x| = 10 and flat above |x| = 60, so two
    // raw logits of 114 and 200 both arrive at 29.97 and the softmax cannot tell
    // them apart.
    {
        let mut order: Vec<&f32> = logits_f32.iter().collect();
        order.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let cap = config.final_logit_softcapping;
        let raw = |capped: f32| cap * (capped / cap).clamp(-0.999_999, 0.999_999).atanh();
        let top: Vec<String> = order
            .iter()
            .take(4)
            .map(|value| format!("{value:.4}->{:.2}", raw(**value)))
            .collect();
        let flattened = logits_f32
            .iter()
            .filter(|value| raw(**value) > 60.0)
            .count();
        println!(
            "  [d6] top4 capped->raw [{}], {flattened} of {} raw logits exceed 60 (the softcap's \
             flat region)",
            top.join(", "),
            logits_f32.len()
        );
    }
    let logit_max = logits_f32.iter().cloned().fold(f32::MIN, f32::max);
    let saturated = logits_f32.iter().filter(|value| value.abs() > 29.9).count();
    let near_flat = logits_f32
        .iter()
        .filter(|value| logit_max - **value < 0.05)
        .count();
    println!(
        "  [d5] logits max {logit_max:.5}, rms {logit_rms:.5}, {saturated} of {} within 0.1 of the \
         softcap 30, {near_flat} within 0.05 of the row max",
        logits_f32.len()
    );
    println!(
        "  [d5] block hidden vs real anchor/mask: anchor RMS {anchor_rms:.5}, mask RMS {mask_rms:.5}"
    );

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

    // DIAGNOSTIC (D13): where the drafter *ranks* the token the target wants.
    //
    // alpha = sum min(p, q) is zero for two very different illnesses. Either the
    // drafter gives the target's token a sane probability in a flattened
    // distribution (fixable: the ranking is right and only the sharpness is
    // lost), or it ranks that token so far down that it is not in the top 64 at
    // all (the ranking itself is wrong and no temperature fixes it). The rank
    // below is the difference, and it is the one number that says which of the
    // two this is.
    println!("  [d13] --- rank of the target's top token in the drafter's row ---");
    for row in 1..draft_rows.min(5) {
        let Some((token, _)) = targets.get(row).and_then(|dist| dist.first()) else {
            continue;
        };
        let logits_row = &block_logits[row * vocab..(row + 1) * vocab];
        let widened_row = widen(logits_row);
        let mine = widened_row[*token as usize];
        let rank = widened_row.iter().filter(|value| **value > mine).count();
        let max = widened_row.iter().cloned().fold(f32::MIN, f32::max);
        let sum: f64 = widened_row
            .iter()
            .map(|value| ((*value - max) as f64).exp())
            .sum();
        let p = ((mine - max) as f64).exp() / sum;
        let own = draft.tokens.get(row).copied().unwrap_or(u32::MAX);
        println!(
            "  [d13] row {row}: target wants {token} (p_target {:.4}); drafter ranks it {rank} of \
             {vocab} with p {p:.3e}; logit {mine:.4} against the row's max {max:.4} (softcapped). \
             The drafter's own pick is {own}",
            targets[row].first().map_or(0.0, |(_, prob)| *prob)
        );
    }

    // DIAGNOSTIC (D11): the head at M=16 vs M=1.
    //
    // `Model::encode_lm_head_rows` refuses to run this 6-bpw head through
    // `encode_gemm`'s M>1 MMA tile, and says why: measured on this same
    // checkpoint, M=1 matched `encode_gemv` to half an fp16 quantum while M=2
    // "diverged by 19.2 on softcapped logits that only span -30..30".
    // `DflashModel::draft_logits` goes through `gemm_chunk_buffer_sliced` with
    // all 16 rows at once. If that is the broken path, the drafter's logits are
    // not a wrong model -- they are a wrong kernel, and every row of them is
    // noise.
    {
        let width = config.hidden_size;
        let mut per_row = Vec::with_capacity(block_size * vocab);
        let mut ok = true;
        for row in 0..block_size {
            let slice = &draft.block.hidden[row * width..(row + 1) * width];
            match drafter.draft_logits(slice, head, config.final_logit_softcapping, 1) {
                Ok(logits) => per_row.extend(logits),
                Err(error) => {
                    println!("  [d11] per-row head failed at row {row}: {error}");
                    ok = false;
                    break;
                }
            }
        }
        if ok && per_row.len() == block_size * vocab {
            let batched = widen(&block_logits);
            let per_row_f32 = widen(&per_row);
            let mut row_diffs = Vec::new();
            for row in 0..block_size {
                row_diffs.push(max_abs_diff(
                    &per_row_f32[row * vocab..(row + 1) * vocab],
                    &batched[row * vocab..(row + 1) * vocab],
                ));
            }
            let per_row_dists: Vec<Vec<(u32, f64)>> = (0..block_size)
                .map(|row| top_k_distribution(&per_row[row * vocab..(row + 1) * vocab], TOP_K))
                .collect();
            let per_row_diagonal: Vec<f64> = (1..=draft_rows)
                .filter(|row| *row < targets.len())
                .map(|row| acceptance(&targets[row], &per_row_dists[row]))
                .collect();
            let per_row_mean = if per_row_diagonal.is_empty() {
                0.0
            } else {
                per_row_diagonal.iter().sum::<f64>() / per_row_diagonal.len() as f64
            };
            let per_row_best = per_row_dists
                .iter()
                .enumerate()
                .flat_map(|(row, dist)| {
                    targets
                        .iter()
                        .enumerate()
                        .map(move |(t, target_dist)| (row, t, acceptance(target_dist, dist)))
                })
                .fold((0usize, 0usize, 0.0_f64), |best, item| {
                    if item.2 > best.2 {
                        item
                    } else {
                        best
                    }
                });
            let mut distinct = per_row[..]
                .chunks_exact(vocab)
                .map(argmax)
                .collect::<Vec<u32>>();
            distinct.sort_unstable();
            distinct.dedup();
            println!(
                "  [d11] M=1 vs M=16 head: per-row worst diff {:?}",
                row_diffs
                    .iter()
                    .map(|value| format!("{value:.4}"))
                    .collect::<Vec<_>>()
            );
            println!(
                "  [d11] M=1 head: mean alpha {per_row_mean:.4}, best {:.4} (row {} t {}), \
                 {} distinct tokens of {}",
                per_row_best.2,
                per_row_best.0,
                per_row_best.1,
                distinct.len(),
                block_size
            );
        }
    }

    // DIAGNOSTIC (D12): the noise embedding's scale.
    //
    // The anchor and mask rows are the target's **raw** table rows, RMS 0.015,
    // while the fused context is RMS 1.006 -- a ratio of 65x. Under that ratio
    // the block's sixteen rows all collapse onto the context: their own identity
    // (a real token versus `<mask>`) is 1.5% of the input to layer 0, so the only
    // thing left distinguishing one row from the next is the RoPE phase of its
    // attention. Measured, the rows do differ (cosine 0.7) but every one of them
    // returns the same argmax.
    //
    // The target is Gemma, and Gemma does not use its embedding table raw: its
    // own forward multiplies every row by `sqrt(hidden_size)` = 73.3. A drafter
    // trained to continue the target's stream has to be fed the rows the way the
    // target sees them -- and z-lab plumbs exactly that scale as a config knob
    // (`input_embedding_scale`, `arch_specific/dflash.py:126`), defaulting to 1.0
    // only because the DFlash2 export leaves it unset. If that is the missing
    // factor, the block's rows get their identity back and the draft stops being
    // one token repeated sixteen times.
    {
        let scale = (config.hidden_size as f32).sqrt();
        let rescale = |row: &[u16]| -> Vec<u16> {
            row.iter()
                .map(|bits| f16::from_f32(f16::from_bits(*bits).to_f32() * scale).to_bits())
                .collect()
        };
        let scaled_anchor = rescale(&anchor);
        let scaled_mask = rescale(&mask);
        if drafter.set_mask_embedding(&scaled_mask).is_ok() {
            if let Ok(scaled) = drafter.draft_block(&scaled_anchor, total, &cache, head) {
                stats("block hidden (x sqrt(h))", &widen(&scaled.block.hidden));
                if let Ok(scaled_logits) = drafter.draft_logits(
                    &scaled.block.hidden,
                    head,
                    config.final_logit_softcapping,
                    block_size,
                ) {
                    let scaled_dists: Vec<Vec<(u32, f64)>> = (0..block_size)
                        .map(|row| {
                            top_k_distribution(
                                &scaled_logits[row * vocab..(row + 1) * vocab],
                                TOP_K,
                            )
                        })
                        .collect();
                    let scaled_diagonal: Vec<f64> = (1..=draft_rows)
                        .filter(|row| *row < targets.len())
                        .map(|row| acceptance(&targets[row], &scaled_dists[row]))
                        .collect();
                    let scaled_mean = if scaled_diagonal.is_empty() {
                        0.0
                    } else {
                        scaled_diagonal.iter().sum::<f64>() / scaled_diagonal.len() as f64
                    };
                    let scaled_best = scaled_dists
                        .iter()
                        .enumerate()
                        .flat_map(|(row, dist)| {
                            targets.iter().enumerate().map(move |(t, target_dist)| {
                                (row, t, acceptance(target_dist, dist))
                            })
                        })
                        .fold((0usize, 0usize, 0.0_f64), |best, item| {
                            if item.2 > best.2 {
                                item
                            } else {
                                best
                            }
                        });
                    let mut distinct = scaled.tokens.clone();
                    distinct.sort_unstable();
                    distinct.dedup();
                    println!(
                        "  [d12] noise x sqrt(hidden) = {scale:.2}: mean alpha {scaled_mean:.4}, best \
                         {:.4} (row {} t {}), {} distinct tokens of {}",
                        scaled_best.2,
                        scaled_best.0,
                        scaled_best.1,
                        distinct.len(),
                        scaled.tokens.len()
                    );
                    println!("  [d12] drafts: {:?}", scaled.tokens);
                }
            }
        }
        // Restore, so nothing downstream sees the scaled mask.
        let _ = drafter.set_mask_embedding(&mask);
    }

    // DIAGNOSTIC (D7): is the failure *only* a scale?
    //
    // The drafter's raw logits reach 113 while the target's peak at 27, so
    // `tanh(x/30)*30` flattens the drafter's whole top band to 29.97-30.00 and
    // every row's argmax becomes a coin toss among near-ties. Sweeping a scale
    // on the block's hidden state separates that from a deeper error: if some
    // factor restores alpha, the drafter computes the right thing at the wrong
    // magnitude (and the factor names the bug); if none does, the magnitude is
    // a symptom and the search must continue elsewhere.
    println!("  [d7] --- scale sweep on the drafter's block output ---");
    for factor in [1.0_f32, 0.75, 0.5, 0.35, 0.25, 0.125] {
        let scaled: Vec<u16> = draft
            .block
            .hidden
            .iter()
            .map(|bits| {
                half::f16::from_f32(half::f16::from_bits(*bits).to_f32() * factor).to_bits()
            })
            .collect();
        let Ok(scaled_logits) =
            drafter.draft_logits(&scaled, head, config.final_logit_softcapping, block_size)
        else {
            continue;
        };
        let scaled_dists: Vec<Vec<(u32, f64)>> = (0..block_size)
            .map(|row| top_k_distribution(&scaled_logits[row * vocab..(row + 1) * vocab], TOP_K))
            .collect();
        let scaled_diagonal: Vec<f64> = (1..=draft_rows)
            .filter(|row| *row <= targets.len())
            .map(|row| acceptance(&targets[row - 1], &scaled_dists[row]))
            .collect();
        let scaled_mean = if scaled_diagonal.is_empty() {
            0.0
        } else {
            scaled_diagonal.iter().sum::<f64>() / scaled_diagonal.len() as f64
        };
        let scaled_best = scaled_dists
            .iter()
            .enumerate()
            .flat_map(|(row, dist)| {
                targets
                    .iter()
                    .enumerate()
                    .map(move |(t, target_dist)| (row, t, acceptance(target_dist, dist)))
            })
            .fold((0usize, 0usize, 0.0_f64), |best, item| {
                if item.2 > best.2 {
                    item
                } else {
                    best
                }
            });
        let raw_max = widened_max(&scaled_logits);
        println!(
            "  [d7] scale {factor:<6} mean alpha {scaled_mean:.4}  best {:.4} (row {} t {}), \
             softcapped logit max {raw_max:.4}",
            scaled_best.2, scaled_best.0, scaled_best.1
        );
    }
    // The round's alignment: block row `r` sits at position `total + r` and
    // predicts the token at `total + r`, which is exactly what `targets[r]`
    // holds. (Row `r - 1` would be right if row 0 were the *previous* position;
    // it is not -- row 0 is the anchor at `total` itself.) The best-over-all-pairs
    // search above is alignment-free, so this is a display convention only.
    let diagonal: Vec<f64> = (1..=draft_rows)
        .filter(|row| *row < targets.len())
        .map(|row| acceptance(&targets[row], &dists[row]))
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
    println!(
        "  [d4] {} distinct draft tokens of {}",
        distinct.len(),
        draft.tokens.len()
    );
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
        block_logits,
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
    println!("  self control (p0 vs itself) must be 1.0: {:.6}", {
        let positions: Vec<u32> = (0..tokens.len() as u32).collect();
        target.reset_gpu_caches();
        let logits = target.forward_buffer(&tokens, &positions).expect("control");
        let p = top_k_distribution(&logits, TOP_K);
        acceptance(&p, &p)
    });

    let mut summary = Vec::new();
    // The third case is the drafter's own version of
    // `real_model_permutation_sweep`: its checkpoint has no `layout` key, so the
    // loader leaves every linear at the row-major default and nothing has ever
    // scored that choice. A row-major label (or a mode-0 read transposed for
    // non-square tiles) decodes to plausible-magnitude, wrong-direction weights,
    // which every engine-vs-host test is blind to because both sides share the
    // reader. Acceptance is the score: the right permutation is the one that
    // makes the drafter predict the target.
    for (label, layers, permutation) in [
        ("as-configured", base_layers.clone(), 0u32),
        ("config+1 (z-lab offset=1)", shifted_layers.clone(), 0u32),
        ("as-configured, colmajor", base_layers.clone(), 1u32),
    ] {
        drafter.set_permutation_mode(permutation);
        let case = measure_case(&mut target, &mut drafter, &tokens, &layers, &config);
        println!();
        println!("=== tap layers {label}: {layers:?}, permutation_mode {permutation} ===");
        println!("  target p0 top3: {}", case.top_target);
        println!("  drafts: {:?}", case.drafts);
        // T12: the softcap's flat region, per row. `draft_logits` caps at
        // `final_logit_softcapping` = 30, and `argmax_row` breaks ties to the
        // lowest index, which is context-independent. If whole rows sit inside
        // the cap, every row's argmax is decided by tie order and no amount of
        // context can move it -- which looks exactly like a dead context path.
        let pinned = case
            .block_logits
            .iter()
            .filter(|bits| half::f16::from_bits(**bits).to_f32().abs() > 29.9)
            .count();
        // Row width from the buffer itself: the vocab is 262144 here, but a
        // literal would silently mis-slice every row if the checkpoint changed.
        let vocab = case.block_logits.len() / config.block_size;
        let rows_pinned = (0..config.block_size)
            .filter(|row| {
                case.block_logits[row * vocab..(row + 1) * vocab]
                    .iter()
                    .any(|bits| half::f16::from_bits(*bits).to_f32() > 29.9)
            })
            .count();
        println!(
            "  [t12] {pinned} of {} capped logits sit at the softcap (> 29.9), in {rows_pinned} of \
             {} rows",
            case.block_logits.len(),
            config.block_size
        );
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

    // T12's measurement, and the one that separates the two illnesses that both
    // print `alpha = 0.0000`: compare the drafter's block *logits* across the two
    // contexts, not just their argmaxes.
    //
    //   * logits differ (and differ by a lot) -> the context DOES reach the block
    //     and the argmaxes coincide because the rows are saturated or degenerate
    //     at the cap. The bug is downstream of the context: the logit scale or
    //     the head, not the taps.
    //   * logits byte-identical -> the block's output does not depend on its
    //     context at all, so the loss is in `append_context`'s fusion/projection,
    //     the cached rows, or the attention.
    {
        let a = &summary[0].2.block_logits;
        let b = &summary[1].2.block_logits;
        if a.len() == b.len() {
            let differing = a.iter().zip(b).filter(|(x, y)| x != y).count();
            let worst = a
                .iter()
                .zip(b)
                .map(|(x, y)| {
                    (half::f16::from_bits(*x).to_f32() - half::f16::from_bits(*y).to_f32()).abs()
                })
                .fold(0.0_f32, f32::max);
            let left = a.iter().map(|bits| *bits as u64).sum::<u64>();
            let right = b.iter().map(|bits| *bits as u64).sum::<u64>();
            println!();
            println!("=== [t12] do two different contexts change the block LOGITS? ===");
            println!(
                "  bit-pattern sums: {} vs {} (equal: {})",
                left,
                right,
                left == right
            );
            println!(
                "  {differing} of {} logits differ; worst |delta| {worst:.5}",
                a.len()
            );
            println!(
                "  drafts identical: {}",
                summary[0].2.drafts == summary[1].2.drafts
            );
        }
    }
    println!();
    println!("DECISION RULE (measured round times, 15.25 tok/s baseline):");
    println!("  alpha < 0.919 -> slower than plain decode, DO NOT SHIP");
    println!("  alpha 0.919-0.977 -> a real speedup, but under 30 t/s at b32");
    println!("  alpha >= 0.977 -> 30 t/s at b32 is reachable");

    // The two cases tap different layers, so the taps differ, so the drafter's
    // context differs. **The block logits must therefore differ**: the drafter is
    // supposed to condition on the target's hidden states, and this is the one
    // property that separates "the context is not reaching the block" from "the
    // context reaches the block but the argmaxes coincide" -- two failures that
    // both print `alpha = 0.0000` and need opposite fixes.
    //
    // Asserted on the logits, not on the drafts, because the drafts are argmaxes:
    // `draft_logits` caps at the softcap 30 (`final_logit_softcapping` = 30) and
    // `argmax_row` breaks ties to the lowest index, so a context change that
    // moves every logit by 10 can still leave the argmaxes identical. An
    // assertion on the drafts alone would accuse the wrong component.
    assert_ne!(
        summary[0].2.block_logits, summary[1].2.block_logits,
        "the drafter's block logits are byte-identical from two different tap layer \
         sets ({:?} vs {:?}), so its context does not reach its output at all. \
         alpha = 0.0000 on every row follows from that, and no amount of context \
         quality can fix it: the loss is in `append_context`'s fusion/projection, the \
         cached rows, or the attention. `enable_taps` does update the layer list and \
         T7's test shows different layers produce different tap contents, so the loss \
         is downstream of the taps.",
        summary[0].1, summary[1].1
    );

    // And the drafts are recorded separately, because "logits differ but drafts do \
    // not" is the saturation signature and is the next thing to chase.
    if summary[0].2.drafts == summary[1].2.drafts {
        println!();
        println!(
            "NOTE: the logits differ but the drafts do not. That is the softcap \
             tie-break, not a dead context: the argmax is decided by index order among \
             logits pinned at the cap. Chase the logit scale, not the taps."
        );
    }
}
