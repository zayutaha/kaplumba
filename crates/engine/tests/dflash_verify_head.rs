//! spec/dflash-correctness T9: the batched-verify head must be equivalent to
//! `n` sequential single-row forwards.
//!
//! plan.md risk #7, and it is not a formality. `forward_verify` runs
//! `encode_rms_norm(rows = n)` + `encode_gemm(M = n)` where the decode path
//! runs `encode_rms_norm(rows = 1)` + `encode_gemv(M = 1)`. Those are different
//! kernels with different accumulation orders, so a divergence would corrupt
//! every token the round "verifies" while still reporting a plausible acceptance
//! rate. This is the test that makes T9 admissible.
//!
//! Ignored by default: it loads the 16 GB target. `KAPLUMBA_TEST_MODEL` selects
//! which checkpoint.

use std::path::Path;

use kaplumba_engine::model::TextModel;
use kaplumba_engine::tokenizer::GemmaTokenizer;

fn model_path() -> String {
    std::env::var("KAPLUMBA_TEST_MODEL").expect("KAPLUMBA_TEST_MODEL is required")
}

fn argmax_row(matrix: &[u16], row: usize, vocab: usize) -> u32 {
    let base = row * vocab;
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for column in 0..vocab {
        let value = half::f16::from_bits(matrix[base + column]).to_f32();
        if value > best_value {
            best_value = value;
            best = column;
        }
    }
    best as u32
}

fn top_two_gap(row: &[u16]) -> f32 {
    let mut best = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for bits in row {
        let value = half::f16::from_bits(*bits).to_f32();
        if value > best {
            second = best;
            best = value;
        } else if value > second {
            second = value;
        }
    }
    best - second
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; loads the target and runs real prompt tokens"]
fn batched_m3_matches_sequential_at_dflash_prompt_token_77() {
    let path = model_path();
    let dir = Path::new(&path);
    let tokenizer = GemmaTokenizer::from_model_dir(dir).expect("tokenizer");
    let mut model = TextModel::load_from_dir(dir).expect("target model");
    let vocab = model.vocab_size();
    let prompt = "Explain the theory of relativity in simple terms. Special relativity says the speed of light is the same for every observer.";
    let prompt_tokens = tokenizer.encode_prompt(prompt).expect("encode prompt");
    let prompt_positions: Vec<u32> = (0..prompt_tokens.len() as u32).collect();

    // Save an exact sequential prefix and the target row after every generated
    // token. Output token 77 is the known divergence from the forced-first-
    // proposal experiment (99382 batched, 84750 sequential).
    model.reset_gpu_caches();
    let mut logits = model
        .forward_buffer(&prompt_tokens, &prompt_positions)
        .expect("prompt prefill");
    let mut generated = Vec::with_capacity(80);
    let mut after_token_logits = Vec::with_capacity(80);
    for index in 0..80usize {
        let token = argmax_row(&logits, 0, vocab);
        generated.push(token);
        logits = model
            .forward_buffer(&[token], &[(prompt_tokens.len() + index) as u32])
            .expect("sequential target decode");
        after_token_logits.push(logits.clone());
    }
    assert_eq!(generated[77], 84750, "natural-prefix fixture changed");

    // B=4 DFlash verifies three proposal rows. At this prefix the preceding
    // target rows are sequentially resident, so this isolates M=3 arithmetic.
    let start = 74usize;
    model.reset_gpu_caches();
    model
        .forward_buffer(&prompt_tokens, &prompt_positions)
        .expect("re-prefill prompt");
    for (index, token) in generated.iter().take(start).enumerate() {
        model
            .forward_buffer(&[*token], &[(prompt_tokens.len() + index) as u32])
            .expect("replay sequential prefix");
    }
    let tokens = &generated[start..start + 3];
    let positions: Vec<u32> = (0..3usize)
        .map(|index| (prompt_tokens.len() + start + index) as u32)
        .collect();
    let batched = model
        .forward_verify(tokens, &positions)
        .expect("M=3 target verify");
    for row in 0..3usize {
        let expected = &after_token_logits[start + row];
        let actual = &batched[row * vocab..(row + 1) * vocab];
        let mut max_delta = 0.0f32;
        for (actual, expected) in actual.iter().zip(expected) {
            max_delta = max_delta.max(
                (half::f16::from_bits(*actual).to_f32() - half::f16::from_bits(*expected).to_f32())
                    .abs(),
            );
        }
        let batched_token = argmax_row(&batched, row, vocab);
        let sequential_token = argmax_row(expected, 0, vocab);
        eprintln!(
            "M=3 row={row} output_index={} batched={batched_token} sequential={sequential_token} sequential_gap={:.6} max_logit_delta={max_delta:.6}",
            start + row + 1,
            top_two_gap(expected)
        );
        assert_eq!(
            batched_token,
            sequential_token,
            "M=3 greedy mismatch at output token {} (sequential gap {:.6}, max logit delta {max_delta:.6})",
            start + row + 1,
            top_two_gap(expected)
        );
    }
    // The DFlash round retains only the first accepted proposal here and
    // retracts the two speculative tail rows. The next target correction is
    // token `start+1`; feeding it must still match sequential decode.
    model.retract_gpu_caches(2);
    let correction = generated[start + 1];
    let continued = model
        .forward_buffer(&[correction], &[(prompt_tokens.len() + start + 1) as u32])
        .expect("continue from retracted M=3 cache");
    let expected = &after_token_logits[start + 1];
    let actual_token = argmax_row(&continued, 0, vocab);
    let expected_token = argmax_row(expected, 0, vocab);
    eprintln!(
        "M=3 retracted continuation output_index={} actual={actual_token} sequential={expected_token} sequential_gap={:.6}",
        start + 2,
        top_two_gap(expected)
    );
    assert_eq!(
        actual_token,
        expected_token,
        "M=3 cached continuation mismatch at output token {} (sequential gap {:.6})",
        start + 2,
        top_two_gap(expected)
    );

    // Reproduce a long series of real M=3 rounds with one accepted proposal
    // each. The accepted proposal is the ground-truth token; two speculative
    // tail rows are retracted, then the target correction is processed as the
    // next single-row anchor. This catches small KV differences that compound
    // across rounds even when each individual batched argmax matches.
    model.reset_gpu_caches();
    model
        .forward_buffer(&prompt_tokens, &prompt_positions)
        .expect("re-prefill for repeated M=3 rounds");
    model
        .forward_buffer(&[generated[0]], &[(prompt_tokens.len()) as u32])
        .expect("process first target anchor");
    let mut round_start = 1usize;
    while round_start + 3 <= 80 {
        let tokens = &generated[round_start..round_start + 3];
        let positions: Vec<u32> = (0..3usize)
            .map(|index| (prompt_tokens.len() + round_start + index) as u32)
            .collect();
        model
            .forward_verify(tokens, &positions)
            .expect("repeated M=3 verify");
        model.retract_gpu_caches(2);

        let correction_index = round_start + 1;
        let continuation = model
            .forward_buffer(
                &[generated[correction_index]],
                &[(prompt_tokens.len() + correction_index) as u32],
            )
            .expect("process target correction and next anchor");
        let actual = argmax_row(&continuation, 0, vocab);
        let expected = generated[correction_index + 1];
        let gap = top_two_gap(&after_token_logits[correction_index]);
        eprintln!(
            "repeated M=3 start={round_start} output_index={} actual={actual} sequential={expected} gap={gap:.6}",
            correction_index + 1
        );
        assert_eq!(
            actual,
            expected,
            "repeated M=3 cached continuation mismatch at output token {} (gap {gap:.6})",
            correction_index + 1
        );
        round_start += 2;
    }
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; loads the 16 GB target"]
fn batched_verify_head_matches_sequential_single_row() {
    let path = model_path();
    let dir = Path::new(&path);
    let mut model = TextModel::load_from_dir(dir).expect("target model");
    let vocab = model.vocab_size();

    // Three short real token runs, long enough to cover row selection at each
    // supported speculative width while keeping this ignored GPU test quick.
    for prompt in 0..3u32 {
        let tokens: Vec<u32> = (0..48u32)
            .map(|index| 1 + (index * 7919 + prompt * 1237) % 4096)
            .collect();

        // Compare the whole batched model path, including the head, against the
        // autoregressive path. This includes the M=1 GEMV and M=2..16 MRHS routes.
        for rows in [1usize, 2, 4, 8, 16] {
            // The window is prefilled from a reset cache, so its positions are
            // `0..rows` -- not a slice of a longer conversation's positions.
            let window = &tokens[..rows];
            let positions: Vec<u32> = (0..rows as u32).collect();

            // --- one batched pass ------------------------------------------------
            // The whole window as one chunk, like a prefill. Causal attention means
            // row `i` sees tokens `0..=i`, which is exactly what the sequential path
            // below sees, so the head is the only variable.
            model.reset_gpu_caches();
            let batched = model
                .forward_verify(window, &positions)
                .expect("batched verify");
            assert_eq!(
                batched.len(),
                rows * vocab,
                "the batched head returned {} elements for {rows} rows of {vocab}",
                batched.len()
            );

            // --- `rows` sequential single-row passes ----------------------------
            // One forward per token, with the cache growing the way a decode loop
            // grows it. `positions` must start at the (reset) cache cursor, i.e. 0.
            model.reset_gpu_caches();
            let sequential_positions: Vec<u32> = (0..rows as u32).collect();
            let mut worst = 0.0_f32;
            let mut worst_at = (0usize, 0usize);
            for (index, (token, position)) in
                window.iter().zip(sequential_positions.iter()).enumerate()
            {
                let mut row_worst = 0.0_f32;
                let row = model
                    .forward_buffer(&[*token], &[*position])
                    .expect("single row");
                assert_eq!(
                    argmax_row(&batched, index, vocab),
                    argmax_row(&row, 0, vocab),
                    "greedy token differs at verify row {index} for M={rows}"
                );
                let base = index * vocab;
                for column in 0..vocab {
                    let a = half::f16::from_bits(batched[base + column]).to_f32();
                    let b = half::f16::from_bits(row[column]).to_f32();
                    let diff = (a - b).abs();
                    row_worst = row_worst.max(diff);
                    if diff > worst {
                        worst = diff;
                        worst_at = (index, column);
                    }
                }
                eprintln!("rows {rows} row {index}: worst {row_worst:.6}");
                if rows == 1 {
                    let column = worst_at.1;
                    let a = half::f16::from_bits(batched[column]).to_f32();
                    let b = half::f16::from_bits(row[column]).to_f32();
                    eprintln!(
                        "  col {column}: batched {a:.4} sequential {b:.4} | argmax batched {} sequential {}",
                        argmax_row(&batched, 0, vocab),
                        argmax_row(&row, 0, vocab)
                    );
                }
            }
            let scale = half::f16::from_bits(batched[worst_at.1])
                .to_f32()
                .abs()
                .max(1.0);
            eprintln!(
                "rows {rows}: worst |batched - sequential| = {worst:.6} at row {} col {} \
             (relative {:.4}%)",
                worst_at.0,
                worst_at.1,
                100.0 * worst / scale
            );
            // fp16 softcapped logits use distinct M=1 and batched kernels, whose
            // accumulation order differs. Greedy identity is the correctness gate;
            // The 2.0 absolute bound catches large routing failures (the pre-fix
            // defect exceeded 12); exact greedy row identity is the acceptance gate.
            assert!(
                worst < 2.0,
                "the batched head diverges from the sequential path by {worst} at row {} col {}",
                worst_at.0,
                worst_at.1
            );
        }
    }
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; loads the 16 GB target"]
fn batched_and_sequential_layer_states_match() {
    let path = model_path();
    let dir = Path::new(&path);
    let mut model = TextModel::load_from_dir(dir).expect("target model");
    let tap_layers: Vec<usize> = (0..model.layer_count()).collect();
    model.enable_taps(&tap_layers).expect("enable layer taps");
    let tokens: Vec<u32> = (0..2u32).map(|index| 1 + (index * 7919) % 4096).collect();
    let positions = [0u32, 1];

    model.reset_gpu_caches();
    let _ = model
        .forward_verify(&tokens, &positions)
        .expect("batched verify");
    let batched_taps = model.take_taps(2).expect("batched layer taps");
    let mut batched_cache = Vec::new();
    for layer in 0..model.layer_count() {
        batched_cache.push(model.layer_kv_to_host(layer).expect("batched layer KV"));
    }

    model.reset_gpu_caches();
    let mut sequential_taps = vec![Vec::new(); tap_layers.len()];
    for (token, position) in tokens.iter().zip(positions.iter()) {
        let _ = model
            .forward_buffer(&[*token], &[*position])
            .expect("sequential decode");
        for (layer_taps, row) in sequential_taps
            .iter_mut()
            .zip(model.take_taps(1).expect("sequential layer taps"))
        {
            layer_taps.extend(row);
        }
    }
    let mut hidden_diffs = Vec::with_capacity(tap_layers.len());
    for layer in 0..tap_layers.len() {
        let width = model.hidden_size();
        let mut max_diff = 0.0_f32;
        for row in 0..2 {
            let start = row * width;
            for (left, right) in batched_taps[layer][start..start + width]
                .iter()
                .zip(&sequential_taps[layer][start..start + width])
            {
                max_diff = max_diff.max(
                    (half::f16::from_bits(*left).to_f32() - half::f16::from_bits(*right).to_f32())
                        .abs(),
                );
            }
        }
        hidden_diffs.push(max_diff);
    }
    let mut max_k_diff = 0.0_f32;
    let mut max_v_diff = 0.0_f32;
    let mut max_k_layer = 0usize;
    let mut max_v_layer = 0usize;
    for (layer, (batched, sequential)) in batched_cache
        .iter()
        .zip(
            (0..model.layer_count())
                .map(|layer| model.layer_kv_to_host(layer).expect("sequential layer KV")),
        )
        .enumerate()
    {
        let (batched_k, batched_v, batched_positions, _, kv_heads, head_dim) = batched;
        let (sequential_k, sequential_v, sequential_positions, _, sequential_heads, sequential_dim) =
            sequential;
        assert_eq!(batched_positions[..2], [0, 1], "layer {layer}");
        assert_eq!(sequential_positions[..2], [0, 1], "layer {layer}");
        assert_eq!((*kv_heads, *head_dim), (sequential_heads, sequential_dim));
        let elements = 2 * kv_heads * head_dim;
        let max_diff = |left: &[u16], right: &[u16]| {
            left[..elements]
                .iter()
                .zip(&right[..elements])
                .map(|(a, b)| {
                    (half::f16::from_bits(*a).to_f32() - half::f16::from_bits(*b).to_f32()).abs()
                })
                .fold(0.0_f32, f32::max)
        };
        let k_diff = max_diff(&batched_k, &sequential_k);
        let v_diff = max_diff(&batched_v, &sequential_v);
        if k_diff > max_k_diff {
            max_k_diff = k_diff;
            max_k_layer = layer;
        }
        if v_diff > max_v_diff {
            max_v_diff = v_diff;
            max_v_layer = layer;
        }
    }
    let (max_hidden_layer, max_hidden_diff) = hidden_diffs
        .iter()
        .copied()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .expect("model has layers");
    eprintln!(
        "batched/sequential M=2 max hidden={max_hidden_diff} (L{max_hidden_layer}), K={max_k_diff} (L{max_k_layer}), V={max_v_diff} (L{max_v_layer})"
    );
    assert!(
        max_k_diff < 0.01,
        "layer {max_k_layer} K diverges: {max_k_diff}"
    );
    assert!(
        max_v_diff < 0.1,
        "layer {max_v_layer} V diverges: {max_v_diff}"
    );
    if let Some((layer, max_diff)) = hidden_diffs
        .iter()
        .copied()
        .enumerate()
        .find(|(_, diff)| *diff >= 0.5)
    {
        panic!("layer {layer} hidden states diverge: max {max_diff}");
    }
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; loads the 16 GB target"]
fn batched_verify_head_carries_signal_on_every_row() {
    // A second, independent guard: `head_rows_matrix` returning the same row 16
    // times (a stride bug in the read-back) would make every verify decide from
    // one row's logits and pass any comparison that only checks the diagonal.
    let path = model_path();
    let dir = Path::new(&path);
    let mut model = TextModel::load_from_dir(dir).expect("target model");
    let vocab = model.vocab_size();

    let tokens: Vec<u32> = (0..40u32)
        .map(|index| 1 + (index * 104_729) % 8192)
        .collect();
    model.reset_gpu_caches();
    let positions: Vec<u32> = (0..16u32).collect();
    let batched = model
        .forward_verify(&tokens[..16], &positions)
        .expect("batched verify");
    assert_eq!(batched.len(), 16 * vocab);

    let mut differing = 0usize;
    let mut compared = 0usize;
    for row in 0..16usize {
        let next = (row + 1) % 16;
        for column in 0..vocab {
            compared += 1;
            let a = half::f16::from_bits(batched[row * vocab + column]).to_f32();
            let b = half::f16::from_bits(batched[next * vocab + column]).to_f32();
            if (a - b).abs() > 1e-6 {
                differing += 1;
            }
        }
    }
    eprintln!("adjacent rows differ on {differing}/{compared} vocabulary entries");
    // Adjacent positions genuinely have different distributions. If the rows
    // were a stride bug they would be identical everywhere.
    assert!(
        differing > compared / 100,
        "adjacent verify rows are nearly identical ({differing}/{compared}); the \
         read-back stride is probably wrong"
    );
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; loads the 16 GB target and crosses the sliding-window boundary"]
fn batched_verify_preserves_keys_at_sliding_window_boundary() {
    let path = model_path();
    let mut model = TextModel::load_from_dir(Path::new(&path)).expect("target model");
    let vocab = model.vocab_size();

    // The first query is at position 1023 and must still see key 0. The second
    // query writes position 1024, which aliases slot 0 in the old 1024-slot
    // physical ring. Batched forwards write all rows before attention, so that
    // write must not evict key 0 before row 0 has consumed it.
    let prefix: Vec<u32> = (0..1023u32)
        .map(|index| 1 + (index * 7919 + 17) % 4096)
        .collect();
    let prefix_positions: Vec<u32> = (0..prefix.len() as u32).collect();
    let boundary_tokens = [137u32, 941];
    let boundary_positions = [1023u32, 1024];

    model.reset_gpu_caches();
    model
        .forward_buffer(&prefix, &prefix_positions)
        .expect("prefill through position 1022");
    let batched = model
        .forward_verify(&boundary_tokens, &boundary_positions)
        .expect("batched boundary verify");
    assert_eq!(model.gpu_next_position(), 1025);
    let (_, _, batched_positions, _, _, _) = model
        .layer_kv_to_host(0)
        .expect("read first sliding-layer cache");
    assert!(
        batched_positions.contains(&0),
        "batched row at position 1024 evicted key position 0 before row 0 attended"
    );

    model.reset_gpu_caches();
    model
        .forward_buffer(&prefix, &prefix_positions)
        .expect("sequential control prefill through position 1022");
    let mut worst = 0.0f32;
    let mut worst_at = (0usize, 0usize);
    for row in 0..2 {
        let sequential = model
            .forward_buffer(&[boundary_tokens[row]], &[boundary_positions[row]])
            .expect("sequential boundary row");
        assert_eq!(model.gpu_next_position(), 1024 + row);
        assert_eq!(
            argmax_row(&batched, row, vocab),
            argmax_row(&sequential, 0, vocab),
            "greedy token differs at sliding boundary row {row}"
        );
        for column in 0..vocab {
            let left = half::f16::from_bits(batched[row * vocab + column]).to_f32();
            let right = half::f16::from_bits(sequential[column]).to_f32();
            let diff = (left - right).abs();
            if diff > worst {
                worst = diff;
                worst_at = (row, column);
            }
        }
    }
    eprintln!(
        "sliding boundary M=2: max logit delta {worst:.6} at row {} col {}",
        worst_at.0, worst_at.1
    );
    assert!(
        worst < 2.0,
        "batched boundary verify diverges from sequential by {worst} at row {} col {}",
        worst_at.0,
        worst_at.1
    );
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; loads the 16 GB target"]
fn retracted_verify_rows_do_not_leak_into_the_following_decode() {
    let path = model_path();
    let mut model = TextModel::load_from_dir(Path::new(&path)).expect("target model");
    let vocab = model.vocab_size();
    let contexts = [
        ("before-wrap", 40usize),
        ("at-wrap", 1023),
        ("after-wrap", 1030),
    ];
    let draft = [137u32, 941, 211, 1501];

    for (label, prefix_len) in contexts {
        let prefix: Vec<u32> = (0..prefix_len as u32)
            .map(|index| 1 + (index * 7919 + 17) % 4096)
            .collect();
        let prefix_positions: Vec<u32> = (0..prefix_len as u32).collect();

        for accepted in [0usize, 2, draft.len()] {
            model.reset_gpu_caches();
            model
                .forward_buffer(&prefix, &prefix_positions)
                .expect("prefill rollback prefix");
            let verify_positions: Vec<u32> = (0..draft.len())
                .map(|row| (prefix_len + row) as u32)
                .collect();
            model
                .forward_verify(&draft, &verify_positions)
                .expect("verify speculative rows");
            model.retract_gpu_caches(draft.len() - accepted);
            assert_eq!(
                model.gpu_next_position(),
                prefix_len + accepted,
                "{label}, accepted={accepted}: cache cursor after retract"
            );

            let correction = 2048 + accepted as u32;
            let correction_position = (prefix_len + accepted) as u32;
            let rolled_back = model
                .forward_buffer(&[correction], &[correction_position])
                .expect("feed correction at first rejected position");
            let continuation = argmax_row(&rolled_back, 0, vocab);
            let rolled_back_next = model
                .forward_buffer(&[continuation], &[correction_position + 1])
                .expect("decode after correction");

            // Independent sequential reference contains only the accepted draft
            // prefix, followed by the target correction and its next greedy token.
            model.reset_gpu_caches();
            model
                .forward_buffer(&prefix, &prefix_positions)
                .expect("sequential reference prefill");
            for row in 0..accepted {
                model
                    .forward_buffer(&[draft[row]], &[(prefix_len + row) as u32])
                    .expect("sequential accepted draft");
            }
            let reference = model
                .forward_buffer(&[correction], &[correction_position])
                .expect("sequential target correction");
            assert_eq!(
                argmax_row(&rolled_back, 0, vocab),
                argmax_row(&reference, 0, vocab),
                "{label}, accepted={accepted}: correction greedy token"
            );
            let reference_continuation = argmax_row(&reference, 0, vocab);
            let reference_next = model
                .forward_buffer(&[reference_continuation], &[correction_position + 1])
                .expect("sequential continuation");
            let mut worst = 0.0f32;
            for column in 0..vocab {
                let actual = half::f16::from_bits(rolled_back_next[column]).to_f32();
                let expected = half::f16::from_bits(reference_next[column]).to_f32();
                worst = worst.max((actual - expected).abs());
            }
            assert!(
                worst < 2.0,
                "{label}, accepted={accepted}: post-retract logits differ by {worst}"
            );
            eprintln!(
                "rollback {label}, accepted={accepted}/{}: next logits max delta {worst:.6}",
                draft.len()
            );
        }
    }
}
