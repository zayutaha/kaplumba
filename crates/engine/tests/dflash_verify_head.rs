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
