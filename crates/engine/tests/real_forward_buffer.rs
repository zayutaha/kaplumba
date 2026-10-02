//! Real-checkpoint exercise of the GPU-resident forward path.
//!
//! Isolated and ignored by default so a crash here points at `forward_buffer`
//! rather than at generation, sampling, or detokenisation.

use kaplumba_engine::model::{PREFILL_CHUNK, TextModel};

fn model_path() -> String {
    std::env::var("KAPLUMBA_TEST_MODEL").expect("KAPLUMBA_TEST_MODEL is required")
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; runs the full GPU forward path"]
fn real_model_buffer_forward_single_token() {
    let mut model = TextModel::load_from_dir(std::path::Path::new(&model_path())).unwrap();
    model.reset_gpu_caches();
    let logits = model.forward_buffer(&[2], &[0]).unwrap();
    assert_eq!(logits.len(), model.vocab_size());
    assert!(logits.iter().any(|value| *value != 0));
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; runs the full GPU forward path"]
fn real_model_buffer_forward_prompt() {
    let mut model = TextModel::load_from_dir(std::path::Path::new(&model_path())).unwrap();
    model.reset_gpu_caches();
    let prompt = [
        2_u32, 105, 2364, 107, 9259, 106, 107, 105, 4368, 107, 100, 45518, 107, 101,
    ];
    let positions: Vec<u32> = (0..prompt.len() as u32).collect();
    let logits = model.forward_buffer(&prompt, &positions).unwrap();
    assert_eq!(logits.len(), model.vocab_size());
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; prefill then repeated decode on the GPU path"]
fn real_model_buffer_forward_prefill_then_decode() {
    let mut model = TextModel::load_from_dir(std::path::Path::new(&model_path())).unwrap();
    model.reset_gpu_caches();
    let prompt = [
        2_u32, 105, 2364, 107, 9259, 106, 107, 105, 4368, 107, 100, 45518, 107, 101,
    ];
    let positions: Vec<u32> = (0..prompt.len() as u32).collect();
    let logits = model.forward_buffer(&prompt, &positions).unwrap();
    assert_eq!(logits.len(), model.vocab_size());
    let mut argmax = 0;
    for (index, value) in logits.iter().enumerate() {
        if half::f16::from_bits(*value).to_f32() > half::f16::from_bits(logits[argmax]).to_f32() {
            argmax = index;
        }
    }
    let mut position = prompt.len() as u32;
    for step in 0..4 {
        let logits = model.forward_buffer(&[argmax as u32], &[position]).unwrap();
        assert_eq!(logits.len(), model.vocab_size(), "step {step}");
        argmax = 0;
        for (index, value) in logits.iter().enumerate() {
            if half::f16::from_bits(*value).to_f32() > half::f16::from_bits(logits[argmax]).to_f32()
            {
                argmax = index;
            }
        }
        position += 1;
    }
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; compares slice and buffer paths on the real model"]
fn real_model_buffer_forward_matches_slice_forward() {
    let mut slice_model = TextModel::load_from_dir(std::path::Path::new(&model_path())).unwrap();
    let mut buffer_model = TextModel::load_from_dir(std::path::Path::new(&model_path())).unwrap();
    buffer_model.reset_gpu_caches();
    let prompt = [2_u32, 105, 2364, 107, 9259];
    let positions: Vec<u32> = (0..prompt.len() as u32).collect();

    let slice = slice_model.forward(&prompt, &positions).unwrap();
    let buffer = buffer_model.forward_buffer(&prompt, &positions).unwrap();
    let vocab = slice_model.vocab_size();
    let last = &slice[(prompt.len() - 1) * vocab..];

    let mut worst = 0.0_f32;
    let mut worst_index = 0;
    for (index, (a, b)) in last.iter().zip(buffer.iter()).enumerate() {
        let a = half::f16::from_bits(*a).to_f32();
        let b = half::f16::from_bits(*b).to_f32();
        let delta = (a - b).abs();
        if delta > worst {
            worst = delta;
            worst_index = index;
        }
    }
    let argmax_slice = last
        .iter()
        .enumerate()
        .max_by(|a, b| {
            half::f16::from_bits(*a.1)
                .to_f32()
                .partial_cmp(&half::f16::from_bits(*b.1).to_f32())
                .unwrap()
        })
        .map(|(i, _)| i);
    let argmax_buffer = buffer
        .iter()
        .enumerate()
        .max_by(|a, b| {
            half::f16::from_bits(*a.1)
                .to_f32()
                .partial_cmp(&half::f16::from_bits(*b.1).to_f32())
                .unwrap()
        })
        .map(|(i, _)| i);
    eprintln!(
        "worst delta {worst} at {worst_index}; argmax slice={argmax_slice:?} buffer={argmax_buffer:?}"
    );
    assert!(worst <= 0.2, "worst logit delta {worst} at {worst_index}");
}

/// Reports the residual-stream magnitude after every layer for the real prompt.
///
/// A correct 60-layer stack keeps a slowly growing, bounded norm. A blown-up or
/// collapsed norm localises the defect to a layer instead of leaving us with only
/// "the text is incoherent".
#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; per-layer residual diagnostic"]
fn real_model_per_layer_hidden_norms() {
    use half::f16;
    let mut model = TextModel::load_from_dir(std::path::Path::new(&model_path())).unwrap();
    let prompt = [
        2_u32, 105, 2364, 107, 9259, 106, 107, 105, 4368, 107, 100, 45518, 107, 101,
    ];
    let positions: Vec<u32> = (0..prompt.len() as u32).collect();

    let mut hidden = model.embedding(&prompt);
    for index in 0..model.layer_count() {
        hidden = model.forward_layer(index, &hidden, &positions);
        let sum: f64 = hidden
            .iter()
            .map(|value| {
                let value = f16::from_bits(*value).to_f32() as f64;
                value * value
            })
            .sum();
        let rms = (sum / hidden.len() as f64).sqrt();
        let finite = hidden
            .iter()
            .all(|value| f16::from_bits(*value).to_f32().is_finite());
        println!(
            "layer {index:2} hidden_rms={rms:.5} finite={finite} len={}",
            hidden.len()
        );
    }
}

/// Internal trace of a healthy full-attention layer (53) and the final layer
/// (59), each fed the hidden state it actually receives in a single stack pass.
#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; layer-internal trace"]
fn real_model_trace_layers_53_and_59() {
    let mut model = TextModel::load_from_dir(std::path::Path::new(&model_path())).unwrap();
    let prompt = [
        2_u32, 105, 2364, 107, 9259, 106, 107, 105, 4368, 107, 100, 45518, 107, 101,
    ];
    let positions: Vec<u32> = (0..prompt.len() as u32).collect();

    let mut hidden = model.embedding(&prompt);
    let mut captured: Vec<(usize, Vec<u16>)> = Vec::new();
    for index in 0..model.layer_count() {
        if index == 53 || index == 59 {
            captured.push((index, hidden.clone()));
        }
        hidden = model.forward_layer(index, &hidden, &positions);
    }
    for (index, input) in captured {
        model.reset_gpu_caches();
        model.reset_caches().unwrap();
        // Replay to the capture point so the layer's cache is populated.
        let mut warm = model.embedding(&prompt);
        for step in 0..index {
            warm = model.forward_layer(step, &warm, &positions);
        }
        let trace = model.trace_layer(index, &input, &positions);
        println!("--- layer {index} (input_rms captured separately) ---");
        for (name, value) in trace {
            println!("    {name:16} {value:.6}");
        }
    }
}

/// Print the top-10 next-token candidates with probabilities.
///
/// A correct EXL3 decode gives a confident, plausible distribution. Scrambled
/// weights give a near-uniform spread over junk, which makes this the cheapest
/// high-signal check for the tile permutation.
#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; top-k next-token diagnostic"]
fn real_model_top_tokens() {
    let mut model = TextModel::load_from_dir(std::path::Path::new(&model_path())).unwrap();
    model.reset_gpu_caches();
    let prompt = [
        2_u32, 105, 2364, 107, 9259, 106, 107, 105, 4368, 107, 100, 45518, 107, 101,
    ];
    let positions: Vec<u32> = (0..prompt.len() as u32).collect();
    let logits = model.forward_buffer(&prompt, &positions).unwrap();

    let mut scored: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(index, bits)| (index, half::f16::from_bits(*bits).to_f32()))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let max = scored[0].1;
    let sum_exp: f32 = scored
        .iter()
        .take(2000)
        .map(|(_, v)| ((v - max) as f64).exp() as f32)
        .sum();
    println!(
        "logit max={max:.3}  top1_prob_approx={:.4}",
        1.0 / sum_exp.max(1.0)
    );
    for (rank, (index, value)) in scored.iter().take(10).enumerate() {
        let p = (((*value - max) as f64).exp() / sum_exp.max(1.0) as f64) as f32;
        println!("  {rank:2} id={index:6} logit={value:8.3} p={p:.4}");
    }
    // Entropy over the top 2000 as a degeneracy probe.
    let entropy: f32 = scored
        .iter()
        .take(2000)
        .map(|(_, v)| {
            let p = (((*v - max) as f64).exp() / sum_exp.max(1.0) as f64) as f32;
            if p > 0.0 { -p * p.ln() } else { 0.0 }
        })
        .sum();
    println!("top2000 entropy={entropy:.3} (uniform would be ~7.6)");
}

/// Predictability probe: a correct model nails trivially predictable
/// continuations. Broken weight decode cannot. Loads once, scores several.
#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; predictability probe"]
fn real_model_predictability_probe() {
    let mut model = TextModel::load_from_dir(std::path::Path::new(&model_path())).unwrap();
    let tokenizer = kaplumba_engine::tokenizer::GemmaTokenizer::from_model_dir(
        std::path::Path::new(&model_path()),
    )
    .unwrap();
    let cases = [
        "Count: 1 2 3 4 5 6 7 8 9",
        "The capital of France is",
        "one two three four five six seven eight nine ten",
        "aaaaaaaaaa",
        "2 + 2 = 4, 4 + 4 = 8, 8 + 8 =",
    ];
    for case in cases {
        let ids = tokenizer.encode_prompt(case).unwrap();
        let positions: Vec<u32> = (0..ids.len() as u32).collect();
        model.reset_gpu_caches();
        let logits = model.forward_buffer(&ids, &positions).unwrap();
        let mut scored: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, b)| (i, half::f16::from_bits(*b).to_f32()))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let max = scored[0].1;
        let z: f64 = scored
            .iter()
            .take(500)
            .map(|(_, v)| ((*v - max) as f64).exp())
            .sum();
        let top: Vec<String> = scored
            .iter()
            .take(5)
            .map(|(i, v)| {
                let p = (((*v - max) as f64).exp() / z) as f32;
                format!(
                    "{}:{:.3}",
                    tokenizer
                        .decode_all(&[*i as u32], true)
                        .unwrap_or("?".into()),
                    p
                )
            })
            .collect();
        println!("PROBE {case:?}");
        println!("   top1={:.3}  top5=[{}]", 1.0 / z as f32, top.join("  "));
    }
}

/// Sweep the candidate EXL3 tile permutations and score each with a
/// predictability probe. The correct permutation is the one that makes the model
/// actually know things, e.g. "The capital of France is" -> " Paris".
#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; permutation sweep"]
fn real_model_permutation_sweep() {
    let owned = model_path();
    let path = std::path::Path::new(&owned);
    let tokenizer = kaplumba_engine::tokenizer::GemmaTokenizer::from_model_dir(path).unwrap();
    let cases = [
        ("The capital of France is", " Paris"),
        ("Count: 1 2 3 4 5 6 7 8 9", " 10"),
    ];
    for mode in 0..=9_u32 {
        let mut model = TextModel::load_from_dir(path).unwrap();
        model.set_permutation_mode(mode);
        let mut line = format!("MODE {mode:2}");
        let mut best = 0.0_f32;
        for (prompt, want) in cases {
            let ids = tokenizer.encode_prompt(prompt).unwrap();
            let positions: Vec<u32> = (0..ids.len() as u32).collect();
            model.reset_gpu_caches();
            let logits = model.forward_buffer(&ids, &positions).unwrap();
            let mut scored: Vec<(usize, f32)> = logits
                .iter()
                .enumerate()
                .map(|(i, b)| (i, half::f16::from_bits(*b).to_f32()))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let max = scored[0].1;
            let z: f64 = scored
                .iter()
                .take(500)
                .map(|(_, v)| ((*v - max) as f64).exp())
                .sum();
            let top1 = 1.0 / z as f32;
            let winner = scored[0].0 as u32;
            let text = tokenizer
                .decode_all(&[winner], true)
                .unwrap_or_else(|_| "?".to_string());
            let hit = if text.trim() == want.trim() {
                "HIT"
            } else {
                "   "
            };
            line.push_str(&format!("  [{hit} p={top1:.3} {:?}]", text));
            if top1 > best {
                best = top1;
            }
        }
        line.push_str(&format!("   best_p={best:.3}"));
        println!("{line}");
    }
}

/// Bisect: compare the GPU-resident and host forward paths at 1 token and at a
/// multi-token prefill, to isolate which path diverges.
#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; path bisect"]
fn real_model_path_bisect() {
    use half::f16;
    let path_str = model_path();
    let p = std::path::Path::new(&path_str);
    let prompt = [2_u32, 105, 2364, 107, 9259];

    for tokens in [1_usize, 5_usize] {
        let mut host = TextModel::load_from_dir(p).unwrap();
        let mut gpu = TextModel::load_from_dir(p).unwrap();
        gpu.reset_gpu_caches();
        let ids = &prompt[..tokens];
        let positions: Vec<u32> = (0..tokens as u32).collect();
        let h = host.forward(ids, &positions).unwrap();
        let g = gpu.forward_buffer(ids, &positions).unwrap();
        let vocab = host.vocab_size();
        let last = &h[(tokens - 1) * vocab..];
        let arg = |v: &[u16]| -> usize {
            v.iter()
                .enumerate()
                .max_by(|a, b| {
                    f16::from_bits(*a.1)
                        .to_f32()
                        .partial_cmp(&f16::from_bits(*b.1).to_f32())
                        .unwrap()
                })
                .map(|(i, _)| i)
                .unwrap()
        };
        let mut worst = 0.0f32;
        for (a, b) in last.iter().zip(g.iter()) {
            worst = worst.max((f16::from_bits(*a).to_f32() - f16::from_bits(*b).to_f32()).abs());
        }
        println!(
            "BISECT tokens={tokens}  argmax_host={} argmax_gpu={}  worst_delta={worst:.4}",
            arg(last),
            arg(&g)
        );
    }
}

/// AGENTS.md §8: the activation scratch must be explicitly owned, fixed-size and
/// independent of context length.
///
/// Before chunked prefill, a prompt of N tokens made `ForwardScratch` N rows
/// deep -- `raw` alone is `tokens * inter * 4` bytes, 672 MiB at 8192 tokens.
/// Now the pass runs in `PREFILL_CHUNK`-row chunks, so the scratch is capped no
/// matter how long the prompt is and only the pre-allocated KV cache grows.
///
/// PROMPT is deliberately greater than one chunk, so the assertion is that the
/// scratch did *not* scale to the prompt's full length. It is written relative
/// to `PREFILL_CHUNK` because that constant is a measured performance choice
/// that has moved before (64 -> 256 -> 1024) and a hardcoded 300 would have
/// silently stopped spanning a chunk the day it crossed 300.
#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL; asserts scratch does not scale with prompt length"]
fn real_model_prefill_scratch_is_independent_of_prompt_length() {
    let mut model = TextModel::load_from_dir(std::path::Path::new(&model_path())).unwrap();
    model.reset_gpu_caches();

    const SLACK: usize = 44;
    let PROMPT: usize = PREFILL_CHUNK + SLACK;
    assert!(PROMPT > PREFILL_CHUNK, "test must span more than one chunk");
    let tokens = vec![2_u32; PROMPT];
    let positions: Vec<u32> = (0..PROMPT as u32).collect();

    let logits = model.forward_buffer(&tokens, &positions).unwrap();
    assert_eq!(logits.len(), model.vocab_size());

    assert_eq!(
        model.scratch_max_tokens(),
        PREFILL_CHUNK,
        "scratch grew to {} rows for a {PROMPT}-token prompt; chunking regressed",
        model.scratch_max_tokens()
    );
    check_scratch_budget(model.scratch_resident_bytes(), model.scratch_max_tokens());
    let (weights, kv, scratch) = model.allocation_breakdown();
    println!("weights={weights} kv={kv} scratch={scratch}");
}

/// Bytes the scratch really occupies at `PREFILL_CHUNK` = 1024 rows.
///
/// Derived from the allocation in `ForwardScratch::new` (model.rs:2978-3042) at
/// the checkpoint's shapes (`hidden` 5376, `inter` 21504, `q_max` 32x512 =
/// 16384, `kv_max` 16x256 = 4096, `vocab` 262144, `attn_partials`
/// `decode_partial_elements(32, 512)` = 16x32x514 = 263,168):
///
/// ```text
/// 5 x hidden x u16          55,050,240
/// 4 x inter  x u16         176,160,768
/// 3 x inter  x f32         264,241,152   <- raw, raw_b, raw_c
/// q_max     x u16          33,554,432
/// 2 x kv_max x u16          16,777,216
/// fixed (query_positions, token_ids, staging, host_tokens,
///        last_hidden, logits, capped, attn_partials, argmax_keys)
///                              2,153,232
///                            -----------
///                            547,937,040
/// ```
const SCRATCH_BYTES_AT_PREFILL_CHUNK: u64 = 547_937_040;

/// Per-row budget, `~10 %` above [`SCRATCH_BYTES_AT_PREFILL_CHUNK`] / 1024 =
/// 535,094.77 B. Still 143x below what a prompt-sized scratch would need (`raw`
/// alone is 672 MiB at 8192 tokens).
const SCRATCH_BUDGET_BYTES_PER_ROW: u64 = 590_000;

/// The budget was `400 * 1024 = 409,600` B/row, described as "1.09x the real
/// figure" where the "real figure" was
/// `resident_bytes() / 1024 = 354.4 KiB/row`. That number was 168 MiB light:
/// `ForwardScratch::resident_bytes()` (model.rs:3045-3076) sums 24 buffers while
/// `ForwardScratch::new` (model.rs:2989-2990) allocates 26. `raw_b` and `raw_c`
/// are absent from the list and both are genuinely used --
/// `encode_gemv_fused3` is handed `&scratch.raw, &scratch.raw_b, &scratch.raw_c`
/// (model.rs:1525-1527, `:1582`, `:2210`). So the budget was **0.766x** the
/// truth: it passed *because* of the omission and breaks the day that is fixed.
///
/// Fixed by `spec/memory-swap-cliff` F5, which replaces the hand-maintained list
/// with one `all_buffers()` source of truth. Until then the constant here is the
/// honest number and `scratch_resident_bytes()` is 168 MiB short of it, which is
/// why this check has to accept both figures.
fn check_scratch_budget(scratch: u64, max_tokens: usize) {
    // Compile-time tripwire: the budget must stay above the derived figure, and
    // since both are constants this is a build failure rather than a test
    // failure. (It is how the sabotage "restore 400 KiB/row" was caught:
    // E0080, evaluation panicked.)
    const _: () = assert!(
        SCRATCH_BUDGET_BYTES_PER_ROW > (SCRATCH_BYTES_AT_PREFILL_CHUNK / 1024)
    );
    assert!(max_tokens > 0);
    assert!(
        SCRATCH_BUDGET_BYTES_PER_ROW > (SCRATCH_BYTES_AT_PREFILL_CHUNK / 1024),
        "budget {SCRATCH_BUDGET_BYTES_PER_ROW} must exceed the derived {}",
        SCRATCH_BYTES_AT_PREFILL_CHUNK / 1024,
    );
    assert!(
        scratch <= SCRATCH_BUDGET_BYTES_PER_ROW * max_tokens as u64,
        "scratch is {scratch} bytes ({:.1} KiB/row), over the {} byte budget \
         ({:.1} KiB/row) for {max_tokens} rows -- a new scratch buffer?",
        scratch as f64 / 1024.0 / max_tokens as f64,
        SCRATCH_BUDGET_BYTES_PER_ROW * max_tokens as u64,
        SCRATCH_BUDGET_BYTES_PER_ROW as f64 / 1024.0,
    );
}

/// The budget, without a checkpoint.
///
/// Two things are being asserted here and neither is obvious:
///
///  * the budget accepts *both* the figure `resident_bytes()` reports today
///    (371,776,272 B, 24 buffers) and the figure actually allocated
///    (547,937,040 B, 26 buffers), so it does not break when F5 lands, and
///  * the old 400 KiB/row budget did **not** -- it is below the truth, which is
///    how a budget calibrated on an under-reported measurement passes for years.
#[test]
fn the_scratch_budget_accepts_the_truth_and_rejects_a_new_buffer() {
    const REPORTED_TODAY: u64 = 371_776_272;
    const ACTUALLY_ALLOCATED: u64 = SCRATCH_BYTES_AT_PREFILL_CHUNK;
    const OLD_BUDGET_PER_ROW: u64 = 400 * 1024;

    assert_eq!(ACTUALLY_ALLOCATED - REPORTED_TODAY, 176_160_768);
    assert_eq!(REPORTED_TODAY + 88_080_384, 459_856_656, "one raw* buffer");

    // Both the reported and the true figure are inside the budget.
    check_scratch_budget(REPORTED_TODAY, PREFILL_CHUNK);
    check_scratch_budget(ACTUALLY_ALLOCATED, PREFILL_CHUNK);

    // The old budget could not have held the truth -- it passed only because the
    // accounting was short by 168 MiB.
    assert!(
        OLD_BUDGET_PER_ROW * (PREFILL_CHUNK as u64) < ACTUALLY_ALLOCATED,
        "the premise of this test changed: 400 KiB/row would now be above the \
         real {} bytes",
        ACTUALLY_ALLOCATED,
    );
    assert!(
        std::panic::catch_unwind(|| {
            check_scratch_budget(
                SCRATCH_BUDGET_BYTES_PER_ROW * (PREFILL_CHUNK as u64) + 1,
                PREFILL_CHUNK,
            )
        })
        .is_err(),
        "the budget must reject a scratch that is over it"
    );

    // One more inter-sized f32 accumulator -- what F5's omission amounts to --
    // must be caught.
    assert!(
        std::panic::catch_unwind(|| {
            check_scratch_budget(ACTUALLY_ALLOCATED + 88_080_384, PREFILL_CHUNK)
        })
        .is_err(),
        "a new 84 MiB scratch buffer must be caught by the budget"
    );
}
