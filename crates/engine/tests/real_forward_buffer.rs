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
    // The point of the check is that the scratch is a function of PREFILL_CHUNK
    // and not of PROMPT, so the budget is per row rather than a fixed 100 MB.
    // The scratch is ~366 KB/row and barely changes per row (93.7 MB at 256,
    // 372 MB at 1024), so 400 KB/row is 1.09x the real figure -- tight enough
    // to catch a new buffer, and still 22x below what a prompt-sized scratch
    // (raw alone is 672 MiB at 8192 tokens) would need. At 1024 rows that
    // 372 MB is a deliberate, measured trade; see PREFILL_CHUNK.
    const BUDGET_PER_ROW: u64 = 400 * 1024;
    let scratch = model.scratch_resident_bytes();
    assert!(
        scratch <= BUDGET_PER_ROW * PREFILL_CHUNK as u64,
        "scratch is {scratch} bytes, over the {} byte budget for {PREFILL_CHUNK} rows",
        BUDGET_PER_ROW * PREFILL_CHUNK as u64
    );
    let (weights, kv, scratch) = model.allocation_breakdown();
    println!("weights={weights} kv={kv} scratch={scratch}");
}
