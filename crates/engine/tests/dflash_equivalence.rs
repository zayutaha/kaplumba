//! T12 greedy end-to-end DFlash correctness against the target's own decode.
//! Ignored by default because it loads the target and DFlash checkpoints.

use std::path::Path;
use std::time::Instant;

use kaplumba_engine::dflash::{
    DflashConfig, DflashKvCache, DflashModel, DflashSamplingConfig, DflashWeights,
};
use kaplumba_engine::model::TextModel;
use kaplumba_engine::tokenizer::GemmaTokenizer;

const PROMPTS: [&str; 3] = [
    "Explain the theory of relativity in simple terms. Special relativity says the speed of light is the same for every observer.",
    "Explain why coastal regions often have milder temperatures than inland areas, and give two examples.",
    "Describe a practical way to plan a small garden, from choosing plants to watering them through the first week.",
];
const OUTPUT_TOKENS: usize = 200;

fn model_path() -> String {
    std::env::var("KAPLUMBA_TEST_MODEL").expect("KAPLUMBA_TEST_MODEL is required")
}

fn drafter_path() -> String {
    std::env::var("KAPLUMBA_DFLASH_DIR").expect("KAPLUMBA_DFLASH_DIR is required")
}

fn argmax(logits: &[u16], vocab: usize) -> u32 {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (token, bits) in logits.iter().take(vocab).enumerate() {
        let value = half::f16::from_bits(*bits).to_f32();
        if value > best_value {
            best = token;
            best_value = value;
        }
    }
    best as u32
}

fn greedy_reference(target: &mut TextModel, prompt: &[u32], tokens: usize) -> (Vec<u32>, f64) {
    target.reset_gpu_caches();
    target.enable_taps(&[]).expect("disable taps for baseline");
    let positions: Vec<u32> = (0..prompt.len() as u32).collect();
    let mut logits = target
        .forward_buffer(prompt, &positions)
        .expect("reference prefill");
    let vocab = target.vocab_size();
    let mut output = Vec::with_capacity(tokens);
    let mut position = prompt.len() as u32;
    let generation_start = Instant::now();
    while output.len() < tokens {
        let token = argmax(&logits, vocab);
        output.push(token);
        logits = target
            .forward_buffer(&[token], &[position])
            .expect("reference decode");
        position += 1;
    }
    (output, generation_start.elapsed().as_secs_f64())
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL and KAPLUMBA_DFLASH_DIR; loads both models"]
fn dflash_greedy_stream_matches_target_for_200_tokens_on_three_prompts() {
    let model_dir = model_path();
    let drafter_dir = drafter_path();
    let tokenizer = GemmaTokenizer::from_model_dir(Path::new(&model_dir)).expect("tokenizer");
    let mut target = TextModel::load_from_dir(Path::new(&model_dir)).expect("target model");
    let device = metal::Device::system_default().expect("Metal device");
    let weights = DflashWeights::load(Path::new(&drafter_dir), &device).expect("DFlash weights");
    let config = DflashConfig::load(Path::new(&drafter_dir)).expect("DFlash config");
    let mut drafter = DflashModel::new(weights, config.clone()).expect("DFlash model");
    let mask = target.raw_embedding(&[config.mask_token_id]);
    drafter.set_mask_embedding(&mask).expect("mask embedding");
    let vocab = target.vocab_size();
    let sampling = DflashSamplingConfig {
        temperature: 0.0,
        ..DflashSamplingConfig::default()
    };

    for (prompt_index, prompt) in PROMPTS.iter().enumerate() {
        let prompt_tokens = tokenizer.encode_prompt(prompt).expect("encode prompt");
        assert!(
            prompt_tokens.len() + OUTPUT_TOKENS < 8192,
            "prompt {prompt_index} does not fit the target context"
        );

        target.reset_gpu_caches();
        target
            .enable_taps(&config.target_layer_ids)
            .expect("enable DFlash taps");
        let prompt_positions: Vec<u32> = (0..prompt_tokens.len() as u32).collect();
        let prompt_logits = target
            .forward_buffer(&prompt_tokens, &prompt_positions)
            .expect("DFlash prompt prefill");
        let prompt_taps = target.take_taps(prompt_tokens.len()).expect("prompt taps");
        let prompt_tap_refs: Vec<&[u16]> = prompt_taps.iter().map(Vec::as_slice).collect();
        let mut drafter_cache = DflashKvCache::new(&config);
        drafter
            .append_context(
                &prompt_tap_refs,
                prompt_tokens.len(),
                prompt_tokens.len(),
                &mut drafter_cache,
            )
            .expect("initialize DFlash context");

        let mut output = vec![argmax(&prompt_logits, vocab)];
        let mut anchor_token = output[0];
        let mut anchor_position = prompt_tokens.len();
        let generation_start = Instant::now();
        let mut rounds = 0usize;
        let mut accepted_total = 0usize;
        let mut anchor_seconds = 0.0f64;
        let mut draft_verify_seconds = 0.0f64;
        let mut verify_tap_seconds = 0.0f64;
        let mut context_seconds = 0.0f64;
        while output.len() < OUTPUT_TOKENS {
            let anchor_start = Instant::now();
            let anchor_logits = target
                .forward_buffer(&[anchor_token], &[anchor_position as u32])
                .expect("process round anchor");
            let anchor_taps = target.take_taps(1).expect("save anchor taps");
            anchor_seconds += anchor_start.elapsed().as_secs_f64();

            let round_start = Instant::now();
            let round = drafter
                .draft_and_verify(
                    &mut target,
                    &mut drafter_cache,
                    anchor_token,
                    &anchor_logits,
                    anchor_position,
                    sampling,
                    || 0.5,
                )
                .expect("draft and verify");
            draft_verify_seconds += round_start.elapsed().as_secs_f64();
            let tap_start = Instant::now();
            let verify_taps = target
                .take_taps(round.drafted)
                .expect("capture verify taps before next target forward");
            verify_tap_seconds += tap_start.elapsed().as_secs_f64();

            let accepted = round.accepted.len();
            let mut context_rows = Vec::with_capacity(1 + accepted);
            for layer in 0..config.num_taps() {
                let mut rows = anchor_taps[layer].clone();
                rows.extend_from_slice(&verify_taps[layer][..accepted * config.hidden_size]);
                context_rows.push(rows);
            }
            let context_refs: Vec<&[u16]> = context_rows.iter().map(Vec::as_slice).collect();
            let correction_position = anchor_position + 1 + accepted;
            let context_start = Instant::now();
            drafter
                .append_context(
                    &context_refs,
                    1 + accepted,
                    correction_position,
                    &mut drafter_cache,
                )
                .expect("append anchor and accepted verify taps once");
            context_seconds += context_start.elapsed().as_secs_f64();

            accepted_total += accepted;
            output.extend(round.accepted);
            output.push(round.next);
            anchor_token = round.next;
            anchor_position = correction_position;
            rounds += 1;
        }
        let elapsed = generation_start.elapsed().as_secs_f64();
        output.truncate(OUTPUT_TOKENS);

        let (reference, plain_elapsed) =
            greedy_reference(&mut target, &prompt_tokens, OUTPUT_TOKENS);
        assert_eq!(
            output, reference,
            "greedy DFlash output differs for prompt {prompt_index}"
        );
        eprintln!(
            "prompt {prompt_index}: greedy exact, rounds={rounds}, accepted={accepted_total}, emitted={OUTPUT_TOKENS}, DFlash={elapsed:.3}s ({:.2} tok/s), plain={plain_elapsed:.3}s ({:.2} tok/s), per-round anchor={:.2}ms round={:.2}ms tap-read={:.2}ms context={:.2}ms",
            OUTPUT_TOKENS as f64 / elapsed,
            OUTPUT_TOKENS as f64 / plain_elapsed,
            anchor_seconds * 1000.0 / rounds as f64,
            draft_verify_seconds * 1000.0 / rounds as f64,
            verify_tap_seconds * 1000.0 / rounds as f64,
            context_seconds * 1000.0 / rounds as f64,
        );
    }
}
