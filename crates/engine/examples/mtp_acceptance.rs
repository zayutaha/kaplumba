//! spec/mtp-40tps T4: **the acceptance gate**. Measure real drafter-vs-target
//! token agreement for K draft steps, under greedy and temp-1.0 sampling.
//!
//! This is the single number that decides whether MTP can reach 40 tok/s on this
//! box. It runs the real Gxma target (60 layers, EXL3 3 bpw) and the real
//! drafter (Gemma4 assistant, affine-4bit) and measures, per round, how many of
//! the K drafted tokens the target's own greedy/sampled continuation confirms.
//!
//! It does NOT implement the fast M>1 verify — it verifies one token at a time
//! (M=1) because the acceptance *rate* is a property of the drafter and the
//! target, not of how the verify is batched. Batching only changes how fast the
//! confirmed tokens are produced, never which tokens they are. So this measures
//! the gate number without building T8a first; T8a is still required for the
//! 40 tok/s *speed*, and the gate tells us whether that speed is worth building.
//!
//! Usage:
//!   cargo run --release --example mtp_acceptance -- \
//!     --target ~/.kaplumba/models/Gxma/Gxma \
//!     --drafter ~/.omlx/models/mlx-community/gemma-4-31B-it-qat-assistant-4bit \
//!     --rounds 40 --k 5 [--temperature 1.0 --top-k 64 --seed 7]

use half::f16;
use kaplumba_engine::config::load_config;
use kaplumba_engine::drafter::DrafterWeights;
use kaplumba_engine::drafter_model::{DrafterModel, DrafterSharedKv, KvSlice};
use kaplumba_engine::model::TextModel;
use kaplumba_engine::sampler::{Sampler, SamplerConfig};
use metal::Device;
use std::time::Instant;

struct Args {
    target: String,
    drafter: String,
    rounds: usize,
    k: usize,
    temperature: f32,
    top_k: usize,
    seed: u64,
    prompt: String,
}

fn parse_args() -> Args {
    let mut target = String::new();
    let mut drafter = String::new();
    let mut rounds = 40usize;
    let mut k = 5usize;
    let mut temperature = 0.0f32;
    let mut top_k = 64usize;
    let mut seed = 7u64;
    let mut prompt =
        "The quick brown fox jumps over the lazy dog. The weather today is".to_string();
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--target" => {
                target = argv.get(i + 1).cloned().unwrap_or_default();
                i += 2;
            }
            "--drafter" => {
                drafter = argv.get(i + 1).cloned().unwrap_or_default();
                i += 2;
            }
            "--rounds" => {
                rounds = argv[i + 1].parse().unwrap_or(rounds);
                i += 2;
            }
            "--k" => {
                k = argv[i + 1].parse().unwrap_or(k);
                i += 2;
            }
            "--temperature" => {
                temperature = argv[i + 1].parse().unwrap_or(temperature);
                i += 2;
            }
            "--top-k" => {
                top_k = argv[i + 1].parse().unwrap_or(top_k);
                i += 2;
            }
            "--seed" => {
                seed = argv[i + 1].parse().unwrap_or(seed);
                i += 2;
            }
            "--prompt" => {
                prompt = argv.get(i + 1).cloned().unwrap_or(prompt);
                i += 2;
            }
            _ => i += 1,
        }
    }
    Args {
        target,
        drafter,
        rounds,
        k,
        temperature,
        top_k,
        seed,
        prompt,
    }
}

/// Softmax sampling from a capped-logit row (f16 bit patterns), matching the
/// engine's sampler semantics: greedy at temperature 0, else top-k + temperature
/// + top_p.
fn sample_token(logits: &[u16], sampler: &mut Sampler) -> u32 {
    let row: Vec<f32> = logits.iter().map(|v| f16::from_bits(*v).to_f32()).collect();
    sampler.sample(&row).expect("sample")
}

fn greedy_token(logits: &[u16]) -> u32 {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, v) in logits.iter().enumerate() {
        let f = f16::from_bits(*v).to_f32();
        if f > best_val {
            best_val = f;
            best = i;
        }
    }
    best as u32
}

/// Reorder one layer's device KV cache into the drafter's `[position][kv*dim]`
/// layout. The sliding caches RING, so a position is not its own slot; `slots`
/// carries the mapping produced by the extractor.
fn build_kv_slice(
    keys: &[u16],
    values: &[u16],
    slots: &[usize],
    kv_heads: usize,
    head_dim: usize,
) -> KvSlice {
    let stream = kv_heads * head_dim;
    let len = slots.len();
    let mut k_out = vec![0u16; len * stream];
    let mut v_out = vec![0u16; len * stream];
    for (position, &slot) in slots.iter().enumerate() {
        let src = slot * stream;
        let dst = position * stream;
        k_out[dst..dst + stream].copy_from_slice(&keys[src..src + stream]);
        v_out[dst..dst + stream].copy_from_slice(&values[src..src + stream]);
    }
    KvSlice {
        keys: k_out,
        values: v_out,
        kv_heads,
        head_dim,
        len,
        positions: (0..len as u32).collect(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    if args.target.is_empty() || args.drafter.is_empty() {
        eprintln!("usage: mtp_acceptance --target <dir> --drafter <dir> [--rounds N --k K --temperature T --top-k 64 --seed S]");
        std::process::exit(2);
    }

    let device = Device::system_default().ok_or("no metal device")?;
    eprintln!("loading target {}...", args.target);
    let t0 = Instant::now();
    let mut target = TextModel::load_from_dir(std::path::Path::new(&args.target))?;
    eprintln!("target loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let dweights = DrafterWeights::load(std::path::Path::new(&args.drafter), &device)?;
    let drafter = DrafterModel::new(dweights)?;
    eprintln!("drafter loaded");

    let target_cfg = load_config(std::path::Path::new(&args.target).join("config.json").as_path())?;
    // Find the last sliding layer and the last full layer: the drafter reads one
    // shared K/V per layer type, which in mlx-vlm is the last layer of each type
    // to write into its cache.
    let mut last_sliding = 0usize;
    let mut last_full = 0usize;
    for (i, kind) in target_cfg.layer_types.iter().enumerate() {
        if kind.is_full() {
            last_full = i;
        } else {
            last_sliding = i;
        }
    }
    eprintln!("shared KV from sliding layer {last_sliding}, full layer {last_full}");

    let sampler_cfg = SamplerConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: 0.95,
    };
    let mut sampler = Sampler::new(sampler_cfg, args.seed)?;

    // Tokenize the prompt with the engine's tokenizer is more machinery than the
    // gate needs; use the known-safe single-token seed and grow context from
    // there. (A real prompt tokenizer hookup is T7's job.)
    // BOS for Gemma is 2.
    let mut context: Vec<u32> = vec![2];

    let mut accepted_hist: Vec<usize> = Vec::new();
    let mut rounds_done = 0usize;
    let start = Instant::now();

    // Round loop: draft K, verify token-by-token with the target (M=1).
    while rounds_done < args.rounds {
        let position = context.len();
        // 1) Target forward on the last token to get its hidden + logits.
        let last_token = *context.last().unwrap();
        let (hidden, logits) = target.forward_with_hidden(&[last_token], &[position as u32])?;
        let target_token = if args.temperature == 0.0 {
            greedy_token(&logits)
        } else {
            sample_token(&logits, &mut sampler)
        };

        // 2) Snapshot shared KV from the last sliding + full layers.
        let (s_keys, s_values, _, s_slots, s_heads, s_head_dim) =
            target.layer_kv_to_host(last_sliding)?;
        let (f_keys, f_values, _, f_slots, f_heads, f_head_dim) =
            target.layer_kv_to_host(last_full)?;
        let shared = DrafterSharedKv {
            sliding: build_kv_slice(&s_keys, &s_values, &s_slots, s_heads, s_head_dim),
            full: build_kv_slice(&f_keys, &f_values, &f_slots, f_heads, f_head_dim),
        };

        // 3) Drafter autoregressively drafts K tokens from the same position,
        //    feeding its own previous draft back each step (position held constant,
        //    per the reference draft_block).
        // 3) Drafter autoregressively drafts K tokens. Per the reference
        //    `draft_block`, `position_ids` is held CONSTANT at kv_offset
        //    (= position + 1, the next slot) across all K steps, and each step
        //    feeds its own previous draft back.
        let draft_position = position + 1;
        let mut tok_embed = target_embedding_scaled(&target, target_token);
        let mut prev_hidden = hidden.clone(); // the target's hidden that produced target_token
        let mut drafts = Vec::with_capacity(args.k);
        for _ in 0..args.k {
            let mut input = vec![0u16; tok_embed.len() + prev_hidden.len()];
            input[..tok_embed.len()].copy_from_slice(&tok_embed);
            input[tok_embed.len()..].copy_from_slice(&prev_hidden);
            let step = drafter.forward(&input, &shared, draft_position)?;
            prev_hidden = step.last_hidden;
            let logits = drafter_logits(&drafter, &step.hidden);
            let drafted = if args.temperature == 0.0 {
                greedy_token(&logits)
            } else {
                sample_token(&logits, &mut sampler)
            };
            tok_embed = target_embedding_scaled(&target, drafted);
            drafts.push(drafted);
        }

        // 4) Verify against the target's own continuation, one token per forward
        //    (M=1 — enough to measure the acceptance RATE, which does not depend
        //    on how the verify is batched).
        //
        //    G0 = target_token already predicted after `last_token` (position).
        //    draft[0] predicts the slot AFTER G0, so the first verification
        //    forward feeds G0 @ position+1 to get G1, and compares draft[0] to G1.
        //    Each acceptance then feeds the accepted token forward to learn the
        //    next target prediction. After accepting `a` drafts, `next_target`
        //    is the bonus/correction token.
        let mut accepted = 0usize;
        let (_, logit_row) =
            target.forward_with_hidden(&[target_token], &[(position + 1) as u32])?;
        let mut next_target = if args.temperature == 0.0 {
            greedy_token(&logit_row)
        } else {
            sample_token(&logit_row, &mut sampler)
        };
        for (i, &drafted) in drafts.iter().enumerate() {
            if drafted == next_target {
                accepted = i + 1;
                let pos = position + 1 + accepted;
                let (_, logit_row) = target.forward_with_hidden(&[drafted], &[pos as u32])?;
                next_target = if args.temperature == 0.0 {
                    greedy_token(&logit_row)
                } else {
                    sample_token(&logit_row, &mut sampler)
                };
            } else {
                break;
            }
        }

        accepted_hist.push(accepted);
        rounds_done += 1;

        // Commit: G0, the confirmed drafts, and the bonus/correction token.
        context.push(target_token);
        for i in 0..accepted {
            context.push(drafts[i]);
        }
        context.push(next_target);
        if context.len() > 1024 {
            context.drain(..context.len() - 512);
        }
    }

    let elapsed = start.elapsed();
    let total_accepted: usize = accepted_hist.iter().sum();
    let mean_accepted = total_accepted as f64 / accepted_hist.len().max(1) as f64;
    let target_pass_ms = 89.5; // measured single-token target GPU span, spec/decode-3bpw-amplification
    let achieved_tps = mean_accepted * 1000.0 / target_pass_ms;

    eprintln!("\n=== MTP acceptance gate (K={}, temperature={:.1}) ===", args.k, args.temperature);
    eprintln!("rounds: {}", accepted_hist.len());
    eprintln!("mean accepted drafts/round: {mean_accepted:.3}");
    eprintln!("accepted histogram: {:?}", accepted_hist);
    eprintln!("implied tok/s at {target_pass_ms:.0}ms target pass: {achieved_tps:.2}");
    eprintln!("40 tok/s needs >= 3.14 accepted/pass: {}", if mean_accepted >= 3.14 { "PASS" } else { "FAIL" });
    eprintln!("wall clock: {:.1}s", elapsed.as_secs_f64());
    Ok(())
}

/// Scaled target embedding for one token. The reference multiplies the target's
/// embedding by `embed_scale = sqrt(hidden_size)` before handing it to the
/// drafter (`gemma4_assistant.py` `draft_block`, `language.py:377`).
fn target_embedding_scaled(target: &TextModel, token: u32) -> Vec<u16> {
    let raw = target.embedding(&[token]);
    let scale = (target.hidden_size() as f32).sqrt();
    raw.into_iter()
        .map(|v| f16::from_f32(f16::from_bits(v).to_f32() * scale).to_bits())
        .collect()
}

/// Project the drafter's post-projection hidden through the target's embedding
/// to get draft logits (the drafter's lm_head IS the target's tied embed_tokens).
fn drafter_logits(drafter: &DrafterModel, hidden: &[u16]) -> Vec<u16> {
    drafter.logits(hidden).expect("drafter logits")
}