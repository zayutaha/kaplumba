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

    // A real prompt. A BOS-only context makes the target loop on one token, and
    // a loop measures acceptance of ~1.0 for free — the gate must be measured on
    // text the model actually wants to continue.
    let tokenizer = kaplumba_engine::tokenizer::GemmaTokenizer::from_model_dir(std::path::Path::new(
        &args.target,
    ))?;
    let mut context: Vec<u32> = tokenizer.encode_prompt(&args.prompt)?;
    eprintln!("prompt tokenized to {} tokens", context.len());

    let mut accepted_hist: Vec<usize> = Vec::new();
    let mut rounds_done = 0usize;
    let (mut depth1_measured, mut depth1_exact, mut depth1_top5, mut depth1_top20) =
        (0usize, 0usize, 0usize, 0usize);
    let start = Instant::now();

    // Prefill the whole prompt in ONE M>1 forward so the target's KV cache and
    // its next-token prediction are real. Without this the context is a single
    // BOS token, the target loops on one id, and acceptance reads ~1.0 for free
    // — a gate measured on a degenerate loop is not a gate.
    let prefill_len = context.len();
    let positions: Vec<u32> = (0..prefill_len as u32).collect();
    let (_, prefill_logits) = target.forward_with_hidden(&context, &positions)?;
    eprintln!(
        "prefilled {prefill_len} tokens; cache primed; gpu_next_position={}",
        target.gpu_next_position()
    );

    // Invariant for the rest of the program:
    //
    //   * `context` = every COMMITTED token plus exactly one PENDING token,
    //     which is `context.last()`;
    //   * the target's KV cache holds `context[..len-1]` — the pending token is
    //     predicted but not yet fed.
    //
    // So the pending token is fed at slot `context.len() - 1` at the top of each
    // round, which is also where the cache expects the next write. Getting this
    // off by one either trips the cache's monotonic-position check or, worse,
    // silently double-counts a token.
    context.push(if args.temperature == 0.0 {
        greedy_token(&prefill_logits)
    } else {
        sample_token(&prefill_logits, &mut sampler)
    });

    // Round loop: draft K, verify token-by-token with the target (M=1).
    while rounds_done < args.rounds {
        let position = context.len() - 1;
        let last_token = *context.last().unwrap();
        if std::env::var_os("MTP_VERBOSE").is_some() {
            eprintln!(
                "round {rounds_done}: feeding {last_token} at pos={position} (cache at {})",
                target.gpu_next_position()
            );
        }
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

        // 3) Draft K tokens autoregressively.
        //
        //    Two details are load-bearing and both were wrong in the first
        //    version of this harness, which measured a meaningless 0.675/round:
        //
        //    * Position. `_mtp_draft_position` is `kv_valid_len - 1`
        //      (`mtp.py:714-719`), i.e. the drafter queries at the slot of the
        //      LAST token in the shared KV, not the slot after it. The KV holds
        //      `position + 1` tokens here, so the drafter's position is
        //      `position`.
        //    * Conditioning. `draft_block(last_bonus, hidden, ...)` pairs
        //      `embed(last_bonus)` with the hidden state OF THAT SAME TOKEN
        //      (`gemma4_assistant.py:224-246`). So the first drafter input is
        //      `[embed(last_token), hidden(last_token)]`, and its output predicts
        //      the slot AFTER `last_token` — the same slot `target_token`
        //      predicts. That makes draft[0] directly comparable to
        //      `target_token` with no extra forward.
        let draft_position = position;
        let mut tok_embed = target_embedding_scaled(&target, last_token);
        let mut prev_hidden = hidden.clone(); // the target's hidden for `last_token`
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

        // 4) Verify against the target's own continuation, one forward per
        //    accepted token (M=1). The acceptance RATE does not depend on how
        //    the verify is batched, so this measures the gate without T8a.
        //
        //    `expected` is always "the target's prediction for the slot that
        //    draft[i] proposes". For i=0 that slot is the one `target_token`
        //    already predicts; afterwards it is `bonus`, learned by feeding the
        //    just-accepted token forward.
        let mut accepted = 0usize;
        let mut expected = target_token;
        // Is draft[0] merely losing an argmax tie, or is the drafter nowhere near?
        // Top-k containment of the drafter's first proposal against the target's
        // row separates "drafter understands this context" from "drafter is not
        // conditioned on this target at all" — the difference between "too weak
        // for 40 tok/s" and "this harness is still wrong".
        let target_top20: Vec<usize> = top_k_indices(&logits, 20);
        let target_top5: Vec<usize> = target_top20[..5.min(target_top20.len())].to_vec();
        for (i, &drafted) in drafts.iter().enumerate() {
            if drafted != expected {
                break;
            }
            accepted = i + 1;
            let pos = position + accepted;
            if std::env::var_os("MTP_VERBOSE").is_some() {
                eprintln!(
                    "  verify[{i}] feeding {drafted} at pos={pos} (cache at {})",
                    target.gpu_next_position()
                );
            }
            let (_, logit_row) = target.forward_with_hidden(&[drafted], &[pos as u32])?;
            expected = if args.temperature == 0.0 {
                greedy_token(&logit_row)
            } else {
                sample_token(&logit_row, &mut sampler)
            };
        }
        let next_target = expected;

        accepted_hist.push(accepted);
        rounds_done += 1;
        if let Some(&first_draft) = drafts.first() {
            depth1_measured += 1;
            if first_draft == target_token {
                depth1_exact += 1;
            }
            if target_top5.contains(&(first_draft as usize)) {
                depth1_top5 += 1;
            }
            if target_top20.contains(&(first_draft as usize)) {
                depth1_top20 += 1;
            }
        }

        // Diagnostics: a 0% acceptance that is really a wiring bug looks exactly
        // like a 0% acceptance that is a real result, so print both token streams
        // until the gate has been shown to be trustworthy.
        if std::env::var_os("MTP_VERBOSE").is_some() {
            let h0 = f16::from_bits(hidden[0]).to_f32();
            let any_nan = hidden.iter().any(|v| f16::from_bits(*v).to_f32().is_nan());
            let kv_s_nan = shared.sliding.keys.iter().any(|v| f16::from_bits(*v).to_f32().is_nan());
            let kv_f_nan = shared.full.keys.iter().any(|v| f16::from_bits(*v).to_f32().is_nan());
            eprintln!("[round {rounds_done}] pos={position} target_hidden[0]={h0} nan_in_hidden={any_nan} nan_in_sliding_kv={kv_s_nan} nan_in_full_kv={kv_f_nan} sliding_len={} full_len={}", shared.sliding.len, shared.full.len);
            eprintln!("  target G0 = {target_token}");
            eprintln!("  drafts    = {drafts:?}");
            eprintln!("  accepted  = {accepted}");
        }

        // Commit.
        //
        //   * `accepted == 0`: no verification forward ran, so the cache still
        //     covers 0..=position and the pending token IS `target_token`. Pushing
        //     it as both a committed and a pending token double-counted it and
        //     desynchronised `context` from the cache by one slot — which is what
        //     produced "positions must begin at the current cache position" on
        //     round 1.
        //   * `accepted > 0`: the cache covers 0..=position+accepted, holding
        //     `target_token` (== draft[0]) then draft[1..accepted].
        if accepted > 0 {
            context.push(target_token);
            for i in 1..accepted {
                context.push(drafts[i]);
            }
        }
        context.push(next_target);
        // Trimming `context` would desynchronise it from the KV cache (the cache
        // has no "roll back" that a `drain` implies), so stop instead of
        // pretending. A long run is a `--rounds` job, not a trimming one.
        if context.len() >= 900 {
            eprintln!("context reached {} tokens; stopping measurement", context.len());
            break;
        }
    }

    let elapsed = start.elapsed();
    let total_accepted: usize = accepted_hist.iter().sum();
    let rounds = accepted_hist.len().max(1) as f64;
    let mean_accepted = total_accepted as f64 / rounds;
    // P(the i-th draft is reached AND accepted). Depth-1 is the health check on
    // the drafter itself: a correctly conditioned drafter accepts its FIRST
    // proposal far more often than not. A low depth-1 rate would mean the wiring
    // is wrong rather than that the drafter is weak, which is the difference
    // between "MTP is too slow" and "this harness is broken".
    let mut depth_hits = vec![0usize; args.k];
    for &a in &accepted_hist {
        for slot in depth_hits.iter_mut().take(a) {
            *slot += 1;
        }
    }
    let target_pass_ms = 89.5; // measured single-token target GPU span
    // Each round emits `accepted + 1` tokens: the confirmed drafts PLUS the
    // target's own next token, which the verify pass yields for free because it
    // already streamed every weight to produce the logits the drafts were
    // checked against.
    //
    // The first version of this harness omitted that +1 and reported 9.87 tok/s
    // for a configuration that sustains ~21. That 2.1x error is why speculation
    // was reported as worthless when it is in fact close to a 2x win on this
    // box, and it is the reason `spec/mtp-40tps` T4 was written up as "stop".
    let tokens_per_pass = mean_accepted + 1.0;
    let achieved_tps = tokens_per_pass * 1000.0 / target_pass_ms;

    eprintln!("\n=== MTP acceptance gate (K={}, temperature={:.1}) ===", args.k, args.temperature);
    eprintln!("rounds: {}", accepted_hist.len());
    eprintln!("mean accepted drafts/round: {mean_accepted:.3}");
    eprintln!("accepted histogram: {accepted_hist:?}");
    eprintln!(
        "P(accept at depth d): [{}]",
        depth_hits
            .iter()
            .map(|h| format!("{:.3}", *h as f64 / rounds))
            .collect::<Vec<_>>()
            .join(", ")
    );
    if depth1_measured > 0 {
        let n = depth1_measured as f64;
        eprintln!(
            "drafter draft[0] vs target: exact {}/{} ({:.3}), in top-5 {:.3}, in top-20 {:.3}",
            depth1_exact,
            depth1_measured,
            depth1_exact as f64 / n,
            depth1_top5 as f64 / n,
            depth1_top20 as f64 / n,
        );
    }
    eprintln!(
        "tokens per pass = accepted + 1 bonus = {tokens_per_pass:.3}"
    );
    eprintln!("implied tok/s at {target_pass_ms:.0}ms target pass: {achieved_tps:.2}");
    eprintln!(
        "40 tok/s needs tokens_per_pass >= {:.2} (i.e. accepted >= {:.2})",
        40.0 * target_pass_ms / 1000.0,
        40.0 * target_pass_ms / 1000.0 - 1.0
    );
    eprintln!("wall clock: {:.1}s", elapsed.as_secs_f64());
    Ok(())
}

/// Scaled target embedding for one token.
///
/// `TextModel::embedding` already returns `embed_tokens[token] * sqrt(hidden)`,
/// which is exactly what the reference multiplies by `embed_scale`
/// (`gemma4_assistant.py` `draft_block`, `language.py:377`), so this is now a
/// straight pass-through. It previously re-interpreted raw BF16 checkpoint bits
/// as FP16, which produced `-inf`/`NaN` and a meaningless 0% acceptance rate.
fn target_embedding_scaled(target: &TextModel, token: u32) -> Vec<u16> {
    target.embedding(&[token])
}

/// Top-k token indices of an f16 logit row, descending by value.
fn top_k_indices(logits: &[u16], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        f16::from_bits(logits[b])
            .to_f32()
            .partial_cmp(&f16::from_bits(logits[a]).to_f32())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order.truncate(k);
    order
}

/// Project the drafter's post-projection hidden through the target's embedding
/// to get draft logits (the drafter's lm_head IS the target's tied embed_tokens).
fn drafter_logits(drafter: &DrafterModel, hidden: &[u16]) -> Vec<u16> {
    drafter.logits(hidden).expect("drafter logits")
}