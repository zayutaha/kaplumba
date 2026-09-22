//! Streaming generation loop (T1.4): prefill → sample → emit, with the stats the
//! bench report needs (R7.1/R7.4).
//!
//! The naive path has **no KV cache**: every step recomputes the whole prefix, so
//! `computed_tokens` grows quadratically and `tg_tps` is not comparable with a
//! KV-cached engine. Reported honestly rather than hidden; T4.2/T5.1 fix it.

use std::time::Instant;

use crate::gemma4::{Gemma4Model, GemmaError};
use crate::sampler::{Pcg32, SampleParams, sample};

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// An EOS/stop id was produced (it is included in `tokens`).
    Eos,
    /// `max_new` tokens were produced.
    Length,
    /// The caller's callback returned `false`.
    Stopped,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Eos => "eos",
            FinishReason::Length => "length",
            FinishReason::Stopped => "stopped",
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenParams {
    pub max_new: usize,
    pub sample: SampleParams,
    pub seed: u64,
    /// Stop ids; the matching token is emitted and generation ends.
    pub eos: Vec<u32>,
    /// Depth limit for debugging/benching (only the first N decoder layers).
    pub layers: Option<usize>,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            max_new: 128,
            sample: SampleParams::default(),
            seed: 0,
            // Gemma 4: <eos> = 1, <turn|> = 106 (turn boundary).
            eos: vec![1, 106],
            layers: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenStats {
    pub prompt_tokens: usize,
    pub gen_tokens: usize,
    /// Time of the first forward (the whole prompt, since there is no KV cache).
    pub prefill_s: f64,
    /// Summed time of the forwards after the first (one per generated token).
    pub decode_s: f64,
    pub decode_steps: usize,
    /// Time from request start to the **first emitted token**.
    pub ttft_ms: f64,
    pub finish_reason: FinishReason,
    pub tokens: Vec<u32>,
    /// Prompt tokens actually pushed through the model (naive path recomputes the
    /// prefix each step, so this is much larger than `prompt_tokens`).
    pub computed_tokens: usize,
}

impl GenStats {
    pub fn pp_tps(&self) -> f64 {
        if self.prefill_s > 0.0 {
            self.prompt_tokens as f64 / self.prefill_s
        } else {
            0.0
        }
    }

    /// Tokens generated *after* the prefill step, per second of decode time.
    pub fn tg_tps(&self) -> f64 {
        if self.decode_s > 0.0 {
            self.decode_steps as f64 / self.decode_s
        } else {
            0.0
        }
    }

    /// Total model passes over prompt+generated tokens.
    pub fn redone_fraction(&self) -> f64 {
        if self.computed_tokens == 0 {
            0.0
        } else {
            self.computed_tokens as f64 / (self.prompt_tokens + self.gen_tokens).max(1) as f64
        }
    }
}

/// Generate from `prompt`, calling `on_token` for every emitted token (including
/// the EOS token). Returning `false` from the callback stops generation.
pub fn generate(
    model: &Gemma4Model<'_>,
    prompt: &[u32],
    params: &GenParams,
    mut on_token: impl FnMut(u32) -> bool,
) -> Result<GenStats, GemmaError> {
    if prompt.is_empty() {
        return Err(GemmaError::Config("empty prompt".into()));
    }
    let hidden = model.cfg.hidden;
    let mut toks: Vec<u32> = prompt.to_vec();
    let mut rng = Pcg32::new(params.seed);
    let mut tokens: Vec<u32> = Vec::with_capacity(params.max_new);
    let mut prefill_s = 0.0f64;
    let mut decode_s = 0.0f64;
    let mut decode_steps = 0usize;
    let mut computed_tokens = 0usize;
    let mut ttft_ms = 0.0f64;
    let mut finish = FinishReason::Length;
    let t_start = Instant::now();

    for step in 0..params.max_new {
        computed_tokens += toks.len();
        let t0 = Instant::now();
        let h = match params.layers {
            Some(n) => model.forward_seq_layers(&toks, n)?,
            None => model.forward_seq(&toks)?,
        };
        let last = &h[(toks.len() - 1) * hidden..toks.len() * hidden];
        let logits = model.logits(last)?;
        let dt = t0.elapsed().as_secs_f64();
        if step == 0 {
            prefill_s = dt;
        } else {
            decode_s += dt;
            decode_steps += 1;
        }

        let next = sample(&logits, &params.sample, &mut rng);
        if step == 0 {
            ttft_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        }
        tokens.push(next);

        let is_eos = params.eos.contains(&next);
        let keep_going = on_token(next);
        if is_eos {
            finish = FinishReason::Eos;
            break;
        }
        if !keep_going {
            finish = FinishReason::Stopped;
            break;
        }
        toks.push(next);
    }

    Ok(GenStats {
        prompt_tokens: prompt.len(),
        gen_tokens: tokens.len(),
        prefill_s,
        decode_s,
        decode_steps,
        ttft_ms,
        finish_reason: finish,
        tokens,
        computed_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemma4::{Gemma4Model, GemmaError};
    use crate::testutil::synth_model;

    fn greedy_params(max_new: usize) -> GenParams {
        GenParams {
            max_new,
            sample: SampleParams::greedy(),
            seed: 0,
            eos: vec![1],
            layers: None,
        }
    }

    #[test]
    fn greedy_matches_greedy_generate_and_reports_stats() {
        let (src, cfg) = synth_model(Some(30.0));
        let model = Gemma4Model::new(cfg, &src);
        let prompt = [2u32, 17, 5, 300, 42];
        let params = greedy_params(3);

        let mut streamed = Vec::new();
        let stats = generate(&model, &prompt, &params, |t| {
            streamed.push(t);
            true
        })
        .expect("generate");

        let direct = model.greedy_generate(&prompt, 3, &[1]).expect("greedy");
        assert_eq!(stats.tokens, direct, "stream equals greedy_generate");
        assert_eq!(streamed, direct, "callback saw every emitted token");
        assert_eq!(stats.gen_tokens, stats.tokens.len());
        assert_eq!(stats.prompt_tokens, prompt.len());
        assert!(stats.prefill_s > 0.0 && stats.decode_s > 0.0);
        assert_eq!(stats.decode_steps, stats.tokens.len() - 1);
        assert!(stats.ttft_ms > 0.0);
        assert!(stats.pp_tps() > 0.0 && stats.tg_tps() > 0.0);
        // Naive path recomputes the prefix: 5 + 6 + 7 tokens for 3 steps.
        assert_eq!(stats.computed_tokens, 5 + 6 + 7);
        assert!(stats.redone_fraction() > 1.0);
        assert!(stats.tokens.iter().all(|t| (*t as usize) < model.cfg.vocab));
    }

    #[test]
    fn eos_stops_immediately_and_is_included() {
        let (src, cfg) = synth_model(Some(30.0));
        let model = Gemma4Model::new(cfg, &src);
        let prompt = [2u32, 9];
        let first = model.greedy_generate(&prompt, 1, &[]).expect("probe")[0];

        let params = GenParams {
            max_new: 8,
            sample: SampleParams::greedy(),
            seed: 0,
            eos: vec![first],
            layers: None,
        };
        let stats = generate(&model, &prompt, &params, |_| true).expect("generate");
        assert_eq!(stats.tokens, vec![first], "EOS token is emitted then stops");
        assert_eq!(stats.finish_reason, FinishReason::Eos);
        assert_eq!(stats.finish_reason.as_str(), "eos");
        assert_eq!(stats.decode_steps, 0, "no decode step needed");
        assert_eq!(stats.tg_tps(), 0.0, "no decode time → no tg rate");
    }

    #[test]
    fn callback_false_stops_with_stopped_reason() {
        let (src, cfg) = synth_model(Some(30.0));
        let model = Gemma4Model::new(cfg, &src);
        let prompt = [2u32, 9];
        let mut seen = 0;
        let stats = generate(&model, &prompt, &greedy_params(5), |_| {
            seen += 1;
            false
        })
        .expect("generate");
        assert_eq!(seen, 1);
        assert_eq!(stats.tokens.len(), 1);
        assert_eq!(stats.finish_reason, FinishReason::Stopped);
        assert_eq!(stats.gen_tokens, 1);
        assert_eq!(stats.decode_steps, 0);
    }

    #[test]
    fn zero_max_new_is_a_no_op() {
        let (src, cfg) = synth_model(Some(30.0));
        let model = Gemma4Model::new(cfg, &src);
        let stats = generate(&model, &[2u32], &greedy_params(0), |_| true).expect("generate");
        assert!(stats.tokens.is_empty());
        assert_eq!(stats.finish_reason, FinishReason::Length);
        assert_eq!(stats.gen_tokens, 0);
        assert_eq!(stats.computed_tokens, 0);
        assert_eq!(stats.ttft_ms, 0.0);
        assert_eq!(stats.pp_tps(), 0.0, "no prefill ran");
        assert_eq!(stats.redone_fraction(), 0.0);
    }

    #[test]
    fn length_reason_when_max_new_is_reached() {
        let (src, cfg) = synth_model(Some(30.0));
        let model = Gemma4Model::new(cfg, &src);
        // EOS list empty → cannot stop early.
        let params = GenParams {
            max_new: 3,
            sample: SampleParams::greedy(),
            seed: 0,
            eos: vec![],
            layers: None,
        };
        let stats = generate(&model, &[2u32, 4], &params, |_| true).expect("generate");
        assert_eq!(stats.tokens.len(), 3);
        assert_eq!(stats.finish_reason, FinishReason::Length);
    }

    #[test]
    fn sampling_is_seed_reproducible() {
        let (src, cfg) = synth_model(Some(30.0));
        let model = Gemma4Model::new(cfg, &src);
        let prompt = [2u32, 17, 5];
        let base = GenParams {
            max_new: 8,
            sample: SampleParams::default(),
            seed: 7,
            eos: vec![],
            layers: None,
        };
        let a = generate(&model, &prompt, &base, |_| true).expect("a");
        let b = generate(&model, &prompt, &base, |_| true).expect("b");
        assert_eq!(a.tokens, b.tokens, "same seed → same stream");

        let other = GenParams {
            seed: 8,
            ..base.clone()
        };
        let c = generate(&model, &prompt, &other, |_| true).expect("c");
        assert_ne!(a.tokens, c.tokens, "different seed → different stream");
        assert!(
            a.tokens.iter().all(|t| (*t as usize) < model.cfg.vocab),
            "all sampled ids in vocab"
        );
    }

    #[test]
    fn depth_limit_shortens_the_forward_but_keeps_plumbing() {
        let (src, cfg) = synth_model(Some(30.0));
        let model = Gemma4Model::new(cfg, &src);
        let params = GenParams {
            layers: Some(0),
            ..greedy_params(2)
        };
        let stats = generate(&model, &[2u32, 5], &params, |_| true).expect("generate");
        assert_eq!(stats.tokens.len(), 2);
        assert!(stats.tokens.iter().all(|t| (*t as usize) < model.cfg.vocab));
        let full = generate(&model, &[2u32, 5], &greedy_params(2), |_| true).expect("full");
        assert_ne!(
            stats.tokens, full.tokens,
            "depth 0 must not accidentally equal the full model"
        );
    }

    #[test]
    fn empty_prompt_is_rejected() {
        let (src, cfg) = synth_model(Some(30.0));
        let model = Gemma4Model::new(cfg, &src);
        let err = generate(&model, &[], &greedy_params(1), |_| true).unwrap_err();
        assert!(matches!(err, GemmaError::Config(_)), "got {err}");
    }
}
