use std::time::{Duration, Instant};

use thiserror::Error;

use crate::metrics::{
    BenchmarkPlan, BenchmarkResult, MetricsError, RunMetrics, benchmark_result, measure_rss,
};
use kaplumba_kernels::encoder::MetalEncoder;
use crate::model::{ModelError, TextModel};
use crate::sampler::{Sampler, SamplerConfig, SamplerError};
use crate::tokenizer::{GemmaTokenizer, TokenizerError};

pub const DEFAULT_CONTEXT_LIMIT: usize = 8192;

/// Set `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` once, before the first command buffer.
///
/// llama.cpp does exactly this at process start
/// (`ggml/src/ggml-metal/ggml-metal.cpp:1025-1029`) as a documented workaround
/// for the AGX driver stalling on a very large command-buffer context store,
/// reported upstream as an interactivity hang (ggml-org/llama.cpp#20141). Our
/// prefill is the same shape of problem: hundreds of dispatches submitted back
/// to back with no host wait, so the driver can be asked to store a context far
/// larger than it is tuned for. It has to be set before the driver reads it, so
/// it is done at the earliest point in a generation, which is before any
/// `MetalEncoder` in this process has been constructed.
///
/// `overwrite = 1`, matching llama.cpp, so an operator can still override it.
fn ensure_agx_relax_ctxstore_timeout() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let name = std::ffi::CString::new("AGX_RELAX_CDM_CTXSTORE_TIMEOUT").expect("no NUL");
        let value = std::ffi::CString::new("1").expect("no NUL");
        // SAFETY: both pointers are NUL-terminated and outlive the call.
        let result = unsafe { libc::setenv(name.as_ptr(), value.as_ptr(), 1) };
        if result != 0 {
            eprintln!("kaplumba: could not set AGX_RELAX_CDM_CTXSTORE_TIMEOUT");
        }
    });
}

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub eos_ids: Vec<u32>,
    pub sampler: SamplerConfig,
    pub seed: u64,
    pub context_limit: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            eos_ids: vec![1, 106, 50],
            sampler: SamplerConfig::default(),
            seed: 0,
            context_limit: DEFAULT_CONTEXT_LIMIT,
        }
    }
}

#[derive(Debug, Error)]
pub enum GenerationError {
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    #[error(transparent)]
    Sampler(#[from] SamplerError),
    #[error("generation requires at least one new token")]
    ZeroNewTokens,
    #[error("context limit must be positive")]
    InvalidContextLimit,
    #[error("prompt plus generation exceeds context limit {limit}")]
    ContextExceeded { limit: usize },
    #[error("invalid generation input: {0}")]
    InvalidInput(String),
    #[error("output callback failed: {0}")]
    Output(String),
    #[error(transparent)]
    Metrics(#[from] MetricsError),
}

#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub generated_ids: Vec<u32>,
    pub stopped_on_eos: bool,
    pub ttft_ms: f64,
    pub prefill_tokens_per_second: f64,
    pub decode_tokens_per_second: f64,
}

pub fn generate<F>(
    model: &mut TextModel,
    tokenizer: &GemmaTokenizer,
    prompt: &str,
    config: &GenerationConfig,
    mut on_text: F,
) -> Result<GenerationResult, GenerationError>
where
    F: FnMut(&str) -> Result<(), String>,
{
    if config.max_new_tokens == 0 {
        return Err(GenerationError::ZeroNewTokens);
    }
    if config.context_limit == 0 {
        return Err(GenerationError::InvalidContextLimit);
    }
    ensure_agx_relax_ctxstore_timeout();
    let request_started = Instant::now();
    let prompt_ids = tokenizer.encode_prompt(prompt)?;
    let total_limit = prompt_ids.len().checked_add(config.max_new_tokens).ok_or(
        GenerationError::ContextExceeded {
            limit: config.context_limit,
        },
    )?;
    if total_limit > config.context_limit {
        return Err(GenerationError::ContextExceeded {
            limit: config.context_limit,
        });
    }
    // Greedy decode has its own loop because the token is the argmax rather than
    // a sample, and because it is timed from the prefill's pick. See
    // `generate_gpu_greedy`.
    if config.sampler.temperature == 0.0 {
        let mut on_text = on_text;
        return generate_gpu_greedy(
            model,
            tokenizer,
            &prompt_ids,
            config,
            &mut move |chunk| on_text(chunk).map_err(GenerationError::Output),
            request_started,
        );
    }

    let mut sampler = Sampler::new(config.sampler, config.seed)?;
    let mut detokenizer = crate::tokenizer::Detokenizer::new(tokenizer);
    let positions = (0..prompt_ids.len())
        .map(|position| position as u32)
        .collect::<Vec<_>>();
    let prefill_started = Instant::now();
    let mut logits = model.forward(&prompt_ids, &positions)?;
    let prefill_duration = prefill_started.elapsed();
    let mut decode_duration = Duration::ZERO;
    let mut ttft_duration = Duration::ZERO;
    let mut logits_offset = (prompt_ids.len() - 1) * model.vocab_size();
    let mut generated_ids = Vec::with_capacity(config.max_new_tokens);
    let mut text = String::new();
    let mut stopped_on_eos = false;
    for step in 0..config.max_new_tokens {
        let vocab_size = model.vocab_size();
        let last_logits = logits[logits_offset..logits_offset + vocab_size]
            .iter()
            .map(|value| half::f16::from_bits(*value).to_f32())
            .collect::<Vec<_>>();
        let token = sampler.sample(&last_logits)?;
        if generated_ids.is_empty() {
            ttft_duration = request_started.elapsed();
        }
        if config.eos_ids.contains(&token) {
            stopped_on_eos = true;
            break;
        }
        generated_ids.push(token);
        if let Some(chunk) = detokenizer.push(token)? {
            text.push_str(&chunk);
            on_text(&chunk).map_err(GenerationError::Output)?;
        }
        if step + 1 == config.max_new_tokens {
            break;
        }
        let position = prompt_ids.len() + generated_ids.len() - 1;
        let position = u32::try_from(position).map_err(|_| GenerationError::ContextExceeded {
            limit: config.context_limit,
        })?;
        let decode_started = Instant::now();
        logits = model.forward(&[token], &[position])?;
        decode_duration += decode_started.elapsed();
        logits_offset = 0;
    }
    if let Some(chunk) = detokenizer.finish()? {
        text.push_str(&chunk);
        on_text(&chunk).map_err(GenerationError::Output)?;
    }
    let prefill_tokens_per_second = if prefill_duration.is_zero() {
        0.0
    } else {
        prompt_ids.len() as f64 / prefill_duration.as_secs_f64()
    };
    let decode_steps = generated_ids.len().saturating_sub(1);
    let decode_tokens_per_second = if decode_duration.is_zero() {
        0.0
    } else {
        decode_steps as f64 / decode_duration.as_secs_f64()
    };
    Ok(GenerationResult {
        text,
        prompt_tokens: prompt_ids.len(),
        generated_tokens: generated_ids.len(),
        generated_ids,
        stopped_on_eos,
        ttft_ms: ttft_duration.as_secs_f64() * 1000.0,
        prefill_tokens_per_second,
        decode_tokens_per_second,
    })
}

/// Generate using the fully GPU-resident forward path.
///
/// Identical sampling and detokenisation to [`generate`], but every forward runs
/// as a single command buffer and only the final softcapped logits come back to
/// the host. The GPU path always yields just the sampled row, so there is no
/// logits offset to track.
pub fn generate_gpu<F>(
    model: &mut TextModel,
    tokenizer: &GemmaTokenizer,
    prompt: &str,
    config: &GenerationConfig,
    mut on_text: F,
) -> Result<GenerationResult, GenerationError>
where
    F: FnMut(&str) -> Result<(), String>,
{
    if config.max_new_tokens == 0 {
        return Err(GenerationError::ZeroNewTokens);
    }
    if config.context_limit == 0 {
        return Err(GenerationError::InvalidContextLimit);
    }
    ensure_agx_relax_ctxstore_timeout();
    let request_started = Instant::now();
    let prompt_ids = tokenizer.encode_prompt(prompt)?;
    let total_limit = prompt_ids.len().checked_add(config.max_new_tokens).ok_or(
        GenerationError::ContextExceeded {
            limit: config.context_limit,
        },
    )?;
    if total_limit > config.context_limit {
        return Err(GenerationError::ContextExceeded {
            limit: config.context_limit,
        });
    }
    model.reset_gpu_caches();
    let mut sampler = Sampler::new(config.sampler, config.seed)?;
    let mut detokenizer = crate::tokenizer::Detokenizer::new(tokenizer);
    let positions = (0..prompt_ids.len())
        .map(|position| position as u32)
        .collect::<Vec<_>>();
    let prefill_started = Instant::now();
    let mut logits = model.forward_buffer(&prompt_ids, &positions)?;
    let prefill_duration = prefill_started.elapsed();
    let mut decode_duration = Duration::ZERO;
    let mut ttft_duration = Duration::ZERO;
    let mut generated_ids = Vec::with_capacity(config.max_new_tokens);
    let mut text = String::new();
    let mut stopped_on_eos = false;
    let vocab_size = model.vocab_size();
    for step in 0..config.max_new_tokens {
        let last_logits = logits
            .iter()
            .map(|value| half::f16::from_bits(*value).to_f32())
            .collect::<Vec<_>>();
        let token = sampler.sample(&last_logits)?;
        if generated_ids.is_empty() {
            ttft_duration = request_started.elapsed();
        }
        if config.eos_ids.contains(&token) {
            stopped_on_eos = true;
            break;
        }
        generated_ids.push(token);
        if let Some(chunk) = detokenizer.push(token)? {
            text.push_str(&chunk);
            on_text(&chunk).map_err(GenerationError::Output)?;
        }
        if step + 1 == config.max_new_tokens {
            break;
        }
        let position = prompt_ids.len() + generated_ids.len() - 1;
        let position = u32::try_from(position).map_err(|_| GenerationError::ContextExceeded {
            limit: config.context_limit,
        })?;
        let decode_started = Instant::now();
        logits = model.forward_buffer(&[token], &[position])?;
        decode_duration += decode_started.elapsed();
    }
    if let Some(chunk) = detokenizer.finish()? {
        text.push_str(&chunk);
        on_text(&chunk).map_err(GenerationError::Output)?;
    }
    let prefill_tokens_per_second = if prefill_duration.is_zero() {
        0.0
    } else {
        prompt_ids.len() as f64 / prefill_duration.as_secs_f64()
    };
    let decode_steps = generated_ids.len().saturating_sub(1);
    let decode_tokens_per_second = if decode_duration.is_zero() {
        0.0
    } else {
        decode_steps as f64 / decode_duration.as_secs_f64()
    };
    let _ = vocab_size;
    Ok(GenerationResult {
        text,
        prompt_tokens: prompt_ids.len(),
        generated_tokens: generated_ids.len(),
        generated_ids,
        stopped_on_eos,
        ttft_ms: ttft_duration.as_secs_f64() * 1000.0,
        prefill_tokens_per_second,
        decode_tokens_per_second,
    })
}

pub fn run_benchmark(
    model: &mut TextModel,
    prompt_ids: &[u32],
    plan: BenchmarkPlan,
) -> Result<BenchmarkResult, GenerationError> {
    plan.validate()?;
    ensure_agx_relax_ctxstore_timeout();
    if prompt_ids.is_empty() || prompt_ids.len() != plan.prompt_tokens {
        return Err(GenerationError::InvalidInput(
            "benchmark prompt length does not match the plan".to_string(),
        ));
    }
    model.reset_caches()?;
    let positions = (0..prompt_ids.len())
        .map(|position| position as u32)
        .collect::<Vec<_>>();
    let prefill_started = Instant::now();
    let mut logits = model.forward(prompt_ids, &positions)?;
    let prefill = prefill_started.elapsed();
    let mut logits_offset = (prompt_ids.len() - 1) * model.vocab_size();
    let mut decode = Duration::ZERO;
    let steps = plan.warmup_steps + plan.measured_tokens;
    for step in 0..steps {
        let token = argmax_logits(&logits[logits_offset..logits_offset + model.vocab_size()]);
        if step + 1 == steps {
            break;
        }
        let position = prompt_ids.len() + step;
        let position = u32::try_from(position).map_err(|_| GenerationError::ContextExceeded {
            limit: plan.prompt_tokens + steps,
        })?;
        let decode_started = Instant::now();
        logits = model.forward(&[token], &[position])?;
        if step >= plan.warmup_steps {
            decode += decode_started.elapsed();
        }
        logits_offset = 0;
    }
    let metrics = RunMetrics {
        prefill,
        decode,
        prefill_tokens: prompt_ids.len() as u64,
        decode_tokens: plan.measured_tokens as u64,
        peak_rss_bytes: measure_rss()?,
        ..RunMetrics::default()
    };
    Ok(benchmark_result(metrics, plan)?)
}

fn argmax_logits(logits: &[u16]) -> u32 {
    let mut best = 0;
    for index in 1..logits.len() {
        if half::f16::from_bits(logits[index]).to_f32()
            > half::f16::from_bits(logits[best]).to_f32()
        {
            best = index;
        }
    }
    best as u32
}

/// Greedy decode over the fully GPU-resident forward path.
///
/// One step per barrier: the argmax is taken on the host off the single softcapped
/// logits row the forward returns, exactly as [`generate_gpu`] does, and the
/// sampled token goes back into the next step's `token_ids` on the host. That is
/// the only correct shape available, and the reasons the previous shape was not
/// are worth keeping, because every one of them looks like a scheduling bug from
/// the outside and none of them is:
///
/// *The device-side token hand-off cannot be expressed.* Step N+1's embedding
/// reads `tokens_data[0]` (`crates/kernels/src/elements.metal`), so step N's
/// pick would have to write `scratch.token_ids[0]` to keep the host out of the
/// loop. The only host-visible entry point for that write,
/// `ElementKernels::encode_greedy_pick`, refuses a destination index of 0. The
/// loop queued its steps with `token_ids = [0]`, so every step after the first
/// was fed token 0.
///
/// *The device-side pick is wrong for this vocabulary anyway.* `greedy_key`
/// packs `0xFFFF - index` into 16 bits, which underflows at and past index
/// 65,536 — most of a 262,144-entry vocabulary — so the packed key is no longer
/// ordered and the "winner" is an arbitrary low token id. Measured on the real
/// checkpoint: the host argmax over the same buffer is index 9,259 at 28.5, the
/// device reduction returns 0xFFFFFFFF, i.e. index 0.
///
/// *The overlapping encodes raced.* Steps were encoded back to back and waited
/// once, but `forward_buffer_into_step` stages `token_ids[0]` and
/// `query_positions[0]` into the shared scratch *from the host*, so those writes
/// landed while the previous step's kernels could still be reading them, and
/// which way that landed depended on how far the GPU got while the host encoded.
///
/// *The bookkeeping was off by one.* Each step overwrote log slot `step`, so the
/// prefill's own pick was clobbered by the first step and the loop read `depth`
/// tokens for a batch that had produced `depth + 1`.
///
/// Verified consequence: `--legacy-path --show-ids` printed 24 zeros for every
/// reference prompt, while the commit that introduced this loop claimed "correct
/// on all reference prompts, suite green" — no test reaches it, because
/// `SamplerConfig::default()` has `temperature: 1.0` and only the `--temperature 0`
/// CLI flag enters it.
///
/// # What the metrics mean
///
/// `ttft` is stamped when the first token is sampled, which is right after the
/// prefill's wait: no decode step is included, so it does not move with
/// `max_new_tokens`. The previous version stamped it after the first *batch's*
/// wait, so it carried up to 7 decode steps of extra GPU time (~1.9 s measured at
/// 614 tokens of context) and grew with `max_new_tokens`. `decode_duration` is
/// the sum of the per-step waits and `decode_tokens_per_second` divides the
/// generated-minus-one steps by it; the previous version divided steps it never
/// ran by the wall time of a whole batch, which is where `decode_tok_s=0.24` at
/// `--max-new-tokens 2` came from.
fn generate_gpu_greedy<F>(
    model: &mut TextModel,
    tokenizer: &crate::tokenizer::GemmaTokenizer,
    prompt_ids: &[u32],
    config: &GenerationConfig,
    on_text: &mut F,
    request_started: Instant,
) -> Result<GenerationResult, GenerationError>
where
    F: FnMut(&str) -> Result<(), GenerationError> + ?Sized,
{
    let mut detokenizer = crate::tokenizer::Detokenizer::new(tokenizer);
    let mut text = String::new();
    let mut generated_ids: Vec<u32> = Vec::with_capacity(config.max_new_tokens);
    let mut ttft_duration = Duration::ZERO;
    let mut decode_duration = Duration::ZERO;
    let mut prefill_duration = Duration::ZERO;
    let mut stopped_on_eos = false;
    let prompt_len = prompt_ids.len();
    let queue = model.forward_queue();

    let result = (|| -> Result<(), GenerationError> {
        let prefill_started = Instant::now();
        let positions: Vec<u32> = (0..prompt_len as u32).collect();
        let mut prefill = MetalEncoder::new(&queue).map_err(ModelError::from)?;
        model
            .forward_buffer_into(prompt_ids, &positions, &mut prefill)
            .map_err(ModelError::from)?;
        // Commit the prefill. Encoding is not execution: without this the buffer
        // is dropped un-ended, the KV cache is never populated, and the decode
        // steps that follow run against an empty cache. It also makes
        // `prefill_duration` honest -- it now covers the GPU, not just the host.
        prefill.commit_and_wait().map_err(ModelError::from)?;
        prefill_duration = prefill_started.elapsed();

        // Read after the wait: the prefill's last row is the first generated
        // token's logits, and reading it earlier returns the previous row.
        let mut logits = model.last_logits();
        loop {
            let token = argmax_logits(&logits);
            if generated_ids.is_empty() {
                ttft_duration = request_started.elapsed();
            }
            if config.eos_ids.contains(&token) {
                stopped_on_eos = true;
                break;
            }
            generated_ids.push(token);
            if let Some(chunk) = detokenizer.push(token)? {
                text.push_str(&chunk);
                on_text(&chunk)?;
            }
            if generated_ids.len() == config.max_new_tokens {
                break;
            }
            let position = prompt_len + generated_ids.len() - 1;
            let position = u32::try_from(position).map_err(|_| GenerationError::ContextExceeded {
                limit: config.context_limit,
            })?;
            let step_started = Instant::now();
            let mut encoder = MetalEncoder::new(&queue).map_err(ModelError::from)?;
            model
                .forward_buffer_into_step(&mut encoder, position, token)
                .map_err(ModelError::from)?;
            // `commit_terminal` ends the step's trailing partial slice of rolled
            // command buffers; `wait` covers all of them, in commit order, which
            // is the barrier the next step's host-side staging needs.
            encoder.commit_terminal().map_err(ModelError::from)?;
            encoder.wait();
            decode_duration += step_started.elapsed();
            logits = model.last_logits();
        }

        if let Some(chunk) = detokenizer.finish()? {
            text.push_str(&chunk);
            on_text(&chunk)?;
        }
        Ok(())
    })();
    result?;

    let decode_steps = generated_ids.len().saturating_sub(1);
    Ok(GenerationResult {
        text,
        prompt_tokens: prompt_len,
        generated_tokens: generated_ids.len(),
        generated_ids,
        stopped_on_eos,
        ttft_ms: ttft_duration.as_secs_f64() * 1e3,
        prefill_tokens_per_second: if prefill_duration.is_zero() {
            0.0
        } else {
            prompt_len as f64 / prefill_duration.as_secs_f64()
        },
        decode_tokens_per_second: if decode_duration.is_zero() {
            0.0
        } else {
            decode_steps as f64 / decode_duration.as_secs_f64()
        },
    })
}

#[cfg(test)]
mod tests {
    use half::f16;

    use super::argmax_logits;

    fn row(values: &[f32]) -> Vec<u16> {
        values.iter().map(|v| f16::from_f32(*v).to_bits()).collect()
    }

    /// Greedy decode is exactly this function applied to one row of softcapped
    /// logits, so it is the whole of the sampling decision.
    ///
    /// This test exists because the greedy path had *no* coverage at all: with
    /// `argmax_logits` sabotaged to `0u32` the engine's 20 suites / 81 tests
    /// still passed, since `SamplerConfig::default()` has `temperature: 1.0` and
    /// only `--temperature 0` reaches greedy decode, and every test that loads the
    /// real checkpoint is `#[ignore]`d. A fix that fed the model token 0 on every
    /// step was indistinguishable from a correct one to `cargo test`.
    #[test]
    fn argmax_picks_the_largest_logit() {
        assert_eq!(argmax_logits(&row(&[-5.0, 3.0, -1.0, 12.5, 2.0])), 3);
        assert_eq!(argmax_logits(&row(&[1.0, 2.0])), 1);
    }

    /// A tie must resolve to the lowest index, and an all-negative row must not
    /// fall through to index 0 by accident: both are properties of the order the
    /// tokens are scanned in, and both are what makes the pick reproducible.
    #[test]
    fn argmax_breaks_ties_toward_the_lowest_index() {
        assert_eq!(argmax_logits(&row(&[7.0, 7.0, 7.0])), 0);
        assert_eq!(argmax_logits(&row(&[0.0, 0.0])), 0);
        assert_eq!(argmax_logits(&row(&[-4.0, -9.0, -4.0, -1.0])), 3);
        // Every logit negative, and the first is the largest.
        assert_eq!(argmax_logits(&row(&[-0.5, -30.0, -2.0])), 0);
    }

    /// One-element and fp16-precision boundaries: the argmax runs over a `u16`
    /// row, so values that are equal in `f32` can differ in fp16 and vice versa.
    #[test]
    fn argmax_handles_single_element_and_fp16_precision() {
        assert_eq!(argmax_logits(&row(&[42.0])), 0);
        // Two distinct f32 values that round to the same fp16 must tie, and the
        // scan order then decides.
        let a = f16::from_f32(1.0000001);
        let b = f16::from_f32(1.0);
        assert_eq!(a.to_bits(), b.to_bits());
        assert_eq!(argmax_logits(&[a.to_bits(), b.to_bits()]), 0);
        // The softcap's 30.0 ceiling is exactly representable and is the largest
        // value the head can produce.
        assert_eq!(argmax_logits(&row(&[30.0, 29.999, 30.0])), 0);
    }
}
