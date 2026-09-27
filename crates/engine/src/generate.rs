use std::time::{Duration, Instant};

use thiserror::Error;

use crate::metrics::{
    BenchmarkPlan, BenchmarkResult, MetricsError, RunMetrics, benchmark_result, measure_rss,
};
use crate::model::{ModelError, TextModel};
use crate::sampler::{Sampler, SamplerConfig, SamplerError};
use crate::tokenizer::{GemmaTokenizer, TokenizerError};

pub const DEFAULT_CONTEXT_LIMIT: usize = 8192;

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
