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
    // Greedy decode pipelines: the pick runs on the device so steps queue back to
    // back. See `generate_gpu_greedy_pipelined`.
    if config.sampler.temperature == 0.0 {
        let mut on_text = on_text;
        return generate_gpu_greedy_pipelined(
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

/// Greedy decode with the pick on the GPU and the wait batched.
///
/// Steps are encoded back to back, each into its own `MetalEncoder`. The greedy
/// pick runs as a kernel and leaves the sampled token in device memory, so step
/// N+1's embedding kernel reads it without any host involvement. The host then
/// waits once per `PIPELINE_DEPTH` steps and reads back that many token ids
/// together, which keeps latency bounded while the GPU stays fed.
///
/// Each step's encoder rolls on its own: `MetalEncoder::tick` counts every
/// dispatch, so a step's ~2100 dispatches are committed in
/// `MAX_DISPATCHES_PER_BUFFER` slices — 42 command buffers per step, measured
/// with `MetalEncoder::command_buffers_created` — and the encoder only holds the
/// trailing partial slice when it is committed. Rolling inside a step is safe
/// for the pipeline because the slice boundaries are invisible to it: a single
/// Metal queue runs committed buffers in commit order, and the wait is
/// unchanged.
fn generate_gpu_greedy_pipelined<F>(
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
    /// Steps queued before the host waits. Deep enough to hide sampling and
    /// detokenization, shallow enough that text still appears promptly.
    const PIPELINE_DEPTH: usize = 8;

    let mut detokenizer = crate::tokenizer::Detokenizer::new(tokenizer);
    let mut text = String::new();
    let mut generated_ids: Vec<u32> = Vec::with_capacity(config.max_new_tokens);
    let mut ttft_duration = Duration::ZERO;
    let mut decode_duration = Duration::ZERO;
    let mut prefill_duration = Duration::ZERO;
    let mut stopped_on_eos = false;
    let prompt_len = prompt_ids.len();

    let queue = model.forward_queue();
    model.set_pick_on_device(true);
    let result = (|| -> Result<(), GenerationError> {
        let prefill_started = Instant::now();
        let positions: Vec<u32> = (0..prompt_len as u32).collect();
        let mut prefill = MetalEncoder::new(&queue)
            .map_err(ModelError::from)?;
        model
            .forward_buffer_into(prompt_ids, &positions, &mut prefill)
            .map_err(ModelError::from)?;
        // Commit the prefill. Encoding is not execution: without this the buffer
        // is dropped un-ended, the KV cache is never populated, and the decode
        // steps that follow run against an empty cache. It also makes
        // `prefill_duration` honest -- it now covers the GPU, not just the host.
        prefill.commit_and_wait().map_err(ModelError::from)?;
        prefill_duration = prefill_started.elapsed();

        let mut position = prompt_len as u32;
        let mut decoded = 0_usize;
        while decoded < config.max_new_tokens {
            let depth = (config.max_new_tokens - decoded).min(PIPELINE_DEPTH);
            let decode_started = Instant::now();

            // One command buffer per step, all submitted before any wait. Each
            // step's forward reads the token the previous pick left in device
            // memory, so the host is not in the loop. Inside a step the
            // `MetalEncoder` rolling scheme applies as usual -- it counts every
            // dispatch -- so a step is committed in slices of
            // `MAX_DISPATCHES_PER_BUFFER`, not as one giant encoder;
            // `commit_terminal` then ends the trailing partial slice.
            let mut steps = Vec::with_capacity(depth);
            for step in 0..depth {
                let mut encoder = MetalEncoder::new(&queue)
                    .map_err(ModelError::from)?;
                model
                    .forward_buffer_into_step(&mut encoder, position + step as u32, step)
                    .map_err(ModelError::from)?;
                // Terminal: the encoder is dead after this, and `wait()` below is
                // the only thing left that may be called on it. `Drop` ends an
                // encoder that is still open, so the `?` on the encode above is
                // an error return rather than a Metal trap.
                encoder.commit_terminal().map_err(ModelError::from)?;
                steps.push(encoder);
            }
            for encoder in &steps {
                encoder.wait();
            }
            decode_duration += decode_started.elapsed();

            let tokens = model.take_greedy_tokens(depth);
            for token in tokens {
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
            }
            decoded += depth;
            position += depth as u32;
            if stopped_on_eos {
                break;
            }
        }

        if let Some(chunk) = detokenizer.finish()? {
            text.push_str(&chunk);
            on_text(&chunk)?;
        }
        let decode_steps = generated_ids.len().saturating_sub(1);
        let _ = decode_steps;
        Ok(())
    })();
    model.set_pick_on_device(false);
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
