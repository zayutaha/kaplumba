use std::time::{Duration, Instant};

use thiserror::Error;

use crate::metrics::{
    BenchmarkPlan, BenchmarkResult, MetricsError, RunMetrics, benchmark_result, measure_rss,
};
use crate::model::{ModelError, TextModel};
use crate::sampler::{Sampler, SamplerConfig, SamplerError};
use crate::tokenizer::{GemmaTokenizer, TokenizerError};
use kaplumba_kernels::encoder::MetalEncoder;

/// The default usable context window, in tokens.
///
/// Raised from 8192 to 32768 after measuring that the model supports it:
/// `config.json` declares `max_position_embeddings = 262144`, so 8192 was a
/// *policy* choice, not a capability limit. See
/// `spec/decode-20tps/implementation_notes.md` §7 for the measurements.
///
/// The cost is KV memory, and only the full-attention layers pay it, because the
/// 50 sliding layers are fixed at their 1024-token window and ring. Per position,
/// full layers hold 2 x 4 global KV heads x 512 global head dim x fp16 = 8 KiB,
/// sliding layers 2 x 16 KV heads x 256 head dim x fp16 = 16 KiB:
///
/// ```text
/// full KV   = 10 layers x positions x 8 KiB  =  0.078 MiB/token
/// sliding   = 50 layers x 1024 x 16 KiB      =  0.781 GiB, constant
/// ```
///
/// i.e. 0.64 + 0.78 = 1.4 GiB at 8k, 1.3 + 0.78 = 2.1 GiB at 16k and 2.5 + 0.78 =
/// 3.3 GiB at 32k, on top of ~11.0 GiB of weights. Measured peak footprint at 32k
/// is 14.4 GiB against this box's 24 GiB, so the window fits with headroom -- but
/// it is not free, and it is why the cache still starts at 512 positions and
/// doubles on demand rather than allocating the worst case at load.
pub const DEFAULT_CONTEXT_LIMIT: usize = 32768;

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
    model.reset_gpu_caches();
    let mut sampler = Sampler::new(config.sampler, config.seed)?;
    let mut detokenizer = crate::tokenizer::Detokenizer::new(tokenizer);
    let positions = (0..prompt_ids.len())
        .map(|position| position as u32)
        .collect::<Vec<_>>();
    let prefill_started = Instant::now();
    let mut logits = model.forward_buffer(&prompt_ids, &positions)?;
    let prefill_duration = prefill_started.elapsed();
    // Timed here, on the same clock as the decode steps below, and added to
    // nothing: a decode total that absorbed the prefill would be flat in
    // `--max-new-tokens`, which is how the prefill table came to be read as a
    // decode table.
    crate::decode_profile::record_prefill(prefill_duration.as_secs_f64() * 1e3, prompt_ids.len());
    let mut decode_duration = Duration::ZERO;
    let mut ttft_duration = Duration::ZERO;
    let mut generated_ids = Vec::with_capacity(config.max_new_tokens);
    let mut text = String::new();
    let mut stopped_on_eos = false;
    let vocab_size = model.vocab_size();
    // `KAPLUMBA_MARGIN=1` / `KAPLUMBA_SEGMENTS=1`. See `MarginLog` and
    // `SegmentLog`.
    //
    // Instrumented HERE, in `generate_gpu`, and not only in `generate_gpu_greedy`,
    // because this is the function the CLI calls (`crates/cli/src/main.rs:102`).
    // The greedy path is reached only via `generate`, which is `--legacy-path`.
    // An instrument placed in the greedy path alone compiles, passes its unit
    // tests, and never fires — which is exactly what happened the first time
    // this was written.
    let mut margins = MarginLog::new();
    let mut segments = SegmentLog::new();
    // At temperature 0 the sampler is an argmax, so the margin is meaningful. At
    // temperature > 0 the sampled token is not the argmax and the fragility
    // number would describe a decision the engine did not make.
    let greedy = config.sampler.temperature == 0.0;
    let track_margin = greedy && std::env::var_os("KAPLUMBA_MARGIN").is_some();
    // Where pipeline compilation stood when load finished; `warmup_decode` armed
    // it. Read here and not only in `generate_gpu_greedy`, for the reason the
    // margin instrumentation above spells out: `generate_gpu` is the function the
    // CLI calls, so an instrument that lives only in the greedy path compiles,
    // passes its tests, and never fires.
    let load_mark = kaplumba_kernels::pipeline_audit::armed_mark().unwrap_or(0);
    for step in 0..config.max_new_tokens {
        let token = if greedy {
            argmax_logits(&logits)
        } else {
            let last_logits = logits
                .iter()
                .map(|value| half::f16::from_bits(*value).to_f32())
                .collect::<Vec<_>>();
            sampler.sample(&last_logits)?
        };
        if track_margin {
            let last_logits = logits
                .iter()
                .map(|value| half::f16::from_bits(*value).to_f32())
                .collect::<Vec<_>>();
            margins.record(generated_ids.len(), margin_from_f32(&last_logits));
        }
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
        model.forward_buffer_into_vec(&[token], &[position], &mut logits)?;
        let step_ms = decode_started.elapsed().as_secs_f64() * 1e3;
        if std::env::var_os("KAPLUMBA_STEP_TRACE").is_some() {
            eprintln!("GPU_STEP_TRACE: forward_buffer={:.2}ms", step_ms);
        }
        decode_duration += decode_started.elapsed();
        segments.record(generated_ids.len(), position, step_ms);
        // The measurement hazard this guards: a kernel compiled inside this loop
        // charges Metal's shader build to `step_ms`, which is what
        // `decode_tokens_per_second` is made of. Checked AFTER the step is
        // timed, so a panic here cannot be mistaken for a slow step, and gated,
        // so a default run pays one relaxed atomic load.
        kaplumba_kernels::pipeline_audit::assert_no_builds_since(load_mark);
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
    // One report for the whole run, after the loop. The accumulator latches
    // `printed`, so if both this entry point and `generate_gpu_greedy` reach it
    // in one process only the first produces a table.
    crate::decode_profile::report();
    // `KAPLUMBA_COUNT_BUFFERS`: command buffers per decode step. The tally is in
    // `forward_buffer`, the only scope that holds the encoder, so it has to be
    // *read* from each of the two decode entry points.
    crate::model::report_buffers_per_step();
    // `KAPLUMBA_PIPELINE_AUDIT`: what was compiled, when, and whether any of it
    // happened inside the loop above. Printed here, after the loop, so
    // `late_builds` is a statement about the whole timed region rather than
    // about whichever step happened to sample it.
    if load_mark != 0 && std::env::var_os("KAPLUMBA_PIPELINE_AUDIT").is_some() {
        eprintln!(
            "{}",
            kaplumba_kernels::pipeline_audit::audit_line_with_mark("after-decode", load_mark)
        );
    }
    margins.report();
    segments.finish();
    segments.report();
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
        peak_resident_pages_bytes: measure_rss()?,
        ..RunMetrics::default()
    };
    Ok(benchmark_result(metrics, plan)?)
}

fn argmax_logits(logits: &[u16]) -> u32 {
    let mut best = 0;
    let mut best_val = half::f16::from_bits(logits[0]).to_f32();
    for index in 1..logits.len() {
        let val = half::f16::from_bits(logits[index]).to_f32();
        if val > best_val {
            best_val = val;
            best = index;
        }
    }
    best as u32
}

/// The argmax with `skip` excluded, using `argmax_logits`'s exact tie rule.
///
/// Deliberately a second pass over the same row with the same comparison, rather
/// than a single pass tracking a runner-up. A fused version has to decide what
/// "second" means when several entries share the top value, and every one of
/// those decisions is a place the fragility number could quietly disagree with
/// the token that was actually emitted. This way the runner-up is by
/// construction the argmax the sampler would have picked had the winner been
/// absent, which is the only definition that means anything here.
fn argmax_excluding(logits: &[u16], skip: usize) -> u32 {
    let mut best = usize::MAX;
    for index in 0..logits.len() {
        if index == skip {
            continue;
        }
        if best == usize::MAX
            || half::f16::from_bits(logits[index]).to_f32()
                > half::f16::from_bits(logits[best]).to_f32()
        {
            best = index;
        }
    }
    assert!(best != usize::MAX, "a 1-entry vocabulary has no runner-up");
    best as u32
}

/// How far the greedy decision was from flipping, in units of the fp16 ulp at
/// the winning logit.
///
/// This is the measurement that makes "decode is reproducible" a claim instead
/// of an observation. The logits are fp16 over a 262,144-entry vocabulary, so at
/// a winning logit of ~28 the representable step is 2^(4-10) = 0.015625. Any
/// top-2 gap below one ulp means a single rounding decision in the last matvec
/// would emit a different token, and a run of N tokens is only as reproducible
/// as its smallest margin.
///
/// Reported in ulps rather than in logits because the absolute gap is
/// meaningless on its own: 0.01 is a comfortable margin at logit 4 and a coin
/// flip at logit 28. This is also why an fp32 accumulator difference that is far
/// below the fp16 grid can still change the output — which is the mechanism
/// behind the FMA-contraction defect found in `spec/decode-bandwidth-240gb`
/// (T2), where re-association moved 2 of 21,504 outputs by one ulp.
#[derive(Clone, Copy, Debug)]
pub struct ArgmaxMargin {
    pub token: u32,
    /// Winning logit, fp32 view of the fp16 row.
    pub top: f32,
    /// Runner-up logit.
    pub runner_up: f32,
    /// `(top - runner_up)`, fp32.
    pub gap: f32,
    /// `gap` in units of the fp16 ulp at `top`. `<= 1.0` means one rounding
    /// decision could change the token.
    pub ulps: f32,
}

/// The fp16 ulp at `value`: `2^(exponent - 10)`, with fp16's 10 explicit
/// mantissa bits. Subnormals and zero are floored at `f32::MIN_POSITIVE`'s ulp
/// so the divisor is never zero and the ratio stays finite.
fn fp16_ulp(value: f32) -> f32 {
    if !value.is_finite() || value == 0.0 {
        return f32::from_bits(1 << 23); // f32::MIN_POSITIVE
    }
    let exponent = value.abs().log2().floor() as i32;
    // fp16 max exponent is 15; anything above is not a value this format can
    // hold, so clamp rather than produce a denormal-flushing subnormal.
    let exponent = exponent.clamp(-14, 15);
    2.0f32.powi(exponent - 10)
}

/// `argmax_logits` plus the distance to the runner-up.
pub fn argmax_with_margin(logits: &[u16]) -> ArgmaxMargin {
    assert!(
        logits.len() >= 2,
        "a margin needs a runner-up, so the vocabulary needs two entries"
    );
    let token = argmax_logits(logits);
    let runner_up = argmax_excluding(logits, token as usize);
    let top = half::f16::from_bits(logits[token as usize]).to_f32();
    let second = half::f16::from_bits(logits[runner_up as usize]).to_f32();
    let gap = top - second;
    ArgmaxMargin {
        token,
        top,
        runner_up: second,
        gap,
        ulps: gap / fp16_ulp(top),
    }
}

/// Per-segment decode rate, bucketed by how much context had accumulated.
///
/// The question this exists to answer is "does decode get slower as the
/// generation gets longer?", and the aggregate `decode_tok_s` cannot answer it:
/// a run that starts fast and decays reports one number that describes neither
/// end. Context grows by one token per step, so any per-token cost that depends
/// on it -- attention over a longer KV, a larger working set, more TLB pressure,
/// more of the KV cache falling out of a cache level -- shows up here as a
/// downward trend and is invisible in the average.
///
/// Buckets are fixed-size windows of *steps*, not of context length, because the
/// step index is what the loop actually knows and it makes the boundaries exact.
/// 128 steps is a reasonable default: the reference run is ~128 tokens, so the
/// first bucket is directly comparable to the benchmark number and the rest show
/// the drift from there.
///
/// Gated on `KAPLUMBA_SEGMENTS=1`. The bookkeeping is a push and an add per step
/// and never runs otherwise.
struct SegmentLog {
    enabled: bool,
    bucket: usize,
    /// `(first_step, last_step, total_ms, n_steps)` per completed bucket.
    buckets: Vec<(usize, usize, f64, usize)>,
    current: Option<(usize, f64, usize)>,
}

impl SegmentLog {
    fn new() -> Self {
        Self {
            enabled: std::env::var("KAPLUMBA_SEGMENTS").is_ok_and(|v| v != "0"),
            bucket: std::env::var("KAPLUMBA_SEGMENTS_BUCKET")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|n| *n > 0)
                .unwrap_or(128),
            buckets: Vec::new(),
            current: None,
        }
    }

    /// One decode step's wall time. `context` is the position being computed,
    /// which is the KV length that step attends over.
    fn record(&mut self, step: usize, context: u32, step_ms: f64) {
        if !self.enabled {
            return;
        }
        let entry = match &mut self.current {
            Some(e) => e,
            None => {
                self.current = Some((step, 0.0, 0));
                self.current.as_mut().expect("just assigned")
            }
        };
        entry.1 += step_ms;
        entry.2 += 1;
        if entry.2 == self.bucket {
            let (first, ms, n) = self.current.take().expect("just matched");
            self.buckets.push((first, first + n - 1, ms, n));
        }
        let _ = context;
    }

    fn finish(&mut self) {
        if let Some((first, ms, n)) = self.current.take() {
            if n > 0 {
                self.buckets.push((first, first + n - 1, ms, n));
            }
        }
    }

    fn report(&self) {
        if !self.enabled || self.buckets.is_empty() {
            return;
        }
        eprintln!("=== SEGMENTED DECODE (KAPLUMBA_SEGMENTS) ===");
        eprintln!(
            "{:>6} {:>7} {:>10} {:>9} {:>8}",
            "steps", "context", "ms/step", "tok/s", "vs_first"
        );
        let first_rate = self
            .buckets
            .first()
            .map(|(_, _, ms, n)| *n as f64 / (*ms / 1000.0));
        for (first, last, ms, n) in &self.buckets {
            let rate = *n as f64 / (*ms / 1000.0);
            let ratio = match first_rate {
                Some(fr) if fr > 0.0 => rate / fr,
                _ => 1.0,
            };
            eprintln!(
                "{:>6} {:>7} {:>10.3} {:>9.3} {:>7.1}%",
                format!("{first}-{last}"),
                format!("~{last}"),
                *ms / *n as f64,
                rate,
                ratio * 100.0
            );
        }
        // The trend is the result. A single number per bucket does not say
        // whether decode is holding up, and "it was fine" is exactly the
        // conclusion that a drifting rate gets filed under.
        let last_rate = self
            .buckets
            .last()
            .map(|(_, _, ms, n)| *n as f64 / (ms / 1000.0));
        if let (Some(first_rate), Some(last_rate)) = (first_rate, last_rate) {
            let drift = (last_rate / first_rate - 1.0) * 100.0;
            eprintln!(
                "SEGMENT-DRIFT first={first_rate:.3} tok/s last={last_rate:.3} tok/s \
                 drift={drift:+.1}%"
            );
            if drift < -10.0 {
                eprintln!(
                    "SEGMENT-NOTE decode loses more than 10% from the first bucket to the last; \
                     per-token cost is growing with context"
                );
            }
        }
    }
}

/// `argmax_with_margin` for a row the caller has already widened to `f32`.
///
/// `generate_gpu` converts the fp16 row to `f32` before sampling, because
/// `Sampler::sample` takes `&[f32]`. Re-running the fp16 path would mean a second
/// definition of "the row" that could disagree with the one the token came from —
/// and a fragility number that describes a different row than the one sampled is
/// worse than no number.
///
/// The ulp divisor is still the **fp16** ulp, because that is the precision the
/// logits were rounded to before anything else happened. Dividing by the `f32`
/// ulp would report a gap of one fp16 step as ~4600 "ulps", which reads as
/// enormous safety on what is actually a coin flip.
pub fn margin_from_f32(row: &[f32]) -> ArgmaxMargin {
    assert!(
        row.len() >= 2,
        "a margin needs a runner-up, so the vocabulary needs two entries"
    );
    let mut token = 0usize;
    for index in 1..row.len() {
        if row[index] > row[token] {
            token = index;
        }
    }
    let mut runner_up = usize::MAX;
    for index in 0..row.len() {
        if index == token {
            continue;
        }
        if runner_up == usize::MAX || row[index] > row[runner_up] {
            runner_up = index;
        }
    }
    assert!(runner_up != usize::MAX, "no runner-up in a >= 2 entry row");
    let gap = row[token] - row[runner_up];
    ArgmaxMargin {
        token: token as u32,
        top: row[token],
        runner_up: row[runner_up],
        gap,
        ulps: gap / fp16_ulp(row[token]),
    }
}

/// Collects the per-step margin so a run can report its own fragility.
///
/// Gated on `KAPLUMBA_MARGIN=1`. The cost when off is one `Once` read, because
/// a 262,144-entry second pass per step is real work and this is a diagnostic,
/// not something the hot path should carry.
struct MarginLog {
    enabled: bool,
    steps: Vec<(usize, ArgmaxMargin)>,
}

impl MarginLog {
    fn new() -> Self {
        Self {
            enabled: std::env::var("KAPLUMBA_MARGIN").is_ok_and(|v| v != "0"),
            steps: Vec::new(),
        }
    }

    fn record(&mut self, step: usize, margin: ArgmaxMargin) {
        if self.enabled {
            self.steps.push((step, margin));
        }
    }

    /// Per-step lines, then a summary naming the tightest step.
    ///
    /// The summary is the number that matters and it is reported as a *worst
    /// case*, because a run is reproducible exactly when its least decisive
    /// step is decisive. A median margin would be the wrong statistic and would
    /// hide exactly the step anyone needs to look at.
    fn report(&self) {
        if !self.enabled || self.steps.is_empty() {
            return;
        }
        for (step, margin) in &self.steps {
            eprintln!(
                "MARGIN step={step} token={} top={:.6} runner_up={} gap={:.6} ulps={:.3}",
                margin.token, margin.top, margin.runner_up, margin.gap, margin.ulps
            );
        }
        let (worst_step, worst) = self
            .steps
            .iter()
            .min_by(|a, b| {
                a.1.ulps
                    .partial_cmp(&b.1.ulps)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("non-empty by the guard above");
        let ties = self.steps.iter().filter(|(_, m)| m.ulps <= 0.0).count();
        let within_ulp = self.steps.iter().filter(|(_, m)| m.ulps <= 1.0).count();
        let median = {
            let mut v: Vec<f32> = self.steps.iter().map(|(_, m)| m.ulps).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            v[v.len() / 2]
        };
        eprintln!(
            "MARGIN-SUMMARY steps={} min_ulps={:.3} at_step={} (token={}) median_ulps={:.3} \
             within_1_ulp={} exact_ties={}",
            self.steps.len(),
            worst.ulps,
            worst_step,
            worst.token,
            median,
            within_ulp,
            ties
        );
        if ties > 0 {
            eprintln!(
                "MARGIN-WARN {ties} step(s) have an exact fp16 tie at the top; the token is \
                 decided by lowest index and any perturbation of the winner flips it"
            );
        } else if within_ulp > 0 {
            eprintln!(
                "MARGIN-NOTE {within_ulp} step(s) decided within one fp16 ulp; reproducible only \
                 because no arithmetic on this path is nondeterministic, not because the margin \
                 is comfortable"
            );
        }
    }
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
    // `KAPLUMBA_MARGIN=1`: per-step distance from the greedy decision flipping,
    // in fp16 ulps. See `argmax_with_margin`. This is the measurement that makes
    // "decode reproduces" a claim rather than an observation.
    let mut margins = MarginLog::new();
    // `KAPLUMBA_SEGMENTS=1`: decode rate bucketed by context length, so a run
    // says whether per-token cost grows as the KV does. See `SegmentLog`.
    let mut segments = SegmentLog::new();
    // Where pipeline compilation stood when load finished. `warmup_decode`
    // armed this; if a step compiles something, the assert below says so by
    // name rather than leaving it inside a tok/s figure.
    let load_mark = kaplumba_kernels::pipeline_audit::armed_mark().unwrap_or(0);

    let result = (|| -> Result<(), GenerationError> {
        model.reset_gpu_caches();
        let prefill_started = Instant::now();
        let positions: Vec<u32> = (0..prompt_len as u32).collect();
        let mut prefill = MetalEncoder::new(&queue).map_err(ModelError::from)?;
        prefill.set_max_dispatches(2500);
        model
            .forward_buffer_into(prompt_ids, &positions, &mut prefill)
            .map_err(ModelError::from)?;
        // Commit the prefill. Encoding is not execution: without this the buffer
        // is dropped un-ended, the KV cache is never populated, and the decode
        // steps that follow run against an empty cache. It also makes
        // `prefill_duration` honest -- it now covers the GPU, not just the host.
        prefill.commit_and_wait().map_err(ModelError::from)?;
        prefill_duration = prefill_started.elapsed();
        // Timed here, on the same clock as the decode steps below, and added to
        // nothing: a decode total that absorbed the prefill would be flat in
        // `--max-new-tokens`, which is how the prefill table came to be read as
        // a decode table.
        crate::decode_profile::record_prefill(prefill_duration.as_secs_f64() * 1e3, prompt_len);

        // Read after the wait: the prefill's device argmax yields the first generated token.
        let mut token = model.last_token();
        if margins.enabled {
            let logits = model.last_logits();
            margins.record(generated_ids.len(), argmax_with_margin(&logits));
        }
        let pipelined = !margins.enabled
            && crate::decode_profile::active_mode() == crate::decode_profile::Mode::Off
            && std::env::var_os("KAPLUMBA_NO_PIPELINE").is_none();

        if pipelined {
            if generated_ids.is_empty() {
                ttft_duration = request_started.elapsed();
            }
            if config.eos_ids.contains(&token) {
                stopped_on_eos = true;
            } else {
                generated_ids.push(token);
                if let Some(chunk) = detokenizer.push(token)? {
                    text.push_str(&chunk);
                    on_text(&chunk)?;
                }

                if generated_ids.len() < config.max_new_tokens {
                    let total_to_generate = config.max_new_tokens - generated_ids.len();

                    let pos0 = u32::try_from(prompt_len)
                        .map_err(|_| GenerationError::ContextExceeded {
                            limit: config.context_limit,
                        })?;
                    let mut enc0 = MetalEncoder::new(&queue).map_err(ModelError::from)?;
                    enc0.set_max_dispatches(kaplumba_kernels::encoder::decode_max_dispatches_per_buffer());
                    model.forward_step_pipelined(&mut enc0, pos0, Some(token), 0, 0)?;
                    crate::model::record_decode_step_tallies(&enc0);
                    enc0.commit_terminal().map_err(ModelError::from)?;

                    let mut in_flight_enc = enc0;
                    let mut in_flight_step = 0usize;
                    let mut in_flight_pos = pos0;
                    let mut step_start = Instant::now();

                    for step_idx in 1..total_to_generate {
                        let slot = step_idx % 2;
                        let pos = u32::try_from(prompt_len + step_idx)
                            .map_err(|_| GenerationError::ContextExceeded {
                                limit: config.context_limit,
                            })?;

                        let mut next_enc = MetalEncoder::new(&queue).map_err(ModelError::from)?;
                        next_enc.set_max_dispatches(kaplumba_kernels::encoder::decode_max_dispatches_per_buffer());
                        model.forward_step_pipelined(&mut next_enc, pos, None, slot, step_idx as u32)?;
                        crate::model::record_decode_step_tallies(&next_enc);
                        next_enc.commit_terminal().map_err(ModelError::from)?;

                        in_flight_enc.wait();
                        let step_ms = step_start.elapsed().as_secs_f64() * 1e3;
                        decode_duration += Duration::from_secs_f64(step_ms / 1e3);
                        step_start = Instant::now();
                        segments.record(generated_ids.len(), in_flight_pos, step_ms);
                        kaplumba_kernels::pipeline_audit::assert_no_builds_since(load_mark);

                        let step_token = model.read_host_token(in_flight_step);
                        generated_ids.push(step_token);
                        if let Some(chunk) = detokenizer.push(step_token)? {
                            text.push_str(&chunk);
                            on_text(&chunk)?;
                        }

                        if config.eos_ids.contains(&step_token) {
                            stopped_on_eos = true;
                            next_enc.wait();
                            model.retract_gpu_caches(1);
                            break;
                        }

                        in_flight_enc = next_enc;
                        in_flight_step = step_idx;
                        in_flight_pos = pos;
                    }

                    if !stopped_on_eos {
                        in_flight_enc.wait();
                        let step_ms = step_start.elapsed().as_secs_f64() * 1e3;
                        decode_duration += Duration::from_secs_f64(step_ms / 1e3);
                        segments.record(generated_ids.len(), in_flight_pos, step_ms);
                        kaplumba_kernels::pipeline_audit::assert_no_builds_since(load_mark);

                        let final_token = model.read_host_token(in_flight_step);
                        generated_ids.push(final_token);
                        if let Some(chunk) = detokenizer.push(final_token)? {
                            text.push_str(&chunk);
                            on_text(&chunk)?;
                        }
                        if config.eos_ids.contains(&final_token) {
                            stopped_on_eos = true;
                        }
                    }
                }
            }
        } else {
            loop {
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
                let position =
                    u32::try_from(position).map_err(|_| GenerationError::ContextExceeded {
                        limit: config.context_limit,
                    })?;
                let step_started = Instant::now();
                let mut encoder = MetalEncoder::new(&queue).map_err(ModelError::from)?;
                encoder.set_max_dispatches(kaplumba_kernels::encoder::decode_max_dispatches_per_buffer());
                crate::decode_profile::begin_step(&mut encoder, |encoder| {
                    model.calibration_dispatch(encoder)
                })
                .map_err(ModelError::from)?;
                let t_encode_start = Instant::now();
                model
                    .forward_buffer_into_step(&mut encoder, position, token)
                    .map_err(ModelError::from)?;
                let t_encode_ms = t_encode_start.elapsed().as_secs_f64() * 1e3;
                crate::model::record_decode_step_tallies(&encoder);
                let step_mode =
                    crate::decode_profile::active_mode() == crate::decode_profile::Mode::Step;
                let t_wait_start = Instant::now();
                if step_mode {
                    crate::decode_profile::close_step(&mut encoder).map_err(ModelError::from)?;
                } else {
                    encoder.commit_terminal().map_err(ModelError::from)?;
                    encoder.wait();
                }
                let t_wait_ms = t_wait_start.elapsed().as_secs_f64() * 1e3;
                let step_ms = step_started.elapsed().as_secs_f64() * 1e3;
                if std::env::var_os("KAPLUMBA_STEP_TRACE").is_some() {
                    eprintln!(
                        "STEP_TRACE: encode={:.2}ms wait={:.2}ms total={:.2}ms",
                        t_encode_ms, t_wait_ms, step_ms
                    );
                }
                crate::decode_profile::end_step(&mut encoder, step_ms).map_err(ModelError::from)?;
                decode_duration += Duration::from_secs_f64(step_ms / 1e3);
                segments.record(generated_ids.len(), position, step_ms);
                kaplumba_kernels::pipeline_audit::assert_no_builds_since(load_mark);
                token = model.last_token();
                if margins.enabled {
                    let logits = model.last_logits();
                    margins.record(generated_ids.len(), argmax_with_margin(&logits));
                }
            }
        }

        if let Some(chunk) = detokenizer.finish()? {
            text.push_str(&chunk);
            on_text(&chunk)?;
        }
        Ok(())
    })();
    result?;

    // One report for the whole run, after the loop: printed per step it would be
    // N tables of the same thing and unreadable.
    crate::decode_profile::report();
    // `KAPLUMBA_COUNT_BUFFERS`: command buffers per decode step. The tally lives
    // in `forward_buffer`, the only scope that holds the encoder.
    crate::model::report_buffers_per_step();
    // `KAPLUMBA_PIPELINE_AUDIT`: what was compiled, when, and whether any of it
    // happened inside the loop above. Printed here, after the loop, so the
    // `late_builds` field is a statement about the whole timed region rather
    // than about whichever step happened to sample it.
    if load_mark != 0 && std::env::var_os("KAPLUMBA_PIPELINE_AUDIT").is_some() {
        eprintln!(
            "{}",
            kaplumba_kernels::pipeline_audit::audit_line_with_mark("after-decode", load_mark)
        );
    }
    margins.report();
    segments.finish();
    segments.report();

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

    use super::{argmax_logits, argmax_with_margin, fp16_ulp};

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

    // ---- argmax_with_margin: the fragility instrument ----
    //
    // The failure mode of this instrument is a *plausible number*, not a crash:
    // a margin off by a factor of a thousand, or computed against the wrong
    // logit, looks like a real measurement and would be filed as one. So the
    // teeth checks below assert a *divergence* from the wrong reading rather than
    // only the right value — see `fp16_ulp_is_pinned_to_the_mantissa_width` and
    // `one_ulp_gap_reads_as_one_ulp_not_as_thousands`.

    /// The ulp divisor has to be the ulp *at the winning logit*, not a constant.
    /// `2^(e-10)` for fp16's 10 mantissa bits, where `2^e <= |top| < 2^(e+1)`.
    ///
    /// Worked check on the 16.0 boundary, because it is the one that is easy to
    /// get wrong: 16.0 = 1.0 x 2^4, so e = 4 and ulp = 2^-6 = 0.015625 — the
    /// next representable value is 16.015625. The `2^-7` it is easy to write is
    /// ulp(8.0). This test was written with that error and failed; see
    /// `spec/decode-bandwidth-240gb/implementation_notes.md`.
    #[test]
    fn fp16_ulp_tracks_the_exponent_band() {
        assert_eq!(fp16_ulp(1.0), 0.0009765625); // 2^-10
        assert_eq!(fp16_ulp(1.999), 0.0009765625);
        assert_eq!(fp16_ulp(2.0), 0.001953125); // 2^-9
        assert_eq!(fp16_ulp(8.0), 0.0078125); // 2^-7
        assert_eq!(fp16_ulp(16.0), 0.015625); // 2^-6, NOT 2^-7
        assert_eq!(fp16_ulp(28.5), 0.015625); // 2^-6, the measured band
        assert_eq!(fp16_ulp(30.0), 0.015625);
        assert_eq!(fp16_ulp(0.0), f32::MIN_POSITIVE); // divisor never zero
        assert_eq!(fp16_ulp(f32::NAN), f32::MIN_POSITIVE);
        assert_eq!(fp16_ulp(-28.5), 0.015625); // sign-independent
        // The 16.0 boundary from both sides: 16.0 is the *first* value of the
        // [16,32) band, so the value immediately below it is in [8,16) and has a
        // half-sized ulp. This is the exact step where `fp16_ulp` must change.
        assert_eq!(fp16_ulp(16.0), 0.015625);
        assert_eq!(fp16_ulp(16.0 - 0.015625), 0.0078125); // 15.984375, band [8,16)
        assert_eq!(fp16_ulp(15.999), 0.0078125);
        assert_eq!(fp16_ulp(16.0 - 1e-4), 0.0078125); // just under the boundary
    }

    /// Teeth check: an exponent offset of -8 (as if fp16 had 8 mantissa bits)
    /// makes every band assertion above fail by exactly 4x.
    ///
    /// Asserted as a divergence so the test states what it guards — the value is
    /// pinned to the fp16 *mantissa width*, not merely to something monotonic in
    /// the exponent. A looser approximation would pass a smaller version of this.
    #[test]
    fn fp16_ulp_is_pinned_to_the_mantissa_width() {
        let correct = fp16_ulp(28.5);
        let wrong_width = 2.0f32.powi(4 - 8); // 8 mantissa bits -> 2^-4
        assert_eq!(correct, 0.015625);
        assert_eq!(wrong_width, 0.0625);
        assert!((wrong_width / correct - 4.0).abs() < 1e-6);
        assert_ne!(correct, wrong_width);
    }

    /// A comfortable margin reports a large ulp count, and reports the *same*
    /// token `argmax_logits` does. The two must never disagree: if they do, the
    /// fragility number is describing a decision the engine did not make.
    #[test]
    fn margin_agrees_with_argmax_and_scales_the_gap() {
        let m = argmax_with_margin(&row(&[1.0, 20.0, 2.0]));
        assert_eq!(m.token, argmax_logits(&row(&[1.0, 20.0, 2.0])));
        assert_eq!(m.top, 20.0);
        assert_eq!(m.runner_up, 2.0);
        assert_eq!(m.gap, 18.0);
        // 18 / 0.015625 = 1152
        assert!((m.ulps - 1152.0).abs() < 1e-3, "ulps was {}", m.ulps);
    }

    /// The number the instrument exists for: a one-ulp gap must report ~1.0.
    ///
    /// The failure mode guarded is a *smaller* divisor. Using the fp32 ulp at
    /// 28.5 (3.4e-6) instead of the fp16 one (0.015625) divides by ~4600x less and
    /// reports ~4600 ulps for a coin flip — which reads as enormous safety. Both
    /// readings appear below so the test asserts their divergence, not a constant
    /// that a wrong formula could satisfy in both places at once.
    #[test]
    fn one_ulp_gap_reads_as_one_ulp_not_as_thousands() {
        let top = 28.5f32;
        let fp16_step = fp16_ulp(top);
        let m = argmax_with_margin(&row(&[top, top - fp16_step, 0.0]));
        assert_eq!(m.token, 0);
        assert!(
            (m.ulps - 1.0).abs() < 1e-3,
            "a one-ulp gap must read as 1.0 ulp, got {}",
            m.ulps
        );
        let fp32_ulp = top * 2.0f32.powi(-23);
        let wrong = m.gap / fp32_ulp;
        assert!(
            wrong > 1000.0,
            "the wrong divisor must be obviously different, got {wrong}"
        );
        assert!((m.ulps - wrong).abs() > 1000.0);
    }

    /// An exact fp16 tie is the maximally fragile step and must report 0.0, not a
    /// tiny positive number. With top-1 and top-2 equal, `gap` is exactly 0.
    #[test]
    fn exact_tie_reports_zero_ulps() {
        let m = argmax_with_margin(&row(&[7.0, 7.0, 7.0]));
        assert_eq!(m.token, 0, "ties resolve to the lowest index");
        assert_eq!(m.gap, 0.0);
        assert_eq!(m.ulps, 0.0);
    }

    /// fp16 quantisation is the whole reason this instrument exists, so the test
    /// has to show that two **distinct f32** values can collapse to the same fp16
    /// and that the margin reports a tie rather than the f32 gap.
    ///
    /// The perturbation has to be sized between the two grids: at 28.5 the f32 ulp
    /// is 2^-19 = 1.9e-6 and the fp16 ulp is 0.015625, so `1e-5` is ~5 f32 ulps
    /// (distinct values) and 1/1500th of an fp16 ulp (same fp16). `1e-9`, which
    /// this test was first written with, does not even survive the *f32* add and
    /// fails `assert_ne!` on the f32 values rather than testing anything.
    #[test]
    fn fp16_quantisation_is_visible_as_a_zero_margin() {
        let a = 28.5f32;
        let b = 28.5f32 + 1e-5;
        assert_ne!(a, b, "must be two distinct f32 values");
        let (fa, fb) = (f16::from_f32(a), f16::from_f32(b));
        assert_eq!(fa.to_bits(), fb.to_bits(), "...that are the same fp16");
        let zero = f16::from_f32(0.0).to_bits();
        let m = argmax_with_margin(&[fa.to_bits(), fb.to_bits(), zero]);
        assert_eq!(m.ulps, 0.0, "a collapsed pair is an exact tie");
        assert_eq!(m.gap, 0.0);
    }

    /// Negative logits, and a case where the runner-up is the LAST entry rather
    /// than an early one — the scan must not stop early.
    #[test]
    fn margin_handles_negative_and_late_runner_up() {
        let m = argmax_with_margin(&row(&[-30.0, -29.0, -28.0, -1.0, -28.0]));
        assert_eq!(m.token, 3);
        assert_eq!(m.runner_up, -28.0);
        assert_eq!(m.gap, 27.0);
        // -1.0 is in [0.5,1) so ulp is 2^-11 = 0.00048828125; but the divisor uses
        // |top| = 1.0 -> 2^-10.
        assert!((m.ulps - 27.0 / fp16_ulp(1.0)).abs() < 1e-3);
    }

    /// All-negative row where the winner is the first entry: the fragile case a
    /// fused runner-up tracker gets wrong.
    #[test]
    fn margin_when_winner_is_index_zero_and_others_are_lower() {
        let m = argmax_with_margin(&row(&[-0.5, -30.0, -2.0]));
        assert_eq!(m.token, 0);
        assert_eq!(m.runner_up, -2.0);
        assert_eq!(m.gap, 1.5);
        // |top| in [0.25,0.5) -> 2^-12.
        assert!((m.ulps - 1.5 / fp16_ulp(-0.5)).abs() < 1e-3);
    }

    /// A one-entry row has no runner-up and must say so rather than inventing one.
    /// The whole instrument is meaningless without a runner-up, so a silent
    /// fallback here would produce a fabricated margin.
    #[test]
    #[should_panic(expected = "needs two entries")]
    fn margin_rejects_a_single_entry_row() {
        let _ = argmax_with_margin(&row(&[42.0]));
    }
}
