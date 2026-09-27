use kaplumba_engine::generate::GenerationResult;

#[derive(Debug, Clone, Copy, Default)]
pub struct RunMetrics {
    pub load_ms: f64,
    pub ttft_ms: f64,
    pub prefill_tokens_per_second: f64,
    pub decode_tokens_per_second: f64,
    pub peak_rss_bytes: u64,
}

pub fn format_metrics(result: &GenerationResult, metrics: RunMetrics) -> String {
    format!(
        "prompt_tokens={} generated_tokens={} stopped_on_eos={} load_ms={:.3} ttft_ms={:.3} prefill_tok_s={:.3} decode_tok_s={:.3} peak_rss_bytes={}",
        result.prompt_tokens,
        result.generated_tokens,
        result.stopped_on_eos,
        metrics.load_ms,
        metrics.ttft_ms,
        metrics.prefill_tokens_per_second,
        metrics.decode_tokens_per_second,
        metrics.peak_rss_bytes
    )
}

pub fn format_diagnostic_ids(result: &GenerationResult) -> String {
    result
        .generated_ids
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}
