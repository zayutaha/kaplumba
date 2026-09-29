use kaplumba_engine::generate::GenerationResult;

#[derive(Debug, Clone, Copy, Default)]
pub struct RunMetrics {
    pub load_ms: f64,
    pub ttft_ms: f64,
    pub prefill_tokens_per_second: f64,
    pub decode_tokens_per_second: f64,
    /// Peak CPU-resident pages (`getrusage().ru_maxrss`).
    ///
    /// DELIBERATELY NOT the memory budget figure. This counts resident pages
    /// only and misses the Metal buffer pages entirely. Measured on the same
    /// run at 614 tokens, by `/usr/bin/time -l`:
    ///
    ///   peak_resident_pages_bytes  6 691 717 120  = 6.23 GiB
    ///   peak memory footprint     12 555 691 320  = 11.69 GiB
    ///
    /// and at 8192 tokens the footprint reaches ~12.96 GiB. Reading this field
    /// as a footprint understated real usage by 5.5 GiB and made a
    /// 93%-of-budget situation look like 57%. The budget-relevant number is
    /// `phys_footprint`, readable only externally via `/usr/bin/time -l`; see
    /// `crates/engine/tests/memory_proof.rs`.
    pub peak_resident_pages_bytes: u64,
}

pub fn format_metrics(result: &GenerationResult, metrics: RunMetrics) -> String {
    format!(
        "prompt_tokens={} generated_tokens={} stopped_on_eos={} load_ms={:.3} ttft_ms={:.3} prefill_tok_s={:.3} decode_tok_s={:.3} peak_resident_pages_bytes={}",
        result.prompt_tokens,
        result.generated_tokens,
        result.stopped_on_eos,
        metrics.load_ms,
        metrics.ttft_ms,
        metrics.prefill_tokens_per_second,
        metrics.decode_tokens_per_second,
        metrics.peak_resident_pages_bytes
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
