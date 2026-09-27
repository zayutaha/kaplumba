mod args;
mod report;

use std::env;
use std::process::ExitCode;
use std::time::Instant;

use args::{CliArgs, help_text};
use kaplumba_engine::config::load_generation_config;
use kaplumba_engine::generate::{GenerationConfig, generate, generate_gpu, run_benchmark};
use kaplumba_engine::metrics::{BenchmarkPlan, fixed_benchmark_prompt, measure_rss};
use kaplumba_engine::model::TextModel;
use kaplumba_engine::sampler::SamplerConfig;
use kaplumba_engine::tokenizer::GemmaTokenizer;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let arguments = env::args();
    let Some(arguments) = CliArgs::parse(arguments).map_err(|error| error.to_string())? else {
        print!("{}", help_text());
        return Ok(());
    };
    let prompt = if arguments.benchmark {
        String::new()
    } else {
        arguments.read_prompt().map_err(|error| error.to_string())?
    };
    let load_started = Instant::now();
    let mut model =
        TextModel::load_from_dir(&arguments.model_path).map_err(|error| error.to_string())?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    let tokenizer =
        GemmaTokenizer::from_model_dir(&arguments.model_path).map_err(|error| error.to_string())?;
    if arguments.benchmark {
        let prompt_ids =
            fixed_benchmark_prompt(&tokenizer, 2048).map_err(|error| error.to_string())?;
        let result = run_benchmark(
            &mut model,
            &prompt_ids,
            BenchmarkPlan {
                prompt_tokens: prompt_ids.len(),
                warmup_steps: 8,
                measured_tokens: 128,
            },
        )
        .map_err(|error| error.to_string())?;
        println!(
            "benchmark prefill_tok_s={:.3} decode_tok_s={:.3} measured_tokens={} warmup_steps={} peak_rss_bytes={} load_ms={:.3}",
            result.prefill_tokens_per_second,
            result.decode_tokens_per_second,
            result.measured_tokens,
            result.warmup_steps,
            result.peak_rss_bytes,
            load_ms
        );
        return Ok(());
    }
    let checkpoint_generation =
        load_generation_config(&arguments.model_path.join("generation_config.json"))
            .map_err(|error| error.to_string())?;
    let generation_config = GenerationConfig {
        max_new_tokens: arguments.max_new_tokens,
        eos_ids: checkpoint_generation.eos_token_ids,
        sampler: SamplerConfig {
            temperature: arguments.temperature,
            top_k: arguments.top_k,
            top_p: arguments.top_p,
        },
        seed: arguments.seed_or_random(),
        ..GenerationConfig::default()
    };
    let stream = |chunk: &str| {
        print!("{chunk}");
        use std::io::Write;
        std::io::stdout().flush().map_err(|error| error.to_string())
    };
    let result = if arguments.legacy_path {
        generate(&mut model, &tokenizer, &prompt, &generation_config, stream)
    } else {
        generate_gpu(&mut model, &tokenizer, &prompt, &generation_config, stream)
    }
    .map_err(|error| error.to_string())?;
    println!();
    if arguments.show_ids {
        println!("token_ids: {}", report::format_diagnostic_ids(&result));
    }
    if arguments.show_metrics {
        let metrics = report::RunMetrics {
            load_ms,
            ttft_ms: result.ttft_ms,
            prefill_tokens_per_second: result.prefill_tokens_per_second,
            decode_tokens_per_second: result.decode_tokens_per_second,
            peak_rss_bytes: measure_rss().unwrap_or(0),
        };
        println!("{}", report::format_metrics(&result, metrics));
    }
    Ok(())
}
