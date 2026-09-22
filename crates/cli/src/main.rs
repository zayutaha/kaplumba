//! kaplumba-engine CLI: `metal-smoke` (T1.0) and `generate` (T1.4).
//!
//! `generate` streams decoded text and can emit a `kaplumba-bench/1` report
//! (R7.4) so Rust-engine runs are diffable against `scripts/bench.py`.

mod args;
mod report;

use std::io::Write;
use std::time::Instant;

use kaplumba_engine::{
    Gemma4Model, GemmaTokenizer, GenParams, MetalContext, SampleParams, generate, load_exl3,
};

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let code = match argv.first().map(String::as_str) {
        Some("metal-smoke") => metal_smoke(),
        Some("generate") => {
            if argv.iter().any(|a| a == "-h" || a == "--help") {
                print!("{}", args::usage());
                0
            } else {
                match args::parse_generate(&argv[1..]) {
                    Ok(a) => generate_cmd(&a),
                    Err(e) => {
                        eprintln!("error: {e}\n");
                        eprint!("{}", args::usage());
                        2
                    }
                }
            }
        }
        Some("--help") | Some("-h") => {
            print!("{}", args::usage());
            0
        }
        Some(other) => {
            eprintln!("unknown command: {other} (try --help)");
            2
        }
        None => {
            eprint!("{}", args::usage());
            2
        }
    };
    std::process::exit(code);
}

fn metal_smoke() -> i32 {
    match MetalContext::new() {
        Ok(ctx) => {
            let n = 1 << 16;
            match kaplumba_engine::clear_buffer_u32(&ctx, n, 0xA5A5_A5A5) {
                Ok(v) if v.iter().all(|&x| x == 0xA5A5_A5A5) => {
                    println!("metal-smoke: OK  elems={n} pattern=0xA5A5A5A5");
                    0
                }
                Ok(_) => {
                    eprintln!("metal-smoke: FAIL buffer not fully filled");
                    1
                }
                Err(e) => {
                    eprintln!("metal-smoke: FAIL fill error: {e}");
                    1
                }
            }
        }
        Err(e) => {
            eprintln!("metal-smoke: FAIL context: {e}");
            1
        }
    }
}

/// Where streamed text goes while generation runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sink {
    None,
    Stdout,
    Stderr,
}

/// Incremental detokenizer: re-decodes the growing id list and emits only the new
/// suffix (handles multi-byte pieces and skipped specials without artifacts).
struct Streamer<'a> {
    tok: &'a GemmaTokenizer,
    ids: Vec<u32>,
    printed: usize,
    sink: Sink,
    failed: Option<String>,
}

impl<'a> Streamer<'a> {
    fn new(tok: &'a GemmaTokenizer, sink: Sink) -> Self {
        Self {
            tok,
            ids: Vec::new(),
            printed: 0,
            sink,
            failed: None,
        }
    }

    fn push(&mut self, id: u32) {
        self.ids.push(id);
        if matches!(self.sink, Sink::None) {
            return;
        }
        let text = match self.tok.decode(&self.ids) {
            Ok(t) => t,
            Err(e) => {
                if self.failed.is_none() {
                    self.failed = Some(e.to_string());
                }
                return;
            }
        };
        if text.len() <= self.printed || !text.is_char_boundary(self.printed) {
            return;
        }
        let delta = &text[self.printed..];
        self.printed = text.len();
        match self.sink {
            Sink::Stdout => {
                print!("{delta}");
                let _ = std::io::stdout().flush();
            }
            Sink::Stderr => {
                eprint!("{delta}");
                let _ = std::io::stderr().flush();
            }
            Sink::None => {}
        }
    }
}

fn generate_cmd(a: &args::GenerateArgs) -> i32 {
    let progress = |msg: String| {
        if !a.quiet {
            eprintln!("{msg}");
        }
    };

    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: metal context: {e}");
            return 1;
        }
    };

    let t_load = Instant::now();
    let model = match load_exl3(&ctx, &a.model) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: load {}: {e}", a.model.display());
            return 1;
        }
    };
    let load_s = t_load.elapsed().as_secs_f64();
    progress(format!(
        "loaded {} tensors / {} shards / {:.3} GB in {:.1}s",
        model.n_tensors(),
        model.n_shards(),
        model.resident_gb(),
        load_s
    ));

    let tokenizer = match GemmaTokenizer::from_dir(&a.model) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: tokenizer: {e}");
            return 1;
        }
    };

    let prompt_text = match (&a.prompt, &a.prompt_file) {
        (Some(t), _) => t.clone(),
        (None, Some(p)) => match std::fs::read_to_string(p) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("error: read {}: {e}", p.display());
                return 1;
            }
        },
        (None, None) => {
            eprintln!("error: --prompt or --prompt-file is required");
            return 2;
        }
    };

    let mut ids = match tokenizer.encode(&prompt_text) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("error: encode: {e}");
            return 1;
        }
    };
    if a.bos {
        ids.insert(0, tokenizer.gemma_special_ids().bos);
    }
    if ids.is_empty() {
        eprintln!("error: prompt tokenized to zero tokens");
        return 2;
    }

    let gemma = match Gemma4Model::from_model(&model) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: model config: {e}");
            return 1;
        }
    };

    let params = GenParams {
        max_new: a.max_tokens,
        sample: SampleParams {
            temp: a.temp,
            top_k: a.top_k,
            top_p: a.top_p,
        },
        seed: a.seed,
        eos: a.eos.clone(),
        layers: a.layers,
    };

    let sink = if !a.stream {
        Sink::None
    } else if a.json_to_stdout() {
        Sink::Stderr
    } else {
        Sink::Stdout
    };
    let mut streamer = Streamer::new(&tokenizer, sink);
    progress(format!(
        "prompt: {} tokens (bos={}) → generating up to {} tokens{}",
        ids.len(),
        a.bos,
        a.max_tokens,
        match a.layers {
            Some(n) => format!(" (layers={n})"),
            None => String::new(),
        }
    ));

    let t_wall = Instant::now();
    let stats = match generate(&gemma, &ids, &params, |id| {
        streamer.push(id);
        true
    }) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: generate: {e}");
            return 1;
        }
    };
    let wall_ms = t_wall.elapsed().as_secs_f64() * 1000.0;
    if let Some(err) = streamer.failed {
        eprintln!("warning: decode: {err}");
    }
    if matches!(sink, Sink::Stdout) && stats.gen_tokens > 0 {
        println!();
    }

    let text = tokenizer.decode(&stats.tokens).unwrap_or_default();
    let (metal_alloc_bytes, metal_working_set_bytes) = report::metal_bytes(&ctx);
    let outcome = report::RunOutcome {
        load_s,
        wall_ms,
        stats: &stats,
        text: &text,
        backend: "rust-naive-cpu",
        n_layers: gemma.cfg.n_layers,
        weights_bytes: model.resident_bytes(),
        peak_rss_bytes: report::peak_rss_bytes(),
        metal_alloc_bytes,
        metal_working_set_bytes,
    };
    let rep = report::build_report(a, &outcome);

    if let Some(path) = &a.json {
        let text = match serde_json::to_string_pretty(&rep) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("error: encode report: {e}");
                return 1;
            }
        };
        if a.json_to_stdout() {
            println!("{text}");
        } else if let Err(e) = std::fs::write(path, format!("{text}\n")) {
            eprintln!("error: write {}: {e}", path.display());
            return 1;
        }
    }

    progress(format!(
        "prefill {:.1}ms ({:.1} tok/s) · ttft {:.1}ms · {} tokens · {} · {:.3} tok/s decode ({} steps, {:.1}x recompute)",
        stats.prefill_s * 1000.0,
        stats.pp_tps(),
        stats.ttft_ms,
        stats.gen_tokens,
        stats.finish_reason.as_str(),
        stats.tg_tps(),
        stats.decode_steps,
        stats.redone_fraction(),
    ));
    0
}
