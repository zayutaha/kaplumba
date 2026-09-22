//! Hand-rolled CLI argument parsing.
//!
//! Deliberately dependency-free: this project has no clap/structopt, and the
//! flag surface is small enough that a tiny parser is cheaper than a new dep.

use std::path::PathBuf;

/// DE2 memory model of record — the roofline denominator.
pub const DEFAULT_WEIGHTS_GB: f64 = 15.5;
/// M4 Pro (20 GPU cores) unified-memory bandwidth.
pub const DEFAULT_BANDWIDTH_GBPS: f64 = 273.0;

/// Default artifact location (`KAPLUMBA_MODEL_DIR` overrides).
pub fn default_model_dir() -> PathBuf {
    if let Ok(d) = std::env::var("KAPLUMBA_MODEL_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(format!("{home}/.kaplumba/models/Gxma/Gxma"))
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerateArgs {
    pub model: PathBuf,
    pub prompt: Option<String>,
    pub prompt_file: Option<PathBuf>,
    pub max_tokens: usize,
    /// 0 → greedy.
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
    /// Debug/bench depth limit (decoder layers actually run).
    pub layers: Option<usize>,
    pub eos: Vec<u32>,
    /// Prepend the tokenizer's BOS id.
    pub bos: bool,
    /// Write the `kaplumba-bench/1` report here (`-` = stdout).
    pub json: Option<PathBuf>,
    /// Stream decoded text to stdout as it is produced.
    pub stream: bool,
    pub quiet: bool,
    /// Roofline inputs: weights (DE2 memory model of record) and bandwidth.
    pub weights_gb: f64,
    pub bandwidth_gbps: f64,
}

impl Default for GenerateArgs {
    fn default() -> Self {
        Self {
            model: default_model_dir(),
            prompt: None,
            prompt_file: None,
            max_tokens: 32,
            // OQ4/DE9 primary sampling mode.
            temp: 1.0,
            top_k: 64,
            top_p: 0.95,
            seed: 0,
            layers: None,
            eos: vec![1, 106],
            bos: true,
            json: None,
            stream: true,
            quiet: false,
            weights_gb: DEFAULT_WEIGHTS_GB,
            bandwidth_gbps: DEFAULT_BANDWIDTH_GBPS,
        }
    }
}

impl GenerateArgs {
    /// JSON report goes to stdout → text must not pollute it.
    pub fn json_to_stdout(&self) -> bool {
        matches!(&self.json, Some(p) if p.as_os_str() == "-")
    }
}

pub fn usage() -> &'static str {
    "kaplumba-engine — pure Rust + raw Metal Gemma 4 EXL3 engine\n\
     \n\
     usage: kaplumba-engine <command> [options]\n\
     \n\
     commands:\n\
     \x20 metal-smoke            GPU fill-buffer smoke test\n\
     \x20 generate [options]     stream text from a prompt and report bench stats\n\
     \n\
     generate options:\n\
     \x20 --model DIR            artifact dir (default $KAPLUMBA_MODEL_DIR or ~/.kaplumba/models/Gxma/Gxma)\n\
     \x20 --prompt TEXT          prompt text\n\
     \x20 --prompt-file PATH     prompt text from file\n\
     \x20 --max-tokens N         max new tokens (default 32)\n\
     \x20 --temp F               0 = greedy (default 1.0)\n\
     \x20 --top-k N              top-k cutoff, 0 = off (default 64)\n\
     \x20 --top-p F              nucleus cutoff, 1.0 = off (default 0.95)\n\
     \x20 --seed N               sampler seed (default 0)\n\
     \x20 --layers N             only run the first N decoder layers (debug/bench)\n\
     \x20 --eos ID,ID            stop ids (default 1,106)\n\
     \x20 --no-bos               do not prepend the BOS token\n\
     \x20 --json PATH            write the kaplumba-bench/1 report ('-' = stdout)\n\
     \x20 --no-stream            do not stream decoded text to stdout\n\
     \x20 --quiet                suppress progress lines on stderr\n\
     \x20 --weights-gb F         roofline weights (default 15.5)\n\
     \x20 --bandwidth-gbps F     roofline bandwidth (default 273.0)\n"
}

/// Parse `generate` arguments (excluding the command word itself).
pub fn parse_generate(args: &[String]) -> Result<GenerateArgs, String> {
    let mut out = GenerateArgs::default();
    let mut i = 0usize;
    while i < args.len() {
        let raw = args[i].as_str();
        let (flag, inline) = match raw.split_once('=') {
            Some((f, v)) => (f, Some(v.to_string())),
            None => (raw, None),
        };
        let mut take_value = |name: &str| -> Result<String, String> {
            if let Some(v) = inline.clone() {
                return Ok(v);
            }
            i += 1;
            args.get(i)
                .cloned()
                .ok_or_else(|| format!("{name} requires a value"))
        };
        match flag {
            "--model" => out.model = PathBuf::from(take_value("--model")?),
            "--prompt" => out.prompt = Some(take_value("--prompt")?),
            "--prompt-file" => out.prompt_file = Some(PathBuf::from(take_value("--prompt-file")?)),
            "--max-tokens" => {
                out.max_tokens = parse_num("--max-tokens", &take_value("--max-tokens")?)?
            }
            "--temp" => out.temp = parse_num("--temp", &take_value("--temp")?)?,
            "--top-k" => out.top_k = parse_num("--top-k", &take_value("--top-k")?)?,
            "--top-p" => out.top_p = parse_num("--top-p", &take_value("--top-p")?)?,
            "--seed" => out.seed = parse_num("--seed", &take_value("--seed")?)?,
            "--layers" => out.layers = Some(parse_num("--layers", &take_value("--layers")?)?),
            "--eos" => {
                let v = take_value("--eos")?;
                out.eos = v
                    .split(',')
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| {
                        s.trim()
                            .parse::<u32>()
                            .map_err(|e| format!("--eos: bad id {s:?}: {e}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--weights-gb" => {
                out.weights_gb = parse_num("--weights-gb", &take_value("--weights-gb")?)?
            }
            "--bandwidth-gbps" => {
                out.bandwidth_gbps =
                    parse_num("--bandwidth-gbps", &take_value("--bandwidth-gbps")?)?
            }
            "--json" => out.json = Some(PathBuf::from(take_value("--json")?)),
            "--no-bos" => out.bos = false,
            "--no-stream" => out.stream = false,
            "--quiet" => out.quiet = true,
            other => return Err(format!("unknown flag: {other}")),
        }
        i += 1;
    }

    if out.prompt.is_some() && out.prompt_file.is_some() {
        return Err("--prompt and --prompt-file are mutually exclusive".into());
    }
    if out.temp < 0.0 {
        return Err("--temp must be >= 0".into());
    }
    if out.top_p < 0.0 {
        return Err("--top-p must be >= 0".into());
    }
    if out.weights_gb <= 0.0 || out.bandwidth_gbps <= 0.0 {
        return Err("--weights-gb and --bandwidth-gbps must be > 0".into());
    }
    Ok(out)
}

fn parse_num<T: std::str::FromStr>(flag: &str, raw: &str) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    raw.trim()
        .parse::<T>()
        .map_err(|e| format!("{flag}: bad value {raw:?}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn defaults_match_the_spec_sampling_mode() {
        let a = parse_generate(&[]).expect("defaults");
        assert_eq!(a.max_tokens, 32);
        assert_eq!(a.temp, 1.0);
        assert_eq!(a.top_k, 64);
        assert_eq!(a.top_p, 0.95);
        assert_eq!(a.eos, vec![1, 106]);
        assert!(a.bos && a.stream && !a.quiet);
        assert_eq!(a.layers, None);
        assert_eq!(a.weights_gb, 15.5);
        assert_eq!(a.bandwidth_gbps, 273.0);
        assert!(!a.json_to_stdout());
    }

    #[test]
    fn parses_every_flag_in_both_forms() {
        let a = parse_generate(&v(&[
            "--model",
            "/tmp/m",
            "--prompt=hello world",
            "--max-tokens",
            "7",
            "--temp=0",
            "--top-k",
            "1",
            "--top-p",
            "0.5",
            "--seed=42",
            "--layers",
            "2",
            "--eos",
            "1, 5,106",
            "--no-bos",
            "--no-stream",
            "--quiet",
            "--weights-gb",
            "16.5",
            "--bandwidth-gbps=300",
            "--json",
            "-",
        ]))
        .expect("parse");
        assert_eq!(a.model, PathBuf::from("/tmp/m"));
        assert_eq!(a.prompt.as_deref(), Some("hello world"));
        assert_eq!(a.max_tokens, 7);
        assert_eq!(a.temp, 0.0);
        assert_eq!(a.top_k, 1);
        assert_eq!(a.top_p, 0.5);
        assert_eq!(a.seed, 42);
        assert_eq!(a.layers, Some(2));
        assert_eq!(a.eos, vec![1, 5, 106]);
        assert!(!a.bos && !a.stream && a.quiet);
        assert_eq!(a.weights_gb, 16.5);
        assert_eq!(a.bandwidth_gbps, 300.0);
        assert!(a.json_to_stdout(), "- means stdout");
    }

    #[test]
    fn rejects_bad_input() {
        assert!(
            parse_generate(&v(&["--nope"]))
                .unwrap_err()
                .contains("unknown flag")
        );
        assert!(
            parse_generate(&v(&["--max-tokens"]))
                .unwrap_err()
                .contains("requires a value")
        );
        assert!(parse_generate(&v(&["--max-tokens", "x"])).is_err());
        assert!(parse_generate(&v(&["--eos", "1,x"])).is_err());
        assert!(parse_generate(&v(&["--temp", "-1"])).is_err());
        assert!(parse_generate(&v(&["--weights-gb", "0"])).is_err());
        assert!(
            parse_generate(&v(&["--prompt", "a", "--prompt-file", "b"]))
                .unwrap_err()
                .contains("mutually exclusive")
        );
        // Negative numbers are parsed as flag values, not as flags.
        assert_eq!(parse_generate(&v(&["--top-p", "0"])).unwrap().top_p, 0.0);
        assert!(parse_generate(&v(&["--prompt-file"])).is_err());
    }

    #[test]
    fn prompt_file_and_prompt_are_optional() {
        let a = parse_generate(&v(&["--prompt-file", "/tmp/p.txt"])).unwrap();
        assert_eq!(a.prompt_file, Some(PathBuf::from("/tmp/p.txt")));
        assert!(a.prompt.is_none());
    }

    #[test]
    fn usage_lists_every_flag_the_parser_accepts() {
        let u = usage();
        for flag in [
            "--model",
            "--prompt",
            "--prompt-file",
            "--max-tokens",
            "--temp",
            "--top-k",
            "--top-p",
            "--seed",
            "--layers",
            "--eos",
            "--no-bos",
            "--json",
            "--no-stream",
            "--quiet",
            "--weights-gb",
            "--bandwidth-gbps",
            "metal-smoke",
            "generate",
        ] {
            assert!(u.contains(flag), "usage is missing {flag}");
        }
    }

    #[test]
    fn default_model_dir_prefers_env() {
        // Only asserts the shape; env may or may not be set in CI.
        let d = default_model_dir();
        assert!(
            d.to_string_lossy().contains("models") || std::env::var("KAPLUMBA_MODEL_DIR").is_ok()
        );
    }
}
