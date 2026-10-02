use std::fmt;
use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug)]
pub enum ArgsError {
    MissingValue(String),
    Unknown(String),
    Invalid(String),
    Io(String),
}

impl fmt::Display for ArgsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingValue(name) => write!(formatter, "missing value for {name}"),
            Self::Unknown(argument) => write!(formatter, "unknown argument {argument}"),
            Self::Invalid(message) => formatter.write_str(message),
            Self::Io(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for ArgsError {}

/// Sampling defaults, in one place so the parser and the help text cannot
/// disagree about what a bare invocation does.
pub const DEFAULT_TEMPERATURE: f32 = 1.0;
pub const DEFAULT_TOP_K: usize = 64;
pub const DEFAULT_TOP_P: f32 = 0.95;
/// Generation budget when `--max-new-tokens` is not given.
///
/// 4096 fits well inside the 32768 context limit. It has to be: `fits_in_context`
/// reserves the *whole* budget up front, so a default equal to the limit would
/// make every prompt fail on the first turn (`used + need + max_new_tokens >
/// limit` is then true for any non-empty prompt). Half leaves room for the
/// prompt and the chat template's turn markers. 4096 is 1/8 of 32768, so
/// long generations still leave room for several turns of history.
pub const DEFAULT_MAX_NEW_TOKENS: usize = 4096;

#[derive(Debug, Clone)]
pub struct CliArgs {
    pub model_path: PathBuf,
    pub prompt: Option<String>,
    pub prompt_file: Option<PathBuf>,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: Option<u64>,
    pub show_ids: bool,
    pub show_metrics: bool,
    pub benchmark: bool,
    /// Interactive multi-turn chat: read lines from stdin, keep the KV cache
    /// across turns. Rejects `--prompt`/`--prompt-file`/`--benchmark`.
    pub chat: bool,
    /// Use the legacy host-slice forward path instead of the GPU-resident one.
    pub legacy_path: bool,
}

impl CliArgs {
    pub fn parse<I>(arguments: I) -> Result<Option<Self>, ArgsError>
    where
        I: IntoIterator<Item = String>,
    {
        let mut args = arguments.into_iter();
        let _program = args.next();
        let mut model_path = default_model_path();
        let mut prompt = None;
        let mut prompt_file = None;
        let mut max_new_tokens = DEFAULT_MAX_NEW_TOKENS;
        // Sampling defaults. The three are the usual llama.cpp-style trio and
        // are named in one place so the parser and the `--help` text cannot
        // drift: they are the numbers a user gets when they pass nothing, and a
        // help string that lies about them is worse than no help string.
        //
        // temperature 1.0 is full-distribution sampling, top_k 64 truncates the
        // tail, top_p 0.95 then trims within what is left. `--temperature 0`
        // is the one that changes the code path rather than just the values --
        // it selects greedy argmax in `ChatSession::new_greedy` -- so it is a
        // separate decision and not part of this default.
        let mut temperature: f32 = DEFAULT_TEMPERATURE;
        let mut top_k = DEFAULT_TOP_K;
        let mut top_p: f32 = DEFAULT_TOP_P;
        let mut seed = None;
        let mut show_ids = false;
        let mut show_metrics = false;
        let mut benchmark = false;
        let mut chat = false;
        let mut legacy_path = false;
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--help" | "-h" => return Ok(None),
                "--show-ids" => show_ids = true,
                "--metrics" => show_metrics = true,
                "--benchmark" => benchmark = true,
                "--chat" => chat = true,
                "--legacy-path" => legacy_path = true,
                "--model" => model_path = PathBuf::from(value(&mut args, "--model")?),
                "--prompt" => prompt = Some(value(&mut args, "--prompt")?),
                "--prompt-file" => {
                    prompt_file = Some(PathBuf::from(value(&mut args, "--prompt-file")?))
                }
                "--max-new-tokens" => {
                    max_new_tokens = value(&mut args, "--max-new-tokens")?
                        .parse()
                        .map_err(|_| ArgsError::Invalid("invalid --max-new-tokens".to_string()))?
                }
                "--temperature" => {
                    temperature = value(&mut args, "--temperature")?
                        .parse()
                        .map_err(|_| ArgsError::Invalid("invalid --temperature".to_string()))?
                }
                "--top-k" => {
                    top_k = value(&mut args, "--top-k")?
                        .parse()
                        .map_err(|_| ArgsError::Invalid("invalid --top-k".to_string()))?
                }
                "--top-p" => {
                    top_p = value(&mut args, "--top-p")?
                        .parse()
                        .map_err(|_| ArgsError::Invalid("invalid --top-p".to_string()))?
                }
                "--seed" => {
                    seed = Some(
                        value(&mut args, "--seed")?
                            .parse()
                            .map_err(|_| ArgsError::Invalid("invalid --seed".to_string()))?,
                    )
                }
                other => return Err(ArgsError::Unknown(other.to_string())),
            }
        }
        if prompt.is_some() && prompt_file.is_some() {
            return Err(ArgsError::Invalid(
                "choose only one of --prompt and --prompt-file".to_string(),
            ));
        }
        if chat && (prompt.is_some() || prompt_file.is_some() || benchmark) {
            return Err(ArgsError::Invalid(
                "--chat takes its prompts from stdin; drop --prompt, --prompt-file and --benchmark".to_string(),
            ));
        }
        if max_new_tokens == 0 {
            return Err(ArgsError::Invalid(
                "--max-new-tokens must be greater than zero".to_string(),
            ));
        }
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(ArgsError::Invalid(
                "--temperature must be finite and non-negative".to_string(),
            ));
        }
        if top_k == 0 {
            return Err(ArgsError::Invalid(
                "--top-k must be greater than zero".to_string(),
            ));
        }
        if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            return Err(ArgsError::Invalid("--top-p must be in (0, 1]".to_string()));
        }
        Ok(Some(Self {
            model_path,
            prompt,
            prompt_file,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            seed,
            show_ids,
            show_metrics,
            benchmark,
            chat,
            legacy_path,
        }))
    }

    pub fn read_prompt(&self) -> Result<String, ArgsError> {
        let prompt = if let Some(prompt) = &self.prompt {
            prompt.clone()
        } else if let Some(path) = &self.prompt_file {
            fs::read_to_string(path)
                .map_err(|error| ArgsError::Io(format!("{}: {error}", path.display())))?
        } else {
            let mut prompt = String::new();
            io::stdin()
                .read_to_string(&mut prompt)
                .map_err(|error| ArgsError::Io(error.to_string()))?;
            prompt
        };
        if prompt.trim().is_empty() {
            return Err(ArgsError::Invalid("prompt must not be empty".to_string()));
        }
        Ok(prompt)
    }

    pub fn seed_or_random(&self) -> u64 {
        self.seed.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos() as u64)
                .unwrap_or(0x9e3779b97f4a7c15)
        })
    }
}

/// Built with `format!` so the defaults in the text come from the constants the
/// parser uses. A hand-written string is how a help text ends up advertising
/// numbers the code stopped using.
pub fn help_text() -> String {
    format!(
        "Usage: kaplumba [--model PATH] [--prompt TEXT | --prompt-file PATH] [options]\n\n\
         Options:\n  \
         --prompt TEXT              Use TEXT as the user prompt\n  \
         --prompt-file PATH         Read the user prompt from PATH\n  \
         --model PATH               Model directory (default: ~/.kaplumba/models/Gxma/Gxma)\n  \
         --max-new-tokens N         Maximum generated tokens (default: {DEFAULT_MAX_NEW_TOKENS})\n  \
         --temperature VALUE        Sampling temperature; 0 selects greedy (default: {DEFAULT_TEMPERATURE})\n  \
         --top-k N                  Keep the top K logits (default: {DEFAULT_TOP_K})\n  \
         --top-p VALUE              Nucleus probability threshold (default: {DEFAULT_TOP_P})\n  \
         --seed N                   Deterministic sampling seed\n  \
         --show-ids                 Print diagnostic token IDs separately\n  \
         --metrics                  Print timing and memory metrics\n  \
         --benchmark                Run the fixed 2k/8-warmup/128-token benchmark\n  \
         --chat                     Interactive multi-turn chat on stdin (no --prompt)\n  \
         -h, --help                 Show this help\n"
    )
}

fn value<I>(arguments: &mut I, name: &str) -> Result<String, ArgsError>
where
    I: Iterator<Item = String>,
{
    arguments
        .next()
        .ok_or_else(|| ArgsError::MissingValue(name.to_string()))
}

fn default_model_path() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".kaplumba/models/Gxma/Gxma")
}

#[cfg(test)]
mod tests {
    use super::{
        ArgsError, CliArgs, DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_K,
        DEFAULT_TOP_P, help_text,
    };
    use kaplumba_engine::generate::DEFAULT_CONTEXT_LIMIT;

    fn args(values: &[&str]) -> Vec<String> {
        std::iter::once("kaplumba")
            .chain(values.iter().copied())
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn parses_sampling_and_prompt_options() {
        let parsed = CliArgs::parse(args(&[
            "--prompt",
            "hello",
            "--max-new-tokens",
            "12",
            "--temperature",
            "0",
            "--top-k",
            "8",
            "--top-p",
            "0.9",
            "--seed",
            "7",
            "--show-ids",
            "--metrics",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(parsed.prompt.as_deref(), Some("hello"));
        assert_eq!(parsed.max_new_tokens, 12);
        assert_eq!(parsed.temperature, 0.0);
        assert_eq!(parsed.top_k, 8);
        assert_eq!(parsed.top_p, 0.9);
        assert_eq!(parsed.seed, Some(7));
        assert!(parsed.show_ids);
        assert!(parsed.show_metrics);
    }

    #[test]
    fn rejects_conflicts_and_invalid_boundaries() {
        assert!(matches!(
            CliArgs::parse(args(&["--prompt", "a", "--prompt-file", "p"])),
            Err(ArgsError::Invalid(_))
        ));
        assert!(matches!(
            CliArgs::parse(args(&["--max-new-tokens", "0"])),
            Err(ArgsError::Invalid(_))
        ));
        assert!(matches!(
            CliArgs::parse(args(&["--temperature", "-1"])),
            Err(ArgsError::Invalid(_))
        ));
        assert!(matches!(
            CliArgs::parse(args(&["--top-k", "0"])),
            Err(ArgsError::Invalid(_))
        ));
        assert!(matches!(
            CliArgs::parse(args(&["--top-p", "1.1"])),
            Err(ArgsError::Invalid(_))
        ));
        assert!(matches!(
            CliArgs::parse(args(&["--unknown"])),
            Err(ArgsError::Unknown(_))
        ));
    }

    #[test]
    fn help_and_prompt_boundaries() {
        assert!(CliArgs::parse(args(&["--help"])).unwrap().is_none());
        let parsed = CliArgs::parse(args(&["--prompt", "a"])).unwrap().unwrap();
        assert_eq!(parsed.read_prompt().unwrap(), "a");
        let empty = CliArgs::parse(args(&["--prompt", " "])).unwrap().unwrap();
        assert!(matches!(empty.read_prompt(), Err(ArgsError::Invalid(_))));
    }

    /// The help text is generated from the same constants the parser uses, so
    /// it cannot advertise defaults the code stopped honouring. Asserted on all
    /// four, because the failure mode is silent: the text still renders, it
    /// just lies.
    #[test]
    fn help_text_reports_the_actual_defaults() {
        let text = help_text();
        assert!(
            text.contains(&format!("(default: {DEFAULT_MAX_NEW_TOKENS})")),
            "help text does not mention the max-new-tokens default of \
             {DEFAULT_MAX_NEW_TOKENS}: {text}"
        );
        assert!(
            text.contains(&format!("(default: {DEFAULT_TEMPERATURE})")),
            "help text does not mention the temperature default of \
             {DEFAULT_TEMPERATURE}: {text}"
        );
        assert!(
            text.contains(&format!("(default: {DEFAULT_TOP_K})")),
            "help text does not mention the top-k default of {DEFAULT_TOP_K}: {text}"
        );
        assert!(
            text.contains(&format!("(default: {DEFAULT_TOP_P})")),
            "help text does not mention the top-p default of {DEFAULT_TOP_P}: {text}"
        );
    }

    /// A bare invocation gets the documented sampling defaults. This is the
    /// property a user relies on when they type no flags at all.
    #[test]
    fn a_bare_invocation_uses_the_documented_defaults() {
        let parsed = CliArgs::parse(args(&[])).unwrap().unwrap();
        assert_eq!(parsed.temperature, DEFAULT_TEMPERATURE);
        assert_eq!(parsed.top_k, DEFAULT_TOP_K);
        assert_eq!(parsed.top_p, DEFAULT_TOP_P);
        assert_eq!(parsed.max_new_tokens, DEFAULT_MAX_NEW_TOKENS);
    }

    /// The default generation budget must leave room in the context for a
    /// prompt. `fits_in_context` reserves the whole budget up front, so a
    /// default at or above the 32768 context limit would reject every prompt --
    /// a failure at the first turn rather than at the limit.
    #[test]
    fn the_default_budget_leaves_room_for_a_prompt_in_the_context() {
        assert!(
            DEFAULT_MAX_NEW_TOKENS < DEFAULT_CONTEXT_LIMIT,
            "a default of {DEFAULT_MAX_NEW_TOKENS} against a context limit of \
             {DEFAULT_CONTEXT_LIMIT} cannot fit any prompt"
        );
        // A realistic first turn: a short prompt plus the chat template's
        // markers is well under 1k tokens.
        let prompt = 512;
        assert!(
            kaplumba_engine::chat::fits_in_context(
                0,
                prompt,
                DEFAULT_MAX_NEW_TOKENS,
                DEFAULT_CONTEXT_LIMIT,
            )
            .is_ok(),
            "a {prompt}-token first turn does not fit with the default budget"
        );
    }

    #[test]
    fn chat_flag_and_conflicts() {
        let parsed = CliArgs::parse(args(&["--chat"])).unwrap().unwrap();
        assert!(parsed.chat);
        assert!(matches!(
            CliArgs::parse(args(&["--chat", "--prompt", "a"])),
            Err(ArgsError::Invalid(_))
        ));
        assert!(matches!(
            CliArgs::parse(args(&["--chat", "--prompt-file", "p"])),
            Err(ArgsError::Invalid(_))
        ));
        assert!(matches!(
            CliArgs::parse(args(&["--chat", "--benchmark"])),
            Err(ArgsError::Invalid(_))
        ));
    }
}
