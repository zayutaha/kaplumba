use std::io::{self, BufRead, Write};

use kaplumba_engine::chat::{ChatRole, ChatSession};
use kaplumba_engine::config::load_generation_config;
use kaplumba_engine::generate::GenerationConfig;
use kaplumba_engine::model::TextModel;
use kaplumba_engine::sampler::SamplerConfig;
use kaplumba_engine::tokenizer::GemmaTokenizer;

use crate::args::CliArgs;

const HELP: &str = "commands: /help /exit (/quit) /reset /history /stats";

fn print_help() {
    println!("{HELP}");
    println!("  /reset    forget the conversation and start over");
    println!("  /history  show the conversation so far, with token counts");
    println!("  /stats    context used / limit, turns, last-turn and session tok/s");
}

pub fn run(
    arguments: &CliArgs,
    model: &mut TextModel,
    tokenizer: &GemmaTokenizer,
    load_ms: f64,
) -> Result<(), String> {
    let checkpoint_generation =
        load_generation_config(&arguments.model_path.join("generation_config.json"))
            .map_err(|error| error.to_string())?;
    let sampler = SamplerConfig {
        temperature: arguments.temperature,
        top_k: arguments.top_k,
        top_p: arguments.top_p,
    };
    let mut session = if arguments.temperature == 0.0 {
        ChatSession::new_greedy(
            model,
            tokenizer,
            sampler,
            arguments.seed_or_random(),
            arguments.max_new_tokens,
            checkpoint_generation.eos_token_ids,
            GenerationConfig::default().context_limit,
        )
    } else {
        ChatSession::new(
            model,
            tokenizer,
            sampler,
            arguments.seed_or_random(),
            arguments.max_new_tokens,
            checkpoint_generation.eos_token_ids,
            GenerationConfig::default().context_limit,
        )
    }
    .map_err(|error| error.to_string())?;
    println!(
        "kaplumba chat (model loaded in {load_ms:.0} ms, {}k context). {HELP}",
        session.context_limit() / 1024
    );
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        print!("> ");
        io::stdout().flush().map_err(|error| error.to_string())?;
        let line = match lines.next() {
            None => break, // EOF (Ctrl-D) exits cleanly.
            Some(line) => line.map_err(|error| error.to_string())?,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(command) = trimmed.strip_prefix('/') {
            match command.to_ascii_lowercase().as_str() {
                "help" => print_help(),
                "exit" | "quit" => {
                    break;
                }
                "reset" => {
                    session.reset();
                    println!("conversation reset.");
                }
                "history" => print_history(&session),
                "stats" => print_stats(&session),
                _ => {
                    println!("unknown command /{command}. {HELP}");
                }
            }
            continue;
        }
        let stream = |chunk: &str| {
            print!("{chunk}");
            io::stdout().flush().map_err(|error| error.to_string())
        };
        // A failed turn returns to the prompt; only a failed load is fatal,
        // and the load already happened above.
        match session.turn(&line, stream) {
            Ok(_) => {
                println!();
                // Per turn, not only at exit. A tok/s figure you can only see by
                // quitting is no use while you are deciding whether to keep
                // chatting -- and the session rate is the number worth watching,
                // because early turns are short and unrepresentative.
                print_turn_rate(&session);
            }
            Err(error) => println!("error: {error}"),
        }
    }
    // Session summary, on every exit path: `/exit`, `/quit`, and Ctrl-D all
    // land here, because they all leave the loop by `break`.
    print_session_totals(&session);
    Ok(())
}

/// One line after each turn: the last turn's rate and the session's.
fn print_turn_rate(session: &ChatSession) {
    let Some(stats) = session.last_stats() else {
        return;
    };
    let totals = session.totals();
    println!(
        "[{} tok/s last turn, {:.2} tok/s session, {} turns, {} ctx]",
        stats.decode_tok_s, totals.decode_tok_s, totals.turns, session.context_used()
    );
}

/// The end-of-chat summary.
///
/// `decode_tok_s` is the token-weighted rate over the whole session, which is
/// what makes it comparable to `--metrics`'s `decode_tok_s` on a single
/// generation. Both exclude prefill and the model load, deliberately: prefill
/// cost depends on prompt length and the load is a one-off, so folding either in
/// would make the number mostly a measure of how the session was started rather
/// than of decode speed.
fn print_session_totals(session: &ChatSession) {
    let totals = session.totals();
    if totals.turns == 0 {
        return;
    }
    println!(
        "{} turns, {} prefill + {} generated tokens in {:.1} s decode",
        totals.turns, totals.prefill_tokens, totals.generated_tokens, totals.decode_s
    );
    println!("decode: {:.2} tok/s (session average)", totals.decode_tok_s);
}

fn print_history(session: &ChatSession) {
    if session.messages().is_empty() {
        println!("(empty conversation)");
        return;
    }
    for message in session.messages() {
        let role = match message.role {
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        };
        println!("--- {role} ---");
        println!("{}", message.text);
    }
    println!(
        "({} turns, {} context tokens)",
        session.turns(),
        session.context_used()
    );
}

fn print_stats(session: &ChatSession) {
    println!(
        "context: {} / {} tokens, turns: {}",
        session.context_used(),
        session.context_limit(),
        session.turns()
    );
    match session.last_stats() {
        Some(stats) => println!(
            "last turn: {} prefill + {} generated tokens, {:.2} tok/s",
            stats.prefill_tokens, stats.generated_tokens, stats.decode_tok_s
        ),
        None => println!("no turns yet"),
    }
    let totals = session.totals();
    if totals.turns > 0 {
        println!(
            "session:   {} turns, {} generated tokens, {:.2} tok/s",
            totals.turns, totals.generated_tokens, totals.decode_tok_s
        );
    }
}
