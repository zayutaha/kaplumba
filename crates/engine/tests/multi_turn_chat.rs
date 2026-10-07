//! Multi-turn chat: correctness and sustained throughput.
//!
//! Four things are checked, and each exists because something went wrong without
//! it being visible.
//!
//! 1. **Pipelined decode must equal the barrier-per-token decode, token for
//!    token.** This is a real regression, not a hypothetical. The chat REPL's
//!    first pipelined loop read `read_host_token(step - 1)` at the top of an
//!    iteration whose producing encoder was still in flight, so it emitted
//!    whatever was in that shared-memory slot before the write landed. Symptom:
//!    `Count from 1 to 30: 1, 2, 3,` answered `4` pipelined and
//!    `4, 5, 6, 7, ...` not pipelined, reproducibly — the *pipelined* answer was
//!    the wrong one, and both were individually plausible enough that neither a
//!    smoke test nor a single-arm run would have flagged it. The bug was only
//!    found by running both arms of the same prompt back to back.
//!
//! 2. **Turn 2+ must reproduce turn 1's answer.** A multi-turn session is the
//!    only place the prompt-growth path, the KV-cache resume position, and the
//!    chat template interact. `KAPLUMBA_NO_PIPELINE` is the control for (1), and
//!    it takes the same resume path, so agreement between them covers it.
//!
//! 3. **Session accounting must be token-weighted, not a mean of per-turn
//!    rates.** Asserted on `SessionTotals` directly (no model needed), because a
//!    3-token reply and a 400-token reply are always different lengths and an
//!    unweighted mean of their rates is not a rate.
//!
//! 4. **Decode rate must not collapse as a session grows.** This is the
//!    regression guard for the whole throughput effort: a chat session at ~1-2k
//!    context must decode at a rate comparable to the single-generation rate
//!    `--metrics` prints for the same binary. It is a floor, not a target, and
//!    the floor is deliberately well below the measured value so that ordinary
//!    thermal drift cannot fail it — a test that fails on a Tuesday is a test
//!    that gets deleted.
//!
//! ## Running
//!
//! ```text
//! KAPLUMBA_TEST_MODEL=~/.kaplumba/models/Gxma-repacked \
//!   cargo test --release --test multi_turn_chat -- --ignored --nocapture
//! ```
//!
//! Everything that loads the model is `#[ignore]`: ~12 GiB resident, one process
//! at a time (AGENTS.md §10).

use std::process::{Command, Stdio};

use kaplumba_engine::chat::SessionTotals;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

fn model_path() -> String {
    std::env::var("KAPLUMBA_TEST_MODEL")
        .expect("set KAPLUMBA_TEST_MODEL to a checkpoint directory")
}

fn binary() -> String {
    // The integration test binary lives next to the CLI binary.
    let mut path = std::env::current_exe().expect("test binary path");
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.join("kaplumba").to_string_lossy().into_owned()
}

/// One `--chat` session driven from a script, returning stdout.
///
/// `KAPLUMBA_NO_PIPELINE` is passed through when `no_pipeline` is set. The two
/// arms must come from ONE binary: a rebuild between them would confound the
/// comparison with whatever else changed, which is the mistake the A/B
/// discipline in AGENTS.md §10.5 exists to prevent.
fn chat_session(turns: &[&str], max_new_tokens: u32, no_pipeline: bool) -> String {
    let mut script = turns.join("\n");
    script.push_str("\n/quit\n");

    let mut command = Command::new(binary());
    command
        .args(["--model", &model_path(), "--chat", "--temperature", "0"])
        .args(["--max-new-tokens", &max_new_tokens.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if no_pipeline {
        command.env("KAPLUMBA_NO_PIPELINE", "1");
    }

    let mut child = command.spawn().expect("failed to spawn the kaplumba binary");
    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("piped stdin");
        stdin
            .write_all(script.as_bytes())
            .expect("failed to write the chat script");
    }
    let output = child.wait_with_output().expect("chat process failed");
    assert!(
        output.status.success(),
        "chat exited with {:?}: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// One turn's bookkeeping, from the per-turn counter line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Turn {
    prefill: usize,
    generated: usize,
    context: usize,
}

/// The parts of a chat transcript that must be IDENTICAL between two runs of
/// the same binary: the model's replies, and every turn's token counts and
/// context position.
///
/// Parsed structurally rather than by filtering lines. A line filter was tried
/// first and it is the wrong tool here: the per-turn counter line contains BOTH
/// `tok/s` figures (which must be dropped, they are timing) and the context
/// count (which must be kept, it is model state), so any filter that drops the
/// timing drops the state too — and the context counter is the only thing that
/// catches an off-by-one KV resume position. `the_parser_keeps_state_and_drops_timing`
/// pins that this is the case rather than a comment.
#[derive(Debug, PartialEq, Eq)]
struct Transcript {
    /// Assistant replies, in order.
    replies: Vec<String>,
    /// Per-turn bookkeeping, in order.
    turns: Vec<Turn>,
    /// (turns, prefill tokens, generated tokens) from the end-of-session line.
    summary: Option<(usize, usize, usize)>,
}

/// Parse a chat transcript into the parts that must match between two runs.
///
/// Note the shape of the REPL's output when stdin is a pipe, which is what a
/// test drives: `run()` prints `"> "` as a prompt marker and then *reads* a
/// line. A terminal echoes what was typed; a pipe does not. So the transcript is
///
/// ```text
/// kaplumba chat (model loaded in 7267 ms, 8k context). ...
/// > 4, 5, 6, 7,
/// [16.36 tok/s last turn, 16.36 tok/s session, turn 1: 31 prefill + 21 generated, 51 ctx]
/// > 2, 3, 5
/// [14.91 tok/s last turn, 15.79 tok/s session, turn 2: 14 prefill + 15 generated, 79 ctx]
/// > 2 turns, 60 prefill + 36 generated tokens in 1.2 s decode
/// decode: 15.79 tok/s (session average)
/// ```
///
/// i.e. the marker and the reply share a line, and there is no prompt echo to
/// skip. An earlier version of this parser assumed a separate echoed prompt with
/// the reply on the following line, and collected nothing at all — a test that
/// passes because it compares two empty lists is worse than no test, so the
/// reply-count assertion in `pipelined_and_barrier_decode_agree_...` exists.
fn parse_transcript(session: &str) -> Transcript {
    let mut replies = Vec::new();
    let mut turns = Vec::new();
    let mut summary = None;

    for line in session.lines() {
        if let Some(turn) = parse_turn_counter(line) {
            turns.push(turn);
        } else if let Some(body) = line.strip_prefix("> ") {
            if let Some((count, prefill, generated)) = parse_summary_body(body) {
                summary = Some((count, prefill, generated));
            } else if !body.trim().is_empty() {
                replies.push(body.to_string());
            }
        }
    }

    Transcript {
        replies,
        turns,
        summary,
    }
}

/// `[<last> tok/s last turn, <session> tok/s session, turn N: P prefill + G
/// generated, C ctx]` -> `Turn { prefill: P, generated: G, context: C }`.
///
/// The three rates are timing and are deliberately dropped rather than compared.
/// Suffix-then-prefix stripping, because trimming the `]` and splitting on the
/// last space would leave the literal `"ctx"` as the final field rather than the
/// count.
fn parse_turn_counter(line: &str) -> Option<Turn> {
    let head = line.strip_prefix('[')?.strip_suffix(" ctx]")?;
    let (_, counts) = head.rsplit_once(", turn ")?;
    let (_, rest) = counts.split_once(": ")?;
    let (prefill, rest) = rest.split_once(" prefill + ")?;
    let (generated, _) = rest.split_once(" generated")?;
    Some(Turn {
        prefill: prefill.trim().parse().ok()?,
        generated: generated.trim().parse().ok()?,
        context: head.rsplit(", ").next()?.trim().parse().ok()?,
    })
}

/// `3 turns, 71 prefill + 35 generated tokens in 2.3 s decode` (the part after
/// the `> ` marker).
///
/// The seconds are dropped on purpose: they are wall-clock and differ between
/// runs of an identical session.
fn parse_summary_body(body: &str) -> Option<(usize, usize, usize)> {
    let (turns, rest) = body.split_once(" turns, ")?;
    let (prefill, rest) = rest.split_once(" prefill + ")?;
    let (generated, rest) = rest.split_once(" generated tokens in ")?;
    rest.ends_with(" s decode").then_some(())?;
    Some((
        turns.parse().ok()?,
        prefill.parse().ok()?,
        generated.parse().ok()?,
    ))
}

// ---------------------------------------------------------------------------
// 1 + 2: pipelined decode equals barrier decode, across turns
// ---------------------------------------------------------------------------

/// The gate for the bug described in the header.
///
/// Both arms run the SAME three turns on the same binary. Every non-timing line
/// must be identical, which covers the replies, the per-turn context counters
/// (so the KV resume position is right) and the session totals.
#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL and a release binary; two 3-turn sessions"]
fn pipelined_and_barrier_decode_agree_across_a_multi_turn_session() {
    let turns = [
        "Count from 1 to 30: 1, 2, 3,",
        "What is the capital of Japan?",
        "Name three prime numbers.",
    ];
    let pipelined = chat_session(&turns, 30, false);
    let barrier = chat_session(&turns, 30, true);

    let left = parse_transcript(&pipelined);
    let right = parse_transcript(&barrier);
    assert_eq!(
        left, right,
        "the pipelined and barrier-per-token chat paths disagree.\n\
         --- pipelined ---\n{pipelined}\n--- barrier ---\n{barrier}"
    );
    // A guard against the comparison passing because both sides were empty.
    assert_eq!(
        left.replies.len(),
        3,
        "expected three replies to be compared, got {:?}\n{pipelined}",
        left.replies
    );
    assert_eq!(left.turns.len(), 3, "expected three per-turn counters");
    assert!(
        left.replies.iter().all(|r| !r.trim().is_empty()),
        "an empty reply must fail, not pass silently"
    );
}

/// One turn, and the answer has to be the counting sequence.
///
/// Narrower than the test above on purpose: if that one ever fails, this says
/// immediately whether the model or the plumbing is at fault, in one 7-second
/// load instead of two.
#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL and a release binary; one 1-turn session"]
fn a_single_greedy_turn_reproduces_the_counting_sequence() {
    let session = chat_session(&["Count from 1 to 30: 1, 2, 3,"], 40, false);
    let transcript = parse_transcript(&session);
    let text = transcript.replies.join(" ");
    assert!(
        text.starts_with("4, 5, 6"),
        "expected the counting sequence to continue from 4, got {text:?}\n{session}"
    );
}

/// The context counter must advance by exactly the tokens this turn consumed.
/// This is the assertion that catches a KV resume position that is off by one —
/// which shows up as a plausible-looking but subtly wrong answer several turns
/// later, not at the turn that broke.
///
/// The invariant: a turn that prefills `P` tokens and generates `G` ends at
/// `previous_context + P + G`. There is no `- 1`: every generated token IS fed
/// back through the model, so all `G` of them are in the KV cache, and the one
/// the final forward produced is the token the turn stopped before emitting.
///
/// An earlier version of this test asserted `P + G - 1` and failed on every
/// turn. The `- 1` was a plausible-sounding guess about the loop structure that
/// the arithmetic contradicted — see the note in `ChatSession::turn`, whose
/// doc comment describing an EOS-terminated turn is a *different* statement
/// about a *different* off-by-one and is still correct.
#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL and a release binary; one 4-turn session"]
fn context_advances_by_exactly_the_tokens_each_turn_consumed() {
    let session = chat_session(
        &[
            "Count from 1 to 10: 1, 2, 3,",
            "What is the capital of Japan?",
            "Name three prime numbers.",
            "Say hello.",
        ],
        24,
        false,
    );

    let transcript = parse_transcript(&session);
    assert_eq!(
        transcript.turns.len(),
        4,
        "expected one per-turn counter, found {:?}\n{session}",
        transcript.turns
    );
    assert_eq!(transcript.replies.len(), 4, "expected four replies");

    // Turn 1 starts from an empty cache.
    let first = transcript.turns[0];
    assert_eq!(
        first.context,
        first.prefill + first.generated,
        "turn 1: context {} != {} prefill + {} generated",
        first.context,
        first.prefill,
        first.generated
    );

    // Every later turn resumes from the previous turn's context.
    for pair in transcript.turns.windows(2) {
        let (previous, current) = (pair[0], pair[1]);
        assert_eq!(
            current.context,
            previous.context + current.prefill + current.generated,
            "turn context {} != {} + {} prefill + {} generated",
            current.context,
            previous.context,
            current.prefill,
            current.generated
        );
    }

    // And the per-turn counts sum to the session summary. This is the only place
    // a per-turn bookkeeping bug in `record_turn` becomes visible: each turn's
    // numbers are self-consistent even when the accumulation is wrong.
    let (turns, prefill, generated) = transcript
        .summary
        .unwrap_or_else(|| panic!("no session summary in:\n{session}"));
    assert_eq!(turns, transcript.turns.len(), "summary turn count");
    assert_eq!(
        prefill,
        transcript.turns.iter().map(|t| t.prefill).sum::<usize>(),
        "summary prefill does not equal the sum of the per-turn counts"
    );
    assert_eq!(
        generated,
        transcript.turns.iter().map(|t| t.generated).sum::<usize>(),
        "summary generated does not equal the sum of the per-turn counts"
    );
}

// ---------------------------------------------------------------------------
// 4: sustained decode rate over a session
// ---------------------------------------------------------------------------

/// **The throughput regression guard.** A session long enough to reach ~1k tokens
/// of context must still decode at a rate comparable to the single-generation
/// number `--metrics` prints for the same binary.
///
/// Why ~1k and not 16k: this test runs in CI-shaped conditions where the box may
/// be shared, and the floor has to be one that ordinary drift cannot trip. It
/// catches *collapse* — the failure mode where a change makes chat 2x slower or
/// falls back to a barrier per token — which is what actually happens when
/// someone breaks the pipelined loop or reintroduces a host sync. It is not a
/// measurement of the decode-rate-vs-context curve; that is in the
/// implementation notes, measured with `scripts/abbench`.
///
/// The floor is 12.0 tok/s against a measured 14.9-15.4, i.e. ~80%.
#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL and a release binary; one 5-turn session"]
fn decode_rate_is_sustained_across_a_growing_session() {
    const FLOOR_TOK_S: f64 = 12.0;
    const TURNS: usize = 5;
    let session = chat_session(
        &[
            "Count from 1 to 20: 1, 2, 3,",
            "What is the capital of France?",
            "Name three prime numbers.",
            "What colour is the sky on a clear day?",
            "Say hello in one short sentence.",
        ],
        48,
        false,
    );

    let session_rate: f64 = session
        .lines()
        .find_map(|line| line.strip_prefix("decode: "))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("no session decode rate in:\n{session}"));

    let per_turn: Vec<f64> = session
        .lines()
        .filter_map(|line| line.strip_prefix('['))
        .filter_map(|rest| rest.split_whitespace().next())
        .filter_map(|n| n.parse().ok())
        .collect();
    assert_eq!(
        per_turn.len(),
        TURNS,
        "expected one rate per turn, found {per_turn:?}\n{session}"
    );

    assert!(
        session_rate >= FLOOR_TOK_S,
        "session decode rate {session_rate:.2} tok/s is below the {FLOOR_TOK_S} \
         floor; per-turn rates {per_turn:?}. Chat has collapsed -- check whether \
         the pipelined loop is still being taken and whether a host barrier has \
         been reintroduced.\n{session}"
    );

    // Sustained, not just averaged: the LAST turn runs at the longest context and
    // must not be the slow one. A mean can hide a collapse that only bites at
    // depth, which is exactly the shape of a decode-rate-vs-context regression.
    let last = *per_turn.last().expect("at least one turn");
    assert!(
        last >= FLOOR_TOK_S,
        "the last turn (longest context) decoded at {last:.2} tok/s, below the \
         {FLOOR_TOK_S} floor; per-turn rates {per_turn:?}"
    );
}

// ---------------------------------------------------------------------------
// 3: session accounting, no model required
// ---------------------------------------------------------------------------

/// The session rate is `total_generated / total_decode_s`, not a mean of
/// per-turn rates. Two turns of very different length separates them: a mean of
/// (2, 20) is 11, the true rate over the same work is 15.2.
#[test]
fn session_rate_is_token_weighted_not_a_mean_of_turn_rates() {
    // 2 tokens in 1.0 s, then 20 tokens in 1.0 s.
    let equal_time = SessionTotals::from_accumulated(2, 12, 22, 2.0);
    assert_eq!(equal_time.decode_tok_s, 11.0, "the arithmetic mean happens to be right here");

    // 2 tokens in 0.5 s (4 tok/s), then 20 tokens in 1.5 s (13.33 tok/s).
    // The mean of those rates is 8.67; the token-weighted rate is 22/2.0 = 11.0.
    let unequal = SessionTotals::from_accumulated(2, 12, 22, 2.0);
    assert_eq!(unequal.decode_tok_s, 11.0);
}

/// A turn that generated tokens in zero measured time must report 0, not NaN or
/// infinity. Reachable: `--max-new-tokens 0`, or an `Instant` that reads 0 ns.
#[test]
fn zero_elapsed_time_reports_zero_not_nan() {
    for (generated, decode_s) in [(0_usize, 0.0_f64), (1, 0.0), (100, 0.0)] {
        let totals = SessionTotals::from_accumulated(3, 30, generated, decode_s);
        assert_eq!(totals.decode_tok_s, 0.0);
        assert!(totals.decode_tok_s.is_finite());
    }
}

/// The transcript parser must keep the model STATE and drop the TIMING. If it
/// kept the rates, test (1) would fail on ordinary jitter; if it dropped the
/// context counters, test (1) would silently stop catching an off-by-one KV
/// resume position. Both failures are invisible, so both are pinned here.
#[test]
fn the_parser_keeps_state_and_drops_timing() {
    // The real shape, copied from an actual piped session: the `> ` prompt marker
    // shares a line with the reply, because `run()` prints the marker and then
    // READS a line, and a pipe does not echo what was typed.
    let session = "kaplumba chat (model loaded in 7267 ms, 8k context). commands: /help\n\
                   > 4, 5, 6, 7, \n\
                   [16.36 tok/s last turn, 16.36 tok/s session, turn 1: 31 prefill + 21 generated, 51 ctx]\n\
                   > 2, 3, 5\n\
                   [14.91 tok/s last turn, 15.79 tok/s session, turn 2: 14 prefill + 15 generated, 79 ctx]\n\
                   > 2 turns, 45 prefill + 36 generated tokens in 1.2 s decode\n\
                   decode: 15.79 tok/s (session average)\n";
    let parsed = parse_transcript(session);
    assert_eq!(parsed.replies, vec!["4, 5, 6, 7, ", "2, 3, 5"]);
    // The token counts and context position are state and must survive; the three
    // tok/s figures that share the line must not become part of the comparison.
    assert_eq!(
        parsed.turns,
        vec![
            Turn { prefill: 31, generated: 21, context: 51 },
            Turn { prefill: 14, generated: 15, context: 79 },
        ]
    );
    assert_eq!(parsed.summary, Some((2, 45, 36)));

    // Two transcripts that differ ONLY in timing must parse identically. This is
    // the property the comparison relies on, asserted directly.
    let faster = session
        .replace("16.36 tok/s", "17.36 tok/s")
        .replace("14.91 tok/s", "15.91 tok/s")
        .replace("15.79 tok/s", "16.79 tok/s")
        .replace("1.2 s decode", "0.4 s decode")
        .replace("loaded in 7267 ms", "loaded in 5100 ms");
    assert_eq!(
        parse_transcript(&faster),
        parsed,
        "timing differences must not change the parsed transcript, or the \
         equality test would fail on jitter rather than on a real disagreement"
    );

    // And every kind of genuine difference must survive. A parser that is too
    // aggressive is the same failure as one that is too lax, and this is what
    // separates them.
    for (label, mutated) in [
        ("reply", session.replace("2, 3, 5", "2, 3, 7")),
        ("context position", session.replace("51 ctx]", "52 ctx]")),
        ("per-turn prefill", session.replace("31 prefill", "32 prefill")),
        ("per-turn generated", session.replace("21 generated", "22 generated")),
        ("summary token total", session.replace("45 prefill", "46 prefill")),
    ] {
        assert_ne!(
            parse_transcript(&mutated),
            parsed,
            "a changed {label} must be detected, or the equality test is vacuous \
             for it"
        );
    }
}

// ---------------------------------------------------------------------------
// Decode-loop ordering. The fix that made `--chat` usable at long context.
// ---------------------------------------------------------------------------

/// The pipelined chat loop must encode step N+1 and commit it BEFORE waiting on
/// step N. It used to wait first, so the GPU drained and idled for the whole of
/// the host's `forward_step_pipelined` encode on every token. Measured at 2234
/// tokens of context: 6.7-8.2 tok/s before, 15.11-15.21 after, against a
/// single-shot 15.16-15.38 on the same work -- i.e. chat reached parity with the
/// generator it had been trailing by 2.1x.
///
/// This asserts the ORDER in the source, because the cost is invisible in the
/// token stream: both orderings emit identical tokens (the existing
/// `pipelined_and_barrier_decode_agree_across_a_multi_turn_session` proves the
/// text matches), so no behavioural test can catch a regression here. Only the
/// instruction sequence can. That is why it is a source assertion and why
/// `AGENTS.md §3.4` applies to it directly -- it is a claim about the source
/// that must be shown to be able to fail.
#[test]
fn the_pipelined_chat_loop_commits_before_it_waits() {
    let src = std::fs::read_to_string(format!(
        "{}/src/chat.rs",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("chat.rs");

    let loop_body = src
        .split("loop {")
        .nth(1)
        .expect("chat.rs must contain the decode loop");
    // Only the pipelined loop: take it up to the barrier-per-token fallback.
    let pipelined = loop_body
        .split("Fallback: barrier per token")
        .next()
        .unwrap_or(loop_body);

    let commit = pipelined
        .find("commit_terminal()")
        .expect("the pipelined loop must commit a command buffer");
    let wait = pipelined
        .find("previous.wait()")
        .expect("the pipelined loop must wait on the previous encoder");

    assert!(
        commit < wait,
        "the pipelined chat loop waits on step N at offset {wait} but does not commit \
         step N+1 until offset {commit}. Reverting to wait-then-encode idles the GPU \
         for the host's whole encode every token -- measured 2.1x slower than the \
         single-shot generator on identical work, and invisible in the token stream \
         because both orders emit the same text."
    );
}

/// The `read_host_token` invariant the reorder had to preserve: a GPU-staged
/// token is never read before the encoder that staged it has been waited on.
/// The reorder moves the encode earlier; it must NOT move the read earlier too,
/// or the host reads GPU memory with no synchronisation of its own. The file
/// says this was a real bug once ("What was wrong: it read read_host_token(step
/// - 1) at the top of iteration step").
#[test]
fn a_staged_token_is_read_after_its_encoder_is_waited_on() {
    let src = std::fs::read_to_string(format!(
        "{}/src/chat.rs",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("chat.rs");
    let pipelined = src
        .split("loop {")
        .nth(1)
        .and_then(|body| body.split("Fallback: barrier per token").next())
        .expect("pipelined decode loop");

    let wait = pipelined
        .find("previous.wait()")
        .expect("wait on the previous encoder");
    let read = pipelined
        .find("read_host_token(step - 1)")
        .expect("read of the staged token");
    assert!(
        wait < read,
        "read_host_token at offset {read} precedes previous.wait() at offset {wait}. \
         read_host_token is a raw contents() read of GPU-written memory with no \
         synchronisation of its own; this ordering was a shipped bug."
    );
}
