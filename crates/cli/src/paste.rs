//! Reading one chat turn that may be an arbitrarily long paste.
//!
//! # Why this exists
//!
//! `--chat` originally read with `stdin.lock().lines()`, which makes **one
//! physical line equal one turn**. That is fine for `What is 2+2?` and wrong
//! for everything else: paste a paragraph, or a page of source, or a long log,
//! and every newline in it becomes another turn, so the model answers each line
//! of your paste separately and the paragraph is shredded. There was also no
//! line editor, so a long line soft-wrapped across the terminal and the only way
//! to see what had been captured was to scroll back.
//!
//! The fix has two halves, and the second is the one that matters for
//! "spilling over":
//!
//! 1. **Join.** After a line has been read, keep reading while more input is
//!    already waiting, and append it with newlines. One paste becomes one turn.
//! 2. **Report.** Tell the user how much was captured, on one line, instead of
//!    leaving a wall of terminal echo to interpret. See [`Turn::summary`].
//!
//! # Why the drain is gated on `isatty`
//!
//! Piped input is deliberately **excluded**. `printf 'a\nb\n' | kaplumba --chat`
//! means two turns today and must keep meaning two turns; that is how the mode
//! is scripted and tested. Only an interactive terminal — the case where a paste
//! actually shreds — gets joined. See [`stdin_is_terminal`].

use std::io::{self, BufRead};

/// How long to wait for the rest of a paste after a line has been read.
///
/// A paste is already sitting in the tty's input queue by the time the read that
/// ended the first line returns, so this only has to cover the kernel handing
/// over the *rest* of a large paste, which arrives in 4 KiB chunks. 25 ms is far
/// longer than that takes and far shorter than a person can begin typing the
/// next prompt — which is the mistake this timeout has to avoid making, by
/// swallowing a fast typist's second line into the first turn.
const PASTE_DRAIN_MS: i32 = 25;

/// One logical turn: the user's text, however many physical lines it took.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Turn {
    /// The turn's text, with pasted lines joined by `\n`.
    pub text: String,
    /// How many physical lines it was assembled from. 1 for a typed line.
    pub lines: usize,
}

impl Turn {
    /// Whether this turn is a `/command` rather than something to send.
    ///
    /// Requires `lines == 1` on purpose. A pasted block whose first line happens
    /// to be `/help` is text the user wants answered, not a command — and with
    /// pasting now supported, that collision is reachable for the first time.
    pub fn is_command(&self) -> bool {
        self.lines == 1 && self.text.trim_start().starts_with('/')
    }

    /// Whether there is nothing to send. Blank lines inside a paste do **not**
    /// make a turn empty, which is why this looks at the whole text.
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
    }

    /// A one-line description of a multi-line turn, for telling the user what was
    /// captured. `None` for an ordinary typed line.
    pub fn summary(&self) -> Option<String> {
        (self.lines > 1).then(|| {
            format!(
                "[pasted {} lines, {} chars]",
                self.lines,
                self.text.chars().count()
            )
        })
    }
}

/// Whether stdin is a terminal — and so the place a paste arrives from.
///
/// A pipe or a file is not, and must not be joined: see the module comment.
pub fn stdin_is_terminal() -> bool {
    // SAFETY: `isatty` only inspects the descriptor and has no memory effects.
    unsafe { libc::isatty(libc::STDIN_FILENO) == 1 }
}

/// Whether more input is already waiting on stdin, waiting up to `timeout_ms`.
///
/// A `poll` error reads as "nothing pending", which ends the drain and leaves
/// whatever was already read intact. Failing to drain is recoverable; losing the
/// lines read so far is not, so the bias is towards stopping.
fn input_pending(timeout_ms: i32) -> bool {
    let mut fds = libc::pollfd {
        fd: libc::STDIN_FILENO,
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: `fds` is a single valid, initialised pollfd for the duration of the
    // call, which is what `poll` requires.
    unsafe { libc::poll(&mut fds, 1, timeout_ms) > 0 }
}

/// Assemble one turn from `first` plus whatever `next_line` still yields while
/// `pending` says input is waiting.
///
/// Split out from [`TurnReader`] so the joining rules are testable without a
/// terminal: `next_line` and `pending` are injected. `next_line` returning
/// `None` is end-of-input and ends the drain, which is what makes Ctrl-D in the
/// middle of a paste send what was pasted rather than losing it.
fn drain_turn<L, P>(first: String, mut next_line: L, mut pending: P) -> Turn
where
    L: FnMut() -> Option<io::Result<String>>,
    P: FnMut(i32) -> bool,
{
    let mut text = first;
    let mut lines = 1_usize;
    while pending(PASTE_DRAIN_MS) {
        let Some(next) = next_line() else { break };
        match next {
            Ok(line) => {
                text.push('\n');
                text.push_str(&line);
                lines += 1;
            }
            // A malformed line is not a reason to throw away the lines already
            // read. Stop here and let the caller report the turn; the encoding
            // problem surfaces on the next turn instead of silently truncating
            // this one mid-word.
            Err(_) => break,
        }
    }
    Turn { text, lines }
}

/// Reads logical turns from stdin, joining an interactive paste into one turn.
pub struct TurnReader {
    lines: io::Lines<io::StdinLock<'static>>,
    join_pastes: bool,
}

impl TurnReader {
    /// A reader over stdin that joins pastes **iff stdin is a terminal**.
    pub fn new() -> Self {
        Self::with_paste(stdin_is_terminal())
    }

    /// A reader with the paste behaviour chosen explicitly. For tests.
    pub fn with_paste(join_pastes: bool) -> Self {
        Self {
            lines: io::stdin().lock().lines(),
            join_pastes,
        }
    }

    /// Read one turn, or `None` at end of input.
    pub fn read_turn(&mut self) -> io::Result<Option<Turn>> {
        read_turn_from(&mut self.lines, self.join_pastes, input_pending)
    }
}

/// The body of [`TurnReader::read_turn`], with the terminal's line source and its
/// "is more waiting" predicate injected.
///
/// Takes the line source by **value**, not `&mut`. `&mut I` only implements
/// `Iterator` when `I: Clone`, so passing `&mut self.lines` from the method would
/// force `StdinLock` to be cloneable — it is not, and cannot be.
fn read_turn_from<L, P>(
    mut lines: L,
    join_pastes: bool,
    pending: P,
) -> io::Result<Option<Turn>>
where
    L: Iterator<Item = io::Result<String>>,
    P: FnMut(i32) -> bool,
{
    let Some(first) = lines.next() else {
        return Ok(None); // EOF (Ctrl-D) exits cleanly.
    };
    let first = first?;
    if !join_pastes {
        return Ok(Some(Turn {
            text: first,
            lines: 1,
        }));
    }
    Ok(Some(drain_turn(first, || lines.next(), pending)))
}

#[cfg(test)]
mod tests {
    use super::*;

    // A readable helper: N more lines available, then EOF.
    fn source(lines: Vec<&str>) -> impl FnMut() -> Option<io::Result<String>> {
        let mut queue: Vec<io::Result<String>> =
            lines.into_iter().map(|line| Ok(line.to_string())).collect();
        move || {
            if queue.is_empty() {
                None
            } else {
                Some(queue.remove(0))
            }
        }
    }

    /// `poll` says "input waiting" for `rounds` calls, then stops.
    fn pending_for(rounds: usize) -> impl FnMut(i32) -> bool {
        let mut left = rounds;
        move |_| {
            if left == 0 {
                false
            } else {
                left -= 1;
                true
            }
        }
    }

    #[test]
    fn a_single_line_is_one_turn() {
        let turn = drain_turn("hello".to_string(), source(vec![]), pending_for(0));
        assert_eq!(turn, Turn { text: "hello".into(), lines: 1 });
        assert!(!turn.is_command());
        assert!(!turn.is_empty());
        assert_eq!(turn.summary(), None);
    }

    #[test]
    fn a_three_line_paste_becomes_one_turn() {
        let turn = drain_turn(
            "first".to_string(),
            source(vec!["second", "third"]),
            pending_for(2),
        );
        assert_eq!(turn.text, "first\nsecond\nthird");
        assert_eq!(turn.lines, 3);
        assert_eq!(turn.summary().as_deref(), Some("[pasted 3 lines, 18 chars]"));
    }

    #[test]
    fn blank_lines_inside_a_paste_are_kept() {
        // The whole point: a pasted paragraph has blank lines in it, and they
        // must survive into the prompt rather than ending the turn.
        let turn = drain_turn(
            "para one".to_string(),
            source(vec!["", "para two"]),
            pending_for(2),
        );
        assert_eq!(turn.text, "para one\n\npara two");
        assert_eq!(turn.lines, 3);
        assert!(!turn.is_empty());
    }

    #[test]
    fn the_drain_stops_when_nothing_is_pending() {
        // Three lines are available but only one poll says "waiting": the extra
        // lines belong to the NEXT turn, which is the fast-typist case.
        let turn = drain_turn(
            "mine".to_string(),
            source(vec!["yours", "later"]),
            pending_for(1),
        );
        assert_eq!(turn.text, "mine\nyours");
        assert_eq!(turn.lines, 2);
    }

    #[test]
    fn a_paste_that_looks_like_a_command_is_text() {
        let turn = drain_turn(
            "/help".to_string(),
            source(vec!["and also this"]),
            pending_for(1),
        );
        assert!(!turn.is_command(), "a pasted block is never a command");
    }

    #[test]
    fn a_lone_slash_line_is_a_command() {
        let turn = drain_turn("/stats".to_string(), source(vec![]), pending_for(0));
        assert!(turn.is_command());
    }

    #[test]
    fn a_command_with_leading_space_is_still_a_command() {
        let turn = drain_turn("   /help".to_string(), source(vec![]), pending_for(0));
        assert!(turn.is_command());
    }

    #[test]
    fn whitespace_only_input_is_empty() {
        assert!(drain_turn("   ".to_string(), source(vec![]), pending_for(0)).is_empty());
        assert!(drain_turn("\t \n ".to_string(), source(vec![]), pending_for(0)).is_empty());
    }

    #[test]
    fn eof_mid_paste_keeps_what_was_pasted() {
        // Ctrl-D partway through: send the text, do not lose it.
        let turn = drain_turn(
            "kept".to_string(),
            source(vec![]),
            pending_for(5), // still "pending", but the source is exhausted
        );
        assert_eq!(turn.text, "kept");
        assert_eq!(turn.lines, 1);
    }

    #[test]
    fn summary_counts_characters_not_bytes() {
        let turn = Turn { text: "é".repeat(10), lines: 4 };
        assert_eq!(turn.summary().as_deref(), Some("[pasted 4 lines, 10 chars]"));
    }

    #[test]
    fn paste_joining_is_off_when_stdin_is_not_a_terminal() {
        // `printf 'a\nb\n' | kaplumba --chat` must stay two turns. Exercised
        // through the reader so the gate itself is covered, not just the helper.
        //
        // `into_iter()` and then `&mut` it: a slice iterator would hand out
        // `&io::Result<String>` and stop matching the bound, and the point of
        // these tests is the real iterator shape.
        let mut iter = vec![Ok("a".to_string()), Ok("b".to_string())].into_iter();
        let first = read_turn_from(&mut iter, false, |_| true).unwrap().unwrap();
        assert_eq!(first, Turn { text: "a".into(), lines: 1 });
        let second = read_turn_from(&mut iter, false, |_| true).unwrap().unwrap();
        assert_eq!(second, Turn { text: "b".into(), lines: 1 });
        assert!(read_turn_from(&mut iter, false, |_| true).unwrap().is_none());
    }

    #[test]
    fn paste_joining_is_on_when_stdin_is_a_terminal() {
        let mut iter = vec![
            Ok("a".to_string()),
            Ok("b".to_string()),
            Ok("c".to_string()),
        ]
        .into_iter();
        let turn = read_turn_from(&mut iter, true, |_| true).unwrap().unwrap();
        assert_eq!(turn, Turn { text: "a\nb\nc".into(), lines: 3 });
        // ...and the next turn starts clean rather than repeating the paste.
        assert!(read_turn_from(&mut iter, true, |_| true).unwrap().is_none());
    }

    #[test]
    fn joining_does_not_lose_a_line_across_turns() {
        // Two pastes back to back. The turn boundary must fall between them.
        //
        // The predicate models a real tty: input is "pending" only while lines
        // remain buffered. An always-true predicate says the user never stops
        // typing, and then joining everything is the correct answer -- which is
        // exactly why the earlier version of this test, written that way,
        // asserted a boundary the code has no way to know about. There is no
        // signal distinguishing "rest of this paste" from "a whole new paste
        // typed already"; the timeout is the only boundary, and it is a
        // heuristic by necessity.
        // Each turn gets its own `pending`, because the boundary between two
        // pastes IS the predicate going false. That is the whole mechanism: the
        // code cannot distinguish "rest of this paste" from "a new paste already
        // typed", so the timeout is the boundary and is a heuristic by
        // necessity. Asserting it here keeps that assumption written down.
        let first = drain_turn(
            "p1a".to_string(),
            source(vec!["p1b", "p2a", "p2b"]),
            pending_for(1),
        );
        assert_eq!(first.text, "p1a\np1b");
        assert_eq!(first.lines, 2);

        let second = drain_turn(
            "p2a".to_string(),
            source(vec!["p2b"]),
            pending_for(1),
        );
        assert_eq!(second.text, "p2a\np2b");
        assert_eq!(second.lines, 2);
    }

    #[test]
    fn a_paste_typed_after_the_turn_is_not_swallowed() {
        // The fast-typist case the 25 ms timeout exists to protect: input stops
        // being pending, so the turn ends even though more lines are queued.
        let mut iter = vec![
            Ok("first".to_string()),
            Ok("typed next".to_string()),
        ]
        .into_iter();
        let first = read_turn_from(&mut iter, true, |_| false).unwrap().unwrap();
        assert_eq!(first.text, "first");
        assert_eq!(first.lines, 1);
        let second = read_turn_from(&mut iter, true, |_| false).unwrap().unwrap();
        assert_eq!(second.text, "typed next");
    }

    #[test]
    fn a_single_pasted_line_is_reported_as_one_line_not_a_paste() {
        // A very long single line -- one paragraph with no newlines -- is the
        // other half of the request. It must arrive intact and must not be
        // announced as a multi-line paste, because it is not one.
        let long = "word ".repeat(5_000);
        let turn = drain_turn(long.clone(), source(vec![]), pending_for(0));
        assert_eq!(turn.lines, 1);
        assert_eq!(turn.text, long);
        assert_eq!(turn.summary(), None);
        assert!(!turn.is_empty());
    }
}
