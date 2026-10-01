# SPECIFICATION: Interactive Chat REPL

## 1. Goal

Add an interactive multi-turn chat mode to `kaplumba`, invoked as
`kaplumba --chat`, that reads a message from stdin, generates a streamed
response, and repeats until the user exits — the interaction shape of
`llama-cli`. Conversations must carry context across turns: turn _n_ must see
turns 1.._n_-1. The model loads once and stays resident for the whole session.

Success is: a human can hold a coherent multi-turn conversation with
`Gxma-repacked` at `~/.kaplumba/models/Gxma-repacked` without the process
restarting, without losing the thread, and without the cache growing faster
than the 8192-token context allows.

---

## 2. Non-Goals & Restrictions

1. **No quality or determinism change to the engine.** This feature adds a
   caller. It must not alter a single kernel, weight layout, or numeric
   reduction. `crates/kernels/**` is out of scope entirely.
2. **No re-prefill of history.** Turn _n_ must not re-run the conversation
   through the model from scratch. See R3.
3. **No silent truncation.** When the context limit is reached the tool says so
   and stops that turn. It must never drop the oldest turns to make room
   (R7).
4. **No streaming *input*.** The prompt is a complete line of stdin. No partial
   lines, no paste detection, no multi-line heredoc mode.
5. **Single user, single session.** No session save/restore, no history file,
   no multi-turn transcript persistence.
6. **No Python.** Engine and CLI are Rust only (AGENTS.md §8). `scripts/` may
   use Python for offline tooling, as it already does.
7. **No speculative decoding, no quantization change, no KV quantization.**
   Standing project constraints (AGENTS.md §8).
8. **`--prompt` / `--benchmark` behaviour is unchanged.** Chat is an additional
   mode selected by `--chat`, not a replacement default.

---

## 3. Requirements

### R1 — Invocation
`kaplumba --chat [--model PATH]` enters the REPL. The model and tokenizer load
exactly once, before the first prompt, and the load time is printed so the user
knows the wait is expected. All existing one-shot flags (`--prompt`,
`--prompt-file`, `--benchmark`, `--legacy-path`) are rejected in combination
with `--chat` with a clear error rather than being silently ignored.

### R2 — Prompt loop
Print a prompt (default `> `) to stdout, flush, read one line from stdin. An
empty or whitespace-only line is ignored without ending the session. EOF
(Ctrl-D) exits cleanly with status 0.

### R3 — Incremental context, no re-prefill
The engine's GPU KV cache persists across turns. Turn 1 prefills the rendered
prompt; turn _n_ (_n_ > 1) prefills **only the new turn's tokens** at their
absolute positions. This is required, not an optimisation: re-prefilling a
conversation of length _L_ before every turn costs _O(L²) total work and makes
the tool unusable after a few exchanges.

### R4 — Chat template
Conversations must be rendered in the exact format the checkpoint's own
`chat_template.jinja` produces, so that token-for-token the model sees what it
was trained on. The relevant markup and its token IDs:

| markup | id | role in the format |
|---|---|---|
| `<bos>` | 2 | start of sequence, first turn only |
| `<eos>` | 1 | pad/fallback stop |
| `<pad>` | 0 | unused |
| `<\|turn>` | 105 | opens a role block |
| `<turn\|>` | 106 | closes a role block; also the model's turn terminator |
| `<\|channel>` | 100 | opens a channel block |
| `<channel\|>` | 101 | closes a channel block |

Turn 1 renders exactly what the existing `GemmaTokenizer::encode_prompt` renders
today, so `--chat`'s first turn and `--prompt` are byte-identical:

```
<bos><|turn>user\n{text}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
```

Each subsequent turn renders:

```
<turn|>\n<|turn>user\n{text}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
```

The leading `<turn|>` of turn _n_ > 1 is the terminator the model itself
produced and which was **not** fed to the cache (see R5), so feeding it here is
what closes the assistant's previous turn. Omitting it would leave the model's
turn unterminated and degrade the response.

### R5 — Stop handling
`generation_config.json` gives `eos_token_ids = [1, 106, 50]`. Generation of a
turn stops on any of them, and **the stopping token is not fed into the cache** —
this matches `generate_gpu`, where the EOS check precedes the forward. If the
model stopped on 1 or 50 rather than 106, the next turn still emits `<turn|>`
to close the block: the template requires a closed role block, so this is the
well-formed choice even though the model chose a different terminator.

### R6 — Streaming
Response text is emitted token-by-token via `Detokenizer::push`, flushing stdout
on each chunk, so the user sees the answer as it is produced rather than after
it completes.

### R7 — Context limit
The hard limit is 8192 tokens (`DEFAULT_CONTEXT_LIMIT`,
`generate.rs:13`; the KV cache is capped at `max_sequence_len = 8192`,
`model.rs:3273`). Before each turn, if `gpu_next_position() + new_tokens +
max_new_tokens > 8192`, print how many tokens are already committed, how many
the new message needs, and stop that turn with a non-fatal message. The session
continues; the user can `/reset`. History is never silently dropped.

### R8 — Slash commands
- `/help` — list commands
- `/exit`, `/quit` — leave (also EOF)
- `/reset` — clear the conversation, reset the KV cache, start fresh
- `/history` — print the conversation as it would be rendered, plus a token count
- `/stats` — context tokens used / 8192, and tok/s for the last turn

An unknown `/`-prefixed line prints the unknown-command message and the help
text; it is not treated as a chat message.

### R9 — Sampling
Defaults come from `generation_config.json` via `load_generation_config`:
`temperature 1.0`, `top_k 64`, `top_p 0.95`. `--temperature`, `--top-k`,
`--top-p`, `--seed` and `--max-new-tokens` override them for the session.
`--temperature 0` selects greedy.

### R10 — Errors
An error in one turn (context overflow, sampler failure, tokenizer failure)
prints a message and returns to the prompt. It does not exit the session. Only
a failure to load the model is fatal.

---

## 4. Acceptance Criteria

1. **AC1 — Coherent multi-turn context.** In one session, turn 1 asks the model
   to remember a nonce (e.g. "remember the word platypus"), turn 2 asks what it
   was. Turn 2's answer contains `platypus`. This is the acceptance test that
   distinguishes real context reuse from a tool that silently forgets.
2. **AC2 — First turn matches `--prompt`.** For the same first message, `--chat`
   and `--prompt "<same text>"` produce identical generated token IDs under
   `--temperature 0`. R4 makes this true by construction; the test proves it.
3. **AC3 — No re-prefill.** Instrumentation or the profiler shows turn 2 prefills
   only the new turn's token count, not the whole conversation.
4. **AC4 — Streaming.** Output appears incrementally, not in one burst at the
   end.
5. **AC5 — Overflow is reported, not hidden.** A conversation driven past 8192
   tokens prints the committed/required counts and returns to the prompt.
6. **AC6 — Engine untouched.** `git diff --stat` over `crates/kernels/**` and
   `crates/engine/src/model.rs` is empty for this feature. New code lives in
   `crates/engine/src/chat.rs` and `crates/cli/src/chat.rs`; existing files are
   edited only to add the module declaration, the `--chat` flag, and dispatch.
7. **AC7 — Tests.** New unit tests cover the template renderer (exact token
   sequences for turn 1 and turn 2), the stop/terminator bookkeeping, the
   context-overflow arithmetic, and the slash-command parser. All existing tests
   still pass, including the bit-exactness suites.
8. **AC8 — Golden prompt intact.** `--prompt "The capital of France is"` still
   yields `Paris`, proving the engine's numerics did not move.

---

---

## 5. Handoff

`handoff_prompt.md` in this directory is a self-contained brief for a fresh
agent session: the task, the process the repo requires, the verified engine API
surface, the two silent-failure traps, the interface, and the test plan. Start
there.

---

## 6. Open Questions

None blocking. Two defaults were chosen where the user did not specify and can
be changed without touching the design:

- **Overflow behaviour** is *error and stop the turn* (R7), not auto-truncate.
  Auto-dropping the oldest turns is the alternative; it was rejected because
  silent history loss is worse than an explicit message.
- **A system prompt / persona flag is not included.** It is a small addition
  (`<|turn>system\n{text}<turn|>\n` as a turn-0 prefill, per the template's
  system-block branch) and is left out of scope to keep this change to one
  reviewable unit.