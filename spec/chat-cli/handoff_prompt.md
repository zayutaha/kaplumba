# Handoff prompt — interactive chat CLI for `kaplumba`

Paste everything below the line into a fresh agent session with working
directory `/Users/zayaantaha/coding/kaplumba`.

---

## Your task

Add an interactive multi-turn chat mode to `kaplumba`, invoked as
`kaplumba --chat`. It reads a message from stdin, generates a streamed response,
and repeats until the user exits — the interaction shape of `llama-cli`.

**Conversation context must carry across turns.** Turn _n_ must see turns
1.._n_-1. This is the entire point of the feature and the thing most likely to be
got wrong, so read §3 and §4 of the plan before writing code.

This is a pure **caller-side** feature. Do not modify a single line under
`crates/kernels/`, and do not change any numeric path in
`crates/engine/src/model.rs`. The engine already supports everything you need.

### Read these first, in this order

1. `AGENTS.md` — the repo's operating rules. §9 is the checklist you must follow,
   §8 the standing constraints, §10 the GPU discipline. This repo enforces its own
   process and you will be judged on following it.
2. `spec/chat-cli/spec.md` — requirements R1–R10, acceptance criteria AC1–AC8.
3. `spec/chat-cli/plan.md` — the technical design. Verified file:line references,
   the exact token sequences, the traps, the test plan.

Both documents are written against the current tree and their line references
were checked. If a reference has drifted, re-derive it and say so rather than
trusting it.

### Process you must follow (AGENTS.md §9)

This is a non-trivial task — it spans multiple files and adds a user-facing
subsystem — so the full workflow applies:

1. Read `AGENTS.md` and the two spec documents above.
2. Create `spec/chat-cli/tasks.md` with small, independently verifiable steps,
   each naming the files touched and how it will be tested.
3. Implement **one task at a time**. After each: run the tests, update
   `tasks.md` with status + commit hash, append anything surprising to
   `implementation_notes.md`, and commit.
4. Run the full suite before each commit (§6 below).
5. Prove the acceptance criteria before declaring done, and record what you
   observed — including the platypus test in AC1.

Do not batch unrelated changes. Do not commit broken code or a half-updated
`tasks.md`.

---

## 1. What already exists (verified — do not rebuild it)

A Rust workspace: `crates/kernels` (raw Metal shaders), `crates/engine`
(inference), `crates/cli` (the `kaplumba` binary). The engine is a complete,
working Gemma 4 31B EXL3 inference engine running at **15.85 tok/s** decode with
**bit-exact** greedy output. You are adding a front end, not an engine.

The critical insight, and the reason this task is smaller than it looks:

> **Multi-turn context requires no KV-cache management of any kind.** The cache is
> already append-only and self-tracking. You append tokens at absolute positions
> and it just works.

- `TextModel::forward_buffer_into_vec(token_ids, positions, out)`
  (`crates/engine/src/model.rs:3684`) runs a forward for arbitrary tokens at
  **absolute** positions, and the cache advances itself during the encode.
- `GpuKeyValueCache::reserve` (`crates/engine/src/gpu_cache.rs:230`) **returns an
  error unless `position == next_position`**. That is a safety net, not an
  obstacle: append contiguously from `gpu_next_position()` and it holds by
  construction. A bug that skips or repeats a position fails loudly instead of
  silently corrupting output.
- Full-attention layers grow on demand via `ensure_capacity`
  (`model.rs:1718`), doubling as needed up to the cap. No pre-allocation step,
  no reallocation logic.
- `TextModel::gpu_next_position()` (`model.rs:4407`) gives the committed
  position — the number the context-limit check uses.

**Do not call `reset_gpu_caches()` between turns.** It is called once at session
start and on `/reset`. There are only two non-test call sites today
(`generate.rs:258` and `generate.rs:882`); if you add a third inside a turn, you
have broken the feature.

Existing CLI shape: `crates/cli/src/args.rs` holds `CliArgs::parse` with
hand-rolled validation and its own unit tests; `crates/cli/src/main.rs` loads
the model and tokenizer, then dispatches. There is no `tests/` directory under
`crates/cli` — the parsing tests live inline in `args.rs`, so follow that.

## 2. Model facts you need (all verified)

From `~/.kaplumba/models/Gxma-repacked/config.json` and
`crates/engine/src/config.rs:225-241`:

- 60 layers, `hidden_size 5376`, `intermediate_size 21504`, `vocab_size 262144`
- 32 query heads × 256 head_dim, 16 KV heads, 4 global KV heads
- `sliding_window 1024`; layer types are `index % 6 == 5 → Full`, i.e. **50
  sliding layers and 10 full-attention layers**
- KV cache capped at `max_sequence_len = 8192` (`model.rs:3273`), matching
  `DEFAULT_CONTEXT_LIMIT = 8192` (`generate.rs:13`). Treat 8192 as one constant.
- `generation_config.json`: `eos_token_ids = [1, 106, 50]`
- Full-attention caches start at 512 positions and double on demand
  (`INITIAL_FULL_CACHE`, `model.rs:1075`)

Because 10 layers attend over the whole conversation, **memory grows linearly
with conversation length** — roughly 0.84 GiB of KV at the 8192 cap, on top of
an 11.58 GiB resident model, in a 24 GiB machine. The context limit is a real
resource ceiling. That is why the spec errors and stops the turn instead of
silently dropping old turns.

### Chat template markers

`~/.kaplumba/models/Gxma-repacked/chat_template.jinja` is authoritative. The
markers and their IDs, verified against `tokenizer.json`:

| markup | id |
|---|---|
| `<bos>` | 2 |
| `<eos>` | 1 |
| `<pad>` | 0 |
| `<\|turn>` | 105 |
| `<turn\|>` | 106 |
| `<\|channel>` | 100 |
| `<channel\|>` | 101 |

Turn 1 must render **exactly** what `GemmaTokenizer::encode_prompt`
(`crates/engine/src/tokenizer.rs:87`) already renders today:

```
<bos><|turn>user\n{text}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
```

Reuse that function. Do not hand-roll a first-turn renderer — AC2 requires
`--chat` and `--prompt` to produce identical token IDs for the same message, and
reusing the existing renderer makes that true by construction instead of by a
test that has to be maintained.

Resolve the marker IDs **through the tokenizer** at startup and assert they match
the table above; fail loudly on mismatch. Do not assume a checkpoint swap left
them alone.

## 3. The two traps

These are the bugs you will write if you do not read this section. Both fail
*silently* — the tool runs, prints plausible text, and is subtly wrong.

### Trap 1: the terminator is not fed to the cache

`generate_gpu` (`generate.rs:300-338`) checks EOS **before** forwarding:

```rust
let token = argmax_logits(&logits);
if config.eos_ids.contains(&token) { stopped_on_eos = true; break; }  // not fed
generated_ids.push(token);
...
model.forward_buffer_into_vec(&[token], &[position], &mut logits);
```

So when a turn ends on EOS, the cache holds every generated token **except the
terminator**. Your loop must do the same: break on EOS without forwarding it.

Getting this wrong shifts every subsequent position by one and the model
degrades with no error anywhere. The check: after a turn,
`gpu_next_position() == prompt_start + prompt_len + (generated_tokens - 1)`.
Assert that in a test.

### Trap 2: turn _n_ > 1 needs a leading `<turn|>`

The model produced a terminator that was never fed to the cache. To render turn
_n_ correctly you must emit it:

```
<turn|>\n<|turn>user\n{text}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
```

Note `106` appears **twice**: once leading, to close the assistant's previous
turn, and once mid-string to close the *user's* turn. Omitting the leading one
leaves the model's turn unterminated — the model answers turn 2 somewhat worse
than it should, silently. Assert the exact `Vec<u32>` in a test.

If the model stopped on EOS 1 or 50 rather than 106, still emit `<turn|>`: the
template requires a closed role block, so that is the well-formed choice.

## 4. What you cannot reuse

`generate_gpu` (`generate.rs:218`) and `generate_gpu_greedy` (`generate.rs:849`)
both take `prompt: &str` and call `reset_gpu_caches()` before prefilling. Calling
either per turn re-prefills the entire conversation: _O(L²)_ work across a
session, visibly slower every turn.

So the chat needs its own generate loop over
`forward_buffer_into_vec` / `forward_buffer_into_step`. That duplicates ~40
lines of `generate_gpu`. This duplication is **deliberate and correct** — justify
it in `tasks.md` against R3 and move on. Refactoring `generate_gpu` to accept a
pre-populated cache is a separate change carrying real risk to the bit-exactness
suites, and it is out of scope.

One performance note, so you don't mistake slowness for a bug: the 15.85 tok/s
figure comes from the **pipelined** single-token path
(`forward_step_pipelined`, `model.rs:3622` + `read_host_token`, `model.rs:3671`).
A naive `forward_buffer_into_vec`-per-token loop will be slower. Build the
simple version first, measure it, and only then consider pipelining — and do not
break bit-exactness to chase the number. The point of this feature is a usable
chat tool, not a speed record.

## 5. Interface

```
kaplumba --chat [--model PATH] [--temperature F] [--top-k N] [--top-p F]
                [--seed N] [--max-new-tokens N]
```

- Reject `--chat` combined with `--prompt`, `--prompt-file`, or `--benchmark`
  with a clear `ArgsError::Invalid` — do not silently ignore them.
- Load the model and tokenizer **once**, print the ~7.1 s load time, then enter
  the loop. Branch *after* the load so chat costs one load, not one per turn.
- Defaults from `generation_config.json` via `load_generation_config`:
  temperature 1.0, top_k 64, top_p 0.95. Existing flags override for the
  session. `--temperature 0` selects greedy.
- Loop: print `> `, flush, read a line. Empty/whitespace lines are ignored and do
  not end the session. EOF (Ctrl-D) exits 0.
- Slash commands: `/help`, `/exit` (`/quit`), `/reset`, `/history`, `/stats`.
  An unknown `/`-prefixed line prints a message and the help; it is **not**
  treated as a chat message.
- Stream via `Detokenizer::push` (`tokenizer.rs:195`), flushing on each chunk.
  `Detokenizer` is **per turn**, not per session — construct one per turn and
  call `finish()` (`tokenizer.rs:230`) at the end, or the final partial UTF-8
  sequence is dropped. `finish()` returning a chunk is normal, not an error.
- An error in one turn prints a message and returns to the prompt. Only a failure
  to load the model is fatal.

Preserve the existing behaviour of `main.rs`'s exit paths, including the GPU-busy
message and `ExitCode::from(75)`.

## 6. Tests

Unit tests (no GPU, no TTY) belong inline in `crates/engine/src/chat.rs`
following the `args.rs` convention:

- `render_first_turn` produces the same ids as `GemmaTokenizer::encode_prompt`
  for the same text (this is AC2)
- `render_next_turn` produces an exact literal `Vec<u32>`, so a template change
  shows up as a visible diff
- marker resolution returns 2/1/0/105/106/100/101
- the context-limit boundary: exactly 8192 allowed, one over refused
- slash-command parsing, including unknown commands
- EOS bookkeeping: for a turn of _k_ generated tokens ending in EOS, the
  committed position is `start + prompt_len + k - 1`

Extend the existing `args.rs` tests for the new flag — that file already asserts
boundary conditions, and a new flag with no test is a gap.

Before every commit:

```bash
cargo test --release --workspace
```

The bit-exactness suites are the ones that matter most — they are what proves
you did not perturb the numerics:

```bash
cargo test --release --test live_kernel_bit_exact --test fused_bit_exact \
  --test repacked_golden --test argmax_engine_parity --test norm_rope_qkv_bit_exact
```

Manual GPU verification, **one model process at a time** (AGENTS.md §10 — a
second instance overruns physical memory and destroys the measurements):

```bash
scripts/gpuq try -- ./target/release/kaplumba --chat \
  --model ~/.kaplumba/models/Gxma-repacked
```

Prove AC1 by hand: turn 1 "remember the word platypus", turn 2 "what word did I
ask you to remember?" — the answer must contain `platypus`. That test is what
distinguishes real context reuse from a tool that quietly forgets. Then prove
AC5 by driving the conversation past 8192 tokens and confirming it reports the
committed/required counts rather than truncating. Finally confirm AC8:
`--prompt "The capital of France is"` still yields `Paris`.

`AGENTS.md` §10.7 warns that a missing output is not a hang: a prompt over 8192
tokens is rejected *before* loading with "prompt plus generation exceeds context
limit", which looks exactly like a silent failure if you only read the tail of a
pipe.

## 7. Definition of done

- Every task in `tasks.md` complete, each committed separately with its hash
  recorded.
- AC1–AC8 demonstrated, with what you observed written into
  `implementation_notes.md`.
- `git diff --stat` over `crates/kernels/` is empty for this feature.
- Full workspace suite green; golden prompt intact.
- `implementation_notes.md` updated with anything the plan did not predict —
  especially if a documented file:line reference was wrong, or if the naive
  decode loop turned out faster or slower than §4 suggests.

## 8. Honest notes on scope

The spec makes two judgement calls you should know about rather than discover:

- **Context overflow errors instead of auto-truncating.** When the limit is
  hit, the tool says so and stops that turn; the user can `/reset`. Silently
  dropping the oldest turns was rejected because invisible history loss is worse
  than an explicit message. If the user actually wants auto-truncate, it is a
  small change to the `fits` check — but ask before making it.
- **No system-prompt flag.** The template supports a system block
  (`<|turn>system\n{text}<turn|>\n` as a turn-0 prefill), but it is left out to
  keep this one reviewable unit. Mention it as a natural follow-up; do not build
  it unasked.

There is no ambiguity left in the requirements that needs asking about. If you
find one, ask rather than guessing — but the design is settled.