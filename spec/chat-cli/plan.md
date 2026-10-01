# PLAN: Interactive Chat REPL

Read `spec.md` first. This document is the technical design: what to build, where,
and why each piece is shaped the way it is. Everything here was verified against
the current tree; the file:line references are real and should be re-checked if
they have moved.

---

## 1. Where the code goes

```
crates/engine/src/chat.rs      NEW   conversation state machine, no I/O
crates/cli/src/chat.rs         NEW   REPL: stdin/stdout, slash commands
crates/engine/src/lib.rs       EDIT  + pub mod chat;
crates/cli/src/args.rs         EDIT  + `--chat` field, parse arm, help text
crates/cli/src/main.rs         EDIT  dispatch to the REPL after load
```

`chat.rs` in the engine owns the token bookkeeping and the model calls. It takes
no `std::io` types, so it is unit-testable without a terminal and without a GPU.
The CLI file owns everything the user sees. That split is the reason the tests
in R7 can assert exact token sequences.

---

## 2. The engine already supports this; do not build a cache

The single most important thing to understand before writing code: **multi-turn
context requires no cache management at all.** The KV cache is already
append-only and self-tracking.

- `TextModel::forward_buffer_into_vec(token_ids, positions, out)`
  (`model.rs:3684`) runs a forward for arbitrary `token_ids` at **absolute**
  `positions`, and `GpuKeyValueCache::advance` runs during the encode, so
  `next_position` is correct for the next call. Positions are absolute, so the
  caller passes `gpu_next_position() + i` for token `i`.
- `GpuKeyValueCache::reserve` (`gpu_cache.rs:230`) **errors unless
  `position == next_position`**. This is a helpful invariant, not an obstacle: as
  long as the caller appends contiguously from `gpu_next_position()`, it holds by
  construction. A bug that skipped or duplicated a position fails loudly.
- Full-attention layers grow on demand: `ensure_capacity(&device, offset+tokens,
  config.max_sequence_len)` (`model.rs:1718`) doubles the buffer as needed.
  Sliding layers ring at a fixed 1024 and are untouched. So there is no
  pre-allocation step and no reallocation logic to write.
- `TextModel::gpu_next_position()` (`model.rs:4407`) reports the committed
  position — this is the number R7's overflow arithmetic uses.

**Do not** call `reset_gpu_caches()` between turns. That is the whole bug this
feature exists to avoid; it is called once at session start and on `/reset`.

---

## 3. Attention shape, and why the context limit is 8192

From `config.json` (`text_config`) and `config.rs:225-241`:

- 60 layers, `hidden_size 5376`, `intermediate_size 21504`, `vocab_size 262144`
- `sliding_window 1024`, and `layer_types` is `index % 6 == 5 → Full` — so
  **50 sliding layers and 10 full-attention layers** (indices 5, 11, 17, …, 59)
- 32 query heads × 256 head_dim, 16 KV heads, 4 global KV heads

Consequence for the chat: the 10 full-attention layers hold every token of the
conversation, so **memory grows linearly with conversation length** and the
8192-token cap is a real resource ceiling, not a formality. At 8192 positions
those layers hold roughly 10 × 8192 × (16+4) × 256 × 2 × 2 B ≈ 0.84 GiB on top
of an 11.58 GiB resident model. This is why R7 errors rather than truncates: the
user should learn the ceiling from the tool, not from the machine swapping.

`max_sequence_len` is set to 8192 at `model.rs:3273`, matching
`DEFAULT_CONTEXT_LIMIT` (`generate.rs:13`). Treat 8192 as one constant, not two.

---

## 4. Turn-by-turn data flow

### Turn 1

```
ids = [2, 105] + tokenize("user\n" + text) + [106, 107, 105] + tokenize("model\n")
      + [100] + tokenize("thought\n") + [101]
```

Better: reuse `GemmaTokenizer::encode_prompt(text)`
(`tokenizer.rs:87`), which already produces exactly this string:

```
<bos><|turn>user\n{text}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
```

**Turn 1 must call `encode_prompt` and nothing else.** It is the R4/AC2
invariant: identical to `--prompt`, by construction rather than by a
hand-rolled renderer that could drift.

```
start = model.gpu_next_position()          // 0 after reset
positions = (start..start + ids.len())
model.forward_buffer_into_vec(&ids, &positions, &mut logits)
```

then decode: `argmax(logits)` (or `Sampler::sample`) → check EOS (R5) →
`Detokenizer::push` → stream → `forward_buffer_into_vec(&[token], &[pos], …)`.

### Turn _n_ > 1

```
delta = [106, 107]                       // close previous model turn
      + [105] + tokenize("user\n" + text)
      + [106, 107] + [105] + tokenize("model\n")
      + [100] + tokenize("thought\n") + [101]
start = model.gpu_next_position()
```

Note `106` (`<turn|>`) appears twice: once leading, to terminate the assistant's
previous turn, and once mid-string to close the *user's* turn. Getting the
leading one wrong produces a model that answers turn 2 slightly worse than it
should, with no error — verify it explicitly in a test.

**Marker tokens must be emitted as IDs, not tokenized from text.** All seven
markers are single vocab entries, so `tokenizer.encode("<turn|>")` returns the
right id — but only if the tokenizer treats them as special/added tokens. Do not
assume: resolve each marker once through the tokenizer at startup and assert the
ids are 2/1/0/105/106/100/101. If any differ, **fail loudly at startup** rather
than rendering a subtly wrong template. Hardcoding the numbers in the renderer
while separately asserting them is the right split: the renderer stays readable,
the assertion catches a checkpoint swap.

---

## 5. R5 in detail — what is in the cache after a turn

This is the part that is easy to get wrong, so state it plainly.

`generate_gpu` (`generate.rs:300-350`) checks EOS **before** forwarding:

```rust
let token = argmax_logits(&logits);
if config.eos_ids.contains(&token) { stopped_on_eos = true; break; }  // not fed
generated_ids.push(token);
…
model.forward_buffer_into_vec(&[token], &[position], &mut logits);
```

So when a turn ends on EOS, the cache holds *every generated token except the
terminator*. The chat loop must do the same: **break on EOS without forwarding
it.** Forwarding the terminator would push every position on by one and the
next turn's `<turn|>` would appear twice in the rendered stream.

Consequence, stated as a rule: after a turn, `gpu_next_position()` equals
`prompt_start + ids.len() + (generated_tokens - 1)`. Assert this in a test for a
hand-run case; it is the cheapest available check that the bookkeeping is right.

---

## 6. Why `generate_gpu` cannot be reused as-is

`generate_gpu` (`generate.rs:218`) calls `model.reset_gpu_caches()` at line 258
before prefilling, and takes `prompt: &str` — it renders one prompt and starts
from an empty cache. Calling it per turn would re-prefill the entire
conversation: _O(L²) work across a session, and visibly slower every turn.
`generate_gpu_greedy` (line 849) has the same shape and the same reset.

So the chat needs its own generate loop over
`forward_buffer_into_vec` / `forward_buffer_into_step`. That is a deliberate
duplication of ~40 lines of `generate_gpu`. Justify it in `tasks.md` as R3 and
move on; refactoring `generate_gpu` to accept a pre-populated cache is a
separate change with its own risk to the bit-exact suites.

If greedy (`--temperature 0`) is wanted, note `forward_step_pipelined`
(`model.rs:3622`) + `read_host_token` (`model.rs:3671`) is the pipelined
single-token path that `generate_gpu_greedy` uses, and it is what makes greedy
decode reach 15.85 tok/s. A naive `forward_buffer_into_vec` per token in the REPL
will be slower. Start with the simple path, measure, and only then consider
pipelining — and see the memory note in §8 before adding it.

---

## 7. CLI shape

`args.rs` gains one `bool` field and one match arm:

```rust
"--chat" => chat = true,
```

and in the existing validation block, reject `--chat` combined with
`--prompt`/`--prompt-file`/`--benchmark` (`ArgsError::Invalid`). Extend
`help_text()` and the `args.rs` unit tests to match — those tests assert
boundary conditions, so a new flag with no test is a gap.

`main.rs` splits after the model+tokenizer load (which is shared) into:

```
if arguments.chat { chat::run(...) } else { /* existing one-shot path */ }
```

Loading once and then branching is why chat costs one 7.1 s load rather than one
per turn.

REPL skeleton:

```
loop {
    print!("> "); flush;
    let line = read_line()?;            // None on EOF -> break
    if line.trim().is_empty() { continue }
    match parse_command(&line) { ... }  // /help /exit /reset /history /stats
    let reply = engine_chat.turn(&line, &mut |chunk| { print!("{chunk}"); flush })?;
    println!();
}
```

`Detokenizer` (`tokenizer.rs:180`) is per-turn, not per-session: construct one at
the start of each turn and call `finish()` (`tokenizer.rs:230`) at the end, or
the last partial UTF-8 sequence is dropped. `finish()` returning a chunk is
normal, not an error.

---

## 8. Memory and performance constraints that apply to this feature

From the measured state of the tree (`spec/decode-25tok-peak-bandwidth/`):

- **Resident model is 11.58 GiB** (`phys_footprint`, not `ru_maxrss` — see
  AGENTS.md §10.3; `peak_resident_pages_bytes` is `getrusage` and understates by
  ~5 GiB). Total budget is 24 GiB, so a long conversation has ~10 GiB of room
  and the 8192 cap is what stops it being exceeded.
- **Decode is 15.85 tok/s** (63.1 ms/token). A 200-token answer is ~12.6 s.
  Print the per-turn tok/s in `/stats` so the user can see this rather than
  think it has hung.
- **Model load is ~7.1 s.** Print it before the first prompt.
- One model process at a time: the engine holds a kernel `flock` and a second
  process exits 75. `main.rs` already maps that to `ExitCode::from(75)` and a
  "GPU busy" line — leave that path intact.

---

## 9. Requirement → task map

| Req | Satisfied by |
|---|---|
| R1 invocation | T2 (`--chat` flag + validation), T5 (dispatch in `main.rs`) |
| R2 prompt loop | T5 |
| R3 incremental context | T3 (`ChatSession::turn`), T4 (the reset guard test) |
| R4 chat template | T1 (marker resolution + assertion), T3 |
| R5 stop handling | T3, tested in T4 |
| R6 streaming | T3, T5 |
| R7 context limit | T3 (`fits` arithmetic), T4 |
| R8 slash commands | T5 |
| R9 sampling | T5 |
| R10 errors | T5 |
| AC2 first turn == `--prompt` | T1 + T4 |

---

## 10. Risks

| Risk | Why it bites | Mitigation |
|---|---|---|
| Terminator fed to the cache | Silently shifts every later position; model degrades with no error | R5 rule + the `gpu_next_position()` assertion in T4 |
| Missing leading `<turn|>` on turn _n_ | Model answers turn 2 worse; no error anywhere | Explicit token-sequence test in T4 |
| Marker IDs assumed rather than resolved | A checkpoint swap renders a wrong template invisibly | Resolve through the tokenizer, assert at startup (T1) |
| Re-prefill creep | Someone "simplifies" by calling `generate_gpu` per turn; _O(L²) | R3 is explicit; note the call sites that reset |
| 8192 overflow reached mid-conversation | Full-attention layers grow ~0.84 GiB by the cap | R7 checks before the turn, reports, never truncates |
| Greedy decode slower in the REPL than the benchmark | The pipelined path is not the obvious one | §6; measure before optimising; don't break bit-exactness to gain it |
| `AGENTS.md` §9 process skipped | Non-trivial feature, spans 6 files | Follow the spec/plan/tasks/commit discipline throughout |

---

## 11. Test plan

`crates/engine/src/chat.rs` gets `#[cfg(test)] mod tests` for everything that is
pure logic:

- `render_first_turn` equals `GemmaTokenizer::encode_prompt`'s ids for the same
  text (AC2, and it fails loudly if R4 drifts from the existing renderer)
- `render_next_turn` produces the exact expected id sequence, asserted as a
  literal `Vec<u32>` so a change to the template is a visible diff
- marker resolution returns 2/1/0/105/106/100/101
- `fits()` boundary: exactly at 8192 is allowed, one over is refused
- slash-command parsing, including unknown-command and case handling
- EOS bookkeeping: given a simulated turn of _k_ generated tokens ending in EOS,
  the committed position is `start + prompt + k - 1`

GPU-dependent behaviour is verified by hand with `scripts/gpuq try --` (one model
process at a time, AGENTS.md §10), driving AC1's platypus test and AC5's overflow
message.

Full suite before committing:

```bash
cargo test --release --test live_kernel_bit_exact --test fused_bit_exact \
  --test repacked_golden --test argmax_engine_parity --test norm_rope_qkv_bit_exact
cargo test --release --workspace
```

---

## 12. References verified for this plan

- `spec/chat-repl/spec.md` — requirements R1-R10, acceptance criteria
- `spec/decode-25tok-peak-bandwidth/implementation_notes.md` §16-18 — measured
  decode rate, resident memory, why the bandwidth ceiling is 17.2 tok/s (this is
  why the chat does not attempt to make generation faster)
- `crates/engine/src/generate.rs:218,258,300-350,849` — the reset and the
  check-EOS-before-forward ordering
- `crates/engine/src/model.rs:3684,3774,4407,3273,1718` — the forward entry
  points, position reporting, cache cap, on-demand growth
- `crates/engine/src/gpu_cache.rs:104,162,230` — `next_position`,
  `ensure_capacity`, `reserve`'s contiguity invariant
- `crates/engine/src/tokenizer.rs:87,119,180,230` — `encode_prompt`,
  `encode_text`, `Detokenizer::push`/`finish`
- `~/.kaplumba/models/Gxma-repacked/chat_template.jinja` — the authoritative
  template; `generation_config.json` — `eos_token_ids`
- `AGENTS.md` §8 (no Python, single user, memory), §9 (spec discipline),
  §10 (one model process, `gpuq`, `phys_footprint`)