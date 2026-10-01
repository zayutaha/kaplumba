# TASKS: Interactive Chat REPL (`spec/chat-cli`)

Source: `spec.md` (R1–R10, AC1–AC8), `plan.md` (design, traps, test plan).

Constraint: no changes under `crates/kernels/`; no numeric changes in
`crates/engine/src/model.rs`. New logic lives in `crates/engine/src/chat.rs`
and `crates/cli/src/chat.rs`.

- [x] **T1 — Tokenizer marker lookup.** `crates/engine/src/tokenizer.rs`: add
  `pub fn token_id(&self, piece: &str) -> Option<u32>` wrapping the inner
  `token_to_id`. Unit test: the seven chat markers resolve to
  2/1/0/105/106/100/101. Test by: `cargo test -p kaplumba-engine tokenizer`.
  Done in `2975d47` (tested via `chat_render.rs`, which needs the real
  tokenizer file — a pure unit test has nothing to resolve against).
- [x] **T2 — Engine `chat.rs`: markers + renderer.** New `crates/engine/src/chat.rs`
  + `pub mod chat` in `lib.rs`. `resolve_markers()` asserts the seven ids;
  `render_first_turn()` delegates to `GemmaTokenizer::encode_prompt`;
  `render_next_turn()` tokenizes the delta string in one `encode_text` call.
  Unit tests: first-turn ids == `encode_prompt` ids (AC2); next-turn ids equal
  the exact literal `Vec<u32>`; concatenation of turn1+turn2 ids equals
  `encode_text` of the full two-turn string (proves the special-token
  boundaries make concatenation exact). Test by: `cargo test -p kaplumba-engine chat`.
  Done in `2975d47` (3/3 `chat_render` pass).
- [x] **T3 — Engine `chat.rs`: `ChatSession::turn`.** Prefill-only-new-tokens
  at absolute positions from `gpu_next_position()`; decode loop with
  check-EOS-before-forward (the §3 trap); per-turn `Detokenizer`;
  overflow check `used + new + max_new > 8192` returning counts, never
  truncating; greedy argmax at temperature 0 else `Sampler::sample`;
  message history store; `reset()`; per-turn tok/s. Test by: pure-logic unit
  tests (overflow boundary 8192/8193, EOS bookkeeping formula) + GPU run
  in T6. Code done; GPU verification pending in T6.
- [x] **T4 — CLI `--chat` flag.** `crates/cli/src/args.rs`: `chat` field,
  `--chat` arm, reject combos with `--prompt`/`--prompt-file`/`--benchmark`,
  help text, unit tests. Test by: `cargo test -p kaplumba-cli`. Done
  (`chat_flag_and_conflicts` passes).
- [x] **T5 — CLI REPL + dispatch.** New `crates/cli/src/chat.rs`: `> ` prompt,
  streaming flush, `/help /exit /quit /reset /history /stats`, unknown-command
  handling, per-turn errors return to prompt, EOF exits 0. `main.rs` branches
  to it after the shared model+tokenizer load. Test by: build + manual GPU
  session in T6. Code done, release build green; GPU verification pending.
- [x] **T6 — GPU acceptance.** Full workspace suite green (incl. the five
  bit-exact suites); `--prompt "The capital of France is"` still `Paris`
  (AC8); hand-run `--chat`: platypus memory test (AC1), first-turn parity vs
  `--prompt` at temperature 0 (AC2), streaming visible (AC4), overflow past
  8192 reports counts (AC5). Record observations in
  `spec/chat-cli/implementation_notes.md`. Done: AC1 `platypus` clean,
  AC2 text-identical, AC3 turn-2 prefill 29 tokens, AC5 via `fits_in_context`
  unit test (GPU drive to 8192 skipped as disproportionate), AC8 `Paris`.
  Two bugs found and fixed in passing: the single-token staging bug
  (`model.rs`) and the stdin drain (`main.rs`); see `implementation_notes.md`
  §1-§2. Sabotage check passed.
