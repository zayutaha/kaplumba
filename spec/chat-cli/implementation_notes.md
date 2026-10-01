# IMPLEMENTATION NOTES: Interactive Chat REPL

Running log. Implementation on branch `chat-cli` (recovered via reflog after
an external `git reset fc36ee1` wiped the branch pointer mid-session; the
commits survived as objects and the branch now protects them).

---

## 1. The single-token staging bug (pre-existing, fixed here)

**Symptom.** `--chat` at `--temperature 0` answered "hi" with
"HelloSSSSSSSSS" while `--prompt "hi"` answered "Hello! How can I help you
today?" + EOS. First token agreed (9259); trajectories diverged at token 2.

**Bisection.**
- Same seed at temperature 1.0: `--chat` and `--prompt` produce byte-identical
  text ("HelloMSSLLKCL single"). So the chat's forward is numerically identical
  to `--prompt`'s; the defect is temp-0-specific.
- `KAPLUMBA_NO_PIPELINE=1 --prompt` at temp 0 reproduces the chat's answer
  exactly (ids `9259 236773 ×9`). So the split is pipelined vs non-pipelined
  decode, and the chat faithfully reproduces the non-pipelined behavior.

**Root cause.** A one-token forward reads its input from the decode slot
buffers (`ForwardScratch::active_token_ids(1)` /
`active_query_positions(1)`, `model.rs:2808`), which only
`forward_step_pipelined` stages (`model.rs:3639-3647`). The host-staged copy in
`forward_chunk` runs only for `tokens > 1` (`model.rs:3924`). So every
single-token call through `forward_buffer_into` embedded whatever the slot
buffers still held — zeros on a fresh model — i.e. token 0 at position 0, every
step. First token comes from the (correct, host-staged, multi-token) prefill,
then one repeated token forever. This silently broke `--temperature > 0`
(`generate.rs:337`), `KAPLUMBA_NO_PIPELINE` (`generate.rs:1032`), and any
single-token `forward_buffer_into_vec` caller. Only the pipelined greedy path
ever worked, which is why the bit-exact suites (kernel-level, multi-token) and
the "Paris" golden prompt (pipelined temp-0) never caught it.

**Fix** (`model.rs`, in `forward_buffer_into` right after validation): if
`token_ids.len() == 1`, write the token and position into
`token_ids_decode[active_slot]` / `query_positions_decode[active_slot]`. No
`flush_and_wait`: `forward_buffer_into_vec` ended its previous call with
`commit_and_wait` so nothing is in flight, and `forward_buffer_into_step`
already documents the barrier as its caller's duty. The pipelined path calls
`forward_chunk` directly and never comes through here, so its overlap is
untouched. 29-line diff including the comment.

**Proof.** After the fix, `--chat` temp-0 "hi" gives "Hello! How can I help you
today?" + EOS, identical to `--prompt`. `KAPLUMBA_NO_PIPELINE=1 --prompt` agrees
too. All 13 bit-exact tests still pass (the fix only adds writes on a path that
previously read uninitialized buffers; multi-token numerics are untouched).

**Sabotage check** (AGENTS.md §3.4): temporarily wrote `token_ids[0] + 1` into
the slot buffer; "hi" decoded as "Hello Hello! Hellot Helloaryier Hello" —
visibly corrupt, confirming the staging is load-bearing. Reverted; rebuilt;
re-verified the correct answer.

## 2. The stdin drain (chat bug, fixed)

`main.rs` called `read_prompt()` (which reads stdin to EOF) before the chat
branch, so `--chat` always saw empty stdin and exited after one prompt. The
loop was correct; the dispatch order was not. Fix: skip `read_prompt` when
`--chat` is set. Diagnosed via `lseek(0, SEEK_CUR)` == file size before the
first read, with temporary debug output that was removed afterwards.

## 3. Acceptance results (all on `Gxma-repacked`, greedy unless noted)

- **AC1**: turn 1 "Please remember the word platypus. Reply with just OK." →
  `OK`; turn 2 "What word did I ask you to remember?..." → `platypus`. Context
  carries. Turn 2 prefilled 29 tokens (the delta), not the conversation (AC3).
- **AC2**: `--chat` turn 1 "hi" == `--prompt "hi"` temp-0 output, both text and
  (by the `render_first_turn == encode_prompt` unit test) ids. Plus the
  `chat_render` concatenation test proves turn1+turn2 ids equal whole-string
  tokenization.
- **AC4**: chunks print with per-chunk flush; observed incrementally in the
  terminal. (In a pipe the same bytes arrive; nothing buffers a full turn.)
- **AC5**: not driven to 8192 on GPU (350+ s of prefill for one check). The
  boundary logic is extracted as pure `fits_in_context` with a unit test at
  8191/8192/8193; the error carries used/need/limit and the turn is refused,
  never truncated.
- **AC6**: `crates/kernels/` untouched; `model.rs` has only the §1 fix.
- **AC7**: `chat_render.rs` (3 ignored-GPU tests, run with
  `KAPLUMBA_TEST_MODEL`), `chat::tests::context_limit_boundary`,
  `args::tests::chat_flag_and_conflicts`. Full workspace suite green; the five
  bit-exact suites 13/13 green.
- **AC8**: `--prompt "The capital of France is"` → `Paris`, temp 0.

## 4. Environment hazards hit during this work

- An external `git reset fc36ee1` (another agent's cleanup, or the user's)
  wiped six commits mid-session. Objects survived; recovered with
  `git checkout -b chat-cli cf7b6b0` + `git stash` round-trip for the one
  uncommitted file. Lesson: **do feature work on a branch, never on master**,
  in this repo while ~30 worktrees are live.
- A concurrent process rebuilt `target/release/kaplumba` from old source into
  the shared `target/` (AGENTS.md §10.2.4's exact hazard), producing a stale
  binary that rejected `--chat`. Rebuilding fixed it. If `--chat` is ever
  "unknown" again, check `strings target/release/kaplumba | grep -c
  "kaplumba chat"` before debugging the CLI.
- `scripts/gpuq run` (and bare `< file` redirects) deliver stdin correctly;
  the earlier "EOF immediately" was the §2 bug, not the harness.

## 5. Commands

```bash
cargo test --release --workspace
cargo test --release --test live_kernel_bit_exact --test fused_bit_exact \
  --test repacked_golden --test argmax_engine_parity --test norm_rope_qkv_bit_exact
KAPLUMBA_TEST_MODEL=~/.kaplumba/models/Gxma-repacked cargo test --release \
  --test chat_render -- --ignored
printf 'hi\n/stats\n' | scripts/gpuq run -- ./target/release/kaplumba \
  --model ~/.kaplumba/models/Gxma-repacked --chat --temperature 0 --max-new-tokens 10
```
