# Task Checklist: Sub-300ms TTFT in Interactive `--chat`

- [x] **Task 1: Warmup `forward_step_pipelined` in `model.rs` (R1)**
  - **Files Touched**: `crates/engine/src/model.rs`
  - **Intent**: Exercise `forward_step_pipelined` with dummy inputs in `warmup_decode` so all Metal pipelines and device-side argmax/sampling kernels are pre-compiled and resident in GPU instruction memory prior to Turn 1.
  - **Testing**: `cargo test --package kaplumba-engine --lib model` (6/6 passed).

- [ ] **Task 2: Token 1 immediate streaming and `TurnStats::ttft_ms` in `chat.rs` (R2, R3, R4)**
  - **Files Touched**: `crates/engine/src/chat.rs`
  - **Intent**: Track `turn_started`, record `ttft_ms` immediately after prefill completes, stream Token 1 text via `on_text` without delay, and store `ttft_ms` in `TurnStats`.
  - **Testing**: `cargo test --package kaplumba-engine --lib chat`.

- [ ] **Task 3: Turn Rate TTFT telemetry in `crates/cli/src/chat.rs` (R3)**
  - **Files Touched**: `crates/cli/src/chat.rs`
  - **Intent**: Update `print_turn_rate` to display `[ttft: {stats.ttft_ms:.0} ms, ...]` on every turn.
  - **Testing**: `cargo check --workspace`.

- [ ] **Task 4: Automated test probe & sabotage verification (AC1, AC2)**
  - **Files Touched**: `crates/engine/tests/probe_chat_ttft.rs`, `spec/chat-ttft-300ms/implementation_notes.md`
  - **Intent**: Implement unit probe asserting immediate Token 1 streaming and `ttft_ms` recording. Sabotage Token 1 delivery (delay to decode step 1), observe test failure, revert sabotage, and confirm pass.
  - **Testing**: `cargo test --package kaplumba-engine --test probe_chat_ttft`.

- [ ] **Task 5: Workspace tests and end-to-end verification (AC3, AC4)**
  - **Files Touched**: `spec/chat-ttft-300ms/implementation_notes.md`
  - **Intent**: Run full workspace test suite (`cargo test --workspace --lib`, `multi_turn_chat.rs`), verify release binary build, and record final findings.
  - **Testing**: `cargo test --workspace --lib` and `cargo test --package kaplumba-engine --test multi_turn_chat`.
