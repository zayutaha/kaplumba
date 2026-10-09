# Implementation Notes: Sub-300ms TTFT in Interactive `--chat`

Append as work proceeds. Factual and dense.

## 0. Hardware & Environment Facts
- **Machine:** Apple M4 Pro (20 GPU cores, 24 GiB unified RAM, 273 GB/s peak bandwidth).
- **OS:** macOS 15+ (Darwin kernel 27.0).
- **Model:** Gemma 4 31B dense (`~/.kaplumba/models/Gxma-repacked`), EXL3 ~3.0 bpw.
- **Weights Residency:** 12.04 GiB wired via `LlamaMlock` + `MTLResidencySet`.
- **Target:** Time-To-First-Token $\le 300\text{ ms}$ on typical interactive chat inputs ($\le 30$ delta tokens).

## 1. Measured Baseline & Root Causes of Latency
- Historical Turn 1 Token 2 stall: **1759 ms** (`spec/decode-14tok-17k-context` Spec §5 Finding 4).
  - Cause: `warmup_decode` ran `forward_buffer(&[1], &[0])` but never executed `forward_step_pipelined`. The Metal pipeline state and GPU timeline allocation for pipelined decode executed cold on Token 2 of Turn 1.
- Turn 2+ Token 2 stall: **331 ms**.
- Buffered Token 1: The user had to wait for Token 2 to be decoded before any characters rendered on screen.
- Total perceived latency before response: **2.0–3.5+ seconds**.

## 2. Optimization Strategy
1. Warm up `forward_step_pipelined` with `HeadPick::Argmax` at model load time.
2. Deliver Token 1 text directly to `on_text` and stdout immediately following prefill wait.
3. Record `ttft_ms` in `TurnStats` and display on every turn in `--chat`.
4. Preserve standing RAM KV cache across turns without re-prefilling past context.

## 3. Sabotage Verification & Task Log (AGENTS.md §3.4)

### Task 1 (`2515904`)
- Warmed up `forward_step_pipelined` with `HeadPick::Argmax` during `warmup_decode` in `crates/engine/src/model.rs`.
- Eliminated the 1759 ms cold Metal pipeline specialization stall on Turn 1 Token 2.

### Task 2 (`42f9d60`)
- Added `ttft_ms: f64` to `TurnStats` and `record_turn` in `crates/engine/src/chat.rs`.
- Added `turn_started = Instant::now()` and calculated `ttft_ms` immediately after prefill forward completion.
- Unit test sabotage: sabotaged `turn_stats_records_ttft_ms` asserting `stats.ttft_ms == 999.0` (expected 125.4). Test failed as expected with `assertion left == right failed; left: 125.4, right: 999.0`. Reverted sabotage and verified 8/8 passed.

### Task 3 (`4406903`)
- Updated `print_turn_rate` in `crates/cli/src/chat.rs` to format `[ttft: {stats.ttft_ms:.0} ms, {stats.decode_tok_s:.2} tok/s last turn, ...]`.
- Updated `/stats` command (`print_stats`) to include `ttft: {stats.ttft_ms:.0} ms`.
- Structural check passed: `multi_turn_chat`'s `the_parser_keeps_state_and_drops_timing` passed without modification because `parse_turn_counter` matches from the right on `, turn `.

### Task 4
- Added automated unit test probe `crates/engine/tests/probe_chat_ttft.rs` verifying:
  1. `TurnStats` captures positive `ttft_ms`.
  2. `turn_started` precedes prefill forward and `ttft_ms` is computed immediately following prefill forward.
  3. Token 1 is actively emitted to `on_text` strictly before entering `loop {` and before `previous.wait()`.
- **Sabotage execution:**
  - Sabotaged Token 1 emission by deleting pre-loop `on_text(&chunk)` call in `crates/engine/src/chat.rs`.
  - Executed `cargo test --package kaplumba-engine --test probe_chat_ttft`.
  - Observed diagnosable failure:
    `thread 'test_token_1_streamed_before_decode_loop' panicked at crates/engine/tests/probe_chat_ttft.rs:58:5:`
    `token 1 must have an active uncommented on_text(&chunk) call BEFORE loop begins! Emitting inside or after the loop introduces a full decode step delay to TTFT.`
  - Reverted sabotage and verified all 3 probe tests passed cleanly in 0.00s.

