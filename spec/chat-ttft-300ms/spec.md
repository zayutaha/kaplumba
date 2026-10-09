# Specification: Sub-300ms TTFT in Interactive `--chat`

## 1. Goal
In interactive `--chat`, eliminate all extraneous startup and step transition delays so that Time-To-First-Token (TTFT) is strictly bounded by the raw prefill computation time plus at most 100 ms of overhead:
$$\text{TTFT} \le \text{Prefill Time} + 100\text{ ms}$$
For typical chat inputs (15–30 tokens), the model must respond in **$\le 300\text{ ms}$** across all turns (Turn 1 and subsequent turns). The 1.76-second first-decode stall is eliminated via startup warmup, Token 1 streams to stdout immediately, previous turns remain resident in RAM KV cache without re-prefill, and per-turn TTFT is displayed transparently.

---

## 2. Non-Goals & Restrictions
- **No KV cache quantization** (AGENTS.md §8 standing constraint).
- **No quality loss** (greedy-identical output; AC2 golden parity).
- **No Python in product code** (AGENTS.md §8 standing constraint; pure Rust + Metal).
- **No regressions in single-turn `--prompt` or `--benchmark`**.
- **No O(L²) scratch memory expansion**.

---

## 3. Requirements

### R1 — Pipelined Decode Warmup during Model Load
[`TextModel::warmup_decode`](file:///Users/zayaantaha/coding/kaplumba/crates/engine/src/model.rs#L3859) must execute a dummy step of [`forward_step_pipelined`](file:///Users/zayaantaha/coding/kaplumba/crates/engine/src/model.rs#L4023) in addition to `forward_buffer`. This forces Metal to compile, specialize, and load the pipelined decode dispatches, slot buffers, and device argmax/sampling kernels into GPU instruction memory prior to session start, eliminating the 1759 ms first-decode stall on Turn 1.

### R2 — Immediate Token 1 Streaming
In [`ChatSession::turn`](file:///Users/zayaantaha/coding/kaplumba/crates/engine/src/chat.rs#L340), Token 1 produced by the prefill argmax must be decoded and streamed to stdout immediately via `on_text` as soon as prefill completes, prior to encoding or waiting on subsequent decode steps.

### R3 — Per-Turn TTFT Telemetry
[`TurnStats`](file:///Users/zayaantaha/coding/kaplumba/crates/engine/src/chat.rs#L210) must record `ttft_ms` (elapsed time from turn start to Token 1 production/emission). In `--chat`, each turn summary must print `ttft: {stats.ttft_ms:.0} ms` alongside decode tok/s.

### R4 — Standing RAM KV Cache & Delta-Only Prefill
Across multiple turns in `--chat`, the KV cache must remain resident in GPU/unified RAM. Each new turn $n > 1$ must strictly prefill only the delta tokens formatted by `render_next_turn`, beginning at `gpu_next_position()`.

### R5 — Strict Overhead Bound
The non-prefill overhead in TTFT (tokenization, encoder creation, buffer staging, host readback, and stdout flush) must be $\le 100\text{ ms}$. For prompts where prefill computation is under 200 ms (e.g. $\le 25$ tokens), total TTFT must not exceed 300 ms.

---

## 4. Acceptance Criteria
1. **Automated Probe Pass:** [`probe_chat_ttft.rs`](file:///Users/zayaantaha/coding/kaplumba/crates/engine/tests/probe_chat_ttft.rs) passes, validating that `ttft_ms` is recorded and Token 1 is emitted before any decode wait.
2. **Sabotage Verification:** Sabotaging Token 1 streaming (e.g. delaying emission until after step 0 decode wait) causes `probe_chat_ttft` to fail with an explicit diagnostic message.
3. **Regression Safety:** Workspace tests pass (`cargo test --workspace --lib` and `multi_turn_chat.rs`).
4. **End-to-End Latency:** Measured `ttft_ms` in `kaplumba --chat` meets $\le 300\text{ ms}$ on short interactive turns.

---

## 5. Open Questions
*(None remaining; resolved during clarification interview.)*
