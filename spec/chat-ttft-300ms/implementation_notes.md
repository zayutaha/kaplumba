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
