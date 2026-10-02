# spec/mtp-40tps — 40 tok/s decode for Gemma 4 31B-it via MTP speculative decoding (mlx-vlm)

## Goal

Sustain a **median ≥ 40 tok/s** decode rate for `gemma-4-31B-it` on this M4 Pro
(20 GPU cores, 24 GiB unified memory, ~220 GB/s achievable DRAM), using Multi-Token
Prediction speculative decoding through **omlx / mlx-vlm**, with output that is
coherent and **distribution-faithful at temperature 1.0, top_k = 64**.

Runtime choice is fixed by the user: mlx-vlm. MTP is **not** being ported into
`crates/engine` (explicitly rejected).

## Non-goals / restrictions

- No Python in `crates/` or `scripts/` (AGENTS.md §8). All Python/MLX work lives in
  `/private/var/folders/tp/.../opencode/` (scratch) and `~/.omlx/`.
- No multi-user / batching work. Single stream only.
- No KV-cache quantization (§8: "No KV cache quantization").
- No quality loss: speculative decoding must be distribution-exact, not
  "accept the draft and hope".
- Do not kill another agent's model process; exactly one model process at a time.
- Do not report a peak number, a short burst, or a single run as the result.

## Requirements

1. **R1 — target weights.** An MLX-format 4-bit `gemma-4-31B-it` checkpoint whose
   tokenizer/chat template match the drafter (vocab 262144).
2. **R2 — baseline.** Measured non-speculative decode of the same checkpoint at
   `temperature 1.0, top_k 64`, 1024 generated tokens, ≥3 runs, interleaved A/B,
   reporting median tok/s and the median of the middle 50 % of per-token intervals.
3. **R3 — MTP.** Same run with `--draft-model <drafter> --draft-kind mtp`.
4. **R4 — verification audit.** Read the installed mlx-vlm source and state, with
   `path:line`, whether the non-greedy path is proper speculative sampling.
5. **R5 — the bar.** Median ≥ 40 tok/s over 1024 tokens, median of ≥3 runs, and
   the median of the middle 50 % of per-token intervals also ≥ 40.
6. **R6 — distribution faithfulness.** Many samples from (a) non-speculative and
   (b) MTP decode at the same seed/temperature/top_k; compare token-frequency
   distributions with total-variation distance and a two-sample χ² test with p-values.
7. **R7 — coherence.** Long-form multi-turn chat at temp 1.0 / top_k 64; check for
   degenerate repetition, mid-sentence truncation, EOS handling
   (`eos_token_id = [1, 106, 50]`), and cross-turn coherence, against the
   non-speculative baseline on identical prompts.

## Acceptance criteria

- Both halves of the bar met **on the same measured run**, with spread reported, or
- the measured ceiling is reported together with roofline arithmetic, the measured
  acceptance rate, and what would be required to close the gap.

## Open questions

- None blocking. The blocking constraint is physical and is being measured rather
  than assumed — see `implementation_notes.md`.