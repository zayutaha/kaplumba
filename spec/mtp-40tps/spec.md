# spec/mtp-40tps — 40 tok/s decode for Gemma 4 31B-it via MTP speculative decoding

## Goal

Sustain a **median ≥ 40 tok/s** decode rate for `gemma-4-31B-it` on this M4 Pro
(20 GPU cores, 24 GiB unified memory, ~220 GB/s achievable DRAM), using Multi-Token
Prediction speculative decoding, with output that is coherent and
**distribution-faithful at temperature 1.0, top_k = 64**.

## Runtime: kaplumba Rust+Metal engine (AMENDED 2026-10-02)

Originally written for mlx-vlm with a freshly downloaded 18.41 GB
`mlx-community/gemma-4-31b-it-4bit` target. **That direction was abandoned** — the
user rejected it and the numbers are strictly worse:

| | Gxma (EXL3, Rust engine) | mlx-community 4bit (mlx-vlm) |
|---|---|---|
| resident weights | **11.6 GiB** | 17.27 GiB |
| headroom on 24 GiB | comfortable | **0.09 GiB left** (measured) |
| per-pass ceiling | **19 tok/s already achieved** | 12.74 tok/s computed |
| accepted tokens/pass needed for 40 t/s | **2.1** | 3.14 |

MTP is being integrated into `crates/engine` against the already-local
`~/.kaplumba/models/Gxma/Gxma`. The drafter must be converted from MLX affine-4bit
into the engine's own format. No new multi-GB model download is required.

The mlx-vlm source stays the **reference implementation** to port the verify rule
from — it is read-only reference material under
`~/.local/share/uv/tools/mlx-vlm/`.

## Non-goals / restrictions

- No Python in `crates/` or `scripts/` (AGENTS.md §8). All Python/MLX work lives in
  `/private/var/folders/tp/.../opencode/` (scratch) and `~/.omlx/`.
  **Weight conversion must be a Rust tool, not a Python script in the repo.**
- No multi-user / batching work. Single stream only.
- No KV-cache quantization (§8: "No KV cache quantization").
- No quality loss: speculative decoding must be distribution-exact, not
  "accept the draft and hope".
- Do not kill another agent's model process; exactly one model process at a time.
- Do not report a peak number, a short burst, or a single run as the result.
- **Do not use `vm_stat` `Pages free` as a memory-pressure signal** (§10.3). It
  counts only never-touched pages and ignores the file cache.

## Requirements

1. **R1 — drafter weights.** The MLX affine-4bit drafter at
   `~/.omlx/models/mlx-community/gemma-4-31B-it-qat-assistant-4bit` converted into
   the engine's own weight format and loaded, verified against a reference that
   states whether the check is bit-level or distributional.
2. **R2 — baseline.** Measured non-speculative decode of Gxma at
   `temperature 1.0, top_k 64`, 1024 generated tokens, ≥3 runs, interleaved A/B,
   reporting median tok/s and the median of the middle 50 % of per-token intervals.
3. **R3 — MTP.** Same run with the drafter enabled.
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

- **Q1 — the real acceptance rate is unmeasured, and it gates everything.** The
  synthetic probes measured p–q overlap c ≈ 0.20 (flat) to 0.40 (peaked), which
  would put 40 t/s out of reach. Those were *synthetic* distributions, not the real
  drafter against the real target. **This is measured first, before any
  optimization work.**
- **Q2 — does the drafter actually match Gxma?** Both are vocab 262144 with tied
  embeddings, and the drafter's `backbone_hidden_size` 5376 equals Gxma's
  `hidden_size`. Unverified: a drafter whose embedding semantics disagree with the
  target is useless no matter how fast it is.

## Amendment 2026-10-03 — structural gaps acknowledged (was never measured)

Post-review, three problems with the earlier plan as written, recorded so the
next attempt does not re-derive them:

1. **T8 hides a new-kernel project.** "One target forward over
   `[bonus, cand_1..cand_k]`" requires an M=K+1 forward. The engine's only fast
   GEMV path is M=1 (`linear_gemv_*`); there is no measured 3 bpw M>1 kernel in
   this repo. T8 is therefore not "integration" — it is a kernel spec of its
   own. Until it exists and is benchmarked, R5's 40 tok/s rests on a pass that
   does not exist.
2. **Q1/T4 was never run.** T4 is gated behind T5 (drafter conversion), which
   was never done, so the make-or-break acceptance number is still open.
3. **The 3 bpw verify-side advantage is plausible but unproven in our
   engine.** In verify, the book is read once per K+1 tokens, so the lane-load
   amplification that caps M=1 decode at ~200 GB/s should be amortized. Whether
   a 3 bpw M=4..8 path then beats a hypothetical 4 bpw one is a measurement,
   not a premise.

Sequencing consequence: **T5 → T6 → T4(gate) → [only if acceptance ≥ ~0.73]
M>1 verify kernel as its own spec → T7/T8 → …**. T8–T10 in
`implementation_notes` as originally written are frozen until the gate is
answered.