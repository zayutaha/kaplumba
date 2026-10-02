# plan — mtp-40tps

## Runtime chosen

`mlx-vlm 0.6.8` / `mlx-lm 0.31.3` / `mlx 0.32.0`, installed as uv tools under
`~/.local/share/uv/tools/`. No new venv is needed; the drafter README's
`uv run mlx_vlm.generate --draft-kind mtp` maps to the already-installed
`mlx_vlm.generate` entry point.

## Model layout

| role | path | size | notes |
|---|---|---|---|
| drafter | `~/.omlx/models/mlx-community/gemma-4-31B-it-qat-assistant-4bit` | 0.264 GB | `Gemma4AssistantForCausalLM`, 4 layers, `block_size=4` default, shares target K/V |
| target | `~/hf_models/gemma-4-31b-it-4bit` (`mlx-community/gemma-4-31b-it-4bit`) | 18.41 GB on disk | `Gemma4ForConditionalGeneration`, 60 layers (48 sliding w=1024 + 12 full), hidden 5376, tie_word_embeddings, affine 4-bit g64 |

Per-tensor byte accounting of the target (read from the four safetensors headers
over HTTP range requests, `../opencode/hdr2.py`):

```
text_linear          16.474 GB   (all 60 layers' quantised q/k/v/o + mlp)
embed_tokens          0.793 GB   (262144 x 5376, itself affine-4bit g64)
norms + layer_scalar  0.003 GB
vision tower          1.143 GB   (strippable for a text-only run)
                    ---------
total                18.412 GB   ->  17.27 GB text-only
```

## Data flow of one MTP round (`mlx_vlm/speculative/mtp.py:_mtp_rounds`, :523)

1. drafter autoregresses `block_size-1` candidate tokens from
   `[embed(last_bonus) ; target_hidden]`, reading the *target's* shared K/V
   (`gemma4_assistant.py:217 draft_block`).
2. target runs **one** forward over `[last_bonus, cand_1 … cand_k]` — a single
   weight read, `k+1` positions of hidden state (`mtp.py:163`).
3. acceptance walk compares each candidate against an independent sample from the
   target distribution and stops at the first mismatch, emitting the target sample
   as the bonus token.
4. `rollback_speculative_cache` trims the target KV; the drafter's shared-KV view is
   re-sliced.

## Interfaces used

- `mlx_vlm.generate.dispatch.stream_generate(...)` — wrapped by a scratch harness
  so per-token wall-clock intervals can be recorded. The stock CLI prints only
  average tok/s and has **no `--top-k`**, which the bar requires.
- `mlx_vlm.sample_utils.make_sampler(temp=1.0, top_k=64)`.
- `mlx_vlm.speculative.drafters.load_drafter(path, kind="mtp")`.

## Measurement discipline

- Exactly one model process; `pgrep -fl 'mlx_vlm|omlx|kaplumba'` must be empty first.
- A/B interleaved (A B A B …), median of ≥3 runs, spread reported.
- Memory from `top -l 1 -n 0 | grep PhysMem` (`unused`) and `sysctl -n vm.swapusage`.
  `vm_stat` `Pages free` is used only to *decompose* memory, never as a pressure signal.

## Risks

1. **Memory (primary).** 24 GiB box vs 17.27 GB of mandatory-resident weights.
2. **Roofline.** 17.27 GB / 220 GB/s = 78.5 ms per target pass = 12.74 tok/s ceiling
   before any speculation.
3. **Acceptance at temp 1.0.** The accept rule's per-position rate is bounded by
   the p–q overlap, which shrinks sharply as the target distribution flattens.
4. **Network.** huggingface.co delivers ~1 MB/s on this box; the 18.41 GB fetch is
   the wall-clock long pole.

## Requirement → task map

| req | task |
|---|---|
| R1 | T1 target acquisition |
| R4 | T2 verification-source audit + synthetic exactness test |
| R2 | T3 baseline harness + runs |
| R3 | T4 MTP harness + runs |
| R5 | T5 block-size / KV / flag sweep to the bar |
| R6 | T6 distribution-faithfulness test |
| R7 | T7 coherence / multi-turn test |
| — | T8 write-up |