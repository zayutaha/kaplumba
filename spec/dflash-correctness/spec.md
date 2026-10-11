# spec/dflash-correctness — the z-lab DFlash drafter on Gxma-repacked, greedy

## Goal

Run speculative decoding with the real, already-trained **z-lab DFlash**
drafter at `~/.kaplumba/models/Drafter` against the EXL3 3-bit target
`~/.kaplumba/models/Gxma-repacked`, and reach **>= 2.4 accepted tokens per
target pass** in greedy decode while the emitted sequence stays
**bit-identical** to non-speculative greedy decode.

This is the "is the thing even correct" half. It delivers a drafter that is
loaded, forward-ed, and verified against golden reference arrays, plus the
whole speculative loop (context-feature fusion, KV injection, block diffusion
draft, target verification) and an acceptance measurement. Non-greedy
(`temperature > 0`) distribution-exact sampling is deliberately **not** here —
it is `spec/dflash-sampling`.

## Why this replaces our own drafter plans

Three earlier specs each spent effort on a drafter we did not have:

| spec | outcome |
|---|---|
| `spec/mtp-40tps` | the released `Gemma4AssistantForCausalLM` MTP drafter measured **0.883 accepted/round** and a depth-1 hit rate of 63% |
| `spec/kv-free-drafter` | designed an architecture *we would have to train*; superseded by this spec, no training needed |
| `spec/decode-*` | pure-decode roofline work; the 3-bit ceiling is bandwidth, not code |

DFlash drafts an entire 16-token block in **one** parallel forward instead of
15 sequential autoregressive ones, and conditions on the target's own
hidden states rather than a small model's guesses. exllamav3, which supports
this exact `DFlashDraftModel` family, measured **2.4–2.9 accepted/round** on
`gemma4-31b-it-dflash`; the paper reports tau ≈ 6.5 at block 16 on Qwen3.

## The arithmetic that gates it

```
11.834 GB of 3-bit weights per decode token, ~250 GB/s hardware ceiling
=> single-token decode can never exceed ~21 tok/s; 40 tok/s needs 2.40/pass
tokens_per_pass = tok_s * pass_s
   pass=89.5ms -> 3.58/pass      pass=60ms -> 2.40/pass
```
The 2.4 gate in R9 is that same number, unchanged.

## The drafter, measured from the checkpoint

| | value | source |
|---|---|---|
| quant | EXL3 **4.0 bpw, mcg**, version 0.0.32, `head_bits 6`, `out_scales always` | `quantization_config.json` |
| mcg multiplier | `3417055213` = `0xCBAC1FED` | equals `MCG_MULTIPLIER` in `crates/engine/src/exl3.rs:6` |
| file | 166 tensors, 769 057 588 B, 1.54 G params | `model.safetensors` header |
| encoder | `fc.trellis [2016, 336, 64]` i16 → **32256 in, 5376 out**; `fc.suh [32256]`, `fc.svh [5376]` fp16 | header |
| layers | 5, hidden 5376, **64 q heads × head_dim 128**, **8 kv heads**, inter 10752 | `config.json` |
| taps | `target_layer_ids [1, 12, 23, 35, 46, 57]` of `num_target_layers 60` | `dflash_config` |
| block | `block_size 16`, `mask_token_id 4` | `config.json` |
| windows | `layer_types` = 4 × `sliding_attention` + 1 × `full_attention`, `sliding_window 2048` | `config.json` |
| head | **none of its own** — the target's tied `embed_tokens` and softcap 30.0 | `tie_word_embeddings true`, `final_logit_softcapping 30.0` |

## How DFlash works (the part the engine does not have yet)

Per decode round:

1. **Target taps.** The target's forward emits the post-layer hidden states at
   the 6 tap layers for every position it processes. `extract_context_feature`
   uses `offset = 1`, i.e. `hidden_states[layer_id + 1]`, which in our
   0-based `DecoderLayer` indexing is **the output of layers 1, 12, 23, 35,
   46, 57** — the same numbers the config states. (exllamav3's `tap_shift = 1`
   is the same offset expressed against its own export convention;
   `architecture/dflash.py:69` records that `gemma4-31b-it-dflash` wants it:
   2.4–2.9 accepted/round with it, 0.3 without.)
2. **Fusion.** The 6 taps are concatenated to 32256, projected by `fc`, then
   RMS-normed by `hidden_norm`.
3. **KV injection.** That fused feature, and *not* the drafter's own token
   embeddings, supplies K/V for every position of the drafter's history. The
   drafter layers run `k_proj`/`v_proj` on it, apply RoPE at the token's real
   position, and write it into the drafter's KV cache. This is the difference
   from EAGLE-3, which fuses target features only at the block input; the paper
   (§5.5.5) measures KV injection at 4.2 vs 3.5 tau.
4. **Block diffusion.** One 16-token block `[last accepted token, mask × 15]`
   is embedded **raw** (`F.embedding`, no `sqrt(hidden)`), run through 5
   layers, final-normed, lifted through the target's head with softcap 30.0,
   and argmaxed. All 15 future tokens come from a single pass.
5. **Verification.** The target runs the 16 tokens and greedily accepts the
   longest prefix match; the first mismatch position is the bonus token.

## Non-goals / restrictions

1. **No training.** The drafter arrives trained. Nothing in this spec writes
   weights.
2. **No quality loss.** Greedy output must be bit-identical to
   non-speculative greedy decode. This is a hard gate (AC2), not an average.
3. **No Python in `crates/` or `scripts/`** (AGENTS.md §8). Any reference
   dumping is a throwaway harness in
   `/private/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/`,
   never committed.
4. **No change to the target's numerics.** The EXL3 3-bit mcg colmajor decode
   path is not touched; DFlash adds a *reader* of its hidden states, not a
   different way to compute them.
5. **No KV cache quantization**, no multi-token parallelism beyond the DFlash
   block, no quality knobs on the target.
6. **No scratch that grows with context.** The 6 tap hidden states for a whole
   prompt are 6 × 5376 × 2 B per position — 528 MB at 8192. Prefill tap
   capture must be chunked and streamed into the drafter cache, never
   materialised whole (AGENTS.md §8).
7. **The MTP drafter path stays.** `crates/engine/src/drafter.rs`,
   `drafter_model.rs`, and every test that uses them keep compiling and
   passing; the DFlash drafter lives in its own module.
8. **One model process at a time**, `scripts/gpuq` discipline (AGENTS.md §10).
9. **Do not report a coverage claim without sabotage evidence** (AGENTS.md
   §3.4). Every new kernel or numerical path in this spec is checked by
   breaking it on purpose and confirming the test fails diagnosably.
10. **Do not add machinery on the hot path without a profile.** The R11
    benchmark decides whether `gemm_chunk` at M=16 is good enough; a new
    kernel is only written if it is not.

## Requirements

### Loading and configuration

- **R1 — drafter loader.** Load `~/.kaplumba/models/Drafter/model.safetensors`
  (single shard) as EXL3 mcg 4.0 bpw, honouring the packed-trellis and scale
  layout already validated in `crates/engine/src/exl3.rs`. Every one of the 166
  tensors present in the header must be either loaded or explicitly rejected
  with a named error — no silent skip. `mc_g` per tensor resolves to
  `MCG_MULTIPLIER`; a mismatch is a hard error.
- **R2 — norm offset.** Every RMSNorm weight in the drafter
  (`hidden_norm`, `norm`, `layers.*.input_layernorm`,
  `layers.*.post_attention_layernorm`, `layers.*.self_attn.q_norm`,
  `layers.*.self_attn.k_norm`) is stored **without** Gemma's `+1` offset:
  `layers.0.input_layernorm.weight` measures 0.30–0.82 BF16 and our
  `rms_norm_weighted` (`crates/kernels/src/elements.metal:110`) multiplies by
  `weight` alone. The loader adds `1.0` after BF16→F16 conversion. A test
  asserts the offset is present, with sabotage evidence.
- **R3 — drafter config.** Parse `block_size`, `mask_token_id`,
  `target_layer_ids` (sorted, unique, all < `num_target_layers`),
  `layer_types`, `sliding_window`, `num_hidden_layers`, `hidden_size`,
  `num_attention_heads`, `num_key_value_heads`, `head_dim`,
  `intermediate_size`, `rms_norm_eps`, `rope_theta`,
  `final_logit_softcapping`. Unknown-but-present keys inside `dflash_config`
  (e.g. `tap_shift`) must be read and reported, not ignored.

### Target-side feature export

- **R4 — tap export.** A target forward returns, for a set of layer indices,
  the post-layer hidden state of those layers for **every** position of the
  pass. Prefill path chunks it (R6); the decode/verify path runs M=16.
  Existing `Model::hidden_states_all` (`model.rs:5474`) is the model for
  chunking but only exports the post-final-norm row — this is a new path.
- **R5 — zero cost when unused.** With the drafter disabled, the tap export
  adds no work and no allocation. `KAPLUMBA_GPU_PROFILE` on the same prompt
  with and without DFlash must show identical target kernel time; the
  difference is the drafter itself.
- **R6 — chunked prefill.** Tap capture during prefill processes at most
  `PREFILL_CHUNK` positions at a time and streams each chunk through `fc`,
  `k_proj`, `v_proj` into the drafter cache, then releases it. Peak tap
  scratch must be independent of prompt length and reported as a number.

### Drafter forward

- **R7 — block construction.** Input is `[token_at_kv_position]` followed by
  `block_size - 1` copies of `mask_token_id`, embedded **raw** (no
  `sqrt(hidden)` scaling) at F16. A second, flag-gated variant applies the
  target's `sqrt(hidden)` scale; Q1 in the open questions decides which is
  the default.
- **R8 — attention mask (z-lab semantics).** `reference` behaviour, from
  `dflash/model.py:157` and `:367`:
  - layers whose `layer_types[i] == "sliding_attention"`
    (`is_causal` absent from the config → `is_causal = True`):
    `key_pos <= q_pos` **and** `q_pos - key_pos < 2048`.
  - the `full_attention` layer (`is_causal = False`, `sliding_window = None`):
    **fully bidirectional** within the block.
  A `KAPLUMBA_DFLASH_CAUSAL_BLOCK=1` flag switches every layer to the
  exllamav3 reading (causal everywhere) so the two can be A/B-measured
  without a rebuild.
- **R9 — RoPE.** `RopeStyle::Neox` (rotate_half), `theta = 1e6` on **all five**
  layers — the drafter config has one `rope_theta`, no separate local base and
  no `rope_scaling`. Full rotation over all 128 dims, no partial factor.
  Applied to q and to k (both the context K and the block K), never to v.
- **R10 — layer arithmetic, exactly.**
  ```
  h = norm(input)                                   q = q_norm(q_proj(h))
  k = k_norm(k_proj(ctx))  ;  v = v_proj(ctx)       (context rows, RoPEed)
  kn = k_proj(h)  ;  vn = v_proj(h)                 (block rows, RoPEed)
  k = cat([k, kn]) ; v = cat([v, vn])
  o = o_proj(attn(q, k, v)) ; h = h + o
  m = silu(gate_proj(norm(h))) * up_proj(norm(h))
  h = h + down_proj(m)
  ```
  No layer_scale, no attention sinks, no value scale, no bias anywhere.
- **R11 — head.** `norm` (RMSNorm +1), then the target's tied embedding table
  as the output head, then `tanh(logits / 30.0) * 30.0`
  (`final_logit_softcapping`). Greedy argmax of the softcapped logits.

### Drafter state

- **R12 — drafter KV cache.** A separate cache, **5 layers × 8 kv heads ×
  128 dim**, K and V in F16. Layers 0–3 are ring caches of 2048 positions
  (the drafter's own `sliding_window`); layer 4 is a full cache. Sized from
  the drafter's config, not the target's. Per token this is 20 480 B, ~168 MB
  at 8192.
- **R13 — cache content.** The persistent drafter cache holds **only**
  rows derived from the fused target context feature. The block's own K/V
  exist for the duration of the block pass and must not survive into the next
  round in a way that changes the next round's output
  (z-lab `_crop_to(past_key_values_draft, start)`); exllamav3 instead
  overwrites them next round. Either is acceptable *provided* the equivalence
  is asserted by a test that runs two rounds and checks the second round's
  draft is unchanged when the block K/V of round one were left in place.
- **R14 — rewind.** When verification rejects at position `a`, the drafter's
  cache and window rings must end at exactly the accepted length `a + 1` (the
  target's new `kv_position`). The pre-written, rejected rows are dropped.

### Loop and correctness

- **R15 — the round.** One round = drafter block pass → 15 draft tokens →
  target verify pass over 16 positions → greedy acceptance → advance by
  `accepted + 1` → refresh the drafter cache rows for the accepted positions
  from the freshly exported taps. Mirrors `dflash_generate`
  (`dflash/model.py:180`).
- **R16 — golden reference.** Dump the drafter's per-stage activations
  (`fc`, `hidden_norm`, each layer's `q/k/v/o/mlp`, final `norm`, logits) for
  a fixed prompt from the z-lab reference implementation, and check the
  engine's values against them. State in the test header whether the
  comparison is bit-level or distributional, and run it on the **release**
  build at the same shapes used in production.
- **R17 — greedy equivalence.** Greedy output with DFlash must equal greedy
  output without it, token for token, over >= 5 prompts of >= 200 tokens
  each. Near-ties where the top two logits differ by < 0.25 are exempt
  (exllamav3's `test_speculative.py` uses the same `NEAR_TIE`).
- **R18 — acceptance.** Measured accepted/round over >= 200 rounds of greedy
  decode on held-out prompts. **Gate: >= 2.4.** Report the median, the
  spread, and the per-depth acceptance histogram.
- **R19 — kernel decision.** Benchmark `gemm_chunk` at M=16 for every drafter
  shape: `fc` 32256→5376, `q_proj` 5376→8192, `k_proj`/`v_proj` 5376→1024,
  `o_proj` 8192→5376, `gate`/`up`/`down` 5376→10752, 10752→5376. Report ms
  and GB/s effective. If the whole block is < 4 ms, use it as-is; otherwise
  open a follow-up spec for a fused M=16 path. AGENTS.md: profile before
  adding machinery.
- **R20 — memory.** Measure the process peak footprint
  (`/usr/bin/time -l`, `peak memory footprint`, **not**
  `peak_resident_pages_bytes` — AGENTS.md §10.3) at 1k/2k/8k context with the
  drafter resident. Budget: current 12.96 GiB + 769 MB weights + 168 MB KV.
  Report the number; if it crosses ~20 GiB the block size must be reduced
  and that must be said out loud.

## Acceptance criteria

- **AC1** all 166 tensors load, shapes and codebook verified against the
  header; a malformed file fails with a named error.
- **AC2** R16 golden reference passes, with sabotage evidence recorded in
  `implementation_notes.md` (what was broken, the failure message, the revert
  commit).
- **AC3** R17 greedy equivalence passes on >= 5 prompts × >= 200 tokens.
- **AC4** R18 measured accepted/round **>= 2.4**, median with spread, over
  >= 200 rounds.
- **AC5** R19 benchmark table written into `implementation_notes.md`, and the
  resulting decision (reuse / new kernel) recorded with the number that drove
  it.
- **AC6** R20 peak footprint measured and under the stated budget.
- **AC7** every new numerical or kernel path has sabotage evidence
  (AGENTS.md §3.4).

## Open questions

- **Q1 — raw vs scaled block embedding.** z-lab feeds raw `F.embedding`;
  exllamav3 routes through the target's embedding module, which applies
  `sqrt(5376) = 73.3`. Both cannot be right for one checkpoint. Default to
  raw (it is the training code), measure both, and let acceptance decide.
  If scaled wins, R7's default flips and the reason is written here.
- **Q2 — tap index derivation.** We read the config's `target_layer_ids` as
  0-based `DecoderLayer` indices directly, because `offset = 1` in
  `extract_context_feature` cancels the `hidden_states` list's leading
  embedding entry. If Q1 is resolved and acceptance is still far below 2.4,
  run the +1 / −1 alternatives and record the numbers; a 0.3-vs-2.5 swing is
  the documented symptom (`architecture/dflash.py:69`).
- **Q3 — does `gemm_chunk` at M=16 stay under 4 ms for `fc`?** `fc` reads
  32256 × 5376 × 4 bpw ≈ 86.7 MB per block; at 250 GB/s that is a 350 ms
  floor if it is not reused across the 16 rows. This is the single largest
  drafting cost and R19 exists to settle it.
- **Q4 — does the block's own K/V need RoPE at all?** z-lab RoPEs them
  (`k = cat([k_ctx, k_noise])` then `apply_rotary_pos_emb`) — keep it, but
  note it as the second-most-likely divergence if the golden reference fails
  on the context rows rather than the block rows.
- **Q5 — prompt longer than the drafter's 2048 sliding window.** Layers 0–3
  ring at 2048; a >2048-token prompt means those layers genuinely cannot see
  the prompt's start. The paper's long-context table shows acceptance
  degrading past 4K without fine-tuning, so this is expected behaviour, not a
  bug — but the spec should state the acceptance we measure at 8192 context.
