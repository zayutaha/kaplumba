# spec/dflash-correctness — implementation notes

Running log. Append as work proceeds (AGENTS.md §5). Nothing below is a
measurement from this spec's own runs yet — the entries marked **MEASURED**
are facts read directly off the checkpoint or the reference code, the ones
marked **TO MEASURE** are the numbers this spec exists to produce.

---

## 2026-10-11 — investigation findings (before implementation)

### What the drafter checkpoint is

`~/.kaplumba/models/Drafter/` — 3 files: `config.json` (1 524 B),
`quantization_config.json` (48 796 B), `model.safetensors` (769 057 588 B,
single shard, header 16 540 B, **166 tensors**).

MEASURED (`python3` over the safetensors header):
- `fc.trellis [2016, 336, 64]` i16 = 86 704 128 B → in 2016·16 = **32256**
  (= 6 × 5376), out 336·16 = **5376**. `fc.suh [32256]`, `fc.svh [5376]` fp16.
- `layers.{0..4}.self_attn.{q,k,v,o}_proj.trellis` i16; q `[336, 512, 64]`
  (5376→8192), k/v `[336, 64, 64]` (5376→1024), o `[512, 336, 64]`
  (8192→5376).
- `layers.{i}.mlp.{gate,up}_proj.trellis [336, 672, 64]` (5376→10752),
  `down_proj.trellis [672, 336, 64]` (10752→5376).
- Norm tensors are **BF16** (`hidden_norm.weight`, `norm.weight`,
  `layers.*.input_layernorm.weight`, `*.post_attention_layernorm.weight`,
  `*.self_attn.q_norm.weight [128]`, `*.self_attn.k_norm.weight [128]`)
  — 2 560 B of BF16 each, nontrivial in count.
- `q_norm`/`k_norm` are per-`head_dim` (128), not per-`hidden`.
- **No `v_norm`, no `layer_scalar`, no `attention_sink_bias`, no
  `mask_embedding`.** exllamav3's `DFlashConfig` (`architecture/dflash.py:83-98`)
  probes for all three; this checkpoint has none.

MEASURED (`quantization_config.json`):
- `quant_method exl3`, `version 0.0.32`, `bits 4.0`, `head_bits 6`,
  `out_scales always`, `codebook mcg`, per-tensor `mcg_multiplier
  3417055213`.
- `3417055213 == 0xCBAC1FED` — **identical** to
  `MCG_MULTIPLIER` in `crates/engine/src/exl3.rs:6`. Same codebook family as
  the 4-bit Gxma checkpoints, but a different `version` string (0.0.32 vs
  0.0.37/0.0.38) and a different bitrate. Do not assume the loader's
  version gate accepts it without checking.

MEASURED (`config.json`): `architectures ["DFlashDraftModel"]`,
`model_type qwen3` (a z-lab template default — the model is Gemma-normed, see
below), `num_hidden_layers 5`, `hidden_size 5376`, `intermediate_size 10752`,
`num_attention_heads 64`, `num_key_value_heads 8`, `head_dim 128`,
`vocab_size 262144`, `tie_word_embeddings true`,
`layer_types ["sliding_attention" ×4, "full_attention"]`,
`sliding_window 2048`, `max_window_layers 5`, `rms_norm_eps 1e-06`,
`rope_theta 1000000`, `rope_scaling null`, `final_logit_softcapping 30.0`,
`block_size 16`, `dflash_config {mask_token_id 4, target_layer_ids [1,12,23,35,46,57]}`,
`num_target_layers 60`, `attention_bias false`, `hidden_act silu`.

Total params ≈ 1.54 G (5 × 272 M layers + 173 M `fc`) → 769 MB at 4 bpw.
MAKES SENSE: 5·(5376·8192 + 2·5376·1024 + 8192·5376 + 3·5376·10752) ≈ 1.362 G,
plus 32256·5376 ≈ 0.173 G.

### The target it pairs with

`~/.kaplumba/models/Gxma-repacked` — MEASURED: `bits 3.0`, `codebook mcg`,
`head_bits 6`, `layout colmajor`, `version 0.0.37`, `out_scales always`.
60 layers, hidden 5376, 32 q heads, 16 kv heads, head_dim 256, sliding_window
1024, `layer_types` = 50 × sliding + 10 × full, `final_logit_softcapping
30.0`, vocab 262144, inter 21504. The drafter's `hidden_size 5376` and
`vocab_size 262144` match the target exactly, which is required for the shared
head.

### The Gemma `+1.0` norm offset (a real, measured trap)

MEASURED, drafter BF16 norm weights, first 6 values:
```
hidden_norm.weight               [1.0938 1.1562 1.2656 1.4141 1.4297 1.1953]
norm.weight                      [3.1562 2.6719 3.1250 3.2188 3.0469 3.0938]
layers.0.input_layernorm.weight  [0.7539 0.4023 0.7969 0.7578 0.8164 0.7070]
layers.0.post_attention_layernorm [0.3125 0.4023 0.3770 0.3672 0.3750 0.2578]
layers.0.self_attn.q_norm.weight [1.8281 1.6094 1.1797 1.8516 1.4609 0.9922]
layers.0.self_attn.k_norm.weight [1.3984 1.4219 1.5859 1.5469 1.3281 0.6797]
```
Values < 1 prove the Gemma `(1 + weight)` offset is **not** baked into this
checkpoint: the trainer used HF's Gemma RMSNorm, which adds 1 at runtime.

MEASURED, by contrast, the repacked **target** checkpoints
(`Gxma-repacked` and `Gxma-uncensored-repacked`,
`model.language_model.norm.weight`, BF16): `[6.188 8.0 5.844 6.094 6.5 6.469
5.875 6.25]`. The engine's `rms_norm_weighted`
(`crates/kernels/src/elements.metal:110`) multiplies by `weight` **only** —
there is no `+1` — and the target decodes correctly, so the repack already
folded the offset in.

**Consequence for the loader**: BF16→F16 convert, then `+1.0`, for the
drafter's norms. Not doing this silently destroys them (a 0.75 weight becomes
0.75 instead of 1.75, i.e. a ~2.3× error on every residual stream).

### Attention mask, block semantics — the two references disagree

z-lab, `dflash/model.py:157` `_attention_mask` + `:367` in
`Qwen3DFlashAttention.__init__`:
```python
is_causal = getattr(config, "is_causal", None)     # absent in our config
self.is_causal = layer_type == "sliding_attention" if is_causal is None else bool(is_causal)
self.sliding_window = config.sliding_window if layer_type == "sliding_attention" else None
# visible &= key_position <= query_position                        (if is_causal)
# visible &= query_position - key_position < sliding_window         (if sliding_window)
# visible &= key_position - query_position < sliding_window         (if not is_causal)
```
→ layers 0–3: causal **and** `q_pos - k_pos < 2048`. Layer 4
(`full_attention`, `is_causal False`, `sliding_window None`):
**no mask at all inside the block**.

exllamav3, `architecture/dflash.py:107` `block_window` returns `(-1, 0)` for
the non-sliding layer and `(2047, 0)` for the sliding ones; the triton paged
kernel (`modules/attention_fn/triton_paged.py:311-316`) turns `window_right
= 0` into `offs_n <= q_abs`, i.e. **causal on every layer**.

→ We implement z-lab (it trained the checkpoint) and keep exllamav3's reading
behind `KAPLUMBA_DFLASH_CAUSAL_BLOCK=1` for A/B.

### Block input embedding — the other disagreement

z-lab `dflash/model.py:135`:
```python
def _raw_input_embeddings(target, input_ids, scale=1.0):
    return F.embedding(input_ids, target.get_input_embeddings().weight) * scale
```
**No `sqrt(hidden)`.** exllamav3's `DFlashInputLayer.forward`
(`modules/arch_specific/dflash.py:121-123`) routes through the target's
embedding module, which applies Gemma's `sqrt(5376) = 73.33`.

→ Default to z-lab (raw), keep the scaled variant behind a flag, let
acceptance decide. Note our engine's `Model::embedding` (`model.rs:5285`)
already does the scaled version, so the raw path is a *new* small helper, not
a modification.

### The target-tap index, resolved

z-lab `dflash/model.py:37`:
```python
def extract_context_feature(hidden_states, layer_ids):
    offset = 1
    selected_states = [hidden_states[layer_id + offset] for layer_id in layer_ids]
```
`hidden_states[0]` is the embedding output, so `hidden_states[l + 1]` is the
**output of engine layer `l`**. The config's `target_layer_ids` are therefore
already the 0-based `DecoderLayer` indices: **[1, 12, 23, 35, 46, 57]**.
exllamav3's `tap_shift = 1` (`architecture/dflash.py:27`, applied at `:74`) is
the same offset against its own convention, and its comment at `:69` records
the measured consequence for this checkpoint family: `+1` gives 2.4–2.9
accepted/round, `0` gives 0.3.

### Engine seams found (so the next session does not re-derive them)

- `crates/engine/src/exl3.rs` — `TILE_SIZE 256`, `HADAMARD_SIZE 128`,
  `MCG_MULTIPLIER 0xCBAC1FED`, `MUL1_MULTIPLIER 0x83DCD12D`,
  `Codebook::{Plain,Mcg,Mul1}`, `BitRate` accepting `1.0..=8.0` integers and
  `1.5/2.5/3.5` fractional. The drafter's `bits 4.0` → `BitRate::Integer(4)`.
- `crates/kernels/src/linear.rs:1171` `gemm_chunk` / `:1064`
  `gemm_chunk_buffer` — the existing M>1 EXL3 path (prefill). M=1 is
  `gemv`/`encode_gemv`. **No EXL3 M=16 kernel exists**; T12 decides whether
  one is needed.
- `crates/engine/src/model.rs:5474` `hidden_states_all` — chunked prefill
  exporter of the post-**final**-norm hidden only; `run_head: false` skips the
  262 144-wide head. It is the model to copy for tap capture, but it does not
  export intermediate layers.
- `crates/engine/src/model.rs:4648` `forward_chunk` — the layer loop at
  `:4820-4840` writes into `hidden_a`/`hidden_b` alternating; the final row
  lives in `hidden_a` for an even layer count (`:4876-4880`). Tapping a layer
  means blitting out of whichever buffer that layer *wrote*.
- `crates/engine/src/model.rs:5285` `embedding` — scaled (×`sqrt(hidden)`),
  reads through `embedding_scaled_view` honouring the offloaded
  page-aligned BF16 table. The raw path must go through the same helper with
  `scale = 1.0`, or it will read garbage for an offloaded table. (The code
  comment at `:5282-5284` records exactly this bug class: a host-mirror
  version made `mtp_acceptance`'s drafter input `-inf`/NaN and the measured
  acceptance read 0.)
- `crates/engine/src/kv_cache.rs` — ring caches for sliding layers, full for
  full-attention; `model.rs:3694` documents `Gxma`'s 10 full + 50 sliding
  split. The drafter's 1 full + 4×2048-ring cache is the same primitives.
- `crates/engine/src/drafter.rs` (96 lines) / `drafter_model.rs` (367) — the
  MTP drafter, **untouched** by this spec. Its loader is the pattern to copy
  for the single-shard BF16→F16 conversion.
- `crates/engine/examples/mtp_acceptance.rs` and `export_hidden.rs` exist;
  T4/T11's harnesses are patterned on them.

### Roofline for the drafting pass (TO CONFIRM with T12)

`fc` alone is `32256 × 5376 × 4 bpw` = **86.7 MB** of trellis read per block
if the weights are not kept resident in cache. At ~250 GB/s that is a **350 ms
floor** — which would make the drafter slower than the target. This is why
T12 is scheduled before T6: the whole design of the block pass depends on
whether `gemm_chunk` at M=16 amortises that read across 16 rows (it must be
one weight read per 16-row block, not per row) or whether a fused path is
required. The same arithmetic applies to the target's own 16-row verify pass
at 3 bpw (11.8 GB read per pass, which is exactly why a verify pass must
amortise the book across 16 tokens).

### Environment facts

- Working tree at investigation time: `5df4065 WIP non mcg quant support`
  (the mul1 work in `spec/mul1-checkpoint-support`).
- `reference/exllamav3` cloned 2026-10-11 at `3a513ac QCache: Fix regression`.
  z-lab's `dflash` reference cloned to
  `/private/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/zlab/dflash`.
- Prompt budget is 8192 tokens; the test sentence is 6.01 tokens
  (AGENTS.md §10.2).

## Dead ends / abandoned approaches (recorded as they happen)

- *(none yet)*

## TO MEASURE

- [ ] T12 — `gemm_chunk` ms and effective GB/s at M=16 for every drafter shape.
- [ ] T11 — accepted/round median + spread + per-depth histogram.
- [ ] T13 — `peak memory footprint` at 1k / 2k / 8k with the drafter resident.
- [ ] Q1 — acceptance under raw vs `sqrt(hidden)`-scaled block embedding.
- [ ] Q2 — acceptance under the ±1 tap-index variants if the gate misses.
- [ ] z-lab mask vs `KAPLUMBA_DFLASH_CAUSAL_BLOCK=1` acceptance.

## 2026-10-11 — T1/T2 landed: loader + Gemma norm offset

**T1 — `crates/engine/src/dflash.rs` (new, ~250 lines).** Single-shard loader
for `~/.kaplumba/models/Drafter`. Builds `QuantizedLinear`s from the EXL3 mcg
packing directly (`.trellis` i16 `[in/16, out/16, 64]`, `.suh`/`.svh` fp16,
`.mcg` i32), so the existing EXL3 gemv/gemm kernels apply unchanged — no new
quantised-matmul path. BF16 `.weight` tensors become the RMSNorm scales.

Design decisions not in the plan:
- **`unknown_tensors` is surfaced, not ignored.** The loader collects every
  header name it does not recognise into a `Vec<String>` and the test asserts
  it is empty. exllamav3's `DFlashConfig`
  (`architecture/dflash.py:83-98`) probes for `attention_sink_bias`,
  `mask_embedding` and per-tap `hidden_norms.*`; our checkpoint has none, and
  if a future one does, this makes the loader refuse rather than half-load.
- **`QuantizedLinear::validate` already enforces `in_features` and
  `out_features` multiples of 128** (`model.rs:429`). Warned in the plan:
  `fc` is 32256 (ok, 252 × 128) and `hidden_norm` is 5376 (ok, 42 × 128), so
  every drafter line passes without an explicit check here.
- **`.su`/`.sv` packed-scale suffixes are accepted but unused** — this
  checkpoint stores `.suh`/`.svh` outright.

**T2 — the Gemma `+1.0` offset, and the trap that proves it matters.** First
working version had a real bug: it widened BF16 via
`bf16→f16 bits`, then reused *those* bits as the top half of an f32, which
converts the value a second time. `hidden_norm[0]` came out 1.0137 instead of
2.0937 and the test caught it immediately. Final form: BF16 widens exactly as
`f32::from_bits(bits << 16)`, add 1.0 in f32, narrow once to f16.

**Sabotage evidence (AGENTS.md §3.4), `dflash_norm_weights_carry_the_gemma_offset`:**
removing the `+1.0` fails with
`hidden_norm[0]: stored 1.09375, file 1.09375 + 1.0 = 2.09375`.
Reverted, re-run green.

**Tests, `crates/engine/tests/dflash_load.rs` (new, 3 tests):**
- `dflash_load_has_every_tensor_and_the_right_shapes` — `unknown_tensors`
  empty; one linear per trellis stem; `fc` = 32256→5376; per-layer q/k/o/gate/
  down shapes; `mcg` = `MCG_MULTIPLIER` for three sampled tensors.
- `dflash_norm_weights_carry_the_gemma_offset` — all 22 norm tensors compared
  element-by-element against the file, plus the hand-measured
  `layers.0.input_layernorm[0] == 1.7539`.
- `dflash_load_rejects_a_mismatched_codebook` — copies the shard, flips one
  byte of `fc.mcg`, and asserts the error names both the tensor and the
  codebook and stays a `DflashLoadError::Tensor`.

**Also needed and landed with it: a `silu` / `gated_silu` Metal kernel**
(`crates/kernels/src/elements.metal`, `elements.rs`). The DFlash drafter is
Gemma-family, so its MLP is SwiGLU (`hidden_act: "silu"` in `config.json`);
the engine only had `gated_gelu` for the MTP drafter. Using GeLU where SiLU
belongs is a *silent* quality error — nothing crashes — so it got its own
discriminator test: `gated_silu_is_not_gated_gelu` scans −8..8 in 0.25 steps
and asserts the two activations disagree by more than 0.1 somewhere, which is
50× the fp16 tolerance `assert_close` uses. Sabotage: rewiring `gated_silu`
to `gelu_tanh_value` fails **both** `element_gated_silu_matches_cpu`
(`gated silu[0] expected 0.15650958 actual 0.17114258`) and the
discriminator. Reverted, green.

### Commands

```bash
cargo test -p kaplumba-engine --release --test dflash_load
cargo test -p kaplumba-engine --release --test elements_metal
cargo check -p kaplumba-engine --release
```
