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

## 2026-10-11 — T3 (part 1): `dflash_block_attn` Metal kernel

`online_attention` **cannot** express the drafter's `full_attention` layer: its
liveness test is unconditional (`k_abs <= q_abs`, `attention.metal:836`), so a
bidirectional block is unreachable. z-lab's reference has that layer bidirectionally
(`dflash/model.py:367`, `_attention_mask` with `is_causal=False` and
`sliding_window=None` adds no branch), and the plan says implement z-lab. So a
new kernel: `dflash_block_attn` in `crates/kernels/src/attention.metal`.

Design (follows the repo's no-score-matrix rule, AGENTS.md §8): one
threadgroup per (block row, kv head), `q_heads/kv_heads` simdgroups, one per
query head in the group, 32 lanes each owning `head_dim/32` dimensions, online
softmax in fp32. Threads per threadgroup = `32 * q_heads/kv_heads` = 256 for the
real shapes (64/8). Capped at 32 simdgroups on the host side.

`crates/kernels/src/attention.rs` gained `DflashBlockRequest` (queries,
`ctx_keys`/`ctx_values`, `block_keys`/`block_values`, `key_positions` of length
`n_ctx + rows`, `rows`, `n_ctx`, `q_heads`, `kv_heads`, `head_dim`, `scale`,
`window`, `bidirectional`) and `AttentionKernels::dflash_block_attention`.

### Two real kernel bugs, both found by the tests rather than by reading

1. **`simd_shuffle_down` tree left only lane 0 holding the sum.** The original
   code did `dot += simd_shuffle_down(dot, offset)` for 16,8,4,2,1 and then a
   *no-op* `simd_broadcast(dot, 0) * 0.0f`. The online softmax merges `m`, `l`
   and `acc` **across lanes**, so a lane holding a partial sum corrupts the
   normaliser. Symptom: every output was exactly `0` (because `l` stayed 0 after
   the second bug). Fixed with `dot = simd_sum(dot)`, a butterfly that leaves
   all 32 lanes with the true total.
2. **`m_new = simd_max(m)` omitted the current key.** On the first key
   `m == -inf`, so `m_new == -inf` and the code hit an `if (m_new == -inf)
   continue` — every first key was dropped and the row normalised against
   nothing. Fixed to `m_new = max(m, score)`; since `score` is uniform across
   the simdgroup this is a scalar max, no reduction.

### Sabotage evidence (R8, `crates/engine/tests/dflash_block_attention.rs`)

| sabotage | failing test | message |
|---|---|---|
| `bidirectional` flag ignored (always causal) | `dflash_block_attention_matches_host_in_every_mask_mode` | `full_attention (bidirectional)[0] expected 0.21587108 actual 0.2244873` |
| `window` ignored (always open) | both the mask-mode test and `..._window_is_narrower_than_causal` | `window 1 barely differs from no window (0 of 16384 elements)` |

Both reverted, green.

### Agreement numbers (fp16, 128-dim dots)

```
full_attention (bidirectional): worst diff 0.000122   (vs scalar host reference)
sliding_attention window 2048:   worst diff 0.000122
sliding_attention window 4:      worst diff 0.000244
window 1 (own-position only):    worst diff 0.000000
dflash kernel vs online_attention (independent causal path): 0.000122
```

The cross-check against `online_attention` is the one worth keeping: two
independent implementations of the same causal-plus-window mask agreeing to
1.2e-4 is strong evidence the mask is right, and it is the only test that would
catch a bug shared between the kernel and the host reference.

## 2026-10-11 — T4: the golden harness, and the blocker that stops T5/T9

State: **T1/T2 committed (`5c8b9fe`); T3a committed (`05d1208`).** `DflashConfig`
+ `DflashModel::fuse_context / project_context_kv / forward_block / draft_logits`
and the three test files below are **written but NOT committed**.

### What T4 is now

`crates/engine/tests/dflash_golden.rs` — three bounded tests, each with its own
scalar f32 host reference dequantised from the same checkpoint:

| test | covers | result |
|---|---|---|
| `dflash_fuse_context_matches_host` | `fc` + mask/anchor fusion | **pass**, worst 0.006263 |
| `dflash_context_kv_matches_host` | `project_context_kv` (k/v proj + k_norm + RoPE) | **pass** |
| `dflash_forward_block_matches_host` | the 5-layer block | **FAIL — see blocker** |

`crates/engine/tests/dflash_nan_probe.rs` supersedes the proposed
`examples/dflash_taps.rs`: it prints per-stage `bad/max` (16 stages) and is how
the NaN below was localised.

### Two engine bugs found on the way (both fixed)

1. **Empty ctx K/V rejected.** `crates/kernels/src/attention.rs`:
   `dflash_block_attention` went through `make_u16_buffer`, which rejects empty
   inputs. With `n_ctx == 0` the context K/V are empty, so the block forwarded an
   `Err(InvalidInput)` where a self-attention-only block is perfectly legal. Fix:
   allocate a 2-element dummy buffer when the slice is empty. It is never
   dereferenced — the Metal loop only indexes `ctx_keys` when `k < n_ctx`, which
   is never true at `n_ctx == 0`. *(uncommitted)*
2. **Absolute tolerance too tight.** `v_ctx` reaches magnitude ~45, where fp16
   rounding is ~0.09. An absolute 0.05 threshold failed a correct implementation
   (`0.075` disagreement, 0.17% relative). Threshold is now relative to the
   reference's max.

### THE BLOCKER: the drafter overflows fp16, and exllamav3 runs it in fp32

With synthetic context, the **f32 host reference** — not the engine — produces:

```
v_block max  45.83      o_proj out max   2169.47
mlp activated max 5811.90   mlp_out max 152287.50
```

`152287` is past fp16's `65504`, so the engine's fp16 element path goes `inf` in
layer 0 and every later layer is `NaN`. `dflash_forward_block_matches_host`
therefore trips its own vacuity guard (`engine carries no signal (energy NaN)`).

**This is not an engine bug.** exllamav3 deliberately runs the drafter with fp32
activations — verified in `reference/exllamav3/exllamav3/architecture/dflash.py`:

- `:277` the attention block computes with `out_dtype = torch.float`
- `:308` `GatedMLP(..., interm_dtype=torch.half, out_dtype=torch.float)`
- `:316` only the *final* RMSNorm steps back down to `torch.half`
- `modules/arch_specific/dflash.py:26` `DFlashInputLayer(out_dtype = torch.float)`

So the residual stream is **fp32 throughout the drafter**, and that is a
requirement, not an optimisation. Consequence: T3/T5's fp16 block forward cannot
be made to pass by scaling or re-tuning; the element path needs an fp32 variant
for the drafter (norms, residuals, `silu`, softcap, and the gemm input/output the
EXL3 kernels consume). **This is the largest piece of remaining work in the spec
and it is not in `plan.md`.**

### The 9.58x gap (measured, unexplained — recorded, not resolved)

`crates/engine/tests/dflash_weight_scale.rs` dequantises `layers.0.self_attn.o_proj`
(8192→5376 in both checkpoints) on the host and compares RMS against the
known-good target:

```
target  (3 bpw, colmajor): rms 0.011386 = 1.031x unit variance  (1/sqrt(8192))
drafter (4 bpw, rowmajor): rms 0.109117 = 9.88x unit variance    (9.58x target)
```

The target landing at **1.031x unit variance** validates the rate/codebook/state
decode: a residual-preserving projection has RMS `1/sqrt(in_features)` by
construction, and a misread one does not. The drafter's ~10x is **not**
explained by 4 bpw vs 3 bpw. It is consistent with the overflow above (larger
weights → larger activations), so it is probably real — but "probably" is why
this is written down as an open question rather than a fact. First thing to check
if acceptance floors: whether the drafter's weights are genuinely 10x, or
whether 4 bpw 0.0.32 needs a different decode than the target's 3 bpw 0.0.37.

**Sabotage evidence (AGENTS.md §3.4), all three attempted:**

| sabotage | caught? | result |
|---|---|---|
| target rate `Integer(3)` → `Integer(4)` | yes, but coarsely | `TrellisLength` error, not the assertion — the width check fires first |
| `packed[0] ^= 0x1` (one corrupt trellis word) | **no** | test still green |
| global `0.01` scale on the decoded matrix | yes | `decodes to 0.01x unit variance; a correct read of a residual-preserving projection sits near 1.0` |

The middle row is the honest scope limit and it is now stated in the test's own
doc comment: **RMS is invariant under permutation**, so this test can never see a
row-major/column-major misread or a single bad tile. It is a magnitude check
only. The real guard on tile order is `repacked_golden`, not this.

### Checkpoint-format gotcha (cost an hour; do not re-derive)

The two checkpoints store the trellis with **different grid orders**, so a shape
cannot be used to derive in/out features:

```
drafter layers.0.self_attn.o_proj.trellis  [512, 336, 64]   tiles_k, tiles_n (stock EXL3)
target  ...layers.0.self_attn.o_proj.trellis [336, 512, 48] tiles_n, tiles_k (repacker, colmajor)
```

Both have `suh [8192]` / `svh [5376]`, so **use the sign-vector lengths** for
in/out — they are unambiguous. The target's on-disk tiles are already transposed;
the tile-grid transpose is its own inverse, so applying `reorder_tensor_to_colmajor`
to what is on disk converts it back to the row-major order `reconstruct_matrix`
reads.

### Exact next commands

```bash
# The three golden tests, with per-stage diagnostics on failure
cargo test -p kaplumba-engine --release --test dflash_golden -- --nocapture
# Stage-by-stage NaN/max probe (no model load, no GPU)
cargo test -p kaplumba-engine --release --test dflash_nan_probe -- --nocapture
# Host-only weight-scale check (no GPU, no gpuq)
cargo test -p kaplumba-engine --release --test dflash_weight_scale -- --nocapture
```

**Order for the next session:** (1) T12's M=16 `gemm_chunk` bench — it is
unblocked, host-measurable, and it decides whether T6's shape survives;
(2) the fp32 element path above, without which T4's third test cannot pass;
(3) commit, then T6.

## 2026-10-11 — T14 landed: the fp32 path, and T4 is green (4/4)

**The blocker above is resolved, and it cost three small kernels rather than the
element-path rewrite the first reading suggested.** The trick is in
`linear.rs::gemm_chunk_buffer`: the M=16 prefill GEMM is not one kernel, it is
three -- `input_rows` (f16 transform) -> the tiled GEMV into an **f32 `raw`
accumulator** -> `linear_finalize_rows` (f32 -> f16, applying `transform_vector`
and the output signs). The f32 result already exists in a buffer; only the last
pass narrowed it. So the f32 GEMM output is a *new finalize kernel*, not a new
GEMM.

### What was added

| file | addition | why |
|---|---|---|
| `crates/kernels/src/linear.metal` | `linear_finalize_rows_f32` | the f32 twin of `linear_finalize_rows`; identical apart from the output dtype |
| `crates/kernels/src/linear.rs` | `finalize_rows_f32_pipeline`, `gemm_chunk_buffer_f32`, `read_f32_buffer` | the wrapper; same three dispatches as `gemm_chunk_buffer` |
| `crates/kernels/src/elements.metal` | `rms_norm_weighted_f32_in_tg`, `residual_add_f32` | read the f32 stream; the norm still writes f16 |
| `crates/kernels/src/elements.rs` | `rms_norm_f32_in`, `residual_add_f32`, `make_f32_buffer`, `read_f32_buffer` | wrappers |
| `crates/engine/src/dflash.rs` | `gemm_f32`, `residual_f32`, `rms_norm_f32_in`, `trace_stage_f32`; `forward_block`'s `hidden_state` is now `Vec<f32>` | the drafter's residual stream never narrows |

**Only `o_proj` and `down_proj` take the f32 gemm.** exllamav3 gives the attention
block and the GatedMLP `out_dtype = torch.float` but leaves their inputs, and the
gated activation, at half (`architecture/dflash.py:277,308`). Mirroring that split
exactly means q/k/v/gate/up keep the fast f16 path. The single dtype boundary is
the norm: **f32 in, f16 out** -- which is also what exllamav3 does (its norm
writes half, and only the *final* norm steps the stream back down).

`trace_stage_f32` exists because tracing the f32 stream through the f16 helper
would narrow it to `inf` precisely for the stages under investigation.

### Result

```
dflash_fuse_context_matches_host ................ ok   worst 0.006263
dflash_context_kv_matches_host .................. ok
dflash_forward_block_matches_host ............... ok   worst 0.068110  (ref max 77.2833 = 8.8e-4 rel)
dflash_exl3_decode_matches_host_on_every_linear . ok   all 36 linears, worst 0.003663
test result: ok. 4 passed; 0 failed
```

`forward_block`: engine max **77.2500** vs reference max **77.2833** — the engine
now carries signal where it previously produced `NaN`. No regression:
`dflash_load` 3/3, `dflash_block_attention` 3/3, `dflash_weight_scale` 1/1,
`dflash_nan_probe` 1/1, `kaplumba-kernels` 33+5+4+4+2+1 green.

### Sabotage evidence (AGENTS.md §3.4)

`residual_add_f32` rewritten to narrow its result (`float(half(left + right))`) —
i.e. undo the fix without touching anything else:

```
forward_block worst diff 0.000000 at 0 of 86016; engine max 0.0000 ... (energy NaN)
thread 'dflash_forward_block_matches_host' panicked at dflash_golden.rs:700:5:
the comparison is vacuous: the engine carries no signal (energy NaN)
test result: FAILED
```

That is the *same* failure the fp32 path was introduced to fix, reproduced from
the opposite direction, so the test can fail and the f32 add is load-bearing.
Reverted; 4/4 green again.

### Still open from this session

- `crates/engine/tests/dflash_golden.rs` had **half-applied edits** from the
  previous session (`let (in, out) = ...`, a stray `(line.0, line.1)`, an
  unfixed `mut harness`). The first made the whole test *file* fail to compile,
  which reads as "no tests ran" rather than "a test failed". Fixed. If a future
  session sees a test file that will not build, check for that pattern first.
- The 9.58x weight-RMS gap (below) is still unexplained and is not on the path to
  T4 passing. Keep it in mind if acceptance floors.
- `KAPLUMBA_DFLASH_CAUSAL_BLOCK` is still unwired (mask selection reads
  `layer_is_sliding` only).

## 2026-10-11 — T12: the M=16 GEMM is the binding constraint, not the drafter

`crates/engine/tests/dflash_gemm_m16.rs`, all 36 drafter linears at M=16, median
of 9 with 3 warmups. Full table is printed by the test; the shape of it:

```
fc                          32256x5376    86.78 MB    8.722 ms    9.9 GB/s
layers.N.self_attn.q_proj    5376x8192    22.05 MB    ~2.50 ms    8.8 GB/s
layers.N.self_attn.k_proj    5376x1024     2.77 MB    ~0.69 ms    4.1 GB/s
layers.N.mlp.gate_proj       5376x10752   28.93 MB    ~3.08 ms    9.4 GB/s
layers.N.mlp.down_proj      10752x5376   28.93 MB    ~3.13 ms    9.0 GB/s
one block, 35 per-layer projections: 78.89 ms      slowest single: fc, 8.722 ms
```

**9.9 GB/s on a box with 273 GB/s and a measured ~51 GB/s on the M=1 decode
path.** The M=16 tiled GEMM is therefore *instruction-bound*, not bandwidth-bound,
and it is ~25x off streaming peak. That is a much larger finding than the one T12
was commissioned to make, and it changes what is worth doing next.

### The arithmetic this pins down

| stage | cost | source |
|---|---|---|
| 35 per-layer projections | 78.89 ms | measured |
| `fc`, once per block | 8.72 ms | measured |
| the output head | **~105 ms** | extrapolated: the drafter has no head, so the round uses the target's `lm_head` (5376x262144 at 6 bpw ~= 1.05 GB) at the measured 9.9 GB/s |
| target verify, M=16 | **~1.55 s** | extrapolated: one M=16 pass over the target's 15.5 GB at 9.9 GB/s |

So one round costs ~193 ms of drafting **plus ~1.55 s of verify** and can yield at
most 15 tokens: **a ceiling of ~8.6 tok/s at 100% acceptance and zero rejection.**
The current non-speculative decode is 3.3-6.4 tok/s. DFlash as specified is
therefore worth perhaps 1.5-2.5x, not the 10x the paper's shape suggests.

**A 40 tok/s target is not reachable by work on the drafter, the acceptance
rate, or the round loop.** It requires the M=16 GEMM to reach a normal fraction
of bandwidth first. The head at ~105 ms and the verify at ~1.55 s are both the
same bottleneck: `gemv_tiled` at M=16.

### Caveats on the extrapolations

- The head and verify rows are **extrapolated from the measured per-byte rate**,
  not measured. Measuring them needs the 13 GB target loaded (`scripts/gpuq`), and
  the head row in particular assumes the target's 6 bpw `lm_head` behaves like the
  drafter's 4 bpw linears, which the per-shape table suggests but does not prove.
  **Both should be measured before any decision is made on them.**
- The 51 GB/s figure for the M=1 path is taken from the decode rate already in
  the repo (15.5 GB at ~3.3 tok/s), not re-measured here.
- `k_proj`/`v_proj` are the efficient outliers at 4.1 GB/s because 2.77 MB per
  dispatch is too little work to amortise the dispatch; they would want fusing,
  not a new kernel.

### What is NOT the problem

The drafter's own forward is fine: T4 agrees with the f32 reference to 8.8e-4 and
T6's cache invariants hold. Optimising the drafter further buys ~1.4x of the
mispredicted 10x.

## 2026-10-11 — T6 landed: the drafter context cache

`DflashKvCache` + `DflashModel::append_context` / `rewind_context` / `draft_block`.
Three tests in `crates/engine/tests/dflash_cache.rs`, 3/3 green.

### The design change the plan did not predict

`plan.md` assumed one `n_ctx` for all five layers. **That cannot work with the
sliding window**: layers 0-3 cap at `sliding_window` (2048) while the last layer
attends the whole context, so past 2048 rows the layers genuinely hold different
row counts and start at different absolute positions. `BlockContext` now carries
`ctx_first: &[usize]` and derives each layer's count from its own buffer; the mask
is on position *distance*, so a ring that renumbered its surviving rows from zero
would silently attend the wrong keys.

Cache storage is `keys: Vec<Vec<u16>>` / `values: Vec<Vec<u16>>` — one flat vector
per layer, which is *exactly* `BlockContext`'s shape, so a block forward borrows it
instead of copying 5x2x[rows x kv_width] per round.

`push` is layer-local and `commit(rows)` advances the global context length once
per chunk; that is what keeps `first_position(layer) = total - held(layer)` correct
for both a ring that has wrapped and one that has not.

### Deviations from `plan.md`

- `draft_block` takes the target's head (`&QuantizedLinear`) as an argument and
  returns a `DflashDraft { tokens, block }`. The plan's signature had no head,
  but the drafter has none of its own (`tie_word_embeddings`) and the round needs
  the target's, so it cannot be implicit.
- `rewind_context` returns `Result` rather than `()`: rewinding further than the
  context is a caller bug and should not be silent.

### Sabotage evidence (AGENTS.md §3.4)

| sabotage | failing test | message |
|---|---|---|
| ring eviction skipped (`if false && held > capacity`) | `ring_keeps_absolute_positions` | `assertion failed: layer 0 should hold exactly the window` |
| RoPE offset uses `position` instead of `position - rows` | `is_chunk_invariant` **and** `rewind_equals_never_having_appended` | `assertion failed: layer 0 keys differ between a whole and a chunked append` / `layer 0 keys differ after rewind` |

Both reverted, 3/3 green. Note the second sabotage attributes to two tests, which
is expected: the same offset error corrupts any two-schedule comparison, and it is
why chunk invariance and rewind equivalence are not redundant tests.

## 2026-10-11 — CORRECTION: the T12 extrapolation above is WRONG for the target

I extrapolated the target's M=16 verify cost from the *drafter's* measured 9.9 GB/s.
That was a mistake — the two are not comparable — and it has now been measured
directly. **The table in §"T12" under "the arithmetic this pins down" must not be
used.** Corrected numbers below.

### What was measured (this branch, `Gxma-repacked`, 3 bpw)

```text
prompt 28 tok, 1 new:   ttft 546.7 ms   prefill 51.26 tok/s   (28.4 GB/s over ~15.5 GB)
prompt 62 tok, 24 new:  ttft 1351.9 ms  prefill 45.87 tok/s   decode 15.39 tok/s
```

Two things fall out, and both matter more than the T12 result did:

1. **The target prefills at 28-46 GB/s, not 9.9.** The drafter's 9.9 GB/s figure
   is specific to a 4 bpw **row-major** (`perm_mode = 0`) trellis. The target is
   3 bpw **colmajor**, and the repo's colmajor path is the faster one
   (`spec/decode-19tok-repacked-250gb`). So an M=16 verify over the target is
   ~550 ms, not ~1.55 s, and the head is ~37 ms, not ~105 ms.
2. **The baseline decode on this branch is 15.4 tok/s, not the 3.7-6.4 tok/s that
   `AGENTS.md` §10.2 #5 records.** That figure is stale for `stepwise-clean-3724c01`,
   which carries the colmajor/19-tok work. This changes the value proposition
   completely: DFlash is being asked to beat 15.4 tok/s, not 3.3.

### Corrected round arithmetic

| stage | corrected cost | basis |
|---|---|---|
| 35 per-layer projections | 78.89 ms | measured (T12) |
| `fc` | 8.72 ms | measured (T12) |
| head, ~15 rows | ~37 ms | 1.05 GB at the target's measured ~28 GB/s |
| verify, M=16 | ~550 ms | 15.5 GB at ~28 GB/s |

Per block: ~125 ms drafting + ~550 ms verify = **~675 ms for at most 15 tokens =
~22 tok/s at 100% acceptance.** Against a 15.4 tok/s baseline that is ~1.4x.

**40 tok/s** needs 15 tokens per 375 ms, i.e. verify + draft <= 375 ms, i.e. the
M=16 verify at >= ~62 GB/s. Whether that is reachable is exactly the open question
(see below). It is *not* reachable by drafter-side work: the drafter is already
only 19% of the block.

### Hard caveats on these numbers

- `sysctl vm.swapusage` after the second run: **5994.75 M swap in use**, and the
  engine printed `sys_free_pct_min=15`, `sys_wired_peak_gb=20.36`. The box was
  under real memory pressure. Per `AGENTS.md` §10 these numbers should be treated
  as indicative only and re-measured on a quiet box before any decision.
- The 62-token run needed `KAPLUMBA_FORCE_LOAD=1`; admission control refused it
  (demand 16.28 GiB vs 15.95 GiB available). A forced load is a benchmark-validity
  risk, which is the second reason to re-measure.
- The M=16 verify cost is still **inferred**, now from a prefill-rate measurement
  rather than a drafter number. It has not been measured at M=16 specifically.
- I did not determine **why** decode is 15.4 tok/s when `AGENTS.md` says 3.7-6.4.
  If the engine is already running a speculative (MTP) path, then DFlash's
  marginal value is much smaller again, and the 15.4 tok/s is not the baseline
  DFlash has to beat. **This should be established before implementing T7-T9.**

### T12's real conclusion

The drafter is not the bottleneck; the M=16 GEMM is, and the target's own M=16
path is the number that decides whether any of this pays. T12's 9.9 GB/s remains a
valid measurement **of the drafter's row-major 4 bpw path only** and should not be
generalised.

## 2026-10-11 — RESEARCH: why decode is 15.4 tok/s, and the exact T7 edit sites

### It is NOT speculative decoding — verified

`grep -rn "speculative|mtp|Mtp|draft_model|drafter"` across the run path returns
exactly one real hit: `generate.rs:1068`, whose comment says "the GPU never
executes this speculative step". That is the **pipelined encoder commit** (encode
the next step, commit only once the token is known not to be EOS) — not
speculative decoding. There is no MTP head and no draft model in `Model::load` or
the CLI. The 15.4 tok/s is the genuine non-speculative decode rate.

### Why it is that fast: decode is at the bandwidth ceiling

`Gxma-repacked` is **15 G on disk** (`du -sh`). One decode step reads all of it:

```text
15.5 GB / 64.9 ms = 239 GB/s    (vs 273 GB/s peak, and the ~227-238 GB/s ceiling
                                 already recorded in spec/decode-19tok-repacked-250gb)
```

64.9 ms/token is exactly 15.4 tok/s. **Decode is bandwidth-saturated at ~88% of
peak — there is essentially nothing left to win there**, which is precisely the
regime where speculative decoding is supposed to pay.

### Why prefill looked slow: it is compute-bound, and it amortises

Measured, three prompt lengths, `--max-new-tokens 1`:

| prompt tokens | ttft ms | prefill tok/s | marginal ms/token |
|---|---|---|---|
| 28 | 546.7 | 51.26 | — |
| 62 | 1351.9 | 45.87 | 23.7 (28->62) |
| 445 | 5189.2 | 85.77 | 10.0 (62->445) |

The marginal cost **falls** with batch size, so prefill amortises the weight read
across the batch (and is then compute-bound), while decode at M=1 cannot amortise
anything. This fixes the mistake in §"CORRECTION": a verify is **one pass**, not
per-token. Extrapolating the measured M=28 pass (546 ms total) to M=16 gives
**~300-550 ms**, not 1.55 s. So a round is ~125 ms drafting + ~300-550 ms verify
for up to 15 tokens:

- at 550 ms: ~22 tok/s (1.4x over the 15.4 baseline)
- at 300 ms: ~35 tok/s (2.3x)

**The M=16 verify cost is the single number that decides whether this spec pays,
and it is still not directly measured.** Measuring it is cheap: prefill exactly 16
tokens and read `ttft_ms`. That should be the first action of the next session,
before any more implementation.

### T7 edit sites (researched, not yet implemented)

Everything needed is known; I ran out of budget before writing it and chose not to
leave unverified edits in a 5800-line hot-path file.

| what | where |
|---|---|
| `ForwardScratch` struct | `model.rs:3163` |
| `scratch: ForwardScratch` field | `model.rs:3419` |
| Model construction literal (`scratch,`) | `model.rs:3591` |
| layer loop | `model.rs:4807` — `for index in 0..layers` |
| ping-pong select | `model.rs:4808` — `let (input, output) = if index % 2 == 0 { (hidden_a, hidden_b) } else { (hidden_b, hidden_a) }` |
| layer call, after which `output` holds layer `index`'s post-layer state | `model.rs:4836` (`this[0].forward_buffer(...)`) |
| GPU->GPU copy primitive | `ElementKernels::encode_copy(encoder, input, output, count)` — `elements.rs:1776` |
| read-back precedent | `hidden_states_all` — `model.rs:5474` |
| `ensure_scratch` | `model.rs:5165` |

Notes that cost real time to establish and should not be re-derived:

- **Tap layers `[1, 12, 23, 35, 46, 57]` have mixed parity**, so the blit must
  pick its source per layer: odd -> `hidden_a`, even -> `hidden_b`. A hardcoded
  `hidden_a` happens to be right for four of the six.
- The blit **must be inside the layer loop**. `hidden_states_all` captures only the
  *final* layer's pre-final-norm state after the whole chunk, so intermediate
  taps cannot be recovered post-hoc. This is the one part of T7 that has no
  existing precedent in the tree.
- Add the tap buffers as an **opt-in struct**, not an unconditional
  `ensure_scratch` allocation: 6 x 1024 x 5376 x 2 = 66 MB, and the default path
  must not pay it (AGENTS.md §8; the footprint budget is 12.96 GiB measured).
- `ensure_scratch` resolves its device handle inside its own body (via the
  `u16_elements`/`f32_elements` helpers). **I did not confirm that idiom** — read
  `model.rs:5165-5300` first; it is the only unknown left in T7.

## 2026-10-11 — MEASURED: the M=16 verify floor is ~545 ms, so 40 tok/s is out of reach

The one measurement everything hinged on. Prefill short prompts and read `ttft_ms`:

| prompt tokens | ttft ms |
|---|---|
| 25 | **543.9** |
| 28 | **548.1** |
| 62 | 1351.9 |
| 445 | 5189.2 |

**25 and 28 tokens cost the same (543.9 vs 548.1 ms).** So `ttft` for a small prompt
is a *fixed floor of ~545 ms*, not a per-token cost — the batch is padded or the
chunk is padded, and an M=16 verify sits inside that floor. This is the number
that supersedes the "~300-550 ms" estimate above; it is the pessimistic end.

### Final round arithmetic

| stage | cost | basis |
|---|---|---|
| 35 per-layer projections | 78.89 ms | measured (T12) |
| `fc` | 8.72 ms | measured (T12) |
| head, ~15 rows | ~37 ms | 1.05 GB at a prefill-like rate |
| verify, M=16 | **~545 ms** | **measured floor** |

Round = ~125 ms drafting + ~545 ms verify = **~670 ms for at most 15 tokens =
~22 tok/s at 100% acceptance.** Against the measured 15.4 tok/s baseline that is
**1.4x**.

**40 tok/s needs 15 tokens per 375 ms, i.e. a verify at <= ~250 ms.** The measured
floor is 545 ms, so 40 tok/s requires the M=16 prefill path to get ~2.2x faster
than it currently is. It is not reachable by anything in this spec: the drafter is
19% of a round and the verify floor alone is 15% more than the entire 375 ms
budget. This is now a GEMM/prefill-efficiency question, not a DFlash question.

### Caveats

- `sysctl vm.swapusage`: 5275 MB in use, `sys_free_pct_min=18`. Still memory
  pressure. But the two short-prompt runs agree with each other to 4 ms, so the
  ~545 ms floor is robust to it in a way a single reading would not be.
- All runs used `KAPLUMBA_FORCE_LOAD=1`.
- The 62- and 445-token rows show cost *rising* with length past the floor, so
  545 ms is a floor for M<=~28 and not a general prefill rate.

### Bottom line for the spec

DFlash as specified is worth ~1.4x (22 vs 15.4 tok/s), not the 10x its shape
implies, and the ceiling is set by the target's M=16 prefill path. T7-T11 are
still the only way to *confirm* the 1.4x, but they cannot exceed it. If 40 tok/s
is the goal, the work is a faster M=16 tiled GEMM / prefill, and this spec should
be paused rather than extended.

## 2026-10-11 — HARD RESULT: 25 tok/s is impossible without speculation, and needs 100% acceptance with it

### The bandwidth wall (measured weights, measured peak)

```
weights on disk:  8.498 + 7.529 GB = 16.03 GB   (the two safetensors shards)
box peak:         273 GB/s
```

Every decode step must read all 16.03 GB of weights, so:

```text
theoretical max decode at 100% of peak = 16.03e9 / 273e9 = 58.7 ms/token = 17.0 tok/s
measured decode                       = 15.4 tok/s  = 87% of that ceiling
25 tok/s needs 401 GB/s  -> more than the hardware can do
40 tok/s needs 641 GB/s  -> likewise
```

**25 tok/s is physically unreachable by single-token decode** — not "hard", not
"needs a better kernel": it would require 1.47x the memory bandwidth this machine
has, and the existing path is already at 87% of peak. The same is true of the
earlier 40 tok/s request.

So 25 tok/s **requires** emitting more than one token per weight read. That is
what speculative decoding is for, and it is the only route.

### But DFlash at block_size=16 cannot reach it either

A round is `draft + verify` and yields at most 15 tokens (anchor dropped):

```text
draft                      ~125 ms   (measured: 78.89 projections + 8.72 fc + ~37 head)
verify, M=16               ~545 ms   (measured floor)
round                      ~670 ms
tokens per second at 100% acceptance = 15 / 0.670 = 22.4 tok/s
```

**25 tok/s needs a round of <= 600 ms** (15 / 0.600 = 25.0). With the verify floor
at 545 ms that leaves **<= 55 ms for drafting, against the measured 125 ms**, i.e.
drafting must drop by **2.3x**, and acceptance must be **100%**.

Both conditions are required and neither is sufficient alone. In practice
acceptance will not be 100%, so:

| acceptance | round needed for 25 tok/s | implication |
|---|---|---|
| 100% (15 tok) | 600 ms | drafting must be <= 55 ms |
| 90% (13.5 tok avg) | 540 ms | drafting must be <= 0 (already impossible) |

At 90% acceptance, 25 tok/s is out of reach **regardless of drafting cost**,
because the verify floor alone exceeds the budget.

### What would actually get there

1. **Raise `block_size`.** This is the single highest-leverage change and it was
   not in the spec. At M=32 the verify is a ~700-900 ms pass (bounded by the
   measured 62-token pass at 1352 ms) for up to 31 tokens: `31 / (0.8 + 0.25) =
   29.5 tok/s` at 100% acceptance, and still ~26 tok/s at 90%. That clears the
   target with margin, from the *same drafter checkpoint semantics* the paper's
   §5.5.4 says generalises down from b16. This is worth measuring before anything
   else.
2. **Cut the drafting cost.** The drafter's M=16 path runs at 9.9 GB/s where the
   target's colmajor path reaches 28-46 GB/s — ~3-4x of real headroom in the same
   kernel family (T12). This converts the 78.89 ms of projections into ~20-25 ms.
3. **Accept that 25 tok/s is a speculation-only number.** Any plan that reaches it
   by tuning the decode path is chasing 401 GB/s on a 273 GB/s machine.

### Status of the spec against this target

T7-T11 are still unimplemented, so no end-to-end DFlash tok/s exists. Even
completed, at block_size=16 they top out at ~22 tok/s. **The measured evidence says
the spec should be re-scoped around `block_size` before T7-T11 are built**, because
block_size decides whether 25 tok/s is reachable at all and it is a config/verify-shape
question rather than a drafter question.

## 2026-10-11 — BREAKTHROUGH: the verify is PADDED TO A POWER OF TWO, so block_size=32 doubles throughput for ~free

### The measurement

Prefill cost vs prompt length (all `--max-new-tokens 1`, `Gxma-repacked`):

| prompt tokens | ttft ms |
|---|---|
| 25 | 543.9 |
| 28 | 548.1 |
| **30** | **595.2** |
| — boundary — | |
| **36** | **1325.4** |
| 37 | 1323.8 |
| 39 | 1328.0 |
| 43 | 1329.9 |
| 62 | 1351.9 |
| 445 | 5189.2 |

**This is a step function, and the step is exactly at 32.** 25-30 tokens cost
~545-595 ms; 36-62 tokens cost ~1324-1352 ms. The engine pads the prefill batch to
a power of two, so **anything up to 32 rows costs the same as 16.**

### Why that decides the target

An M=16 verify (block_size=16 → 15 draft tokens) and an M=32 verify (→ **31** draft
tokens) sit in the **same 545-600 ms bucket**. Doubling the block roughly doubles
the tokens per round for ~50 ms of extra verify.

Derived from the measured M=16 numbers (78.89 ms projections = 4.93 ms/row,
~2.4 ms/row head), with verify from the table above:

| block_size | verify ms | draft ms | round ms | 100% acc | 90% acc | 80% acc |
|---|---|---|---|---|---|---|
| 16 | 545 | 126 | 671 | 22.4 | 20.1 | 17.9 |
| **32** | **~595** | **~242** | **~837** | **~37** | **~33** | **~29** |
| 64 | 1325 | 478 | 1803 | 34.9 | 31.5 | 28.0 |

**block_size=32 clears 25 tok/s even at 80% acceptance (~29 tok/s).**
**block_size=16 cannot reach 25 tok/s even at 100% acceptance (22.4).**

So the answer to "get it to 25 tok/s" is not a kernel optimisation and not the
round loop: it is **block_size**. At 16 the target is arithmetically impossible; at
32 it is reached with ~20% of headroom to spare. 64 is worse than 32 because it
pays for the next padded bucket.

### The risk, and the mitigation

The checkpoint was trained at b16, and the paper's §5.5.4 only claims a b16-trained
drafter **generalises down** to b8. Interpreting its geometry at b32 is not
supported and may cost acceptance — which is why the 80% column matters, and 29
tok/s there is still above target.

A safer route to the same 31 tokens that stays on the trained b16 geometry: **run
the drafter block twice per round** (15 drafts, then re-anchor on the 15th and draft
15 more), concatenate both blocks' rows into one M=32 verify, and feed block 1's
own K/V into the cache for block 2. `DflashBlock` already hands back
`block_keys`/`block_values` for exactly this, so the plumbing exists; what does not
is a caller that uses them.

### Consequence for the spec

T7-T11 should be re-scoped to **block_size=32** before being built. Building them
at 16 would produce the best possible implementation of a configuration that
cannot hit the target (22 tok/s ceiling), which is why this measurement had to come
before the pipeline rather than after it.

Caveats: the M=32 verify figure is read from the 30-token row (595.2 ms) and the
drafter's M=32 cost is scaled linearly from its measured M=16 cost, not measured.
Both should be confirmed with the drafter benchmark extended to M=32 before the
round is built.

## 2026-10-11 — T7 landed: target tap export

`TextModel::enable_taps(&[usize])` / `take_taps(tokens) -> Vec<Vec<u16>>` /
`tap_layers()`, plus `TapScratch`. The capture is a GPU->GPU `encode_copy`
(`elements.rs:1776`) inside `forward_chunk`'s layer loop, after
`forward_buffer` returns and before the profiling block.

**The one non-obvious thing: the blit must select its ping-pong source per
layer.** Layer `k` writes `hidden_b` when `k` is even and `hidden_a` when `k` is
odd (`model.rs:4808`), so the offset written for layer `index` is `output`. The
DFlash taps are `[1, 12, 23, 35, 46, 57]` — four odd, two even — so a hardcoded
`hidden_a` is right for four of the six and silently wrong for the other two.

It also **cannot** be done after the loop: `hidden_states_all` reads the *final*
layer's state once a chunk completes, but intermediate layers are overwritten by
the ping-pong before then. That is why this is in-loop and why T7 had no existing
precedent to copy.

`TapScratch` is deliberately **not** a field of `ForwardScratch`: six taps is
6 x 1024 x 5376 x 2 = 66 MB, and the default path must not carry it (AGENTS.md §8,
and the 12.96 GiB footprint budget is measured). `enable_taps(&[])` releases it.

### Test and sabotage evidence (AGENTS.md §3.4)

`crates/engine/tests/dflash_taps.rs`, `#[ignore]`d (needs `KAPLUMBA_TEST_MODEL`),
one test with four assertions:

| property | passing | sabotaged (source hardcoded to `hidden_a`) |
|---|---|---|
| shape + signal | 2 buffers, `rows * hidden` each, non-zero | — |
| forward unperturbed | logits bitwise identical with taps on/off | **still passed** |
| parity | tap[1] vs tap[2] differ on **75180/75264** | differ on **0/75264** -> FAILED |
| request order | `[2,1]` returns the layers swapped | — |

Note the second row: the perturbation check **still passed under the sabotage**,
which is the point of having the parity assertion. Reading the wrong buffer does
not corrupt the forward; it silently mislabels the data, and only a check on the
*tap contents* can see it. Run command:

```bash
KAPLUMBA_TEST_MODEL=~/.kaplumba/models/Gxma-repacked \
  cargo test -p kaplumba-engine --release --test dflash_taps -- --ignored --nocapture
```

## 2026-10-11 — T8 landed: chunked prefill streaming, plus a real T7 bug it caught

`crates/engine/tests/dflash_streaming.rs` (ignored; needs the 16 GB target *and*
the drafter). It feeds the same 32 tokens two ways — one 32-row chunk, and two
16-row chunks streamed through `append_context` — with **real target taps** from
T7, and requires the drafter's cache to be byte-identical.

Result: **0 of 32768 elements differ on all 5 layers.** R6 holds end to end.

This is strictly stronger than `dflash_cache`'s synthetic-tap invariance test: it
covers tap capture, `fc`/`hidden_norm` fusion, `k_proj`/`v_proj` + `k_norm` +
RoPE, and the cache, across a real chunking.

### It caught a genuine out-of-bounds bug I had just introduced in T7

`enable_taps` sized the tap buffers from `self.scratch.max_tokens` **at the time
it was called** — which is before the first forward, when the scratch is still at
its initial size. The first forward calls `ensure_scratch(tokens)`, which *grows*
the scratch. The tap buffers did not grow with it, so the in-loop blit was writing
`tokens * hidden` elements into a buffer sized for one row.

Symptom: `take taps: InvalidConfig("tap buffer is smaller than the requested row
count")` — the read-back failing was the *lucky* part; had `take_taps` not
re-checked the length, this would have been a silent device out-of-bounds write.

Fixed with `rebuild_tap_buffers`, called from `ensure_scratch` after the scratch is
rebuilt, which re-allocates when the row count changed and is a no-op otherwise.
`dflash_taps` passed *before* this fix only because its 14-token prompt happened to
fit the scratch's initial size — a reminder that a test with a too-small input can
pass while the code is wrong.

### Sabotage evidence (AGENTS.md §3.4)

| sabotage | result |
|---|---|
| RoPE base from the chunk offset (`start = 0`) instead of `position - rows` | **~13130/32768 keys differ, 0 values**, on every layer -> FAILED |

That split is the correct signature and worth keeping as a diagnostic: R9 rotates
q and k and **never** v, so a position/RoPE error must move keys and leave values
untouched. A failure that moved both would mean a different bug (e.g. the fusion or
the projection), not a position error.
