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

## 2026-10-11 — T9 blocked on a real alignment subtlety (found, not yet solved)

I wrote `draft_and_verify`, then removed it before committing. It compiled, but it
is **wrong**, and this is the finding that matters.

### The bug

The block is `[anchor, mask x 15]` at positions `position .. position + block - 1`,
where `anchor` is the last already-generated token. Verifying row `r` requires the
target's logits *at that row*. The obvious implementation feeds
`[last_token] ++ drafts` to the target at `position .. position + block - 1` — which
is what I wrote.

**That re-feeds the last token.** After any prefill the target's cache already
holds the row for `last_token`; feeding it again at the same position appends a
second, identical K/V row there. With a causal mask the block's rows then attend
to that key twice, which does not cancel — it changes the softmax and therefore
the logits. So the "target" being verified against would be a subtly different
model, and acceptance would be measured against corrupted logits.

### The correct shape

Do not re-feed the anchor. Carry the previous step's logits instead:

```rust
pub fn draft_and_verify(
    &self, target, cache,
    last_token: u32,
    position: usize,             // position of the FIRST token to predict
    previous_logits: &[u16],     // target logits at row position - 1, vocab-wide
) -> Result<DflashRound, DflashError>
```

- Feed only `draft.tokens[0 .. n-1]` at `position .. position + n - 2`.
- Let `L[k]` be the target logits row that predicts `draft.tokens[k]`: `L[0]` is
  `previous_logits`, and `L[k]` for `k >= 1` is the row produced by feeding
  `draft.tokens[k-1]`.
- First `k` with `argmax(L[k]) != draft.tokens[k]` decides: accept
  `draft.tokens[..k]`, emit `argmax(L[k])` as `next`.
- The cache advanced by `n - 1` rows but only `k + 1` are correct, so retract
  `(n - 1) - (k + 1)`.

The consequence for the caller is that the round must **own the logits** from the
previous forward — the prefill's returned last-row logits, then each round's
`L[k]`. `forward_buffer` already returns exactly that for a single row, so the
plumbing exists; it just is not free, and the round's signature has to carry it.

### Why this is worth writing down rather than coding through

Every wrong variant here still *runs* and still produces a number. Re-feeding the
anchor inflates or deflates acceptance silently; verifying `draft.tokens[1..]`
against `L[0..]` (off by one) accepts tokens the target would not have emitted,
which breaks R6's "no quality loss" without failing anything obvious. A T10
equivalence test catches those — but only if the test compares full greedy
sequences, which is why T10 has to be written before T9 is believed.

### What did land

- `TextModel::head()` and `TextModel::raw_embedding(&[u32])` — the latter is
  `embedding_scaled_view` with `scale = 1.0`, so it reuses the proven kernel path
  rather than a host mirror (the host mirror is what once made `mtp_acceptance`
  read `-inf`/`NaN` and report 0 acceptance). This is the R7/Q1 flag: Q1 flips
  raw -> scaled by changing that one argument.
- `DflashRound { accepted, next, drafted }` with `emitted()`, the shape the round
  will return.

---

## Decode baseline measured, 2026-10-11 (3-bit target, no speculation)

Command (3 runs, 200 tokens, through gpuq):

    M=~/.kaplumba/models/Gxma/Gxma
    for i in 1 2 3; do
      ~/.../gpuq run -- ./target/release/kaplumba --model "$M" \
        --prompt "Explain the theory of relativity." --max-new-tokens 200 \
        --temperature 0 --metrics
    done

Result — decode tok/s: **15.283, 15.403, 15.265** (median 15.283, spread
0.138 = 0.9%). ttft_ms: 1388.6, 1342.6, 1337.7.

This is the first tight decode figure in this repo. AGENTS.md §10.2 #5 records
3.7-6.4 tok/s from the *same* binary minutes apart; this session's runs are
within 0.9% of each other across three runs. Treat 15.3 as the baseline, not the
older numbers — those were measured under different conditions.

### What 30 tok/s requires (arithmetic, not opinion)

    weights            = 16.028 GB (8.499 + 7.529)
    15.283 tok/s       = 65.4 ms/token = 245 GB/s = **90% of the 273 GB/s peak**
    non-spec ceiling   = 273 / 16.028 = **17.03 tok/s**
    30 t/s needs       = 481 GB/s = **1.76x the machine's bandwidth**

**30 t/s is impossible on any single-token decode path.** The existing decode is
already at 90% of the hardware ceiling; there are ~1.7 tok/s of headroom, not
13. This closes the question the same way 25 did, with a measured baseline
rather than an assumed one.

### Therefore 30 t/s is reachable only through speculation, and only at b32

              block=16    block=32
    acc 100%    22.4        37.0
    acc  90%    20.1        33.3
    acc  81%    18.1        30.0   <- 30 t/s threshold
    acc  80%    17.9        29.6

- **block_size=16 cannot reach 30 t/s even at 100% acceptance (22.4).**
- **block_size=32 reaches 30 t/s at 81% acceptance** (33.3 at 90%, 37.0 at 100%).

So the acceptance bar is now an exact number: **81% at block_size=32**. Between
81% and 100% every extra point of acceptance is worth 0.37 tok/s.

### Residual uncertainty in the b32 column

The 595 ms verify is read from the 30-token prefill row (measured); the 242 ms
drafter is the M=16 figure scaled 2x, **not measured**. If the drafter's M=32
path is superlinear the round grows and the acceptance bar rises in step. The
cheap check is extending `dflash_gemm_m16` to M=32 — it is the one number
between this table and a decision.

---

## Drafter M=32 measured — the b32 column is now measured, not projected

`dflash_gemm_m16` gained a `KAPLUMBA_DFLASH_M` override so the same 36 linears
can be timed at a candidate block size without editing the checkpoint config.
An override prints its numbers and does not trip the R19 `< 100 ms` assertion,
which is scoped to the checkpoint's own block size.

Command (both runs, release, no model load):

    cargo test -p kaplumba-engine --release --test dflash_gemm_m16 -- --nocapture
    KAPLUMBA_DFLASH_M=32 cargo test -p kaplumba-engine --release --test dflash_gemm_m16 -- --nocapture

| | M=16 | M=32 |
|---|---|---|
| 35 per-layer projections | 78.65 ms | 139.94 ms |
| `fc` | 8.713 ms | 16.774 ms |
| **drafting total** | **87.36 ms** | **156.71 ms** |
| effective bandwidth | 9.4 GB/s | 5.1 GB/s |

Two things this settles:

1. **M=32 is SUBlinear** — 1.794x the time for 2x the rows, so the earlier
   2x-scaled 242 ms estimate was **wrong by 85 ms** in the favourable direction.
2. **The drafter is compute-bound, not read-bound.** Doubling the rows does not
   double the bytes; if the kernel were bandwidth-bound M=32 would cost the same
   as M=16. Instead GB/s *falls* (9.4 -> 5.1) while time rises 1.79x. So the
   earlier "3-4x of headroom vs the target's 28-46 GB/s colmajor path" is real
   *and* the limit at M=32 is arithmetic throughput, not weight traffic. Fixing
   it is a tiling problem, not a staging problem.

## The 30 t/s answer, both sides measured

    block=16:  545 verify +  87.4 draft =  632 ms round -> 23.7 t/s at 100% acc
    block=32:  595 verify + 156.7 draft =  752 ms round -> 41.2 t/s at 100% acc

               block=16    block=32
    acc 100%     23.7        41.2
    acc  90%     21.3        37.1
    acc  80%     19.0        33.0

    block=16: 30 t/s needs **126.5% acceptance** -> arithmetically impossible
    block=32: 30 t/s needs **72.7% acceptance** -> the real bar

**At block_size=16 the target is unreachable at any acceptance rate.** At
block_size=32 it needs 73% acceptance, and every point above that is worth
about 0.4 tok/s. That is the whole decision: `block_size`, 32, 73% acceptance.

Residual: the 595 ms verify is the 30-token prefill row (measured step function,
upper bound sound for a 32-row batch); the 156.7 ms drafter is now measured
directly. The one unmeasured interaction is that a b16-trained drafter read at
b32 may lose acceptance — which is exactly what the 73% bar is asking about.

---

## 2026-10-11 — T9's remaining blocker is on the target, and it is a hardcoded `1`

I went to implement T9 with the corrected (non-anchor-refeeding) shape and found the
structural blocker that makes it more than a caller change. It is worth recording
precisely because it invalidates the "T9 is just wiring" reading.

### The target returns logits for exactly ONE row

`TextModel::encode_lm_head` (`crates/engine/src/model.rs:5050`) runs:

    self.elements.encode_rms_norm(.., &self.scratch.last_hidden, .., 1, hidden, eps)
                                                                       ^
    self.linear.encode_gemv(.., input: &self.scratch.last_hidden, ..)

A literal `1` for the row count, over `scratch.last_hidden`, which is a single
row. `forward_buffer_into`'s own doc says it outright: *"The LM head already runs
on the last row only, so only the final chunk's logits are returned."* That is
correct for prefill (only the last row predicts the next token) and for decode
(one row by definition).

**A batched verify needs `n` logits rows, one per draft position.** `L[k]` is the
full vocabulary distribution at position `position + k - 1`; there is no way to
decide acceptance for `draft.tokens[k]` without it. So the verify path cannot use
`forward_buffer`, and T9 is blocked on a target-side change, not on the round
loop:

1. the final hidden state must survive for **all `n` rows**, not just row `n-1`
   (`scratch.last_hidden` currently keeps only that one), and
2. the head must run at `n` rows — `encode_rms_norm(rows = n)` plus a GEMM at
   `M = n` instead of the `encode_gemv` — and expose an `[n, vocab]` read-back.

### The cost is small once written, which is the point

The head is ~1.05 GB at 6 bpw. `encode_gemv` at M=1 reads all of it; a batched
GEMM at M=n reads the same bytes once and amortises them over `n` rows, so the
head costs ~4.3 ms at the measured 245 GB/s whether n is 1 or 32. **The verify
head is nearly free.** What is not free is that it does not exist yet, and that
adding it touches the same 6 500-line hot-path file that `forward_buffer` lives
in — the file whose single-row behaviour every decode measurement depends on.

### So the full T9 chain is now specified end to end

| # | piece | state |
|---|---|---|
| 1 | drafter tap export (T7) | **done** — `dflash_taps` 1/1, sabotaged |
| 2 | chunked tap streaming (T8) | **done** — `dflash_streaming` 1/1, sabotaged |
| 3 | the corrected round shape | **specified** (see above): feed `draft.tokens[0..n]` at `position..position+n-1`, carry `previous_logits` for `L[0]`, read `L[1..=n]` off the returned rows, retract `n - k` |
| 4 | **multi-row head on the target** | **missing** — the blocker above |
| 5 | greedy-equivalence test (T10) | not written; must precede believing (3) |
| 6 | CLI wiring + measurement (T11) | not written |

Piece 4 is the one that has to be built carefully, and it is the reason I am not
writing T9 in the remaining budget: an unverified edit to `encode_lm_head` risks
the decode path that every number in this note is measured through. The
single-row `KAPLUMBA_FORCE_LOAD` decode baseline (15.283 tok/s) is the thing worth
protecting, and a batched head is exactly the change that could silently move it.

Nothing above changes the decision the measurements support: **block_size=32,
~73% acceptance for 30 tok/s.**

---

## 2026-10-11 — multi-row head: design closed, including the memory tradeoff

I scoped the multi-row head to the point of implementation and found the last
unknown, which is a **sizing** decision rather than a correctness one.

### The final hidden state for ALL rows is already there

`forward_chunk` keeps every row's hidden through the layer stack and only narrows
at the head:

    let final_hidden = if layers.is_multiple_of(2) { &self.scratch.hidden_a }
                       else                         { &self.scratch.hidden_b };
    self.elements.encode_copy_last_row(encoder, final_hidden, &self.scratch.last_hidden, tokens, hidden)?;

So the multi-row head is **additive** — a new method, not a re-plumbing of
`forward_buffer_into`. `forward_buffer_into`'s signature and its 3 external call
sites (`generate.rs:967`, `examples/prefill_probe.rs:98`, plus the internal
pipelined path) do not have to change. `scratch.normed` is already
`max_tokens x hidden`, so even the normalise target exists.

### The one thing that does NOT fit: the head's fp32 accumulator

`encode_gemm`/`encode_gemv_mrhs` accumulate into `scratch.raw`, sized

    raw_elements = (max_tokens * inter).max(vocab)      // 32*21504 = 688128 elems = 2.75 MB

but a head GEMM over `queries` rows needs `queries * out_features` = `queries *
262144`:

| rows | head raw (f32) | logits (f16) | total |
|---|---|---|---|
| 1 | 1.0 MB | 0.5 MB | 1.6 MB |
| 8 | 8.4 MB | 4.2 MB | 12.6 MB |
| 16 | 16.8 MB | 8.4 MB | 25.2 MB |
| **32** | **33.6 MB** | **16.8 MB** | **50.3 MB** |

At 32 rows the needed accumulator is **12x the existing `raw`**. A new buffer is
required either way; the question is its size, which is exactly the kind of
decision AGENTS.md §8 ("explicitly owned, fixed-size, independent of context
length") and R20 (peak footprint) govern rather than a late edit.

### Recommended shape: batch the head at 8 rows

Run the head in fixed 8-row batches inside the verify. 32 rows = 4 batches,
allocating 12.6 MB instead of 50.3 MB, at a cost of 3 extra reads of the 1.05 GB
head:

    3 * 1.05 GB / 245 GB/s = **12.9 ms per round**, against a 752 ms round => 1.7%

So the batching is 1.7% of the round to quarter the buffer, and it keeps the
allocation **independent of block size** — raising `block_size` to 64 later does
not grow it. That satisfies §8 properly instead of merely passing this round.

### Where this leaves T9

Every piece is now specified: the round shape, the retract count, the accessors
(landed), the head change (design above), the T10 equivalence test that must
precede believing any of it, and the T11 measurement. Nothing here changes the
settled conclusion — **30 t/s needs `block_size=32` and >=72.7% acceptance** — and
none of this design work moves the decode baseline, which is the number worth
protecting at 15.283 tok/s.

---

## 2026-10-11 — TWO CORRECTIONS: the model path, and the acceptance arithmetic

Both of these invalidate numbers I wrote earlier in this file. Correcting them in
place rather than leaving the earlier text to mislead.

### Correction 1 — the checkpoint path in every command above is wrong

Every command block in this file says `--model ~/.kaplumba/models/Gxma/Gxma`.
**That path does not exist.** The real 3-bit target is

    ~/.kaplumba/models/Gxma-repacked

confirmed by size: `model-00001-of-00002.safetensors` is 8 498 859 202 B and
`model-00002-of-00002.safetensors` is 7 529 529 792 B, the exact pair the
bandwidth arithmetic uses (16.028 GB). The **numbers** recorded above are right
(they came from a real run); only the path label was wrong. Anyone re-running
these commands verbatim got
`checkpoint inspection failed: ... config.json: No such file or directory`.

Other checkpoints present: `Gxma-4-bit-split` (19 G), `Gxma-uncensored*` (15 G),
`Drafter` (733 M, the DFlash drafter).

### Correction 2 — the acceptance arithmetic was wrong, by up to 2.9x

I wrote that a b32 round yields `31 * acceptance` tokens, and derived "30 t/s needs
72.7% acceptance" from it. **That is not the formula.** In speculative decoding the
draft is a *prefix*: the round emits tokens while the acceptance coin keeps coming
up heads, so with per-token acceptance `alpha` and `gamma` draft tokens,

    E[tokens per round] = sum_{j=0}^{gamma} alpha^j = (1 - alpha^(gamma+1)) / (1 - alpha)

The linear form agrees only at `alpha = 1` and diverges fast -- at `b32, alpha=0.9`
it overstated by **2.9x** (37.1 t/s claimed, 12.8 t/s correct).

Round times used (both sides measured): b16 = 545 verify + 87.4 draft = 0.6324 s;
b32 = 595 + 156.7 = 0.7517 s.

| alpha | E[b16] | E[b32] | t/s b16 | t/s b32 |
|---|---|---|---|---|
| 0.80 | 4.86 | 5.00 | 7.7 | 6.6 |
| 0.90 | 8.15 | 9.66 | 12.9 | 12.8 |
| 0.92 | 9.21 | 11.63 | 14.6 | 15.5 |
| 0.94 | 10.47 | 14.37 | 16.6 | 19.1 |
| 0.96 | 11.99 | 18.23 | 19.0 | 24.3 |
| 0.977 | 13.52 | 22.83 | 21.4 | **30.4** |
| 1.00 | 16.00 | 32.00 | 25.3 | 42.6 |

**Baseline to beat: 15.25 tok/s** (measured below, this box, clean).

- **b16: break-even alpha = 0.928; 30 t/s is UNREACHABLE — even at alpha = 1 the
  ceiling is 25.3 t/s.** This conclusion survives the correction, but for a
  different reason than I gave: not "acceptance is capped", but "16 tokens per
  round at 545 ms is 25.3 t/s, period".
- **b32: break-even alpha = 0.919; 30 t/s needs alpha = 0.977.**

**Below alpha ~= 0.92, DFlash is SLOWER than plain decode.** At alpha = 0.9, b32
gives 12.8 t/s against the 15.25 baseline: the drafting cost (156.7 ms) is not
repaid. That is the single most important number in this file, because it means
the whole spec is net-negative until acceptance is high.

### The spec's own gate was below break-even

AC4 said "acceptance >= 2.4 tokens per target forward" at b16. Under the correct
formula, 2.4 tokens/round at b16 is `alpha ~= 0.58`, which is **4.4 t/s** -- a
**3.5x slowdown** against plain decode. I set that gate without checking it
against the baseline it had to beat. It is retired below.

### Temp 1.0 / top_k 64 changes the VERIFIER, not just the sampler

The requested target is *coherent prose at temperature 1.0, top_k = 64*. Greedy
`argmax` comparison is **not a valid verifier for that** and would violate R6:
matching argmax turns the output greedy, which is a different distribution from
the one asked for.

Distribution-exact sampling requires **speculative sampling** (Leviathan et al.
2023; Chen et al. 2023): draft `x ~ q`, accept with probability
`min(1, p(x)/q(x))`, and on rejection resample from the normalised residual
`(p - q)+`. Then per-token acceptance is

    alpha = sum_x min(p(x), q(x)) = 1 - TV(p, q)

i.e. acceptance *is* the total-variation distance between the drafter and target
distributions **at temperature 1.0 under top_k = 64 truncation** -- not the
argmax agreement rate, and not the temperature-0 number. So T9's design and T10's
test both need restating for sampling:

- T9 must carry `p` and `q` rows (not argmax) and implement the accept/residual
  step; the corrected round shape in the section above still holds for the
  positions and the retract count.
- **T10's "greedy sequence equivalence" is the wrong criterion here.** For
  temp > 0 the admissible test is distribution-exactness: with a fixed seed and a
  fixed draft stream, the round must reproduce the *same sampling procedure's*
  distribution -- checked by comparing the empirical token distribution (and the
  rejection/residual path) against non-speculative sampling, not by byte-equality
  of sequences. Byte-equality is only the temp-0 special case.

**`alpha` is now the blocking unknown.** Nothing above tells us what it is: it
depends on how close a b16-trained 733 MB drafter's temp-1.0 top-64 distribution is
to a 31B target's, and on whether a b16 checkpoint read at b32 keeps it. If it
lands below 0.92 the honest answer is **do not ship this** -- it is a pessimization
-- and that is measurable *before* building T9, by comparing the drafter's and
target's next-token distributions on real contexts.

---

## 2026-10-11 — `draft_block` contradicts its own documented contract, on an untested path

Found while trying to measure `alpha` (which is now the blocking unknown). This
must be resolved **before** any alpha number means anything, because alpha is a
function of exactly which logits rows become the drafts.

### The evidence

Three sources say the anchor (block row 0) must be dropped:

1. `DflashConfig::draft_tokens` (`dflash.rs:396`):
   > `// A block is [anchor, mask x (block_size - 1)] and the anchor row predicts a
   > // token the target has already emitted, so it is dropped.`
   > `self.block_size - 1`
2. `DflashModel::draft_block` (`dflash.rs:1588`):
   > `// Only the block_size - 1 masked rows are new tokens; the anchor's own`
   > `// logits predict a token the target has already produced.`
3. `forward_block` (`dflash.rs:1310`) — the anchor is row 0, unambiguously:

        let source = if row == 0 { context.anchor_embedding } else { mask_row };

But the selection loop (`dflash.rs:1596`) takes **rows `0..draft_rows`**, i.e.
`0..block_size-1`:

    let tokens = (0..draft_rows)
        .map(|row| argmax_row(&logits, vocab, row))
        .collect();

That is **the anchor (row 0) included and the last masked row
(`block_size - 1`) omitted**. The count is right; the start index is not. The
reading consistent with all three sources is `1..=draft_rows` (= `1..block_size`,
exactly the `block_size - 1` masked rows).

### Consequences if it is the bug it looks like

- The **first** draft token is the anchor's argmax: a token the target has
  already emitted, so the round would spend its first verify slot re-checking a
  token that is not a prediction at all.
- The **last** of the `block_size - 1` real drafts is never produced, so the
  block is one token short of what the geometry says.
- At temperature 1.0 both effects corrupt the draft *stream*, so any measured
  `alpha` would be an artefact of the selection rather than of the drafter's
  quality.
- **No test covers this.** `dflash_golden` exercises `forward_block` (4/4) and
  never calls `draft_block`; `dflash_cache` tests cache arithmetic. Grepping the
  test tree for `draft_block` finds no assertion on `tokens`. This is the
  AGENTS.md §3.4 pattern again: a documented invariant with nothing that could
  fail.

### Why I am recording it rather than "fixing" it in place

One thing does not fit the story, and it is the reason I stopped instead of
patching: `BlockContext::position` is documented as *"absolute position of the
first block row (== the target's `next_position`, i.e. one past the last context
row)"*, and the first block row **is** the anchor. If the anchor sits at
`next_position` while the last emitted token sits at `next_position - 1`, then the
anchor's logits predict a genuinely new token and "already emitted" is wrong --
which would make `(0..draft_rows)` correct and the two doc comments wrong.

I cannot settle that by reading position arithmetic, and guessing is precisely how
this file already acquired a re-fed-anchor bug in T9. So this is an **open
question requiring the z-lab reference** (`dflash/model.py`, `_rejection_sample` /
`dflash_generate`, and `reference/exllamav3/architecture/dflash.py`, which the
grep above did not surface a match in) to say which rows become drafts.

### The sharp test, either way

Whichever way the reference rules, `draft_block`'s `tokens` must equal the argmax
of **one specific** contiguous row range of `draft_logits(&block.hidden, ...)`,
and today nothing asserts it. `DflashBlock::hidden` and `draft_logits` are both
public, so the test is: recompute the logits from the returned block and assert
`tokens == argmax(rows X)`, plus `tokens != argmax(rows 0..draft_rows)` so the
test cannot pass vacuously by the two ranges agreeing. `argmax_row` is private, so
the test needs a local f16 argmax -- which also makes it independent of the
engine's own helper.

---

## 2026-10-11 — RESOLVED against the reference: `draft_block` was off by one. Fixed.

The open question above is settled. The z-lab reference **is** present
(`/private/var/folders/tp/.../opencode/zlab/dflash`) and is explicit.

### The reference says rows `1..`, so the engine was wrong

    dflash/model.py:249   draft_hidden = model(...)[:, 1 - verify_size :, :]
    dflash/model.py:265   block_output_ids[:, 1:] = _sample_probs(draft_probs)
    dflash/model.py:268   block_output_ids[:, 1:] = torch.argmax(draft_logits, dim=-1)
    dflash/model.py:281   _rejection_sample(block_output_ids[:, 1:], ...)
    dflash/model.py:291   (block_output_ids[:, 1:] == posterior[:, :-1])...

`[:, 1 - verify_size :, :]` with `verify_size == block_size` is `[:, -(block_size-1):, :]`
-- **the last `block_size - 1` rows**, i.e. the masks, with the anchor's row 0
excluded. `argmax` of exactly those fills slots `1..`. The engine's `(0..draft_rows)`
took the anchor and dropped the last mask. **Fixed to `(1..=draft_rows)`** in
`crates/engine/src/dflash.rs`, with the citation in the comment.

So `BlockContext::position`'s "first block row == next_position" is *not* in
conflict after all: the anchor is re-fed at the block's first position by design,
and the reference's round re-writes `output_ids[start]` with it
(`output_ids[start : start+acc+1] = block_output_ids[:acc+1]`), so the anchor is
the already-emitted token that seeds the block and is then overwritten in place,
not a duplicated row.

### `draft_block` and `draft_logits` have no callers at all

    $ grep -rn "draft_block\|draft_logits" crates/ --include=*.rs | grep -v dflash.rs
    (only doc-comment mentions, plus the new test)

They are unexercised dead code. That is *why* the off-by-one survived a spec, a
plan, a golden harness and a task list: R15/T9 was never wired, so nothing ever
ran the selection.

### The round semantics, now read off the reference (for T9)

- Block ids: `[anchor, draft_1 .. draft_{b-1}]` at positions
  `position_ids[start : start + verify_size]`.
- Drafts from the drafter's **last `b-1` rows**; `block_output_ids[:, 1:] = drafts`.
- Target runs once over the whole block; `verify_size > 1` implies
  `output_hidden_states=True`.
- `temperature > 0` -> `target_probs`/`draft_probs` are
  `_sampling_probs(logits, temperature, top_p, top_k)`, then **`_rejection_sample`**
  (`model.py:94`): accept while `rand()*q < p`, and on rejection resample from
  `residual = clamp(p - q, 0)` renormalised. This is the distribution-exact
  verifier the temp-1.0 requirement needs -- confirming the note above that greedy
  argmax matching is not the right criterion here.
- `temperature == 0` -> `acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(1).sum(1)`,
  the greedy prefix match, and `bonus = posterior[:, acceptance_length]`.
- Emit `acc` drafts + 1 bonus; `output_ids[start : start+acc+1] = block_output_ids[:acc+1]`,
  `output_ids[start+acc+1] = bonus`; then `_crop_to(cache, start + produced)`.

Note the acceptance is over `posterior[:, :-1]` vs `drafts` -- **`b-1` drafts, not
`b`**. The engine's round must use `b-1` verify positions, which matches
`Config::draft_tokens()`.

### Test written; it does not yet reach its assertion

`crates/engine/tests/dflash_draft_rows.rs` (ignored). It compiles and runs, and
aborts inside `draft_block` with

    Kernels("invalid kernel input: packed buffer length 2208470208 does not match 1056964608")

`1056964608 == 262144 * 5376 * 6 / 8`, the target head's packed size, so the
expectation is right and the supplied buffer is the wrong object --
`draft_logits` hands `head.metal_buffer` straight to the drafter's encoder, and
since nothing ever called it, this plumbing has never been exercised. The
assertion (tokens == argmax of rows `1..=draft_rows`, with an `assert_ne!` guard so
it cannot pass vacuously when both ranges agree) is in place and will be the
immediate check once the buffer view is fixed.

### The fix is now covered, without either checkpoint

The row selection was extracted from `draft_block` into a free function,
`select_draft_tokens(logits, vocab, draft_rows)`, and unit-tested in
`dflash.rs`'s own `#[cfg(test)] mod tests`. Two tests, both GPU-free and in the
**default** suite (`cargo test -p kaplumba-engine --release --lib dflash::`):

- `draft_tokens_are_the_masked_rows_not_the_anchor` -- the anchor's row is given a
  winner no masked row shares, so including it is observable rather than
  coincidental.
- `the_two_candidate_ranges_are_distinguishable` -- guards the first against
  passing vacuously, by asserting the two candidate ranges differ on this input.

**Sabotage, run and observed (§3.4):** reverting the function to `(0..draft_rows)`
fails both with

    assertion `left == right` failed: the drafts must be rows 1..=draft_rows;
      taking rows 0..draft_rows would put the anchor's token first and drop the
      last masked row
    assertion `left != right` failed: the two ranges must differ on this input

exit 101. Reverted, re-run green, and the 16-test integration suite re-run green
after the `dflash.rs` change (no regressions).

`dflash_draft_rows.rs` (ignored) stays as the end-to-end version and still does
not reach its assertion, for the head-buffer reason above. The unit tests are the
ones that actually cover the fix; the integration test is the one that would cover
the plumbing, and that plumbing is still first-touched-today.

---

## 2026-10-11 — self-disproof attempt: the verify estimate HOLDS; D2 diagnosed to the byte

Asked to disprove my own pessimistic conclusion, I attacked the two assumptions it
rests on. The first broke in my favour; the second did not.

### ttft is NOT mostly fixed overhead -- the 595 ms verify is real

My whole bar depends on the M=32 verify costing ~595 ms, and I had taken that from
a `ttft` number, which bundles startup cost. If the fixed part had been large, a
mid-decode verify would be far cheaper and the bar would collapse. Measured
directly (`--max-new-tokens 1`, varying prompt length, reading `prompt_tokens` and
`ttft_ms`):

| prompt tokens | ttft ms |
|---|---|
| 14 | 317.5 |
| 21 | 457.5 |
| 29 | 587.8 |
| 37 | 1326.5 |
| 61 | 1341.0 |
| 109 | 1374.8 |

Linear fit for the `<= 32` bucket: **`ttft ~= 65.5 + 18.0 * n` ms**. The intercept is
**65.5 ms -- exactly one decode step** (the one unavoidable weight read), not the
400 ms that would have rescued the estimate. At n = 32 that is ~641 ms, so my
595 ms was mildly **optimistic, not pessimistic**. The conclusion survives.

### New finding: the 32-row cliff is `SmallBatchMrhs`, and it argues for b32 hard

The step from 29 -> 37 rows costs **+739 ms and then stays flat** (61, 77 and 109
rows all land in 1341-1375 ms). That is not chunking -- taller prompts would keep
growing. It is the route switch documented in the code:

> `LinearRoute::SmallBatchMrhs => // ... Falls back to encode_gemm internally for
> ineligible shapes (rate/codebook != MCG 3 bpw, queries > 32)`

So `queries <= 32` takes the fast small-batch path at **18.0 ms/row**, and anything
past 32 falls onto `encode_gemm` at ~1330 ms flat. **This is an independent,
measured argument for `block_size = 32` and against 64** -- I previously rejected
b64 only on the acceptance arithmetic.

### D2 root cause, measured rather than guessed

`draft_block`'s first-ever caller produced the printed head metadata:

    head: in=5376 out=262144 rate=Integer(6) packed_len=Some(528482304)
          host_packed_words=0 metal_buffer=Some(7529545728) metal_offset=5321075520

- `metal_buffer` is the **whole 7.53 GB shard** (`model-00002-of-00002` is
  7 529 529 792 B).
- The trellis starts at `metal_offset = 5 321 075 520` and is
  `packed_len * 2 = 1 056 964 608` B.
- `length - offset = 7 529 545 728 - 5 321 075 520 = 2 208 470 208`, which is
  exactly the number the error reported, so **1 151 505 600 B of further tensors
  follow the head in the same buffer**.
- `host_packed_words = 0`: the head is fully buffer-resident, so a test cannot
  rebuild it from host words.

`validate_buffer_shape` (`linear.rs:4197`) requires
`length - offset == expected`, which is only true when the tensor is the buffer's
**tail**. The drafter's own linears get dedicated buffers and satisfy it; the
target's head does not. `encode_gemv` -- the head's only ever-exercised path --
accepts the layout. **Two validators disagree, and only one has ever run.**

**The fix, and why I did not apply it:** the preferred change is an explicit packed
length on `LinearBufferRequest` (keeps the equality check for all other callers);
the alternative is bounding the region by `packed_len` instead of the buffer end.
Both touch `validate_buffer_shape`, which guards **every GEMM in the engine**,
including the MMA path this file's comments already record as having shipped
numerically wrong once. That is not a change to make blind at the end of a session
on a box that is currently swapping (5.83 GiB in use, 1.2 GiB "available", which is
why the run needed `KAPLUMBA_FORCE_LOAD=1`).

`dflash_draft_rows.rs` is therefore an **expected-fail diagnostic**: it is
`#[ignore]`d, so the default suite is unaffected, and its failure message carries
the whole diagnosis above.

---

## 2026-10-11 — D2 FIXED. The target-head plumbing works, and the round is unblocked.

### The change

`gemm_chunk_buffer`'s validator took the packed tensor to be everything from
`packed_offset` to the buffer's end, which is false for a tensor that is not the
buffer's tail. Rather than relax that check -- it is the only thing catching a
mislocated tensor for every other caller -- a length-aware path was added:

- `linear.rs`: `gemm_chunk_buffer_with_stats` is now a 4-line wrapper that measures
  the region and delegates to a private
  `gemm_chunk_buffer_with_stats_region(request, packed_bytes)`, which is the old
  body with `validate_buffer_shape` receiving the supplied length. **All 13
  existing construction sites are untouched and take byte-identical code paths.**
- New public `gemm_chunk_buffer_sliced(request, packed_bytes)`.
- `draft_logits` now passes `head.packed_len * 2` (falling back to the region when
  `packed_len` is absent), with the measured numbers in the comment.

### Result: the first end-to-end run of the drafter's head path

    head trellis 1056964608 B at offset 5321075520 inside a 2208470208 B region
      (1151505600 B of trailing tensors); sliced validation used
    test draft_tokens_are_the_masked_rows_not_the_anchor ... ok

`dflash_draft_rows` is no longer an expected-fail diagnostic: it loads the 16 GB
target **and** the drafter, builds the double-drift context, runs `draft_block`,
and asserts the drafts are the argmax of rows `1..=draft_rows`.

**Sabotage, run and observed:** reverting `select_draft_tokens` to `0..draft_rows`
fails it on the real models with

    assertion `left == right` failed: draft_block must return the argmax of rows
      1..=draft_rows (the masked rows); it returned the argmax of rows 0..draft_rows,
      which includes the anchor's row -- a token the target has already emitted --
      and drops the last masked row

exit 101. Reverted, re-run green (1/1), and the 16-test non-ignored suite re-run
green (16/16, exit 0) -- **and checked unfiltered**, because a first pass with
`--lib dflash::` in the same invocation propagated that filter to every target and
reported `0 passed; N filtered out` for all of them, which is a vacuous pass rather
than a passing test.

### What this unblocks

`draft_logits` on the real target head is now exercised, so **alpha is finally
measurable**: the drafter returns one logits row per block row, the target's own
`last_logits()` gives `p`, and `alpha = sum_x min(p(x), q(x)) = 1 - TV(p, q)` over
the temp-1.0 / top-64 truncated distributions. That is D3, and it decides whether
30 t/s is reachable at all: break-even is 0.919 at b32, 30 t/s needs 0.977.
