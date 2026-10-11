# spec/dflash-correctness — plan

## Architecture and data flow

Two models, two caches, one loop per round.

```
                 ┌───────────────────────── target (Gxma-repacked, EXL3 3bpw mcg) ─────────────────────────┐
                 │                                                                                      │
 prompt ids ───► │  prefill (chunked)  ──►  tap hidden states h_1, h_12, h_23, h_35, h_46, h_57          │
                 │                              for every position                                      │
                 │                                                                                      │
                 │  each round: 16 tokens ──► verify logits for all 16 positions + 6 tap rows            │
                 └───────────┬──────────────────────────────────────┬───────────────────────────────────┘
                             │ 6 × [rows, 5376] F16                  │ greedy argmax / bonus
                             ▼                                       ▼
              ┌──────────────────────────────┐            acceptance = longest prefix match
              │  context-feature fusion      │
              │  ctx = RMSNorm( fc( cat taps ) )│  32256 ──► 5376, one gemm_chunk
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼────────────────────────────────────────────┐
              │  drafter KV cache (5 layers, 8 kv heads × 128, F16)        │
              │  per layer:  k = k_proj(RoPE(ctx))   v = v_proj(ctx)       │
              │  layers 0-3 ring at 2048; layer 4 full                     │
              └──────────────┬────────────────────────────────────────────┘
                             │ history K/V
                             ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  block pass: [last accepted token, mask ×15]                          │
  │    raw embedding (no sqrt(hidden))                                    │
  │    layer 0..3: causal + 2048 window over block AND ctx                │
  │    layer 4   : bidirectional within the 16-block rows                 │
  │    q_norm/k_norm over head_dim 128, RoPE neox theta 1e6, no v_norm    │
  │    final norm ─► target tied embed as head ─► softcap 30 ─► argmax    │
  └──────────────┬───────────────────────────────────────────────────────┘
                 │ 15 draft tokens
                 └──── back up to the target's verify pass
```

The drafter's **block rows** attend to the **context rows** that sit earlier
in the same flattened cache (positions `0..kv_position-1`), and to each other
under the block mask. There is no separate cross-attention: the drafter is an
ordinary paged-attention transformer whose history happens to be derived from
the target's features. That is why the drafter needs its own cache rather
than borrowing the target's.

## File / module touchpoints

| path | current behaviour | intended change |
|---|---|---|
| `crates/engine/src/dflash.rs` | — | **new.** `DflashWeights` (EXL3 mcg 4bpw + fp16 scales + BF16→F16 norm weights with `+1.0`), `DflashConfig`, `DflashModel::forward_block`, `DflashKvCache` |
| `crates/engine/src/dflash.rs` (cfg parse) | — | parse `config.json` of the drafter dir; validate against the shard header |
| `crates/engine/src/model.rs` | `forward_chunk` (`:4648`) alternates `hidden_a`/`hidden_b` per layer and returns only the final row | add an optional tap sink: after a layer whose index is in a caller-supplied set, blit `tokens × hidden` F16 out of `hidden_a`/`hidden_b` into a caller buffer. `hidden_states_all` (`:5474`) stays as-is |
| `crates/engine/src/model.rs` | `ensure_scratch` sizes `hidden_a/b/normed/attended/proj` | add `taps: [Buffer; N]` sized `PREFILL_CHUNK * hidden * n_taps`, allocated only when a tap set is configured |
| `crates/engine/src/kv_cache.rs` | ring caches for the target's sliding layers, full caches for full layers | reuse the existing ring/full primitives for the drafter's 4×2048 + 1×full; no new cache type |
| `crates/engine/src/generate.rs` | single-token decode loop | add the DFlash round (R15); keep the non-speculative path byte-identical |
| `crates/engine/src/lib.rs` | module list | register `dflash` |
| `crates/kernels/src/linear.rs` | `gemm_chunk` (`:1171`) is the M>1 EXL3 path used by prefill | **no change in this spec**; R19 only measures it. A fused M=16 path is a separate spec if the benchmark demands it |
| `crates/kernels/src/elements.rs` | `rms_norm`, `rms_norm_unweighted`, `rope`, `residual_add`, `gated_gelu` | no change; reuse. The `+1.0` is folded into the weights at load (R2), not added in the kernel |
| `crates/kernels/src/attention.rs` | `online_attention` with `window` | reuse for the block pass; the bidirectional layer-4 mask needs a `window == usize::MAX`-style "no lower bound" call plus an explicit mask for the 16×16 block |
| `crates/engine/examples/dflash_taps.rs` | — | **new.** the golden-reference dump/compare harness (R16) |
| `crates/engine/examples/dflash_acceptance.rs` | — | **new.** the acceptance harness (R18), patterned on `mtp_acceptance.rs` |
| `crates/engine/src/drafter.rs`, `drafter_model.rs` | MTP drafter | **untouched** (R7 of the non-goals) |

## Interfaces

```rust
// crates/engine/src/dflash.rs
pub struct DflashConfig {
    pub block_size: usize,          // 16
    pub mask_token_id: u32,         // 4
    pub target_layer_ids: Vec<usize>,// [1, 12, 23, 35, 46, 57]  (0-based DecoderLayer indices)
    pub hidden_size: usize,         // 5376
    pub num_layers: usize,          // 5
    pub q_heads: usize,             // 64
    pub kv_heads: usize,            // 8
    pub head_dim: usize,            // 128
    pub intermediate: usize,        // 10752
    pub sliding_window: usize,      // 2048
    pub layer_is_sliding: [bool; 5],
    pub rms_norm_eps: f32,          // 1e-6
    pub rope_theta: f32,            // 1e6
    pub final_softcap: f32,         // 30.0
}

pub struct DflashModel { /* weights + kernels, borrowed &'static like DrafterModel */ }

impl DflashModel {
    /// Fuse one chunk of taps and append the resulting K/V to `cache`.
    /// `taps` is `n_taps` rows of `rows * hidden` F16, in `target_layer_ids` order.
    pub fn append_context(&self, taps: &[&[u16]], rows: usize,
                          position: usize, cache: &mut DflashKvCache) -> Result<(), DflashError>;
    /// Remove the last `n` context rows (rewind after a rejected tail).
    pub fn rewind_context(&self, n: usize, cache: &mut DflashKvCache);

    /// Draft one block. `last_token` is the token at `kv_position`.
    pub fn draft_block(&self, last_token: u32, anchor_embedding: &[u16],
                       kv_position: usize, cache: &DflashKvCache)
        -> Result<[u32; 15], DflashError>;
}

pub struct DflashKvCache { /* 5 layers × (k, v) */ }
```

`anchor_embedding` is the *raw* target embedding row for `last_token` (no
`sqrt(hidden)`), which R7 requires and Q1 may flip.

## On-disk / in-memory layout

- **Drafter weights**: one 769 MB `model.safetensors`, loaded into
  `StorageModeShared` Metal buffers, exactly like `DrafterWeights::load`
  (`drafter.rs:33`). BF16 tensors → F16 in place; **norm tensors only** then
  get `+1.0` added per element.
- **EXL3 packing**: per-linear `trellis` `[in/16, out/16, 64]` i16, `suh` fp16
  `[in]`, `svh` fp16 `[out]`, `mcg` i32. This is the same packing
  `crates/engine/src/exl3.rs` already validates, so R1 is a loader, not a
  decoder.
- **Drafter cache**: `5 × 2 × (rows × 8 × 128)` F16. Rows for layers 0–3 cap
  at 2048 (ring); layer 4 grows with context. 20 480 B per position, 168 MB
  at 8192.
- **Tap buffers**: `n_taps × PREFILL_CHUNK × 5376` F16 holding
  `StorageModeShared`. Bounded by `PREFILL_CHUNK`, never by prompt length
  (R6).

## Sequencing and dependencies

```
T1  loader + config parse ──────────────┐
T2  norm-offset unit test (sabotage)     ├─► nothing else can run before T1/T2
T3  drafter block forward, single layer  │
T4  golden reference vs z-lab (R16)      │   T3 is the first correctness signal
T5  full 5-layer block forward           │
T6  drafter KV cache + context append    │   needs T5's k/v projections
T7  target tap export (model.rs)         │   independent of T1-T6, can run in parallel
T8  chunked prefill tap streaming (R6)   │   needs T6 + T7
T9  round loop in generate.rs (R15)      │   needs T8
T10 greedy equivalence (R17)             │   needs T9
T11 acceptance harness + measurement (R18)│  needs T9
T12 gemm_chunk M=16 benchmark (R19)      │   independent, run it EARLY, it may change the plan
T13 peak footprint (R20)                 │   needs T9
```

**T12 should be run before T6.** If `gemm_chunk` at M=16 turns out to be far
too slow, the whole drafting shape changes (fused kernels) and T6's structure
with it. It is a measurement, not an implementation task, so it is cheap.

## Risks and fallbacks

| # | risk | mitigation |
|---|---|---|
| 1 | **Tap index wrong** → acceptance ≈ 0.3 | Q2 gives the A/B; the ±1 variants are one-line config changes |
| 2 | **Mask semantics wrong** → degrades but does not zero | R8 keeps both behind a flag; measure both |
| 3 | **Raw vs scaled embedding wrong** | Q1 keeps both behind a flag |
| 4 | **`gemm_chunk` at M=16 too slow for `fc`** (86.7 MB read per block) | T12 measures first; fallback is a follow-up fused-kernel spec, not a rushed kernel |
| 5 | **Memory**: 12.96 GiB + 769 MB + 168 MB | R20 measures; R6 bounds the tap scratch; if over budget, reduce `block_size` (the paper §5.5.4 says a b16-trained drafter generalises to b8 with acceptable loss) |
| 6 | **614 MB tap scratch leaking into the decode path** | R6 + a test that asserts tap buffer count/size is invariant with prompt length |
| 7 | **The M=16 target verify pass is a new path for the 3-bit target** | it is the same prefill `gemm_chunk` the target already uses for prompts, only 16 rows wide; assert equivalence against 16 sequential M=1 passes |
| 8 | **Existing MTP tests break** | non-goal 7 forbids touching those files; a regression in them is a revert signal |
| 9 | **`gpuq` contention** during A/B runs | every model run goes through `scripts/gpuq try --`, exit 75 means "do non-GPU work" |

## Requirement → task mapping

| requirement | tasks |
|---|---|
| R1 loader | T1 |
| R2 norm offset | T2 |
| R3 config | T1 |
| R4 tap export | T7 |
| R5 zero cost unused | T7 |
| R6 chunked prefill | T8 |
| R7 block construction | T3, T5 |
| R8 attention mask | T5 |
| R9 RoPE | T5 |
| R10 layer arithmetic | T3, T5 |
| R11 head | T5 |
| R12 drafter KV cache | T6 |
| R13 cache content | T6, T9 |
| R14 rewind | T9 |
| R15 the round | T9 |
| R16 golden reference | T4 (and re-run after T5) |
| R17 greedy equivalence | T10 |
| R18 acceptance | T11 |
| R19 kernel decision | T12 |
| R20 memory | T13 |
| AC1–AC7 | T1, T2, T4, T10, T11, T12, T13 |
