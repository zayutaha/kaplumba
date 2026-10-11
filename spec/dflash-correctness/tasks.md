# spec/dflash-correctness — tasks

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done (commit hash appended).
Update this file **immediately** after each task (AGENTS.md §3.3).

> **HANDOVER STATE (2026-10-11).** Last commit `05d1208`. **T1, T2, T3a are
> committed.** `DflashConfig` + `DflashModel` (fuse_context /
> project_context_kv / forward_block / draft_logits) plus `dflash_golden.rs`,
> `dflash_nan_probe.rs` and `dflash_weight_scale.rs` are **written but NOT
> committed**. See `implementation_notes.md` §"2026-10-11 — T4" for the exact
> next commands.
>
> **Read this before touching the block forward.** The drafter overflows fp16 in
> layer 0 (`mlp_out` reaches 152 287) and exllamav3 runs its attention and MLP
> outputs in **fp32** (`architecture/dflash.py:277,308`). The fp16 element path
> cannot be tuned into passing; it needs an fp32 variant. **T14 below is that
> work and it blocks T4, T9, T10 and T11.**

## Pre-flight

- [x] **P0** Engine builds and loads `~/.kaplumba/models/Gxma-repacked`;
      `reference/exllamav3/exllamav3/architecture/dflash.py:107-208` and
      z-lab's `dflash/model.py:340-480` read end to end. **Tested by**:
      `cargo check -p kaplumba-engine --release` is clean and the reference
      paths were transcribed into `implementation_notes.md`.

## Correctness prerequisites

- [x] **T1 — drafter loader.** `crates/engine/src/dflash.rs`: EXL3 mcg 4.0 bpw
      into `QuantizedLinear`, `fc` = 32256→5376, per-layer shapes, `mcg` =
      `MCG_MULTIPLIER`, `unknown_tensors` surfaced. **Commit `5c8b9fe`.**
      **Tested by** `crates/engine/tests/dflash_load.rs` (3 tests).
- [x] **T2 — Gemma `+1.0` norm offset.** BF16→F16 then `+1.0` in f32.
      Caught a real double-widening bug on the way. **Commit `5c8b9fe`.**
      **Tested by** `dflash_norm_weights_carry_the_gemma_offset`.
- [x] **T3 — config + block input construction.** `DflashConfig::load` parses
      every field the plan lists, validates taps against
      `num_target_layers`, and checks `kv_heads` divides `q_heads`.
      `forward_block` builds `[anchor, mask × 15]` from the caller's raw
      embedding rows. *(uncommitted)*
- [x] **T3a — `dflash_block_attn` Metal kernel.** New kernel because
      `online_attention` is unconditionally causal and the drafter's
      `full_attention` layer is bidirectional. **Commit `05d1208`.**
      **Tested by** `crates/engine/tests/dflash_block_attention.rs` (3 tests,
      including a cross-check against `online_attention`).
- [x] **T5 — full 5-layer block forward.** `forward_block` + `project_context_kv`
      + `fuse_context` + `draft_logits` (target's tied head + softcap 30).
      `KAPLUMBA_DFLASH_CAUSAL_BLOCK` flag is **not yet wired** — the mask
      selection currently reads `layer_is_sliding` only; add it when the A/B
      decision is made. *(uncommitted)*
- [x] **T4 — golden reference harness.** `crates/engine/tests/dflash_golden.rs`,
      a scalar f32 host reference dequantised from the same checkpoint, four
      bounded tests, **4/4 green**: `fuse_context` (worst 0.006263),
      `project_context_kv`, `forward_block` (worst 0.068110 against reference max
      77.2833 = 8.8e-4 relative; engine max 77.2500 — it carries signal), and
      `dflash_exl3_decode_matches_host_on_every_linear` (all 36 linears, worst
      0.003663). `crates/engine/examples/dflash_taps.rs` is **superseded** by
      `crates/engine/tests/dflash_nan_probe.rs`. Sabotage evidence recorded in
      `implementation_notes.md`. *(uncommitted)*
- [x] **T14 — fp32 element path for the drafter.** Landed as **three new Metal
      kernels**, not a wholesale rewrite: `linear_finalize_rows_f32` (the tiled
      GEMV already accumulated in f32, so only the finalize's narrowing was
      removed), `rms_norm_weighted_f32_in_tg` (f32 in, f16 out — the single
      dtype boundary), and `residual_add_f32`. Only `o_proj` and `down_proj` use
      the f32 gemm, matching exllamav3's `out_dtype = torch.float` on the
      attention block and GatedMLP
      (`architecture/dflash.py:277,308`). **Tested by** T4's `forward_block`.
      **Sabotage:** re-narrowing `residual_add_f32` reproduces the original
      failure exactly — `the comparison is vacuous: the engine carries no signal
      (energy NaN)` — so the f32 add is load-bearing, not incidental.
- [ ] **T12 — `gemm_chunk` M=16 benchmark (do this before T6).** Benchmark
      every drafter shape at M=16; report ms and effective GB/s.
      *Unblocked by T1; do it first — its result decides whether T6's shape
      survives. Host-measurable, needs no model load.*
- [x] **T4b — weight-scale sanity check.** `crates/engine/tests/dflash_weight_scale.rs`
      dequantises `layers.0.self_attn.o_proj` from both checkpoints on the host
      (no model load, no GPU). Target **1.031x unit variance** — validates the
      rate/codebook/state decode; drafter **9.88x**, which is an open question.
      **Passes.** *(uncommitted)*

## Drafter state

- [~] **T6 — drafter KV cache.** `project_context_kv` exists and is validated
      against the host reference. **Not built:** the `DflashKvCache` itself
      (5 layers × 8 kv heads × 128, layers 0–3 ring at 2048, layer 4 full),
      `append_context`, `rewind_context`, and the R13 equivalence test.
      *TODO after T12.*

## Target-side feature export

- [ ] **T7 — tap export.** `model.rs::forward_chunk` gains an optional tap
      sink; `[1, 12, 23, 35, 46, 57]` blits out of `hidden_a`/`hidden_b`.
      Tap buffers added to `ensure_scratch`. **Tested by**: R5's
      `KAPLUMBA_GPU_PROFILE` A/B, and a tap-`[0]` comparison against
      `hidden_states_all`.
- [ ] **T8 — chunked prefill streaming.** Stream each prefill chunk's taps into
      the drafter cache and release. **Tested by**: the chunk-vs-prompt-length
      invariance test.
- [ ] **T9 — the round.** `generate.rs` DFlash loop; rewind on rejection.
      **Tested by**: the 0/5/15 stub-forced acceptance cases and the R13
      two-round test.

## Measurement

- [ ] **T10 — greedy equivalence.** `crates/engine/tests/dflash_equivalence.rs`.
- [ ] **T11 — acceptance.** `crates/engine/examples/dflash_acceptance.rs`.
- [ ] **T13 — peak footprint.** `/usr/bin/time -l` at 1k/2k/8k.

## Stage-gate

- [ ] **S1** acceptance >= 2.4 (AC4) **and** greedy equivalence (AC3) **and**
      footprint under budget (AC6) on the same measured runs. If acceptance
      lands below 2.4, do not tune — report Q1/Q2's A/B numbers and the
      histogram, and stop.
