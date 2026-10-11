# spec/dflash-correctness — tasks

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done (commit hash appended).
Update this file **immediately** after each task (AGENTS.md §3.3).

> **STATE (2026-10-11).** Last commit `7b20fa1`. **T1, T2, T3a, T3, T5, T4, T4b
> and T14 are done and committed.** `DflashConfig`, `DflashModel`
> (fuse_context / project_context_kv / forward_block / draft_logits) and the
> four-test golden harness are all in tree. See `implementation_notes.md`
> §"2026-10-11 — T14 landed" for the fp32 findings and the exact commands.
>
> **The fp16 overflow that blocked this spec is fixed — see T14.** The drafter's
> residual stream is now f32 end to end and T4 is green 4/4 (`99db335`). T6 is
> green 3/3.
>
> **READ T12'S RESULT BEFORE PLANNING ANYTHING ELSE.** The M=16 GEMM path runs
> at **9.9 GB/s**, ~25x below streaming peak. The target's verify pass is an
> M=16 pass over 15.5 GB of weights, so it is ~1.55 s, not ~300 ms. Per block
> that is ~193 ms of drafting + ~1.55 s of verify for at most 15 tokens — a
> ceiling of **~8.6 tok/s even at 100% acceptance**. A 40 tok/s target is not
> reachable by improving the drafter, acceptance, or the round; it requires the
> M=16 GEMM to be fixed first. See `implementation_notes.md` §"T12".

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
      **Commit `99db335`.**
- [x] **T12 — `gemm_chunk` M=16 benchmark.** `crates/engine/tests/dflash_gemm_m16.rs`,
      all 36 linears at M=16, median of 9. **Result: the M=16 path runs at 9.9
      GB/s — ~25x below this box's streaming peak, so it is instruction-bound,
      not bandwidth-bound.** The 35 per-layer projections cost 78.89 ms per
      block and `fc` 8.72 ms. R19's gate (< 100 ms for the projections) passes at
      78.89 ms, but see the note below: this is the binding constraint on the
      whole spec, not the drafter. *(uncommitted)*
- [x] **T4b — weight-scale sanity check.** `crates/engine/tests/dflash_weight_scale.rs`
      dequantises `layers.0.self_attn.o_proj` from both checkpoints on the host
      (no model load, no GPU). Target **1.031x unit variance** — validates the
      rate/codebook/state decode; drafter **9.88x**, which is an open question.
      **Passes.** *(uncommitted)*

## Drafter state

- [x] **T6 — drafter KV cache.** `DflashKvCache` (per-layer flat K/V, layers
      0–3 ring at `sliding_window`, last layer unbounded), `append_context`,
      `rewind_context`, `draft_block`, `DflashDraft`, and a host-side
      `argmax_row`. **`BlockContext` was restructured**: it now carries
      `ctx_first: &[usize]` and derives each layer's row count from its own
      buffer, because a sliding layer and the full-attention last layer hold
      different rows once the context passes the window — the old single `n_ctx`
      could not express that. **Tested by** `crates/engine/tests/dflash_cache.rs`
      (3 tests: chunk invariance R6, rewind equivalence R13, ring absolute
      positions), **3/3 green**. Sabotage: skipping the ring eviction fails the
      ring test with `layer 0 should hold exactly the window`; dropping the
      chunk offset from the RoPE position fails chunk invariance and rewind with
      `layer 0 keys differ`. *(uncommitted)*

## Target-side feature export

- [x] **T7 — tap export.** `TextModel::enable_taps` / `take_taps` / `tap_layers`
      plus an opt-in `TapScratch` (66 MB for six taps, deliberately NOT in
      `ForwardScratch` so the default path does not pay it). The blit sits inside
      `forward_chunk`'s layer loop and **selects its ping-pong source per layer**
      (`hidden_b` for even, `hidden_a` for odd) — the taps
      `[1, 12, 23, 35, 46, 57]` are four odd and two even, so a hardcoded source is
      wrong for two of six. **Tested by** `crates/engine/tests/dflash_taps.rs`
      (ignored; needs `KAPLUMBA_TEST_MODEL`): shape+signal, bitwise-identical
      logits with taps on/off, the parity check, and request order. **Passes:**
      tap[1] vs tap[2] differ on 75180/75264 elements. **Sabotage:** hardcoding
      the source to `hidden_a` makes them differ on **0/75264** and fails the
      parity assertion. *(uncommitted)*
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
