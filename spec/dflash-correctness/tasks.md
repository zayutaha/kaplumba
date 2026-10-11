# spec/dflash-correctness — tasks

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done (commit hash appended).
Update this file **immediately** after each task (AGENTS.md §3.3).

## Pre-flight

- [ ] **P0** Re-clone `reference/exllamav3` and
      `~/.kaplumba/models/Drafter` presence checks; confirm
      `~/.kaplumba/models/Gxma-repacked` is the 3.0 bpw mcg colmajor
      checkpoint and that the engine already loads it
      (`./target/release/kaplumba --model ~/.kaplumba/models/Gxma-repacked
      --prompt "hi" --max-new-tokens 4`). Read `reference/exllamav3/exllamav3/architecture/dflash.py:107-208`
      and `/private/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/zlab/dflash/dflash/model.py:340-480`
      end to end. **Tested by**: the smoke run above succeeds.

## Correctness prerequisites

- [ ] **T1 — drafter loader.** `crates/engine/src/dflash.rs`: load
      `~/.kaplumba/models/Drafter/model.safetensors` (single shard), EXL3 mcg
      4.0 bpw, trellis `[in/16, out/16, 64]` i16 + `suh`/`svh` fp16 + `mcg`
      i32. BF16 → F16 for the norm tensors. Parse the config
      (`block_size`, `mask_token_id`, `target_layer_ids`, `layer_types`,
      `sliding_window`, `rope_theta`, `rms_norm_eps`,
      `final_logit_softcapping`, dims) and validate every one against the
      header's tensor shapes. **Files**: `crates/engine/src/dflash.rs` (new),
      `crates/engine/src/lib.rs`. **Tested by**:
      `crates/engine/tests/dflash_load.rs` asserting (a) 166 tensors accounted
      for, (b) `fc` resolves to in 32256 / out 5376, (c) every `mcg` equals
      `0xCBAC1FED`, (d) a truncated/garbled header fails with the loader's
      named error. *Sabotage*: change the expected mcg constant and confirm
      the test fails naming the tensor.
- [ ] **T2 — Gemma `+1.0` norm offset.** After BF16→F16 conversion add `1.0`
      to every RMSNorm weight. **Files**: `crates/engine/src/dflash.rs`.
      **Tested by**: `crates/engine/tests/dflash_load.rs` asserting
      `input_layernorm.weight[0]` f16 ≈ 1.7539 (0.7539 + 1.0) for layer 0 and
      ≈ 0.3125 + 1.0 for `post_attention_layernorm`. *Sabotage*: drop the
      `+1.0`, confirm the test fails with the actual value printed.
- [ ] **T3 — single-layer block forward, isolated.** Implement the layer
      arithmetic of R10 for one layer with the context rows supplied as a
      host array (no cache yet), plus the raw-embedding block input and the
      `q_norm`/`k_norm`/RoPE. **Files**: `crates/engine/src/dflash.rs`.
      **Tested by** `crates/engine/tests/dflash_forward.rs`: a deterministic
      hand-built `ctx` and `block` of 16 rows, compared element-wise against
      golden arrays dumped from z-lab (see T4). Start with `fc` +
      `hidden_norm` + one layer + final norm + head.
- [ ] **T4 — golden reference harness.** `crates/engine/examples/dflash_taps.rs`:
      run the drafter's `fc`, `hidden_norm`, each layer's `q/k/v/o/mlp`, the
      final `norm` and the softcapped logits for one fixed prompt, and compare
      against reference dumps produced out-of-tree with the z-lab code
      (PyTorch, CPU, in the scratch dir — never committed). Report
      max-abs-diff and the element index of the worst diff per stage. **Files**:
      `crates/engine/examples/dflash_taps.rs` (new). **Tested by**: the
      example exits non-zero on any stage above tolerance, and the tolerance is
      stated in the header as bit-level or distributional. *Sabotage*: flip
      `+1.0` to `+1.01`, run, confirm the failing stage is the one expected.
- [ ] **T12 — `gemm_chunk` M=16 benchmark (do this before T6).** Benchmark
      `gemm_chunk` at M=16 for every drafter shape and for the 16-row target
      verify pass. Report ms and effective GB/s per shape and for the whole
      block. **Files**: `crates/kernels/examples/dflash_m16.rs` (new).
      **Tested by**: the numbers land in
      `spec/dflash-correctness/implementation_notes.md` with the measured
      total, and the reuse-vs-new-kernel decision is written there with the
      number that drove it (AC5).

## Drafter state

- [ ] **T5 — full 5-layer block forward.** All five layers, z-lab mask
      semantics (R8), bidirectional on the `full_attention` layer, RoPE on
      both context and block K, final norm + tied target head + softcap 30 +
      argmax. Expose `KAPLUMBA_DFLASH_CAUSAL_BLOCK=1` for the all-causal
      variant. **Files**: `crates/engine/src/dflash.rs`. **Tested by**:
      `dflash_forward.rs` takes T4's golden arrays for all 5 layers and the
      15 draft token ids; assert the 15 ids match the reference argmax
      exactly. *Sabotage*: swap layer 4 from bidirectional to causal and
      confirm the test fails at layer 4 rather than earlier.
- [ ] **T6 — drafter KV cache and context append.** `DflashKvCache`: 5 layers,
      8 kv heads × 128, layers 0–3 ring at 2048, layer 4 full. `append_context`
      runs `fc` → `hidden_norm` → per-layer `k_proj`/`v_proj` → RoPE → ring
      write. `rewind_context` truncates. **Files**:
      `crates/engine/src/dflash.rs`. **Tested by**: append 3 chunks of 16
      rows, rewind 5, re-append, and assert the resulting K/V is identical to
      a single append of the same rows; assert the ring never exceeds 2048 and
      never reads a stale row. *Sabotage*: make `rewind_context` a no-op and
      confirm the round-trip test fails.
- [ ] **T7 — target tap export.** `model.rs::forward_chunk` gains an optional
      tap sink; after each layer whose index is in the tap set, blit
      `tokens × hidden` F16 out of `hidden_a`/`hidden_b` into
      `scratch.taps[i]`. Tap set is `[1, 12, 23, 35, 46, 57]`. **Files**:
      `crates/engine/src/model.rs`, `crates/engine/src/exl3.rs` if scale
      handling is needed. **Tested by**: (a) `KAPLUMBA_GPU_PROFILE` at 8k
      prompt with the tap set empty vs `[1, 12, 23, 35, 46, 57]` — the target
      kernel time must be identical, only the blits added (R5); (b) a test
      that taps `[0]` and compares against `hidden_states_all`'s output for
      the same prompt, within tolerance.
- [ ] **T8 — chunked prefill streaming.** Prefill runs chunk by chunk; each
      chunk's taps are fused and appended to the drafter cache, then the tap
      buffers are overwritten. **Files**: `crates/engine/src/model.rs`,
      `crates/engine/src/dflash.rs`. **Tested by**: a test that prefills at
      1000 and 8000 tokens and asserts the tap buffer allocation and the
      drafter cache bytes are the same function of *chunk* size, not prompt
      length; the 8000-token case must not exceed the 1000-token tap footprint
      (R6). *Sabotage*: remove the per-chunk release so taps accumulate, and
      confirm the test fails.
- [ ] **T9 — the round.** `generate.rs`: drafter block pass → 15 draft tokens →
      target verify over 16 positions → greedy acceptance (longest prefix match
      vs argmax) → advance `accepted + 1` → refresh the drafter cache rows for
      the accepted positions from the freshly exported taps (R13, R14).
      **Files**: `crates/engine/src/generate.rs`. **Tested by**: an
      integration test with a tiny stub target/drafter that forces acceptance
      at 0, 5, and all 15 draft tokens, and asserts the drafter cache length
      and the target `next_position()` both equal the accepted count; plus a
      two-round test asserting round 2's draft is unchanged whether or not
      round 1's block K/V were left in the cache (R13).
- [ ] **T10 — greedy equivalence.** Run DFlash-speculative greedy decode vs
      plain greedy decode on >= 5 prompts × >= 200 tokens; allow divergence
      only where the plain run's top-2 logit gap < 0.25. **Files**:
      `crates/engine/tests/dflash_equivalence.rs` (new). **Tested by**: the
      test itself. *Sabotage*: make the drafter emit off-by-one tokens
      (repeat the last draft token) and confirm the test fails with a
      diagnosable "first divergence at position N, draft vs target" message.
- [ ] **T11 — acceptance.** `crates/engine/examples/dflash_acceptance.rs`:
      >= 200 greedy rounds on held-out prompts, reporting median, spread, and
      the per-depth acceptance histogram. **Files**: new example. **Tested
      by**: AC4's >= 2.4, plus the histogram which is what tells us *where* the
      drafter fails if the gate misses.
- [ ] **T13 — peak footprint.** `/usr/bin/time -l` at 1k/2k/8k context with
      the drafter resident, reporting `peak memory footprint` (never
      `peak_resident_pages_bytes`, AGENTS.md §10.3). **Files**: notes only.
      **Tested by**: AC6's budget statement.

## Stage-gate

- [ ] **S1** acceptance >= 2.4 (AC4) **and** greedy equivalence (AC3) **and**
      footprint under budget (AC6) on the same measured runs. If acceptance
      lands below 2.4, do not tune — report Q1/Q2's A/B numbers and the
      histogram, and stop.

## Notes

- No task may be marked done without its test written, run, and — where the
  task adds a kernel or numerical path — sabotaged (AGENTS.md §3.4). Record the
  failure mode in `implementation_notes.md`.
- Commit after every task: `spec/dflash-correctness: T<n> <one line>`.
