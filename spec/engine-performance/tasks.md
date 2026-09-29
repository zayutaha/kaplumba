# Concrete Task Checklist: Engine Performance

Status as of 2026-09-28, commit `1dd4da9`. **Prefill 9.67 → 68.8 tok/s.**
Items that reality invalidated are marked `~~struck~~` with a pointer to the
section of `implementation_notes.md` that explains why.

- [x] **T0 — Fix `attention_metal` test harness encoder lifecycle.** ✅ `c8360b5`
  - Files: `crates/kernels/src/attention.rs` (not the test file — it contains no
    Metal encoder; it calls `AttentionKernels::online_attention`).
  - Root cause (two faults): (1) the pipeline was resolved *after* the encoder
    opened, so any other `head_dim` returned with a live encoder;
    (2) the generic `online_attention` kernel was compiled but **never
    dispatched**.
  - Verification: `--test attention_metal` → **4 passed; 0 failed**.
  - See `spec.md` Q6.

- [x] **T0b — Fractional-rate path for the tiled prefill GEMM.** ✅ `c8360b5`
  - Files: `crates/kernels/src/linear.metal`
  - `word_count` was `256*rate/32` unconditionally, which for a fractional rate
    gave `128 + 8*bits` words instead of `8*bits + 4` — tile stride and ring wrap
    both wrong. Added `fractional_window_tg` and branched on `rate >= 16`.
  - Verification: `--test linear_metal` → **6 passed; 0 failed**.
  - The deployed checkpoint is integer-only (`bits: 3.0`), so this removed a
    silent wrong-numbers path, not a performance problem.

- [x] **T0c — Make prefill scratch independent of prompt length.** ✅ `817d5b9`
  - Files: `crates/engine/src/model.rs`
  - Added `pub const PREFILL_CHUNK` and split `forward_buffer_into` into a chunk
    loop over a private `forward_chunk`. Chunking is semantically inert:
    positions are absolute, `GpuKvCache::advance` runs during each chunk's
    encode, and the LM head already ran on the last row only.
  - Chunk size was 64, forced by the oversized `attn_scores`; it is **256** now
    (`model.rs:52`) after T0d.

- [x] **T0d — Right-size `attn_scores`.** ✅ `c8360b5`, then **deleted entirely** in `1dd4da9`
  - Was `q_heads * head_dim * 1024` floats = 67.11 MB against a real requirement
    of `q_heads * key_count` = 1.05 MB — over-allocated by ~512x.
  - Step 1: threaded the KV capacity into `ForwardScratch::new` and added the
    missing `scores.length() >= q_heads * key_count * 4` assertion. Scratch at a
    256-row chunk 160,707,332 → 93,729,540 B, so `PREFILL_CHUNK` went back to 256.
  - Step 2 (`1dd4da9`): `attn_scores`, `attn_row_max`, `attn_row_sum` and the
    **five kernels** that used them are **deleted**. Decode now runs the same
    online-softmax body as prefill via `PARTIALS`
    (`attention.metal:724`) plus `attention_decode_combine_{256,512}`, over a
    fixed **1,052,672 B** partial buffer. Fixed a hard failure past 1024 keys.
  - Verification: 2414 tokens of context, previously fatal, now prefills at
    75.1 tok/s. See notes §12.5.

- [x] **T1 — Prefill GEMM weight-decode ablation.** ✅ **closed as rejected** (`1dd4da9`)
  - ~~Implement a 256-entry codebook LUT.~~ **Measured worse, and it is not a
    close call.** Our codebook has only **8** distinct values, so a `select` chain
    needs 7 selects and beats `decode_state`'s ~6 ops. oMLX explicitly rejects
    dynamically-indexed tables on Apple
    (`bonsai/csrc/quantized.h:612-613`) because they spill to device memory.
  - The real win was structural, not a table: **decode the eight codes a lane
    owns from one window setup** (`exl3_decode8` form, `linear.metal:2424-2493`),
    ~400 → ~40 instructions, bit-identical codes. This is most of the 2.14x in
    notes §12.4.

- [x] **T2 — Prefill attention parallelisation.** ✅ **closed** (`c056b48`)
  - ~~Path A (bit-exact prefix-max scan)~~ **not worth building**: the serial
    body it was aimed at (`online_attention_tg2_body`,
    `attention.metal:539-655`) is **dead code with no entry point**, and the live
    kernel already reduces chunk max and sum in parallel. A/B was 10.394 vs
    10.373 — noise. See notes §11a.
  - **Path B landed, but as a structural transpose rather than a merge.** All
    query heads sharing a KV head now sit in one threadgroup
    (`attention_gqa_body`, `attention.metal:725`), from llama.cpp
    `fa_common.metal:411-454`. KV re-read factor **512 → 1**; attention
    1324 → 338 ms, **3.9x**.

- [x] **T3 — Metal tensor-core simdgroup layout probe fix.** ✅ `da55d1d`
  - `simdgroup_load(ta, a, 0)` — the third argument is the **leading dimension
    (elements per row)**, not a stride or count. `0` made every row load from the
    same address, hence 8/64 matched. `LD = 8` → **64/64 PASS**.
  - Also measured the real tensor peak: **15.18–15.64 TFLOPS** with 16
    independent accumulator chains. An array of chains
    (`simdgroup_half8x8 acc[32]`) spills and collapses to 0.74 TFLOPS.
  - Consequence: tensor GEMM-only ceiling ~264 tok/s vs 152.3 for SIMT fp32 —
    ~1.7x, not 4-6x.

- [x] **T3.5 — Fix the `bstage` overflow and enable the MMA path.** ✅ `60ae014` — **11.3 → 24.5 tok/s**
  - `bstage[QTILE * OUTS]` was declared 32×32 but written 64×32, so the upper
    half aliased `astage` and every `k >= 32` weight read activations.
  - Also: `key_count` returned `capacity` for sliding layers forever (+3.8% once
    fixed); latent NaN from loading masked values in the PV loop.
  - **Why it shipped:** `gemm_chunk` (`linear.rs:579`) hardcodes `gemv_tiled`, so
    the MMA path had **zero** coverage and the "6/6" claim counted the other
    kernel. See T6.

- [x] **T4 — Resume decode path optimizations.** ❌ **not done, deliberately** — and the original list was wrong
  - ~~Codebook LUT~~ — rejected, T1.
  - ~~Cooperative 128-bit ring loads~~ — superseded: `exl3_decode8` in
    `ishizuki_gemv` already does one window setup per eight codes.
  - ~~Grid-stride sweep~~ — still unswept, still not the bottleneck.
  - **What decode actually needed, and got (`1dd4da9`): the O(L²) score matrix
    deleted** (§T0d). Decode *hard-failed* past 1024 keys. That was worth far
    more than every item on this list.

- [x] **T5 — Full context scaling & memory audit (1k / 2k / 8k).** ⚠️ **partial**
  - Done: scaling fit over 164–3974 tokens, `t = 1567 + 27.93·L + 0.000533·L²`
    ms, exponent **+0.066** (nearly flat — context is not the constraint, GEMM
    efficiency is). Peak RSS flat at 7.81–8.06 GiB, projected 8.20 at 8192.
  - **Not done: the 8192-token run.** Free memory fell below the 16 GiB gate.
  - **Stale:** the fit was taken at 32.2 tok/s and predates `1dd4da9`. Re-measure
    before quoting it.

- [x] **T6 — Make the test suite honest.** ✅ `c056b48`, `1dd4da9`
  - `mma_prefill_gemm_matches_cpu_reference` set `KAPLUMBA_DISABLE_MMA_PREFILL=1`,
    which **disables the path under test**; a sabotaged kernel still passed.
    Now `remove_var` (`linear_metal.rs:900`) and query counts
    1/127/128/129/255/256/257 for the 128-row tail.
  - **Every new coverage claim is now verified by breaking the code and
    watching the test fail** (sabotage → `worst normalised error inf`; neutered
    `Drop` → `Command encoder released without endEncoding`).
  - **Still open (`spec.md` Q11):** `linear_metal.rs:745-753` carries a stale
    doc comment saying the flag is *set* to select the MMA path (backwards) and
    that the gate is `% 32` when `linear.rs:976-987` gates on `% 64`.

---

## Open

- [ ] **T7 — Get past the 76.7 tok/s ceiling of the current tile shape.**
      A staging-stripped diagnostic caps 128q × 64n at 76.7 tok/s, so ~11% is
      staging/dequant and the rest needs a **different tile shape**. 5.31 TFLOPS
      today is 35% of the 15.18 TFLOPS measured tensor peak. This is now the
      whole remaining project. Target metric: pp/s at 8192.
- [ ] **T8 — Run the 8192-token prefill end to end** and re-measure the scaling
      law, the RSS figure, and the `exp(m_s − M)` combine path at 16 splits.
      Blocked on free memory, not on code.
- [ ] **T9 — Decide `lm_head.trellis` residency** (`spec.md` Q12).
      `weights.rs:530-533` says mapping it is "actively harmful" and "Left
      resident"; `weights.rs:552` maps it. Needs an A/B and a decision.
- [ ] **T10 — Drain the autorelease pool** (`notes` §16). 0 B leaked per
      buffer/encoder pair after the handle fix, but the pool is still never
      drained.
- [ ] **T11 — Explain the decode regression**, 7.87 → 5.3–6.9 tok/s at 614.
      Probably the machine-level shift flagged in notes §8.3; unconfirmed.
- [ ] **T12 — Fix the stale test doc comment** at `linear_metal.rs:745-753`
      (see `spec.md` Q11). Trivial, but it will mislead the next reader.

---

## Session 2026-09-29: converged search + memory accounting

- [x] **M1 — Register-pipelined dequant.** `1b5de3c`. 77.8 -> 81.3 tok/s @614.
      Bit-identical, 20/20 suites, 5/5 prompts.
- [x] **M2 — Hoist Q's conversion out of the per-key attention loop.**
      `fc474ff`. 73.7 -> 75.0 tok/s @8192. Bit-exact, no change at 614.
- [x] **M3 — Stop reporting `ru_maxrss` as a footprint.** `ee14f62`. Renamed to
      `peak_resident_pages_bytes`, documented the measured 6.23 vs 11.69 GiB
      pair, fixed the dangling `measure_footprint_with_time` reference, and
      recorded that the `task_info` FFI has been tried twice and rejected.
- [x] **M4 — Fix `scripts/gpuq holder()` naming dead pids.** `ee14f62`. Stale /
      live / missing-lock cases all verified.
- [x] **M5 — Correct the budget 12 -> 14 GiB** in AGENTS.md §10.1/§10.2/§10.3
      and the `gpuq` warning, and add the §10.3 warning against reading
      `peak_resident_pages_bytes` as a footprint.
- [x] **M6 — Establish where the 1.5 GiB context delta goes.** No leak: 11.696
      GiB @622 -> 12.963 GiB @8126, with 1038 and 1630 byte-identical. KV growth
      + sliding-window fill + scratch + a ~0.25-0.5 GiB driver Metal residual.
- [x] **M7 — Rule out the sliding-KV "333 MiB waste".** Not waste: the sliding
      caches ring, so the full 1024 window is required from the start, and
      untouched buffers charge ~0.016 MiB. No code change, deliberately.
- [x] **M8 — Rule out the in-flight window as the residual.** 2 vs 8 at 7814 tok
      is 23 MiB at no speed cost. The window is saturated at 8 and the bulk of
      the residual is committed-then-released buffers, which A14 showed cannot be
      batched without a 0.913x regression.
- [x] **M9 — Record the negative results** in `implementation_notes.md` so the
      closed questions (8 accumulators, KCHUNK, dequant cost, host overhead,
      14.79 TFLOPS, fp8, repacking, page faults, Metal pipelines, Q_TILE) are not
      re-litigated. Includes corrections to the earlier KV arithmetic.
- [ ] **T13 — Fix the nondeterminism.** AGENTS.md §8 requires greedy-identical;
      14 runs give 13 identical and 1 outlier. **Live violation. Not fixed.**
- [ ] **T14 — Build a quality gate that can see 2e-3.** `output_identity.rs` is
      `#[ignore]`d, 160 tokens total, tolerance 2e-2 vs 2.16e-3 actual. Replace
      with a 16-prompt logit fingerprint + 12x256 exact text + 5x determinism.
- [ ] **T15 — Land and verify `decode-matvec-k3`** (`e360769`, on that branch;
      bit-exact, 8.7 -> 9.3 tok/s, unverified against current HEAD).
- [ ] **T16 — Cost the "less math" directions** for 120 tok/s: exploit the 12.51
      bits of real per-weight value content. fp8 is confirmed absent on this
      M4 Pro, so operand-narrowing is closed.
- [ ] **T12 — Fix the stale test doc comment** at `linear_metal.rs:745-753`
      (carried forward; still open).
