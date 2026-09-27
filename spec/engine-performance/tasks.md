# Concrete Task Checklist: Engine Performance

- [ ] **T-1 — Read the mandatory references before touching any kernel.**
  - Files: `reference/omlx`, `/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/ishizuki`, `/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/exllamav3`
  - Action: Read oMLX's `omlx_sdpa_decode` / `omlx_sdpa_prefill` (32-simdgroup attention + threadgroup merge), oMLX `qwen35_packed_linear.py` `project()` (weight concat), ishizuki's matvec and attention, and exllamav3's `exl3_gemm_inner.cuh` (`cp.async` staging). Record every technique we have not yet tried, with `path:line`.
  - Verification: A new "Techniques we have not tried" section exists in `implementation_notes.md`, each entry citing `path:line` and a guess at whether it is bit-exact.

- [x] **T0 — Fix `attention_metal` test harness encoder lifecycle.** ✅ done
  - Files: **`crates/kernels/src/attention.rs`** (the plan named the test file; the
    test contains no Metal encoder at all — it calls `AttentionKernels::online_attention`).
  - Root cause (two faults, not one):
    1. `online_attention` resolved its pipeline via
       `encoder.set_compute_pipeline_state(match head_dim { 256, 512, _ => return Err })`
       **after** opening the encoder, so any other `head_dim` returned with a live
       encoder open → `Command encoder released without endEncoding`.
    2. The generic `online_attention` kernel was already compiled and loaded but
       **never dispatched** — the 256/512 specialisation dropped the documented
       fallback for any other `head_dim` up to `MAX_HEAD_DIM`.
  - Fix: resolve the pipeline *before* creating the encoder, and fall back to
    `online_attention_pipeline` instead of erroring.
  - Verification: `cargo test --release -p kaplumba-engine --test attention_metal`
    → **4 passed; 0 failed** (was failing on `head_dim` 16 and 32).

- [x] **T0b — Give the tiled prefill GEMM a fractional rate path.** ✅ done
  - Files: `crates/kernels/src/linear.metal`
  - Fault: `linear_gemv_tiled` decoded every rate with `exl3_code_at_tg` (integer
    ring arithmetic) and computed `word_count = 256*rate/32` unconditionally. For
    a fractional rate (`encoded == 16 + bits`) that is `128 + 8*bits` words
    instead of `8*bits + 4`, so both the tile stride and the ring wrap were
    wrong. Prefill silently produced wrong numbers for fractional rates.
  - Fix: added `fractional_window_tg` (threadgroup twin of the global
    `fractional_window` the non-tiled `linear_gemv` already used) and branched on
    `params.rate >= 16`. `index = lr*16 + lc` is the plain row-major identity
    `decode_tile` uses, so `perm_mode = 0` is correct. No new buffers.
  - Verification: `cargo test --release -p kaplumba-engine --test linear_metal`
    → **6 passed; 0 failed** (was 4 passed, 2 failed).
  - Note: the deployed checkpoint is integer-only (`quantization_config.json`:
    `bits: 3.0`), so this changes no measured throughput — it removes a silent
    wrong-numbers path and a red suite.

- [x] **T0c — Make the prefill scratch independent of prompt length.** ✅ done
  - Files: `crates/engine/src/model.rs`, `crates/engine/tests/real_forward_buffer.rs`
  - Fault: `forward_buffer_into` ran the whole prompt through every layer in one
    pass, and `ensure_scratch(tokens)` sized `ForwardScratch` by the prompt. At
    8192 tokens that is ~2.5 GiB of activation scratch — the context-length
    dependence AGENTS.md §8 forbids.
  - Fix: added `pub const PREFILL_CHUNK` and split `forward_buffer_into` into a
    chunk loop over a new private `forward_chunk`. Chunking is semantically
    inert: positions are absolute, `GpuKvCache::advance` runs during each chunk's
    encode so `next_position` is right for the next chunk, and the LM head
    already runs on the last row only.
  - Chunk size: **64**, chosen from measurement, not estimation. At 256 rows the
    measured scratch was 160,707,332 bytes (over the 100 MiB cap); at 64 it is
    **91,401,476 bytes**. `attn_scores` (67.11 MB) dominates and does not shrink
    with the chunk — see T0d.
  - Verification: new ignored test
    `real_model_prefill_scratch_is_independent_of_prompt_length` prefills 300
    tokens (>1 chunk) and asserts `scratch_max_tokens() == PREFILL_CHUNK` and
    `scratch_resident_bytes() <= 100 MiB`. Passes in ~35 s:
    `weights=11001026276 kv=881029120 scratch=91401476`.

- [ ] **T0d — Right-size `attn_scores` (frees 67 MB, allows a 256-row chunk).**
  - Files: `crates/engine/src/model.rs`
  - Fault: `attn_scores` is allocated `f32_elements(q_max, 1024)` =
    `q_heads * head_dim * 1024` = 16,777,216 floats (67.11 MB). It is used only
    on the `tokens == 1` decode path, where `AttentionEncodeRequest.method`
    requires `q_heads * key_count` floats = 32 * 8192 = 262,144 floats (1.05 MB).
    Over-allocated by roughly one `head_dim` (512x).
  - Action: thread the KV capacity into `ForwardScratch::new` and allocate
    `max over layers of (q_heads * key_capacity)`. **Do not do this blind:**
    getting the capacity wrong under-allocates and corrupts memory. Requires an
    assertion that `q_heads * key_count` never exceeds the allocation.
  - Payoff: total scratch drops to ~24 MB at a 256-row chunk, so `PREFILL_CHUNK`
    can return to 256 (or higher) and recover the per-chunk dispatch overhead.
  - Verification: `real_model_prefill_scratch_is_independent_of_prompt_length`
    passes with `PREFILL_CHUNK = 256` and `scratch_resident_bytes()` well under
    100 MiB; decode attention output unchanged.

- [ ] **T1 — Prefill GEMM weight decode ablation & lookup table.**
  - Files: `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`
  - Action: Measure exact time spent in `exl3_code_at_tg` unpacking logic. Implement a 256-entry table for MCG decode state to eliminate integer div/mod ALU instructions.
  - Verification: Benchmark on 600-token prompt with `cargo build --release -p kaplumba-cli`; confirm bit-exact output on five reference prompts.

- [ ] **T2 — Prefill attention parallelization.**
  - Files: `crates/kernels/src/attention.metal`, `crates/kernels/src/attention.rs`
  - Action:
    - *Path A (Bit-Exact):* Implement prefix-max scan across 256 threads to find rescale points; parallelize `exp()` evaluations across all threads while thread 0 performs ordered additions.
    - *Path B (Relaxed-Float):* Port oMLX 32-simdgroup interleaved online softmax with 4 KB threadgroup merge.
  - Verification: Measure attention time reduction (target: <10 ms/token); verify generation output against reference prompts.

- [x] **T3 — Metal tensor-core simdgroup layout probe fix.** ✅ done
  - Files: `crates/engine/examples/metal_probe.rs`, `crates/engine/examples/mma_peak.rs` (new)
  - Fault: `simdgroup_load(ta, a, 0)` — the third argument is the **leading
    dimension (elements per row)**, not a stride or count. `0` made every row
    load from the same address, hence 8/64 matched.
  - Fix: `LD = 8`. Probe now reports **64/64 correct -> PASS**.
  - Also measured the real tensor peak, replacing the spec's unverified
    "30-50+ TFLOPS" guess: **15.64 TFLOPS** with 16 independent accumulator
    chains (1 chain = 5.75, 8 = 7.65, and a 32-element matrix array spills to
    local memory and collapses to 0.74).
  - Verification:
    `cargo run --release -p kaplumba-engine --example metal_probe` (64/64 PASS),
    `cargo run --release -p kaplumba-engine --example mma_peak` (15.64 TFLOPS).
  - Consequence: the tensor GEMM-only ceiling is 272 tok/s versus 152.3 for
    SIMT fp32 — only ~1.8x, not 4-6x. 150 tok/s is still not reachable (needs
    >86% GEMM efficiency plus a ~3 ms attention budget). See notes §7.3.

- [ ] **T4 — Resume decode path optimizations.**
  - Files: `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`
  - Action: Implement decode codebook LUT and cooperative 128-bit ring loads in `linear_gemv_exl3_k*`. Sweep grid-stride parameters to match 20 GPU cores.
  - Verification: Decode throughput advances from 7.87 tok/s toward 12–16 tok/s; full test suite remains green.

- [ ] **T5 — Full context scaling & memory audit (1k / 2k / 8k).**
  - Files: `crates/engine/src/model.rs`, `crates/engine/src/generate.rs`
  - Action: Measure prefill throughput and peak RSS at 1024, 2048, and 8192 tokens.
  - Verification: Verify prefill throughput, confirm KV memory footprint matches 1.41 GiB at 8192, and record `/usr/bin/time -l` physical footprint.
