# Implementation Tasks: Gemma 4 Metal Engine

- [x] **T1 — Scaffold the Rust/Metal workspace**
  - Files: `Cargo.toml`, `crates/engine/Cargo.toml`, `crates/kernels/Cargo.toml`, `crates/cli/Cargo.toml`, minimal `src/lib.rs`/`main.rs` files.
  - Intent: Recreate the confirmed three-crate workspace without runtime Python or forbidden acceleration dependencies.
  - Test: `cargo metadata`, `cargo check --workspace`, and `cargo test --workspace --no-run`.
  - Commit: `4a263a2`.

- [ ] **T2 — Parse configuration, safetensors headers, and text-only tensor selection**
  - Files: `crates/engine/src/config.rs`, `crates/engine/src/safetensors.rs`, config/header unit tests.
  - Intent: Validate the Gemma 4 graph, locate shards/offsets/dtypes, and exclude all vision tensors.
  - Test: Synthetic headers, missing/overlap/type/shape failures, and the real checkpoint metadata without uploading vision.
  - Commit: pending.

- [ ] **T3 — Load and convert text weights into Metal-owned buffers**
  - Files: `crates/engine/src/weights.rs`, `crates/engine/src/metal.rs`, loader tests.
  - Intent: Mmap source shards, convert BF16 plain tensors to FP16, retain packed EXL3 data, and copy all text weights into shared Metal buffers.
  - Test: Synthetic dtypes/offsets, real text tensor count, vision exclusion, residency accounting, and explicit allocation failures.
  - Commit: pending.

- [ ] **T4 — Implement the CPU EXL3 reference codec and variant fixtures**
  - Files: `crates/engine/src/exl3.rs`, `crates/engine/tests/exl3_format.rs`, fixture data under the test module.
  - Intent: Cover integer K=1–8 plain/mcg/mul1, fractional 1.5/2.5/3.5, packed/unpacked signs, tile permutation, and block Hadamards.
  - Test: Pack/unpack round trips, codebook vectors, transform identities, and every supported variant combination.
  - Commit: pending.

- [ ] **T5 — Implement Metal EXL3 decode and transform kernels**
  - Files: `crates/kernels/src/exl3.metal`, `crates/kernels/src/exl3.rs`, parity tests.
  - Intent: Decode packed trellis tiles and apply the exact tensor-core permutation/Hadamard path on Metal.
  - Test: Metal output against the CPU reference for every variant, including fp16/bf16 scale boundaries.
  - Commit: pending.

- [ ] **T6 — Implement quantized batch-one decode linear kernels**
  - Files: `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`, `crates/engine/src/weights.rs` dispatch wiring.
  - Intent: Provide packed EXL3 GEMV for all model projection shapes without dense reconstruction.
  - Test: Random matrix parity, all current checkpoint shapes, K/codebook matrix, and repeated-dispatch stability.
  - Commit: pending.

- [ ] **T7 — Implement tiled quantized prefill linear kernels**
  - Files: `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`, prefill benchmark tests.
  - Intent: Reuse decoded 16x16 tiles across bounded query chunks and provide prefill GEMM without materializing dense weights.
  - Test: Chunked versus reference outputs, query-count boundaries, all variant fixtures, and profiling counters.
  - Commit: pending.

- [ ] **T8 — Implement embedding, norm, RoPE, activation, residual, and head kernels**
  - Files: `crates/kernels/src/elements.metal`, `crates/kernels/src/elements.rs`, model math tests.
  - Intent: Match Gemma 4 scaled embeddings, weighted/unweighted RMSNorm, local/proportional RoPE, tanh GELU, residuals, layer scalars, and final softcap.
  - Test: Scalar/vector CPU parity, position zero/large offsets, 256/512 head dimensions, and finite-value checks.
  - Commit: pending.

- [ ] **T9 — Allocate and manage full and sliding KV caches**
  - Files: `crates/engine/src/kv_cache.rs`, cache tests, Metal buffer helpers.
  - Intent: Implement contiguous full caches and 1024-slot sliding rings with absolute-position mapping and no KV quantization.
  - Test: Empty, first-token, exact-window, rollover, multi-chunk, and 8k-capacity cases.
  - Commit: pending.

- [ ] **T10 — Implement online full and sliding attention kernels**
  - Files: `crates/kernels/src/attention.metal`, `crates/kernels/src/attention.rs`, attention tests.
  - Intent: Compute causal/GQA attention with FP32 online softmax and fixed working memory for prefill and decode.
  - Test: CPU parity for short and boundary contexts, GQA mapping, window inclusion, causal masking, and no full score/mask allocation.
  - Commit: pending.

- [ ] **T11 — Assemble one decoder layer and gated MLP execution**
  - Files: `crates/engine/src/model.rs`, layer/MLP integration tests.
  - Intent: Execute attention, residual normalization, gated MLP, post-norm, and layer scalar in the confirmed order.
  - Test: Small random Gemma-shaped layer parity, sliding/full variants, shared-value behavior, and chunk independence.
  - Commit: pending.

- [ ] **T12 — Assemble the complete 60-layer text model and quantized head**
  - Files: `crates/engine/src/model.rs`, full-graph tests, loader integration.
  - Intent: Execute the exact config-driven text graph and 6-bit packed language head without vision or dense materialization.
  - Test: Synthetic multi-layer graph parity, real checkpoint load, finite hidden states, and head/top-token smoke.
  - Commit: pending.

- [ ] **T13 — Implement tokenizer, template, sampler, streaming, and CLI**
  - Files: `crates/engine/src/tokenizer.rs`, `crates/engine/src/sampler.rs`, `crates/engine/src/generate.rs`, `crates/cli/src/args.rs`, `crates/cli/src/main.rs`, `crates/cli/src/report.rs`, CLI tests.
  - Intent: Render the direct one-turn template, reject media, implement seeded sampling/top-k/top-p, stop IDs, clean streaming, and prompt sources.
  - Test: Exact prompt IDs, special-token rejection, seed reproducibility, EOS/stop cases, empty/zero boundaries, stdin/file input, and stream formatting.
  - Commit: pending.

- [ ] **T14 — Add metrics, memory accounting, and deterministic benchmark harness**
  - Files: `crates/engine/src/metrics.rs`, `crates/cli/src/report.rs`, benchmark/tests and recorded prompt fixture.
  - Intent: Report post-load TTFT, prefill/decode throughput, token counts, and peak RSS with the agreed 18 GB accounting.
  - Test: Metric calculations, 1k/2k/8k runs, fixed 2k benchmark prompt, 8 warmups, and 128-token measurement window.
  - Commit: pending.

- [ ] **T15 — Run strict real-checkpoint integration and acceptance gates**
  - Files: integration tests, benchmark command/configuration, implementation notes.
  - Intent: Demonstrate text-only model load, prefill equivalence, streamed generation, full resident memory, and 20 tok/s decode.
  - Test: Actual checkpoint 1k/2k/8k prefill, finite/coherent smoke output, no media leakage, RSS ceiling, and sustained greedy decode.
  - Commit: pending.

- [ ] **T16 — Complete the full variant matrix and final validation**
  - Files: all test modules, `tasks.md`, `implementation_notes.md`, final CLI documentation in existing source help text.
  - Intent: Close every acceptance criterion, rerun lint/typecheck/tests, and record exact commands and measurements.
  - Test: `cargo fmt --check`, `cargo check --workspace`, `cargo clippy --workspace --all-targets`, `cargo test --workspace`, and the final benchmark/memory commands.
  - Commit: pending.
