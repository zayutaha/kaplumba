# Implementation Notes: Gemma 4 Metal Engine

## 2026-09-25 — Research and scope confirmation

- No previous repository commit history was inspected. The existing Rust source and old spec files were already deleted in the working tree at the start of this task and were not restored.
- Authorized reference checkout: `reference/omlx`, cloned from `https://github.com/jundot/omlx.git` into the ignored reference directory.
- Independent EXL3 format research used the public upstream format source at `https://github.com/turboderp-org/exllamav3`; no forbidden third-party implementation was used.
- Hardware: Apple M4 Pro, 20 GPU cores, 24 GB unified memory, Metal 4. Toolchain observed: Rust 1.95.0-nightly, Cargo 1.95.0-nightly, Xcode Metal compiler available.
- Model path: `/Users/zayaantaha/.kaplumba/models/Gxma/Gxma`.
- Checkpoint files: two safetensors shards totaling 16,028,050,244 indexed bytes; text tensor bytes total 14,876,563,172, including 60 BF16 layer-scalar tensors worth 120 bytes; the quantization metadata covers 14,876,563,052 bytes; vision tensor bytes total 1,151,487,072.
- Text graph: 60 layers; full attention at 5, 11, 17, 23, 29, 35, 41, 47, 53, 59; 50 sliding layers; 10 full layers.
- Text dimensions: hidden 5376, intermediate 21504, vocabulary 262144, query heads 32, sliding KV 16x256, full KV 4x512.
- Current quantization: 410 three-bit mcg linear descriptors and one six-bit mcg `lm_head`; all use EXL3 packed trellises, mcg marker `0xCBAC1FED`, `suh`/`svh` scale/sign vectors, and 16x16 tiles.
- Current `lm_head` trellis shape is 336x16384x96; ordinary three-bit linears use packed width 48.
- Attention semantics: sliding window 1024 total positions, full attention K reused as V, attention scale 1.0, final logit softcap 30.
- Verified tokenizer prompt for `Hello`: `<bos><|turn>user\nHello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`, token IDs `[2, 105, 2364, 107, 9259, 106, 107, 105, 4368, 107, 100, 45518, 107, 101]`.
- Generation defaults read from the checkpoint: temperature 1.0, top-k 64, top-p 0.95, EOS IDs 1/106/50.
- Read-only tokenizer inspection used the existing Python environment once to verify the template; product/runtime implementation remains Rust and Metal only.
- No speculative decoding, draft/verification, n-gram lookup, or n-gram cache is permitted in this implementation.
- Agreed release gates: fixed 2k benchmark prompt, 8 warmup steps, 128-token greedy average, at least 20 decode tok/s, fully resident text weights, and peak RSS below 18,000,000,000 bytes at 8k context.

## Format invariants

- Safetensors data offsets are relative to the payload following the JSON header.
- Quantized tensors are represented as packed `trellis` plus input/output sign-scale vectors; both `su`/`sv` and `suh`/`svh` must be accepted by the generic loader.
- Trellis tiles contain 256 K-bit states. The loader must apply the tensor-core permutation before interpreting tile row/column positions.
- The normalized transform dimension is 128 with scale `1/sqrt(128)`.
- mcg decoding uses multiplication by `0xCBAC1FED`, mask `0x8fff8fff`, and xor mask `0x3b603b60`, followed by the codebook reconstruction defined by the upstream format.
- Fractional execution is limited to 1.5, 2.5, and 3.5 bits with mul1, matching the upstream executable kernel set.

## Planned validation commands

- `cargo fmt --check`
- `cargo check --workspace`
- `cargo clippy --workspace --all-targets`
- `cargo test --workspace`
- Actual model benchmark with the recorded 2k prompt, 8 warmups, and 128 measured greedy tokens.
- 1k, 2k, and 8k chunked-prefill runs with peak RSS measurement.

## T1 — Workspace scaffold

- Recreated `crates/engine`, `crates/kernels`, and `crates/cli` as the three workspace members.
- Added shared `half` and `rayon` workspace dependencies to match the locked dependency graph.
- Added minimal Rust library and CLI entry points; no runtime model code or acceleration path exists yet.
- Verified with `cargo metadata --no-deps --format-version 1`, `cargo check --workspace`, `cargo test --workspace --no-run`, and `cargo fmt --check`.
- T1 commit: `4a263a2` (`spec/gemma4-metal-engine: T1 scaffold workspace`).

## T2 — Configuration and safetensors metadata

- Added strict typed parsing for `config.json`, `generation_config.json`, and `quantization_config.json`.
- Added structural and current-checkpoint graph validation, including the exact 60-layer full/sliding pattern, RoPE parameters, generation defaults, supported quantization rates, and MCG marker validation.
- Added safetensors header parsing with payload-relative offsets, dtype/shape byte checks, overlap detection, index consistency, shard path safety, and total-size validation.
- Added text-only selection that excludes `model.vision_tower.*` and `model.embed_vision.*` without reading or uploading tensor payloads.
- Added exact text tensor inventory/projection geometry validation; full-attention layers correctly omit `v_proj` because K is reused as V.
- Corrected the text-byte accounting: the complete text selection is 14,876,563,172 bytes; the quantization metadata sum is 14,876,563,052 bytes because 60 BF16 layer-scalar tensors add 120 bytes.
- Synthetic tests cover valid headers, missing index entries, overlap, dtype/shape mismatches, path traversal, total-size mismatch, vision exclusion, config boundaries, and quantization failures.
- Real metadata test passed with `KAPLUMBA_TEST_MODEL=/Users/zayaantaha/.kaplumba/models/Gxma/Gxma cargo test -p kaplumba-engine -- --ignored`; it reads headers only and reports 14,876,563,172 text bytes and 1,151,487,072 vision bytes excluded.
- T2 commit: `8e5eca9` (`spec/gemma4-metal-engine: T2 parse checkpoint metadata`).

## T3 — Metal-owned text weights

- Added `MetalContext` with system-device/queue ownership, shared-storage buffer creation, maximum-buffer validation, and explicit allocation-size errors.
- Added a chunked `WeightArena`; the default 512 MiB segments avoid single-buffer limits for the 2.8 GB embedding and 1.06 GB language head, while allowing tensors to span segments.
- Added read-only mmap with RAII `munmap`; source shards are released when loading returns.
- Plain BF16 tensors are converted in 8 MiB chunks to FP16; FP16 and packed EXL3 tensor bytes are copied without transformation.
- The arena enforces the 18,000,000,000-byte resident limit before allocation and reports logical versus allocated capacity separately.
- Synthetic tests cover BF16 conversion, packed-byte retention, F16 spans crossing segments, vision exclusion, unsupported dtypes, allocation-limit failure, and real text-selection compatibility.
- The real selection test passed without allocating model weights; all 14,876,563,172 text bytes classify successfully and vision tensors remain excluded.
- With 512 MiB segments, the current text selection plans for 28 segments and 15,032,385,536 allocated bytes, below the 18 GB gate; this is a capacity calculation, not a completed model-load RSS measurement.
- T3 commit: `391164d` (`spec/gemma4-metal-engine: T3 load Metal-owned weights`).

## Scope amendment — token-to-text verification

- User requires generated token IDs to be converted through the actual tokenizer into proper Unicode text, not represented as packed bits or byte fragments.
- T13 must use incremental detokenization with multi-byte UTF-8 and special-token handling; raw token IDs are diagnostic-only.
- T15 must include manual inspection of real streamed text and record the observed transcript/result here before acceptance.

## T4 — CPU EXL3 reference codec

- Implemented upstream-compatible plain, MCG, and mul1 codebook decoding, including the lop3 MCG transform and FP16 mul1 affine reconstruction.
- Implemented integer trellis window extraction for K=1–8 and the alternating 1.5/2.5/3.5 fractional ring layout for mul1.
- Added exact tensor-core permutation/inverse helpers, packed/unpacked sign handling, and normalized blockwise 128 Hadamard transforms.
- Added matrix reconstruction helpers that apply row/column transforms and sign scales without changing the packed representation.
- Integration fixtures cover all integer K/codebook combinations, all fractional rates, packed and unpacked signs, malformed layouts, transform identities, and known codebook bit patterns.
- Real K=3 MCG tile decoding passed against the supplied checkpoint’s first q-projection trellis tile.
- T4 commit: `5555c0a` (`spec/gemma4-metal-engine: T4 add CPU EXL3 reference`).

## T5 — Metal EXL3 decode and transforms

- Added runtime-compiled raw Metal kernels for integer/fractional trellis windows, all three codebooks, tensor-core permutation, and 128-point Hadamard transforms.
- Added a retained `Exl3Kernels` context with shared input/output buffers, bounded dispatch, command completion checks, and readback helpers.
- Metal parity passes for K=1–8 plain/MCG/mul1, fractional 1.5/2.5/3.5 mul1, and multi-vector Hadamard output.
- Aligned the CPU mul1 reference with the upstream fused FP16 multiply-add after Metal parity exposed the intermediate-rounding difference.
- T5 commit: `3e13ab4` (`spec/gemma4-metal-engine: T5 add Metal EXL3 kernels`).

## T6 — Packed batch-one linear

- Added a three-stage raw Metal GEMV: FP16 input Hadamard/sign transform, packed trellis-tile FP32 dot products, and FP16 output Hadamard/sign finalize.
- Added `LinearKernels` with runtime shader compilation, shared buffers, shape/rate validation, command completion checks, and readback.
- Added `TextWeights::packed_u16` handoff for packed EXL3 tensors without dense reconstruction.
- Parity passes for 128×128 matrices across K=1–8 plain/MCG/mul1 and fractional 1.5/2.5/3.5 mul1.
- The current correctness-first kernel uses one output thread and serial input reduction; T7 now provides tiled multi-query dispatch while retaining the batch-one path for decode.
- T6 commit: `0cbd089` (`spec/gemma4-metal-engine: T6 add packed linear kernels`).

## T7 — Tiled quantized prefill linear

- Added `linear_input_rows`, `linear_gemv_tiled`, and `linear_finalize_rows` to the raw Metal source.
- The tiled kernel uses a 16x16 threadgroup tile, decodes each packed 16x16 weight tile once per bounded 16-query chunk, and reuses it across those query rows without dense materialization.
- Added `LinearKernels::gemm_chunk` and `gemm_chunk_with_stats`; the latter reports query chunks, decoded tiles, weight applications, and dispatch groups.
- Added row-transform/finalization support for multiple 128-value blocks and non-square projection shapes.
- Parity passes for query counts 1, 15, 16, 17, and 32; 128x128 and 256x384 matrices; integer K=1–8 plain/MCG/mul1; and fractional 1.5/2.5/3.5 mul1.
- Added invalid-shape and sign-length rejection tests. Workspace check, test, clippy, and format checks pass.
- T7 implementation is verified in the working tree; no commit was created because the current session was not asked to commit.

## T8 — Element, norm, and rotary kernels

- Added `crates/kernels/src/elements.metal` and `elements.rs` for scaled token embedding lookup, weighted RMSNorm, unweighted V RMSNorm, residual addition, layer scaling, tanh GELU, gated GELU, final logit softcap, and RoPE.
- Weighted norms use the checkpoint’s direct learned multiplier; unweighted V norms omit the multiplier. Reductions are FP32 and outputs are FP16.
- RoPE supports the nontraditional split-half pairing, default frequencies, proportional partial rotation, absolute offsets, and 256/512 head dimensions.
- Added CPU parity fixtures for embedding, both norm modes, residual/layer scalar paths, GELU variants, softcap, default/proportional RoPE, invalid inputs, and token batches longer than the synthetic vocabulary.
- Workspace check, test, clippy, and format checks pass.

## T9 — Full and sliding KV cache management

- Added `FullKvCache`, `SlidingKvCache`, and the `KvCache` enum in `crates/engine/src/kv_cache.rs`.
- Full caches use contiguous absolute positions and distinct transformed K/V storage; the full-attention path shares only the raw K projection before K-norm/RoPE and V-norm. Sliding caches use a fixed-capacity ring with absolute-position-to-slot mapping.
- Sliding visibility is inclusive: after appending position `p`, valid positions are `max(0, p - capacity + 1)..=p`.
- Added strict contiguous-position, shape, capacity, empty-chunk, and chunk-length validation plus resident-byte accounting.
- Tests cover first/chunked appends, full overflow, exact 1024-window rollover, multi-chunk 8k positions, K=V aliasing, enum dispatch, and invalid inputs.

## T10 — Online full and sliding attention

- Added `crates/kernels/src/attention.metal` and `attention.rs` with a fixed local FP32 accumulator per query/head and no score or mask buffer.
- The kernel applies causal position checks, optional inclusive sliding-window filtering, GQA head mapping, configurable scale, and separate transformed K/V inputs for the full-attention raw-projection reuse path.
- One dispatch handles both decode and multi-query prefill; local accumulator storage is bounded by the 512-element maximum head dimension.
- CPU parity covers decode, multi-query prefill, GQA, K=V, 256/512 head dimensions, causal filtering, sliding boundaries, large positions, invalid shapes, and working-set accounting.
- T10 implementation is verified in the working tree; no commit was created because the current session was not asked to commit.

## T11 — Decoder layer and gated MLP

- Added `crates/engine/src/model.rs` with validated layer configuration, packed quantized projection descriptors, and a decoder-layer forward path.
- The layer order is input norm → Q/K/V projections → Q/K/V norms → RoPE → cache append → online attention → output projection → post-attention norm/residual → pre-FFN norm → gate/up projections → gated GELU → down projection → post-FFN norm/residual → layer scalar.
- Full layers reuse the raw K projection for V before separate K/V normalization and cache storage; sliding layers require an explicit V projection.
- Decode uses batch-one GEMV and multi-token chunks use the tiled prefill GEMM; layer tests exercise chunk boundaries and cache continuity.
- CPU parity tests cover full and sliding synthetic layers, quantized projections, GQA attention, RoPE, gated MLP, residuals, layer scaling, invalid weights, and position errors.
- T11 implementation is verified in the working tree; no commit was created because the current session was not asked to commit.

## T12 — Text model assembly (core path)

- Added `TextModel` in `crates/engine/src/model.rs` for scaled embedding lookup, sequential decoder layers, final RMSNorm, packed language-head projection, and final logit softcap.
- Added validation for the confirmed 60-layer graph, including full/sliding pattern, 5376 hidden size, 21504 MLP size, 32 query heads, 16/4 KV heads, 256/512 head dimensions, and 6-bit MCG head.
- Synthetic two-layer CPU parity covers chunked prefill/decode continuity, final head projection, softcap, and cache advancement.
- Added dedicated per-tensor Metal allocation for the real path; the full 14,876,563,172-byte text selection loads into single addressable Metal buffers under the 18 GB accounting limit.
- Added buffer-backed GEMV/GEMM and embedding entry points, plus `TextModel::load_from_dir` for the confirmed checkpoint. The real load test passed in about 75 seconds and constructed all 60 layers, caches, and the packed 6-bit head.
- A real greedy one-token generation smoke test passed with proper Unicode output: prompt `Hello` produced observed text `" Sond"`. This is recorded as a smoke observation, not yet a coherence/throughput acceptance result.
- A real 8-token CLI run produced the streamed text `" Sond’ de de de own de own"`; diagnostic IDs were `168884 236858 569 569 569 1852 569 1852`. Text is valid Unicode and contains no raw IDs, but it is not yet coherent enough for acceptance.
- The 8-token run reported `load_ms=73377.069`, `ttft_ms=28222.022`, and `peak_rss_bytes=15593111552`.
- A later one-token CLI verification reported `load_ms=73996.788`, `ttft_ms=13441.245`, `prefill_tok_s=1.042`, and `peak_rss_bytes=13880508416`; the resident model remains below the 18 GB ceiling. The current implementation is far below the 20 tok/s gate because each layer operation performs host readback/allocation. Performance work remains required.
- T12 core implementation and real residency load are verified in the working tree; no commit was created because the current session was not asked to commit.

## T13 — Tokenizer, sampling, generation, and CLI

- Added real `tokenizers`-backed prompt rendering with media-token rejection, exact Gemma template IDs, incremental UTF-8-safe detokenization, special-token suppression, and diagnostic-only raw IDs.
- Added seeded FP32 top-k/top-p sampling with temperature-zero greedy mode and deterministic tie handling.
- Added generation orchestration with EOS/context bounds and streamed text callbacks; synthetic generation tests pass and the real smoke test emitted `" Sond"`.
- Added CLI argument parsing, prompt/file/stdin handling, seed/sampling/metrics flags, report helpers, and real model execution through `TextModel::load_from_dir`; the end-to-end CLI smoke test streamed proper Unicode text.

## T14 — Metrics core

- Added `crates/engine/src/metrics.rs` with prefill/decode rates, TTFT/load duration fields, peak RSS measurement, fixed prompt construction, and the fixed 2k/8-warmup/128-token benchmark plan validation.
- Added `run_benchmark` and CLI `--benchmark`; metric and benchmark boundary tests pass. The real 1k/2k/8k benchmark runs remain to be executed.

## T15 — Real-checkpoint validation (partial)

- Ran `KAPLUMBA_TEST_MODEL=/Users/zayaantaha/.kaplumba/models/Gxma/Gxma cargo test --workspace -- --ignored --nocapture`.
- Real metadata inspection, text-selection validation, real MCG tile decode, full dedicated-buffer model load, tokenizer tests, and one-token Unicode generation all passed.
- The full load test completed in 75.89 seconds; the real generation test completed in 89.20 seconds and emitted `" Sond"`.
- The 8-token CLI observation remains `" Sond’ de de de own de own"`; it is valid Unicode but not coherent enough for final acceptance.
- 1k/2k/8k prefill, sustained 20 tok/s decode, and the final manual coherence review remain open.

## T14 — Decode throughput root-cause analysis (measured, 2026-09-25)

The 20 tok/s decode gate is missed by ~20x. Root cause is architectural, not kernel quality.

**Op count and sync model.** `DecoderLayer::forward` (`crates/engine/src/model.rs:329`) runs ~20 ops per layer; with 60 layers that is ~1200 ops per generated token. Every op today follows the pattern: allocate a `CommandBuffer`, encode one dispatch, `commit()`, `wait_until_completed()`, then copy the result back to the host into a `Vec<u16>`. The next op uploads that `Vec<u16>` back to a fresh Metal buffer as its input. The `gemv_buffer`/`gemm_chunk_buffer` weight path is already GPU-resident, but activations still round-trip GPU→CPU→GPU between every op.

**Why this dominates.** The weight-streaming floor is ~14.9 GB of text tensors per token at roughly 200 GB/s, i.e. ~75-100 ms/token (7-13 tok/s). Observed is ~1000 ms/token. The gap is dominated by ~1200 `commit()` + `wait_until_completed()` barriers (each a full CPU/GPU sync, on the order of hundreds of microseconds to ~1 ms on unified memory) plus the per-op buffer allocation and activation copy. Overhead is ~90% of wall time; the actual math is a small fraction.

**Conclusion.** The fix is a GPU-resident activation pipeline, not faster kernels. Concretely: keep activations in Metal buffers, batch the whole forward into one (or few) command buffers, and read back only the final logits for sampling.

## T14 — Load-time optimization (done): mmap -> positional reads

- Root cause of the 74s load (and 11.7s in release): bulk tensor bytes were read through an mmap. Reading a multi-GB safetensors shard via mmap costs one minor fault per page; with many threads faulting unrelated file offsets at once the kernel's readahead collapses (~1.1-1.3 GB/s).
- Measured drive sequential rate (`dd bs=8m`): 7.5 GB/s and 5.4 GB/s for the two shards (~2.5s for 16GB). So the disk was never the limit.
- Fix: `ShardReader` uses `read_at` (pread) straight into each tensor's destination Metal buffers, chunks in parallel via rayon, no staging copy for raw tensors; BF16 tensors stage one chunk at a time and convert in place. Tensors are read in ascending file order to keep readahead sequential.
- Result: weight read+convert = 2.5s (was 11.7s release / 74s debug). `cargo test --release` real load test wall time 3.10s. CLI `load_ms=4253` (includes building 60 layers on top of the 2.5s read). Peak RSS dropped 15.59GB -> 7.68GB (no duplicate 16GB file mapping).

## T14 — GPU-resident execution plan (sub-tasks)

1. **T14a — Shared command-buffer encoder.** Introduce a `MetalEncoder` that owns a single command buffer; each kernel host function encodes into it without `commit()`/`wait`. Caller commits and waits once per forward. Parity must be exact vs the existing per-op path.
2. **T14b — Buffer variants of element ops.** Add `*_buffer` (Buffer in, Buffer out) for `rms_norm`, `rms_norm_unweighted`, `residual_add`, `layer_scale`, `gated_gelu`, `rope`, `softcap`. Norm weights become small persistent device buffers, not per-call uploads.
3. **T14c — GPU-resident KV cache append + attention.** Cache storage becomes Metal buffers; append writes K/V on device; `online_attention` reads the cache buffer directly and writes its output buffer. Preserve the no-score-matrix / no-context-mask invariant.
4. **T14d — Persistent activation scratch.** Preallocate fixed-size ping-pong buffers for hidden/q/k/v/mlp activations sized to the max chunk, reused across all 60 layers (no per-op allocation). Sizes independent of context length per the project constraint.
5. **T14e — `forward_buffer` end-to-end.** `DecoderLayer::forward_buffer` and `TextModel::forward_buffer` execute entirely on GPU, committing once, reading back only logits. Decode (`tokens == 1`) targeted first; prefill reuses the same path per chunk.
6. **T14f — Benchmark + acceptance.** Re-measure 1k/2k/8k prefill, sustained decode, and peak RSS. Gate: >= 20 tok/s decode, RSS < 18 GB, and coherent manual text.

Each sub-task keeps the existing host-slice path for tests and adds buffer paths alongside; no behavior change to the already-verified kernels.

## T14 — GPU-resident forward path (done) and measured decode ceiling

### What was built

- `crates/kernels/src/encoder.rs`: `MetalEncoder` batches many dispatches into one command buffer.
- Encode-only buffer variants: `ElementKernels` (`encode_rms_norm`, `encode_rms_norm_unweighted`, `encode_residual_add`, `encode_layer_scale`, `encode_gated_gelu`, `encode_rope`, `encode_softcap`, `encode_embedding_scaled`, `encode_cache_append`, `encode_copy`, `encode_copy_last_row`), `LinearKernels` (`encode_gemv`, `encode_gemm`), `AttentionKernels` (`encode_online_attention`).
- `crates/engine/src/gpu_cache.rs`: `GpuKvCache` keeps K/V/positions in Metal buffers (append + attend fully on device).
- `DecoderLayer::forward_buffer` / `TextModel::forward_buffer`: all 60 layers, the vocabulary head, and the logit softcap encoded into ONE command buffer; only the final softcapped logits are read back.
- `ForwardScratch`: fixed-size ping-pong activation buffers, sized by chunk length only, never by context length.
- `generate_gpu` + CLI default path; `--legacy-path` keeps the host-slice path.
- Parity: `crates/engine/tests/forward_buffer_parity.rs` (synthetic, all pass) and `real_forward_buffer.rs` (real checkpoint; argmax now matches the slice path).

### Two real bugs this work uncovered

1. `copy_last_row` copied **row 0**, not the last row, so the vocabulary head read the wrong position. The synthetic test missed it because its random weights make every row similar. Fixed by packing `(rows-1, width)` into the params word.
2. The hidden-state ping-pong picked `hidden_b` for an even layer count, but layer k writes `hidden_b` when k is even and `hidden_a` when k is odd, so a 60-layer model must finish in `hidden_a`. The 2-layer synthetic model had the same defect and also passed only by luck. Both fixed; real argmax now agrees between paths.

Also: the host KV cache is now created lazily so the GPU path does not hold a second 1.5 GB copy (that duplication pushed the real load past the Metal allocation ceiling and segfaulted at the first full-attention layer). A `write_bytes` count in u32 units vs bytes wrote 4x past the positions buffer; that was the actual segfault.

### Throughput measured (release, 14.88 GB text selection)

| Change | prefill tok/s | decode tok/s | TTFT |
|---|---|---|---|
| host-slice path, per-op commit | 2.08 | ~0.6 | 6.7 s |
| + single-command-buffer GPU path | 1.70 | 0.63 | 8.2 s |
| + bit-rate-specialised decode kernels | 1.70 | 1.21 | 8.2 s |
| + vocabulary head only on the sampled row, both ping-pong fixes | **4.73** | **1.31** | **2.96 s** |

- The prefill win is structural: the 262,144-wide head is 1.06 GB of weights, so running it for 14 prompt rows instead of 1 was 14x wasted work. Only the sampled position needs logits.
- The decode win came from `linear_gemv` doing **two runtime integer divisions (`% word_count`) per weight**. Decode touches all ~39.7 G weights per token, so those divisions dominated. Eight concrete kernels (`linear_gemv_fast1..8`) bake the divisor in as a literal; the window arithmetic is unchanged, so results stay bit-identical (verified against the CPU EXL3 reference for bits 1-8). 1.9x.

### Why 18 tok/s decode is out of reach for this format

Decode must stream 14.88 GB per token, so the memory-bandwidth ceiling is 273/14.88 = **~18.3 tok/s**, and that assumes a decoder whose cost is bandwidth-bound. It is not:

- Each weight costs 2 `read_word` loads (8 bytes) to extract a 3-bit state, i.e. ~10.7x read amplification, plus ~12 ALU ops in `decode_state`.
- 39.7 G weights/token x ~12 ops = ~480 G ops/token. M4 Pro fp32 ALU peak is ~6.6 T ops/s, so the ALU floor alone is ~73 ms/token, i.e. **~13.7 tok/s absolute ceiling even at perfect efficiency**, and ~5-8 tok/s realistically.
- A bandwidth-bound 18 tok/s needs the decode to run at under ~1 op/byte, which requires a fundamentally different kernel: cooperative threadgroup staging of the trellis in shared memory plus a table/vectorised `decode_state` (one lookup instead of ~12 ops). That is a substantial separate piece of kernel engineering, not a tuning pass.

Prefill is the metric with real headroom, because weight reads amortise across the chunk.

### Quality still open (blocking)

Output remains valid Unicode but incoherent (`"de de de ..."`). The slice and buffer paths now agree, so the defect is shared, not a buffer-path bug. The authorised reference could not be used to bisect it: the installed oMLX refuses this checkpoint with `ValueError: Received 1644 parameters not in model` (VLM adapter parameter-name mismatch), and `mlx_lm`/`mlx_vlm` have no `gemma4` module. Remaining suspects: EXL3 weight decode on real data, the shared graph (norm placement / RoPE / attention), or the prompt template.

## T14 — Zero-copy weight residency (done)

- The vision tower was already excluded from the text selection (356 tensors,
  1.151 GB, verified via `inspect_model_dir`: `excluded_vision` names all start
  with `model.vision_tower.*` / `model.embed_vision.*`). The 15.6 GB figure was
  text weights (14.88 GB) plus GPU KV caches (1.51 GB).
- Root change: EXL3 trellis bytes are no longer copied into anonymous memory.
  `ShardReader` now keeps a read-only `mmap` of each shard for the lifetime of
  the weights (the Metal buffers alias those pages, so dropping the mapping would
  leave the GPU reading freed memory), and `.trellis` tensors are bound
  zero-copy via `newBufferWithBytesNoCopy` over a page-aligned window.
- Safetensors tensor offsets are not Metal-aligned, so binding needs byte
  offsets. Added `TensorRef { buffer, offset }` plus offset-aware
  `dispatch_1d_offset` / `dispatch_2d`, and `QuantizedLinear::metal_offset`.
  `packed_len` is now recorded explicitly because a zero-copy buffer is a
  page-aligned window whose length is larger than the tensor.
- Only the 13.1 GB of trellis is zero-copy. The 2.83 GB of BF16 tensors still
  convert into the arena (they must change representation); a GPU conversion
  pass would recover that too.
- Result: load 3951 ms -> ~930 ms, peak RSS 8.34 GB -> 3.33 GB, decode and
  prefill unchanged (1.28 / 1.9 tok/s). 73 tests pass; the packed-bytes test now
  asserts the zero-copy view instead of arena residency.

## Quality investigation (open, with one unverified change)

Verified against the authorised references, all CORRECT:
- Graph, `layer_scalar` placement, and the decoder body match
  `mlx_lm/models/gemma4_text.py` exactly.
- `attention_k_eq_v: true`, `num_kv_shared_layers: 0`, `num_experts: null`, so
  V=K on full layers and dense MLP, as implemented. `layer_types` confirms full
  attention at 5,11,...,59, matching the `index % 6 == 5` heuristic.
- RoPE is verified numerically against `ProportionalRoPE`: reference freqs are
  denominators, so `angle = position / base^(2p/dims)` (mine) is right, the
  split-half layout rotates dims [0..63] and [256..319] (mine), and theta is
  10000 for sliding / 1e6 for full (mine).
- Quantisation rates: every layer projection is 3 bpw, `lm_head` 6 bpw; the
  hardcoded `Integer(3)` is correct. All 411 `.mcg` values are `0xCBAC1FED`,
  matching the cb1 branch. `decode_state` matches `decode_3inst` exactly for all
  three codebooks, and `integer_window` matches `dq` bit for bit.
- The layer-59 hidden-norm collapse (1.87 -> 0.116) is genuine: its
  `layer_scalar` really is 0.0366, and the final RMSNorm undoes the scale.

### The EXL3 trellis decode is the last unverified component

Diagnostic added: `real_model_top_tokens` prints the top-10 next-token
candidates. Current result for the real prompt:

    logit max=26.609  top1_prob_approx=0.0970
    0 id=244376 logit=26.609 p=0.0970     (rare, high-id tokens)
    top2000 entropy=5.545   (uniform would be ~7.6)

A correct instruct model answering "Hello" would be far more confident and the top
ids would be ordinary English continuations. The near-uniform distribution proves
the vocabulary head is reading scrambled weights. Since every other component is
verified against the references, the EXL3 trellis decode is the sole remaining
defect, and it is unverified because my GPU kernels and the CPU reference share
the same formulas, so parity tests cannot catch it.

What is confirmed about the format:
- `SWAP16(x) = __byte_perm(x, 0, 0x1032)` swaps the two 16-bit halves, and the
  reference `dq` reads `ptr[i]` as uint32 from the same little-endian u16 array
  that `read_word` uses, so the swap is already handled consistently.
- `decode_3inst`/`decode_3inst_2` match my `decode_state` for all codebooks.
- BUT the reference `dq` for K=3 returns a full 16-bit window, and the gemv
  kernel does not even use `dq` for K=3: it precomputes
  `b1 = (t_offset + 257) * bits; b2 = b1 + 21` for a specialised 8-state group
  extraction. So my per-weight `integer_window` may be extracting the wrong bits
  for K=3, independent of any tile permutation. The "permutation" may be a red
  herring masking a bit-extraction error.

BLOCKER: a dense ground-truth 16x16 tile is required, and no accessible
implementation can produce one here:
- exllamav3 `reconstruct` is a CUDA kernel and its Python import hard-requires
  torch built for CUDA; unavailable on this Apple Silicon host.
- oMLX stores models in its own `paroquant` format and cannot load EXL3 at all
  (it fails with "Received 1644 parameters not in model" because every layer
  tensor is unmatched, not because of a naming quirk).
- `mlx_lm` / `mlx_vlm` in the app bundle have no `gemma4` module.

### Root cause localised: the trellis decode yields white noise

Added a transform-independent oracle (`effective_rank` = participation ratio
`(tr(A))^2 / tr(A^2)` of a decoded 128x128 block of `q_proj`). Because the
Hadamard is orthogonal it cannot change this number, so it isolates the decode:

    raw / after inverse Hadamard   effective_rank = 62.1     (white noise for 128x128 is 64)

A trained weight block has an effective rank far below 64, so the decoded values
carry no structure at all. This is consistent with the model's behaviour: top-1
next-token probability 9.7%, entropy 5.5, and every predictability probe failing
("The capital of France is" -> "S", "1 2 3 4 5 6 7 8 9" -> "L", "8 + 8 =" -> junk).

Ruled out by measurement or by the references:
- Tile permutation: swept 10 candidate modes end to end (`real_model_permutation_sweep`); none produced "Paris" or "10", so the permutation is NOT the defect. `permutation_index` is now a runtime selector (`set_permutation_mode`) so this stays testable.
- Hadamard: matches `had_hf_r_128_inner` / `had_ff_r_128_inner` (H_128 scaled by 1/sqrt(128), 0.088388347648), and the GEMM passes that exact constant.
- Scale placement: GEMM uses `had_hf_r_128_inner<true,false>` for the input (suh pre-scale) and `<false,true>` for the output (svh post-scale), which matches `linear_input_transform` and `linear_finalize`.
- `suh` rms ~1/sqrt(in_features), `svh` rms ~1.0, so magnitudes are healthy.
- Window extraction: `unpack_integer` is byte-for-byte the reference `dq` (same b0 = t*bits + bits - 16 + 256*bits, i0/i1, funnel shift, &0xffff), and `read_word` matches the reference uint32 view including `SWAP16`.
- Codebooks: `decode_state` matches `decode_3inst` for cb0/cb1/cb2; all 411 `.mcg` multipliers are 0xCBAC1FED.
- Also tested and rejected: the reference gemv's paired `decode_mcg_product_2` decode (two windows `bits` apart). Effective rank stayed 63.4, so the pair variant is not the fix either.

Remaining unknown: the exact EXL3 v2 trellis bit layout. Reading
`pack_trellis_kernel` (exllamav3 `quant/pack.cu`) literally implies every 16-bit
flush discards the top bits of the last state, which cannot be correct, so that
source is being misread and the true layout cannot be derived from it reliably
without executing it. It is a CUDA kernel in a torch-CUDA project, so it cannot
run on this Apple Silicon host, and oMLX stores models in its own `paroquant`
format so it cannot serve as a cross-check either.

Everything outside the EXL3 decode is verified: graph, RoPE (checked numerically
against `ProportionalRoPE`), GQA, layer types, layer scalars, rates, tensor set
(no missing or extra tensors), tokenizer, sampler. 73 tests pass.

`permutation_index` identity is NOT a fix (top-1 is 9.7% either way). It is kept
only because it is the layout the reference packer implies, and because the
permutation is now a runtime selector rather than a baked-in assumption.

## Current risks

- Metal-owned copies of approximately 14.88 GB of text data plus 8k KV and runtime buffers must remain below the agreed 18 GB RSS ceiling.
- The six-bit vocabulary head must remain packed; dense reconstruction would violate the memory target.
- Variant-generic kernels require independent parity fixtures before performance tuning.
- The prefill attention path must preserve the no-score-matrix/no-context-mask invariant while reaching the throughput target.
