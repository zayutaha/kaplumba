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

## Current risks

- Metal-owned copies of approximately 14.88 GB of text data plus 8k KV and runtime buffers must remain below the agreed 18 GB RSS ceiling.
- The six-bit vocabulary head must remain packed; dense reconstruction would violate the memory target.
- Variant-generic kernels require independent parity fixtures before performance tuning.
- The prefill attention path must preserve the no-score-matrix/no-context-mask invariant while reaching the throughput target.
