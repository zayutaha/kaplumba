# Gemma 4 Metal Engine

## Goal

Build a from-scratch, single-user text inference engine in pure Rust and raw Metal that loads the EXL3 Gemma 4 checkpoint from `~/.kaplumba/models/Gxma/Gxma` by default, performs correct chunked prefill and ordinary autoregressive decode, supports the agreed EXL3 format matrix, keeps the vision tower off the GPU, and meets the measured 8k-context performance and memory gates.

## Non-goals and restrictions

- Do not inspect, restore, or depend on previous repository commit history.
- Do not use, import, link, copy, or consult the prohibited third-party implementation.
- Use only independent upstream format documentation/source and the authorized `omlx` reference.
- Product and runtime code must contain no Python. Read-only reference probes are allowed during development only.
- Target one user, one synchronous request, and no batching, server, persistent sessions, or multi-tenant scheduling.
- Text only. Skip every `model.vision_tower.*` and `model.embed_vision.*` tensor; reject image, audio, and video special-token inputs.
- Do not implement speculative decoding, draft heads, verification paths, n-gram lookup, n-gram caches, or other decode acceleration that changes the autoregressive token stream.
- Do not use a CPU inference fallback. Metal is required and failures must be explicit.
- Do not convert the checkpoint to dense weights, requantize it, or use a lossy weight fallback during normal inference.
- Do not quantize the KV cache.
- Do not materialize a prefill score matrix or a context-dependent mask. Prefill attention must use online-softmax/Flash-style streaming with an explicitly owned, fixed-size working set independent of context length.
- Do not add vision, audio, video, tools, structured message history, a general Jinja renderer, or a raw-prompt bypass in this release.
- Preserve FP16 storage/activations with FP32 reductions and FP16-compatible checkpoint conversion. Preserve the checkpoint arithmetic and final logit softcap.

## Checkpoint facts

The supplied checkpoint is a Gemma 4 conditional-generation checkpoint with the following text graph and storage facts:

- 60 text layers, with sliding attention at every layer except 5, 11, 17, 23, 29, 35, 41, 47, 53, and 59.
- 50 sliding layers and 10 full-attention layers.
- Hidden size 5376, intermediate size 21504, vocabulary 262144.
- 32 query heads; sliding layers use 16 KV heads with head dimension 256; full layers use 4 KV heads with head dimension 512.
- Sliding window is 1024 total positions, inclusive of the current position.
- Full-attention K is reused as V; attention scale is 1.0.
- RoPE is default/theta 10000 for sliding layers and proportional/partial factor 0.25/theta 1000000 for full layers.
- No biases, KV sharing, MoE, per-layer input embeddings, or audio weights are present in the text graph.
- Embedding scale is the square root of 5376; final logits use `tanh(logits / 30) * 30`.
- Text tensors total 14,876,563,172 bytes (13.85 GiB), including 60 BF16 layer-scalar tensors worth 120 bytes. The quantization metadata covers 14,876,563,052 bytes of those text tensors. Vision tensors total 1,151,487,072 bytes and are excluded from GPU residency.
- Current text linear tensors use 3-bit mcg EXL3; `lm_head` uses 6-bit mcg EXL3.
- The current `generation_config.json` defaults are temperature 1.0, top-k 64, top-p 0.95, and EOS IDs 1, 106, and 50.

## Requirements

### R1. Workspace and runtime

1. Recreate the existing workspace boundaries `crates/engine`, `crates/kernels`, and `crates/cli`.
2. Implement the runtime in Rust and Metal only.
3. Provide a synchronous CLI with a configurable `--model` path and default `~/.kaplumba/models/Gxma/Gxma`.
4. Accept a prompt from an argument, standard input, or a prompt file, with no raw-template bypass.
5. Reject empty prompts, `--max-new-tokens=0`, media tokens, and prompt-plus-generation lengths above 8192; allow exactly 8192 total tokens.

### R2. Checkpoint and EXL3 loading

1. Parse `config.json`, `generation_config.json`, the safetensors index, safetensors headers, and quantization metadata without loading vision tensors.
2. Validate the Gemma 4 text graph, tensor names, dimensions, dtypes, bitrates, codebook markers, and shard offsets.
3. Support integer K=1 through K=8 with plain, mcg, and mul1 codebooks.
4. Support executable fractional rates 1.5, 2.5, and 3.5 with mul1, matching the upstream layout.
5. Accept both packed `su`/`sv` and unpacked `suh`/`svh` sign/scale representations.
6. Implement the EXL3 16x16 tile layout, packed trellis representation, tensor-core permutation, mcg/plain/mul1 decoding, and blockwise 128 Hadamard transforms without materializing dense matrices in the normal forward path.
7. Copy all required text weights into Metal-owned shared buffers before generation; keep vision bytes out of GPU buffers.
8. Convert BF16 checkpoint tensors to FP16 at load while retaining the exact EXL3 bitstream for quantized tensors.

### R3. Model execution

1. Implement scaled token embedding lookup and all 60 decoder layers.
2. Implement weighted RMSNorm, unweighted V RMSNorm, residual additions, layer scalars, tanh GELU, gated MLP, and final logit softcap.
3. Implement separate full/sliding RoPE parameter paths and position offsets.
4. Implement append-only full-attention KV caches and fixed-capacity sliding ring caches with inclusive 1024-position semantics.
5. Implement chunked prefill with a default chunk size of 256, preserving cache state across chunks and decode steps.
6. Implement online-softmax attention for full and sliding layers without an O(L^2) score or mask buffer.
7. Implement ordinary one-token autoregressive decode with no speculative or n-gram acceleration.

### R4. Sampling and output

1. Use generation-config defaults of temperature 1.0, top-k 64, and top-p 0.95.
2. Expose temperature, top-k, top-p, and seed controls; temperature 0 selects greedy argmax.
3. Apply top-k before top-p using FP32 probabilities and renormalization, with no repetition penalty or logit bias.
4. Use a random runtime seed unless `--seed` is supplied; use a fixed seed for tests and benchmarks.
5. Stop on any configured EOS ID 1, 106, or 50, or when the context-bounded generation limit is reached.
6. Convert every generated token ID through the loaded tokenizer into proper Unicode text; normal output must never expose packed bits, byte fragments, or tokenizer artifacts.
7. Detokenize incrementally so multi-byte UTF-8 characters and special-token boundaries are emitted correctly, and optionally expose raw token IDs only behind an explicit diagnostic flag.
8. Suppress protocol/special tokens from normal output.

### R5. Observability and validation

1. Report post-load TTFT, prefill tokens/sec, decode tokens/sec, prompt/generated token counts, and peak memory.
2. Define TTFT as tokenization plus prefill plus the first decode step, excluding model loading.
3. Use a fixed, recorded deterministic 2k-token English benchmark prompt, 8 warmup decode steps, and a 128-token greedy decode average.
4. Require at least 20 decode tokens/sec on the supplied checkpoint under that benchmark.
5. Enforce a maximum of 18,000,000,000 bytes peak process RSS for a fully resident 8k run, without double-counting unified-memory pages.
6. Test every supported EXL3 variant with synthetic codec and linear parity fixtures.
7. Test small-model layer parity, sliding/full attention boundaries, prefill/decode equivalence, sampling determinism, and error cases.
8. Run a strict real-checkpoint smoke test with a fixed short prompt, finite outputs, coherent nonempty text, no media-token leakage, and no Metal/runtime errors.
9. Manually inspect the real-checkpoint streamed text after token-to-Unicode conversion and record the observed transcript/result in `implementation_notes.md`; acceptance requires coherent human-readable text, not merely finite numeric values.

## Acceptance criteria

- `cargo check`, `cargo test`, `cargo clippy`, and the final Metal build succeed without Python runtime dependencies.
- The loader proves that all required text tensors are resident in Metal-owned buffers and that no vision tensor is uploaded.
- All required EXL3 variant combinations pass synthetic dequantization, transform, linear, and prefill/decode parity tests.
- Chunked prefill at 1k, 2k, and 8k prompt lengths preserves the same next-token result as an unchunked reference within the documented numerical tolerance.
- Full and sliding attention pass boundary tests, including 1024-position inclusion, cache rollover, and causal masking.
- The real checkpoint produces coherent streamed Unicode text with finite values, no forbidden acceleration paths, and a manually inspected transcript recorded in the implementation notes.
- Normal CLI output is tokenizer-decoded text; packed bits, byte fragments, and raw token IDs never appear unless explicitly requested for diagnostics.
- The fixed benchmark reaches at least 20 generated tokens/sec after 8 warmup steps over 128 measured greedy tokens.
- The 8k fully resident run stays below 18,000,000,000 bytes peak RSS under the agreed measurement method.
- The CLI emits the agreed metrics and handles all specified boundary/error cases.
- `tasks.md` is updated after every completed task, implementation notes are current, and each completed task has its authorized commit.

## Open questions

None. The user explicitly confirmed the scope and constraints before this specification was created.
