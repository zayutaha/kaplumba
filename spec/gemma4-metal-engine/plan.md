# Technical Plan: Gemma 4 Metal Engine

## Architecture and data flow

1. The CLI parses arguments, resolves the model directory, loads the tokenizer, and renders the verified one-turn Gemma 4 template.
2. The engine parses the JSON configuration and safetensors headers, builds a text-only tensor view, validates the graph and EXL3 metadata, and copies text weights into Metal-owned shared buffers.
3. The graph allocator creates fixed activation scratch buffers, full-attention KV storage for the selected context, 1024-position sliding ring storage, and sampler/logit buffers.
4. Prefill tokenizes and embeds a bounded chunk. Each layer performs its input norm, quantized projections, RoPE, KV append, online attention, gated MLP, residual/norm operations, and layer scaling. The cache remains live between chunks.
5. After the final chunk, the final norm and quantized language-model head produce logits. The sampler selects a token using the configured distribution.
6. Decode repeats the same path for one token, updates the cache, samples, streams clean text, and reports metrics.
7. No dense weight reconstruction, speculative path, n-gram path, CPU inference path, or vision execution is present in the normal runtime.

## Planned file and module touchpoints

- `Cargo.toml`: retain the three-crate workspace and add only the Rust dependencies required for JSON, tokenization, half precision, platform mmap, and Metal.
- `crates/engine/src/config.rs`: typed checkpoint configuration, generation configuration, validation, and constants derived from the model.
- `crates/engine/src/safetensors.rs`: mmap-backed shard/header reader, tensor lookup, dtype/shape/offset validation, and text-only filtering.
- `crates/engine/src/weights.rs`: BF16/FP16 conversion, EXL3 tensor descriptors, sign formats, Metal-owned buffer creation, and residency accounting.
- `crates/engine/src/exl3.rs`: CPU reference codec, bit packing/unpacking, codebooks, tile permutation, Hadamard helpers, and variant fixtures.
- `crates/engine/src/metal.rs`: device/queue/library/pipeline ownership and command-buffer helpers.
- `crates/engine/src/kv_cache.rs`: full and sliding cache layouts, append/rollover rules, position mapping, and memory accounting.
- `crates/engine/src/model.rs`: layer descriptors and the complete Gemma 4 text forward graph.
- `crates/engine/src/generate.rs`: chunked prefill, decode loop, stopping, and stream callbacks.
- `crates/engine/src/tokenizer.rs`: tokenizer loading, one-turn template rendering, media-token rejection, and detokenization.
- `crates/engine/src/sampler.rs`: greedy, temperature, top-k, top-p, seeded RNG, and stop handling.
- `crates/engine/src/metrics.rs`: TTFT, throughput, token counts, and peak RSS measurement/reporting.
- `crates/engine/src/lib.rs`: public engine API and error types.
- `crates/kernels/src/*.rs`: Metal library source embedding and pipeline registration helpers.
- `crates/kernels/src/*.metal`: EXL3 decode/linear, norm/RoPE/activation, embedding/head, and online-attention kernels.
- `crates/cli/src/args.rs`: model, prompt, generation, seed, debug-ID, and metric arguments.
- `crates/cli/src/main.rs`: synchronous command flow and error reporting.
- `crates/cli/src/report.rs`: streamed output and final metrics formatting.
- `crates/engine/tests/`: synthetic format, graph, attention, sampler, and integration tests.
- `crates/kernels/tests/`: Metal parity and dispatch tests where host-side fixtures are available.
- `crates/cli/tests/`: argument, template, boundary, and output tests.

## Formats and interfaces

### Quantized linear descriptor

Each executable linear owns:

- `in_features` and `out_features`, both validated against the 16/128 tile constraints.
- `bits`: integer 1–8 or supported fractional rate.
- `codebook`: plain, mcg, or mul1.
- `trellis`: contiguous packed bytes with shape `[in/16, out/16, packed_words]`.
- Input scale/sign data from either `su` or `suh`.
- Output scale/sign data from either `sv` or `svh`.
- Optional bias, rejected for this checkpoint but represented for format validation.

The normal forward path keeps the trellis and scale buffers packed. The CPU codec is a test/reference path only.

### EXL3 numeric path

- Expand the packed trellis into 16-bit trellis states using the K-specific bit layout.
- Decode plain, mcg, or mul1 states with the specified integer/codebook operation.
- Restore the tensor-core 16x16 tile permutation.
- Apply the normalized blockwise 128 Hadamard transform on the input and output dimensions with the stored scale/sign vectors.
- Use FP16 storage and FP32 dot-product accumulation.
- Instantiate Metal variants for every supported K/codebook combination; the current checkpoint uses K=3 for all text linears and K=6 for the head.

### KV and attention layout

- Full layers use contiguous `[sequence, 4, 512]` K and V storage for the selected context.
- Sliding layers use a 1024-slot ring with `[slot, 16, 256]` K and V storage.
- Position IDs are absolute and monotonically increasing.
- Sliding attention attends to `[max(0, position - 1023), position]`.
- Attention uses online max/sum/output accumulators in FP32 and never writes a full score matrix or context-sized mask.

### CLI contract

- `--model <path>` defaults to `~/.kaplumba/models/Gxma/Gxma`.
- Prompt input comes from `--prompt`, `--prompt-file`, or stdin.
- `--max-new-tokens` defaults to 128 and is bounded by the remaining 8192-token context.
- `--temperature`, `--top-k`, `--top-p`, and `--seed` control sampling; temperature 0 is greedy.
- Default runtime seed is random; tests and benchmarks pass a fixed seed.
- Normal output is streamed Unicode text produced by incremental tokenizer detokenization; it must never print packed bits, byte fragments, or raw token IDs.
- A diagnostic flag may print raw token IDs separately without changing the text stream.
- The real-checkpoint gate includes manual inspection of the resulting text and a recorded transcript/result in `implementation_notes.md`.
- Metrics are emitted after the stream and include post-load TTFT, prefill tok/s, decode tok/s, token counts, and peak RSS.

## Implementation sequencing

1. Scaffold and test the workspace.
2. Build configuration, safetensors, and text-only weight loading.
3. Build the complete CPU EXL3 reference and synthetic fixtures.
4. Build Metal codec and quantized linear kernels, first for batch-one decode and then for prefill tiles.
5. Build elementwise, norm, RoPE, embedding, activation, and language-head kernels.
6. Build KV caches and online full/sliding attention.
7. Assemble the Gemma 4 layer graph and validate it on small synthetic configurations.
8. Add tokenizer/template, sampler, streaming, CLI, and metrics.
9. Run the full variant matrix, actual-checkpoint integration, 1k/2k/8k prefill, memory, and throughput gates.
10. Manually inspect the real streamed Unicode text, record the transcript/result, update notes/tasks, commit each completed task, and run final validation.

## Risks and fallbacks

- **Metal residency:** no-copy mappings are not used for the release path; if a buffer copy or alignment fails, fail with a precise error rather than silently using a CPU or dense fallback.
- **Memory pressure:** measure allocation before copying the language head and fail before exceeding the 18 GB RSS gate; never trade away the no-score-matrix prefill rule.
- **Codec variation:** keep variant-specific CPU references and synthetic fixtures separate from the optimized Metal path; a parity failure blocks promotion.
- **Tile performance:** start with correctness-oriented tiled GEMV/GEMM, then profile dispatch, occupancy, and memory access until the 20 tok/s gate is met.
- **Large head:** the 6-bit language head must remain packed; do not reconstruct a 262144-column dense matrix.
- **Numerical drift:** use FP32 reductions and document per-operation tolerances; do not hide drift with greedy-only tests.
- **Sampling reproducibility:** isolate RNG state and make seed handling independent of Metal scheduling.
- **Checkpoint mismatch:** fail fast on missing text tensors, unsupported EXL3 metadata, malformed headers, or vision-only dependencies.

## Requirement-to-task mapping

- R1 and R6: T1, T2, T13, T14.
- R2 and R3 codec portions: T2, T3, T4, T5, T6, T7.
- R3 model execution: T8, T9, T10, T11, T12.
- R4 sampling and output: T13.
- R5 validation and performance: T4, T5, T7, T10, T12, T14, T15, T16.
