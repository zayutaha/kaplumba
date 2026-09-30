# Spec: Decode 19+ tok/s and 250+ GB/s via repacked weights

## 1. Goal

Prove Apple M4 Pro memory subsystem sustains 250+ GB/s DRAM streaming bandwidth on large buffers (up to 12 GB), eliminate EXL3 structural lane-load amplification via offline on-disk weight repacking without quality loss, overlap non-matvec runtime, and sustain >= 19.0 tok/s autoregressive decode on Gemma 4 31B (`Gxma`).

## 2. Non-goals & restrictions

- **No quality loss:** Repacked weights must dequantize to exact same numeric values. Token generation must match bit-exact greedy decode.
- **No speculative decoding:** No draft models, no Medusa heads, no lookahead speculation. Pure autoregressive single-token decode.
- **No KV cache quantization:** FP16/BF16 KV cache preserved.
- **Weight file safety:** Never modify original model in `~/.kaplumba/models/Gxma/Gxma`.
- **Single disk copy limit:** At most one repacked copy (`~/.kaplumba/models/Gxma-repacked`) on disk at a time. Delete old repacked copy before generating a new version.
- **No Python in engine:** Pure Rust + Metal.
- **Single-user batch-1 only:** No request batching or server abstractions.

## 3. Requirements

1. **R1 (Synthetic probe):** Standalone Metal probe measuring streaming memory bandwidth across 2 GB, 6 GB, and 12 GB buffers. Explore access patterns (vector width `uint4`/`ulong2`, thread concurrency, unrolling, cache policies). Prove hardware capability >= 250 GB/s.
2. **R2 (Fail-first testing):** All measurement instruments and kernel tests must be verified with fail-first sabotage per AGENTS.md §3.4.
3. **R3 (Offline repacker):** Standalone Rust tool converting EXL3 weights on disk to repacked layout that eliminates unaligned multi-word ring loads and 2.67x lane-load amplification.
4. **R4 (Repacked GEMV kernel):** Specialized Metal GEMV kernel consuming repacked layout with 100% coalesced 128-bit aligned reads.
5. **R5 (Non-matvec overlap/fusion):** Overlap RMSNorm, RoPE, and attention scheduling with GEMV command streams to compress the 7.4 ms non-matvec GPU span.
6. **R6 (End-to-end performance):** Achieve >= 19.0 tok/s decode throughput on M4 Pro under `scripts/gpuq run`.

## 4. Acceptance criteria

1. `cargo run --release --example probe_bw_250gb` reports median >= 250.0 GB/s on 12 GB buffer.
2. Repacker generates `~/.kaplumba/models/Gxma-repacked` without touching original directory; disk usage remains within single-copy limit.
3. `crates/engine/tests/decode_matvec_golden.rs` passes bit-exact comparison between original and repacked kernels.
4. `scripts/gpuq run -- ./target/release/kaplumba --model ~/.kaplumba/models/Gxma-repacked --prompt "Explain the theory of relativity." --max-new-tokens 64 --temperature 0 --metrics` outputs >= 19.0 tok/s.

## 5. Open questions

None. Requirements and restrictions confirmed with user.
