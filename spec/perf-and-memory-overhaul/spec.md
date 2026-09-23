# spec/perf-and-memory-overhaul

## Goal
Eliminate memory bloat and performance bottlenecks in the Gemma 4 31B EXL3 engine on Apple M4 Pro (20 GPU cores, 273 GB/s) by: (1) eliminating unused multimodal vision weights from resident memory to reach ≤ 14.6 GiB (15.7 GB) resident footprint; (2) overhauling GEMV decode with `dq4` vectorized decoding, SIMD shuffle activation sharing, and inlined Hadamard input to achieve ≥ 18.0 t/s decode (consistent, verify-gated); and (3) implementing decoupled striped GEMM with an explicitly owned fixed-size 16 MB scratch buffer to achieve ≥ 200 t/s prefill.

## Non-goals / restrictions
- **No MTP or n-gram speculative decoding**: throughput must be achieved via raw kernel execution efficiency.
- **No quality loss**: verify-gated token tolerances hold (fixture argmax 236773, span-vs-loop within tolerance).
- **No Python / MLX in engine**: engine code remains 100% Rust and raw Metal Shading Language.
- **No KV cache quantization**: KV cache remains full FP16.
- **No dynamic or context-growing prefill scratch**: scratch buffer is fixed-size (16 MB), owned by `GpuDecoder`, independent of sequence length, with zero retained-pool semantics (strictly adhering to AGENTS.md §8).
- **No multi-user or batch scheduling**: single-user chat focus.
- Target hardware: Apple M4 Pro (20 GPU cores, 273 GB/s unified memory).

## Requirements (numbered, testable)

1. **R1: Vision Tower Omission in Loader**
   `load_exl3` in `crates/engine/src/load.rs` must not allocate Metal buffers or load bytes for tensors prefixed with `model.vision_tower.` or `model.embed_vision.`.
   *Testable*: `load_real_artifact_if_present` reports resident weights ≤ 14.9 GB (13.9 GiB), down from 16.03 GB.

2. **R2: Resident Memory Budget ≤ 14.6 GiB**
   Total process resident memory (RSS) after model loading and `GpuDecoder` initialization (with 0 context tokens) must not exceed 14.6 GiB (15.7 GB).
   *Testable*: `prefill_memory_profile_slow` asserts baseline RSS ≤ 15.7 GB.

3. **R3: `dq4` and SIMD-Shuffle GEMV Decode Kernel**
   `exl3.metal` and `exl3.rs` must implement `dq4` 4-codeword extraction from a single 64-bit window and simdgroup register shuffle (`simd_shuffle`) for activation loading.
   *Testable*: `gpu_gemv_matches_cpu_decoded_tile` and `bench_passes_slow` pass with increased Gw/s.

4. **R4: Inlined Hadamard Input Prologue**
   The Hadamard right-side butterfly (`had_in`) must be inlined into the GEMV kernel for decode rows=1, eliminating separate kernel dispatches.
   *Testable*: `decode_throughput_slow` dispatches fewer command buffers/encoders and maintains numerical parity.

5. **R5: Decode Generation Throughput ≥ 18.0 t/s**
   Autoregressive generation (`decode_throughput_slow`, 64 tokens, greedy) must sustain ≥ 18.0 tokens/second on Apple M4 Pro.
   *Testable*: `decode_throughput_slow` reports `tg_tps >= 18.0`.

6. **R6: Fixed-Size Striped Prefill GEMM**
   `GpuDecoder` owns a single, constant-sized 16 MB stripe buffer. Prefill decomposes wide linear layers into column stripes (e.g. 512 columns), dequantizing the stripe to FP16 and executing high-throughput tiled GEMM.
   *Testable*: Peak prefill RSS growth remains ≤ 48 MB across 1k/2k/8k context lengths.

7. **R7: Prefill Throughput ≥ 200 t/s**
   Prompt processing throughput (`prefill_throughput_slow`) must achieve ≥ 200 tokens/second at 512, 2048, and 8192 prompt tokens.
   *Testable*: `prefill_throughput_slow` reports `pp_tps >= 200.0`.

## Acceptance Criteria
- `cargo test -p kaplumba-kernels`: All tests pass.
- `cargo test -p kaplumba-engine --lib -- --test-threads=1`: All 114+ tests pass.
- `KAPLUMBA_RUN_SLOW=1 cargo test -p kaplumba-engine --release --lib -- --ignored --nocapture --test-threads=1`:
  - `decode_throughput_slow`: ≥ 18.0 t/s.
  - `prefill_throughput_slow`: ≥ 200.0 t/s at 512 tokens.
  - `prefill_memory_profile_slow`: Baseline RSS ≤ 15.7 GB, prefill RSS delta ≤ 48 MB.
  - `real_artifact_cached_fixture_and_gpu_slow`: Golden fixture argmax token 236773 preserved.

## Open Questions
- None. (Requirements, constraints, and architecture confirmed).
