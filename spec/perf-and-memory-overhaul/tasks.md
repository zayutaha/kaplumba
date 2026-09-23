# Tasks: Performance and Memory Overhaul

- [x] **T1: Filter vision tower tensors during model load**
  - **Files**: `crates/engine/src/load.rs`
  - **Intent**: Skip `model.vision_tower.*` and `model.embed_vision.*` tensors when parsing safetensors and packing shard buffers, saving 1.151 GB of resident memory.
  - **Testing**: `cargo test -p kaplumba-engine load_real_artifact_if_present` asserts weights_nbytes drop from 16.03 GB to 14.88 GB.

- [x] **T2: Verify baseline resident memory reduction**
  - **Files**: `crates/engine/src/gemma4.rs`, `crates/engine/src/load.rs`, `crates/engine/Cargo.toml`
  - **Intent**: Implement HostMmap for embed_tokens.weight so large host table is paged on-demand rather than pre-allocated in resident Metal buffer; verify process baseline RSS with no context tokens is ≤ 14.6 GB (decimal).
  - **Testing**: `prefill_memory_profile_slow` confirms baseline RSS ~3.96 GB (well below 14.6 GB) and 0 KB growth on 8192 prefill; `load_real_artifact_if_present` confirms resident Metal weights 12.058 GB.

- [ ] **T3: Implement vectorized `exl3_decode` and `dq4` window unpacking in Metal**
  - **Files**: `crates/kernels/src/exl3.metal`, `crates/kernels/src/exl3.rs`
  - **Intent**: Replace the 4-pair loop with `dq4` 4-codeword extraction from a single 64-bit window; vectorize decode with `as_type<half2>`.
  - **Testing**: `cargo test -p kaplumba-kernels gpu_gemv_matches_cpu_decoded_tile`.

- [ ] **T4: Implement SIMD-shuffle activation sharing and inlined Hadamard prologue**
  - **Files**: `crates/kernels/src/exl3.metal`, `crates/kernels/src/exl3.rs`
  - **Intent**: Load `xh` once per simdgroup and distribute across rows via `simd_shuffle`; inline 128-point Hadamard butterfly prologue in threadgroup memory.
  - **Testing**: `cargo test -p kaplumba-kernels`, `cargo test -p kaplumba-engine gpu_linear_matches_cpu_reference`.

- [ ] **T5: Wire fast GEMV into `GpuDecoder::encode_token` & verify decode throughput**
  - **Files**: `crates/engine/src/gpu_decode.rs`, `crates/engine/src/metal_linear.rs`
  - **Intent**: Route single-row decode passes through the fast GEMV pipeline; measure generation throughput.
  - **Testing**: `KAPLUMBA_RUN_SLOW=1 cargo test -p kaplumba-engine --release --lib -- --ignored --nocapture --test-threads=1 decode_throughput_slow` achieves ≥ 18.0 t/s.

- [ ] **T6: Implement fixed-size stripe buffer and fast dequantize stripe kernel**
  - **Files**: `crates/kernels/src/exl3.metal`, `crates/kernels/src/exl3.rs`, `crates/engine/src/gpu_decode.rs`
  - **Intent**: Add an explicitly owned 16 MB stripe scratch buffer; implement `exl3_dequant_stripe` to unpack 512 weight columns into FP16 in linear time.
  - **Testing**: `cargo test -p kaplumba-kernels`.

- [ ] **T7: Implement high-throughput tiled stripe GEMM kernel**
  - **Files**: `crates/kernels/src/exl3.metal`, `crates/kernels/src/exl3.rs`
  - **Intent**: Execute hardware-optimized tiled FP16 matrix multiplication (`xh @ stripe_weights`) at 14–16+ TFLOP/s.
  - **Testing**: `cargo test -p kaplumba-kernels`, `cargo test -p kaplumba-engine gpu_gemm_rows_match_per_row_gemv_and_cpu`.

- [ ] **T8: Integrate striped GEMM into `GpuDecoder::encode_span` & verify prefill throughput**
  - **Files**: `crates/engine/src/gpu_decode.rs`, `crates/engine/src/gemma4.rs`
  - **Intent**: Route batched prefill passes through striped GEMM; measure prefill throughput at 512, 2048, and 8192 tokens.
  - **Testing**: `KAPLUMBA_RUN_SLOW=1 cargo test -p kaplumba-engine --release --lib -- --ignored --nocapture --test-threads=1 prefill_throughput_slow` achieves ≥ 200 t/s.

- [ ] **T9: Full suite verification, memory profile audit, and handover notes**
  - **Files**: `spec/perf-and-memory-overhaul/implementation_notes.md`
  - **Intent**: Run all test suites (fast and gated); verify fixture tokens and memory invariants; document final benchmark measurements.
  - **Testing**: Fast test suite (114+ passed), gated suite (all passed).
