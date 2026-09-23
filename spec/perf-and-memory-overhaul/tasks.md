# Tasks: Performance and Memory Overhaul

- [x] **T1: Filter vision tower tensors during model load**
  - **Files**: `crates/engine/src/load.rs`
  - **Intent**: Skip `model.vision_tower.*` and `model.embed_vision.*` tensors when parsing safetensors and packing shard buffers, saving 1.151 GB of resident memory.
  - **Testing**: `cargo test -p kaplumba-engine load_real_artifact_if_present` asserts weights_nbytes drop from 16.03 GB to 14.88 GB.

- [x] **T2: Verify baseline resident memory reduction**
  - **Files**: `crates/engine/src/gemma4.rs`, `crates/engine/src/load.rs`, `crates/engine/Cargo.toml`
  - **Intent**: Implement HostMmap for embed_tokens.weight so large host table is paged on-demand rather than pre-allocated in resident Metal buffer; verify process baseline RSS with no context tokens is ≤ 14.6 GB (decimal).
  - **Testing**: `prefill_memory_profile_slow` confirms baseline RSS ~3.96 GB (well below 14.6 GB) and 0 KB growth on 8192 prefill; `load_real_artifact_if_present` confirms resident Metal weights 12.058 GB.

- [x] **T3: Implement vectorized `exl3_decode` and `dq4` window unpacking in Metal**
  - **Files**: `crates/kernels/src/exl3.metal`, `crates/kernels/src/exl3.rs`
  - **Intent**: Replace the 4-pair loop with `dq4` 4-codeword extraction from a single 64-bit window; vectorize decode with `as_type<half2>`; eliminate threadgroup barriers with SIMD warp reduction.
  - **Testing**: `cargo test -p kaplumba-kernels` (21/21 passed), `gpu_gemv_matches_cpu_decoded_tile` passed, all 115 engine tests passed cleanly.

- [x] **T4: Implement SIMD-shuffle activation sharing and inlined Hadamard prologue**
  - **Files**: `crates/kernels/src/exl3.metal`
  - **Intent**: Load `xh` once per 16 elements in a single coalesced 64-byte burst and distribute across rows via `simd_shuffle`, reducing activation DRAM loads by 8x.
  - **Testing**: `cargo test -p kaplumba-kernels` (21/21 passed), `gpu_linear_matches_cpu_reference` passed, `gpu_gemv_matches_cpu_decoded_tile` passed.

- [ ] **T5: Implement 128-thread cooperative GEMV with coalesced word loads & verify ≥ 20.0 t/s**
  - **Files**: `crates/kernels/src/exl3.metal`, `crates/kernels/src/exl3.rs`, `crates/engine/src/gpu_decode.rs`, `crates/engine/src/gemma4.rs`
  - **Intent**: Implement 4-simdgroup cooperative GEMV (`NSG=4`, 128 threads/TG) with in-tile striding `tk += 4`, private register accumulation `acc[8]`, on-chip threadgroup reduction `tg_w[4][256]`, and ExllamaV3 coalesced 96-byte tile word loading with `simd_shuffle`. Re-tune K-split policy (`min_split_tiles = 128`).
  - **Testing**: `cargo test -p kaplumba-kernels`, `gpu_linear_matches_cpu_reference`, `bench_passes_slow` (linears sustain ≥ 200–240 Gw/s), `trace_chat_prompt_steps_slow` (coherence exact), and `decode_throughput_slow` achieves ≥ 19.5 t/s (target ≥ 20.0 t/s).

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
