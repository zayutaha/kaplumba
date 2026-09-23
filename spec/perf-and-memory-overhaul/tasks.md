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

- [x] **T5.1: Land cooperative 128-thread GEMV + re-tuned K-split policy**
  - **Files**: `crates/kernels/src/exl3.metal`, `crates/kernels/src/exl3.rs`, `spec/perf-and-memory-overhaul/plan.md`
  - **Intent**: In-place `exl3_gemv` rewrite (NSG=4, 128 threads/TG, `tk += 4` striding, on-chip `tg_cols[4][16]` reduction) + `gemv_splits(in,out)` heuristic (`min_split_tiles=128`, target 8192 TGs → lm 1, gate/q/k/v 2, o 4, down 8). Supersedes T2.9 nsplit=16 over-split.
  - **Testing**: `cargo test -p kaplumba-kernels` 21/21, `cargo test -p kaplumba-engine --lib` 116/116.
  - **Commit**: `d3d3ffb`

- [x] **T5.2: Coalesced 96-byte tile word loads + simd_shuffle word fetch**
  - **Files**: `crates/kernels/src/exl3.metal`
  - **Intent**: ExllamaV3-style: when `aligned && twords <= 32`, lanes `0..twords-1` load the tile's u32 words in one contiguous burst (`w_lane`), fetch `ig0/ig1` via `simd_shuffle`. k=6 (lm_head, 48 words) + unaligned keep per-lane `ld_word` fallback. Keep `accA/accB` mapping (no acc[8] remap — approved scope cut).
  - **Testing**: kernels 21/21, `gpu_linear_matches_cpu_reference`, `gpu_gemv_matches_cpu_decoded_tile`, `debug_gemv_rows` one-hot (T2.3 dead-end guard — GPU==CPU all rows).
  - **Commit**: `182a101`

- [ ] **T5.3: Measure per-pass bandwidth + decode throughput**
  - **Files**: `spec/perf-and-memory-overhaul/implementation_notes.md`
  - **Intent**: `bench_passes_slow` (target gate/down/o/q/k/v ≥ 200–240 Gw/s; k/v must jump from 93–94) + `decode_throughput_slow` baseline with new kernel. Tune constants only if a shape regresses.
  - **Testing**: both gated benches run, numbers recorded; sanity gate `tg_tps > 0.5` stays.

- [ ] **T5.4: Coherence & correctness (hard gate)**
  - **Files**: none (verification only; fix kernel/policy if red)
  - **Intent**: Full gated suite + `trace_chat_prompt_steps_slow` (token 818, margin > 8, 20 steps bit-exact) + live CLI France prompt → exact answer, no gibberish.
  - **Testing**: full gated suite green (fixture 236773, span < 0.55, batched < 2e-3); CLI prints "The capital of France is Paris."

- [ ] **T5.5: Close T5 — gate, docs, ceiling-conflict amendments**
  - **Files**: `crates/engine/src/gemma4.rs`, `spec/perf-and-memory-overhaul/{tasks,implementation_notes}.md`, `spec/final-same-algorithm-ceiling/{tasks,spec}.md`
  - **Intent**: If measured ≥ 19.5 t/s → flip `decode_throughput_slow` assert to `>= 19.5` (never land red; if short, keep sanity gate + record honest shortfall). Final notes table. Amend final-same-algorithm-ceiling: T2.12 superseded, T6.1 ceiling caveat (approved 2026-09-23).
  - **Testing**: re-run `decode_throughput_slow` after gate flip; docs committed.

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
