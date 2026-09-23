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

- [x] **T5.0: Revert cooperative-GEMV regression (A/B proven)**
  - **Files**: `crates/kernels/src/exl3.metal`, `crates/kernels/src/exl3.rs`
  - **Intent**: T5.1+T5.2 (128-thread 4-simdgroup structure + min_split_tiles=128 policy + coalesced loads) regressed every shape vs T4 on this machine: gate 165→130, down 168→131, o 109→95, q 111→81, k 70→55, v 76→55, lm 243→225 Gw/s. Causes: discarded T2.6 2-way unroll, ignored the file's own "fat threadgroups slower on M4 Pro" finding, starved k/v to nsplit=2. Revert to T4 shape; keep goal.
  - **Testing**: kernels 21/21, engine fast 116/116, bench_passes back to T4 numbers.
  - **Commit**: `c250fc0`

- [x] **T5.0b: Correct decode bandwidth metric (16.028 → 11.2 GB/token)**
  - **Files**: `crates/engine/src/gemma4.rs`, `spec/perf-and-memory-overhaul/implementation_notes.md`
  - **Intent**: True per-token stream = 11.2 GB (vision removed T1, embed mmap T2) → roofline 11.2÷273 ≈ 41 ms ≈ **24.4 t/s** (user-confirmed 11.2 GB). The old 16.028e9 figure overstated stream ~43% and produced the wrong 17.6 t/s "hard ceiling" used to declare 18/20 unreachable. Print roofline in `decode_throughput_slow`.
  - **Testing**: `decode_throughput_slow` prints corrected GB/s + roofline; fast suite green.

- [x] **T5.1r: Prefetch ring in the T4 32-thread gemv (ExllamaV3's actual trick)** — ATTEMPTED, REVERTED `fb26c21` (impl `a4e15e2`)
  - **Files**: `crates/kernels/src/exl3.metal`
  - **Intent**: PF depth 2–4 register prefetch: load tile i+PF words while decoding tile i (currently load-then-decode serial → latency never hidden → issue-bound ~50%, 60% BW). Keep T4 structure (32 threads, 2-way unroll, no fat TGs).
  - **Result**: PF=4 ring (4-slot, dynamic `I&3` indexing, consume-then-refill I+4) regressed EVERY shape: gate 165→107, down 168→108, o 109→80, q 111→81, k 70→63, v 76→57, lm 243→135 Gw/s. Correctness fine (21/21, one-hot, parity, 116 fast). Likely cause: MSL arrays with dynamic indexing demoted to stack memory (spill) + occupancy cut. See implementation_notes §T5.1r.
  - **Testing**: kernels 21/21, parity tests, one-hot; **bench gate/down ≥ 200 Gw/s or revert** → reverted.

- [x] **T5.2r: Split-policy empirical sweep** — LANDED `d8adb19`
  - **Files**: `crates/kernels/src/exl3.rs`, `crates/engine/src/metal_linear.rs`
  - **Intent**: Measure k/v ∈ {2,4,8,16}, gate ∈ {1,2,4}, o ∈ {4,8,13}, q ∈ {2,4,8} isolated; bake per-shape argmax into `gemv_splits` + policy test. (T2.9 history: k/v best ≈ 4.)
  - **Result**: sweep (gemv-only, 12 reps) argmax → gate 4 (167), down 16 (167), o 8 (114), q 4 (119), k 8 (89), v 8 (78), lm 1 (241). Baked only the 2 outliers vs 4096-TG formula: q 8→4, k/v 16→8. `bench_passes_slow` A/B: gate 165→168, down 168→168, o 109→113, q 111→115, k 70→76, v 76→79, lm 243→242 — no shape below baseline. Sweep harness `bench_gemv_split_sweep_slow` + `gemv_pass_nsplit` + `encode_gemv(nsplit_override)` retained for future splits.
  - **Testing**: every shape ≥ T4 baseline in bench_passes_slow ✓; policy test updated; kernels 21/21, parity, 116 fast.

- [ ] **T5.3r: Coalesced 96B loads on the 32-thread kernel (isolated)**
  - **Files**: `crates/kernels/src/exl3.metal`
  - **Intent**: Re-apply T5.2's lane-coalesced word burst + simd_shuffle fetch *inside the T4 2-way-unroll loop* — never measured without the structure change. Keep only if bench improves.
  - **Testing**: one-hot + parity; bench shows per-shape gain or the change is dropped.

- [ ] **T5.4r: R4 — inline had_in/had_out into `exl3_gemv`**
  - **Files**: `crates/kernels/src/exl3.metal`, `crates/kernels/src/exl3.rs`
  - **Intent**: rows=1 currently dispatches 3 kernels per linear × 420 linears ≈ **840 extra Hadamard dispatches/token** (~0.15–0.2 ms each isolated). Fuse had_in as gemv prologue (128-block butterfly in tg memory) and had_out as epilogue (y is f32 in registers → scale+FWT+store f16 directly). Drops 840 dispatches/token.
  - **Testing**: kernels 21/21, `gpu_linear_matches_cpu_reference`, `gpu_gemv_matches_cpu_decoded_tile`, had-in/out ≤1 ulp parity retained via fused path; bench `full` time ↓.

- [ ] **T5.5r: End-to-end 20 t/s push loop**
  - **Files**: as needed
  - **Intent**: Re-bench after each of T5.1r–T5.4r; iterate (prefill not in scope) until `decode_throughput_slow` ≤ 50 ms/token. Phase split (cpu_encode/gpu_wait/host) to find the next bottleneck if linears are fixed.
  - **Testing**: gated decode bench trend recorded each iteration.

- [ ] **T5.6r: Coherence & correctness (hard gate)**
  - **Files**: none (verification)
  - **Intent**: Full gated suite + `trace_chat_prompt_steps_slow` (token 818, margin > 8) + CLI France prompt → exact answer, no gibberish.
  - **Testing**: fixture 236773, span < 0.55, batched < 2e-3 all hold.

- [ ] **T5.7r: Close T5 — gate, docs, ceiling amendments**
  - **Files**: `crates/engine/src/gemma4.rs`, spec docs, `spec/final-same-algorithm-ceiling/{spec,tasks}.md`
  - **Intent**: If ≥ 19.5 measured → flip `decode_throughput_slow` assert (never land red). Final notes. Amend ceiling spec: 17.6 t/s roofline superseded by 24.4 (11.2 GB corrected); T2.12 split-cap superseded if sweep says so (approved 2026-09-23).
  - **Testing**: re-run decode bench after gate flip; all docs committed.

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
