# Concrete Task Checklist: Engine Performance

- [ ] **T-1 — Read the mandatory references before touching any kernel.**
  - Files: `reference/omlx`, `/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/ishizuki`, `/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/exllamav3`
  - Action: Read oMLX's `omlx_sdpa_decode` / `omlx_sdpa_prefill` (32-simdgroup attention + threadgroup merge), oMLX `qwen35_packed_linear.py` `project()` (weight concat), ishizuki's matvec and attention, and exllamav3's `exl3_gemm_inner.cuh` (`cp.async` staging). Record every technique we have not yet tried, with `path:line`.
  - Verification: A new "Techniques we have not tried" section exists in `implementation_notes.md`, each entry citing `path:line` and a guess at whether it is bit-exact.

- [ ] **T0 — Fix `attention_metal` test harness encoder lifecycle.**
  - Files: `crates/engine/tests/attention_metal.rs`
  - Action: Audit all command encoder instances; ensure `encoder.endEncoding()` is called explicitly before encoders exit scope.
  - Verification: `cargo test --release -p kaplumba-engine --test attention_metal` passes cleanly without assertions.

- [ ] **T1 — Prefill GEMM weight decode ablation & lookup table.**
  - Files: `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`
  - Action: Measure exact time spent in `exl3_code_at_tg` unpacking logic. Implement a 256-entry table for MCG decode state to eliminate integer div/mod ALU instructions.
  - Verification: Benchmark on 600-token prompt with `cargo build --release -p kaplumba-cli`; confirm bit-exact output on five reference prompts.

- [ ] **T2 — Prefill attention parallelization.**
  - Files: `crates/kernels/src/attention.metal`, `crates/kernels/src/attention.rs`
  - Action:
    - *Path A (Bit-Exact):* Implement prefix-max scan across 256 threads to find rescale points; parallelize `exp()` evaluations across all threads while thread 0 performs ordered additions.
    - *Path B (Relaxed-Float):* Port oMLX 32-simdgroup interleaved online softmax with 4 KB threadgroup merge.
  - Verification: Measure attention time reduction (target: <10 ms/token); verify generation output against reference prompts.

- [ ] **T3 — Metal tensor-core simdgroup layout probe fix.**
  - Files: `crates/engine/examples/metal_probe.rs`, `crates/kernels/src/linear.metal`
  - Action: Correct the SIMD lane fragment mapping for `simdgroup_matrix<half, 8, 8>` so that the identity test matches 64 of 64 elements.
  - Verification: `cargo run --release -p kaplumba-engine --example metal_probe` passes 64/64 identity check and reports sustained TFLOPS.

- [ ] **T4 — Resume decode path optimizations.**
  - Files: `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`
  - Action: Implement decode codebook LUT and cooperative 128-bit ring loads in `linear_gemv_exl3_k*`. Sweep grid-stride parameters to match 20 GPU cores.
  - Verification: Decode throughput advances from 7.87 tok/s toward 12–16 tok/s; full test suite remains green.

- [ ] **T5 — Full context scaling & memory audit (1k / 2k / 8k).**
  - Files: `crates/engine/src/model.rs`, `crates/engine/src/generate.rs`
  - Action: Measure prefill throughput and peak RSS at 1024, 2048, and 8192 tokens.
  - Verification: Verify prefill throughput, confirm KV memory footprint matches 1.41 GiB at 8192, and record `/usr/bin/time -l` physical footprint.
