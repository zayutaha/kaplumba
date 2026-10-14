# Tasks: Sampled Decode 15.8 tok/s

- [x] **T1 — Revert GEMV kernel register bloat in linear_repacked.metal.**
  - Files: `crates/kernels/src/linear_repacked.metal`.
  - Intent: Restore lean 2-trip burst loop with single `float v[8]` reuse in `linear_gemv_ishizuki_k3_colmajor_2col`, `linear_gemv_ishizuki_k3_colmajor_2col_sk2`, `linear_gemv_ishizuki_k3_fused2_2col`, and `linear_gemv_ishizuki_k3_fused3_2col`.
  - Testing: Sabotage in `linear_gemv_ishizuki_k3_colmajor_2col` produced 22708 mismatches in `live_kernel_bit_exact`. Reverted; `live_kernel_bit_exact`, `repacked_golden`, and `fused_bit_exact` all passed 100%. Commit: 4e5c769.

- [x] **T2 — Enforce strict top_k <= 1024 bound and wire pipelined sampling into generate.rs.**
  - Files: `crates/engine/src/generate.rs`, `crates/engine/src/chat.rs`.
  - Intent: Reject `top_k > 1024` with `SamplerError::DeviceTopKTooLarge`. Support `HeadPick::Sample(params)` in pipelined decode loop in `generate.rs`.
  - Testing: Sabotaged `GenerationConfig::validate` limit to 2048 and confirmed `generation_config_rejects_top_k_above_1024` failed with `expected DeviceTopKTooLarge, got Ok(())`. Reverted. Added unit test in `chat.rs` and `generate.rs`. All 148 library tests passed. Commit: (pending).

- [ ] **T3 — Benchmark and demonstrate >= 15.8 tok/s at temp=1.0 and temp=0.**
  - Files: `spec/sampled-decode-15t8/implementation_notes.md`.
  - Intent: Run canonical 32-token benchmark under `temp = 1.0, top_k = 64, top_p = 0.95` and `temp = 0.0` using `scripts/gpuq try --`.
  - Testing: Verify throughput $\ge 15.8\text{ tok/s}$ and verify output coherence.
