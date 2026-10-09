# Task Checklist: Small-Batch High-Bandwidth Prefill (Multi-RHS GEMV)

- [ ] **Task 1: Implement `linear_gemv_mrhs_exl3_k3_mcg` in `linear.metal` (R1)**
  - **Files Touched**: `crates/kernels/src/linear.metal`
  - **Intent**: Implement multi-RHS GEMV kernel supporting $M \in [1, 32]$ query vectors with register accumulators, single weight fetch, and zero threadgroup barriers in the $K$-loop.
  - **Testing**: `cargo check --package kaplumba-kernels`.

- [ ] **Task 2: Implement `encode_gemv_mrhs` wrapper in `crates/kernels/src/linear.rs` (R2)**
  - **Files Touched**: `crates/kernels/src/linear.rs`
  - **Intent**: Compile pipeline state, expose `encode_gemv_mrhs`, set grid geometry to 640 threadgroups, and bind multi-row activation inputs and outputs.
  - **Testing**: `cargo test --package kaplumba-kernels --lib linear`.

- [ ] **Task 3: Unit test probe & sabotage verification in `crates/kernels/tests/linear_mrhs.rs` (AC1, AC2)**
  - **Files Touched**: `crates/kernels/tests/linear_mrhs.rs`
  - **Intent**: Write unit probe asserting `encode_gemv_mrhs` produces parity against single-vector GEMV across $M \in [1, 16]$. Sabotage multi-RHS accumulation (drop row $m > 0$), confirm test failure with diagnostic message, revert sabotage, and confirm pass.
  - **Testing**: `cargo test --package kaplumba-kernels --test linear_mrhs`.

- [ ] **Task 4: Dynamic shape routing in `crates/engine/src/model.rs` (R3)**
  - **Files Touched**: `crates/engine/src/model.rs`
  - **Intent**: In `DecoderLayer::encode_linear` and `Layer::forward_buffer`, route $1 < \text{tokens} \le 32$ to `encode_gemv_mrhs`, while keeping `tokens > 32` on `encode_gemm`. Guard with `KAPLUMBA_DISABLE_MRHS` fallback.
  - **Testing**: `cargo test --package kaplumba-engine --lib model`.

- [ ] **Task 5: End-to-end benchmark and verification on Gemma 4 31B (R4, R5, AC3, AC4)**
  - **Files Touched**: `spec/small-batch-prefill-mrhs/implementation_notes.md`
  - **Intent**: Run `cargo test --workspace --lib` and `multi_turn_chat.rs`. Measure TTFT with release binary on 14-token and 21-token chat turns to verify TTFT drops below 200 ms.
  - **Testing**: `cargo test --workspace --lib` and `scripts/gpuq try -- ./target/release/kaplumba --model ~/.kaplumba/models/Gxma-repacked --chat`.
