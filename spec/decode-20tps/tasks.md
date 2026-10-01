# TASKS: 17-18 tok/s Decode

Each task is independently verifiable, commits on its own, and is **gated on
beating baseline by measurement**. A task that doesn't beat baseline is reverted
and recorded as a negative (AC6).

---

- [x] **T1: Verify Metal concurrent threadgroup scheduling within a single command buffer**
  - Files: `crates/engine/tests/concurrent_tg_feasibility.rs` (new)
  - Intent: Dispatch two compute kernels in the SAME command buffer — one that
    writes an atomic flag after ~1 ms of work, another that spin-waits on the flag
    before writing output. If the output appears, Metal ran them concurrently.
  - **RESULT: ✅ CONCURRENT.** Marker=0xCAFE, spin_count=0, wall=8.2ms.
    Kernel B found the flag already set, meaning Metal ran both threadgroups
    simultaneously within a single command buffer.
  - **Gate PASSED → proceed with T2a (warmstart kernel).**

---

- [ ] **T2a: o_proj warmstart kernel (if T1 passes)**
  - Files: `crates/kernels/src/linear_repacked.metal`, `crates/kernels/src/linear.rs`,
    `crates/engine/src/model.rs`, `crates/engine/tests/live_kernel_bit_exact.rs`
  - Intent: Implement `linear_gemv_ishizuki_k3_colmajor_2col_sk2_warmstart` with
    `device atomic_uint attn_done` spin-wait. Dispatch it BEFORE the attention
    kernel in `forward_buffer`, so Metal's scheduler can overlap them.
    The kernel begins loading weight tiles from DRAM immediately but waits for
    `attn_done` before reading `x[]` (the attention output).
  - Testing:
    - Bit-exact: o_proj output identical to baseline (`live_kernel_bit_exact`).
    - Sabotage: remove the flag wait → output must be wrong (attention not done).
    - A/B: interleaved 8-pair measurement. Must beat 15.85 tok/s baseline.
  - Commit: `spec/decode-20tps: T2a o_proj warmstart kernel`

---

- [ ] **T2b: Dual-queue attention/o_proj overlap (if T1 fails)**
  - Files: `crates/kernels/src/encoder.rs`, `crates/engine/src/model.rs`
  - Intent: Add secondary `MTLCommandQueue` + `MTLEvent`. Dispatch attention on
    secondary queue. Primary queue dispatches o_proj after qkv but before
    attention completes. MTLEvent E1 (qkv_done) gates attention start,
    E2 (attn_done) gates o_proj's input read.
  - Testing:
    - Bit-exact: greedy output identical.
    - Poison test: run ≥ 1000 decode steps, assert bit-identical tokens every step.
    - A/B measurement.
  - Commit: `spec/decode-20tps: T2b dual-queue attention/o_proj overlap`

---

- [ ] **T3: Cross-layer norm/residual chain fusion**
  - Files: `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`,
    `crates/engine/src/model.rs`
  - Intent: New kernel `finalize_post_ffn_and_next_input_norm_transform_sk2` that
    fuses layer L's `finalize_and_post_ffn_norm_scale` + `residual_add` with
    layer L+1's `input_layernorm` + `input_transform`. Called at the
    `forward_buffer` layer boundary for layers 0-58 (layer 59 uses the existing
    finalize for the lm_head transition).
  - Testing:
    - Bit-exact: assert `to_bits()` equality on every output element.
    - Sabotage: corrupt the norm weight load → test must fail.
    - A/B: interleaved measurement.
  - Commit: `spec/decode-20tps: T3 cross-layer norm fusion`

---

- [ ] **T4: Remaining norm optimizations from decode-3bpw-amplification T2-T3**
  - Files: `crates/kernels/src/elements.metal`, `crates/engine/src/elements.rs`
  - Intent: Hoist second-pass `weight`/`residual` loads above the barrier in
    `rms_norm_weighted_tg` (legal: first loop writes only `threadgroup partial[]`).
    Fuse remaining standalone norm dispatches.
  - Testing: Bit-exact, A/B, sabotage. Each sub-fusion commits independently.
  - Commit: `spec/decode-20tps: T4 norm hoist and fusion`

---

- [ ] **T5: Acceptance — A/B measurement and final validation**
  - Files: `spec/decode-20tps/implementation_notes.md`
  - Intent: Measure steady-state tok/s across ≥ 8 interleaved A/B pairs on
    ≥ 64-token generation. Prove AC1-AC6.
  - Commit: `spec/decode-20tps: T5 acceptance`
