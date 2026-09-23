# spec/final-same-algorithm-ceiling — Task Checklist

### T3.10-finish — prefill small-N K-split acceptance
- [ ] Run `KAPLUMBA_RUN_SLOW=1 cargo test -p kaplumba-engine --release --lib -- --ignored --nocapture prefill_throughput_slow` → verify ≥ 93.7 t/s at 512 tokens
- [ ] Update `spec/rust-metal-engine/tasks.md` T3.10 with result
- [ ] Update `spec/rust-metal-engine/implementation_notes.md` with prefill numbers
- [ ] `cargo fmt` → commit `spec/rust-metal-engine: T3.10 sg small-N K-split` (amend if T3.10 already partially committed; otherwise new)

### T2.10-instrument — decode phase split timing in encode_token
- [ ] In `gpu_decode.rs:encode_token`: wrap encoder build (`new_compute_command_encoder` to `end_encoding`) in `Instant`; wrap `commit`+`wait_until_completed` in `Instant`
- [ ] Return tuple `(cpu_encode_ms, gpu_wait_ms)` from `encode_token` (or store in `GpuDecoder` for `forward_cached` to read)
- [ ] In `gemma4.rs:decode_throughput_slow`: capture the three phases (add `host_overhead_ms` as `wall - cpu_encode - gpu_wait`), print formatted line
- [ ] Print achieved GB/s: `16.028 * tokens_per_sec`
- [ ] Fast suite green; gated suite 114/0/5

### T2.10-output — verify phase split numbers
- [ ] Run `decode_throughput_slow` with instrumentation
- [ ] Record `cpu_encode_ms`, `gpu_wait_ms`, `host_overhead_ms` in implementation_notes
- [ ] If `cpu_encode_ms > 5`: note pipeline opportunity for future; else note "encode share small"

### T2.11-test — long-context decode test
- [ ] Add `decode_long_context_slow` in `metal_linear.rs` (or `gemma4.rs`): prompt `vec![2; 1200]` (or random tokens), 64 steps, `KAPLUMBA_RUN_SLOW=1`
- [ ] Measure t/s; compare to short-context baseline (prompt `"2"`, 64 steps)
- [ ] Print ratio and absolute t/s
- [ ] Assert drop ≤ 10% (if >10%, test fails with diagnostic — fix in T2.11b or report)

### T2.11-analyze — long-context result
- [ ] If drop ≤ 10%: record "stable" in notes
- [ ] If drop > 10%: analyze (rotation memmove + KV reads); document in notes as known limitation; do NOT implement device-side rotation (out of scope)

### T2.12-rms — widen/fuse single-TG rmsnorms
- [ ] Option A (widen): change `encode_rms` grid from `rows` (1 TG) to `rows.div_ceil(RGT)` with `RGT=8` (matching `encode_gemm`), same masking pattern
- [ ] Option B (fuse): fuse `rms` into preceding `had_in` or following `had_out` (more complex, try A first)
- [ ] Must preserve bit-exactness vs CPU reference (existing `gpu_had_in_matches_cpu_reference` etc. gates)
- [ ] Verify: fast suite, gated 114/0/5, `decode_throughput_slow` ≥ prior

### T2.12-split-cap — cap k/v over-split
- [ ] In `exl3.rs:gemv_splits`: add `const MAX_SMALL_OUT_SPLIT: usize = 4;` and `if shape.out_tiles <= 256 { nsplit = nsplit.min(MAX_SMALL_OUT_SPLIT) }`
- [ ] In `sg_splits`: similar cap for `out_tiles <= 256` (but note sg already rows-aware; verify k/v at rows=256/512/8192)
- [ ] Update policy tests `gemv_splits_policy_covers_shapes` and `sg_splits_policy_covers_shapes` with new expectations
- [ ] Verify: k/v isolated bench ≥ 100 Gw/s, gated suite 114/0/5, `decode_throughput_slow` ≥ 8.39 t/s

### T6.1-report — final ceiling documentation
- [ ] Add section to `implementation_notes.md`:
  - Measured ceilings table (decode 9–12 t/s / 53–70% BW; prefill 100–110 t/s / 6.3 TFLOP/s wall)
  - llama.cpp/omlx comparison citations (from research)
  - 18/200 marked unreachable with gate commit hashes
- [ ] Update `spec/rust-metal-engine/tasks.md` final status (all tasks [x] or [~] with reason)
- [ ] `cargo fmt` → commit `spec/final-same-algorithm-ceiling: T6.1 final ceiling report`