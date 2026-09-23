# spec/final-same-algorithm-ceiling — Technical Plan

## Architecture & Data Flow
This spec adds measurement instrumentation and small trims to the existing `spec/rust-metal-engine` decode/prefill paths. No new kernels, no new buffers, no new command buffer structure.

- **Decode path**: `gpu_decode.rs:encode_token` (single encoder, 1 CB, 1 commit/wait) — instrument encode vs wait.
- **Prefill path**: `encode_span` (one CB per ≤256-row chunk) — already has `KAPLUMBA_SPAN_PROFILE`; T3.10 K-split already implemented and gated-suite verified.
- **KV cache**: `kv_cache.rs:reserve_row` — add long-context test to exercise rotation at ctx ≥ 1024.
- **RMS kernels**: `fuse.rs/metal` — widen grid or fuse into linear neighbor.
- **K-split policy**: `exl3.rs:gemv_splits` / `sg_splits` — narrow max splits for small-out-tile shapes.

## File/Module Touchpoints

| File | Current Behavior | Change |
|------|------------------|--------|
| `crates/engine/src/gpu_decode.rs` | `encode_token`: one encoder, `commit` + `wait` at end | Add `Instant` timers around encoder build, `commit`+`wait`, and sample/detok; print or return phase split |
| `crates/engine/src/gemma4.rs` | `decode_throughput_slow`: measures total wall time | Consume phase split from decoder, print `cpu_encode_ms`, `gpu_wait_ms`, `host_overhead_ms`, achieved GB/s |
| `crates/engine/src/kv_cache.rs` | `reserve_row`: CPU `copy_within` at ctx ≥ window | No code change — new test exercises it |
| `crates/engine/src/metal_linear.rs` | `encode_linear` etc. — pure encoder passthroughs | Add new test `decode_long_context_slow` |
| `crates/kernels/src/fuse.rs` | `encode_rms`: grid = rows (1 TG per row) | Widen to `rows.div_ceil(RGT)` with RGT≥8, or fuse `rms` into preceding `had_in`/`had_out` |
| `crates/kernels/src/exl3.rs` | `gemv_splits`: TARGET=4096, MAX=16; `sg_splits`: TARGET=320, MAX=8 | Add `MAX_SMALL_OUT_SPLIT=4` for `out_tiles ≤ 256` to cap k/v regression |
| `crates/kernels/src/exl3.metal` | — | No change (policy only) |
| `spec/rust-metal-engine/implementation_notes.md` | Current ceiling notes at T3.10 | Add final ceiling table + 18/200 closure |

## Interfaces & Formats

### Decode phase split output (to stdout from `decode_throughput_slow`)
```
[T4.0] decode phase split: cpu_encode=X.Y ms, gpu_wait=Y.Z ms, host_overhead=Z.W ms
[T4.0] achieved weight BW: 16.028 GB * t/s = XXX.X GB/s (YY% of 273)
```

### Long-context test signature
```rust
#[test]
fn decode_long_context_slow() { /* KAPLUMBA_RUN_SLOW=1, prompt ≥ 1200, 64 steps */ }
```

### Policy change
```rust
fn gemv_splits(out_tiles: usize) -> usize {
    // ... existing ...
    // Add: if out_tiles <= 256 { nsplit = nsplit.min(4) }
}
```

## Sequencing & Dependencies

1. **T3.10 finish** (prefill bench + docs + commit) — independent, already gated-green.
2. **T2.10 phase split** — instrument `encode_token` + `decode_throughput_slow` output.
3. **T2.11 long-context** — new gated test; if drop >10% → T2.11b device-side rotation (out of scope, report only).
4. **T2.12 glue/churn** — rms widen + k/v split cap; depends on T2.10 data (if encode share is small, rms widen lower priority).
5. **T6.1 report** — after all measurements.

## Risks & Fallbacks

| Risk | Fallback |
|------|----------|
| Long-context drop >10% but device-side rotation is complex | Report as known limitation; document max stable ctx |
| k/v split cap hurts other shapes | Only apply to `out_tiles ≤ 256`; verify full gated suite |
| RMS widen breaks ragged-tail masking | Use same masking pattern as existing rms kernel (grid-stride loop) |
| Phase split instrumentation adds overhead | Use `Instant` (cheap); only in slow test path |

## Requirement → Task Mapping

| Req | Task(s) |
|-----|---------|
| 1 (T3.10 acceptance) | T3.10-finish |
| 2 (T2.10 phase split) | T2.10-instrument, T2.10-output |
| 3 (T2.11 long-context) | T2.11-test, T2.11-analyze |
| 4 (T2.12 glue/churn) | T2.12-rms, T2.12-split-cap |
| 5 (T6.1 report) | T6.1-report |