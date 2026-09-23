# spec/prefill-scratch-fix — Technical Plan

## Architecture & Data Flow
- **No shared grow-on-demand scratch** on `Exl3Gemv` (fields removed; user directive 2026-09-23: "get rid of scratch buffers").
- **One fixed arena** `GpuDecoder::lin_temps: ChunkScratch { xh, y, partial }`, sized once at `GpuDecoder::new` from component-wise max of `MetalLinear::chunk_scratch_needs(lin, rows)` over every layer linear + `lm_head` at both `batch_rows` and `rows==1`.
- **Decode** (`encode_token`, rows=1) and **prefill** (`encode_span`, rows≥2) both call `gpu.encode_linear_with_scratch(..., &self.lin_temps)` — no per-chunk allocation, no grow-on-demand.
- **Host test/bench hooks** (`encode_linear` without scratch, `linear_row`, `had_*_pass`, `bench_linear_phases`) still call `with_scratch`, which now always builds a **transient** arena for that call only and drops it on return.
- **Partial buffers**: `encode_gemv` / `encode_gemm_sg` take `ext_partial: Option<&Buffer>`; production passes `lin_temps.partial`; host hooks without one allocate a transient f32 buffer per call (never retained).
- **k/v arena shrink**: `k_raw`/`k_norm`/`v_raw` sized `batch_rows * max(n_kv * head_dim)` over layers (4096, not 8192).

## File/Module Touchpoints

| File | Change |
|------|--------|
| `crates/kernels/src/exl3.rs` | Remove `scratch`/`partial` fields + `ensure_scratch`/`ensure_partial`/`shared_*` accessors; `with_scratch` always transient; `Scratch` drops `x_elems`/`out_elems`; add `gemv_partial_bytes`; `encode_gemv` takes `ext_partial`; `new_chunk_scratch` kept as arena builder |
| `crates/engine/src/gpu_decode.rs` | `GpuDecoder.lin_temps: ChunkScratch` sized in `new`; all `encode_linear*` → `encode_linear_with_scratch(&self.lin_temps)`; delete per-chunk `chunk_scratch_sizes` + alloc; `k_*` sized to `max_kv_row` |
| `crates/engine/src/metal_linear.rs` | `chunk_scratch_needs` ORs `gemv_partial_bytes`; docs: `encode_linear` test-only, `encode_linear_with_scratch` production |
| `crates/engine/src/gemma4.rs` | T4 `prefill_memory_profile_slow`: RSS-only assertion (shared scratch APIs removed) |
| `spec/prefill-scratch-fix/{spec,plan,tasks}.md` | Scope amended for full scratch elimination |

## `ChunkScratch` layout (fixed decoder arena)
```rust
struct ChunkScratch {
    xh: Buffer,     // f16, max over linears of (xh_rows(batch) * in_features) and rows=1
    y: Buffer,      // f32, max over linears of (rows * out_features) for batch + rows=1 (lm vocab)
    partial: Buffer, // f32, max( sg_partial_bytes @ batch_rows, gemv_partial_bytes ) — k/v ~8.4 MB at 256
}
```
Sizes via `MetalLinear::chunk_scratch_needs` → `Exl3Gemv::chunk_scratch_needs` + `gemv_partial_bytes`.

## Sequencing
1. [x] ChunkScratch + `encode_linear_with_scratch` (T1).
2. [x] Shared scratch gated to rows==1 (T2) — superseded by full removal.
3. [x] Per-chunk allocate/pass/drop (T3) — superseded by fixed `lin_temps`.
4. [x] sg partial into arena + `chunk_scratch_needs` (T3.5).
5. [x] Remove shared scratch fields; fixed decoder arena; k_* shrink (T7 / user directive).
6. [ ] T5: throughput + full gated suite.
7. [ ] T6: docs + commit.

## Risks & Fallbacks
| Risk | Fallback |
|------|----------|
| `lin_temps` too large (permanent ~41 MB) vs old transient-per-chunk | Fixed ownership avoids freelist high-water steps; RSS measured by T4 |
| Host `encode_linear` transient alloc slower in tests | Tests only; production path never allocates |
| Metal buffer drop during in-flight CB | Arena owned by decoder for process lifetime; host transient kept alive until after `set_buffer` (`drop(owned)` after encode) |

## Requirement → Task Mapping
| Req | Task |
|-----|------|
| R1, R2, R3 | T7: fixed `lin_temps`, remove shared scratch, k_* shrink |
| R4 | T4: `prefill_memory_profile_slow` |
| R5, R6 | T5: prefill + decode throughput + gated suite |
| — | T6: docs + commit |
