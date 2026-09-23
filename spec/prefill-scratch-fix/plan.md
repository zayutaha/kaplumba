# spec/prefill-scratch-fix — Technical Plan

## Architecture & Data Flow
- **Decode path** (`encode_token`, `rows==1`): uses existing `Exl3Gemv.scratch` (persistent, grows to decode max, ~1 MB).
- **Prefill path** (`encode_span`, `rows>=2`): `encode_span` allocates a `ChunkScratch` per chunk (size = max `xh`/`y` needed by any linear in that chunk), passes it through `encode_linear` → `encode_gem` → `encode_gemm_sg`, drops after `cb.wait_until_completed()`.

## File/Module Touchpoints

| File | Change |
|------|--------|
| `crates/kernels/src/exl3.rs` | New `ChunkScratch { xh, y, partial }`; `Exl3Gemv` adds `encode_linear_with_scratch` taking `&ChunkScratch`; `encode_gemm_sg`/`encode_gem` take `Option<&Buffer>` external partial; `chunk_scratch_needs` + `shared_partial_bytes` accessors; `ensure_scratch` only grows for `rows==1` |
| `crates/engine/src/gpu_decode.rs` | `encode_span`: allocate `ChunkScratch` before chunk loop via `chunk_scratch_needs` maxima, pass to `gpu.encode_linear_with_scratch`, dropped at end of chunk iteration after `cb.wait_until_completed()` |
| `crates/engine/src/metal_linear.rs` | `chunk_scratch(max_in, max_out, max_partial)` builder + `chunk_scratch_needs` passthrough |
| `crates/engine/src/gemma4.rs` | T4 gated test `prefill_memory_profile_slow` |

## `ChunkScratch` layout (per chunk)
```rust
struct ChunkScratch {
    xh: Buffer,     // half, size = max over linears of (xh_rows * in_features)
    y: Buffer,      // f32,  size = max over linears of (rows * out_features)
    partial: Buffer, // f32, size = max over linears of sg K-split need
                     //       (nsplit * rows_pad * out_features * 4);
                     //       ≥4 bytes even when no shape splits
}
```
Sizes computed once per chunk via `Exl3Gemv::chunk_scratch_needs(shape, trellis, rows) → (xh_elems, y_elems, partial_bytes)`, which single-sources `xh_rows` / `sg_eligible` / `sg_splits` (replaces the duplicated inline heuristic formerly in `gpu_decode.rs`).

## Interfaces

### `Exl3Gemv::encode_linear` (decode, unchanged)
```rust
pub fn encode_linear(&self, enc, x, trellis, suh, svh, out, shape, mode, rows) -> Result
```

### `Exl3Gemv::encode_linear_with_scratch` (prefill)
```rust
pub fn encode_linear_with_scratch(&self, enc, x, trellis, suh, svh, out, shape, mode, rows, scratch: &ChunkScratch) -> Result
```

### `MetalLinear` builds `ChunkScratch`:
```rust
fn chunk_scratch(&self, max_in_elems: usize, max_out_elems: usize, max_partial_bytes: usize) -> ChunkScratch
fn chunk_scratch_needs(lin: &LinearHandles, rows: usize) -> (usize, usize, usize)  // associated
```
`encode_span` scans the chunk's layer linears once per chunk (shapes known; trellis offsets via `LinearHandles::trellis()`), takes the component-wise max.

## Sequencing
1. Add `ChunkScratch` + `encode_linear_with_scratch` in `exl3.rs`.
2. Modify `ensure_scratch` to only grow for `rows==1`.
3. Wire `encode_span` to allocate/pass/drop `ChunkScratch`.
4. Verify: prefill RSS profile, prefill throughput, decode throughput, gated suite.

## Risks & Fallbacks
| Risk | Fallback |
|------|----------|
| ChunkScratch allocation too slow per chunk | One allocation per chunk (~3 buffers); sizes precomputed by scanning layer linears |
| Metal buffer drop during `wait` causes use-after-free | Drop after `wait_until_completed()` (GPU done); Metal retains for in-flight CB |
| Shared sg partial still grows during prefill | **Fixed (T3.5):** partial moved into ChunkScratch; `encode_gemm_sg` takes `Option<&Buffer>` (None only for rows=1 gemv/tests → shared ≤344 KB). Measured need ≤ 8.4 MB at rows=256 (k/v, nsplit×rows_pad×N×4) — plan previously claimed ≤344 KB, that was the gemv figure |

## Requirement → Task Mapping
| Req | Task |
|-----|------|
| R1, R2 | T1: ChunkScratch + encode_linear_with_scratch |
| R2 | T2: ensure_scratch only grows for rows==1 |
| R3 | T3: encode_span allocates/passes/drops ChunkScratch |
| R4 | T4: RSS verification test |
| R5 | T5: prefill + decode throughput verification |