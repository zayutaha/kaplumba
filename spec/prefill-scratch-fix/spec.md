# spec/prefill-scratch-fix

## Goal
Eliminate permanent/shared scratch growth during prefill **and** remove the separate shared/per-chunk scratch concept entirely. GEMM temps (`xh`, `y`, sg/gemv `partial`) live in one fixed decoder arena sized at construction; only the KV cache grows with context. Shrink oversized arena buffers where actual max row width is known (`k_raw`/`k_norm`/`v_raw`).

## Non-goals / restrictions
- **No quality loss** — all token gates hold (greedy identity, argmax, span-vs-loop `max_abs < 0.55`, batched-vs-gemv `max_rel_err < 2e-3`).
- No Python / no MLX.
- Keep KV buffers/layout as-is (f16, separate k and v stores).
- No grow-on-demand production scratch (`Exl3Gemv.scratch` / `Exl3Gemv.partial` fields removed).
- Must not regress prefill throughput (T3.10 K-split still applies).
- Host test/bench hooks (`encode_linear`, `linear_row`, `had_*_pass`, benches) may use a **transient** per-call scratch — never retained on `Exl3Gemv`.

## Requirements
1. **R1**: Production decode (`encode_token`, `rows==1`) and prefill (`encode_span`, `rows>=2`) both use `GpuDecoder::lin_temps: ChunkScratch` via `encode_linear_with_scratch`. One allocation at `GpuDecoder::new`; never reallocated per chunk or per token.
2. **R2**: `Exl3Gemv` has **no** retained `scratch`/`partial` fields. `with_scratch` is host-test-only and always transient. `shared_scratch_elems` / `shared_partial_bytes` / `ensure_partial` are gone.
3. **R3**: `lin_temps` sizes = component-wise max over every layer linear **and** `lm_head` at both `batch_rows` and `rows==1`, via `MetalLinear::chunk_scratch_needs` (which ORs `sg_partial_bytes` and `gemv_partial_bytes`).
4. **R4**: No memory leak — after warmup, 8192-token prefill RSS growth ≤ 48 MB with KV dropped on both sides of the comparison (`prefill_memory_profile_slow`).
5. **R5**: `prefill_throughput_slow` ≥ 86.8 t/s @512 (T3.10 baseline maintained).
6. **R6**: Last-chunk `rows == 1`, non-sg-eligible shapes, and k_eq_v (no `v_proj`) keep working.
7. **R7**: `k_raw`/`k_norm`/`v_raw` sized to `max over layers of (n_kv * head_dim)` (actual max 16×256 = 4096), not `MAX_NKV * MAX_HD` = 8192 (~6.3 MB saved).

## Acceptance criteria
- Fast suite 114/0/6 (`cargo test -p kaplumba-engine --lib -- --test-threads=1`).
- Kernels suite 21/0.
- Full gated suite: `KAPLUMBA_RUN_SLOW=1 cargo test -p kaplumba-engine --release --lib -- --ignored --nocapture --test-threads=1` → **114/0/6**.
- `prefill_memory_profile_slow`: RSS growth ≤ 48 MB after 8192 prefill with KV dropped.
- `prefill_throughput_slow` ≥ 86.8 t/s @512.
- `decode_throughput_slow` ≥ 8.39 t/s.
- No `shared_scratch_elems` / `shared_partial_bytes` / `ensure_partial` / `ensure_scratch` symbols remain in `crates/`.
