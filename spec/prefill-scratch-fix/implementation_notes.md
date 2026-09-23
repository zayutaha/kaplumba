# spec/prefill-scratch-fix — implementation notes

## 2026-09-23 — T7 full scratch elimination (user directive)

**Directive:** "get rid of scratch buffers and save mem wherever you can. Keep KV buffers. No quality loss."

### What landed
- Removed `Exl3Gemv.scratch` / `Exl3Gemv.partial` fields and all grow-on-demand helpers (`ensure_scratch`, `ensure_partial`, `shared_scratch_elems`, `shared_partial_bytes`).
- `with_scratch` is host-test/bench only and always builds a **transient** `Scratch` for that call (drops `x_elems`/`out_elems` fields — never read).
- Production path: `GpuDecoder.lin_temps: ChunkScratch` allocated once in `GpuDecoder::new` via `gpu.chunk_scratch(max_xh, max_y, max_partial)`.
- Sizes: component-wise max of `MetalLinear::chunk_scratch_needs(lin, rows)` over every layer linear (q,k,v,o,gate,up,down; v falls back to k when `k_eq_v`) **and** `lm_head`, at **both** `batch_rows` and `rows==1`. `chunk_scratch_needs` now ORs `Exl3Gemv::gemv_partial_bytes(shape)` so the rows=1 gemv K-split (~344 KB for gate-like shapes) is covered by the same partial buffer as the sg prefill split (~8.4 MB for k/v at rows=256).
- All `encode_linear` / `encode_linear_with_scratch` call sites in `encode_token` and `encode_span` → `encode_linear_with_scratch(..., &self.lin_temps)` (or local `sc`). Per-chunk `chunk_scratch_sizes` + alloc deleted.
- `encode_gemv` now takes `ext_partial: Option<&Buffer>`; production passes arena partial; host hooks (`linear_row`, `gemv_pass`, benches) pass `None` → transient f32 buffer for that call only (`drop(owned)` after `set_buffer`).
- `encode_gemm_sg` same pattern (was already `ext_partial`; fallback changed from shared `ensure_partial` to transient).
- **k/v arena shrink (R7):** `k_raw`/`k_norm`/`v_raw` = `batch_rows * max over layers of (n_kv * head_dim)`. Actual max for this model = 16×256 = 4096 (was `MAX_NKV*MAX_HD` = 8192) → **−6.29 MB** total across 3 buffers at `batch_rows=256`.

### T4 rewrite
- Removed assertions on `shared_scratch_elems` / `shared_partial_bytes` (APIs gone).
- Test is now pure RSS: warmup full-cap (decoder + freelist) → 8192 prefill → baseline RSS → second 8192 prefill with fresh KV → drop → assert growth ≤ 48 MB.

### Verification status (after T7 re-run)
- kernels: **21/0**
- fast engine: **114/0/6** (`cargo test -p kaplumba-engine --lib -- --test-threads=1`)
- symbols `shared_scratch_elems|shared_partial_bytes|ensure_partial|ensure_scratch` in `crates/`: **CLEAN**
- `prefill_memory_profile_slow`: **PASSED** — RSS growth **15072 KB** ≤ 48 MB (192272 → 207344 KB); 8192 prefill 72.4/72.3 t/s
- `prefill_throughput_slow`: **87.6 t/s @512** (≥ 86.8); 81.7 @2048; 72.3 @8192; token-loop 7.67
- `decode_throughput_slow`: **8.39 t/s** forward-only / 8.37 e2e; 134.5 GB/s; ids start 236773
- full gated `-- --ignored`: **6 passed / 0 failed** (494s)

### Gotchas
- Always `--test-threads=1` for gated tests (16 GB each; parallel ≈ 32 GB on 25.8 GB machine).
- Never stage `.freebuff/`.
- Specs amended in-place (`spec.md`/`plan.md`/`tasks.md` rewritten for R1–R7); original per-chunk design in T1–T3.5 marked superseded.

### Memory facts (verified earlier this session)
- Weights disk = 16.028 GB (no host mirror).
- KV f16 k+v: cap=1024 → ~0.92 GB total; cap=8192 → ~1.51 GB total (sliding 0.84 GB fixed + full grows).
- Old shared decode scratch (removed): `(21504, 262144)` ≈ 1.66 MB + partial 344064 B.
- Old per-chunk ChunkScratch @256: ~41.4 MB transient (now permanent in `lin_temps` — same high-water, no freelist step).
