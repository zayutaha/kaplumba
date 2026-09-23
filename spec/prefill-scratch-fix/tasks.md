# spec/prefill-scratch-fix — Task Checklist

Legend: `[x]` done · `[ ]` todo · `[!]` blocked
Update immediately after every task; record commit hash when landed.

### T1 — ChunkScratch struct + encode_linear_with_scratch in exl3.rs
- [x] Add `ChunkScratch { xh, y, partial }` + `encode_linear_with_scratch` (superseded by T7 fixed arena)

### T2 / T3 / T3.5 — intermediate scratch designs (superseded)
- [x] Shared scratch gated to rows==1; per-chunk arena; sg partial into arena — all superseded by T7 (fixed decoder `lin_temps`, no shared scratch fields)

### T4 — RSS regression test
- [x] Gated `prefill_memory_profile_slow` in `gemma4.rs`
- [x] Flow: warmup full cap (decoder + freelist) → 8192 prefill → baseline RSS → second 8192 prefill with KV dropped → assert growth ≤ 48 MB
- [x] Previously PASSED with shared-scratch assertions; rewritten for T7 (RSS-only; shared APIs removed)
- [x] Re-run after T7: **PASSED** — RSS growth **15072 KB** (limit 49152); baseline 192272 KB → after 207344 KB; 8192 prefill 72.4 / 72.3 t/s both passes

### T5 — Throughput verification
- [x] `prefill_throughput_slow` → **87.6 t/s @512** (≥ 86.8 ✓); 81.7 @2048; 72.3 @8192; token-loop 7.67
- [x] `decode_throughput_slow` → **8.39 t/s** forward-only / 8.37 e2e (≥ 8.39 ✓); 134.5 GB/s; ids start 236773
- [x] Full gated suite `-- --ignored --test-threads=1` → **6/0** ignored tests (494s); non-ignored fast suite **114/0/6** already green after T7
- ⚠️ **Always `--test-threads=1`** — each gated test loads ~16 GB; parallel ≈ 32 GB on 25.8 GB machine.

### T6 — Docs + commit
- [x] `cargo fmt`
- [x] `spec/rust-metal-engine/tasks.md`: tick T3.10 with acceptance-failed note (86.8 < 94 / 93.7, ceiling spec on hold)
- [x] `spec/rust-metal-engine/implementation_notes.md`: T3.10 facts + T7/scratch-fix facts + combined-commit deviation
- [x] Specs commit `c11559f`; code commit `6772a8a` — never staged `.freebuff/`

### T7 — Eliminate shared scratch; fixed decoder arena; k_* shrink (user directive 2026-09-23)
- [x] `exl3.rs`: remove `scratch`/`partial` fields, `ensure_scratch`/`ensure_partial`/`shared_scratch_elems`/`shared_partial_bytes`; `with_scratch` always transient; `Scratch` drops size fields
- [x] `exl3.rs`: `encode_gemv` takes `ext_partial`; `gemv_partial_bytes(shape)` public; host hooks without ext allocate transient partial
- [x] `metal_linear.rs`: `chunk_scratch_needs` ORs `gemv_partial_bytes`; docs mark `encode_linear` test-only
- [x] `gpu_decode.rs`: `GpuDecoder.lin_temps: ChunkScratch` sized once in `new` (max over all linears + lm at `batch_rows` and rows=1); all production `encode_linear*` → `encode_linear_with_scratch(&self.lin_temps)`; delete per-chunk `chunk_scratch_sizes` + alloc
- [x] `gpu_decode.rs`: `k_raw`/`k_norm`/`v_raw` = `batch_rows * max(n_kv * head_dim)` (~6.3 MB saved)
- [x] `gemma4.rs`: T4 rewritten RSS-only
- [x] Verify: kernels 21/0, fast 114/0/6
- [x] Re-run T4/T5 after T7: T4 15072 KB ≤ 48 MB; prefill 87.6 @512; decode 8.39; gated ignored 6/0
