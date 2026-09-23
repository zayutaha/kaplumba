# spec/prefill-scratch-fix — Task Checklist

Legend: `[x]` done · `[ ]` todo · `[!]` blocked
Update immediately after every task; record commit hash when landed.

### T1 — ChunkScratch struct + encode_linear_with_scratch in exl3.rs
- [x] Add `ChunkScratch { xh: Buffer, y: Buffer }` in `exl3.rs` (module-level pub struct; no RefCell)
- [x] Add `encode_linear_with_scratch` that takes `&ChunkScratch` and uses `scratch.xh` / `scratch.y` instead of `with_scratch`
- [x] `encode_gem` reuses external `xh`/`y` via the existing signature (scratch passed down as args)
- [x] Keep existing `encode_linear` for decode (uses shared `with_scratch`)
- [x] Verify: kernels crate tests pass (20/0)
- Note: `ChunkScratch` had to live at module level (struct-in-impl is illegal Rust); constructor `Exl3Gemv::new_chunk_scratch` added because `Exl3Gemv.device` is private.

### T2 — shared scratch gated to rows==1 (decode)
- [x] `with_scratch(in, out, rows, f)`: rows==1 → shared slot via `ensure_scratch`; rows>1 → **transient** arena built for this call, dropped on return (prefill/test paths never grow the shared slot)
- [x] All `with_scratch` call sites updated with explicit `rows` (1 for host-path helpers, `rows` for bench)
- [x] Verify: decode path still works (first decode allocates decode-max scratch) — fast suite 114/0/5

### T3 — encode_span allocates/passes/drops ChunkScratch
- [x] In `gpu_decode.rs:encode_span`: per-chunk `chunk_scratch_sizes(rows)` closure computes max `xh`/`y` over the chunk's linears
- [x] Allocate `ChunkScratch` via `MetalLinear::chunk_scratch(max_in, max_out)` before the layer loop
- [x] q, k, v, o, gate, up, down call `gpu.encode_linear_with_scratch(..., &scratch)` (norms don't use scratch; lm_head stays rows=1 decode path)
- [x] `scratch` is scoped to the chunk iteration → dropped after `cb.wait_until_completed()`
- [x] Verify: full gated suite 114/0/5 (354s), fast 114/0/5, kernels 20/0
- Note: inline eligibility heuristic (`inn%128 && outp%128 && off%4`) is equivalent to `sg_eligible` under the HAD≥128 gate; T3.5 replaces it with the single-source helper anyway.

### T3.5 — sg K-split partial into the per-chunk arena (R1/R4)
- [ ] `exl3.rs`: `ChunkScratch` gains `partial: Buffer`; `new_chunk_scratch(max_in, max_out, max_partial_bytes)` (≥4 bytes)
- [ ] `exl3.rs`: public `Exl3Gemv::chunk_scratch_needs(shape, trellis, rows) -> (xh_elems, y_elems, partial_bytes)` and `sg_partial_bytes(...)`; public `shared_partial_bytes()`
- [ ] `exl3.rs`: `encode_gemm_sg` / `encode_gem` take `ext_partial: Option<&Buffer>`; nsplit>1 uses it when `Some`, else `ensure_partial` fallback (rows=1 gemv + benches/tests keep shared ≤344 KB)
- [ ] `exl3.rs`: `encode_linear_with_scratch` passes `Some(&scratch.partial)`; `encode_linear`/`bench_linear_phases` pass `None`
- [ ] `exl3.rs` tests: `chunk_scratch_needs` policy (k/v@256 = 8388608, gate@256 = 0, rows=1 = 0, lm = 0)
- [ ] `metal_linear.rs`: `chunk_scratch` 3-arg; `chunk_scratch_needs(&LinearHandles, rows)` associated fn
- [ ] `gpu_decode.rs`: `chunk_scratch_sizes` → component-wise max of `MetalLinear::chunk_scratch_needs`; allocate with `max_partial`
- [ ] Verify: kernels 20/0, fast 114/0/5, gated 114/0/5
- Why: 2026-09-23 audit — `ensure_partial(need)` with `need = nsplit×rows_pad×N×4` = **8.4 MB** for k/v at rows=256, permanently grown in shared storage on first prefill (plan.md formerly claimed ≤344 KB — that's the rows=1 gemv figure). Violates R4/"only KV allocated".

### T4 — RSS/scratch/partial regression test
- [ ] Add gated test `prefill_memory_profile_slow` in **`gemma4.rs`** (deviation: needs `Gemma4Model` + `artifact_dir`; tasks.md originally said `metal_linear.rs`)
- [ ] Flow: load artifact → warmup at full cap (decoder + decode scratch + gemv partial sized) → snapshot `(shared_scratch_elems, shared_partial_bytes, rss)` → 8192-token span prefill → snapshot again → assert scratch/partial **exactly unchanged**, RSS growth ≤ 48 MB after dropping KV
- [ ] RSS sampled via `ps -o rss= -p <pid>` (KB; no new deps)
- [ ] Verify: `KAPLUMBA_RUN_SLOW=1 cargo test -p kaplumba-engine --release --lib -- --ignored --nocapture prefill_memory_profile_slow`

### T5 — Throughput verification
- [ ] Run `prefill_throughput_slow` → ≥ 86.8 t/s @512 (if measured ≈86.3: amend R5 with the recorded number — T3.10 landed a known sub-1% shift)
- [ ] Run `decode_throughput_slow` → ≥ 8.39 t/s
- [ ] Full gated suite: 114/0/5 (→ 114/0/6 once T4's `#[ignore]` test lands; update spec acceptance count)

### T6 — Docs + commit
- [ ] `cargo fmt`
- [ ] `spec/rust-metal-engine/tasks.md`: tick T3.10 with commit hash + acceptance-failed note (86.3 < 93.7, ceiling spec on hold)
- [ ] `spec/rust-metal-engine/implementation_notes.md`: T3.10 facts (sg_splits policy, 9-u32 params, MSL uniform-width rule, reduce determinism) + scratch-fix T1–T5 facts + combined-commit deviation
- [ ] Commit(s): specs first (ceiling + scratch-fix), then code `spec/prefill-scratch-fix: T1-T5 prefill scratch arena, no permanent growth (+land verified T3.10 sg K-split; working trees entangled — single code commit)` — never stage `.freebuff/`
