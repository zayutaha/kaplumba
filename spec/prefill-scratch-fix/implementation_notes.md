# spec/prefill-scratch-fix — HANDOVER (for next model)

**Status: COMPLETE.** All `tasks.md` items `[x]`. Code + docs committed. Do not re-implement this spec.

Read in order: `AGENTS.md` → this file → `spec.md` → `tasks.md`. Broader project: `spec/rust-metal-engine/{spec,plan,tasks}.md`. Held follow-up: `spec/final-same-algorithm-ceiling/`.

---

## 1. What this spec did (outcome)

User directive 2026-09-23: **"get rid of scratch buffers and save mem wherever you can. Keep KV buffers. No quality loss."**

Original goal (R1–R6, first draft of `spec.md`): stop shared `xh`/`y`/partial growing during prefill so only KV grows with context.

**Amended goal (what actually landed as T7 / R1–R7):** eliminate the shared/per-chunk scratch *concept* entirely.

- Production decode + prefill use **one fixed arena** `GpuDecoder::lin_temps: ChunkScratch` (`xh`, `y`, `partial`), allocated once in `GpuDecoder::new`.
- `Exl3Gemv` has **no** retained `scratch`/`partial` fields. Helpers `ensure_scratch`, `ensure_partial`, `shared_scratch_elems`, `shared_partial_bytes` are **gone** (grep `crates/` = CLEAN).
- Host test/bench hooks (`encode_linear` without temps, `linear_row`, `had_*_pass`, benches) use a **transient** per-call scratch only — never stored on `self`.
- `k_raw` / `k_norm` / `v_raw` sized to actual `max(n_kv * head_dim)` = **4096** (was `MAX_NKV*MAX_HD` = 8192) → **−6.29 MB** at `batch_rows=256`.
- **KV cache layout untouched** (f16, separate k and v stores). Weights still 16.028 GB, no host mirror.

Intermediate designs in T1–T3.5 (per-chunk `ChunkScratch`, shared scratch gated to rows=1, sg partial in per-chunk arena) were implemented then **superseded by T7**. `tasks.md` marks them done-but-superseded.

---

## 2. Commits (do not redo)

| Hash | Message |
|------|---------|
| `a45be45` | specs first: prefill-scratch-fix + final-same-algorithm-ceiling (original R1–R6 drafts) |
| `c11559f` | amend scratch-fix to R1–R7 full elimination + T3.10/T7 notes in rust-metal-engine |
| `6772a8a` | **code**: T1–T7 fixed decoder `lin_temps`, no shared scratch, k_* shrink (+land T3.10 sg K-split) |
| `c4c09d3` | record commit hashes in tasks.md |

Working tree: only untracked `.freebuff/` — **never stage it**.

`spec/rust-metal-engine` T3.10 was **entangled** with this code commit (same working tree). T3.10 acceptance note: measured **87.6 t/s @512** after land — **not** the ≥94 / ceiling 93.7 target. Ceiling spec stays **on hold**.

---

## 3. Architecture after T7 (what another model must not break)

```
GpuDecoder::new(model, gpu, cache_cap)
  ├─ batch_rows = min(cache_cap, 256, sliding_window)
  ├─ lin_temps = gpu.chunk_scratch(max_xh, max_y, max_partial)
  │    max over: every layer linear (q,k,v,o,gate,up,down; v→k if k_eq_v)
  │              + lm_head
  │    × rows in {batch_rows, 1}
  │    via MetalLinear::chunk_scratch_needs (ORs sg_partial_bytes + gemv_partial_bytes)
  └─ k_raw/k_norm/v_raw = half_buf(batch_rows * max(n_kv*head_dim))  // 4096 not 8192

encode_token (rows=1)  ──┐
encode_span (rows=m)   ──┴→ gpu.encode_linear_with_scratch(..., &self.lin_temps)
                             → Exl3Gemv::encode_linear_with_scratch
                             → encode_had_in / encode_gem(ext_partial=lin_temps.partial) / encode_had_out

Host hooks only: Exl3Gemv::with_scratch(...) → transient Scratch, dropped on return
                 encode_gemv/encode_gemm_sg with ext_partial=None → transient f32 partial
```

Key files:
- `crates/kernels/src/exl3.rs` — `ChunkScratch`, `encode_linear_with_scratch`, `encode_gem`/`encode_gemv`/`encode_gemm_sg`, `chunk_scratch_needs`, `gemv_partial_bytes`, transient `with_scratch`
- `crates/kernels/src/exl3.metal` — T3.10 sg K-split + reduce (landed in same code commit)
- `crates/engine/src/gpu_decode.rs` — `GpuDecoder`, `lin_temps`, `encode_token`, `encode_span`, k_* sizing
- `crates/engine/src/metal_linear.rs` — `chunk_scratch` / `chunk_scratch_needs`, production `encode_linear_with_scratch`, test-only `encode_linear`
- `crates/engine/src/gemma4.rs` — gated `prefill_memory_profile_slow` (RSS-only now)

---

## 4. Acceptance already proven (2026-09-23, release, M4 Pro, `--test-threads=1`)

| Check | Result |
|-------|--------|
| kernels `cargo test -p kaplumba-kernels` | **21/0** |
| fast `cargo test -p kaplumba-engine --lib -- --test-threads=1` | **114/0/6** |
| gated `KAPLUMBA_RUN_SLOW=1 ... -- --ignored --nocapture --test-threads=1` | **6/0** (494s) |
| `prefill_memory_profile_slow` | RSS growth **15072 KB** ≤ 48 MB (192272 → 207344) |
| `prefill_throughput_slow` | **87.6 / 81.7 / 72.3 t/s** @512/2048/8192 (token-loop 7.67) |
| `decode_throughput_slow` | **8.39** fwd / **8.37** e2e t/s; 134.5 GB/s; ids start 236773 |
| grep forbidden symbols in `crates/` | CLEAN |

Token gates (unchanged by T7): greedy fixture argmax 236773; span-vs-loop `max_abs < 0.55`; batched-vs-gemv `max_rel_err < 2e-3`.

### Exact commands

```bash
cargo fmt
cargo test -p kaplumba-kernels
cargo test -p kaplumba-engine --lib -- --test-threads=1

# gated — ALWAYS --test-threads=1 (each loads ~16 GB; parallel ≈ 32 GB on 25.8 GB machine)
export KAPLUMBA_RUN_SLOW=1
cargo test -p kaplumba-engine --release --lib -- --ignored --nocapture --test-threads=1

# single gated tests
cargo test -p kaplumba-engine --release --lib -- --ignored --nocapture --test-threads=1 prefill_memory_profile_slow
cargo test -p kaplumba-engine --release --lib -- --ignored --nocapture --test-threads=1 prefill_throughput_slow
cargo test -p kaplumba-engine --release --lib -- --ignored --nocapture --test-threads=1 decode_throughput_slow
```

CLI smoke (after this work):

```bash
cargo run --release -p kaplumba-cli -- generate --prompt "The" --max-tokens 32 --temp 0
```

---

## 5. Standing constraints (from AGENTS.md + user)

- No Python/MLX in product; no multi-tenant; no KV quantization; **no quality loss** (verify-gated only).
- Hardware: Apple M4 Pro, 20 GPU cores, 273 GB/s; machine **~25.8 GB RAM** total (not 24 — measurements used 25.77).
- Model: Gemma 4 31B-class dense, EXL3 ~3 bpw, **16.028 GB** weights on disk (`~/.kaplumba/models`), no host mirror.
- One resident model; single-user chat; `reference/` read-only; never stage `.freebuff/`.
- Optimize awareness order: TTFT, tg/s, pp/s, memory — always state which metric a change targets.
- Spec workflow: clarify → `spec.md` → `plan.md` → `tasks.md` → small tasks, test, update tasks+notes, commit per task (`spec/<name>: T<id> ...`).
- **One LLM session per working tree at a time** (AGENTS §6).

### Memory facts (verified this session — do not re-derive casually)

- Weights: 16.028 GB disk; loader keeps no full host mirror (`load.rs`).
- KV f16 k+v: sliding 50×window1024 ≈ **0.84 GB fixed**; full 10 layers grow: cap1024 total ~0.92 GB; cap8192 total ~1.51 GB. User: **keep KV buffers**.
- `lin_temps` @batch_rows=256 ≈ permanent ~41 MB (y ~22 + xh ~11 + partial ~8.4) — same high-water as old per-chunk arena, but **owned**, no freelist step.
- Activity Monitor ~16.4–16.6 GB ≈ weights + small overhead; ~30 GB incidents = two gated tests each loading 16 GB in parallel.

---

## 6. What to do next (this spec is done — pick up elsewhere)

### A. `spec/final-same-algorithm-ceiling` — **ON HOLD** (do not resume without user OK)

- T3.10-finish: ceiling wants **≥93.7 t/s @512**; measured **87.6**. Acceptance **failed**; recorded in `rust-metal-engine/tasks.md` T3.10.
- Remaining unchecked in that spec: T2.10 decode phase instrumentation, T2.11 long-context decode, T2.12 rms widen / k-v split cap, T6.1 ceiling report.
- Resume only if user lifts hold or asks for ceiling work.

### B. `spec/rust-metal-engine` — main unfinished spec

Open items include (see its `tasks.md` for full list): T0.2 full reference matrix + retire `scripts/bench.py`; T2.2 working-set / TTFT warm-up; T2.4 quality gate; T4.8 spec quality suite; later phases (paged KV, etc.).

First unchecked / high-value candidates after scratch-fix:
1. User-directed next lever (ask them — last priorities were decode → 18 t/s conflict, TTFT T2.2, quality).
2. Do **not** re-open scratch allocation — T7 is the invariant.

### C. Handover ritual for the next model

1. Read this file + `AGENTS.md` + `spec/rust-metal-engine/tasks.md`.
2. `git log --oneline -8` · `git status` (expect clean except `.freebuff/`).
3. Re-run fast suite + kernels before any edit.
4. Interview user (AGENTS §2) before any non-trivial new spec.
5. Append decisions to **this** file (`implementation_notes.md`) as work proceeds — never only at the end.
6. Update `tasks.md` immediately after every task with status + commit hash.

---

## 7. Decision log / dead ends (history)

| Decision | Why |
|----------|-----|
| Per-chunk `ChunkScratch` (T1–T3) then replaced by fixed `lin_temps` (T7) | User rejected *any* separate scratch concept; freelist retained ~41 MB after free |
| Shared `Exl3Gemv.scratch/partial` removed entirely | Grow-on-demand violated "only KV grows"; R4 |
| Host hooks keep transient scratch | Tests/benches need staging without production RSS cost |
| T4 asserts RSS only (not shared_scratch_elems) | Those APIs deleted in T7 |
| k_* shrink to 4096 | Actual max `n_kv*head_dim`; 8192 was over-alloc |
| T3.10 acceptance re-stated 87.6 not 94 | Honest record; ceiling 93.7 not met |
| Combined code commit with T3.10 | Working trees entangled (exl3.rs/metal) |

**Abandoned:** shared decode scratch grow-to-decode-max; per-chunk alloc/drop; claiming T3.10 +8% / ceiling without re-measure.

---

## 8. One-line summary for the next model

> `spec/prefill-scratch-fix` is **done and committed** (`6772a8a`): production has **zero retained scratch** — one fixed `GpuDecoder::lin_temps` arena, k_* shrunk, KV unchanged; all gates green (114+21 fast, 6 gated, prefill 87.6, decode 8.39, RSS +15 MB). Next work is **not** this spec: ask the user, or continue `spec/rust-metal-engine` (ceiling spec stays on hold at 87.6 < 93.7).
