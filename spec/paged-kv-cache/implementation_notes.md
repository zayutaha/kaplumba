# Implementation notes — paged-kv-cache (Rust engine)

Append during implementation (AGENTS.md §5). Newest entries at the bottom of each section. Another agent should be able to take over from this file alone.

Rewritten 2026-09-22 for `spec/rust-metal-engine/` (pure Rust + raw Metal; no omlx, no Python). Prior omlx-specific notes below the migration line are historical.

## Status

- Spec rewritten for Rust engine (pre-implementation). No persistence code yet.
- Open questions OQ1–OQ5 in spec.md still open — resolve before or during T0.
- Depends on `spec/rust-metal-engine` T5.3/T5.4 (restore + persist hooks) for integration tasks T5/T6/T9.

## Decisions

- D1: v1 restore is **block-aligned** (256), re-prefill ≤255 tail tokens (spec R4.3). Exact restore deferred (OQ4).
- D2: Product persistence is **explicit per-turn save** to `./chat/<id>/`, not reliance on opportunistic disk tiering.
- D3: No multi-tenant admission path exists; R5.3 reduces to "no synchronous persist wait on turn advance" (T4 asserts this).
- D4: EXL3/resident size facts for capacity planning: **~15.5 GB weights @ 3.0 bpw + ~1 GB SWA/KV ≈ 16.5 GB**; decode roofline ~273/15.5 ≈ **17.6 tok/s**. Do not use bits/8×31B naively.
- D5: Greenfield Rust — block/refcount/hybrid ideas may be *inspired by* omlx/llama.cpp cache designs but no Python code is ported (engine DE18/DE15).
- D6: On-disk JSON schemas (meta.json / chat.json v1) unchanged from prior design — formats already reviewed; keep `schema: 1` until a breaking change forces a bump.

## Divergences from spec/plan

- DV-P1: Prior plan’s omlx touchpoints (prefix_cache.py, paged_cache.py, scheduler gates ~10714/~8564, factory.py, boundary_snapshot_store.py) **removed** — engine has no equivalent Python modules. Boundary-every-256 idea retained as design note only.
- DV-P2: `PONYEXL3_MEM_LIMIT_GB` / MLX working-set notes are reference-only for the engine allocator (see rust-metal-engine notes), not for this crate.

## Environment facts

- Models: `~/.kaplumba/models` (never in repo tree).
- Reference trees (read-only): `reference/{omlx,llama.cpp,ik_llama.cpp,vllm,PonyExl3}` — pins frozen in `spec/rust-metal-engine/implementation_notes.md`.
- Historical omlx defaults (inspiration only): block size 256; `paged_ssd_cache_dir` example `/tmp/cache` (we use `./chat` / `./prompt_cache` under data_root instead).

## Dead ends / abandoned approaches

- (none yet for Rust port)
- Historical: relying on opportunistic SSD tiering alone as product persistence (prior conversation chose explicit per-turn save).

## Invariants

- I1: Fail closed on any chain/meta/checksum mismatch → recompute, never serve partial trust.
- I2: Next-turn advance must not wait on disk persist completion.
- I3: No KV quantization in this spec.
- I4: Single-user only; one writer process.
- I5: No Python in persistence layer (engine-wide).

## Commands

- (build/test/bench commands to be recorded from T0 onward — `cargo test -p …`)

## Handover checklist for next agent

1. Read AGENTS.md + this spec folder + `spec/rust-metal-engine/`.
2. Resolve OQ1–OQ5 with the user if still open.
3. Continue at first unchecked task in tasks.md; update tasks.md and this file after every task/commit.

---

## Migration appendix (historical omlx notes — do not implement)

Prior decisions D1–D4 and environment facts about omlx cache modules/admission gate line numbers are superseded by DV-P1 above; retained in git history (`git log -p spec/paged-kv-cache/`) if needed for block-size rationale.
