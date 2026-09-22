# Implementation notes — paged-kv-cache

Append during implementation (AGENTS.md §5). Newest entries at the bottom of each section. Another agent should be able to take over from this file alone.

## Status

- Spec drafted (pre-implementation). No code yet.
- Open questions OQ1–OQ5 in spec.md still open — resolve before or during T0.

## Decisions

- D1: v1 restore is **block-aligned** (256), re-prefill ≤255 tail tokens (spec R4.3). Exact restore deferred (OQ4).
- D2: Product persistence is **explicit per-turn save** to `./chat/<id>/`, not reliance on opportunistic SSD tiering (prior conversation: option A over omlx default B).
- D3: Admission decoupling (R5.3) is in-scope and required before async persist is safe in production path (T4 before relying on T6).
- D4: EXL3/resident size facts for capacity planning: **~15.5 GB weights @ 3.0 bpw + ~1 GB SWA/KV ≈ 16.5 GB**; decode roofline ~273/15.5 ≈ **17.6 tok/s**. (User corrected an earlier 11.6 GB estimate — do not use bits/8×31B naively; use measured checkpoint size.)

## Divergences from spec/plan

- (none yet)

## Environment facts

- Models: `~/.kaplumba/models` (never in repo tree).
- Reference trees (read-only): `reference/{omlx,llama.cpp,ik_llama.cpp,vllm,PonyExl3}`.
- omlx defaults today: `paged_ssd_cache_dir` example `/tmp/cache` (`cache/factory.py`); block size 256; prefix API `fetch_cache`/`store_cache`/`store_exact_prefix` in `cache/prefix_cache.py`.
- Known admission gates to bypass in single-tenant mode: `scheduler.py` store-cache backpressure ~line 10714; cache-freshness deferral ~line 8564 (line numbers as of shallow clone — re-verify after any pull).
- Boundary snapshots every 256 tokens already exist (`boundary_snapshot_store.py`) — natural checkpoint alignment.

## Dead ends / abandoned approaches

- (none yet)

## Invariants

- I1: Fail closed on any chain/meta/checksum mismatch → recompute, never serve partial trust.
- I2: Next-turn admission must not wait on SSD persist completion.
- I3: No KV quantization in this spec.
- I4: Single-user only; one writer process.

## Commands

- (build/test/bench commands to be recorded from T0 onward)

## Handover checklist for next agent

1. Read AGENTS.md + this spec folder.
2. Resolve OQ1–OQ5 with the user if still open.
3. Continue at first unchecked task in tasks.md; update tasks.md and this file after every task/commit.
