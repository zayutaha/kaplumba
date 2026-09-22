# Plan: Paged KV cache, prompt cache, and delta prefill

## Architecture

```
Request (chat turn)
  → assemble S = system ‖ history ‖ new_user_msg  (token ids)
  → ChainValidator
       ├─ active prompt_hash? model_id? quant_fp? template_ver?
       ├─ chat.json token log prefix of S?
       └─ pass → Restore | fail → Recompute
  → Restore
       ├─ prompt_cache: load shared system-prompt blocks (RAM paged pool ← SSD)
       ├─ chat tail: load blocks to L (block-aligned v1)
       └─ Prefill(S[L:]) only
  → Decode turn
  → on turn complete:
       ├─ append assistant ids to chat.json token log
       └─ async PersistChatKV (checksummed blocks → ./chat/<id>/)  // never gates next admission
```

Single-tenant engine path bypasses omlx multi-tenant admission gates (store-cache backpressure, cache-freshness deferral) so persistence is fully decoupled (spec R5).

## File / module touchpoints (omlx reference)

| Area | Path (under `reference/omlx/omlx/`) | Current behavior | Intended change |
|------|--------------------------------------|------------------|-----------------|
| Prefix API | `cache/prefix_cache.py` (`BlockAwarePrefixCache`, `fetch_cache` `:622`, `store_cache` `:750`, `store_exact_prefix` `:1460`) | Opportunistic block-hash prefix reuse | Add prompt_hash/chat chain validation layer; exact or block-aligned restore entry points |
| Paged RAM | `cache/paged_cache.py` (`PagedCacheManager`, block size 256) | Block pool + optional SSD manager hook | Refcount/CoW for shared prompt blocks; free-on-cut; chat-private tail |
| SSD tier | `cache/paged_ssd_cache.py` (`PagedSSDCacheManager`) | Hot bytes tier + disk, default dir `/tmp/cache` | Repoint dirs; support two namespaces (`prompt_cache` vs `chat/<id>`); checksums |
| Factory | `cache/factory.py` | `paged_ssd_cache_dir`, 100GB default cap | Config for data root, quotas, enable single-tenant persistence mode |
| Boundary snapshots | `cache/boundary_snapshot_store.py` | Every 256 tokens during prefill/decode | Persist point for block-aligned checkpoints; keep for MTP arming |
| Admission gates | `scheduler.py` `_schedule_waiting` ~`:10714`, `_should_defer_for_cache_freshness` ~`:8564` | New turn can wait on prior async store | Single-tenant: skip both gates (R5.3) |
| Prefill loop | `scheduler.py` `_do_external_prefill` ~`:3517`, chunk loop ~`:3686` | Chunks full prompt (minus skip_lm_head) | Start at restored L; chunk only suffix |
| Engine core | `engine_core.py` `add_request` ~`:670`, wake ~`:791` | Executor hop | Unchanged except: never wait on persist future |
| Settings | `settings.py` | multi-tenant defaults | Flags: data root, single_tenant_persistence, block_aligned_restore |
| Gemma hybrid | `cache/prefix_cache.py` ~`:1697` (gemma3 KVCache+Rotating mix), model paths for gemma4 | Hybrid supported for gemma3; gemma4 audit needed | Save/load SWA rings + global full; K=V global round-trip |
| Server config | `server.py` / `cli.py` | — | Expose `--data-root`, `--prompt-cache-dir`, `--chat-dir` |

New code (owned by kaplumba or a thin omlx patch layer — decided in tasks):

- `spec` implementation may live as patches against vendored omlx or a sibling package; exact location is T0 in tasks.md (see open risk below).
- Chain validator + `chat.json` / `prompt_cache/meta.json` schema (small, new).
- Persist worker: single background thread/queue, atomic writes.

## Data flow: validation

1. Compute active `prompt_hash` from running model + template + system tokens.
2. Load `chat/<id>/chat.json` if exists.
3. Fail closed unless: `prompt_hash`, `model_id`, `quant_fingerprint` all match AND `token_log` is prefix of `S` AND `n_tokens_kv ≤ len(token_log)` consistency holds.
4. On fail: free chat blocks; if prompt meta also mismatches, free prompt entry; path to full recompute.
5. On pass: L = min(validated_kv_len, block_floor(token_log_len)) for v1.

## Data flow: save

1. After turn complete: enqueue PersistJob(chat_id, block_table snapshot, token_log tail, meta).
2. Worker: write new blocks to temp files → checksum → rename → update `chat.json` (atomic).
3. Completion does not feed back into scheduler admission (R5.2/R5.3).
4. Dirty generations counted for shutdown flush (R5.5).

## Interfaces / formats

### `prompt_cache/<prompt_hash>/meta.json`

```json
{
  "schema": 1,
  "prompt_hash": "…",
  "model_id": "…",
  "quant_fingerprint": "…",
  "template_version": "…",
  "n_tokens": 4096,
  "block_size": 256,
  "blocks": [{"id": "…", "sha256": "…", "bytes": 0}],
  "kv_layout": {"kind": "gemma4_hybrid", "window": 1024, "global_kv_shared": true},
  "created_at": "…"
}
```

### `chat/<chat_id>/chat.json`

```json
{
  "schema": 1,
  "chat_id": "…",
  "prompt_hash": "…",
  "model_id": "…",
  "quant_fingerprint": "…",
  "template_version": "…",
  "token_log": [/* int ids, append-only */],
  "kv_n_tokens": 8192,
  "kv_blocks": ["…"],
  "status": "ready|dirty|invalid",
  "updated_at": "…"
}
```

- `token_log` may be stored as binary int32 sidecar if JSON size is a concern (task-level detail).
- `status=dirty` means persist incomplete → R5.4 on load.

### Restore API (conceptual)

```text
restore(chat_id, S) -> RestoreOk(L, sources) | Recompute(reason)
persist_async(chat_id) -> None   # enqueue only
invalidate(chat_id | prompt_hash) -> None
```

## Sequencing (tasks live in tasks.md)

T0: Decide patch location + config surface (data root flags).
T1: Schema + chain validator (pure, unit-testable, no GPU).
T2: Directory wiring (factory → `./prompt_cache`, `./chat`), atomic write helpers.
T3: Prompt cache compute-once path (miss → prefill → store → hit).
T4: Single-tenant admission bypass for store-cache/freshness gates.
T5: Delta prefill entry (block-aligned L restore → prefill suffix only).
T6: Per-turn async persist worker + chat.json updates.
T7: Crash/torn-write handling + status=dirty.
T8: Edit-truncate / free-above-cut.
T9: Gemma 4 SWA + global KV save/load round-trip tests.
T10: Eviction/quotas.
T11: Observability counters + bench harness assertions (AC1–AC11).
T12: Docs + implementation_notes freeze for handover.

Dependencies: T1–T2 before anything touches disk; T4 before T6 in production path; T5 after T3; T9 parallelizable after T5 once restore works for full-attn-only fixtures.

## Risks and fallbacks

| Risk | Mitigation |
|------|------------|
| Warm vs cold greedy not token-identical (R9.2/AC10) | OQ1; force deterministic order or allow tolerance gate; fallback: golden tests compare distributions / allowlisted diffs |
| omlx store-cache path deeply entangled with multi-tenant code | Feature-flag single-tenant mode; keep old path behind flag (AC11) |
| Gemma4 hybrid restore bugs → silent quality loss | T9 bit-level fixture tests per layer type; mismatch → recompute (fail closed) |
| SSD write bandwidth / latency spikes on turn boundary | Async only; write block-at-a-time; never on request critical path |
| Token log too large as JSON | Binary sidecar (format in T1) |
| Patching vendored omlx vs forking | T0 decision; prefer small patch series + notes in implementation_notes.md |
| Reference checkouts drift | Pin commits in implementation_notes when patching |

## Requirement → task mapping

| Requirement | Tasks |
|-------------|-------|
| R1 layout/atomicity | T1, T2 |
| R2 prompt cache | T3, T10 |
| R3 chat validation | T1, T3, T7, T8 |
| R4 delta prefill | T5, T9 |
| R5 persist + decoupling | T4, T6, T7 |
| R6 paged/hybrid | T5, T9, T10 |
| R7 eviction | T10 |
| R8 observability | T11 |
| R9 invariants | T1, T7, T9, T11 |
| AC1–AC11 | T1–T12 with tests named in tasks.md |
