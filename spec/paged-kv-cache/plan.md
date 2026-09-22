# Plan: Paged KV cache, prompt cache, and delta prefill (Rust engine)

Rewritten 2026-09-22 for `spec/rust-metal-engine/` (pure Rust + raw Metal). No omlx modules; persistence is a Rust crate/module alongside the engine.

## Architecture

```
Request (chat turn) — via engine CLI/library
  → assemble S = system ‖ history ‖ new_user_msg  (token ids)
  → ChainValidator (Rust)
       ├─ active prompt_hash? model_id? quant_fp? template_ver?
       ├─ chat.json token log prefix of S?
       └─ pass → Restore | fail → Recompute
  → Restore
       ├─ prompt_cache: load shared system-prompt blocks (RAM paged pool ← disk)
       ├─ chat tail: load blocks to L (block-aligned v1)
       └─ Engine prefill(S[L:]) only   // rust-metal-engine R6.2 / T5.3
  → Decode turn
  → on turn complete:
       ├─ append assistant ids to chat.json token log
       └─ async PersistChatKV (checksummed blocks → ./chat/<id>/)  // never gates next turn
```

Single-tenant by construction: no multi-tenant admission gates to bypass (old R5.3 reduced to "no sync persist wait on turn advance").

## File / module touchpoints (Rust — greenfield)

| Area | Intended location (T0 decides exact crate split) | Responsibility |
|------|--------------------------------------------------|----------------|
| Schemas | `crates/persist/src/schema.rs` | `meta.json`, `chat.json` v1 serde types |
| Chain validator | `crates/persist/src/validate.rs` | Pure, unit-testable, no GPU |
| Atomic IO | `crates/persist/src/io.rs` | temp + fsync + rename; per-block sha256 |
| Paged RAM pool | `crates/engine/src/kv/paged.rs` (or persist crate if shared) | 256-token blocks, block table, refcount/CoW |
| Prompt cache | `crates/persist/src/prompt_cache.rs` | compute-once load/store |
| Chat store | `crates/persist/src/chat_store.rs` | per-turn persist worker, dirty status |
| Restore API | engine prefill entry (`crates/engine`) | `prefill_from(L)` |
| Persist hook | engine turn-complete | enqueue only |
| Hybrid Gemma4 | `crates/engine/src/kv/hybrid.rs` | SWA ring + global full round-trip |
| Config | engine CLI/config | `--data-root`, prompt/chat dirs, quotas, enable flag |
| Observability | bench/stats JSON fields | prefix tokens, delta tokens, restore source, fail reason |

New code is owned by kaplumba; `reference/omlx` cache Python is inspiration-only (block size, boundary every 256, hybrid gemma ideas) — do not port.

## Data flow: validation

1. Compute active `prompt_hash` from running model + template + system tokens.
2. Load `chat/<id>/chat.json` if exists.
3. Fail closed unless: `prompt_hash`, `model_id`, `quant_fingerprint` all match AND `token_log` is prefix of `S` AND `n_tokens_kv ≤ len(token_log)` consistency holds.
4. On fail: free chat blocks; if prompt meta also mismatches, free prompt entry; path to full recompute.
5. On pass: L = min(validated_kv_len, block_floor(token_log_len)) for v1.

## Data flow: save

1. After turn complete: enqueue PersistJob(chat_id, block table snapshot, token_log tail, meta).
2. Worker thread: write new blocks to temp files → checksum → rename → update `chat.json` (atomic).
3. Completion does not feed back into turn-advance (R5.2/R5.3).
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

### Restore API (conceptual — Rust)

```text
restore(chat_id, S) -> RestoreOk(L, sources) | Recompute(reason)
persist_async(chat_id) -> ()   // enqueue only
invalidate(chat_id | prompt_hash) -> ()
```

Engine side (rust-metal-engine R6.2 / T5.3–T5.4):

```text
prefill_from(L, S[L:]) -> ()
on_turn_complete(token_log_tail, block_table) -> ()  // enqueue persist
```

## Sequencing (tasks live in tasks.md)

T0: Decide crate layout + config surface (data root flags).
T1: Schema + chain validator (pure, unit-testable, no GPU).
T2: Directory wiring + atomic write helpers.
T3: Prompt cache compute-once path (miss → prefill → store → hit).
T4: Assert no sync persist on turn-advance (replaces omlx admission-bypass task).
T5: Delta prefill entry (block-aligned L restore → prefill suffix only) — depends on engine T5.3.
T6: Per-turn async persist worker + chat.json updates.
T7: Crash/torn-write handling + status=dirty.
T8: Edit-truncate / free-above-cut.
T9: Gemma 4 SWA + global KV save/load round-trip tests — depends on engine hybrid KV (T5.1).
T10: Eviction/quotas.
T11: Observability counters + bench harness assertions (AC1–AC11).
T12: Docs + implementation_notes freeze for handover.

Dependencies: T1–T2 before anything touches disk; T4 before relying on T6 in production path; T5 after T3 + engine restore hook; T9 after T5 once restore works for full-attn-only fixtures.

## Risks and fallbacks

| Risk | Mitigation |
|------|------------|
| Warm vs cold greedy not token-identical (R9.2/AC10) | OQ1; force deterministic order or allow tolerance gate; fallback: golden tests compare distributions / allowlisted diffs |
| Gemma4 hybrid restore bugs → silent quality loss | T9 bit-level fixture tests per layer type; mismatch → recompute (fail closed) |
| Disk write bandwidth / latency spikes on turn boundary | Async only; write block-at-a-time; never on request critical path |
| Token log too large as JSON | Binary sidecar (format in T1) |
| Crate split churn | T0 decision recorded in notes before T1 code lands |
| Reference checkouts drift | Pin commits (already frozen in rust-metal-engine notes) |

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
