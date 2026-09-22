# Spec: Paged KV cache, prompt cache, and delta prefill (Rust engine)

## Goal

Enable single-user multi-turn chat to restore persisted KV state and prefill **only delta tokens**, so turn-N TTFT is proportional to the new user message (plus template overhead), not to system prompt + full history. Persist system-prompt KV once under `./prompt_cache/`, persist per-chat KV after every turn under `./chat/<chat_id>/`, and recompute from scratch whenever the stored chain does not match the current model/prompt/token stream.

**Rewritten 2026-09-22** for the pure-Rust + raw-Metal engine (`spec/rust-metal-engine/`). omlx Python touchpoints (`BlockAwarePrefixCache`, `PagedCacheManager`, scheduler admission gates, etc.) are removed; the engine is single-tenant by construction so multi-tenant bypass logic is gone. On-disk layout concepts, chain validation, delta prefill, and acceptance goals carry forward.

## Non-goals / restrictions

- No KV cache quantization (fp16/bf16 KV only; layout tricks like Gemma SWA rings and K=V sharing are allowed).
- No multi-user / multi-tenant scheduling; single resident model, one conversation active at a time.
- No speculative-decode interaction design beyond "MTP snapshots must remain valid or be invalidated cleanly" (owned by `rust-metal-engine` MTP tasks).
- No cross-process concurrent writers; one process owns the on-disk layout.
- No fuzzy matching: exact chain validation or full recompute.
- **No omlx cache subsystems** — this is greenfield Rust; do not port Python cache manager code (inspiration for block/refcount ideas OK per rust-metal-engine DE18, fresh implementation only).
- No mid-token saves; persistence points are turn boundaries (and process shutdown if a turn is idle).
- `reference/` checkouts are read-only.
- No Python in the persistence layer (engine-wide DE15).

## Requirements

### R1 — On-disk layout

- R1.1 System/template prompt KV lives under `./prompt_cache/<prompt_hash>/` with `meta.json` (prompt_hash, model_id, quant_fingerprint, template_version, n_tokens, block list, checksums) and block payloads.
- R1.2 Each chat lives under `./chat/<chat_id>/` with `chat.json` (prompt_hash link, model_id, quant_fingerprint, token id log, n_tokens, status) and KV payloads (blocks / snapshot files).
- R1.3 `prompt_hash = H(model_id, quant_fingerprint, template_version, system_prompt_token_ids)`.
- R1.4 Paths are configurable; defaults are `./prompt_cache` and `./chat` relative to a configured data root (not `/tmp/cache`).
- R1.5 Directories and files are safe against torn writes: metadata written via temp + atomic rename; block payloads checksummed.

### R2 — Prompt cache lifecycle

- R2.1 On first use of a prompt_hash: prefill the system prompt, write `prompt_cache` entry, mark ready.
- R2.2 On later startups: if entry exists and meta matches the running model/quant/template, load it; do not re-prefill.
- R2.3 On meta mismatch (model swapped, quant changed, template bumped, prompt edited): invalidate the entry (delete or tombstone) and recompute.
- R2.4 Prompt cache entries are shared across chats with the same prompt_hash.

### R3 — Chat lifecycle and chain validation

- R3.1 Opening or resuming a chat: validate `chat.json.prompt_hash` equals the active prompt_hash; model_id and quant_fingerprint match the loaded model.
- R3.2 Validate the stored token id log is an exact prefix of the current assembled sequence `S = system ‖ history ‖ new_user_msg` (and that history assembly is deterministic).
- R3.3 Any validation failure → discard chat KV entirely and recompute (full prefill of S after prompt_cache restore if prompt itself is still valid; full prefill of S if prompt entry also invalid).
- R3.4 Edits to an earlier message truncate the log at the edit point; KV above the cut is freed; tokens after the cut are re-prefilled from the last valid checkpoint ≤ cut.
- R3.5 After each completed assistant turn (EOS/stop/tool boundary as configured): append assistant token ids to the log and schedule durable KV persistence (R5).

### R4 — Delta prefill

- R4.1 Restore KV to length L = validated prefix length (from prompt_cache + chat tail, or block-aligned conservative L).
- R4.2 Prefill only `S[L:]`. Never re-prefill `S[:L]` on a validated resume.
- R4.3 v1 persistence may round L down to a 256-token block boundary; the remainder of the partial block (≤255 tokens) is re-prefilled. Exact-length restore is a follow-up (see open questions).
- R4.4 LCP is append-only for unedited multi-turn; divergence only at the tail except on edit (R3.4).
- R4.5 Mismatch between cached KV length and token log length is a hard failure → recompute (never guess).

### R5 — Persistence cadence and admission decoupling

- R5.1 Persist chat KV asynchronously after each completed turn; do not block the generation completion path on fsync (durable-enough: successful enqueue + checksummed write; torn write detected on next load → recompute).
- R5.2 Starting the next turn must **not** wait for the previous turn's disk write. RAM-resident paged blocks serve the next turn; background write continues.
- R5.3 (omlx-specific store-cache backpressure gate removed — engine has no multi-tenant admission path.) Requirement reduces to: **no synchronous persist wait anywhere on the turn-advance path** (asserted by test).
- R5.4 If a background write is interrupted (crash), the next load detects checksum/meta failure and recomputes the affected tail only if chain validation otherwise passes; otherwise full recompute (R3.3).
- R5.5 Process shutdown with an in-flight dirty turn: best-effort flush; if flush fails, chat remains valid at last completed turn.

### R6 — Paged KV behavior

- R6.1 KV is managed in 256-token blocks with a block table per conversation (block size may be revisited; 256 retained for continuity with prior design and p-RoPE/align notes).
- R6.2 Prefix/system blocks are shared (refcounted or copy-on-write); chat tail blocks are chat-private.
- R6.3 Free blocks above an edit cut or on chat delete/evict.
- R6.4 Gemma 4 hybrid layout is preserved end-to-end: sliding-window local layers persist only their ring (window 1024); global layers persist full context; global K=V sharing must round-trip through save/load bit-identically (or documented as recompute-trigger if a layer type cannot round-trip).
- R6.5 Disk tier is the explicit per-turn save to `./chat` (product persistence); opportunistic overflow tiering is optional, not required for v1.

### R7 — Eviction and quotas

- R7.1 `./prompt_cache` and `./chat` have configurable max bytes and LRU (or explicit delete) eviction; default quotas documented.
- R7.2 Evicting a prompt_cache entry that a chat still references does not corrupt the chat: next resume recomputes prompt KV (R2) then chat tail or full chat recompute if tail depended on evicted blocks.
- R7.3 Never evict the active conversation's blocks while a request is in flight.

### R8 — Observability

- R8.1 Log per request: prefix tokens matched, delta tokens prefilled, restore source (RAM/disk/recompute), validation failure reason.
- R8.2 Counters live in the engine's bench/stats JSON (schema `kaplumba-bench/1` extension or sidecar fields) — no parallel metrics system.

### R9 — Correctness invariants

- R9.1 Loading never produces a KV state that disagrees with the token log (validate length + per-block checksums).
- R9.2 Greedy outputs with warm cache must be token-identical to greedy outputs with cold full-prefill for the same S (within documented float-reassociation tolerance — if reassociation differs, document and gate warm-path tests with tolerance or force deterministic kernels).
- R9.3 No path may silently serve a partially validated cache.

## Acceptance criteria

- AC1: Cold start, first message: system prompt computed once, written to `prompt_cache`; second process start with same model/prompt loads it (0 system-prompt prefill tokens).
- AC2: Turn 2+ with no edits: measured prefill tokens == len(new user message) (+ template delta), history not re-prefilled.
- AC3: Kill -9 after turn K; restart; resume chat: validation passes through turn K (or cleanly recomputes a torn tail); greedy continuation matches a never-killed reference run (R9.2 policy applied).
- AC4: Change system prompt text → old chats and prompt_cache entries fail validation → full recompute; no cross-prompt KV reuse.
- AC5: Swap model or quant fingerprint → full recompute; no KV reuse.
- AC6: Edit message i → only tokens after last checkpoint ≤ i are re-prefilled; output matches full-recompute reference.
- AC7: Start turn N+1 while turn N's disk write is still in flight → no added wait on the write (assert turn advance not gated on write completion).
- AC8: Torn/corrupted block file → detected via checksum → recompute path, never garbage generation.
- AC9: Unit tests (`cargo test`): hash/meta validation, prefix log check, block align/truncate, eviction refs, hybrid SWA save/load shapes.
- AC10: Integration test: N-turn chat script asserting prefill token counts and golden greedy outputs (joint with `rust-metal-engine` T5.3/T5.4).
- AC11: Engine test suite passes with persistence enabled and disabled (feature flag or config); no behavior change when persistence off.

## Open questions

- OQ1: Deterministic warm-vs-cold greedy identity (R9.2) — enforce bit-identical kernels or allow documented tolerance? (Blocks AC10 strictness.)
- OQ2: Default quotas for `./prompt_cache` and `./chat` (bytes / chat count)? (When do old chats get deleted automatically, if ever?)
- OQ3: Template version source of truth (hash in model config vs hardcoded string in kaplumba)? (Who bumps the version?)
- OQ4: Exact-length restore vs block-aligned v1 — ship exact in first release or accept ≤255-token tail re-prefill? (plan assumes block-aligned first.)
- OQ5: Where does the data root live by default — `./prompt_cache` relative to CWD, or under `~/.kaplumba/`? (Prior conversation used `./prompt_cache` and `./chat`; confirm CWD vs app dir.)
