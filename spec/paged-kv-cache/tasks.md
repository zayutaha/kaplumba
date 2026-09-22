# Tasks: paged-kv-cache (Rust engine)

Rewritten 2026-09-22 for `spec/rust-metal-engine/` — no omlx patch tasks. Update this file after **every** completed task: check the box, add commit hash, note deviations.

Legend: `[ ]` todo · `[x]` done · `[~]` in progress · `[!]` blocked/diverged

## T0 — Crate layout and config surface
- [ ] Decide: single `persist` crate vs module inside engine crate; record in implementation_notes.md
- [ ] Add config: data_root, prompt_cache_dir, chat_dir, block_aligned_restore, persistence_enabled
- Files: engine CLI/config + new crate skeleton
- Test: config loads; defaults resolve to `./prompt_cache`, `./chat` (or OQ5 resolution)
- Commit: `spec/paged-kv-cache: T0 config surface`

## T1 — Schemas + chain validator (no GPU)
- [ ] `prompt_cache/**/meta.json`, `chat/**/chat.json` read/write, schema v1 (serde)
- [ ] `prompt_hash`, model_id, quant_fp, template_ver checks
- [ ] token_log prefix check; kv_n_tokens consistency; status dirty/invalid
- [ ] Binary token_log sidecar if adopted
- Files: `crates/persist/src/{schema,validate}.rs` (or agreed path)
- Test: unit — match, prompt mismatch, model mismatch, quant mismatch, log not prefix, kv_len > log_len, truncated JSON, unknown schema
- Commit: `spec/paged-kv-cache: T1 chain validator`

## T2 — Directory wiring + atomic IO helpers
- [ ] Config points persist dirs at data_root
- [ ] atomic_write_json / atomic_write_bytes (temp + fsync + rename)
- [ ] per-block sha256
- Files: `crates/persist/src/io.rs`
- Test: unit — atomic replace, checksum verify, corrupt file detected
- Commit: `spec/paged-kv-cache: T2 dirs + atomic IO`

## T3 — Prompt cache compute-once
- [ ] Miss → prefill system prompt → store blocks + meta → mark ready
- [ ] Hit with matching meta → load, 0 prefill for system tokens
- [ ] Meta mismatch → invalidate + recompute
- Files: prompt_cache + engine prefill entry
- Test: unit (meta hit/miss/mismatch) + integration (two in-process startups)
- Commit: `spec/paged-kv-cache: T3 prompt cache`

## T4 — No sync persist on turn-advance
- [ ] Assert/code-review: turn N+1 advance never waits on persist worker (R5.3)
- [ ] Persistence-disabled path unchanged when flag off
- Files: engine turn loop, persist worker handshake
- Test: unit (gate skipped / no wait); timing test skeleton for AC7
- Commit: `spec/paged-kv-cache: T4 admission decouple`

## T5 — Delta prefill (block-aligned)
- [ ] restore to L = block_floor(validated n); prefill S[L:] only
- [ ] Wire engine `prefill_from(L)` to start chunking at L
- Files: engine prefill entry (rust-metal-engine T5.3), restore API
- Test: unit — L computation, empty suffix, partial block tail; integration — prefill token count == len(suffix) for turn ≥2
- Commit: `spec/paged-kv-cache: T5 delta prefill`

## T6 — Async per-turn persist
- [ ] On turn complete: append token log, enqueue persist, worker writes blocks + chat.json
- [ ] Next turn does not wait on worker (AC7)
- Files: engine turn-complete hook (rust-metal-engine T5.4), new persist worker
- Test: unit — queue/ordering/shutdown; integration — AC7 (start next turn while write sleeps); crash mid-write simulated
- Commit: `spec/paged-kv-cache: T6 async persist`

## T7 — Torn write / dirty status
- [ ] status=dirty on enqueue; ready after checksummed commit
- [ ] Load dirty/failed → recompute tail or full per R5.4
- Files: validator, persist worker, load path
- Test: unit — kill between temp and rename; checksum mismatch; AC3-style resume
- Commit: `spec/paged-kv-cache: T7 dirty/crash handling`

## T8 — Edit truncate
- [ ] Edit at message i → truncate log, free blocks above last checkpoint ≤ i, re-prefill suffix
- Files: chat assembly, paged free path
- Test: unit + golden greedy vs full recompute (AC6)
- Commit: `spec/paged-kv-cache: T8 edit truncate`

## T9 — Gemma 4 hybrid save/load
- [ ] SWA ring (1024) + global full KV round-trip; K=V global layout preserved
- [ ] Fixture per layer type; mismatch → recompute not garbage
- Files: `crates/engine/src/kv/hybrid.rs`, persist handlers
- Test: unit per type; integration — hybrid chat save/resume matches cold path under R9.2 policy
- Commit: `spec/paged-kv-cache: T9 gemma4 hybrid round-trip`

## T10 — Eviction / quotas
- [ ] max bytes + LRU/explicit delete for prompt_cache and chat
- [ ] Never evict active in-flight blocks
- Files: persist store paths
- Test: unit — quota trip, reference safety, resume after prompt eviction (R7.2)
- Commit: `spec/paged-kv-cache: T10 eviction`

## T11 — Observability + acceptance runs
- [ ] Counters: matched prefix tokens, delta prefill tokens, restore source, validation fail reason → bench/stats JSON
- [ ] Execute AC1–AC11 scripted; record results in implementation_notes.md
- Files: observability, acceptance script (Rust)
- Test: the acceptance script itself
- Commit: `spec/paged-kv-cache: T11 AC harness`

## T12 — Handover freeze
- [ ] implementation_notes.md complete: decisions, divergences, commands, gotchas
- [ ] All tasks checked with hashes; open questions closed or waived
- Commit: `spec/paged-kv-cache: T12 notes + close`
