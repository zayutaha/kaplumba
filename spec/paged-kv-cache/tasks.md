# Tasks: paged-kv-cache

Update this file after **every** completed task: check the box, add commit hash, note deviations. Do not batch updates until the whole spec is done.

Legend: `[ ]` todo · `[x]` done · `[~]` in progress · `[!]` blocked/diverged

## T0 — Patch location and config surface
- [ ] Decide: patches vs thin wrapper package; record in implementation_notes.md
- [ ] Add config: data_root, prompt_cache_dir, chat_dir, block_aligned_restore, single_tenant_persistence
- Files: settings/factory/cli (omlx or wrapper)
- Test: config loads; defaults resolve to `./prompt_cache`, `./chat` (or OQ5 resolution)
- Commit: `spec/paged-kv-cache: T0 config surface`

## T1 — Schemas + chain validator (no GPU)
- [ ] `prompt_cache/**/meta.json`, `chat/**/chat.json` read/write, schema v1
- [ ] `prompt_hash`, model_id, quant_fp, template_ver checks
- [ ] token_log prefix check; kv_n_tokens consistency; status dirty/invalid
- [ ] Binary token_log sidecar if adopted
- Files: new validator module + schemas
- Test: unit — match, prompt mismatch, model mismatch, quant mismatch, log not prefix, kv_len > log_len, truncated JSON, unknown schema
- Commit: `spec/paged-kv-cache: T1 chain validator`

## T2 — Directory wiring + atomic IO helpers
- [ ] Factory points SSD/persist dirs at data_root
- [ ] atomic_write_json / atomic_write_bytes (temp + fsync + rename)
- [ ] per-block sha256
- Files: factory, new io helpers
- Test: unit — atomic replace, checksum verify, corrupt file detected
- Commit: `spec/paged-kv-cache: T2 dirs + atomic IO`

## T3 — Prompt cache compute-once
- [ ] Miss → prefill system prompt → store blocks + meta → mark ready
- [ ] Hit with matching meta → load, 0 prefill for system tokens
- [ ] Meta mismatch → invalidate + recompute
- Files: scheduler prefill entry, prefix_cache, validator
- Test: unit (meta hit/miss/mismatch) + integration (two in-process startups)
- Commit: `spec/paged-kv-cache: T3 prompt cache`

## T4 — Single-tenant admission bypass
- [ ] Flag skips store-cache backpressure and cache-freshness deferral
- [ ] Multi-tenant path unchanged when flag off
- Files: scheduler `_schedule_waiting`, freshness deferral
- Test: unit (gate skipped when flag on); regression (flag off preserves old behavior); timing test skeleton for AC7
- Commit: `spec/paged-kv-cache: T4 admission decouple`

## T5 — Delta prefill (block-aligned)
- [ ] restore to L = block_floor(validated n); prefill S[L:] only
- [ ] Wire skip_lm_head / external prefill chunking to start at L
- Files: external prefill loop, restore API
- Test: unit — L computation, empty suffix, partial block tail; integration — prefill token count == len(suffix) for turn ≥2
- Commit: `spec/paged-kv-cache: T5 delta prefill`

## T6 — Async per-turn persist
- [ ] On turn complete: append token log, enqueue persist, worker writes blocks + chat.json
- [ ] Next turn admission does not wait on worker (AC7)
- Files: engine turn-complete hook, new persist worker
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
- Files: chat API/assembly, paged free path
- Test: unit + golden greedy vs full recompute (AC6)
- Commit: `spec/paged-kv-cache: T8 edit truncate`

## T9 — Gemma 4 hybrid save/load
- [ ] SWA ring (1024) + global full KV round-trip; K=V global layout preserved
- [ ] Fixture per layer type; mismatch → recompute not garbage
- Files: cache layer-type handlers, gemma4 model paths
- Test: unit per type; integration — hybrid chat save/resume matches cold path under R9.2 policy
- Commit: `spec/paged-kv-cache: T9 gemma4 hybrid round-trip`

## T10 — Eviction / quotas
- [ ] max bytes + LRU/explicit delete for prompt_cache and chat
- [ ] Never evict active in-flight blocks
- Files: factory, store paths
- Test: unit — quota trip, reference safety, resume after prompt eviction (R7.2)
- Commit: `spec/paged-kv-cache: T10 eviction`

## T11 — Observability + acceptance runs
- [ ] Counters: matched prefix tokens, delta prefill tokens, restore source, validation fail reason
- [ ] Execute AC1–AC11 scripted; record results in implementation_notes.md
- Files: observability, bench script
- Test: the acceptance script itself
- Commit: `spec/paged-kv-cache: T11 AC harness`

## T12 — Handover freeze
- [ ] implementation_notes.md complete: decisions, divergences, commands, gotchas
- [ ] All tasks checked with hashes; open questions closed or waived
- Commit: `spec/paged-kv-cache: T12 notes + close`
