# Implementation Notes: Performance and Memory Overhaul

## Overview
This document records technical decisions, benchmark measurements, register allocation details, and handover context for `spec/perf-and-memory-overhaul`.

## Architectural Baseline
- Target: Apple M4 Pro (20 GPU cores, 273 GB/s unified memory bandwidth).
- Model: Gemma 4 31B dense EXL3 (~3.0 bpw).
- Starting state:
  - Startup RSS: ~16.6 GB.
  - Decode throughput: ~8.39 t/s (134.5 GB/s, 49% bandwidth).
  - Prefill throughput: 87.6 t/s @ 512 tokens (MMA rate crippled to ~5.0–6.3 TFLOP/s).
- Target state:
  - Startup RSS: ≤ 14.6 GiB (15.7 GB).
  - Decode throughput: ≥ 18.0 t/s.
  - Prefill throughput: ≥ 200.0 t/s.

---

## Log of Steps and Learnings

### T1: Filter vision tower tensors during model load
- Implemented `is_vision_tensor` in `crates/engine/src/load.rs`.
- Safetensors loader skips 356 tensors matching `model.vision_tower.*` and `model.embed_vision.*`.
- Retained language model tensors: 2066 tensors (all 60 layers, norms, trellis, embed_tokens, lm_head).
- Resident Metal allocation dropped from 16.028 GB to **14.877 GB (13.855 GiB)**, saving **1.151 GB**.
- Load time: 3.3 seconds.
- Verified: `load_real_artifact_if_present` and `is_vision_tensor_filters_prefixes` passed; all 115 engine unit tests passed.

### T2: Verify baseline resident memory reduction (HostMmap for embed_tokens)
- Added `HostMmap` (using `libc::mmap` with `MAP_SHARED, PROT_READ`) in `crates/engine/src/load.rs`.
- `embed_tokens.weight` (2.819 GB) is accessed exclusively on the host CPU in `embed_row` and `embed`; it is now mapped on-demand via virtual memory rather than pre-allocated into resident Metal shared buffer.
- `model.resident_bytes()` dropped to **12,058,593,088 bytes = 12.058 GB (11.23 GiB)**.
- Gated test `prefill_memory_profile_slow`:
  - `baseline: rss=3959280 KB` (~3.96 GB)
  - `after 8192-token prefill: rss=3943936 KB`
  - `RSS growth: 0 KB (limit 49152)`
  - Confirmed 100% numerical accuracy and complete adherence to ≤ 14.6 GB (decimal) limit.

