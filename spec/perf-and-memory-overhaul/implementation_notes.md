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
