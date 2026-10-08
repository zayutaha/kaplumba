# Implementation Notes: Green Memory Residency

Append as work proceeds. Factual and dense.

## 0. Hardware & Environment
- **Machine:** Apple M4 Pro (20 GPU cores, 24 GiB unified RAM, 273 GB/s peak bandwidth).
- **OS:** macOS 15+ (Darwin kernel 27.0).
- **Model:** Gemma 4 31B dense (`~/.kaplumba/models/Gxma-repacked`), EXL3 ~3.0 bpw (~15.5 GB resident).
- **Target Context:** 17,500+ tokens.

## 1. Measured Baseline (Before Changes)
- **PID 9079 (`kaplumba --chat` with `prompt_17k.txt`):**
  - Physical footprint: 14.1 GiB (peak 14.2 GiB).
  - System wired memory: **17.87 GiB** (1,090,795 pages).
  - System-wide free percentage: **15% (YELLOW MEMORY PRESSURE)**.
  - VM compressor: **3.15–4.45 GiB**.
  - `mapped file`: 11.2 GiB DIRTY (caused by `libc::mlock`).
  - Available unwired DRAM for system + apps: **only 6.5 GiB**.
- **Post-Exit System Baseline:**
  - System wired memory: **3.13 GiB** (191,199 pages).
  - System-wide free percentage: **84% (GREEN MEMORY PRESSURE)**.
  - Unwired flexible DRAM: **20.5 GiB**.

## 2. Key Architecture Findings
1. **`libc::mlock` is the Root Cause of Yellow Pressure:**
   `mlock` forces Darwin VM to wire pages, removing them completely from the kernel's pageable/reclaimable pool. On a 24 GiB machine, wiring 12 GiB of weights plus 5 GiB OS leaves only 6.5 GiB, so any 1.8 GiB dynamic allocation triggers yellow memory pressure.
2. **MLX Baseline Analysis:**
   MLX defaults `set_wired_limit(0)` and allocates weights into `MTLResourceStorageModeShared` Metal buffers. System wired memory stays at ~3.5 GiB, leaving 20.5 GiB of flexible RAM.
3. **`WeightArena::read_into` is Pre-existing:**
   The engine already has `WeightArena::read_into` with parallel positional `read_at` into shared Metal segments. Routing `.trellis` into the arena requires only setting `skip_copy = false` for trellis weights and removing the `mlock` call.

## 3. Sabotage Verification
- In `crates/engine/tests/probe_shared_weights_footprint.rs`:
  - Normal execution: `mapped_file_dirty = 0`, shared buffer has 0 wired pages.
  - Sabotage execution (`KAPLUMBA_SABOTAGE_WIRED_WEIGHTS=1`): fails with explicit message `SABOTAGE ACTIVE: buffer was mlocked, wiring down physical pages!`.
  - Sabotage reverted: test passes 1/1.

## 4. Post-Implementation Telemetry
- **Model Load:**
  - `load_ms`: 11,143 ms to 16,174 ms.
  - Weights streamed into `WeightArena` via `read_at` across all cores.
- **Generation:**
  - `prompt_tokens=19 generated_tokens=10 decode_tok_s=15.210` on `Gxma-repacked`.
- **17.5k Context Memory Pressure Telemetry:**
  - `memory_pressure` during active 17.5k context inference: **36% free (SOLID GREEN)**, comfortably exceeding the $\ge 20\%$ target.
  - System wired pages: reduced by ~300,000 pages (saving ~4.8 GiB of wired memory).
  - `vm.swapusage`: **0 growth** during prefill/decode.
