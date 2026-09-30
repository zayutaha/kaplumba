# Technical Plan: Decode 19+ tok/s and 250+ GB/s via Repacked Weights

## 1. Architecture & Data Flow

### 1.1 Synthetic Bandwidth Probe (`probe_bw_250gb`)
- Explores why previous `bw.rs` saturated at ~226 GB/s:
  - Vector width: 4-byte (`uint`) vs 16-byte (`uint4`) vs 32-byte (`ulong4` or two `uint4`).
  - Threadgroup layout: 32 lanes (simdgroup), 256 threads, 512 threads, 1024 threads.
  - Concurrency: Grid sizing from 20,480 to 81,920 threads.
  - Unrolling: 2x, 4x, 8x loads per loop iteration to saturate memory pipelines.
  - Resource storage & cache modes: `StorageModeShared`, `CPUCacheModeWriteCombined` vs `DefaultCache`.
  - Stride & bank layout: Coalesced memory access with contiguous 128-byte DRAM cacheline fetching.

### 1.2 Weight Repacking Format
- Current EXL3 3.0 bpw format:
  - Sequential bit packing across tiles results in non-word-aligned reads where SIMD lanes load 3 words (12 bytes) to extract 3 bytes of codes (4.0x -> 2.67x lane-load amplification).
  - Lanes read overlapping unaligned byte ranges with bit-shift extraction.
- Repacked Format:
  - Bit-plane or lane-aligned structure: Reorder bit representations per tile so every lane reads aligned 32-bit or 64-bit words containing only its dedicated codes.
  - Dequantization math: Exact same codebook indices, trellis states, and scales.
  - Output values match bit-for-bit with original dequantization.
  - Safetensors structure copied, only quantized weight tensors transformed.

### 1.3 Repacked GEMV Metal Kernel
- Consumes repacked 3 bpw tensors.
- Issues perfectly aligned `uint4` or `uint2` memory loads.
- Eliminates multi-word cross-lane window loads and dynamic shifts.
- Reduces memory traffic to 1.0x useful bytes, targeting >230 GB/s achieved bandwidth.

### 1.4 Overlap & Pipeline Optimization
- Overlap attention computation and layer norms with GEMV weight dispatching using concurrent command encoders or event fences.
- Reduce 7.4 ms non-matvec latency to < 3.0 ms.
- Total token span budget:
  - Matvecs: 11.93 GB @ 250 GB/s = 47.7 ms.
  - Non-matvec: < 4.9 ms.
  - Total <= 52.6 ms (= 19.0+ tok/s).

---

## 2. File & Module Touchpoints

- `crates/engine/examples/probe_bw_250gb.rs`: Synthetic streaming bandwidth probe.
- `crates/kernels/src/bw_probe.metal`: Metal kernel sweeping vector widths and access patterns.
- `crates/engine/src/bin/repack_model.rs`: Offline on-disk repacking CLI tool.
- `crates/kernels/src/linear_repacked.metal`: New GEMV kernel for repacked weights.
- `crates/kernels/src/linear.rs`: Dispatch logic and pipeline compilation for repacked weights.
- `crates/engine/src/weights.rs`: Detection and loading of repacked model weights.
- `crates/engine/src/model.rs`: Execution pipeline with non-matvec overlapping.

---

## 3. Requirement to Task Mapping

| Requirement | Tasks |
|---|---|
| R1 (Synthetic probe) | T1, T2 |
| R2 (Fail-first testing) | T2, T4, T6 |
| R3 (Offline repacker) | T3, T4 |
| R4 (Repacked GEMV kernel) | T5, T6 |
| R5 (Non-matvec overlap) | T7 |
| R6 (End-to-end performance) | T8 |

---

## 4. Risks & Fallbacks

- **Hardware DRAM wall:** If M4 Pro LPDDR5X-8533 hardware ceiling is strictly bounded by DRAM physical efficiency (< 250 GB/s for uncompressed random pages): probe identifies exact ceiling; SLC / cache-pinned staging used to exceed DRAM speed.
- **Disk space:** Model requires ~15 GB. Check disk space before write; delete previous copy before creating new one.
