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

### 1.2 128-Byte Vector Weight Repacking Format
- Hardware reality: Apple M4 Pro LPDDR5 DRAM bus issues 128-byte burst transactions (two 64-byte cache lines).
- Standard EXL3 3.0 bpw tile is 16x16 weights = 768 bits = 96 bytes (24 uint32s).
- 96 bytes / 128 bytes = 0.75: fetching 96 bytes incurs a 128-byte DRAM transaction (1.333x amplification), capping effective weight bandwidth to 188.5 GB/s even while the DRAM bus is pinned at 251.2 GB/s.
- 128-Byte Repacked Layout:
  - 4 consecutive tiles along K = $4 \times 96\text{ bytes} = 384\text{ bytes} = \mathbf{3 \times 128\text{ bytes}}$ (6 full 64-byte cache lines).
  - Repack the 4-tile block into vector-aligned chunks so that threads issue 16-byte (`uint4`) loads.
  - Zero double-fetching across cache lines; physical DRAM bandwidth directly translates to effective weight bandwidth at 250+ GB/s.
  - Dequantization math: Exact bit-for-bit equivalence; zero quality loss.

### 1.3 128-Byte Vector GEMV Metal Kernel (`_vec128`)
- Consumes 384-byte (4-tile) vector-aligned chunks.
- Loads memory using `const device uint4 *`.
- Dispatches fused pairs (`gate_up`) and triplets (`qkv`) to ensure 1000+ threadgroups remain in flight across all 20 GPU cores.

### 1.4 Single Command Buffer Decode Pipeline
- Pre-allocate a 2500-dispatch capacity on `MetalEncoder` for decode steps (`tokens == 1`).
- Eliminates 10-buffer rolling overhead, allowing all 60 layers to commit in 1 single command buffer per token without CPU-GPU synchronization bubbles.

---

## 2. File & Module Touchpoints

- `crates/engine/src/bin/repack_model.rs`: Offline on-disk repacking CLI tool (128-byte 4-tile coalescing).
- `crates/kernels/src/linear.metal`: `linear_gemv_ishizuki_k3_vec128`, `linear_gemv_ishizuki_k3_fused2`, `linear_gemv_ishizuki_k3_fused3`.
- `crates/kernels/src/linear.rs`: `encode_gemv_fused2`, `encode_gemv_fused3`, and `encode_gemv_vec128`.
- `crates/kernels/src/encoder.rs`: Configurable `max_dispatches` per encoder.
- `crates/engine/src/model.rs`: Fused forward execution and single-buffer decode dispatching.
- `crates/engine/tests/repacked_golden.rs`: 100% bit-exact validation and fail-first sabotage test.

---

## 3. Requirement to Task Mapping

| Requirement | Tasks |
|---|---|
| R1 (Synthetic probe) | T1, T2 (Done) |
| R2 (Fail-first testing) | T2, T4, T6, T7.4 |
| R3 (Offline repacker) | T3 (Done), T7.2 |
| R4 (Repacked GEMV kernel) | T5 (Done), T7 (Done), T7.3 |
| R5 (Non-matvec & buffer optimization) | T7 (Done), T7.1 |
| R6 (End-to-end performance) | T7.5, T8 |

---

## 4. Risks & Fallbacks

- **Hardware DRAM wall:** If M4 Pro LPDDR5X-8533 hardware ceiling is strictly bounded by DRAM physical efficiency (< 250 GB/s for uncompressed random pages): probe identifies exact ceiling; SLC / cache-pinned staging used to exceed DRAM speed.
- **Disk space:** Model requires ~15 GB. Check disk space before write; delete previous copy before creating new one.
