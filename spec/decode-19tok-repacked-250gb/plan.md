# Technical Plan: Decode 15+ tok/s and 250+ GB/s via Repacked Weights

> **THIS FILE IS LARGELY SUPERSEDED. Read `spec.md` first.**
>
> The D=32 revert (`9c85e9a`) rolled this file back to its pre-amendment state along
> with the rest of the T14 work, which resurrected §1.2 and §1.3 below **verbatim** —
> including the 128-byte-burst amplification thesis that `spec.md` §5 explicitly
> withdraws, and the `_vec128` kernel design that `spec.md`'s header records as
> **cancelled**. An agent that starts here and not there will build the wrong kernel.
>
> Specifically stale:
> - **§1.2** asserts the 1.333× DRAM-burst amplification. **Withdrawn** (`spec.md` §5);
>   `reference/omlx` and this repo contain no measurement or citation for it, and
>   `exl3.rs` + `linear.metal` make consecutive K-tiles gapless at 96 B stride.
> - **§1.3** (`_vec128`) and the T7.2/T7.3/T7.5 `_u4` work: **cancelled.** `_u4`
>   measured 180.5 → 173.7 on `gate_up`; the `_u8` deep-unroll gave 1.01–1.03×.
> - **§1.1**'s premise ("explores why previous `bw.rs` saturated at ~226 GB/s") is
>   answered: the 239–244 GB/s ceiling was **not** a load-width artifact.
> - **§2** names `encode_gemv_vec128` and `repack_model.rs`'s 128-byte coalescing as
>   live touchpoints. Neither is on the decode path.
> - **§4's requirement→task map** points R6 at T7.5 and T8; T7.5 is cancelled and T8
>   still says 19 tok/s.
>
> The live architecture is: colmajor tiles (`_contig4`), single command buffer per
> decode step, and a **fusion roadmap budgeted at 3.5 µs per dispatch removed**, not at
> kernel time (`spec.md` §9).

## 1. Architecture & Data Flow (SUPERSEDED — see banner)

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

## 2. File & Module Touchpoints (STALE — names kernels that are not on the decode path)

- `crates/engine/src/bin/repack_model.rs`: Offline on-disk repacking CLI tool (128-byte 4-tile coalescing).
  **Live, but the coalescing it does is the colmajor reorder, not `_vec128` chunking.**
- `crates/kernels/src/linear.metal`: `linear_gemv_ishizuki_k3_vec128`, `linear_gemv_ishizuki_k3_fused2`, `linear_gemv_ishizuki_k3_fused3`.
  **`_vec128` does not exist on the decode path (cancelled).** The live kernel is
  `linear_gemv_ishizuki_k3_colmajor_contig4` in `crates/kernels/src/linear_repacked.metal`.
- `crates/kernels/src/linear.rs`: `encode_gemv_fused2`, `encode_gemv_fused3`, and `encode_gemv_vec128`.
  **`encode_gemv_vec128` is not live.** `encode_ishizuki_gemv_named` →
  `finalize_simd_pipeline` is the finalize path; `encode_gemv_fused2_mlp` and
  `linear_finalize_gated_gelu_and_down_input_transform` are the live fusions.
- `crates/kernels/src/encoder.rs`: Configurable `max_dispatches` per encoder. **Live** (`set_max_dispatches`).
- `crates/engine/src/model.rs`: Fused forward execution and single-buffer decode dispatching. **Live.**
- `crates/engine/tests/repacked_golden.rs`: 100% bit-exact validation and fail-first sabotage test.
  **Its coverage claim is void** — it asserted `to_bits()` for `_colmajor` and
  `_colmajor_simd` while the engine dispatched `_colmajor_contig4`, which it never
  mentioned. The real assertion is `crates/engine/tests/live_kernel_bit_exact.rs`.

---

## 3. Requirement to Task Mapping (partly stale)

| Requirement | Tasks |
|---|---|
| R1 (Synthetic probe) | T1, T2 (Done) |
| R2 (Fail-first testing) | T2, T4, T6, T7.4 |
| R3 (Offline repacker) | T3 (Done) — T7.2 **cancelled** |
| R4 (Repacked GEMV kernel) | T5, T7 (Done) — T7.3 **cancelled** |
| R5 (Non-matvec & buffer optimization) | T7 (Done), T7.1, T7.6.1–T7.6.13 |
| R6 (End-to-end performance) | T8 — **and note T7.5 is cancelled and the whole remaining fusion budget is ~0.8 ms (`spec.md` §9)** |

---

## 4. Risks & Fallbacks

- **Hardware DRAM wall:** If M4 Pro LPDDR5X-8533 hardware ceiling is strictly bounded by DRAM physical efficiency (< 250 GB/s for uncompressed random pages): probe identifies exact ceiling; SLC / cache-pinned staging used to exceed DRAM speed.
  **Partly resolved:** the probe measured 250.5 GB/s median on 12 GB (`spec.md` AC1 still
  gates it, and AC1's 0.2% margin is thinner than this box's documented >=10% drift).
  The "SLC / cache-pinned staging" fallback is **speculative and untested** — no such
  path exists in this repo. Do not plan against it.
- **Disk space:** Model requires ~15 GB. Check disk space before write; delete previous copy before creating new one.
- **The remaining risk is not a wall, it is a plateau.** Every mechanism proposed for
  the 3 bpw rate has been measured and failed (`spec.md` §6.6). The plan has no
  remaining hypothesis; the next one has to be found, not executed.
