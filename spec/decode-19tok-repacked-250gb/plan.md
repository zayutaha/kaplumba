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

## 4. T14 — 32 codes per lane (`D = 32`)

### 4.1 Why this is the shape of the fix

The live kernel `linear_gemv_ishizuki_k3_colmajor_contig4`
(`linear_repacked.metal:962-1080`) gives each of 32 lanes **8 codes**. A lane's codes
are 24 consecutive ring bits, and it needs a 16-bit decode window per code, so it reads
2 words. 32 lanes × 2 = **64 lane-loads per 24-word tile = 2.667×**.

`D = 32` gives 8 lanes per tile. Each lane's codes are 96 consecutive ring bits and it
needs `96 + 13 = 109` bits = 4 words, which fits in 4 words with 19 bits of slack. And
because `96·L mod 32 == 0` for every `L`, **every lane's window starts on a word
boundary** — the four-way `sh0 ∈ {19,11,3,27}` funnel collapses to one constant.
8 lanes × 4 words = **32 lane-loads per 24-word tile = 1.333×**.

Tile count is unchanged (24 words is 24 words), so **no repacking, no new checkpoint,
no format change, and the dequant arithmetic is untouched** — the 16-bit windows handed
to `decode_state_mcg` are bit-identical, only *which lane* decodes them changes.

### 4.2 Consequences that must be designed around

| consequence | detail |
|---|---|
| 4 tiles per simdgroup | 8 lanes no longer fill 32. Lanes 0-7 → tile 0, 8-15 → tile 1, 16-23 → tile 2, 24-31 → tile 3. One threadgroup covers 64 outputs instead of 16. |
| 4× fewer threadgroups | `gate_up` fused 2688 → 672; `down_proj` 336 → **84**. On 20 cores that is 4.2 waves instead of 16.8 — a worse tail, **expected and accepted** (AC9). |
| 32 decodes per lane | 4× the per-lane ALU, same total. §6.6 records that removing decode makes the kernel *slower*, so more decode is not automatically bad — but it is untested. |
| reduction changes | 8 lanes contribute per output instead of 4, so `simd_shuffle_xor` masks become 1/2/4 within an 8-lane group rather than 1/2 across all 32. |
| ring wrap | one lane per tile straddles the tile end. Cost: 1 `uint4` + 2 `uint2` instead of 4 `uint`. Byte-amplification is unchanged at 1.333. |

### 4.3 Sequencing — probe first, deliberately

The 1.333× prediction is an **extrapolation of a law fitted at two points**, one of
which (amp 2.667) is where we already are. Committing to a 4-tiles-per-simdgroup
restructure before checking that amp 1.333 buys anything would be building against a
number no instrument has produced.

So **T14.1 is a probe, not a kernel**: a standalone entry point that issues the
`D = 32` load pattern with the real `decode_state_mcg` body and writes deliberately
garbage output. It is measured on the real shapes against `_contig4`, cold
(`MADV_DONTNEED`), interleaved. **AC7 cancels T14.2 if the median ratio is not within
1.6× of the predicted 2.0.** The probe is sabotage-verified per AGENTS.md §3.4 — a probe
that cannot fail proves nothing, and this repo has already shipped one that could not.

### 4.4 Wiring

New kernels are added **alongside** `_contig4` in `linear_repacked.metal` and selected
by an env var that **defaults off** (AC10). `ishizuki_gemv_pipeline` in `linear.rs`
gains one arm; no existing call site changes. `KAPLUMBA_TRACE_GEMV=1` must show the new
name when selected — the historical failure in this repo was a golden test asserting
kernels that did not run.

---

## 5. Requirement to Task Mapping

| Requirement | Tasks |
|---|---|
| R1 (Synthetic probe) | T1, T2 (Done) |
| R2 (Fail-first testing) | T2, T4, T6, T7.4, **T14.1** |
| R3 (Offline repacker) | T3 (Done), T7.2 |
| R4 (Repacked GEMV kernel) | T5 (Done), T7 (Done), T7.3, **T14.2, T14.3** |
| R5 (Non-matvec & buffer optimization) | T7 (Done), T7.1 |
| R6 (End-to-end performance) | T7.5, T8, **T14.4, T15** |

---

## 6. Risks & Fallbacks

- **Hardware DRAM wall:** If M4 Pro LPDDR5X-8533 hardware ceiling is strictly bounded by DRAM physical efficiency (< 250 GB/s for uncompressed random pages): probe identifies exact ceiling; SLC / cache-pinned staging used to exceed DRAM speed.
- **Disk space:** Model requires ~15 GB. Check disk space before write; delete previous copy before creating new one.
- **The amplification law is a 2-point fit.** Fallback if AC7 fails: the request rate is
  not the wall, and the remaining 3 bpw headroom is bounded by something not yet
  identified. In that case T14.2–T14.4 are cancelled and effort moves to T15 (the 8.9 ms
  host gap), whose +3.21 tok/s does not depend on this law at all.
- **fp16 identity is not guaranteed.** `_contig4` needed only a 4-tile reassignment to
  stay fp16-identical; `D = 32` changes the entire lane→output map. If AC8 fails there
  is no partial credit — the kernel is not wired in, and the fallback is T15.
