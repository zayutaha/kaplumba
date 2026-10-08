# Technical Plan: Green Memory Residency

## 1. Architecture & Memory Flow

### 1.1 Memory State Comparison

```text
CURRENT ARCHITECTURE (MLOCK PATH):
+-------------------------------------------------------------------------+
| Shard File on NVMe (safetensors)                                       |
+-------------------------------------------------------------------------+
       | mmap(PROT_READ, MAP_PRIVATE)
       v
+-------------------------------------+  +--------------------------------+
| Clean File Pages in Page Cache      |  | KV Cache & Scratch Buffers    |
+-------------------------------------+  +--------------------------------+
       | libc::mlock (12.04 GiB)                 | StorageModeShared (dirty anon)
       v                                         v
+-------------------------------------+  +--------------------------------+
| WIRED MEMORY: 17.87 GiB             |  | UNWIRED MEMORY DEMAND: 6.5 GiB |
| (12.04 GiB trellis + 5.5 GiB OS)    |  | Available unwired RAM: 6.5 GiB |
+-------------------------------------+  +--------------------------------+
       ==> 100% of Unwired RAM Exhausted at 13.8 GiB Total Footprint
       ==> FREE MEMORY DROPS TO 15% ==> YELLOW PRESSURE & COMPRESSED SWAP
```

```text
PROPOSED ARCHITECTURE (SHARED ARENA PATH - MLX STYLE):
+-------------------------------------------------------------------------+
| Shard File on NVMe (safetensors)                                       |
+-------------------------------------------------------------------------+
       | Parallel pread via arena.read_into (Ascending Shard Offsets)
       v
+-------------------------------------------------------------------------+
| WeightArena (StorageModeShared Metal Buffers, 512 MiB Segments)         |
| 13.4 GiB Dirty Anonymous Metal Memory (No mlock, No Clean-File Trap)    |
+-------------------------------------------------------------------------+
       |
       +----------------------------+
       | Wired Memory: ~3.5 GiB (OS)|
       | Unwired RAM: 20.5 GiB      |
       +----------------------------+
       |
       v
+-------------------------------------------------------------------------+
| Total Memory at 17.5k Context:                                         |
| 13.4 GiB (Weights) + 6.5 GiB (KV Cache) + 0.5 GiB (Scratch) = 20.4 GiB |
| Hardware Working Set Limit = 22.46 GiB                                  |
| System Free/Purgeable Memory > 20% ==> GREEN PRESSURE & ZERO SWAP      |
+-------------------------------------------------------------------------+
```

---

## 2. File & Module Touchpoints

### 2.1 `crates/engine/src/weights.rs`
- **Current Behavior:**
  - `pin_trellis_pages`: walks `.trellis` tensors and invokes `libc::mlock`.
  - `pin_trellis`: reads `KAPLUMBA_NO_PIN_TRELLIS` env var.
  - `skip_copy` in `TextWeights::load`: sets `skip_copy = true` for all `.trellis` and `lm_head.trellis` tensors unless `KAPLUMBA_RESIDENT_WEIGHTS` is set.
- **Intended Change:**
  - Delete `pin_trellis_pages` and `pin_trellis`.
  - In `TextWeights::load`, configure `skip_copy`:
    - `skip_copy` is **only** `true` for `offload_bf16_embedding` (`model.language_model.embed_tokens.weight`).
    - Layer `.trellis` and `lm_head.trellis` are allocated dedicated or bump spans in `WeightArena` and loaded via `copy_tensor` -> `arena.read_into`.
  - Retain `zero_copy_tensor_buffer` and `raw_zero_copy_tensor_buffer` for `embed_tokens.weight`.

### 2.2 `crates/engine/src/model.rs`
- **Current Behavior:**
  - `load_quantized_linear`: checks `tensor.spans.is_empty()`. If spans exist, calls `weights.tensor_buffer(&trellis_name)?` with `offset = 0`. If spans are empty, falls back to `zero_copy_tensor_buffer` with `align_packed_view`.
- **Intended Change:**
  - Already natively compatible: because `tensor.spans` will now be populated for `.trellis` tensors, `load_quantized_linear` automatically binds directly to the resident arena buffers at `offset = 0`.
  - Verify alignment assertions (`packed_offset % EXL3_PACKED_ALIGNMENT == 0`).

---

## 3. Interfaces & Layout

- **`WeightArena` Layout:**
  - Sized by `DEFAULT_RESIDENT_LIMIT_BYTES` (18.0 GB).
  - Segment size: `DEFAULT_SEGMENT_BYTES` (512 MiB).
  - Storage mode: `MTLResourceOptions::StorageModeShared`.
  - Parallel loading: `pending.par_iter()` sorts by ascending file offset per shard, streaming chunks into arena segments via positional `read_at`.

---

## 4. Sequencing & Dependencies

```mermaid
flowchart LR
    T1[T1: Remove mlock] --> T2[T2: Arena Copy for Trellis]
    T2 --> T3[T3: Sabotage & Regression Probes]
    T3 --> T4[T4: Multi-Turn Long-Context Verification]
```

1. **Task 1:** Remove `pin_trellis_pages` and all `mlock` calls from `crates/engine/src/weights.rs`.
2. **Task 2:** Enable arena materialization for `.trellis` weights and `lm_head.trellis` in `weights.rs`.
3. **Task 3:** Run unit tests and sabotage verification (proving 0 wired pages and zero file-backed dirty pages).
4. **Task 4:** End-to-end multi-turn benchmark at 17.5k context proving Green memory pressure, 0 swap, and $\ge 13.0\text{ tok/s}$.

---

## 5. Risks & Fallbacks

- **Risk 1: Startup time regression.**
  - *Mitigation:* `arena.read_into` uses parallel file reads across all cores sorted by shard offset. Startup on M4 Pro NVMe is expected at 1.5–2.0s, which the user explicitly approved.
- **Risk 2: Arena segment fragmentation or alignment mismatch.**
  - *Mitigation:* `WeightArena::allocate` handles multi-span bounding; `align_packed_view` in `model.rs` checks 4-byte EXL3 alignment. Unit tests guard against arithmetic overflow.

---

## 6. Requirement to Task Mapping

| Requirement | Satisfying Tasks |
|---|---|
| **R1 (Removal of `mlock`)** | Task 1 |
| **R2 (Shared Metal Allocation for Trellis)** | Task 2 |
| **R3 (Green Memory Pressure at 17.5k)** | Task 4 |
| **R4 (Zero Swap Growth)** | Task 4 |
| **R5 (Sustained Decode $\ge 13$ tok/s)** | Task 4 |
| **R6 (Regression & Sabotage Tests)** | Task 3 |
