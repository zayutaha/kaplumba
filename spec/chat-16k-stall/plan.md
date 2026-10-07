# Technical Plan: `spec/chat-16k-stall`

## 1. Architecture and Root Cause

The investigation into the 16k `--chat` collapse (0.03 tok/s, 18–30 s/token) revealed two independent defects:

### Defect 1: The Anonymous Memory / Swap Cliff (Primary Cause of 0.03 tok/s)
* `weights.rs` loads layer weights by allocating ~1,440 separate `StorageModeShared` Metal buffers via `arena.allocate_dedicated` and reading file contents into them via `read_exact_at` (`pread`).
* This marks 13.4 GiB of weights as **dirty anonymous graphics memory** (`IOAccelerator (graphics)`).
* macOS wired memory (WindowServer, display, OS) sits at 8.0–9.4 GiB.
* At short context (e.g. 2k tokens), $13.4\text{ GiB} + 9.0\text{ GiB} \approx 22.4\text{ GiB}$, which barely fits in 24 GiB physical RAM with minimal swap.
* At long context (11k–16k tokens), full-attention KV cache grows to ~3.0 GiB ($10\text{ layers} \times 16\text{k} \times 512 \times 4\text{B}$).
* Total un-evictable memory hits $13.4 + 9.0 + 3.0 \approx 25.4\text{ GiB}$, exceeding the 24 GiB physical capacity and the OS `user_wire_limit` of 18.2 GiB.
* macOS memory pressure transitions to **Yellow**. The OS memory compressor fails to compress the high-entropy EXL3 bitpacks, forcing **12.5 GiB into swap/compressor**.
* Every decode token must touch all 60 layers (13.4 GiB of weights), resulting in continuous major page faults from the SSD (~1,000,000 page faults, ~237,000 pageins measured on live PID 68980), collapsing decode time to 18–30 s/token.
* **Why MLX stays green at 18 GB**: MLX maps weights directly using `mmap` with `new_buffer_with_bytes_no_copy`. Clean file-backed pages are never compressed and never written to swap (`SWAPPED = 0K`).

### Defect 2: Missing KV Cache Retraction on EOS in `chat.rs`
* In `crates/engine/src/chat.rs`, the pipelined decode loop speculatively encodes step $N+1$ before waiting on step $N$.
* When step $N$ produces an EOS token, `generate.rs:1050` calls `model.retract_gpu_caches(1)` to retract the speculative advance.
* `chat.rs:507-512` breaks without calling `retract_gpu_caches(1)`. The KV cache cursor permanently drifts forward by phantom tokens across turns.

---

## 2. File & Module Touchpoints

1. `crates/engine/src/weights.rs`:
   - Switch layer `.trellis` and sign components to zero-copy file-backed mapping (`zero_copy_tensor_buffer` / `new_buffer_with_bytes_no_copy`) by default, avoiding the 13.4 GiB dirty anonymous buffer allocation.
   - For any residual file reads, apply `fcntl(fd, F_NOCACHE, 1)` to prevent doubling the footprint in the Unified Buffer Cache.
   - Pre-fault zero-copy mapped pages (`madvise(MADV_WILLNEED)`) at load time so faults do not occur on the decode hot path.

2. `crates/engine/src/chat.rs`:
   - Add `self.model.retract_gpu_caches(1)` on EOS in the pipelined decode loop.

3. `crates/kernels/src/encoder.rs`:
   - Fix `max_in_flight()` parser to permit values $> 8$ (up to 64) without silent clamp.

---

## 3. Interfaces and Formats

* No changes to on-disk weights format or EXL3 quantization layout.
* `WeightArena` preserves resident handling for BF16 converted tensors (norms/scales, ~2.6 MB).
* Zero-copy buffer views align with `PAGE_ALIGN` as established in `weights.rs:1050-1077`.

### 3.1 Component Offloading Policy
1. **Vision Tower (1.15 GiB)**: Remains **never loaded / excluded** (`is_vision_tensor_name` in `safetensors.rs:516`). 0 bytes allocated.
2. **`embed_tokens` (2.82 GiB)**: Remains **offloaded via zero-copy mmap** (`raw_zero_copy_tensor_buffer` with in-kernel BF16 conversion). 0 bytes resident in arena.
3. **`lm_head.trellis` (1.0 GiB)**: Remains **offloaded via zero-copy mmap** (`zero_copy_tensor_buffer`). 0 bytes resident in arena.
4. **Layer Weights (10.25 GiB)**: Transitioned to clean file-backed Metal views, so they are non-dirty and never eligible for swap.

---

## 4. Sequencing and Task Mapping

1. **Task 1**: Eliminate anonymous weight allocation by enabling zero-copy file-backed mapping for layer weights, preventing swap thrashing (Satisfies R1, R3).
2. **Task 2**: Fix EOS cache retraction in `chat.rs` (Satisfies R5, R7).
3. **Task 3**: Fix `max_in_flight()` parser clamp in `encoder.rs` (Satisfies R1, R2).
4. **Task 4**: Verify token-stream equivalence across multi-turn sessions (Satisfies R5).
5. **Task 5**: Verify parity and memory footprint at 16k context (Satisfies R4, AC1, AC2, AC5).

---

## 5. Risks and Fallbacks

* **Risk:** GPU accessing mmapped pages could incur minor page faults on first touch.
  * **Fallback:** Pre-fault pages sequentially at load time using `madvise(MADV_WILLNEED)` to ensure full residency without dirtying pages.
* **Risk:** Metal buffer alignment requirements for non-page-aligned tensor offsets.
  * **Fallback:** Use existing `align_packed_view` (`model.rs:5680-5705`) which is already proven in production for embedding and lm_head zero-copy views.
