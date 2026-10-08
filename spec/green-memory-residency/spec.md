# Specification: Green Memory Residency (Zero-Swap Long-Context Decode)

## 1. Goal

Eliminate yellow system memory pressure and SSD swap activity during long-context inference (up to 17,500+ tokens) by transitioning model weight residency from POSIX `libc::mlock` to standard Metal shared allocations (`MTLResourceStorageModeShared`), matching MLX's zero-wired-memory architecture while strictly preventing NVMe read-thrashing.

---

## 2. Non-goals & Restrictions

- **No KV Cache Quantization:** KV cache remains full fp16 as mandated by AGENTS.md §8 standing constraints.
- **No Quality Degradation:** Output logits and generated greedy token streams must remain bit-exact identical to reference (`live_kernel_bit_exact`, `multi_turn_chat`).
- **No Python on the hot path:** Runtime remains pure Rust + raw Metal compute.
- **No Hard Context Truncation:** Under extreme memory pressure (>18k tokens), allow macOS VM compressor to manage inactive anonymous memory gracefully; do not artificially crash or drop tokens.
- **Preserve Specialized Tensor Offloading:** `embed_tokens.weight` (2.82 GB) remains on its specialized zero-copy in-kernel row-fetching path; vision tower remains unallocated (0 bytes).

---

## 3. Requirements

1. **R1: Complete Removal of `libc::mlock`**
   - The engine must not call `libc::mlock` under any circumstances.
   - `pin_trellis_pages`, `pin_trellis`, and the `KAPLUMBA_NO_PIN_TRELLIS` toggle in `crates/engine/src/weights.rs` must be completely removed.
   - Total system wired memory (`Pages wired down`) contributed by Kaplumba must be strictly **0 bytes**.

2. **R2: Anonymous Shared Metal Allocation for Layer Trellis Weights**
   - The 12.04 GiB of layer `.trellis` weights and `lm_head.trellis` must be loaded into anonymous `MTLResourceStorageModeShared` Metal buffers (via `WeightArena`), rather than remaining as clean file-backed mmap pages.
   - Because these pages are dirty anonymous Metal buffers, the Darwin VM will never treat them as disposable file cache, eliminating the 0.37 tok/s NVMe eviction cliff.
   - Model startup time into the shared arena must remain within acceptable bounds ($\le 2.5\text{ seconds}$).

3. **R3: Memory Pressure Bounds (Green Zone at 17.5k Context)**
   - During multi-turn interactive chat (`kaplumba --chat`) up to 17,500 context tokens on a 24 GiB machine, system memory pressure reported by `memory_pressure` must remain in the **Normal (Green)** state.
   - System-wide free/reclaimable memory percentage must remain above 20%.

4. **R4: Zero Swap Growth**
   - `/usr/bin/time -l` must report `0` swapouts and `0` swapins during 17.5k token chat execution.
   - `sysctl -n vm.swapusage` must record `0.00M` swap growth.

5. **R5: Sustained Decode Throughput**
   - Steady-state decode throughput at 17,500 context tokens must sustain $\ge 13.0\text{ tok/s}$, proving that neither disk paging nor NVMe re-faulting is taking place.

6. **R6: Regression & Soundness Proof**
   - Sabotage-verified test coverage: prove via an isolated test that disabling shared memory triggers either page-eviction or wired footprint, and that the new path maintains zero wired pages.

---

## 4. Acceptance Criteria

1. `memory_pressure` command executed during active steady-state decode at 17,500 context tokens reports `System-wide memory free percentage: >= 20%` and no yellow/critical pressure warnings.
2. `vmmap -summary <pid>` reports:
   - `mapped file`: `DIRTY = 0K` (no wired file pages).
   - `IOAccelerator (graphics)`: resident and dirty within total physical footprint $< 21\text{ GiB}$.
3. Zero swap activity measured via `vm.swapusage` and `/usr/bin/time -l` across a 17.5k token multi-turn chat session.
4. Sustained decode throughput $\ge 13.0\text{ tok/s}$ at 17.5k context.
5. All 102+ unit and regression tests in `crates/engine` pass cleanly. Golden tests (`live_kernel_bit_exact`, `multi_turn_chat`) pass with bit-exact parity.

---

## 5. Open Questions

*(None — all architectural decisions clarified and confirmed during interview.)*
