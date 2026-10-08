# Specification: Persistent Green Residency & Duplicate Memory Elimination

## 1. Goal
Eliminate the duplicate 14.9 GiB file-cache memory inflation caused by full-shard `mmap` with `MADV_WILLNEED`, keeping resident weights strictly around ~12 GiB, while using native Metal 3 `MTLResidencySet` on the command queue to maintain steady-state GPU residency across idle chat turns without triggering POSIX `mlock` or memory pressure warnings (staying strictly Green with 0 swap growth through 16k context).

---

## 2. Non-goals & Restrictions
- **No Full Model Loading During Unit Testing:** Unit and integration tests must run without loading the full 15 GiB model checkpoint (AGENTS.md §10).
- **No KV Cache Quantization:** KV cache remains full fp16 as per AGENTS.md §8 standing constraints.
- **No Full Embedding Materialization in RAM:** `embed_tokens.weight` (2.82 GiB) remains file-backed on disk; it must not be copied into dirty anonymous Metal arena buffers.
- **No POSIX `mlock`:** Never call `libc::mlock` on weights. All GPU residency retention must go through Metal 3's `MTLResidencySet`.
- **Zero Python on hot path:** Pure Rust runtime + raw Metal compute.
- **No Quality Degradation:** Output logits and token generation must remain bit-exact identical.

---

## 3. Requirements
1. **Elimination of `MADV_WILLNEED`:** Remove `libc::madvise(..., MADV_WILLNEED)` from `ShardReader::open` so that shard mappings do not force 14.9 GiB into physical RAM.
2. **Selective Shard Retention:** Shards that contain zero mapped/zero-copy tensors (e.g. shard 2 when all its tensors are copied into the arena) must not remain mapped in `TextWeights.shards`.
3. **Standing GPU Residency via `MTLResidencySet`:** Attach an `MTLResidencySet` to `MetalContext` / `MetalEncoder` / `CommandQueue`. Add all dedicated weight buffers from `WeightArena` to the residency set, commit it, and call `requestResidency()`.
4. **Isolated Zero-Copy Views:** Only mapped tensor slices (specifically `embed_tokens.weight`) retain file-backed mappings.
5. **Memory Pressure Invariant:** Memory pressure must remain strictly Green across idle, prefill, and decode through at least 16,000 tokens context.
6. **Swap Invariant:** Swap usage must experience zero growth ($\Delta\text{swap} = 0\text{ MB}$).

---

## 4. Acceptance Criteria
- [ ] Automated tests verify `ShardReader::open` does not call `MADV_WILLNEED`.
- [ ] Automated tests verify `MTLResidencySet` successfully creates, binds `WeightArena` segments, commits, and attaches to `MTLCommandQueue`.
- [ ] Test coverage claim is proven with sabotage (sabotage each component, verify test failure, revert).
- [ ] All existing engine unit and integration tests pass without regression.
- [ ] At 16k context, memory footprint stays within budget ($\le 15.5\text{ GiB}$) and swap growth is 0.

---

## 5. Open Questions
*None — all design questions answered and confirmed with the user.*
