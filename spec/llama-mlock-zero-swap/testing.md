# Testing Guidelines: `llama.cpp` `mlock` & Zero-Swap Architecture

Guidelines and exact procedures for verifying memory residency, absence of swap thrashing, KV cache allocation, and sabotage tests.

---

## 1. Quick Verification Commands

```bash
# 1. Run KV Cache unit tests
cargo test --package kaplumba-engine --lib gpu_cache

# 2. Run KV Cache capacity exactness probe
cargo test --package kaplumba-engine --test probe_kv_cache_capacity

# 3. Run LlamaMlock incremental locking probe
cargo test --package kaplumba-engine --test probe_llama_mlock

# 4. Run Weight residency & footprint probe
cargo test --package kaplumba-engine --test probe_shared_weights_footprint
```

---

## 2. Sabotage Testing Procedures (Mandatory per AGENTS.md §3)

> [!IMPORTANT]
> A test-coverage claim is not valid until you deliberately sabotage the code under test, observe that the test fails with a clear diagnostic message, and then revert the sabotage.

### Sabotage 1: KV Cache Double-Allocation
1. **Sabotage**: In `crates/engine/src/gpu_cache.rs:234`, revert `(target * self.stream_len * 2)` back to `(target * self.stream_len * 2 * 2)`.
2. **Run Test**: `cargo test --package kaplumba-engine --test probe_kv_cache_capacity`
3. **Expected Failure**: Test fails with:
   `assertion failed: buffer.length() == expected_bytes`
   `expected: 67108864, got: 134217728 (200% over-allocated!)`
4. **Revert**: Revert the edit back to `* 2` and confirm test passes.

### Sabotage 2: Unhandled `mlock` Failure / Overshoot
1. **Sabotage**: In `probe_llama_mlock.rs`, configure an allocation range that exceeds `RLIMIT_MEMLOCK` or requests an absurd size (e.g. 100 GiB).
2. **Run Test**: `cargo test --package kaplumba-engine --test probe_llama_mlock`
3. **Expected Result**: Assert `mlock.failed_already` is `true`, no panic occurs, and the process completes safely.

### Sabotage 3: Wiring Unpinned Embeddings
1. **Sabotage**: In `crates/engine/src/weights.rs`, force `skip_copy = false` for `embed_tokens` and pin it.
2. **Run Test**: Run with `KAPLUMBA_SABOTAGE_WIRED_WEIGHTS=1 cargo test --package kaplumba-engine --test probe_shared_weights_footprint`.
3. **Expected Failure**: Fails with explicit error: `SABOTAGE ACTIVE: buffer was mlocked, wiring down physical pages!`.

---

## 3. System-Level Memory & Swap Verification

### Baseline Measurement (Before Load)
Run in terminal:
```bash
# Record initial swap usage
sysctl -n vm.swapusage

# Record initial physical memory state (wired, unused, compressor)
top -l 1 -n 0 | grep PhysMem
```

### Active Inference Test (Long Context)
Run generation at $\ge 16\text{k}$ context through `gpuq`:
```bash
M=~/.kaplumba/models/Gxma/Gxma
PP=$(python3 -c "print('Explain quantum mechanics in complete detail. '*200)")

./scripts/gpuq run -- ./target/release/kaplumba --model "$M" --prompt "$PP" \
    --max-new-tokens 32 --temperature 0 --metrics
```

### Post-Inference Evaluation Criteria
1. **Swap Usage**:
   ```bash
   sysctl -n vm.swapusage
   ```
   **Acceptance Criteria**: `used` swap must show **0 MiB delta** (no growth).
2. **Wired Memory**:
   `PhysMem` wired count must not exceed $16.0\text{ GiB}$ ($12.04\text{ GiB}$ trellis + OS wired).
3. **Memory Pressure**:
   Memory pressure graph in macOS Activity Monitor must remain **Green** throughout prefill and generation.
4. **Throughput**:
   Decode throughput must remain $\ge 13.0\text{ tok/s}$ at long context (proving no page decompression stalls).
