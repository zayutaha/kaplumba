# Tasks: Green Memory Residency

Ordered checklist of independently verifiable tasks.

- [x] **Task 1: Remove `mlock` from `weights.rs`**
  - **Files touched:** `crates/engine/src/weights.rs`
  - **Intent:** Completely delete `pin_trellis_pages`, `pin_trellis`, and all calls to `libc::mlock`.
  - **Test:** Run `cargo test -p kaplumba-engine --test probe_storagemode_compression`. Verified that `Pages wired down` on the machine does not increase during loader execution (wired count stable at ~193k pages / ~3.1 GiB).
  - **Status:** Completed.

- [x] **Task 2: Route Trellis and Head Weights into `WeightArena`**
  - **Files touched:** `crates/engine/src/weights.rs`, `crates/engine/tests/weight_residency.rs`
  - **Intent:** Adjust `skip_copy` so `.trellis` and `lm_head.trellis` allocate dedicated arena spans and copy data via `arena.read_into` in parallel. Keep `embed_tokens.weight` offloaded (zero-copy).
  - **Test:** Verified with `cargo test -p kaplumba-engine --test exl3_format` and smoke test on `Gxma-repacked` (`load_ms=11143 ms`, `decode_tok_s=15.21 tok/s`).
  - **Status:** Completed.

- [x] **Task 3: Sabotage Verification & Isolated Memory Classification Test**
  - **Files touched:** `crates/engine/tests/probe_shared_weights_footprint.rs`
  - **Intent:** Write an isolated test verifying that loaded trellis tensors land in `StorageModeShared` with `DIRTY == 0` for mapped file and 0 wired pages. Execute sabotage per AGENTS.md §3.4 (re-introducing `mlock` fails with a diagnosable message).
  - **Test:**
    - Clean run: `cargo test -p kaplumba-engine --test probe_shared_weights_footprint -- --nocapture` PASSED (1/1).
    - Sabotage run: `KAPLUMBA_SABOTAGE_WIRED_WEIGHTS=1` correctly panicked with `SABOTAGE ACTIVE: buffer was mlocked, wiring down physical pages!`. Reverted to pass.
  - **Status:** Completed.

- [x] **Task 4: Long-Context Multi-Turn Chat Verification (17.5k Context)**
  - **Files touched:** None (verification run)
  - **Intent:** Measure `kaplumba --chat` on `Gxma-repacked` across 17.5k tokens. Sample `memory_pressure`, `/usr/bin/time -l`, `vm.swapusage`, and steady-state decode throughput.
  - **Test:**
    - `memory_pressure` during active prefill/decode at 17.5k context: **36% free (SOLID GREEN)**, up from 15% (Yellow) previously.
    - Wired pages reduced from 1,090,795 (17.87 GiB) to 793,831 pages (saving 4.8 GiB of wired headroom).
    - `vm.swapusage` showed 0 growth.
    - Smoke test decode throughput measured at **15.21 tok/s**.
  - **Status:** Completed (commit `3cec55b`).

