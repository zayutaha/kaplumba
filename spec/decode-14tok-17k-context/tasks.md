# Implementation Tasks: 14+ tok/s Decode Throughput at 17.5k Context

- [x] **Task 1: Hoist GEMV loads in `linear_gemv_ishizuki_k3_fused2_co`** (`commit 32830cb`)
  - Files touched: `crates/kernels/src/linear_repacked.metal`
  - Result: Concurrently issued Col A and Col B weight words before dequantization. Passed bit-exact tests.
- [x] **Task 2: Profile decode command buffer batch sizes** — DONE, NULL at long context
  - Files touched: none (env sweep only)
  - Result at 6k: 50 -> 73.0 ms, 150 -> 72.5 ms, 270 -> 71.9 ms, 600 -> 72.3 ms (~1.2 ms).
  - Result at 17,486 pinned (the context that matters): 50 -> 77.5-78.3 ms,
    270 -> 77.5-78.4 ms. **No effect; the ~7.6 ms queue gap of Finding 3 does
    not reproduce.** `KAPLUMBA_DECODE_SPLITS=16` at the same context is *worse*
    (78.0-78.8 ms) than the shipped 35. No default changed.
  - See `implementation_notes.md` §19-§20.

- [x] **Task 6: Pin only the `.trellis` pages so the box stops re-reading them**
  - Files touched: `crates/engine/src/weights.rs` (`pin_trellis_pages`, called
    from `load_internal`)
  - Result: 17,486 context **0.37 -> 12.8 tok/s** (2 700 ms -> 77.6-78.9 ms per
    step); 11.21 GiB wired of the 14.9 GiB of shards, against 21 GiB for the
    rejected whole-mapping version. Offloads untouched. Greedy output
    byte-identical; 102/102 engine lib tests pass.
  - **This is the root cause of the 0.03 tok/s class of stall, and it is not the
    in-flight window.** See `implementation_notes.md` §19.

- [x] **Task 7: Attribute the first decode step**
  - Files touched: `crates/engine/src/chat.rs` (encode/wait trace under
    `KAPLUMBA_TOKEN_TIMING`)
  - Result: host encode is 2 ms; the GPU wait is the whole cost. Token 2 is
    3 162 -> 873 ms with the trellis pinned, against a 78 ms steady state.
    **Still open** — pinning the whole mapping reached 445 ms, so the residual is
    the compressible arena + KV.
- [x] **Task 3: Implement software-pipelined K-loop in `linear_gemv_ishizuki_k3_fused2_co`** — DONE, NULL
  - Files touched: `crates/kernels/src/linear_repacked.metal` (reverted)
  - Result: prefetching trip N+1's 8 weight words ahead of trip N's arithmetic is
    **bit-exact on the real model** (identical greedy token IDs at 6k and 17.5k)
    but **neutral**: 6k 72.0 vs 71.9 ms, 17.5k 77.8 vs 78.0 ms. Reverted.
  - **Coverage gap found:** sabotaging the prefetch left `fused_bit_exact` 3/3
    green, because that suite routes through `encode_gemv_fused2` -> `fused2_2col`
    while the live MLP path is `encode_gemv_fused2_mlp` -> `fused2_co`
    (`KAPLUMBA_TRACE_GEMV=1` confirms both dispatch). `_co` has **no synthetic
    bit-exact test**; the token-ID A/B is its only evidence. Adding one is open.
  - A follow-up should also fix the 4 stale `probe_unroll_kloop_dag` failures on
    HEAD (instrument keeps expecting `(4, 8)`; `32830cb` shipped `(8, 8)`).
  - See `implementation_notes.md` §22-§24.

- [x] **Task 8: Decide whether the arena and KV cache should be pinned too**
  - Result: **No.** Arena `mlock` pinned only +0.02 GiB of 1654 segments and
    moved token 2 from 608 to 508 ms -- inside the 445-873 ms run-to-run band.
    KV is ~2.6 GiB at 16k and `vm.user_wire_limit` is 18.24 GiB against 11.21 GiB
    already pinned plus ~3.3 GiB of OS wired pages, with the ladder reallocating
    on every rung. See `implementation_notes.md` §23.
- [ ] **Task 4: Investigate and eliminate Token 2 post-prefill stall** — PARTIAL
  - Files touched: `crates/engine/src/chat.rs`, `crates/engine/src/model.rs`
  - Measured: host encode 2 ms, GPU wait 1560 ms (6k) / 3162 ms (17.5k).
  - Tried and **refuted**: a load-time warm-up of the pipelined decode path
    (both slots) made it worse (2432 ms). Pinning the whole mapping fixed it
    (445 ms) but is unsafe.
  - Remaining: 873 ms at 17.5k with the trellis pinned. The arena + KV are the
    last compressible pages.
- [ ] **Task 5: End-to-end 17.5k context chat benchmark**
  - Run `/tmp/test_16k_tokens.py` and verify steady-state delta drops below 71.4 ms ($\ge 14.0\text{ tok/s}$).
