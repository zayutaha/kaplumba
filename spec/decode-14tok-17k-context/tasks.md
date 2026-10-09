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

---

## Session of 2026-10-08

- [x] **Task 9: Undo the `e1b05c1` residency regression** — 0.37 -> 13.1 tok/s at 16.5k
  - Files: `crates/engine/src/weights.rs`
  - `offload_trellis` was gated on `!dedicated`, and production always loads via
    `load_dedicated`, so all 12.04 GiB of trellis went into dirty anonymous Metal
    shared buffers and got compressed (swap 4.35 GiB, compressor 8.1 GiB).
  - Dropped `!dedicated`; restored `pin_trellis_pages` (`mlock` on the 411
    `.trellis` ranges only, offloads untouched). Measured: **2,721 -> 76.0 ms**.
  - Amends `spec/green-memory-residency` R1; R1 and its own R5 are unsatisfiable
    together on a 24 GiB box. See notes §25.
  - New regression guard `trellis_stays_zero_copy_even_in_the_dedicated_loader`,
    sabotaged three ways, all observed to fail.

- [x] **Task 10: Two bit-exact grid-shape wins, shipped as defaults** — 13.1 -> 13.40 tok/s
  - Files: `crates/kernels/src/linear.rs`
  - `KAPLUMBA_SK2_1COL` (now default, opt out `KAPLUMBA_NO_SK2_1COL`):
    `_1col_sk2` doubles the decode GEMV grid to `out_features/16*2`.
  - `KAPLUMBA_QKV_SK2` (now default, opt out `KAPLUMBA_NO_QKV_SK2`): fused q/k/v
    go through `fused2_sk2`/`fused3_sk2`, 2048 groups against 512.
  - Six-arm sweep, all arms in notes §27. Both bit-exact (90/90 token ids
    identical); `fused_bit_exact` stays 3/3.
  - Also: `fused2_sk2` never called `trace_gemv`, so the split was invisible to
    `KAPLUMBA_TRACE_GEMV`. Fixed.
  - Probe `probe_mlpdispatch_tree` updated to the new defaults (it was 19/19 green,
    so the 2 failures it threw came from the default flip and are now fixed).

- [x] **Task 11: Fix the documented `linear.metal:1187` under-parallelisation**
  - Files: `crates/kernels/src/linear.metal`, new `crates/engine/tests/probe_finalize_block_stride.rs`
  - A 1024-thread dispatch running `b += 8u`: 24 of 32 simdgroups idle, 6 rounds
    instead of 2. Now `b += 32u`, matching its three siblings.
  - Bit-exact by construction (pure elementwise map, disjoint output slots).
  - New probe pins **stride == threads/32** for all nine block loops, both
    directions; sabotaged three ways (8u, 64u, and over-striding a 256-thread
    kernel) and every failure observed.

- [ ] **Task 5: End-to-end 16.5k context chat benchmark — 74.48 ms (13.40 tok/s)**
  - Target 71.4 ms / 14.0 tok/s **not reached** and, on the evidence in §26, not
    reachable at `--temperature 0` without lifting a standing constraint:
    attention is at the DRAM read roofline (3.54 GB / 11.95 ms = ~296 GB/s) and
    the weight GEMV is at ~198 GB/s against a 203 GB/s measured ceiling on a
    bit-locked ALU pool. Element/finalize/host holds at most ~0.5 ms more.
  - The only lever big enough is byte reduction (`lm_head` at 6 bpw costs 4.18
    ms/token), which is a quality decision. Logged, not taken.

- [x] **Task 12: Sweep the decode attention split count for this checkpoint** — 74.49 -> 73.86 ms
  - Files: `crates/kernels/src/attention.rs`
  - The full-attention arm shipped `key_count.div_ceil(512)` = 35 splits at 16.5k,
    i.e. 128 workgroups against a **measured one-wave budget of 54**. The
    workgroup rule gives 54/4 = **13** and keeps 52 workgroups in one wave.
  - The model-free bench had already found 13 best of `{8,12,13,16,20,24}` for
    exactly this shape; the shipped 32 was never in the sweep.
  - Sliding layers unchanged (1024-deep ring never satisfies `key_count > 1024`).
  - Measured 13.42 -> 13.55 tok/s; greedy token ids identical 90/90.
  - **Caveat recorded, not hidden:** `splitk_bit_exact.rs:115` asserts split counts
    *do* reassociate the f32 partials, so the identity is masked by fp16 rounding,
    not structural. `spec/prefill-24k-90tps` required the gate amended in writing
    before this flip; notes §30 says so.

- [ ] **Task 5 (final): 16.5k chat benchmark = 73.86 ms / 13.55 tok/s** — target
  71.4 ms **not reached**, and notes §31 records the measured wall rather than a
  guess: attention holds <=1.2 ms more, element holds ~0.2 ms and it is the
  bit-exactness-critical reduction tree, the GEMV is at ~198 of a 203 GB/s
  measured ceiling on a bit-locked ALU pool.

- [x] **Task 13: Stop chat paying for a speculative decode step nobody reads** — session 14.06 -> 14.86 tok/s
  - Files: `crates/engine/src/chat.rs`
  - The pipelined loop built *and committed* step N+1 before knowing whether
    token N was EOS, so every EOS-terminated turn submitted and waited on a step
    whose result is discarded. Localised as a **constant 132 ms per turn**
    (outside the timed loop, identical across turns of 7/14/10/2 tokens, absent
    on the cap-terminated turn 1).
  - Now the encoder is built but committed only after the EOS check.
    `MetalEncoder::Drop` calls `end()` only, so an uncommitted encoder is never
    submitted. Host encode still overlaps the previous step's GPU work.
  - 132 -> **66 ms** per turn (the irreducible EOS-discovery step); last turn of
    a 5-turn chat 8.52 -> **11.55 tok/s**.
  - Bit-exact: 4/4 `multi_turn_chat` model tests pass, including
    `pipelined_and_barrier_decode_agree_across_a_multi_turn_session`, and
    **90/90 greedy token ids identical** at 16.5k with steady state unchanged.
  - `generate.rs` has the same waste and is still unfixed (noted in §35).

- [x] **Task 14: `decode_rate_is_sustained_across_a_growing_session` measured
      the prompt, not the engine**
  - Files: `crates/engine/tests/multi_turn_chat.rs`
  - Failed at **8.26 tok/s on the parent commit** (verified in a worktree at
    `c4236e1`), so pre-existing, not this work's.
  - Its last turn generated **2 tokens** against a 12.0 tok/s floor; with the
    irreducible EOS-discovery step that turn caps at **10.2 tok/s with zero other
    overhead**, while the first turn generates 48 and runs at 15.3.
  - Changed the last turn to a counting prompt so token counts are comparable and
    the assertion is a decode-rate signal. **Did not lower the floor** — it still
    catches a broken pipelined loop.
