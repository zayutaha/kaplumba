# TASKS: 4 / 8 bpw EXL3 Support

Branch `wt-bpw4` (isolated). Update this file after every task with status + commit hash.

- [x] **T1 — Rate-general colmajor repack, and stop it failing open** — `098cfc8`
  - Files: `crates/engine/src/exl3.rs` (`reorder_tensor_to_colmajor` takes
    `words_per_tile`; new `trellis_words_per_tile`), `crates/engine/src/bin/repack_model.rs`
    (derive width from shape, hard-error on a malformed trellis, write `layout: colmajor`
    only after `repacked == total`), 11 call sites, new
    `crates/engine/tests/colmajor_rate_general.rs`.
  - Testing: 6/6 new, `repacked_golden` 2/2, `rate4_colmajor_bit_exact` 6/6,
    `live_kernel_bit_exact` 1/1. Sabotage both directions observed: tile off-by-one
    → 3/6 fail; width ignored and 48 hardcoded → 4/6 fail, and `repacked_golden` still
    passes, which is the point.

- [ ] **T2 — Verify 4 bpw decode on real weights** — BLOCKED, GPU held by another agent
  - Files: none. Run only.
  - ```bash
    KAPLUMBA_TEST_MODEL=~/.kaplumba/models/Gxma-bpw4-shim \
      cargo test --release --test rate4_8_real_weights_bit_exact -- --ignored
    ```
  - GPU: needs idle + ~2 GiB, no model load. At time of writing `scripts/gpuq status`
    reported `gpu: load .../Gxma-repacked (pid 58889)`, 0.1 GiB available. AGENTS.md §10
    forbids a second GPU job, so this was NOT run and is NOT claimed.

- [ ] **T3 — Decide production dispatch for the 4/6 bpw colmajor kernels** — open question
  - The `_bench` suffix is a deliberate interlock, asserted by
    `the_engine_cannot_dispatch_the_four_bpw_measurement_kernel`. Recommendation: keep it.
    The shim is row-major and does not need colmajor dispatch.

- [ ] **T4 — Optional: repack the shim to colmajor at 4 bpw** — not started
  - Exercises the new path on real data. ~20 GB out, 58 GiB free. Only worth it if T3
    goes the other way.
