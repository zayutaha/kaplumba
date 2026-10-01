# TASKS: 3 bpw Lane-Load Amplification — 18 tok/s Decode

Order matters: T1 is a **gate**. If it returns a negative, stop and record it — a
measured negative is a valid outcome and cheaper than any production code.
Per AGENTS.md §3.3, update this file after every completed task with status + commit hash.

- [ ] **T1 — Gating micro-experiment: does a lane-contiguous 3 bpw tile actually reach ~250 GB/s?**
  - Files: `crates/engine/examples/bench_rate_sweep.rs` only. **No production code.**
  - Intent: `spec.md` §2.2 derives by hand that placing lane L's 8 codes at bit offset `24·L`
    needs 48 lane-loads per 24-word tile (amp 2.000) at an unchanged 96 B/tile, predicting
    ~250 GB/s. Implement that permutation *inside the harness* (not in `exl3.rs`), add a
    `lanecontig` 3 bpw arm beside the existing `@3bpw` arm, and measure. Also measure the
    `straddle` branch both ways (predicated 1-or-2 loads vs branch-free always-2) because
    a lane-dependent branch previously cost −15…−20% in this kernel
    (`decode-25tok-peak-bandwidth/implementation_notes.md:54-56`).
  - Testing:
    - Amplification reported as an **exact rational** derived from the arm's own addressing
      arithmetic, in the style of `lane_load_amplification_is_2_667_at_3_bpw_and_2_000_at_4_and_6`
      (`rate4_colmajor_bit_exact.rs:682`). Must read 48/24 = 2.000, not "~2".
    - Tile size asserted **96 B**, identical to the `@3bpw` arm. Any padding shows up here.
    - **Sabotage:** perturb one lane's bit offset by 1 and confirm the amplification
      assertion fails with a message naming the tile and lane. Revert. Record the failure.
    - GB/s medians over ≥ 9 interleaved rounds, **min-based estimator**, repeatability
      floor per cell, `gpuq status` showing `gpu: none` first.
  - **Decision rule** (from `plan.md` §4):
    - ≥ 235 GB/s → proceed to T2. The spec's premise holds.
    - 210–235 GB/s → proceed, but re-scope the ≥18 tok/s target before starting T2.
    - < 210 GB/s → **stop.** 3 bpw is retained; write the number into `implementation_notes.md`
      and close the spec. Do not write T2.
  - Commit: `spec/decode-3bpw-amplification: T1 gating lanecontig measurement`

- [ ] **T2 — Productionise the permutation as a provably lossless repack**
  - Files: `crates/engine/src/exl3.rs` (new fns, adjacent to `extract_ishizuki_lane_codes:664`),
    `crates/engine/src/bin/repack_model.rs` (`--layout lanecontig`), new test file
    `crates/engine/tests/lanecontig_repack.rs`
  - Intent: promote T1's permutation to `repack_tile_lanecontig` **and its inverse**,
    applied tile-wise while preserving column-major tile order. Reuse
    `extract_ishizuki_lane_codes` unchanged. **Do not** touch `reorder_tensor_to_colmajor:733`
    and **do not** revive the lossy `repack_trellis_3bpw_tile:687`.
  - Testing:
    - Round-trip: `inverse(forward(tile)) == tile` for all 256 codes, over many **real**
      tiles from `Gxma-repacked` — not synthetic, to avoid the running-LCG trap
      (`rate4_appendix:155-166`).
    - Dequantized values bit-identical to `extract_ishizuki_lane_codes` output, per lane.
    - Output byte count **equal** to the colmajor output byte count (proves R7/no-byte-growth).
    - `quantization_config.json` gains `layout: "lanecontig"`; the loader's
      `packed_offset % 4` check (`linear.rs:1353`) still passes — 64-byte alignment in
      `repack_model.rs:188` must be preserved.
    - **Sabotage, both directions:** flip one bit in the forward permutation and in the
      inverse separately; each must fail the round-trip with a diagnosable message. Record both.
  - Commit: `spec/decode-3bpw-amplification: T2 lossless lanecontig repack + inverse`

- [ ] **T3 — `_lanecontig` decode kernels, bit-exact**
  - Files: `crates/kernels/src/linear_repacked.metal` (add
    `extract_v_mcg_lanecontig` + `_lanecontig` variants of `:580-715` `fused2_co`,
    `:1221-1384` `fused3_2col`, `:1782-1882` `contig4`, `:1884-2018` `colmajor_2col`),
    `crates/engine/tests/live_kernel_bit_exact.rs`, `crates/engine/tests/rate4_colmajor_bit_exact.rs`
  - Intent: new variants, **not** in-place edits, so the default path cannot regress.
    Addressing per `plan.md` §2.3. `w0 <= 23` always and `w0+1` is read only when
    `straddle`, so no out-of-bounds word at the tile seam. Shifts become immediates
    (`>> (24 - 3i - 2)`), so `extract_v_mcg_32`'s 64-bit funnel shift is no longer on
    the path.
  - Testing:
    - `live_kernel_bit_exact`: `_lanecontig` fp16-**bit-identical** to `_colmajor` on the
      three real shapes (down 21504×5376, gate 5376×21504, o 4096×5376). Not "close" — identical.
    - `rate4_colmajor_bit_exact`: add the amp ≤ 2.0 assertion; do **not** modify existing ones.
    - Register/occupancy check: report thread-register count if the Metal toolchain is
      available. It was **absent** on this box (`xcrun -sdk macosx metal` → "missing Metal
      Toolchain"), so if still absent, record that and do not invent numbers.
    - **Sabotage:** skip one lane's second word in the straddle case; expect a diagnosable
      mismatch naming the lane. Revert, record.
  - Commit: `spec/decode-3bpw-amplification: T3 lanecontig kernels bit-exact`

- [ ] **T4 — Dispatch wiring behind `KAPLUMBA_LANECONTIG` (default off)**
  - Files: `crates/kernels/src/linear.rs` (env gate alongside `:45-58`; routing in
    `:1915-2056` `encode_gemv_raw`, `:2221-2399` `encode_gemv_fused2_mlp`, `:2402-2556` `encode_gemv_fused3`)
  - Intent: route gate_up / down_proj / o_proj / qkv to `_lanecontig` only when the layout
    key is `lanecontig` **and** the flag is set. `lm_head` is 6 bpw on the EXL3 arm and is
    **out of scope** (`linear.rs:2012-2038`, trellis at `%4==2` per `linear.rs:1353`).
  - Testing:
    - Flag off ⇒ byte-identical dispatch selection to today. `KAPLUMBA_TRACE_GEMV`
      (`linear.rs:1779`) diffed before/after.
    - Flag on ⇒ `KAPLUMBA_TRACE_GEMV` names the `_lanecontig` kernels for exactly the four
      projections and nothing else.
    - **Sabotage:** set the flag while pointing at a colmajor checkpoint; must fail loudly
      (layout mismatch), not silently compute garbage. Record.
    - End-to-end bit-exact golden text with the flag on.
  - Commit: `spec/decode-3bpw-amplification: T4 gated dispatch wiring`

- [ ] **T5 — End-to-end acceptance**
  - Files: `spec/decode-3bpw-amplification/implementation_notes.md`
  - Intent: prove or disprove AC1–AC6.
  - Testing:
    - **AC1:** ≥ 18.0 tok/s (≤ 55.5 ms/step), median of **≥ 8 interleaved paired A/B runs**
      on a ≥ 128-token generation, spread reported. This is the gate; bandwidth is reported
      but not gated. One reading is not a result.
    - **AC2:** "The capital of France is Paris." with identical token IDs; 0 fp16 divergence.
    - **AC3:** `live_kernel_bit_exact`, `fused_bit_exact`, `repacked_golden`,
      `argmax_engine_parity` all 0 failures.
    - **AC5:** `/usr/bin/time -l` `peak memory footprint` ≤ 16.5 GiB, 0 MiB swap.
      **Not** `peak_resident_pages_bytes` (AGENTS.md §10.3).
    - **AC6:** flag off reproduces today's 15.3 tok/s bit-exactly.
    - Baseline re-measured in the same session, interleaved, so drift cannot be mistaken
      for a win.
  - Commit: `spec/decode-3bpw-amplification: T5 end-to-end acceptance`
