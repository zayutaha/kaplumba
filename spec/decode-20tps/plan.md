# Plan: Deccode to 20 t/s

## Evidence (from deep-dive 2026-09-23)

- `reference/omlx` contains **zero** EXL3 code. It delegates to pip-installed
  `mlx_lm` (`omlx/models/llm.py:163,216`), whose decode is stock
  `mx.quantized_matmul` — not an EXL3 path. Only custom kernels = Bonsai.
  **The EXL3-on-Apple reference is `reference/PonyExl3` alone.**
- PonyExl3 production decode kernel (`ponyexl3/mlx/gemv_metal.py` v12, MT=1):
  - 128 threads/TG, `NSG = 4` simdgroups stride in-tiles (`tk += 4`), zero
    barriers in the hot loop;
  - **8 register accumulators per lane** (`acc[8]`), static funnel windows
    (`ig0/ig1/s3`) computed once, tile-invariant;
  - direct device word reads (the ds4 pattern; a tile's 128–256 B is L1-hot),
    **not** coalesced-with-shuffle (which they call out: two-word ladder only
    for k≥5, and it loses);
  - x fetch per row via `simd_shuffle` (register-speed, no tg memory);
  - split-K on the **in-tile** axis via `grid.z`, epilogue writes partials, one
    conflict-free cross-SG reduction per split in-kernel (`tg_w[NSG][256]`).
  - Split chooser (`gemv_metal.py:1088,1136,1405`): double `n_splits` while
    `tiles_available // (splits * 2) >= min_split` with min_split = 128 (had
    path) and `out_tiles * n_splits < 8192`; then enforce
    `tiles_per_split % 8 == 0`. Our current 4096-TG formula with no 128 floor
    under-splits nothing but sets in-tiles/split to values (q=84, k/v=42) the
    reference deems cache-hostile.
  - measured v12 gemv ~486 Gw/s on M5 Max (~546 GB/s). Normalized to M4 Pro
    (273 GB/s): ~243 Gw/s ≈ 91 GB/s at 3 bpw. Community M3 Max (400 GB/s)
    numbers imply ~158 GB/s ≈ 39% of bus. **Practical plain-decode ceiling on
    M4 Pro ≈ 40–45% of bus ≈ 105–120 GB/s → ~9.5–11 t/s.**
- Our `exl3_gemv` (`crates/kernels/src/exl3.metal:189`) is qualitatively the
  same core: same 64-bit funnel decode, same x-shuffle, same direct loads, same
  in-tile split-K (via `grid.y`), host-side cross-split reduce. Differences:
  **32-thread TG (1 SG) vs 128 (4 SG); 2 accumulators/lane vs 8; split formula
  4096-div-ceil without the 128-in-tile floor; separate had_in host pass vs
  in-kernel prologue.** Measured 168 Gw/s (gate) = 63 GB/s.
- Proven-poison mechanisms in our own benches, to avoid: T5.3r coalesced word
  bursts + shuffle ladder (regressed all), T5.1r dynamic-index prefetch ring
  (regressed all). Note T5.1 was *not* PonyExl3's clean geometry — it used
  dynamic indexing and coalescing; a clean 128-thread/8-acc/static-funnel
  kernel has **never** been measured on this box.

## The bus math

- Per-token weight stream 11.2 GB (T5.0b, user-confirmed). 20 t/s = 224 GB/s =
  82% of a 273 GB/s bus. No M-series EXL3 binary sustains > ~45%. Therefore
  **20 t/s plain is not physically reachable on this hardware**; the plan
  splits into (P0–P2) reach the reference parity for plain decode (~10–11 t/s)
  and (P3) cross the ceiling with verify-gated speculation (emit >1 verified
  token per weight-stream read; PonyExl3's DFlash multiplier 2.28× on M5 Max).

## Architecture / flow

- P0 — ground truth: two stream-sum Metal kernels (contiguous + gemv-strided
  grid) over a synthetic buffer. Decides whether 79 GB/s is a kernel-compute
  wall or a memory-pattern wall. Single gated bench test, no engine changes.
- P1-T1 — parity kernel: rewrite `exl3_gemv` to 128-thread/4-SG geometry,
  8 accs, static funnels, direct loads, in-kernel cross-SG reduce. Keep the
  `aligned`-word path and the k=7 pair-table fallback. Grid: `(out_tiles,
  nsplit)` on x,y as today (z is unused by our gemv; callers unchanged).
- P1-T2 — nsplit chooser → reference formula (8192 target, 128 floor, %8
  divisor) validated by re-running `bench_gemv_split_sweep_slow`; bake argmax.
- P1-T3 — had_in prologue in-kernel for rows=1 (and had_out epilogue if it
  measures; see v18 `_gemv_post_source`, one TG owns 128-col block). Reuses the
  existing 128-point FWT pattern (T4 had logic / CPU `had_r_128`).
- P2 — end-to-end gated decode bench with swap sampling + phase split;
  amend ceiling spec only with measured numbers.
- P3 — new spec `spec/verify-gated-spec-dec`. Draft ~0.5–2B dense, greedy-
  identical verify, acceptance % measured on the real artifact; then
  `decode_throughput_slow` ≥19.5 flips on proof.

## Files/touchpoints

- `crates/kernels/src/exl3.metal` — `exl3_gemv` (P1-T1, P1-T3).
- `crates/kernels/src/exl3.rs` — `gemv_splits` chooser (P1-T2),
  `encode_gemv` grid/params if split-axis changes, tests.
- `crates/engine/src/metal_linear.rs` — `bench_passes_slow`,
  `bench_gemv_split_sweep_slow`, new `bench_stream_bw_slow` (P0).
- `crates/engine/src/gemma4.rs` — `decode_throughput_slow` numbers + gate
  (P2), roofline print.
- `crates/engine/src/gpu_decode.rs` — dispatch structure reference for P2
  phase split; target of P3 integration.
- Specs: `spec/decode-20tps/*`, amend `spec/final-same-algorithm-ceiling`
  (P2), new `spec/verify-gated-spec-dec` (P3).

## Sequencing / dependencies

P0 → P1-T1 → P1-T2 → P1-T3 → P2 (each gated, committed, revertable). P3
independent once P2 baseline stands. P1-T3 depends on P1-T1's grid being
stable. No prefill dependency.

## Risks / fallbacks

- 128-thread/8-acc may regress register/occupancy on M4 Pro (user-accepted).
  Fallback: 2 SG (64 threads) variations; or stay 32-thread and spend the
  remaining budget purely in P3.
- The had epilogue may not pay (heavy when fused). Keep had_in-only as the
  default; measure epilogue before keeping.
- P3 acceptance may be < 1.5× on this artifact → still >15 t/s; then iterate
  draft size/architecture, never loosen the verify gate.
- Ceiling amendment must use measured, not predicted, numbers.

## Requirement → task map

- R1 → P0. R2 → P1-T1. R3 → P1-T2. R4 → P1-T3. R5 → P2. R6 → P3 spec + impl.