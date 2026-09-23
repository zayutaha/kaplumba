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

## The real wall: decode ALU (2026-09-23, post-P1-T1)

- Per-token weight stream 11.2 GB (T5.0b). 20 t/s = 224 GB/s = 82% of a
  273 GB/s bus — but the bus is NOT the binding constraint.
- **ALU arithmetic**: per 16-weight tile ≈ 8 load + funnel ~8, extraction
  8×(u64 shift + and) ≈ 16–24, `exl3_decode_f` ≈ 7 ops × 8 ≈ 56, 16 fma,
  4 shuffles, loop ≈ 4 → **~100+ ops / 16 weights ≈ 6–7 ops/weight**.
  31G weights → ≈ 200G ALU ops/token. M4 Pro GPU int/fp32 ≈ 1.4–2.0T ops/s
  → **plain-decode ceiling ≈ 7–10 t/s. Measured 142 ms / 7.03 t/s is ON it.**
  (The P0 "20 t/s plain credible, 224 GB/s" conclusion was wrong: the strided
  probe reads 380 GB/s with trivial ALU, but the gemv adds ~6.5 ops per byte
  read; ALU binds first.)
- 20 t/s plain would need ~3× ALU reduction = f16x2 packed accumulate, blocked
  by no-quality-loss. **Hence verify-gated speculation (P3) is the PRIMARY
  route**, and the mechanism that crosses is **row-batched target-verify via
  the `exl3_gemm_sg` tensor-core GEMM (≤8 rows)**: decode ALU is spent once
  per weight tile and the simdgroup units do the per-row MACs → per-position
  ALU ≈ decode/8 + 1 MAC ≈ 2 ops/weight/position ≈ 25–40 ms/step for ≤8 draft
  positions, before the ≥1-token/step guarantee (self-consistent greedy) and
  the ~1.5–3× drafted acceptance multiple.

## Architecture / flow

- P0 — ground truth: two stream-sum Metal kernels (contiguous + gemv-strided
  grid) over a synthetic buffer. Decides whether 79 GB/s is a kernel-compute
  wall or a memory-pattern wall. **Done (5e60272)**: not a memory wall (strided
  214–381 GB/s) — in-kernel compute.
- P1-T1 — v12-parity A/B kernel `exl3_gemv4` (128-thread/4-SG geometry, 8 accs,
  static funnels, direct loads, in-kernel cross-SG reduce) built alongside v1.
  **Done (3c571bd) — NEGATIVE**: loses at every nsplit; v1 stays; kernel kept
  as regression bench. Also eliminated dependency-latency as the limiter.
- P1-T2 — nsplit chooser updated to the sweep argmax (gate n2, down n8, o n8,
  q n2, k/v n8, lm n1) — baked override table + formula fallback. ~1–2 ms/token
  expected. Previous reference-formula floor (128 in-tiles/split) contradicts
  the measured k/v argmax (n8), so the argmax table (T5.2r philosophy) wins.
- P1-T3 — had_in prologue in-kernel for rows=1 (and had_out epilogue if it
  measures; see v18 `_gemv_post_source`, one TG owns 128-col block). Reuses the
  existing 128-point FWT pattern (T4 had logic / CPU `had_r_128`).
- P2 — end-to-end gated decode bench with swap sampling + phase split;
  amend ceiling spec only with measured numbers.
- P3 (**primary**) — new spec `spec/verify-gated-spec-dec`. Draft =
  shallow-layer target draft (target's own first 2–4 layers + reduced head,
  zero extra weights), row-batched target-verify on `exl3_gemm_sg`, greedy-
  identical only; acceptance % measured on the real artifact; then
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

- The v12-parity shape (128-thread/8-acc) is disproven on this hardware →
  no longer a fallback; the register/occupancy risk is moot (kernel kept
  bench-only).
- The had epilogue may not pay (heavy when fused). Keep had_in-only as the
  default; measure epilogue before keeping.
- Decode ALU is the plain-decode wall; any further plain-decode win must cut
  ops/weight (e.g. cheaper funnel), not repartition — budget is ~1.3× before
  running out; do not sink far into it.
- P3 is the primary route but depends on: (a) row-batched verify GEMM (8 rows)
  actually amortizing decode on M4 Pro (verify with a 2-row E2E probe before
  full build), (b) shallow-draft acceptance ≥ ~1.5× on the artifact (iterate
  draft depth/head; never loosen the verify gate).
- P3 acceptance may be < 1.5× on this artifact → still >15 t/s; then iterate
  draft size/architecture, never loosen the verify gate.
- Ceiling amendment must use measured, not predicted, numbers.

## Requirement → task map

- R1 → P0. R2 → P1-T1. R3 → P1-T2. R4 → P1-T3. R5 → P2. R6 → P3 spec + impl.