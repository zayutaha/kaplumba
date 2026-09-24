# Implementation Notes: Pure Decode 273 GB/s Bandwidth Saturation

## T1 Completion Audit (2026-09-23, session resumption)

State found in the working tree: T1 kernels, bindings, and engine rewire were
already written but uncommitted and untested (session died mid-T1). Full audit
results:

- **MSL kernels present and structurally complete** (`exl3.metal`):
  - `exl3_gemv_post` (L1806): single-linear GEMV, in-threadgroup SRAM butterfly,
    direct f16 out. Body is a faithful copy of `exl3_gemv`'s dot-product
    (dq4 64-bit window path for k≠7 incl. the aligned+tw≤32 `simd_shuffle`
    sub-branch, pair-table funnel for k=7) + `exl3_had_out`'s epilogue.
  - `exl3_gemv_post_fused2` (L2107) / `exl3_gemv_post_fused3` (L2316): same,
    with per-tile trellis pointer select (`is_b ? trellis_b : trellis_a`,
    3-way `sub` select) and `xh` row offset per sibling.
  - `exl3_gemv_reduce_post` (L1115): sums K-split partials `[nsplit, ot, 16]`
    in split order → f16 round → butterfly → svh → f16 out (128 threads/TG,
    8 out-tiles per TG).
- **Bindings present** (`exl3.rs`): pipelines `gemv_post`, `gemv_post_fused2`,
  `gemv_post_fused3`, `gemv_reduce_post` built in `Exl3Gemv::new`; public
  `encode_gemv_post` / `encode_gemv_post_fused2` / `encode_gemv_post_fused3` /
  `encode_gemv_reduce_post`; `encode_linear_with_scratch` rows=1 branch rewired
  to had_in → (`gemv_post` | `gemv` + `gemv_reduce_post`); fused2/fused3
  `*_with_scratch` entries now end in the post-epilogue kernels (no y, no
  had_out dispatch).
- **Param byte-layouts verified against `set_bytes` sizes**: `Exl3Params` 7×u32
  = 28 B, `Exl3Fused2Params` 8×u32 = 32 B, `Exl3Fused3Params` 9×u32 = 36 B.
  `exl3_gemv_post` never reads `p.nsplit` (harmless trailing member, keeps the
  struct identical to `exl3_gemv`'s).
- **Dispatch geometry**: post kernels run 8 out-tiles per 256-thread TG
  (8 warps × 32 lanes, warp w = out-tile, lanes = columns); butterfly runs on
  tid<128 after all 8 warps fill `sh[128]`. Grid = out_tiles.div_ceil(8).
  `exl3_gemv` stays WARPS=1/32 threads (new rows=1 K-split dispatch in the
  rows=1 branch matches `encode_gemv`'s geometry).
- **Rounding-point equivalence argument (why bit-exactness is preserved)**:
  old path rounds the *full* f32 column sum to f16 inside `had_out`
  (`sh[t] = float((half)y[...])`) before the butterfly; new paths round the
  same full sum (nsplit=1: in-kernel `float((half)accA/accB)` after the
  simd_shuffle_down collapse; nsplit>1: `gemv_reduce_post` sums the same
  partials in the same split order as `gemv_reduce` before rounding). Same
  products, same per-split accumulation order, same reduce order, same single
  rounding point → bit-identical f16 outputs.
- **Split-policy consequence**: with the P1-T2 argmax table (gate 2, q 2,
  down/o 8, k/v 8, lm 1) every decode linear *except lm_head* takes the
  `gemv + gemv_reduce_post` route; lm_head (16384 out-tiles → nsplit=1) is the
  only `exl3_gemv_post` caller in production. Both routes must therefore be
  unit-covered — see T1 test below.
- **Arena sizing**: `chunk_scratch_needs` + `gemv_partial_bytes` already
  single-source the `partial` requirement (`metal_linear.rs` maxes it in), so
  the decoder arena covers the new rows=1 K-split partial use without change.
- **Gap found by the audit**: T1's unit test (`gpu_gemv_post_matches_unfused`)
  did not exist yet — written in this session, see T1 entry in tasks.md.
  Note `gpu_encode_fused2_and_fused3_matches_separate_linears` (T5) now
  implicitly exercises the post epilogue too, because its separate/fused
  comparisons both route through the rewired `*_with_scratch` entries.

## Architecture Log & Live Measurements

- **Hardware**: Apple M4 Pro (20 GPU cores, 273 GB/s memory bandwidth).
- **Target Model**: Gemma 4 31B EXL3 (~3.0 bpw). Total weights per token: 11.833 GB.
- **Baseline Measured State**: 144.5 ms/token = 6.92 t/s (81.89 GB/s = 29.99% bus utilization).
- **Primary Bottlenecks Documented**:
  1. Grid under-occupancy on narrow projections (`k`, `v` at 256 tiles yielding only 18.7 GB/s).
  2. 2,165 dispatches per token incurring ~26 ms of hardware bubble latency.
  3. Intermediate fp32 `y` DRAM round-trips for standalone `had_out` (420 dispatches).
- **Key Invariants & Proven Dead-Ends from History**:
  - Generic arbitrary-stride `simd_shuffle` tile weight ladders (T5.3r) are a net loss on M4 Pro due to 160-ALU barrel shuffle cost on 24-word tiles. Direct L1-resident loads are faster.
  - Weights must stay in their original memory-mapped positions without copying (resident memory limit ≤ 14.6 GB). Fused GEMVs use pointer dispatch.

## Tasks Completed (T2, T3, T4, T5)
- **T4 (`exl3_had_in_batched`)**: Multi-row pre-Hadamard operating on row-stacked `suh_stack`. Replaces 2 separate dispatches for Gate/Up and 3 separate dispatches for Q/K/V with a single multi-row launch (`grid.y = 2` or `grid.y = 3`). Verified bit-for-bit identical via `gpu_had_in_batched_matches_separate_passes`.
- **T2 (`exl3_gemv_fused2`)**: Sibling 2-way GEMV dispatching 2,688 out-tiles (Gate 1344 + Up 1344) over 20 cores (134.4 threadgroups/core). Reaches 107 Gw/s in release mode. Verified exact match against separate GEMVs via `gpu_gemv_fused2_matches_separate_passes`.
- **T3 (`exl3_gemv_fused3`)**: Sibling 3-way GEMV dispatching 1,024 out-tiles (Q 512 + K 256 + V 256). Expands K and V grid from 256 to 1,024 tiles (51.2 threadgroups/core), eliminating the 18.7 GB/s under-occupancy collapse. Measured latency dropped from 1.489 ms (separate Q+K+V) to 0.953 ms. Verified exact match via `gpu_gemv_fused3_matches_separate_passes`.
- **T5 (`Fused2Handles` & `Fused3Handles`)**: Zero-copy device handles created on model load with pre-stacked `suh_stack` and `svh_cat` buffers (~4.4 MB total model overhead). Implemented `encode_fused2_with_scratch` and `encode_fused3_with_scratch`. Verified exact match via `gpu_encode_fused2_and_fused3_matches_separate_linears`.
- **T6 (`gpu_decode.rs` integration & MSL loop fix)**:
  - Fixed critical loop shadowing bug in `exl3_gemv_fused2` and `exl3_gemv_fused3` (`uint it = it0` redeclared in inner unroll and tail loops, causing double loop execution and out-of-bounds weight tile reads).
  - Integrated `qkv_fused` (sliding layers) and `qk_fused` (full layers) and `gate_up_fused` into `GpuDecoder::encode_token`.
  - Allocated `qkv_raw` and `gu_fused` scratch arenas in `GpuDecoder`.
  - All 143 engine and kernel tests passed.
  - Verified 20-step chat prompt in `trace_chat_prompt_steps_slow` bit-for-bit with reference, predicting token 818 ("The") on step 19.
  - Measured decode throughput: 140.9 ms/token (7.10 t/s, 79.5 GB/s bandwidth).
  - Profile analysis: 1.28 ms/layer (76.8 ms/token — 54.5% of total decode time) is spent purely in 8 standalone `had_in` and `had_out` dispatches per layer (each taking ~0.16 ms). Fusing these into GEMV epilogue and producer prologues will recover ~76.8 ms and double throughput toward 20 t/s.

## P6 LUT-decode experiment (2026-09-24) — ABANDONED, measured negative

**Hypothesis**: the per-codeword integer mix (mode×cw mul → xor → half2 add, ~5-6
integer ALU ops / 16-bit window) could be replaced by one 2-byte gather from a
host-precomputed 65536×half LUT (128 KiB, L2-resident). Mix is k-independent,
mode-only, so `decode_table` on the host (exact same integer mix + one IEEE
binary16 add) is bit-identical by construction.

**Implementation (MSL + Rust + engine, then reverted — nothing committed)**:
- `exl3.metal`: `constant bool FC_LUT [[function_constant(3)]]`, helpers
  `exl3_decode_sel(cw, mode, cb)` / `exl3_decode_sel_const(cw, cb)` folding on
  FC_LUT (ternary folds at pipeline build), `device const half *codebook`
  buffer param appended to all 8 decode kernels (gemv@5, gemv4@5, spec@5,
  fused2@6, fused3@7, post@6, post_fused2@7, post_fused3@8). Because the
  generic kernels (exl3_gemv/_fused2/_fused3/_gemv4) now reference a function
  constant, `get_function(name, None)` → "function cannot be used to build a
  pipeline state"; all four generic pipelines must be built via
  `constantValues` with defaults (FC_ALIGNED=false, FC_MODE=0, FC_K=0,
  FC_LUT=false).
- `exl3.rs`: `decode_table(mode) -> Vec<f16>` (65536 entries), cached
  `codebook_buffer(mode)` (single 128 KiB shared buffer per mode), pipeline
  caches keyed (+bool lut), `set_decode_lut(bool)`.
- Engine: `MetalLinear::set_decode_lut`, fast bit-exact gate
  `gpu_decode_lut_matches_alu` (proved bit-identical across k=1..8, both
  modes, both alignments, and the full rows=1 encode / fused2 / fused3 paths),
  gated A/B `bench_decode_lut_slow`.

**Measured result (bench_decode_lut_slow, release, real artifact, min-of-12)**:

| linear | alu (ms) | lut (ms) | delta |
|--------|---------|---------|-------|
| down 21504x5376 k=3 | 0.845 | 1.350 | +59.7% |
| o 8192x5376 k=3 | 0.325 | 0.627 | +92.9% |
| lm 5376x262144 k=6 | 4.614 | 14.369 | +211.4% |
| gate+up (fused2) | 0.951 | 2.520 | +165.0% |
| qkv (fused3) | 0.488 | 1.118 | +129.0% |

LUT loses ~everywhere by 1.6-3.1×. The device-memory gathers (4/tile) thrash
L1/L2 (each lane pulls a random 2-byte value from a 128 KiB region; consecutive
k-splits sweep disjoint windows) and cost far more than the ~5 integer ALU ops
they replace — M4 Pro has abundant integer ALU throughput but a single shared
L2. Absolute times above are also inflated vs production decode by artifact
page-faults under swap pressure (used=3.0/4.0 GiB); RELATIVE result is valid.
LUT cannot fit `constant` address space (needs 128 KiB, constant space is
bounded at 64 KiB total per entry point), so no tighter variant exists.

**Decision**: abandon the LUT. Outputs would already be bit-identical (gate
proved), so flipping would have been safe — but it's ~2× SLOWER. This frees the
decode-stage ALU budget as an anti-knowledge: M4's integer units are not the
bottleneck at ~8.6 t/s; memory load/store path is. Next lever priorities
(revisited): (1) reduce per-layer dispatch/overhead further and attack
memory-side stalls (e.g. read-once weight streaming, larger in-threadgroup
working sets), (2) post-P5 nsplit re-sweep for down/o/k/v to remove
`gemv_reduce_post` round-trips, (3) re-check LM head (k=6, 5376→262144) split
policy — it dominates at ~5.9 ms/token.
Reverted via `git checkout --` the three touched files (exl3.metal, exl3.rs,
metal_linear.rs). HEAD remains `5e4e830` (P5). Fast suites green afterwards
(engine 124 + kernels 21).

## P6 continuation (2026-09-24, after LUT revert)

### Clean post-LUT baselines (fresh, release, HEAD 5e4e830)
- decode_throughput_slow (forward-only): **117.0 ms/token → 8.55 t/s**; e2e 8.53 t/s; implied 95.8 GB/s over 11.200 GB/token; roofline 24.4 t/s @ 273 GB/s.
- decode_layer_scaling_slow: fwd(n) = 6.78 + 1.837·n ms (60 layers → 110.2 ms); non-layer fixed a ≈ 6.8 ms ≈ lm_head (4.64 ms isolated) + embed + waits; L=0 → 5.53 ms/token.
- Roofline math: 50 ms/token (20 t/s) needs 224 GB/s → per-layer ≈ 0.75 ms ≈ 239 GB/s vs measured 1.837 ms/layer ≈ 97.6 GB/s in-context. lm fixed cost ≈ 4.6 ms must also drop proportionally (×~0.5).

### bench_linear_bandwidth_slow (engine, gated) — per-linear isolated, release, warm min-of-20, REAL trellis bytes
gate 0.530ms 81.8 GB/s; up 82.0; down 81.7; o 53.9; q 53.4; k 35.9; v 35.2; lm k6 **228.2 GB/s**; gu2 (fused) 93.0; qkv3 (fused) 69.3. NB: sweep bench prints GB/s=(in×out)/t which overstates by k/8 (weights-count not trellis bytes); scale ×k/8 for real.

### bench_gemv_split_sweep_slow rerun (release, min-of-12, v1 generic)
real GB/s ≈ shown × k/8: down n8 0.575·200→75, o n8 0.328·134→50, q n2 0.333·132→50, k/v ~85→32, lm n1 4.665·302→226. Policy (gate→2, q→2, down/o→8, k/v→8, lm→1, tiny→16) confirmed; no argmax changes.

### 4-way unroll experiment (exl3_gemv_spec k≠7 loop) — NEGATIVE, reverted
Widened 2-way→4-way (8 ld_word in flight per warp), validated by fast bit-exact gates (gpu_gemv_spec_matches_generic; fixture argmax 236773). Bandwidth vs 2-way: down 81.7 (81.8), o 53.9 (57.3), q 53.4 (54.6), k 35.9 (37.9), v 35.2 (36.9), lm 228.2 (227.8), gu2 93.0 (94.7), qkv3 69.3 (72.2). Flat-to-worse ⇒ backend already software-pipelines; more register pressure. Reverted via git checkout; suites green (engine 124, kernels 21).

### Conclusion after P6-T1..T3
Geometry/nsplit/spec-split/LUT/unroll are exhausted within the current shared EXL3 trellis layout. k=3 linears cap ~55-95 GB/s vs k=6 lm 228. Dominating remaining hypothesis: per-byte decode-ALU/dependency (k=3 ≈ 2× k=6 per byte). Only a format/layout repack (new spec, affects converter + on-disk + memory layout) can cut it — needs user decision. No in-band knob untested.

## trellis-dense-repack RT0 (2026-09-24) — ALU hypothesis REJECTED
Probe `bench_gemv_probe_alu_slow` (new kernel `exl3_gemv_probe`, FC k=3/0/1, modes
0=full funnel+decode / 1=half symbols / 2=pure u32 stream-sum; probe_nsplit override;
encode_gemv_probe in Exl3Gemv). Real down/o/k trellis, release, min-of-20, warm:
- down k3: m0 0.613ms(71GB/s) m1 0.475(91) m2(ushort) 1.119(39) m2(u32) 0.656(66)
- o    k3: m0 0.344(48) m1 0.294(56) m2(u32) 0.344(48)
- k    k3: m0 0.231(36) m1 0.207(40) m2(u32) 0.259(32)
- nsplit sweep of m2 on down/o (swap-noisy, relative OK): m2n8 best ≈ m0 decode.
VERDICT: stream-sum ≈ full decode ⇒ per-symbol decode ALU effectively free; halving
symbols gains only ~10-19%. The k=3 plateau (~50-75 GB/s) vs lm_head 228 GB/s is a
load/loop-structure gap (independent loads in flight / contiguous runs), NOT ALU.
=> dense repack's mechanism = long contiguous per-thread read spans (L-wide; L-planar
three streams), not ALU reduction. plan.md/spec.md amended accordingly.
Notes/ops: (1) probe pipeline is FC_ALIGNED=1; misaligned tensors (lm_head known) fault
the device and hang wait_until_completed for minutes — probes on aligned tensors only;
(2) an orphaned cargo/rustc from a killed timed-out run held target/.cargo-lock and made
subsequent `cargo test` appear to hang with no output — check `pgrep cargo` + remove
stale `.cargo-lock` when a cargo test prints nothing.
