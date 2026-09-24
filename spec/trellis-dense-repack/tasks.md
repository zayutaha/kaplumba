# Tasks: trellis-dense-repack

Order matters; commit after every task (AGENTS §4). Gated benchmarks run with
`KAPLUMBA_RUN_SLOW=1`, release profile, `--ignored --nocapture`; record
`vm.swapusage` alongside each.

## RT0 — T0 probe: is k=3 per-symbol ALU the limiter? (go/no-go) — DONE
- **Probe kernel** `exl3_gemv_probe` (FC k=3/0/1, `probe_mode` 0=full funnel+decode,
  1=half symbols, 2=pure u32 stream-sum; `probe_nsplit` override) + `encode_gemv_probe`
  + gated bench `bench_gemv_probe_alu_slow` on real down/o/k trellis bytes.
- **Verdict — hypothesis REJECTED (memory/load-structure-bound)**: m0 ≈ m2 on the same
  bytes (down 0.613 vs 0.656 ms; o 0.344 vs 0.344; k 0.231 vs 0.259); m1 (half the
  decode ALU) only ~10-19% faster. Per-symbol decode ALU is effectively free; a
  symbol-free decoder is NOT the path to 180 GB/s for these shapes.
- **Revised mechanism (plan.md updated)**: lm's 228 vs k=3's ~50-75 comes from
  load/loop structure (wider independent loads in flight, whole-in-range per TG);
  the dense repack targets **long contiguous per-thread read runs** (L-wide; also
  L-planar's three contiguous streams), not ALU reduction.
- **Caveat for future gated runs**: swap pressure inflates absolute ms; Mm
  relative (same-run) comparisons; keep probe records alongside
  `bench_linear_bandwidth_slow`. Misaligned tensors (e.g. lm_head) fault the
  FC_ALIGNED=1 probe → probes only on aligned tensors.
- **Go decision**: proceed to RT1/RT2 with both layouts, keeping the ≥180 GB/s bar
  and bit-exact gates. Exit recorded here + `implementation_notes.md`.

## RT1 — Lossless `repack3` (Rust, offline) with reference round-trip
- **Status**: DONE — commits `e574e19` (packer+tests), `b6f9fef` (notes); scope refined vs plan (see below).
- **Files**: `crates/engine/src/repack3.rs` (new), `crates/engine/src/lib.rs` (pub mod), unit tests in `repack3.rs`.
- **Intent**: For both candidate layouts, implement CPU pack(unpack_ref(blob)) producing
  `dense3` side-car blobs (`<tensor>.trellis.dense3`, magic `D3\0`+layout id) for the
  seven k=3 tensors; unpack side-car back to words and assert full-tensor equality with
  the reference decoder (`decode_ref`) for every weight of every k=3 tensor of the
  artifact (batch over all 60 layers in a test).
- **Delivered (RT1 commit)**: `DenseLayout::Wide4` slab layout
  `[otblock(4)][it][ot_local]` — each 4-out-tile × in-range slab is one contiguous span,
  i.e. the long sequential read runs RT0 identified as the target mechanism. Pure blob
  transforms `pack_wide4` / `unpack_wide4` (byte-identical permutation; generic over k
  — `tile_u16` derived from geometry). Verified: 4 synthetic shapes + real splice of
  layers 0/29/59 (`v_proj`, `down_proj`, `gate_proj`, `o_proj`) full-tensor round-trip,
  corner-tile sanity, BadLength / OutTilesNotDivisible errors. SAFETENSORS trellis shape
  is `[in_tiles, out_tiles, 48]` — k=3, no `.weight` tensor needed to derive geometry.
- **Scope changes vs plan (rationale)**:
  1. Byte-offset alignments 0..7 are not a packer property (pure byte permutation);
     meaningful only at decode time — covered by RT2 gates' "byte-offset 0 and 2".
  2. `dense3` magic/side-car files + loader variant scan + `k≠3` rejection are runtime
     plumbing for the *winning* layout only — deferred to RT5 (unwired until RT4 A/B
     picks the layout; avoids building file IO for a layout we might discard).
- **Testing (this task)**: unit tests in `repack3.rs` as described; all pass.
- **Commit**: after tests green.

## RT2 — Dense decode kernels (both layouts) + fast GPU gates
- **Note**: L-planar superseded by RT0 (decode ≈ stream ⇒ bit-plane ALU savings are nil).
  The A/B is the two contiguous-span layouts **W1** (`[otblock][ot][it]`, per-line runs)
  and **W4** (`[otblock][it][ot 0..4]`, 4-line interleave; packer ships in RT1).
- **Status**: in progress — **parts a+b done** (W1 packer, wide1/wide4 kernels); part c
  (pipeline/encode, exl3.rs) next; part d (gates) after. See `implementation_notes` for
  build/measure details at each step.
- **Deviation (decided during part b)**: no `_post` dense twins. Every k=3 dense
  linear has nsplit≥2 via `gemv_splits` (gate/up n2, down/o n8, q n2, k/v n8), so the
  spec-family (partials) kernels cover ALL production dense paths; nsplit==1 dense uses
  the dense spec kernel + `encode_gemv_reduce_post` (same products, same f32 split
  summation order → bit-identical). Also **rows>1 prefill gates deferred to RT6** —
  dense encode is rows==1 only this iteration (tests below say "rows 1 and 64" → rows 1
  only).
- **Part b (in progress, kernels appended)**: `exl3_gemv_dense1` (W1) and
  `exl3_gemv_dense4` (W4) appended to `exl3.metal` — spec-geometry (one warp per
  out-tile / per 4-out-tile block, grid.y = nsplit chunk, partials in the
  `[(split*out_tiles+ot)*TILE+..]` layout `exl3_gemv_reduce[_post]` reads). W1 tile init
  `dense + (ot*in_tiles+it0)*twords*2`, advance `tile += twords*2` per `it`; W4
  block=grid.x, base_ot=block*4, init `dense+(block*in_tiles+it0)*4*twords*2`, advance
  `tile += 4*twords*2`, slot j = `tile + j*twords*2`, 4×2 accumulators, 4 partial row
  writes. Per-slot accumulation statement order mirrors `exl3_gemv_spec` exactly
  (bit-exact). k==7 path a no-op. Whole-MSL library compile validated via
  `gpu_gemv_spec_matches_generic` (Metal compile errors are library-wide).
- **Part a (commit `df25c3a`)**: `pack_wide1`/`unpack_wide1` =
  `[otblock(4)][ot_local][it]` per-line contiguous runs (per out-tile line ~32 KB);
  W1 == Wide4 with block size 1, no new packer machinery. 3 tests (synthetic incl
  (2,4,1), rejects_bad_length, real artifacts round-trip + per-tile verbatim).
  Wide1/Wide4 both require ot%4==0; real k=3 geometries all divisible by 4.
- **Files**: `crates/engine/src/repack3.rs` (add `pack_wide1`/`unpack_wide1` + tests),
  `crates/kernels/src/exl3.metal` (add `exl3_gemv_dense1` + `exl3_gemv_dense4`, k=3,
  spec geometry; dense twins of the `_post` kernel deliberately NOT added — see
  deviation), `crates/kernels/src/exl3.rs`
  (pipeline/family selector, pool key gains layout dimension),
  `crates/engine/src/metal_linear.rs` (`LinearFormat` on handles, `encode_gemv`-family
  dispatch by format, gates).
- **Testing**: fast GPU gates `gpu_dense3_wide1_matches_word` and
  `gpu_dense3_wide4_matches_word` on real layer-0 linears (4 shapes, both modes,
  byte-offset 0 and 2, rows 1 — prefill rows deferred to RT6): bit-identical outputs
  vs the current word kernels; plus fused2/fused3 dense variants for the winner in RT4.
- **Commit**: gates green.

## RT3 — Broad bit-exactness gates across layers
- **Files**: `crates/engine/src/metal_linear.rs` (test), `crates/engine/src/load.rs`.
- **Intent**: Extend the RT2 equality to spot layers 29 and 59 and to 512-row prefill
  geometry; assert the whole-model GPU fixture argmax 236773 unchanged under
  dense-on decode via `real_artifact_cached_fixture_and_gpu_slow`.
- **Testing**: gated + fast variants; any mismatch = layout killer (revert candidate).
- **Commit**: after green.

## RT4 — A/B microbench: keep winner (≥180 GB/s bar)
- **Files**: `crates/engine/src/metal_linear.rs` (`bench_linear_bandwidth_slow` gains
  dense columns), `spec/trellis-dense-repack/tasks.md`.
- **Intent**: On the real artifact, warm min-of-20, per-linear real-bytes GB/s for the
  dense layouts vs word baselines (gate 81.8 / down 81.7 / o 53.9 / k 37.9). Immediate
  targets: gate/down/o individually ≥ 180 GB/s; fused gu2/qkv3 dense ≥ 180 GB/s
  composite; k/v ≥ their current absolute. Keep the winner kernel family + `_post` +
  `_fused2/_fused3`; delete the loser from `exl3.metal`/`exl3.rs`.
- **Testing**: R4 numbers recorded; regression check of k/v shapes; record swap.
- **Commit**: result + winner/loser recorded in spec §R2 + notes.

## RT5 — Default decode to dense; full-pipeline gates
- **Files**: `crates/engine/src/gpu_decode.rs` (route handles by format; not enabling by
  default yet), `crates/engine/src/gemma4.rs`, `crates/engine/src/load.rs`.
- **Intent**: Wire `use_dense3` (env `KAPLUMBA_DENSE3=1`, default off). With it on:
  `decode_throughput_slow` (gated) reports new t/s; `trace`-style e2e stays bit-identical
  ≥20 steps; fixture 236773. With it off: byte-for-byte current behavior (rollback proof).
- **Testing**: both modes gated; equal to prior baselines with flag off.
- **Commit**: after green.

## RT6 — Prefill co-verification
- **Files**: `crates/engine/src/gemma4.rs` (`prefill_throughput_slow`), prefill benches.
- **Intent**: Run prefill 512/2048/8192 with dense on; assert no regression vs
  87.3/81.3/71.3 t/s (prefer lift). If only decode benefits, keep per-stage format key
  (decode dense, prefill word) — documented decision.
- **Testing**: gated prefill suite × dense-on/off; record.
- **Commit**: after results.

## RT7 — Gate flip + final proof
- **Files**: `crates/engine/src/gemma4.rs` (flip `decode_throughput_slow` assert ≥19.5),
  `spec/trellis-dense-repack/spec.md`, `implementation_notes.md`.
- **Intent**: Flip gate to ≥19.5 t/s (target 20.0), enable `KAPLUMBA_DENSE3=1` by
  default if R5/R6 clean, final gated e2e + fixture run, RSS/swap delta, update notes
  with final per-linear table and pipeline numbers, close R7 + acceptance criteria.
- **Testing**: full gated suites (engine slow set + kernels); acceptance checklist.
- **Commit**: final.

## Post-spec (not in scope unless user approves a new spec)
- Repack lm_head (k=6) for symmetry.
- Repack odd-k tensors (none known today; k=7 fallback path untouched).