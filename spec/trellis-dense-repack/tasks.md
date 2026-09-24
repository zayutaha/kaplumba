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
- **Files**: `crates/engine/src/repack3.rs` (new), `crates/engine/src/load.rs` (variant
  scan), `crates/engine/src/lib.rs` (pub mod), unit tests in `repack3.rs`.
- **Intent**: For both candidate layouts, implement CPU pack(unpack_ref(blob)) producing
  `dense3` side-car blobs (`<tensor>.trellis.dense3`, magic `D3\0`+layout id) for the
  seven k=3 tensors; unpack side-car back to words and assert full-tensor equality with
  the reference decoder (`decode_ref`) for every weight of every k=3 tensor of the
  artifact (batch over all 60 layers in a test).
- **Testing**: unit tests: (i) round-trip equality full tensor × 2 layers; (ii) all
  byte-offset alignments 0..7; (iii) corner tiles (first/last in/out tiles); (iv)
  invariant: side-car length == expected bytes for layout; (v) `k != 3` tensors and
  lm_head rejected with clear error (no-op for them).
- **Commit**: after tests green.

## RT2 — Dense decode kernels (both layouts) + fast GPU gates
- **Files**: `crates/kernels/src/exl3.metal` (add `exl3_gemv_dense3_planar`,
  `exl3_gemv_dense3_wide` + `_post` twins with same 128-thread/8-out-tile/nsplit
  geometry), `crates/kernels/src/exl3.rs` (pipeline/family selector, pool key gains
  format dimension), `crates/engine/src/metal_linear.rs` (`LinearFormat` on handles,
  `encode_gemv`-family dispatch by format).
- **Testing**: fast GPU gates `gpu_dense3_planar_matches_word` and
  `gpu_dense3_wide_matches_word` on real layer-0 linears (7 shapes, both modes,
  byte-offset 0 and 2, rows 1 and 64): bit-identical outputs vs the current word
  kernels; plus fused2/fused3 dense variants for the winner in RT4.
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