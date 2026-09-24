# Spec: trellis-dense-repack (k=3 density repack for decode bandwidth)

## Goal
Raise single-token decode throughput from the measured ~8.55 t/s toward the ≥19.5 t/s
(flip) / 20 t/s gate by removing the per-byte decode cost that caps every k=3 layer
linear at ~55–95 GB/s (vs lm_head k=6 at ~228 GB/s). Achieve this strictly by
**repacking the in-memory/on-disk layout of the k=3 EXL3 trellis symbol stream** —
same symbols, same per-weight values, bit-exact decode output — so that a per-warp
decode pass reads the same bytes with strictly fewer per-symbol ALU/dependency ops.

The decoder stays verify-free because no numerics change: a dense repack only permutes
where each weight's 3-bit symbol lives in memory; the value each symbol decodes to is
bit-identical by construction, and correctness is proven by bit-exact GPU A/B gates.

## Non-goals / restrictions (hard)
1. **No bitstream value change.** Only memory placement of symbols changes. bpw,
   k, scale/exponent grids, and quality are bit-identical to today's artifacts.
   Any A/B showing a single differing output bit = failed experiment.
2. **No Python** in converter or engine. The repacker is pure Rust, run offline
   against artifacts in `~/.kaplumba/models` (PonyExl3 Python stays a build-time-only
   source of the original checkpoint, DE19).
3. **Only the seven k=3 linears** (gate, up, down, o, q, k, v) across all 60 layers.
   lm_head (k=6) and any odd-k tensor remain in the current word layout untouched.
4. No KV-cache quantization (global restriction). No speculative/draft/MTP decode
   (spec/pure-decode-273gb restriction). No quality headroom trade.
5. **Rollback is always one config flip away.** Repacked layout lives in a side-car
   variant (`dense3`) of each tensor blob; the default artifact is never overwritten.
6. No changes to reference checkouts; `reference/` read-only (AGENTS §7).

## Requirements (numbered, testable)
1. **R1 — Lossless repack (CPU, Rust).** `repack3` — a Rust subroutine — reads a k=3
   trellis blob and emits a `dense3` side-car blob such that decode(reuse(k)) equals
   decode(dense3(k)) bit-for-bit for every weight in the tensor (verified per-contact
   CPU round-trip against the packed reference decoder for a full tensor).
2. **R2 — Two candidate layouts.** The spec A/Bs exactly two dense layouts and keeps
   the faster one, measured on real weights:
   - **L-planar**: sign and mantissa bit-planes as separate byte-aligned streams
     (vector-friendly loads, more independent load streams per tensor).
   - **L-wide**: each thread owns one large contiguous aligned span of the k=3 stream
     and decodes it with wide (32-bit) masks/half2 tricks, minimizing funnel ops.
3. **R3 — GPU decode gate.** For the kept layout, a fast unit test proves the on-GPU
   dense decode is bit-identical to the current word-layout decode for all seven
   k=3 linears of layer 0 (and spot-checks layer 29/59), across both `mode` values and
   trellis byte-offsets (aligned and misaligned), single-row (decode) and multi-row
   (prefill) paths.
4. **R4 — Per-linear microbench ≥ 180 GB/s (real bytes).** On the real artifact,
   `bench_linear_bandwidth_slow` (extended to the dense kernels) must report ≥180 GB/s
   for gate, down, and o individually (current: 81.8 / 81.7 / 53.9), and ≥180 GB/s
   composite for the fused gu2/qkv3 dispatches after fusing dense variants. The dense
   path must not regress k/v/o shapes (≥ their current absolute GB/s).
5. **R5 — Full-pipeline throughput gate.** `decode_throughput_slow` (gated,
   `KAPLUMBA_RUN_SLOW=1`) and `real_artifact_cached_fixture_and_gpu_slow` (fixture
   argmax **236773**) pass after the switch; e2e `trace_chat_prompt_steps_slow`-style
   bit-for-bit token trace still matches to ≥20 steps.
6. **R6 — Prefill co-verification.** After the switch, prefill throughput suite
   (512/2048/8192 prompt lenses) shows no regression vs current 87.3/81.3/71.3 t/s
   and preferably a lift (dense decode helps prefill per-symbol ALU too). Prefill draws
   from the same segment/per-token geometry; no quality delta.
7. **R7 — Gate flip.** With all of R1–R6 met, flip the `decode_throughput_slow` sink
   assert to ≥19.5 t/s (target 20.0), re-run the gated e2e, and record the final
   numbers + RSS/swap delta in `implementation_notes.md`.

## Acceptance criteria
- Every k=3 linear's outputs under the dense layout are bit-for-bit equal to the word
  layout on GPU (R3 gate), and the whole-model fixture argmax stays 236773.
- Per-linear 180 GB/s met on gate/down/o + fused variants at kept layout (R4).
- `decode_throughput_slow ≥ 19.5` flipped and green (R7), prefill no-regression (R6).
- `dense3` variant load disabled → identical behavior and default artifact untouched.
- All new code passes `cargo test -p kaplumba-kernels` and `-p kaplumba-engine` fast
  suites; gated suites run with `KAPLUMBA_RUN_SLOW=1`.

## Open questions
(none — all resolved by the clarification interview on 2026-09-24: A/B both layouts
keep winner; hard lossless; all k=3 linears; variant side-car; Rust packer; T0 probe
included; ≥180 GB/s per-linear bar; prefill regression-checked.)