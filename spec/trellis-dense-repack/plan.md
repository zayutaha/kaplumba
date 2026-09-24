# Plan: trellis-dense-repack

## Problem statement (measured)
At HEAD (`5e4e830` + P6 notes), decode is 117.0 ms/token = 8.55 t/s at ≈95.8 GB/s over
11.200 GB/token. Per-layer in-context cost is 1.837 ms ≈ 97.6 GB/s; target 50 ms/token
needs ~224 GB/s → per-layer ≈0.75 ms. Isolated BW (real trellis bytes, warm): gate 81.8,
up 82.0, down 81.7, o 53.9, q 53.4, k 35.9, v 35.2 (all k=3); lm_head k=6 228.2.
`bench_gemv_split_sweep_slow` shows nsplit argmaxes unchanged; spec≈generic (≤5%);
LUT and 4-way-unroll A/B negative (see `spec/pure-decode-273gb/implementation_notes.md`).

The k=3/k=6 ratio ~2.0 in both per-symbol decode ALU and measured GB/s is the correlating
signal: k=3 pays ~2× the funnel+decode ALU per byte of the same trellis word format.

## Working hypothesis (T0 will test; see Task T0)
k=3 linears are limited by per-symbol ALU/dependency (funnel shift+mask per symbol +
decode_f chain), not by raw L2/L1 bandwidth — lm_head shows the hardware streams at
~228 GB/s with the same trellis when symbols per byte are halved. If TF = true, a dense
repack that materially cuts per-symbol ALU must move k=3 toward ~200 GB/s and is the
correct lever. If TF = false (memory-side bound at ~80 for *all* parallel shapes and k),
the repack spec must be amended to focus on read-stream count/geometry instead.

## Current layout to be preserved/read (facts)
- Tile: 16×16 weights (`TILE=16`, `Exl3Shape`, crates/kernels/src/exl3.rs:29).
- k=3 tile = 256 weights × 3 bits = 768 bits = 48 u16 words = 96 bytes
  (`words_per_tile = TILE*k`, exl3.rs:79). k=6 → 96 words = 192 B.
- On-disk: converter output tensors `*.trellis` (blob of packed u16 words), loaded
  verbatim by `load.rs` into one 64-byte-aligned shared device buffer per artifact
  (load.rs:585-632); per-tensor `DeviceSpan = (buffer, byte_offset, nbytes)`.
- Word-layout semantics (MSL): each tile's stream read via `ld_word(tile, w, aligned)`
  (u32 at u16-word index `w`), funnel merges two words into a 64-bit window and extracts
  4 code words at shifts s, s+k, s+2k, s+3k (`exl3_decode_f`/`_const`). The k≠7 path is
  the only path k=3 uses. Geometry/dispatch unchanged by this spec for prefill; decode
  uses `encode_gemv_spec` + `gemv_reduce_post` + fused2/fused3 singles — kernels get new
  dense-format variants (see Dense kernels below).
- Pipeline cache in `MetalLinear`/`Exl3Gemv` keyed by (k, mode, aligned). Dense is a new
  kernel-name family → extend the key with a `format: u8`/kernel-family selector.

## Data flow (decode, kept layout, after repack)
1. Repack (offline, one-time, Rust): for each k=3 tensor, read current trellis blob
   (word layout), decode each weight symbol to its canonical (sign, magnitude, k_scale)
   triple via the existing reference decoder, re-encode into candidate dense layout,
   write side-car `<tensor>.trellis.dense3` + regenerated `n`/scale metadata if the
   layout splits planes (scale/exponent grids are per-word-group today — must travel
   with the repacked planes unchanged).
2. Loader (`load.rs`): if variant `dense3` requested in config/CLI flag, substitute
   dense spans for the seven k=3 names (byte offsets re-aligned to 64B inside a second
   buffer or the same shared buffer; see Feature flag below). Non-k=3 tensors untouched.
3. Decode: `encode_gemv` / `encode_gemv_*_fused*` route to the dense kernel family when
   the handle's `format == dense`. All kernels-compose with existing had_in/had_out/
   reduce_post stages unchanged (dense replaces only the gemv word funnel/decode stage).
4. Output buffers/partials identical; reduce_post identical → any bit diff is the gate.

## Dense kernel designs (Task T2 = A/B)
For each, the per-warp decode work = read `bytes_per_tile` and emit the same 16 subrow
accumulators. Both keep the warm 128-thread/8-out-tile threadgroup geometry of the spec
kernels (that geometry already hits ~228 on lm k=6) and the nsplit policy.

- **L-planar**: three separate byte-aligned plains per tensor: sign plane (1 bit per
  weight → 32 B per tile), magnitude plane (2 bits per weight → 64 B per tile), plus
  the k_scale/exponent word grid unchanged. Decode: one `uchar4`/`half2`-ish load per
  plane; sign-xor, magnitude→ 2^m via small `constant` power table (k-fold, fits), fma.
  ~3 parallel load streams per tile instead of 1, per-symbol ALU ≈ sign/mag only
  (funnel gone).
- **L-wide**: keep the word stream but give each thread a contiguous aligned span of
  full tiles (byte-aligned, e.g. 96 B/tile groups), decode via wide 32-bit loads + a
  single fused mask/extract that yields all symbols of a 32-bit window with ~1 op each
  (precomputed per-position masks in `constant`), average per-symbol extract cost ↓.
  Closer to today's format → packs with zero wasted bits (3/word exactly 10+2 spare
  consolidated by permuting symbols across the 16-subrow grid so spares aggregate).

T0 tells which hypothesis each targets; measure both at R4 (min-of-20, warm), keep winner.
(The funnel transforms and the two plane streams are the concrete per-symbol-op deltas;
exact op counts get instrumented in T0.)

## Feature flag / rollback (R2/R5/R7)
- `LinearFormat` enum on `LinearHandles`: `Word` | `Dense3`. Decode path selects
  kernel family by handle; engine-level switch `use_dense3: bool` (env `KAPLUMBA_DENSE3=1`
  or config, default off until R7 lands).
- Loading dense spans: separate `MTLBuffer` (own region of shared artifact buffer), so
  disabling the variant = old spans, no rebuild, instant rollback.
- Packer reads from `~/.kaplumba/models/<artifact>/`; writes `.dense3` side-cars into the
  same dir. Artifact checksum/version string recorded in side-car manifest for hygiene.

## Requirements → tasks mapping
- R1 → RT1 (reference round-trip pack/unpack; full-tensor CPU equality).
- R2 → RT2 (both kernels + fast GPU gates); winner decided at RT2/R4 data.
- R3 → RT3 (fast bit-exact gates on GPU for 7 linears × modes × aligns × 1/64/512 rows).
- R4 → RT4 (extend `bench_linear_bandwidth_slow` to dense family; run A/B; keep winner).
- R5 → RT5 (switch decode path default to winner; fixture + trace gates green).
- R6 → RT6 (prefill suite re-run + regress check).
- R7 → RT7 (flip gate, gated e2e, notes, final numbers).
- T0 → RT0 (hypothesis probe, go/no-go gate on the working hypothesis; amend plan/spec
  here if it fails before any kernel work).

## Sequencing and dependencies
RT0 must pass before RT2. RT1 can start in parallel only in a second working tree
(AGENTS §6 single-model: serialize; do RT0 → RT1 → RT2 → RT3 → RT4 → RT5 → RT6 → RT7).
Dense fused2/fused3 variants are a strict subset of RT2/RT4 (add only for the winner).

## Risks and fallbacks
- **Hypothesis wrong (T0)**: plan changes to stream-count/geometry of dense layout; the
  packed-symbol storage is reused but the target shifts (documented; spec amended).
- **Dense slower than word (RT4)**: keep word layout; record numbers; spec becomes
  anti-knowledge; decode stays on current lever stack. Rollback is the default path.
- **Bit-exactness break (RT3)**: any mismatch kills the candidate layout immediately;
  the round-trip gate (RT1) plus GPU gates make drift findable at first commit, not last.
- **Disk/RAM for side-cars**: k=3 tensors ≈ 150 MB/layer × 60 → ~9 GB side-car total
  in `~/.kaplumba/models`; builds happen offline once; acceptable on current hardware.
- **Prefill regression (RT6)**: prefill segment kernels also use dense family; if dense
  only helps decode, keep decode-on-dense, prefill-on-word via per-stage format key.
- **Swap pressure**: record `vm.swapusage` for every gated benchmark (prior
  convention); use relative (same-run A/B) deltas for decisions.

## Interfaces / formats
- On-disk side-car: `<tensor>.trellis.dense3` + `<tensor>.scales.dense3` (if plane
  split separates scale grid) + `manifest.dense3.json`-less — plain binary N bytes,
  layout identified by magic `D3\0` + layout id. Loader scans `*.dense3` on open.
- MSL: `exl3_gemv_dense3_planar`, `exl3_gemv_dense3_wide` (+ `_post`, `_fused2`,
  `_fused3` for the winner) with the same parameter block as `exl3_gemv` + plane-offset
  fields. Function constants reused (k=3 fixed, pool per family).
- Rust: `LinearFormat` on handles; `encode*` slice dispatch by format; bench gains a
  dense column.