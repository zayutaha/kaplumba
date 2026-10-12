# spec/dflash-30tps-v2 — implementation notes

Running handover log. Append as work proceeds.

## 2026-10-12 — opened. What `spec/dflash-30tps` measured, and why this spec is
## the opposite design.

Read `plan.md` first. The short version: the previous spec bet on *removing the
FMA bottleneck and the 4× weight re-stream* (fp16 scalar, then MMA). Both were
built, both measured, and both **lost to the shipped fp32 path**:

```
arm                           M        v1 fp32      A fp16/n8          B mma
target 3bpw colmajor          8     0.983 ms     1.976 ms     1.593 ms
target 3bpw colmajor         16     1.621 ms     1.359 ms     1.692 ms
target 3bpw colmajor         32     2.621 ms     3.305 ms     2.507 ms
drafter 4bpw rowmajor        32     1.622 ms     2.572 ms     1.355 ms
```

`spec/dflash-30tps/implementation_notes.md` names the cause: Path A's win was
eaten by register pressure and by 128 activation loads per lane per k-tile at
M_CAP=32, and Path B's `bstage` + per-chunk barrier + scattered threadgroup
stores made its dequant schedule 3.9× worse than v1's register-only path.

What survived from that analysis, and is this spec's starting point:

* v1's *schedule* (K split across 8 simdgroups, register-only dequant, one
  `red[8][16]` reduction) is the fast one. Do not change it.
* The only remaining lever is **instructions retired per MAC**, and the one
  instruction that retires two MACs on this hardware is a packed `half2` FMA.
  Scalar `half` retires one, which is why Path A's "fp16" measured as fp32.
* The pairing is free: the ring's 8 decoded values are the 4 k-positions
  `r0, r0+1, r0+8, r0+9` for column `n` then the same four for `n+8`, and
  `mrhs_load_tile_x` already loads exactly the matching `half2` pairs.

### Measured premises carried in

| fact | value | source |
|---|---|---|
| target | `~/.kaplumba/models/Gxma-repacked`, 16.028 GB, 3 bpw, **colmajor**, 60 layers, inter 21504 | `spec/dflash-30tps/implementation_notes.md` |
| drafter | `~/.kaplumba/models/Drafter`, 769 MB, 4 bpw, rowmajor, **block_size 16** | same |
| non-speculative decode | 15.28-16.05 tok/s (65 ms/token, 245 GB/s ≈ 90% of the 273 GB/s wire) | `perf_probe_simd_alu_roofline.rs` |
| M=32 verify, 60 layers | 606 ms (v1 default route) | `spec/dflash-30tps` |
| prefill bucket | ≤32 rows ≈ 545-600 ms; 36-62 rows ≈ 1325-1352 ms | `dflash-correctness` notes |
| copy ceiling | 250.5 GB/s measured | `spec/decode-19tok-repacked-250gb` |
| M=1 dequant is 82% of ALU | at M=1 only — **not** at M=32 | `perf_probe_simd_alu_roofline.rs` |

### Environment

* GPU lock: `scripts/gpuq status` / `scripts/gpuq try --`. Exit 75 = busy.
* `env_flag` (`crates/kernels/src/linear.rs`) memoises on first read; a test
  cannot flip a kernel-selection flag mid-process. `force_mrhs_path` is the
  atomic hook.
* Box: 24 GiB unified, currently ~23 GB used by browsers/editor before any
  model load. Watch `top -l 1 -n 0 | grep PhysMem` (AGENTS.md §10.3).

## 2026-10-12 — K1/K3 MEASURED (AC1, AC2): the packed-half2 family is correct,
## and its win does not survive the noise

Shipped: `extract_v_mcg_32_h2` + `linear_gemv_mrhs_ishizuki_k3_colmajor_h2_impl`
(`linear_repacked.metal`), `exl3_decode8_mcg_h2` + `linear_gemv_mrhs_exl3_h2_impl`
(`linear.metal`), 10 new kernels, `MrhsPath::H2` + `KAPLUMBA_MRHS_PATH=h2`, the
parity/sabotage suite, and the `C h2` arms in `mrhs_v2_bench`.

`mrhs_v2_bench`, same binary, four runs, median of 9 per cell:

```
arm                           M        v1 fp32      A fp16/n8          B mma       C h2/n16
------------------------------------------------------------------------------------------------
target 3bpw colmajor          8     0.983 ms     1.976 ms     1.593 ms     1.035 ms
target 3bpw colmajor          8     1.351 ms     1.499 ms     1.828 ms     0.941 ms
target 3bpw colmajor          8     1.464 ms     1.088 ms     1.216 ms     1.713 ms
target 3bpw colmajor         16     1.341 ms     1.454 ms     2.354 ms     1.832 ms
target 3bpw colmajor         16     1.366 ms     2.526 ms     1.725 ms     2.270 ms
target 3bpw colmajor         32     2.441 ms     3.312 ms     2.518 ms     3.010 ms   (4x m8 blocks)
target 3bpw colmajor         32     2.426 ms     3.318 ms     2.510 ms     7.479 ms   (single-pass m32)
```

### Verdict — AC2 is NOT met

AC2 asked for ≥ 1.8x v1 at M=32 on the 3 bpw colmajor target. The single-pass
m32 body is **3.06x slower** (7.479 vs 2.426 ms) and the 4x-m8-block route is
**0.81x** (3.010 vs 2.441). At M=8 the packed body measured 1.41x, 1.44x and
**0.86x** on three runs of the same binary — a 1.7x spread, which is the box's
noise floor (`AGENTS.md` §10.2 #5), not a result. **No configuration in this
family is reliably faster than v1**, so `auto` is unchanged (v1 ≤ 8 rows, MMA
above) and nothing in the engine routes to `H2` unless `KAPLUMBA_MRHS_PATH=h2`
says so.

### Why the packed body does not pay: the kernel is already at the issue limit

This is the finding that supersedes `spec/dflash-30tps`'s "it is FMA-bound" and
this spec's own "fewer instructions per MAC will fix it". Fit the measured v1
times against lane-ops:

```
per m8 pass, one 5376x21504 3 bpw projection = 451,584 tiles x 32 lane-tiles
lane-ops per lane-tile = 56 dequant + 64 FMA + 18 loads = 138
                                    1.445e7 x 138 = 1.99e9 lane-ops
measured per pass (2.426 ms / 4)              = 0.607 ms
                                    1.99e9 / 0.607e-3 = 3.29e12 lane-ops/s
```

3.29e12 lane-ops/s is **this GPU's SIMD lane rate** (20 cores x 128 lanes x
1.37 GHz = 3.5e12). v1 is at ~94% of it. That also explains the two results the
previous spec could not:

* a *single pass* over 32 rows is not faster than four 8-row passes, because the
  pass cost is FMAs, not weight traffic — Path B's 2.510 ms against v1's 2.426;
* the dequant is 40% of the instruction stream at M≤8 and 13% at M=32, which is
  why shrinking the FMA count buys so little: at M=32 the floor is
  `0.25 loads + 0.5 packed FMA + 6.75/M dequant` = 0.96 lane-ops/MAC, against
  v1's 2.15. That is a 2.2x *ceiling* for a body that actually reaches issue
  rate — and none of the wide single-pass bodies does.

### The wide single-pass bodies stall, and it is `M_CAP`

h2 at M=32 is 7.479 ms = 4.7e11 lane-ops/s = **13% of issue**, i.e. stalled, not
busy. It degrades superlinearly in `M_CAP` (40% of issue at m8, 27% at m16, 13%
at m32) while the *work* per row is constant, which is the signature of the
compiler losing the accumulator registers: 2 `half2` per row is 64 registers at
`M_CAP=32`, plus 4 float2 per row (128) for the fp32 widen. Removing the widen
entirely (`_h2w_`, never widen) measures the same 7.43 ms, so the widen arrays
are not it — the 64 accumulator registers are. The fp32 m32 bucket fell off the
same cliff in `spec/small-batch-prefill-mrhs`; this is the third independent
confirmation that a 32-row single pass does not fit this register budget.

### Two lines of attack that remain, neither of which this spec lands

1. **MLA / MMA without the `bstage` round trip.** Path B has the right op count
   (`~70` lane-ops per 256 MACs, 0.27/MAC) and still measures 2.510 ms, i.e. it
   is stalled too. Its staging cost is the whole difference. Nothing here
   changes Path B's schedule.
2. **`M_CAP` = 16 with a *paired* accumulator** (one `half2` per row-pair rather
   than two per row) halves the accumulator registers to 32 at M=32, at the cost
   of two strided activations loads and one pack per row-pair. Unmeasured.

### Test coverage that was found by sabotage, not by design

`mrhs_h2_matches_v1_*` runs `in_features` ∈ {256, 1024}, and the second is
load-bearing. `tiles_k = in_features / 16` and the 4-tile unroll advances by
`8 simdgroups x 4 = 32` tiles, so at `in_features = 256` (16 tiles) **the
unrolled body never executes** — every k-tile is served by the 8-tile remainder
loop. A sabotage planted in the unrolled body therefore passed the first version
of the test. With 1024 (64 tiles) the same sabotage fails with
`worst 42.73 on a peak magnitude 77.75`. Recorded because the same blind spot
applies to the existing Path A/B suites, which are all `in_features = 256`.

Sabotage evidence (AGENTS.md §3.4), all reverted after observing the failure:

| sabotage | failing test | message |
|---|---|---|
| colmajor h2: swap `x01`/`x23` in the **unrolled** tk+0 tile | `mrhs_h2_matches_v1_column_major` | `K=1024 diverges from v1 at M=1: worst 42.734375 on a peak magnitude 77.75` |
| colmajor h2: swap `x01`/`x23` in the **remainder** loop | same test | `K=256 ... worst 33.984375 on a peak magnitude 42.40625` |
| row-major h2: `H.xz + H.yw` → `H.xz + H.xz` | `mrhs_h2_matches_v1_row_major` | `3 bpw row-major K=256 diverges from v1 at M=1: worst 36.726563 on a peak magnitude 51.15625` |

The legitimate worst diff is 1-2 fp16 ULP of the peak (0.03125 at peak 44.09 for
K=256; 0.0625 at peak 79.7 for K=1024) — the packed body is numerically as good
as the fp32 one, which is the one thing this work does establish outright.

## 2026-10-12 — the blocker is NOT the kernel: the drafter does not condition on
## its context, so alpha is 0.0000 and AC3/AC4/AC5 are unreachable

`dflash_alpha_probe` run against the real checkpoints (17.5 s, model loaded
cleanly — `vm.swapusage` 4704 MB in / 4704 MB out, no thrash):

```
  [d4] tap checksum 71960269904                  (taps are non-zero)
  [d4] 2 distinct draft tokens of 15
=== tap layers as-configured: [1, 12, 23, 35, 46, 57] ===
  drafts: [84544, 84544, 236774, 236774, ... x13]
  mean alpha on the round's diagonal: 0.0000; best 0.0000 at row 7 vs t=2
=== tap layers config+1 (z-lab offset=1): [2, 13, 24, 36, 47, 58] ===
  drafts: [84544, 84544, 236774, 236774, ... x13]      <-- byte-identical
  mean alpha on the round's diagonal: 0.0000
```

The probe's own assertion fires, and it is the right assertion:

> the drafter returned identical drafts from two different tap layer sets, so its
> context does not affect its predictions. alpha = 0.0000 on every row and every
> target position follows from that.

The drafter's own rows are also near-degenerate — `row 0 q top3:
2584:0.0158 9525:0.0158 13855:0.0158` is a nearly flat distribution, and 15
draft rows collapse to 2 distinct tokens — which is what a drafter whose
residual stream is dominated by an uninformative context would look like.

### Why this ends the 30 t/s question for now, and what it does not say

With alpha = 0 the round emits **one** token per 606 ms verify plus a 157 ms
draft: **~1.3 tok/s**, i.e. 12x *slower* than the 15.28 tok/s plain decode.
No kernel speedup fixes that: the verify cost only matters once the round emits
`sum alpha^j` tokens, and at alpha = 0 that sum is 1. **Every number in
`spec.md` AC3-AC5 is gated on this, not on the MRHS family.**

What the kernel work above therefore is, honestly stated: the m8 packed-half2
body is correct to 1-2 fp16 ULP and *sometimes* 1.4x faster, the wide single-pass
bodies lose their registers, and the verify's 606 ms cannot come down without
the ~2.2x headroom that the lane-op arithmetic allows (0.96 lane-ops/MAC
reachable vs 2.15 achieved). All of that is real. None of it is on the critical
path until the context path works.

### Two suspected causes eliminated by reading, not by guessing

Both were the obvious first candidates, and both are wrong — recorded so the
next session does not spend the same time on them:

1. **NOT the cache length.** `DflashKvCache` tracks `KvLayer.held` explicitly and
   `keys[layer]` is a flat vector of exactly `held * kv_width` f16s appended by
   `push`, so `rows_here = keys / kv_width` in `forward_block` really is the row
   count. The probe's `[d4] cache rows per layer [0, 0, 0, 0, 0]` is
   `first_position` (= `total - held`), not a row count, and is 0 *because* the
   cache is correctly full. There are no phantom zero-capacity rows.
2. **NOT `window: 0` on the full-attention layer.** `dflash_block_attn`
   (`attention.metal:2105-2108`) reads
   `window_ok = (params.window == 0u) || (gap < params.window)`,
   so `0` means "no window". And the sliding layers' ctx rows (positions 0..31)
   against block rows at 32..47 satisfy both `key_positions[k] <= q_pos` and
   `gap < 2048`, i.e. **the context is visible to all five layers**. The mask is
   not what drops it.

### What is left

The tap features themselves, or the fusion/projection of them:

* **The T7 tap parity trap is the leading suspect.** `target_layer_ids` =
  [1, 12, 23, 35, 46, 57] has mixed parity, and the in-loop blit must take odd
  layers from `hidden_a` and even from `hidden_b`. A hardcoded source is right
  for four of six. It would leave a *non-zero* tap checksum (the probe's is
  7.2e10, which is not a useful test — summing raw f16 bit patterns is not a
  checksum, and its magnitude implies ~40 prompt rows of large negative values),
  so nothing in the probe output rules this out.
* **`fuse_context` / `project_context_kv` on real taps**, where T4c's only
  positive evidence (`forward_block` differs with 20 *synthetic* context rows)
  does not transfer. `dflash_forward_block_matches_host` — the one test with an
  independent host reference — has only ever been run with **zero** context
  rows.

### The one test that separates them

Extend `dflash_golden`'s host-reference test to `forward_block` with a
**non-zero** context: build 20 context rows from a known f32 pattern, dequantise
and project them on the host, and compare. That test existing at all is the gap
— the whole context path has never been validated against the reference, which
is exactly why the probe's `alpha = 0` sat unexplained across two specs.
Cheaper first pass: call `forward_block` twice with contexts of different row
counts (32 vs 64) under the same anchor and assert the outputs differ; that
localises the failure to "context ignored" vs "context wrong".

### Build / test commands

```bash
cargo build --release
cargo test -p kaplumba-kernels --release                        # all kernel suites
cargo test -p kaplumba-kernels --release --test linear_mrhs_h2  # the new parity
cargo test -p kaplumba-engine  --release --test mrhs_v2_bench -- --nocapture
cargo test -p kaplumba-engine  --release --test probe_kernsel_census -- --nocapture
```

## 2026-10-12 (later) — TWO REAL BUGS FOUND AND FIXED; alpha is still 0.0000

Session started from the previous section's "the one test that separates them". It
found two bugs on the way, both of which are fixed and committed. Neither restores
acceptance, and the notes below say exactly what is now ruled out and with what
number, because most of the work in this session was eliminating hypotheses.

### BUG 1 (fixed): Gemma's `+1.0` RMSNorm offset was applied to a **Qwen3** drafter

`DflashWeights::load` added 1.0 to every drafter RMSNorm weight, justified by a
comment about "Gemma's RMSNorm is `(1 + weight)`". The *target* is Gemma. The
drafter is not: `config.json` says `model_type: "qwen3"`, and z-lab's drafter uses
`Qwen3RMSNorm` (`dflash/model.py:550-607`), whose forward is `weight * x/rms(x)`
with no offset. exllamav3 -- the format's definition -- says the same thing
explicitly:

* `architecture/gemma3.py:267-310` passes `constant_bias = 1.0` to every RMSNorm;
* `architecture/dflash.py:267,272,287,293,316` and
  `modules/arch_specific/dflash.py:52` pass nothing, taking the `0.0` default
  (`modules/rmsnorm.py:20`, applied at `:86`:
  `x * w if self.constant_bias == 0.0 else x * (w + self.constant_bias)`).

The stored bytes agree. Read straight out of the shard as bf16:

| tensor | mean | min | max |
|---|---|---|---|
| `layers.0.input_layernorm.weight` | +0.765 | +0.023 | +3.44 |
| `layers.0.post_attention_layernorm.weight` | +0.311 | -0.006 | +0.617 |
| `layers.0.self_attn.q_norm.weight` | +1.525 | +0.758 | +2.25 |
| `hidden_norm.weight` | +1.208 | +0.002 | +1.69 |
| `norm.weight` | +3.080 | +1.477 | +7.06 |

Ordinary trained weights, all but a handful positive. Gemma's delta-coded table in
the same models directory looks nothing like that: its
`layers.0.input_layernorm.weight` spans **-5.78 .. +444** with mean +4.88, i.e. it
*is* `w - 1` and the offset is real -- for Gemma.

Cost of the error: `post_attention_layernorm` 0.311 -> 1.311 (**4.2x**),
`input_layernorm` 0.765 -> 1.765 (2.3x), `q_norm`/`k_norm` 1.5 -> 2.5, compounding
through five layers and the MLP. Measured in `dflash_golden` (synthetic inputs,
same test both times): layer-0 MLP peak **152 287 -> 10 422**, and the real
drafter's block output RMS **4.139 -> 3.183** with its fused context RMS
**1.958 -> 1.006** and layer-0 ctx V RMS **20.28 -> 10.73**.

Why no test caught it: `dflash_golden`'s host reference applied the same `+1.0`
(`norm_weights`, "The loader's +1.0, applied here in f32 for the reference"), and
`dflash_load` *asserted* the offset existed. Both now assert verbatim storage, with
the two measured anchors named (0.7539; post-attention mean 0.311) so a regression
is read off the failure message.

### BUG 2 (fixed): the golden host reference turned the one non-causal layer causal

`host_block` passed `config.sliding_window` to `attended_of` **unconditionally**,
including for the last layer, which is `full_attention` and `bidirectional`. z-lab
does not: `self.sliding_window = config.sliding_window if layer_type ==
"sliding_attention" else None` (`dflash/model.py:366`), and the engine passes
`window: if sliding { .. } else { 0 }`. With a window and `bidirectional` true,
`q_pos - k_pos` is negative for the block rows ahead of the query and the u32
subtraction wraps to ~4e9, so `gap < window` was false and every future key was
dropped.

Invisible until the norms were fixed, because with the inflated weights the
drafter's own path dominated. After the fix it appeared as a **7% disagreement at
layer 4's `attn_out`** while layers 0-3 matched to 1e-4 -- and layer 4 is the only
non-sliding layer, which is what localized it. Fixed: the same test's worst diff is
**4.973 (11.5%) -> 0.0232 (0.06%)**, and layer 4's stages now agree
(engine 2456.85 / 735.00 / 2454.35 vs host 2457.65 / 734.53 / 2455.96).

This survived because the with-context test **only printed** its result -- no
assertion, so no failure. It now asserts 3e-2 of the reference maximum and names the
worst row and feature.

### New test: the context is *visible*, not merely agreed upon

`dflash_golden::dflash_context_rows_are_visible_in_the_block_output` -- three
passes over the same anchor and positions: two contexts that differ in value, and no
context at all. Measured today (host RMS 4.1122):

```
different context values: engine 33.6172 (8.1751 of host RMS), host 33.8357
context rows vs none:     engine 33.1875 (8.0706 of host RMS), host 33.1591
```

So `forward_block` + `append_context` are right on the context path: the output
moves by 8x its own RMS when the context moves, in the engine and in the reference
alike. A test that only checks agreement could not say this (two implementations
that both ignore the context agree perfectly). On the real checkpoints the same
measurement is `context reordered: block hidden moves 58.14 (18.26 of its own RMS)`.

`host_block` / `block_case` were extracted so engine and reference run over exactly
the same rows, with the worst difference's row and feature reported.

### Probe instrumentation added (D5-D12) and what each ruled out

All measured on the real checkpoints (`Gxma-repacked` + `Drafter`), `alpha = 0.0000`
throughout. A negative result here is worth more than the positive ones:

| # | measurement | result | conclusion |
|---|---|---|---|
| D5 | real magnitudes | anchor RMS 0.0153, mask 0.0161, fused ctx 1.006, taps 0.84-4.94, ctx V 10.73, block hidden RMS 3.183 max 93.7 | nothing degenerate, nothing inf/NaN |
| D6 | drafter logits vs target's | drafter raw top values **103-113**, capped max 30.0, RMS 12.41; **target's own logits max 27.31, RMS 6.28** | the drafter's pre-softcap logits are ~4x the target's, so `tanh(x/30)*30` flattens its whole top band to 29.94-30.00 |
| D8 | block row structure | row RMS 3.13-3.22; cosine to row 0 = 1.0, 0.739, 0.726, 0.691, 0.679, 0.669 | the rows are *not* degenerate -- they differ |
| D9 | the anchor token | the probe fed `tokens[total-1]` (the prompt's last token, 101) where the block's row 0 sits at position `total`; the correct anchor is the target's own argmax there (3048) | a real off-by-one in the probe, fixed (z-lab takes `output_ids[start]`, `dflash/model.py:249,264`). Correcting it changed nothing |
| D10 | head replay vs oracle | `forward_with_hidden`'s `last_hidden` is **all zeros** (max 0.0) and is pre-final-norm anyway (`encode_lm_head` does the norm internally), so this is not a usable oracle | dead end, recorded so nobody repeats it |
| D11 | head at M=1 vs M=16 | per-row worst diff **0.0000 for all 16 rows** | `draft_logits`' batched GEMM is not the `encode_lm_head_rows` MMA hazard |
| D12 | noise embedding x sqrt(hidden) = 73.32 | mean alpha 0.0000, still 1 distinct token | **immaterial**: the block's own rows go through `input_layernorm`, so only their direction survives and the scale cancels. `_raw_input_embeddings(..., scale)` in z-lab is 1.0 for this checkpoint anyway |
| D7 | scale sweep on the block hidden, 1.0 -> 0.125 | alpha 0.0000 at every factor | a pure scale **cannot** change the argmax or the top-64 support, so this only confirms the support is wrong; it was never able to show anything else and is kept as a negative |
| D-as-configured vs +1 | probe ran both tap conventions | identical drafts | the as-configured taps are the right ones (`hidden_states[layer_id + 1]` = the output of layer `layer_id`; exllamav3's `tap_shift = 1`, `architecture/dflash.py:22-27`), and the +1 case is the wrong one |

### Where it stands

`alpha` is **0.0000** on every row and every target position, and all fifteen draft
rows return the **same** token (236774) with a near-flat top-3 (`236774:0.0321
108705:0.0246 84544:0.0235`) while the target's is sharp (`3048:0.9189`). The
drafter's top-64 never intersects the target's top-64 at *any* of the 16x6 row and
position alignments. Established: it is not the norm convention (fixed), not the
mask (fixed), not the head path, not the anchor, not the noise scale, not the
magnitude, and not a lack of context sensitivity -- the context moves the block
output by 18x its RMS. What is left is that the *content* the drafter conditions on
reaches it in a form it cannot use, or that the drafter's weights are not what the
engine decodes; the taps are now the only input that has never been validated
against anything external, and `dflash_golden`'s block reference validates the
graph *given* the fused context, not the fused context itself.

### T18 dead end: the taps cannot be validated without a single-layer GPU entry point

The plan for T18 was to compare a captured tap against the layer's own output,
computed one layer at a time: `forward_layer(0, embedding)` then
`forward_layer(1, ..)`. Written and run, it panics inside the engine:

```
panicked at crates/engine/src/model.rs:5804:55:
called `Result::unwrap()` on an `Err` value: Kernel(InvalidInput(
  "packed buffer length 5671862592 does not match 16515072"))
```

`DecoderLayer::forward` is the **host** path and hands the kernel the whole
offloaded packed region (5 671 862 592 B = the entire second shard) where the
layer's own trellis is 16 515 072 B. It has no `metal_offset`/`packed_len`
slicing, unlike `Model::gemm_chunk_buffer_sliced`, which is why the verify and
drafter paths use the latter. So the only public single-layer entry point cannot
run on a repacked checkpoint, and there is no intermediate-layer oracle in the
API at all: `hidden_states_all` returns the **final** layer's pre-norm state only.

The test was removed rather than committed in a state that always panics. T18
stays open, and the honest statement of it is narrower than before: the taps are
still the only input of the drafter never validated against an independent
computation, and validating them now requires either teaching
`DecoderLayer::forward` to slice an offloaded trellis (a real, separate defect in
a diagnostic path) or exporting one intermediate layer's rows through the GPU
path.

### D13: the drafter does not rank the target's token at all

This is the measurement that ended the magnitude theory for good. For each draft
row, the rank the drafter gives the token the target actually wants (softcapped
logits, so the softcap's order preservation is not in question):

```
row 1: target wants 236858 (p_target 0.3324); drafter ranks it 250604 of 262144 with p 5.8e-26
row 2: target wants 560    (p_target 0.9984); drafter ranks it 198213 of 262144 with p 1.5e-20
row 3: target wants 3016   (p_target 0.9876); drafter ranks it 115807 of 262144 with p 5.7e-15
row 4: target wants 3847   (p_target 0.4052); drafter ranks it  46891 of 262144 with p 5.8e-10
```

`alpha = 0` has two possible causes and they need opposite fixes: a *flattened*
distribution whose ranking is right (a temperature or softcap problem -- fixable)
or a *wrong* ranking (nothing about the magnitude helps). Row 2 settles it: the
target is 99.84% certain of token 560 and the drafter puts it 198 213rd, below
three quarters of the vocabulary. The softcap, the hidden scale, the head path and
the logit magnitudes are all irrelevant to that; the drafter's hidden state simply
does not point where the target's head needs it to.

Also eliminated this session, by reading rather than running:
`permutation_mode` is `0` on both sides (`QuantizedLinear::from_metal_buffer` sets
it, `model.rs:404`, and the drafter's `quantization_config.json` has no `layout`
key, which is what leaves it at 0), so it matches exllamav3's row-major native
layout and the host's `reconstruct_matrix`; and the `suh`/`svh` vectors really are
*scales*, not signs -- `fc.suh` spans 0.0167..1.09 across its 32 256 entries, so a
loader that treated them as +-1 would be off by 65x -- and both the GPU path and
the host path multiply by the f16 values, which is what
`dflash_exl3_decode_matches_host_on_every_linear` verifies at 3e-4 relative on all
36 of the drafter's linears.

## 2026-10-12 (later still) — the round had two position conventions; fixed (T13 correction)

`draft_and_verify` was marked done (T13) but passed its `position` argument to
**two** callees that read it differently:

* `DflashWeights::draft_block` / `forward_block` take **the anchor's own
  position** — the block is `[anchor, mask x 15]`, so block row 0 (the anchor's
  embedding) sits at the anchor's position and the mask rows at +1..+15;
* the round's own `verify_positions` fed `draft.tokens[k]` at
  `position + k`.

`draft.tokens[k]` is block **row `k + 1`** (`select_draft_tokens` takes rows
`1..=draft_rows`), and a block row's hidden state is the model's prediction for
the token **at that row's own position** — that is the convention z-lab's mask
block is trained and verified with (`dflash/model.py:249` slices
`[:, 1 - verify_size:, :]`; `:268` writes those argmaxes into
`block_output_ids[:, 1:]`, whose slots are `anchor_position + 1 ..`). So the
drafts belong at `anchor_position + 1 + k`, not `anchor_position + k`.

The code therefore placed the block at the *first draft's* position while the
verify assumed the drafts started there: **every draft was verified against the
target's row for the position before the one it proposes.**

Failure mode is the nasty kind — silent and total:

* every draft is rejected, so `alpha` is exactly `0.0000` on every round;
* the emitted text stays **correct** (the round still emits the target's own
  token), so no equivalence test, no golden test and no crash reports anything.

Fix: one convention, named. `draft_and_verify` now takes `anchor_position` (the
same number `draft_block` is handed, and the same number a caller passes to
`append_context`, whose chunk ends at `position - 1`), and the `+ 1` between the
anchor's slot and the first draft's lives in a new named function
`draft_verify_positions(anchor_position, n)` (`dflash.rs`, next to
`select_draft_tokens`). Nothing decrements a position, so an anchor at position 0
(empty context) cannot underflow.

**Sabotage (AGENTS.md §3.4), recorded:** dropping the `+ 1` from
`draft_verify_positions` makes
`dflash::tests::drafts_are_fed_one_position_after_the_anchor` fail with
`the first draft is block row 1, so it belongs one past the anchor at 8, not at
the anchor's own slot 7 (which the target's cache already holds: re-feeding it
appends a second key row beneath the anchor)`. Reverted; the suite is 9/9.

`cargo test -p kaplumba-engine --lib -- dflash` → 9 passed, 0 failed.

### What this does NOT fix

`dflash_alpha_probe`'s `alpha = 0.0000` was measured through the *probe*, which
calls `draft_block(&anchor, total, ...)` directly with `total` = the anchor's
position — i.e. already the correct convention — so this fix does not explain it
and does not repair it. The probe's own byte-identical-drafts result
(`[84544, 84544, 236774, 236774, ...]` for both `[1,12,23,35,46,57]` and
`[2,13,24,36,47,58]`) is still the open question, and T12 — "compare at the
`draft_logits` level, assert the drafter conditions on its context" — is still the
task that settles it. Two candidate explanations survive, and T12 discriminates
between them without needing a new oracle:

1. **Saturation.** The drafter's raw logits peak at 113 against the target's 27.3
   and `final_logit_softcapping` is 30, so entries above the cap collapse to
   exactly `30.0` in f16; `argmax_row` breaks ties to the lowest index, which is
   context-independent — that alone would give byte-identical drafts and the flat
   `1/64`-per-token distributions the probe printed. D7's scale sweep argues
   against it (logits at 0.125 of the hidden, max ~14, are not saturated, and the
   drafts still did not move), but D7 reports "still 1 distinct token", i.e. the
   sweep may not have touched the path that produces the drafts.
2. **The probe's context is not the context the block sees.** `forward_block`'s
   own visibility test (`dflash_context_rows_are_visible_in_the_block_output`)
   shows the context moves the block output by 8x its own RMS in engine *and*
   host, which contradicts "the context does not reach the block" as a blanket
   statement.

Comparison at the logits level separates them: saturation means the *logits*
differ while the *argmaxes* agree; a dead context means the logits are
byte-identical.

## 2026-10-12 (T12) — MEASURED: the context DOES reach the block. The "dead context" conclusion is refuted

`dflash_alpha_probe`, real checkpoints (`Gxma-repacked` + `Drafter`), 25.7 s,
`KAPLUMBA_FORCE_LOAD=1` (admission control refuses 16.28 GiB against 15.27 GiB
available; the load is fine with swap, as documented):

```
[ t12 ] two different tap-layer sets, compared at the LOGITS level:
  bit-pattern sums: 145459805532 vs 145508050646 (equal: false)
  4172920 of 4194304 logits differ; worst |delta| 16.28125
  drafts identical: true
  [d5] block hidden (norm out)    rms 3.202003
  [d5] draft logits (softcapped)  max|v| 30.00000 rms 12.594
  [d6] top4 capped->raw [29.9844->123.79, 29.9844->123.79, 29.9844->123.79, 29.9531->107.31]
  [d6] 500 of 4194304 raw logits exceed 60
  [d5] 68 of 4194304 within 0.1 of the softcap 30, 6 within 0.05 of the row max
  [d13] row 2: target wants 560 (p_target 0.9984); drafter ranks it 198213 of 262144
  [d4] row 1 q top3: 236774:0.0299 84544:0.0268 108003:0.0256
  [d11] M=1 vs M=16 head: per-row worst diff 0.0000 in all 16 rows
```

**99.5% of the block's logits (4 172 920 of 4 194 304) change when the tap layers
change, by up to 16.3 logits on a scale where the softcap is 30.** The context is
not ignored, not dropped by the mask, and not lost in `append_context`: the
drafter's output is massively context-dependent. `alpha = 0.0000` is therefore
**not** "the drafter does not condition on its context" (the previous section's
conclusion, and the probe's own `assert_ne!` on the drafts), and AC3/AC4/AC5 are
not gated on a broken context path.

The probe's assertion was measuring the wrong quantity. It compared *drafts* --
argmaxes -- and argmaxes collapse when the logits pile up at the cap:
`argmax_row` breaks ties to the lowest index, which is context-independent, so a
context change that moves 99.5% of the logits can leave all 15 argmaxes
identical. The assertion is now made on the logits (it passes) and the draft
comparison is printed as the saturation signature instead.

### What the numbers now say the illness is

* The distribution is **near-uniform at the top**: row 1 is
  `236774:0.0299 84544:0.0268 108003:0.0256` -- top-3 within 25% of each other --
  while the *target* for the same slot is `3048:0.9189`.
* The drafter's logits are **12.59 rms against the target's 6.278** (exactly 2x),
  and its block hidden is `3.202` rms against the target's final-norm row of
  `1.6` -- also 2x. The two are consistent through one shared head (6.278/1.6 =
  3.9 = 12.59/3.202), so the drafter's *scale* is the trained checkpoint's, not a
  decode error. D7's hidden-scale sweep (0.125 -> logit max 14.3, well inside the
  cap) still gives `alpha 0.0000`, so the scale is not the illness either.
* `[d13]`: the target's token, which it is 0.998 confident about, sits at rank
  198 213 of 262 144 in the drafter's row. **The ranking is wrong, and only the
  ranking** -- magnitude, support and context sensitivity are all present.

So the remaining suspects, in the order the evidence supports them:

1. **The drafter's weight decode.** `alpha` is 2x-scaled but wrong-ranked and
   logits move a lot with context. Every engine-vs-host test shares the reader,
   so a transposed/differently-ordered non-square tile read is invisible to all
   of them, and the drafter is the *only* model that has never been scored with a
   predictability probe. Its `quantization_config.json` has no `layout` key, so
   it takes the row-major default -- and `Gxma-repacked` (`"layout": "colmajor"`,
   which `model.rs:4025` turns into `permutation_mode = 1` for every layer
   linear) shows the project has already had to *score* a permutation for the
   target. **T12's mode sweep could not test this**: `LinearBufferRequest`
   (`kernels/linear.rs:438`) has no `perm_mode` field and
   `gemm_chunk_buffer_with_stats_region` hardcodes `perm_mode: 0`, so setting the
   field to `1` leaves the block bit-identical (measured: same drafts, same 68/16
   pin counts). The buffer path has to learn `perm_mode` before the drafter's
   permutation can be scored at all.
2. **The drafter's `norm`/`hidden_norm` weights and the head application.**
   `DflashModel::draft_logits` applies the *target's* head through the *drafter's*
   GEMM entry point, and `[d10]`'s head replay is unusable as an oracle because
   `forward_with_hidden` returns a **zero** final hidden (`max|v| 0.00000`), the
   dead end already recorded above. D11 only proves the drafter's head is
   self-consistent (M=1 == M=16, 0.0000 in all 16 rows), which a systematic error
   in that entry point would also satisfy.

### Reproduce

```bash
M=~/.kaplumba/models/Gxma-repacked
KAPLUMBA_TEST_MODEL=$M KAPLUMBA_FORCE_LOAD=1 scripts/gpuq try -- \
  cargo test -p kaplumba-engine --release --test dflash_alpha_probe -- --ignored --nocapture
```

`Gxma-4-bit-split` cannot be used for this probe: it demands 20.02 GiB against
15.27 GiB available, and the drafter plus the 4-bit target is what pushes it over.
