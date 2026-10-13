# spec/dflash-round-30tps — implementation notes

Append as work proceeds. Tone: factual and dense; specifics (paths, numbers,
commit hashes) over narrative.

## Measured premises this spec starts from (all pre-existing, none re-measured here)

Cold-decode baseline, `~/.kaplumba/models/Gxma-repacked`, M4 Pro:
**15.283 tok/s** = 245 GB/s of the 273 GB/s roofline, 90% of it. Anything that
wants to beat plain decode has to beat a memory-bound kernel at 90% of roofline,
which is why four EXL3 M>1 families changing nothing is not a surprise.

Round components, from `spec/dflash-correctness` and `spec/dflash-30tps`:

| component | measured | note |
|---|---|---|
| drafter, one block | **156.71 ms** | 5 layers + `fc` + the tied head over 16 rows |
| verify, one `forward_verify` | **606 ms** | M=16 and M=32 land in the same padded bucket |
| tied head, per-row | **~135 ms** | 6 bpw 262144×5376 = 1.06 GB × ~16 rows |
| plain decode | **15.283 tok/s** | 65.4 ms/token |

### The arithmetic, done out

`tokens_per_round(α, n) = (1 − α^(n+1)) / (1 − α)`.

| round | n | cost | α=1.000 | α=0.95 | α=0.90 | α=0.91 |
|---|---|---|---|---|---|---|
| one block | 15 | 898 ms | 17.8 | 12.5 | 10.0 | 10.6 |
| two blocks | 31 | 1055 ms | 30.3 | 15.3 | 9.0 | 10.0 |
| two blocks, batched head | 31 | ~935 ms | 34.2 | 17.2 | 10.2 | 11.2 |

Break-even against 15.283 tok/s: **α ≈ 0.91 for one block, α ≈ 0.95 for two.**
30 tok/s needs α ≈ 1.00 *and* 31 drafts *and* a batched head. This is the
spec's whole shape: it is not a tuning problem.

## Why this spec does not build a fifth kernel family

Four EXL3 M>1 bodies have now been built and measured here, and the wall clock
did not move for any of them:

| family | M=32, 5376×21504 @3 bpw | single-pass BW |
|---|---|---|
| v1 fp32, 4×m8 passes | 2.621 ms | 16.5 GB/s (66 GB/s of 4× traffic) |
| Path A fp16 scalar, 1 pass | 3.305 ms | 13.1 GB/s |
| Path B MMA, 1 pass | 2.507 ms | 17.3 GB/s |
| `h2` packed-half2 | — | AC2 not met (`spec/dflash-30tps-v2`) |

The reason is measured, not argued: **the EXL3 dequant is the binding
resource.** v1 at M=32 decodes 4.6e8 values in 2.6 ms = **178 Gdecodes/s**;
Path B decodes 1.16e8 in 2.5 ms = **46 Gdecodes/s**. Path B removes the multiply
entirely (16 `simdgroup_multiply_accumulate` issues per thread per chunk vs 256
scalar FMAs) and takes the same wall clock, because the dequant was the
constraint.

A verify at v1's dequant rate, one pass, would be ~170 ms — which is the only
number in the table that would make 30 t/s *arithmetically* comfortable (two
blocks + batched head ≈ 314 + 170 + 15 = 499 ms → 64 tok/s at α=1.000, 32 tok/s
at α=0.95). That body is deliberately not in this spec; if the permutation sweep
and the head oracle raise α and there is appetite for it, it is the next spec.

## Known state of the code the spec touches

* `DflashModel::draft_and_verify` — exists, committed `45aeeb6`, **no non-test
  caller**. Speculative-sampling verification over the *sampler's* distributions
  (not argmax), the anchor never re-fed, `L[0] = previous_logits`, retract
  `n − correct`. Position-convention off-by-one fixed in `10dbd2c`.
* `generate.rs:332-386` — the plain decode loop, one token per
  `forward_buffer_into_vec`. No drafter branch.
* `crates/cli` — hand-rolled parser in `args.rs:63-135`; no drafter flags.
* `TextModel::forward_verify` — already captures the six taps through
  `forward_chunk_pick` (model.rs:4999-5015); the blit is per tapped layer, keyed
  on the layer index, and it selects the ping-pong source rather than assuming
  `hidden_a`. `take_taps(tokens)` reads them back.
* `LinearBufferRequest` (kernels/linear.rs:438) — **no `perm_mode` field**, and
  `gemm_chunk_buffer_with_stats_region` hardcodes `perm_mode: 0`. So
  `DflashModel::set_permutation_mode` is dead on the buffer path: setting it to
  1 leaves the block bit-identical (measured). This is the plumbing bug T5
  fixes; it is why the drafter's tile permutation has never been scored.
* `forward_with_hidden` returns a **zero** final hidden — `max|v| 0.00000`. There
  is currently no oracle for `draft_logits`'s head application.
* `DflashBlock.block_keys` / `block_values` (dflash.rs:695-696) — returned by
  `forward_block` and never consumed by anything. That is what T9 uses.

## Reproduce

```bash
M=~/.kaplumba/models/Gxma-repacked
# the alpha probe (needs the GPU; 25.7 s)
KAPLUMBA_TEST_MODEL=$M KAPLUMBA_FORCE_LOAD=1 scripts/gpuq try -- \
  cargo test -p kaplumba-engine --release --test dflash_alpha_probe -- --ignored --nocapture
```

`Gxma-4-bit-split` cannot be used for the probe: it demands 20.02 GiB against
15.27 GiB available. `KAPLUMBA_FORCE_LOAD=1` on the 3-bit checkpoint is the
documented override.

## 2026-10-13 — D1 LANDED, and `perm_mode = 1` is the drafter's real layout

`crates/engine/tests/buffer_perm_mode.rs` (3 tests, model-free, sabotage-verified:
re-hardcoding the two params sites fails both with the named diagnosis).

`dflash_alpha_probe`, real checkpoints, 52 s, `KAPLUMBA_FORCE_LOAD=1`:

```
=== SUMMARY (mean alpha on the round's diagonal) ===
  as-configured                [1,12,23,35,46,57]  mean 0.0000  best 0.0000
  config+1 (z-lab offset=1)    [2,13,24,36,47,58]  mean 0.0000  best 0.0000
  as-configured, colmajor      [1,12,23,35,46,57]  mean 0.1663  best 0.8479
```

**The drafter's tiles are colmajor.** `quantization_config.json` has no `layout`
key so the loader defaulted to 0, and `gemm_chunk_buffer*` hardcoded `perm_mode:
0`, so the mode-1 read was never available. The signatures of a mis-permuted
decode are all present in the mode-0 case and all gone in mode 1:

| | mode 0 | mode 1 | target's own |
|---|---|---|---|
| drafter logit rms | **12.59** | **4.285** | 6.278 |
| logits pinned within 0.1 of softcap 30 | 68 (6 within 0.05) | **0 of 4 194 304** | — |
| block hidden rms | 3.202 | 3.100 | (final-norm row 1.6) |
| mean diagonal alpha | 0.0000 | **0.1663** | — |
| best alpha | 0.0000 | **0.8479** (row 1 vs t=5) | — |

Logit rms going 12.59 → 4.285 against the target's 6.278 is the decisive one:
mode 0 was exactly 2x the target's scale, which is what a transposed tile read
looks like through a shared head, and the saturation at the softcap (68 logits
pinned, `argmax_row` breaking ties to the lowest index) is what turned that into
`alpha = 0.0000` and byte-identical drafts. **`alpha = 0.0000` was a kernel
plumbing bug, not a drafter-quality result.** Three specifications chased it as
"the drafter does not condition on its context" (T12's logits test refuted that:
99.5% of logits move) and then as "the ranking is wrong" (D13); both were
downstream of the permutation.

### What this does NOT fix

Mean alpha is **0.1663**, and the probe's own decision rule says
`alpha < 0.919 -> slower than plain decode, DO NOT SHIP`. So the round still
does not pay. The permutation was necessary, not sufficient.

### The next lead, and it is a strong one

`best 0.8479 at row 1 vs t=5`. The best (row, t) pair on the whole matrix is
**row 1 against target position `total + 5`**, not `total + 1`. The probe
compares draft row `r` against `targets[r]` (the target's distribution for
`total + r`) on the assumption that the block's row 0 sits at position `total`
and row `r` therefore predicts `total + r`. A best match four positions later
says that assumption is off — and an off-by-four on the diagonal would put a
alpha = 0.85 signal where the probe currently reads 0.17.

That is measurement, not conclusion: 0.8479 could be one row's coincidence. The
next step is the full `(row, t)` acceptance matrix, which the probe already
computes and prints only the best of. If the ridge is genuinely at t = r + 4,
the block's position arithmetic (or the probe's comparison) is wrong and the
real per-depth alpha is much closer to 0.85 than to 0.17.

## 2026-10-13 — restore base weight residency for DFlash

The requested reset to `d660a2e` left target `weights.rs` on its original mmap
and `mlock` path, but DFlash had a separate loader in `dflash.rs` that called
`File::read_exact` for each tensor and copied every packed `.trellis` into a new
Metal buffer. That bypassed the intended old loading behavior for the drafter.

`DflashWeights::load` now builds a single-shard `SafeTensorsIndex` and calls
`TextWeights::load_dedicated`. Small MCG/scales/norm tensors are read from the
base loader arena. Packed tensors use `zero_copy_tensor_buffer` and keep the
`TextWeights` owner alive in `DflashWeights`; `pinned_bytes()` reports the
loader's actual mlock accounting. Explicit `KAPLUMBA_RESIDENT_WEIGHTS` uses the
base arena path. Packed offsets pass through the same `align_packed_view` used
by target model loading; aligned ranges stay zero-copy, while misaligned ranges
take its alignment-copy fallback. `model.rs` exposes that helper crate-wide.
Target loading itself was not modified.

Validation on 2026-10-13:

- `cargo check -p kaplumba-engine` — passed.
- `scripts/gpuq try -- cargo test -p kaplumba-engine --test dflash_load dflash_loader_uses_the_base_mmap_and_mlock_path -- --exact --nocapture` — passed on default mmap path.
- `scripts/gpuq try -- env KAPLUMBA_RESIDENT_WEIGHTS=1 cargo test -p kaplumba-engine --test dflash_load dflash_loader_uses_the_base_mmap_and_mlock_path -- --exact --nocapture` — passed on explicit resident path.
- `scripts/gpuq try -- cargo test -p kaplumba-engine --test dflash_load dflash_loader_realigns_misaligned_shard_views -- --exact --nocapture` — passed.
- Coverage sabotage: temporarily bypassed `align_packed_view`; the unaligned test failed with offset `409` instead of `0`; restored helper and reran successfully.
- The first `target/release/kaplumba` was stale (timestamp 21:28, before the reset and T20). Rebuilt and ran the current release with `KAPLUMBA_LOAD_PROFILE=1 /usr/bin/time -l cargo run --release -p kaplumba-cli -- --model ~/.kaplumba/models/Gxma-repacked --prompt 'Say hello.' --max-new-tokens 1 --temperature 0 --metrics`.
- Full 14.88 GB checkpoint load completed in 7,123 ms. Loader reported `pinned 411 trellis tensors, 11.22 GiB wired`; one-token generation completed. `/usr/bin/time -l` reported 0 swaps, 0 block input operations, and 12,432,015,360-byte max resident set; peak memory footprint was 13,598,074,960 bytes. Global `vm.swapusage` used bytes fell from 2,746.06 MiB before the run to 2,618.06 MiB after, so there was no system swap growth during this run. This validates residency for a short load/generation check, not a 30 tok/s throughput claim or a long decode.
