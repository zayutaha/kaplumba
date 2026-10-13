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
* `forward_with_hidden` returns one final-norm row for a one-token decode.
  Passing a prompt-length chunk and reading its last row was invalid because
  only one row is populated. D10 now captures sequential one-token rows and
  checks the full-vocabulary target-head replay; see T0c below.
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

## 2026-10-13 — T20.1 exact slices for mapped DFlash projections

`TextWeights::zero_copy_tensor_buffer` gives each DFlash line a Metal buffer
covering the entire `.trellis` shard, with the tensor's `metal_offset` and
`packed_len` stored separately. DFlash was calling the general fp16/fp32 buffer
GEMM entry points, whose input validation treats the supplied length as the
whole packed input. That path therefore rejected a correct large shard buffer
as the wrong tensor size. `DflashModel::gemm` and `gemm_f32` now pass
`packed_len * sizeof(u16)` to the sliced GEMM APIs together with `metal_offset`.
This preserves zero-copy mmap/mlock ownership and does not modify target model
loading.

Validation on the Drafter checkpoint:

- `scripts/gpuq try -- env KAPLUMBA_DFLASH_DIR="$HOME/.kaplumba/models/Drafter" cargo test --release -p kaplumba-engine --test dflash_golden dflash_exl3_decode_matches_host_on_every_linear -- --exact --nocapture` — passed, all 36 linears, worst absolute error `0.003663`.
- `scripts/gpuq try -- env KAPLUMBA_DFLASH_DIR="$HOME/.kaplumba/models/Drafter" cargo test --release -p kaplumba-engine --test dflash_golden dflash_forward_block_with_context_matches_host -- --exact --nocapture` — passed, full block/context worst absolute difference `0.023227` (`0.0006` relative); includes fp32 `o_proj`/`down_proj` paths.
- Coverage sabotage: temporarily replaced both sliced APIs with whole-buffer GEMM. The named linear test failed at the first projection with `packed buffer length 768973144 does not match 86704128`; restored exact slices and reran successfully.
- `git diff --check` passed. Tests used `scripts/gpuq try`; each model test was run serially. Target loading and `sampler.rs` were not changed.
- This repairs DFlash mapped projection execution. It does not connect DFlash to generation or establish any throughput gain. The contemporaneous plain decode observation remains about `12.6 tok/s`; the 30 tok/s acceptance gate is open.

## 2026-10-13 — T1b.1 sliding-ring append headroom

The existing target cache allocated exactly the 1024-token attention window.
`forward_verify` appends all K/V rows before attention, so a two-row verify at
positions 1023/1024 wrote position 1024 into slot 0 before the earlier query
could attend to key 0. The existing boundary test caught this: `batched row at
position 1024 evicted key position 0 before row 0 attended`.

Sliding GPU caches now allocate `sliding_window + prefill_chunk()` physical
positions (2048 for the current 1024 window/chunk). `key_count` sees the full
capacity as needed, while attention still receives `config.sliding_window` and
applies the unchanged semantic mask. At Gemma's 50 sliding layers this adds
839,065,600 bytes (0.78 GiB), from the spec's 3,524,730,880-byte reserve to
4,363,796,480 bytes.

Validation on `Gxma-repacked`:

- `scripts/gpuq try -- /usr/bin/time -l env KAPLUMBA_TEST_MODEL="$HOME/.kaplumba/models/Gxma-repacked" cargo test --release -p kaplumba-engine --test dflash_verify_head batched_verify_preserves_keys_at_sliding_window_boundary -- --ignored --exact --nocapture` — passed. The M=2 batched verify retained key position 0; both rows matched sequential greedy output; max logit delta `0.136719`.
- Direct process footprint measurement (the cargo wrapper reports only its own `phys_footprint`; execute the already-built test binary directly): `scripts/gpuq try -- /usr/bin/time -l env KAPLUMBA_TEST_MODEL="$HOME/.kaplumba/models/Gxma-repacked" target/release/deps/dflash_verify_head-56a416db91f6d528 batched_verify_preserves_keys_at_sliding_window_boundary --ignored --exact --nocapture` — peak `phys_footprint` 14,591,895,680 bytes (13.59 GiB), zero swaps and zero block input operations.
- Coverage sabotage: temporarily restored capacity to `config.sliding_window`; the same named assertion failed with `batched row at position 1024 evicted key position 0 before row 0 attended`. Restored `window + prefill_chunk` and reran the test successfully.
- The tests use synthetic valid token IDs rather than tokenizer-produced text. They compare exactly the target paths at the cache boundary; a user-facing prompt-level round test remains part of T12.

The old V0a task row reporting `19.247559` was stale after T18. A fresh run on
2026-10-13 across three prompts and M=1/2/4/8/16 passed per-row greedy identity,
with worst logit delta `0.4375`. Taps/KV M=2 comparison also passed (hidden
0.375, K 0.0022583, V 0.048828125). The sabotage coverage and final V0a
evidence are recorded below.

## 2026-10-13 — T0c repair: valid target-head oracle

The old D10 diagnostic passed a 78-token prompt to `forward_with_hidden`, whose
contract is a single-token decode, then indexed row 77 from a buffer with only
one written row. The resulting all-zero input made its 27.3125 logit error
misleading. Updated `dflash_alpha_probe.rs` to prefill all but the final three
prompt tokens and capture those three final-norm rows and target logits using
sequential one-token calls. For each row it compares the entire 262,144-token
vocabulary output against `DflashModel::draft_logits`, asserts RMS > 0.01,
max-logit error ≤ 1.0 and exact argmax equality. The 1.0 bound was declared
before the corrected run (1/30 of the model's softcap).

Validation used `scripts/gpuq try` with `Gxma-repacked` and the `Drafter`:

- Release integration test compiled and `dflash_alpha_probe` passed 1/1 in
  35.19 s. The three oracle rows had RMS 3.4851 / 3.0591 / 6.4171; full-vocab
  maximum logit errors were 0.0078 / 0.0078 / 0.0156; greedy tokens were
  107 / 101 / 3048 in both target and DFlash head paths.
- Zero-row sabotage failed as intended: `D10 row 0: target-head replay max
  logit error 27.1875 exceeds 1.0` (exit 101). Restored and retested.
- ×2 input-scale sabotage failed as intended: `D10 row 0: target-head replay
  max logit error 9.0156 exceeds 1.0` (exit 101). Restored and retested.
- After restoration, the full probe passed. The configured row-major overlap
  remained mean diagonal alpha 0.2604 (best 0.5740 at row 2 vs target t=2);
  tap+1 measured 0.2439; colmajor measured 0.0000. This supersedes the old
  0.1663/mode-1 claim: the target head path is correct, and permutation mode 1
  is not a useful layout for this checkpoint. Low acceptance is still a
  drafter path/quality problem, not proof of low model quality until T8 isolates
  its remaining stages.

## 2026-10-13 — T7: independent target-head reconstruction

Extended D10 with a CPU reference for five 128-column bands of the 6-bpw
target head: vocabulary columns 0–128, 128–256, 65536–65664, 131072–131200,
and 262016–262144, across three nonzero final-norm rows. The helper decodes
only those EXL3 tiles from the mmap-backed Metal shard buffer, applies the
checkpoint sign vectors and 128×128 Hadamard transforms, then computes the
softcapped dot product in f32. It never allocates the full 5.6 GB fp32 matrix.
Host, target `encode_gemv`, and DFlash `gemm_chunk_buffer_sliced` all receive
the same final-norm row and head. The 0.05 maximum-logit bound was set before
the comparison; full target-vs-DFlash output also retains the T0c 1.0 bound
and exact-argmax assertion.

Validation command: `scripts/gpuq try -- env KAPLUMBA_TEST_MODEL="$HOME/.kaplumba/models/Gxma-repacked" KAPLUMBA_DFLASH_DIR="$HOME/.kaplumba/models/Drafter" cargo test --release -p kaplumba-engine --test dflash_alpha_probe -- --ignored --exact --nocapture`.
It passed 1/1 in 35.52 s. Across all three probe layouts (row-major configured,
tap+1, and colmajor), 45 band comparisons passed; worst host-vs-target and
host-vs-DFlash error was 0.0088. Full-vocabulary target-vs-DFlash maximum
errors remained 0.0078 / 0.0078 / 0.0156 for the three rows, with identical
greedy tokens.

Coverage sabotage results, restored afterward:

- Shift the host band by one 128-column tile: failed with `T7 row 0, columns
  0..128: host vs target GEMV error 38.9325 exceeds 0.05` (exit 101).
- Double the host input scale: failed with `T7 row 0, columns 0..128: host vs
  target GEMV error 8.9751 exceeds 0.05` (exit 101).
- A first attempt to use `head.packed` found it empty under the mmap loader.
  The test reads only selected tile words from `metal_buffer` contents at
  `metal_offset`; the zero-copy loader and mlock path are unchanged.

Conclusion: target-head packing, mapped offset, sign/scaling, vocabulary slice
indexing, and target-vs-DFlash head kernels agree. The head is not the source
of the measured 0.2604 mean diagonal overlap. Continue with T8's model-stage,
tap, and row alignment checks; do not re-enable colmajor from the stale alpha
claim.

## 2026-10-13 — T8: full-depth DFlash alignment and overlap

The alpha probe now obtains target-greedy continuations for all 15 masked rows,
asserts 16 target distributions / 16 block-logit rows / 15 draft tokens, and
prints every row's target-top-token rank and same-position top-64 overlap. It
runs three distinct tokenized prompts (78, 30, and 34 tokens) under the
checkpoint tap indices, the `+1` offset candidate, and the diagnostic colmajor
mode. The old fixed `.919/.977` ship thresholds are removed; the output labels
overlap as diagnostic and explicitly says it is not full-round throughput.

The prompt × layout mean diagonal overlaps were:

| Prompt tokens | configured row-major | tap +1 | colmajor candidate |
|---:|---:|---:|---:|
| 78 | 0.0992 | 0.0920 | 0.0000 |
| 30 | 0.2572 | 0.1768 | 0.0000 |
| 34 | 0.1852 | 0.2097 | 0.0001 |

Configured row-major averaged 0.1805 across these prompts. Its mean overlap
across depths 1–4 was 0.5341, while depths 5–15 averaged 0.0520. The signal is
prompt-dependent at early depths, then collapses for most later rows. No
one-position target-row mismatch was observed; config+1 did not consistently
improve overlap, and colmajor remains unsupported by both the source convention
and these measurements. This identifies a short useful prefix but does not
prove a production acceptance rate or diagnose the checkpoint's training data.

Validation:

- `cargo test --release -p kaplumba-engine --test dflash_alpha_probe dflash_mask_rows_align_with_same_position_target_logits -- --exact` — passed.
- Row-shift sabotage (`target_row_for_mask_row(r) = r - 1`) failed the fixture:
  `assertion left == right`, `left: 0`, `right: 1` (exit 101); restored and
  reran successfully.
- `scripts/gpuq try -- env KAPLUMBA_TEST_MODEL="$HOME/.kaplumba/models/Gxma-repacked" KAPLUMBA_DFLASH_DIR="$HOME/.kaplumba/models/Drafter" cargo test --release -p kaplumba-engine --test dflash_alpha_probe alpha_at_temperature_1_top_k_64 -- --ignored --exact --nocapture` — passed 1/1 in 88.12 s. All three prompts completed all 15 draft depths and each of the three layout/tap diagnostics.
- `rustfmt --edition 2021 --check crates/engine/tests/dflash_alpha_probe.rs` and `git diff --check` passed.

The V0b long-context ceiling constrains any shorter prefix: at context 8192,
M=4 has a processed-anchor ideal ceiling of 25.63 tok/s even before draft cost;
M=8 reaches 33.89 tok/s only with 34.45 ms left for all proposal and round
work. Since the first eight rows already lose most acceptance probability after
the first few depths, shorter-prefix selection alone cannot establish 30 tok/s.
No production behavior changed. Next evaluate proposal-distribution tuning or
a two-block design against the full target cost and exact rejection contract.

## 2026-10-13 — V0b ideal-proposal verifier ceiling

Added ignored integration test `crates/engine/tests/dflash_verify_ceiling.rs`.
It loads the target once, creates deterministic valid token prefixes at 32,
1024, 2048, and 8192 tokens, saves 91 greedy target continuations, re-prefills,
then interleaves M=2/4/8/16 verifier calls three times. Each verify row is
asserted against the saved next greedy token; all 144 calls passed. The focused
test was compiled in release mode and run through `scripts/gpuq try`:

```bash
scripts/gpuq try -- env KAPLUMBA_TEST_MODEL="$HOME/.kaplumba/models/Gxma-repacked" \
  cargo test --release -p kaplumba-engine --test dflash_verify_ceiling \
  ideal_target_continuations_bound_verify_throughput -- --ignored --exact --nocapture
```

Run completed 1/1 in 957.49 s. For M=16 the median over 9 timings (3 prefixes ×
3 repeats) was 377.699 / 410.785 / 421.480 / 461.548 ms at contexts 32 / 1024 /
2048 / 8192. At context 8192, M=16 spread was 458.756–462.491 ms. Its perfect-
proposal ceilings are 34.66 tok/s for the unprocessed-anchor contract (16/M
tokens divided by verify time) and 36.83 tok/s for the processed-anchor
contract (17 tokens divided by verify time). At 30 tok/s, those leave only
71.8 ms and 105.1 ms, respectively, for *all* drafting, sampling, context
updates, and other round overhead. The historical one-block DFlash draft cost
of 156.71 ms already exceeds either budget. At 2k, M=16 median is 421.480 ms;
the budgets are 111.9 ms and 145.2 ms, still below that historical draft cost.
At 1k, the budgets are 122.5 ms and 155.9 ms, also below it. At context 32,
they are 155.6 ms and 189.0 ms, so only the processed-anchor ideal can cover
that historical draft cost, before sampling or any other round overhead.
Therefore the ideal ceiling crosses 30, but current measured DFlash end-to-end
does not have enough cost budget. This test does not demonstrate product
throughput or a representative natural-language prompt distribution.

GPU attribution at M=16/context 32 reported 312.627 ms of body kernels and
65.504 ms of target head GEMM rows (plus 0.115 ms final norm), 378.387 ms total
GPU-own time. The profiler's synchronized full wall span was 495.425 ms, versus
377.699 ms unprofiled median; do not use profiled wall time as a speed result.
It identifies both the 60-layer body and 16-row tied head as substantial cost,
with the head about 17% of GPU-own verify time. The short-context breakdown is
not a decomposition of the 8k verify.

The run had two large M=2 timing outliers (273.649 ms at 1k and 600.699 ms at
2k); medians were used and the raw output retained temporarily at
`/tmp/kaplumba-v0b-unprofiled.log`. V0b is complete. Remaining work is to repair
the drafter signal/quality path and implement a measured complete round; target
30 tok/s remains unmet.

### V0a completion — row and position sabotage

- `scripts/gpuq try -- env KAPLUMBA_TEST_MODEL="$HOME/.kaplumba/models/Gxma-repacked" cargo test --release -p kaplumba-engine --test dflash_verify_head batched_verify_head_matches_sequential_single_row -- --ignored --exact --nocapture` — passed on three deterministic token prefixes at M=1/2/4/8/16. Every row kept greedy identity; worst absolute logit delta was `0.4375`.
- `... cargo test ... batched_and_sequential_layer_states_match ...` — passed at M=2; hidden `0.375` at L42, K `0.0022583` at L25, V `0.048828125` at L45.
- `... cargo test ... batched_verify_head_carries_signal_on_every_row ...` — passed; adjacent rows differed on `4,190,782 / 4,194,304` vocab entries.
- Sliding boundary test (T1b.1 above) checks `gpu_next_position == 1025`, retained key position 0, per-row greedy equality and max logit delta `0.136719` after position 1024.
- Row-selection sabotage: changed `head_rows_matrix` to repeat the first row. The row-signal test failed with `adjacent verify rows are nearly identical (0/4194304)`; restored and retested successfully.
- Position sabotage: shifted staged batched query positions by +1. The sequential comparison failed at M=2 row 0: batched greedy `3824`, sequential `236777`; restored and reran the 3-prompt matrix successfully.
- V0a now closes. These results establish greedy target-logit/cache parity for the tested shapes and position boundary. Sampled distribution equivalence and round rollback remain in T1b.2/T12; V0b ideal-proposal timing is recorded above.

## 2026-10-13 — T10a proposal-temperature diagnostic

Extended `dflash_alpha_probe.rs` to compare proposal q at temperatures
0.5/0.75/1.0/1.25/1.5/2.0 while holding target p at temperature 1.0/top-k 64.
The sweep uses the same three real prompts and all 15 draft positions. It is
diagnostic only: no production sampler behavior changed. Mean overlap by
temperature, averaged equally over the three prompts, was 0.2071/0.1971/0.1805/
0.1610/0.1428/0.1125. Per-prompt bests differed: 0.75 for the relativity prompt
(0.1033 vs 0.0992 at 1.0), 0.5 for coastal temperatures (0.3445 vs 0.2572),
and 1.0 for garden planning (0.1852; 0.1851 at 0.75). The gain is neither
large nor stable enough to select a product temperature, and mean late-depth
overlap remains poor. Changing q temperature also requires T1c to retain the
exact proposal distribution used for each draw before any production trial.

Validation: release `proposal_temperature_changes_q_without_changing_target_p`
passed. Sabotage replacing the temperature divisor with `1.0` failed the named
`cool q should be sharper than hot q` assertion (exit 101); restored and reran
successfully. The full forced-load diagnostic passed 1/1 in 92.63 s through
`scripts/gpuq try`; all 3 prompts completed the 6 temperatures × 15 positions.
Command:

```bash
scripts/gpuq try -- env KAPLUMBA_TEST_MODEL="$HOME/.kaplumba/models/Gxma-repacked" \
  KAPLUMBA_DFLASH_DIR="$HOME/.kaplumba/models/Drafter" \
  cargo test --release -p kaplumba-engine --test dflash_alpha_probe \
  alpha_at_temperature_1_top_k_64 -- --ignored --exact --nocapture
```

The model process emitted one `mlock` warning for an 86,720,512-byte range
(`Resource temporarily unavailable`, after 0 bytes in that range tracker). The
test completed; this diagnostic does not establish whether target or drafter
weight pages remained pinned, and it is not a throughput result. Investigate
before using this run to make any residency claim.

`git diff --check` and rustfmt check passed. T10 remains open for actual
acceptance and complete round cost; temperature tuning alone is not the path to
30 tok/s.

## 2026-10-13 — T1c exact sampler policy

Added public `DflashSamplingConfig` in `crates/engine/src/dflash.rs` with CLI
defaults temperature 1.0/top-k 64/top-p 0.95. `draft_and_verify` now takes this
single policy; temperature zero selects the existing greedy path. In sampled
mode, the same validated config creates target p and each proposal q. Candidate
order is descending logit with ascending token-id ties; top-k is applied before
temperature-scaled softmax, then inclusive top-p truncation and renormalization.
The exact q map used for every sampled draft remains the one used by acceptance
and residual correction. Non-finite/missing logits and invalid policy values
return errors. No caller is wired yet, so CLI integration and full-round behavior
remain T2/T12 work.

Validation: `cargo test --release -p kaplumba-engine --lib
dflash::sampler_tests -- --nocapture` — 11 passed. Coverage includes CLI defaults,
greedy temperature, temperature shaping, inclusive nucleus cutoff, deterministic
top-k ties, top-k > vocabulary, malformed logits/config, retained q, invalid
uniforms, and residual sampling. Sabotage changing the top-p stop condition to
`cumulative >= 1.0` failed the support assertion (`left: [0,1,2,3]`,
`right: [0,1]`; exit 101); restored and reran all 11 successfully. A broad
`cargo test --release -p kaplumba-engine sampler_tests` also tried compiling
every integration target and failed on pre-existing `tests/test_residency_set.rs`
using absent `ResidencySet::attach_to_queue`; using `--lib` correctly isolates
the DFlash unit suite and passes.

This step makes the exact sampled mode configurable but does not reduce draft or
verification cost and therefore does not itself advance throughput.

## 2026-10-13 — T1b.2a cache-retract integration probe

Added ignored forced-load test
`retracted_verify_rows_do_not_leak_into_the_following_decode` to
`crates/engine/tests/dflash_verify_head.rs`. At context positions 40, 1023 (the
ring boundary), and 1030 (after wrap), it verifies four rows, retracts for
accepted prefixes 0/2/4, feeds the correction at the first rejected position,
and compares that plus one more decode to a reset-cache sequential reference.
All nine cases passed. Greedy correction IDs matched and the maximum next-row
logit delta was 0.213806 (tolerance 2.0); the largest delta was at the ring
boundary with four accepted rows. This establishes no observed stale future-row
leak for those scenarios. It does not compare every retained K/V/position entry,
exercise repeated rejection, or inject an error between verify and retract; the
parent T1b.2 remains open for those checks.

Before the GPU run, `scripts/gpuq status` reported 16 GiB available and no swap.
The forced-load test ran via `scripts/gpuq try` and passed 1/1 in 154.52 s; no
`mlock` warning or swap appeared in its log. Exact command:

```bash
scripts/gpuq try -- env KAPLUMBA_TEST_MODEL="$HOME/.kaplumba/models/Gxma-repacked" \
  cargo test --release -p kaplumba-engine --test dflash_verify_head \
  retracted_verify_rows_do_not_leak_into_the_following_decode -- --ignored --exact --nocapture
```

The existing `verify_retract` unit fixture was sabotaged to retract one too few
(`drafted - accepted - 1`); it failed at the full-rejection assertion with
14 versus 15 (exit 101). Restored and verified by the clean forced-load run's
corrected cache cursor assertions. GPU queue returned to idle with 17 GiB
available and 0 swap.
