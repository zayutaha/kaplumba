# spec/dflash-round-30tps — implementation notes

## 2026-10-14 — T9 proposal contract

The two-block B=16 experiment is defined as 30 distinct proposal positions:
first block proposes `p+1..p+15`; the second block anchors on the sampled token
at `p+15` and proposes `p+16..p+30`. Its temporary context retains the
target-feature cache and appends first-block drafter K/V only at `p..p+14`;
the `p+15` row is omitted because the second block recomputes that anchor.
Second-block q distributions are the actual logits from that forward, including
its sampled anchor and temporary context. This is a conditional proposal
process, not a claim that masked-state K/V equals target-feature K/V.

This is a bounded correctness/feasibility probe only; no generation or CLI
integration is authorized by T9. First measure the target's 30-row verify cost,
then run full generation only if exactness and observed-prefix cost leave a
plausible route to 30 tok/s. At this point this is a design contract; no code,
model run, or throughput result has validated it.

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

## Historical 2026-10-13 — initial `perm_mode = 1` claim (superseded by T6)

The overlap interpretation below predates the corrected target-head oracle. Its
claim that mode 1 is the drafter's real layout was invalidated by T7/T8 and the
T5/T6 host-reference validation recorded at the end of this file. Keep these
measurements as historical evidence only; the current checkpoint layout is
mode 0.

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

## 2026-10-14 — T20.2 candidate validation

* `scripts/gpuq try -- env KAPLUMBA_DFLASH_DIR=... cargo test --release -p
  kaplumba-engine --test dflash_load
  dflash_load_has_every_tensor_and_the_right_shapes -- --exact --nocapture`
  passed against the header-normalized 6-bpw candidate. It reported
  `mapped=1151926272 bytes, mlocked=1152516096 bytes`; all 36 packed lines used
  the inferred 6-bpw rate and aligned mmap views.
* The first invocation of `dflash_gpu_encoded_block_matches_slice_block` was
  ignored (the test carries `#[ignore]`). Rerunning with `--ignored` passed.
  Empty/normal/sliding cases had K relative error ≤0.000895, V ≤0.000996,
  hidden ≤0.000494. Normal block medians: encoded 70.41 ms, slice 93.55 ms.
* The independent `HostLinear` in `dflash_golden.rs` had its own fixed-4 rate.
  Candidate test failed before any comparison: `layers.0.self_attn.q_proj:
  packed trellis has 16515072 words, expected 11010048`. It now derives
  `BitRate` from the trellis third dimension and validates rank, tile geometry,
  integer rate and supported range. The all-36-linears GPU-vs-host test passes
  on candidate (worst absolute error 0.002741) and local 4-bpw checkpoint
  (0.003663).
* Deliberately forcing the host oracle back to 4 bpw failed on the candidate at
  the named q-projection packed-word check (16,515,072 vs 11,010,048); restored
  shape inference and reran the candidate test successfully. The candidate-only
  block/linears checks do not load the target model and do not establish stream
  exactness or end-to-end throughput. F1d still requires those checks.
* A mistyped `KAPLUMBA_DFLASH_DIR` once caused the host test's existing
  “checkpoint not present; skipping” path to return success without loading.
  Always verify the path and inspect test output for an actual line result; a
  green test summary alone is insufficient here.

F1d progress: `dflash_greedy_stream_matches_target_for_200_tokens_on_three_prompts`
already checks emitted token IDs against plain greedy decode for all three
prompts and reports per-arm tok/s/round timing. It is `#[ignore]` and requires
both models. Candidate-only encoded block timings were compared in a B-A-B
sequence (6-bpw candidate, current 4-bpw drafter, 6-bpw candidate):
70.41 ms / 69.71 ms / 70.08 ms. The candidate's median was ~0.8% slower for
this isolated block; another 4-bpw repeat measured 69.69 ms. This does not rule
out an end-to-end gain from higher acceptance, but provides no block-speed gain.
After the last candidate-only checks, `scripts/gpuq status` still reported
9.5 GiB available and 2,578 MiB swap; target-plus-drafter generation remains
held below the 14 GiB model-run guard. Once the memory gate clears, run the
ignored exact-stream test with `KAPLUMBA_TEST_MODEL=$HOME/.kaplumba/models/Gxma-repacked`
and `KAPLUMBA_DFLASH_DIR=$HOME/.kaplumba/models/Drafter-eepos-exl3-6bpw`, then
repeat same-prompt B-A-B timing and collect `/usr/bin/time -l` footprint/swap.

F1d harness hardening: `Harness::build` now keeps optional skip behavior only
when `KAPLUMBA_DFLASH_DIR` is unset. With an explicit nonexistent directory,
`dflash_exl3_decode_matches_host_on_every_linear` fails at the named configured
path (`configured DFlash checkpoint has no model.safetensors: ...`, exit 101)
instead of returning a green skip. Rerunning with the real 6-bpw candidate
passed through the harness and compared all 36 linears (worst absolute error
0.002741). This closes the false-green path for explicit drafter selection; it
does not load the target or prove exact stream behavior.

## 2026-10-14 — T20.2/F1d candidate review and loader gap

Found `eepos/gemma-4-31B-it-DFlash-exl3-6.0bpw` while searching for a same-
target DFlash candidate the current EXL3 loader could consume. Its config matches
the local Qwen3 DFlash architecture, `block_size=16`, hidden/intermediate sizes,
attention geometry, sliding/full layer layout, and target taps
`[1,12,23,35,46,57]`. Remote Safetensors header inspection only downloaded the
first 128 KiB; it declares 166 tensors and the same key set as local. All 36
`.trellis/.mcg/.suh/.svh` linears and 22 BF16 `.weight` norms are recognized by
the current suffix rules. Norms, input/output scales and MCG codebook metadata
use the expected dtypes. Every candidate trellis has 96 words per tile (6 bpw),
where local uses 64 words (4 bpw).

The compatibility probe caught a source-level blocker before any model run:
`DflashWeights::load` checks against `BitRate::Integer(4).packed_words()` and
constructs every `QuantizedLinear` with `Exl3Rate::Integer(4)` at
`crates/engine/src/dflash.rs`. Thus the candidate is not loadable correctly even
though its config and tensor names match. T20.2 adds per-line integer-rate
inference from the packed tile shape and 4/6-bpw coverage; fractional MCG layouts
remain rejected. No kernel family is needed because line execution already
accepts a per-line `Exl3Rate`.

Candidate repo metadata: `Apache-2.0`, revision
`48be48d8035ec6c94d6b9b3f867c7dae1e4dc627`, 1,153,033,071-byte
`model.safetensors` (SHA-256 from HF file metadata:
`b86bc095d00b30f73f22f3338981061e2218046a675c51c2871a9b51abeff0cb`). Local
drafter shard is 769,057,588 bytes, so candidate adds 383,975,483 bytes (~366
MiB), not a multi-gigabyte memory increase. HF model card author warns this was
their first EXL3 conversion and only briefly tested; treat all local results as
required. Source: [candidate model card](https://huggingface.co/eepos/gemma-4-31B-it-DFlash-exl3-6.0bpw)
and its `config.json`, `quantization_config.json`, and Safetensors header.

Per the user's existing constraint, weight download started only after config,
tensor-header compatibility, and storage/footprint delta were established. It
is downloading to `~/.kaplumba/models/Drafter-eepos-exl3-6bpw`; neither the local
drafter nor project code has been changed yet. The target-model equivalence test
binary compiled with `cargo test --release -p kaplumba-engine --test
dflash_equivalence --no-run`. Continue after the segmented transfer, verify its
SHA-256, add config files, and run T20.2 before any forced-load generation.

T20.2 implementation update: `DflashWeights::load` now derives integer bpw from
`trellis.shape[2] / 16`, constructs each `QuantizedLinear` with that rate, and
retains the existing exact mapped slice/MCG checks. The loader test checks every
packed line's inferred rate and packed word count instead of assuming `fc` is
always 4 bpw. Unit tests cover integer rates 1–8 and malformed/fractional shapes.
The six `dflash::tests` passed; the rate-specific integration test passed on the
current 4-bpw shard and reported 767,950,848 mapped bytes and 768,540,672 pinned
bytes. A deliberately fixed-rate-4 inference failed the 6-bpw unit assertion
with `left: 4, right: 6`; restoring shape inference made it pass. An initial
test filter used `--exact` without the module-qualified name and ran zero tests;
the corrected filter produced the intended failure and pass. One first draft
of the fractional test incorrectly classified 48 tile words (integer 3 bpw) as
fractional; corrected fractional tile depths to 24/40/56 and reran successfully.

Candidate loader test first failed before rate assertions: `mapped_trellis_bytes`
was 0 while the test expected 1,151,926,272. The source Safetensors header is
16,599 bytes, so the data section begins at byte 16,607; all 36 relative
trellis starts are 0 mod 4, making every absolute start 3 mod 4. The existing
`align_packed_view` therefore takes its copy fallback for every packed line.
The official Safetensors serializer rounds metadata to `next_multiple_of(8)` and
space-pads the header (see `safetensors/safetensors/src/tensor.rs`, `prepare`),
so a single whitespace byte makes this header 16,600 bytes and the data section
start 16,608, aligning every candidate trellis without changing relative tensor
offsets or payload bytes. F1d now requires this isolated header-only normalization
and payload hash comparison before model timing. The unnormalized 6-bpw file is
not yet accepted for throughput tests because it would hide the intended
mmap/no-copy behavior behind 36 copies.

The original file is preserved under
`~/.kaplumba/models/Drafter-eepos-exl3-6bpw-source`; the test copy is under
`~/.kaplumba/models/Drafter-eepos-exl3-6bpw`. Added exactly one trailing JSON
space to the test copy's header and changed the 8-byte length from 16,599 to
16,600. The original whole-file SHA-256 is the HF value above; the normalized
file is 1,153,033,072 bytes. Streaming SHA-256 over tensor payload bytes before
and after matches exactly:
`aeed2f2160e5cf1e412d3300b5e18aa3c5554366c6b0019424c341e08425001a`. Header
recheck reports all 36 packed starts 4-byte aligned. Config files are unchanged.

The first candidate loader run was intentionally not repeated after alignment:
`scripts/gpuq status` reports only 10 GiB available and 2,642 MiB swap, below
the repo's 14 GiB model-run guard. It also showed the previous candidate loader
failure due to unaligned source, not rate inference. `dflash_load --no-run`
compiles with the revised mapped-byte assertion; wait for memory to recover
before testing the aligned candidate or loading the target.

2026-10-14 continuation: reran `cargo test --release -p kaplumba-engine --lib
dflash::tests -- --nocapture` and compiled the `dflash_load` integration target
with `--no-run`; both succeeded. Current `scripts/gpuq status` reports 9.7 GiB
available and 2,610 MiB swap, so the aligned candidate device-load and
target-model tests remain held by the 14 GiB guard. No model process holds the
GPU lock. Do not interpret the successful unit/compile checks as candidate
loader validation.

## 2026-10-14 — T12/T10 reruns and T16b unprocessed-anchor rejection

After T11, the processed-anchor DFlash stream was rerun at trained B=16 for
three prompts × 200 output tokens. All tokens remained exactly greedy. Rates
were 3.77, 3.64 and 4.13 tok/s (mean 3.85), versus plain 12.30–12.33 tok/s;
per-round anchor cost was 115–121 ms, draft+verify 541–542 ms, and context
append 24 ms. This is unchanged from the prior 3.85–4.34 tok/s result within
normal run spread. The verifier-head optimization does not materially improve
the complete DFlash stream. Each run logged an 86,720,512-byte drafter range
that `mlock` could not pin; it remained evictable.

Shorter processed-anchor blocks were also tested on the same three prompts ×
200 tokens using `KAPLUMBA_TEST_DFLASH_BLOCK_SIZE`:

```text
B=4: 5.95, 5.86, 6.01 tok/s; plain 12.30–12.35; accepted 115/85, 116/86, 115/84
B=8: 5.34, 5.17, 5.77 tok/s; plain 12.32–12.34; accepted 127/77, 123/80, 127/72
```

All six streams were greedy-exact. B=4 is faster than B=8/B=16 but remains
slower than plain decode. These were separate runs rather than interleaved
length A/Bs, so T10/T15 remain open and no default changed.

T16b tried the unprocessed-anchor contract after T11 corrected the 6-bpw head.
The pure greedy stream failed at the same first divergence for both B=16 and
B=4: output index 77 emitted 99382, while sequential target emitted 84750. The
run used no T19h known-target proposal override. Unit fixtures for `[anchor,
drafts]` row mapping, bonus row, and cache retention passed, but they cannot
cover the real multi-round cache trajectory. The current unprocessed-anchor
API and integration test were removed. This is enough to reject the path under
the exact-greedy gate; the precise source of accumulated drift remains
unisolated. No throughput from the failed path is reported.

Commands:

```bash
scripts/gpuq try -- env KAPLUMBA_TEST_MODEL="$HOME/.kaplumba/models/Gxma-repacked" \
  KAPLUMBA_DFLASH_DIR="$HOME/.kaplumba/models/Drafter" \
  cargo test --release -p kaplumba-engine --test dflash_equivalence \
  dflash_greedy_stream_matches_target_for_200_tokens_on_three_prompts \
  -- --ignored --exact --nocapture

scripts/gpuq try -- env KAPLUMBA_TEST_MODEL="$HOME/.kaplumba/models/Gxma-repacked" \
  KAPLUMBA_DFLASH_DIR="$HOME/.kaplumba/models/Drafter" \
  KAPLUMBA_TEST_DFLASH_BLOCK_SIZE=4 \
  cargo test --release -p kaplumba-engine --test dflash_equivalence \
  dflash_greedy_stream_matches_target_for_200_tokens_on_three_prompts \
  -- --ignored --exact --nocapture
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

## 2026-10-14 — T12a real greedy stream

Added ignored `crates/engine/tests/dflash_equivalence.rs` to exercise the whole
current round and context handoff. It initializes DFlash context from real target
taps, feeds the target-selected anchor, calls `draft_and_verify`, saves verify
taps before the next target forward, appends exactly the anchor plus accepted
verify taps, and proceeds from the correction token. For each of three real
prompts, 200 emitted greedy tokens matched the target's sequential argmax stream
exactly. The test also times generation after prompt prefill and runs a plain
200-token reference in the same process.

Forced-load run (DFlash and target both resident, no swap) completed 1/1 in
213.40 s. DFlash/plain results:

| prompt | rounds | accepted draft tokens | DFlash tok/s | plain tok/s |
|---|---:|---:|---:|---:|
| relativity | 78 | 127 | 3.94 | 12.30 |
| coastal climate | 80 | 123 | 3.85 | 12.30 |
| garden | 71 | 128 | 4.34 | 12.30 |

Median DFlash is 3.94 tok/s, only 32% of this same-process plain baseline; mean
accepted drafts are 1.65 per round. It is therefore not merely below 30 tok/s:
this DFlash loop is about 3.1× slower than plain target decode on these prompts.
Even if the 16-token verify/draft round accepted every proposal, it has a narrow
cost ceiling; the present checkpoint's acceptance length makes this path
uncompetitive. This evidence points to both round cost and draft quality. It
does not justify an end-to-end speed claim for sampled mode.

The run emitted one `mlock` warning for 86,720,512 bytes while both model sets
were loaded. It completed with no swap, but the warning means those pages were
not guaranteed pinned. The plain and DFlash timings share this same residency
state. Re-run throughput after resolving this pinned-byte gap before treating
these rates as final. Exact command:

```bash
scripts/gpuq try -- env KAPLUMBA_TEST_MODEL="$HOME/.kaplumba/models/Gxma-repacked" \
  KAPLUMBA_DFLASH_DIR="$HOME/.kaplumba/models/Drafter" \
  cargo test --release -p kaplumba-engine --test dflash_equivalence \
  dflash_greedy_stream_matches_target_for_200_tokens_on_three_prompts \
  -- --ignored --exact --nocapture
```

T12a is complete; T12 still needs sampled equivalence and independent branch
sabotage. T13/T14 are not met. Next performance work must first reduce proposal
round cost and address the late-depth horizon collapse; sampler tuning alone
cannot plausibly close the measured gap.

## 2026-10-14 — T16a unprocessed-anchor attempt rejected

Temporarily changed `draft_and_verify` to verify `[anchor, proposals]` in one
chunk, use target rows 0..n for proposal/correction p, retain anchor plus
accepted verify taps, and retract `n - accepted`. This removed the separate
one-token anchor forward from the test loop. The forced-load 200-token T12a
stream failed exact greedy equality on prompt 0 at output index 77: DFlash chose
token 99382, sequential target chose 84750. Failure was reported after 71.69 s;
the assertion aborted before a throughput result was printed. Only the first
prompt ran. The existing processed-anchor T12a stream passed all three prompts.

Restored `dflash.rs` and `dflash_equivalence.rs` to the committed processed-anchor
version. The likely issue is a small batched target-row difference at a near-tie,
but this run did not retain logits at the mismatch and does not prove that cause.
Do not ship or benchmark T16 as a speed result until row-zero parity is isolated
and the exact-output stream passes. T16 remains open; the shortcut's potential
one target decode saving is not enough to justify changing output.

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

## 2026-10-14 — T19a DFlash round timing split

Added test-only timing buckets to the 3-prompt, 200-token greedy equivalence
test: processed anchor target forward plus anchor tap read, draft-and-verify,
verify tap read, and drafter context append. The logged mean per-round times
were stable across all three prompts:

| Prompt | DFlash total | Plain target | Anchor + taps | Draft + verify | Verify tap read | Context append |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 50.747 s (3.94 tok/s) | 16.198 s (12.35 tok/s) | 82.32 ms | 544.52 ms | 0.05 ms | 23.70 ms |
| 1 | 52.035 s (3.84 tok/s) | 16.187 s (12.36 tok/s) | 81.49 ms | 544.43 ms | 0.05 ms | 24.45 ms |
| 2 | 46.174 s (4.33 tok/s) | 16.194 s (12.35 tok/s) | 81.52 ms | 544.46 ms | 0.05 ms | 24.30 ms |

The processed-anchor pipeline remained exactly greedy-identical to sequential
target generation for all 600 output tokens. The largest bucket is
draft-and-verify at ~544 ms per round; context append is ~24 ms, anchor forward
plus tap read ~82 ms, and tap read alone negligible. This confirms that
round-level drafter and target verification work must be redesigned together;
the timing does not identify their individual split yet. Current throughput
remains 3.84–4.33 tok/s, far below the 30 tok/s target. This result is not a
30 tok/s acceptance pass.

Before the run, `scripts/gpuq status` reported 16 GiB available and 0 MiB swap.
The test passed via `scripts/gpuq try` in 213.31 s. As in the prior baseline,
one 86,720,512-byte range failed `mlock`; the test ran under that residency
state, so do not treat the absolute rates as a pinned-memory benchmark. The
plain and DFlash timings share the same process and residency state. Exact
command:

```bash
scripts/gpuq try -- env KAPLUMBA_TEST_MODEL="$HOME/.kaplumba/models/Gxma-repacked" \
  KAPLUMBA_DFLASH_DIR="$HOME/.kaplumba/models/Drafter" \
  cargo test --release -p kaplumba-engine --test dflash_equivalence \
  dflash_greedy_stream_matches_target_for_200_tokens_on_three_prompts \
  -- --ignored --exact --nocapture
```

Validation: `cargo test --release -p kaplumba-engine --test
dflash_equivalence --no-run` compiled the instrumented test; the forced-load
run above passed 1/1. No product throughput code changed. The next T19 step is
to split draft computation from target verification inside the round, then
choose GPU-resident stage work based on those measurements.

## 2026-10-14 — T19b proposal/verify phase split

Temporarily instrumented `DflashModel::draft_and_verify` behind
`KAPLUMBA_DFLASH_PHASE_TIMING=1` and removed the hook after measurement. The
forced-load 3 × 200 greedy run passed exact output equivalence for all 600
tokens. Across 229 rounds, phase means were: `draft_block` 169.94 ms, proposal
selection 0.00 ms (greedy), target `forward_verify` 374.45 ms, and acceptance
plus cache retraction 0.69 ms; total 545.08 ms. This reconciles with T19a's
~545 ms draft-and-verify bucket. There is no proposal-sampling cost in this
greedy run; sampled verification still needs its own profile if considered for
throughput.

This makes the main tradeoff concrete: drafting and verifying 15 proposals
costs ~544 ms while current runs accept only ~1.65 on average. Raising overlap
is necessary, while the 170 ms drafter and 374 ms verifier also need reduction
for 30 tok/s. The measurement alone does not establish either change will
reach 30 tok/s.

Exact timed command:

```bash
scripts/gpuq try -- env KAPLUMBA_DFLASH_PHASE_TIMING=1 \
  KAPLUMBA_TEST_MODEL="$HOME/.kaplumba/models/Gxma-repacked" \
  KAPLUMBA_DFLASH_DIR="$HOME/.kaplumba/models/Drafter" \
  cargo test --release -p kaplumba-engine --test dflash_equivalence \
  dflash_greedy_stream_matches_target_for_200_tokens_on_three_prompts \
  -- --ignored --exact --nocapture
```

The instrumented run passed 1/1 in 213.23 s. The same 86,720,512-byte range
again failed `mlock`; GPUQ reported 16 GiB available and no swap before the
run, and 18 GiB available with no swap afterward. After removing the timing
hook, `cargo test --release -p kaplumba-engine --test dflash_equivalence
--no-run` compiled successfully. No persistent product behavior change was
made by the profiler.

## 2026-10-14 — T19c verified-prefix sweep

Added test-only `KAPLUMBA_TEST_DFLASH_BLOCK_SIZE` and
`KAPLUMBA_TEST_PROMPT_COUNT` controls to `dflash_equivalence`. The block override
is constrained to `[2, trained_block_size]`; runtime configuration and
production defaults are untouched. Each measured candidate generated 200
tokens and compared every token ID against the sequential target. B=4 was then
confirmed across all three prompts.

| Proposal block B | Prompt runs | DFlash tok/s | DFlash rounds / accepted drafts | Plain tok/s |
|---:|---|---:|---|---:|
| 2 | prompt 0 ×3 | 6.34, 6.34, 6.37 | 116 / 84 each | 12.32–12.33 |
| 4 | prompt 0 ×3 | 6.95, 6.97, 6.97 | 85 / 115 each | 12.31–12.33 |
| 4 | prompts 0/1/2 | 6.97, 6.90, 7.05 | 85/115, 86/116, 84/115 | 12.32–12.33 |
| 8 | prompt 0 ×1 | 5.99 | 77 / 127 | 12.35 |
| 16 | prompts 0/1/2, T19a baseline | 3.94, 3.85, 4.33 | 78/127, 80/123, 71/128 | 12.35–12.36 |

B=4 is consistently faster than B=2 and full B=16 on these prompts. B=8
accepts more than B=4 but pays for weaker late rows. This is an interim result
only: B=4 reaches about 7 tok/s, far from 30 tok/s, and was not wired into a
runtime entry point. The confirmed gain justifies reducing avoidable work, but
the primary draft and verify synchronization costs remain. Do not describe
shorter B as a 30 tok/s fix.

Build validation: `cargo test --release -p kaplumba-engine --test
dflash_equivalence --no-run` passed. Forced-load runs used `scripts/gpuq try`;
all passed exact greedy comparison. The three-prompt B=4 run passed 1/1 in
150.58 s. One 86,720,512-byte range failed `mlock` in each process; GPUQ showed
no swap. Exact command for the three-prompt B=4 confirmation:

```bash
scripts/gpuq try -- env KAPLUMBA_TEST_DFLASH_BLOCK_SIZE=4 \
  KAPLUMBA_TEST_PROMPT_COUNT=3 \
  KAPLUMBA_TEST_MODEL="$HOME/.kaplumba/models/Gxma-repacked" \
  KAPLUMBA_DFLASH_DIR="$HOME/.kaplumba/models/Drafter" \
  cargo test --release -p kaplumba-engine --test dflash_equivalence \
  dflash_greedy_stream_matches_target_for_200_tokens_on_three_prompts \
  -- --ignored --exact --nocapture
```

## 2026-10-14 — T19d encoder-facing f32 element operations

Added `ElementKernels::encode_rms_norm_f32_in` and
`encode_residual_add_f32`, which enqueue the existing weighted f32-input norm
and f32 residual pipelines on a caller-owned `MetalEncoder`. The encode APIs
validate dimensions and caller buffer lengths before dispatch. No Metal shader
or slice-path behavior changed.

Added `crates/kernels/tests/dflash_encode.rs`. Its 4 × 32 f32 norm input
contains values around 150k (above fp16 range), and encoded output matched the
existing slice API exactly. A 67-element f32 residual case also matched exactly
and confirmed the result remains above 65,504. Zero count, zero epsilon, and
undersized output/input buffers are rejected. `scripts/gpuq try -- cargo test
--release -p kaplumba-kernels --test dflash_encode` passed 1/1.

Coverage sabotage replaced the encoded residual's right operand with the left
operand. The test failed at the output parity assertion (line 93; GPU output
contained `left + left`, expected `left + right`; exit 101). The binding was
restored and the test passed. This confirms the residual binding is exercised.

## 2026-10-14 — T19e encoder-facing gated SiLU

Added `ElementKernels::encode_gated_silu`, which enqueues the existing gated
SiLU pipeline with caller-owned f16 gate/up/output buffers and rejects zero
count or undersized buffers. The `dflash_encode` integration test compares 67
mixed-sign inputs exactly against `ElementKernels::gated_silu`; the combined
T19d/T19e focused test passed 1/1.

Coverage sabotage bound `gate` in place of `up`. The test failed at the gated
SiLU parity assertion (line 116; exit 101). Restored the binding; clean rerun
passed 1/1. No shader was added.

## 2026-10-14 — T19e.1 encoder-facing f32-output GEMM

Added `LinearF32EncodeRequest` and `LinearKernels::encode_gemm_f32`. It encodes
the existing f16 input transform, f32 accumulator zero, tiled EXL3 GEMM, and
f32 output-sign finalize into the caller's `MetalEncoder`. `packed_bytes` and
`packed_offset` are both validated, and packed offset is passed through
`TensorRef`; all activation/scratch buffers remain caller-owned. No linear or
Metal shader was added.

Added `encoded_f32_buffer_gemm_matches_sliced_path_with_offsets` to
`crates/engine/tests/linear_metal.rs`. For 4/6 bpw and 1/4 rows, the encoded
outputs matched `gemm_chunk_buffer_f32_sliced` exactly with a 128-byte prefix
and suffix canary. A packed region one word too short and an output buffer one
byte too small were rejected. `cargo test --release -p kaplumba-engine --test
linear_metal encoded_f32_buffer_gemm_matches_sliced_path_with_offsets --
--exact` passed 1/1.

Coverage sabotage changed the encoded packed binding from `packed_offset` to
zero. The test failed at the output parity assertion (line 582; nonzero output
was read from the `0xA55A` prefix; exit 101). Restored the offset and reran; it
passed 1/1. The existing sampler edit in `crates/engine/src/sampler.rs` was not
staged.

## 2026-10-14 — T19f encoder-facing DFlash block attention

Added `DflashBlockBufferRequest` and `AttentionKernels::encode_dflash_block_attention`
to dispatch the existing `dflash_block_attention` pipeline into the caller's
`MetalEncoder`. Shape/window/mask params are validated; empty context requires
the same one-u16 dummy buffer as the slice path. No shader changed.

`dflash_encode` now checks exact output parity for four block rows, 4 Q heads,
2 KV heads and head dimension 32 under bidirectional full attention, a sliding
window, and empty context. It rejects a one-byte-short output. The full focused
suite passed 2/2.

Coverage sabotage bound context K as context V. The attention parity assertion
failed with differing fp16 outputs (exit 101). Restored the correct V binding;
the suite passed 2/2. This confirms context V is consumed by the tested path.

## 2026-10-14 — T19g.1 owned DFlash GPU scratch and weight views

`DflashModel::new` now creates a Metal command queue, reusable block-shaped
ping-pong/linear/attention/MLP buffers, and cached Metal buffers for all
drafter norm and EXL3 sign vectors. The scratch allocator accepts only model
and trained block dimensions; context length is not an input. Context K/V
buffers remain separate data staging for T19g.2. The existing `TextWeights`
owner and zero-copy packed views are unchanged.

On the actual Drafter checkpoint (`B=16`, hidden 5376, intermediate 10752,
Q width 8192, KV width 1024, five layers), the sizing test measured 5,636,096
bytes (5.37 MiB) fixed scratch and 1,090,048 bytes (1.04 MiB) cached norm/sign
buffers. Both are below the 32 MiB / 16 MiB task bounds. The forced-load
`dflash_gpu_scratch_is_block_sized_and_caches_norm_sign_buffers` test passed
1/1 in 8.22 s. Sabotaging `gpu_scratch_bytes()` to omit hidden buffers failed
with 4,947,968 vs 5,636,096; restored implementation passed again. No target
model or decode run was involved in this allocation check.

## 2026-10-14 — T19g.2 encoded DFlash block

Added `DflashModel::forward_block_gpu`. It stages each layer's host context K/V
and absolute key positions once, widens anchor/mask rows into owned f32 scratch,
then encodes all five layers and final norm into one `MetalEncoder`. Residual
streams remain f32; q/k norms, RoPE, values, gated SiLU, attention, and cached
norm/sign views stay on Metal. Per-layer block K/V is written to fixed-size
owned scratch; the only block readbacks happen after command completion. The
original `forward_block` remains unchanged as the slice-path oracle.

Added ignored real-checkpoint test `dflash_gpu_encoded_block_matches_slice_block`
in `crates/engine/tests/dflash_golden.rs`. It loads only the drafter (not the
host-dequantized reference matrices) and compares empty context, 20-row normal
context, and a mixed sliding/non-sliding context whose sliding layers retain
exactly the configured window. Clean GPUQ run passed 1/1 in 6.75 s. Maximum
relative errors versus the slice path:

| Context | K | V | final hidden |
|---|---:|---:|---:|
| empty | 8.795e-4 | 1.384e-3 | 4.241e-4 |
| normal | 7.770e-4 | 1.299e-3 | 4.386e-4 |
| sliding | 7.930e-4 | 9.135e-4 | 5.376e-4 |

The encoded GEMM selects the eligible MMA path while `gemm_chunk_buffer_sliced`
uses tiled GEMV, so outputs are not bit-identical; the measured difference is
under 0.14% and the test uses a 0.5% max-relative bound. The next task must
validate exact target generation through verify/rejection sampling and measure
acceptance and complete-round speed before adopting this path.

Coverage sabotage bound context K as context V. The test passed the empty case
(which intentionally has no context) and failed the normal case with V relative
error 5.844e-3 against the 5e-3 bound. Restored the V binding; reran clean and
all three cases passed. Focused commands:

```bash
cargo test -p kaplumba-engine --test dflash_golden dflash_gpu_encoded_block_matches_slice_block --no-run
scripts/gpuq status
scripts/gpuq try -- cargo test -p kaplumba-engine --test dflash_golden dflash_gpu_encoded_block_matches_slice_block -- --ignored --nocapture
```

The third hidden f32 ping-pong buffer raises fixed scratch to 5,980,160 bytes
(5.70 MiB); cached norm/sign data remains 1,090,048 bytes (1.04 MiB). The
initial version of the forced test used `Harness::build()`, which also
dequantized every checkpoint linear into host reference matrices and took
285 s; switching this test to a drafter-only load reduced its full run to under
7 s. No target model or decode run was involved.

## 2026-10-14 — T19g.3–T19g.6 route experiments and measurement correction

Kept an opt-in route from `draft_block_with_context` through the one-encoder
`forward_block_gpu`, selected by `KAPLUMBA_DFLASH_GPU_BLOCK=1`. The default
remains the slice path. On prompt 0, B=4 greedy output matched exactly: the
encoded route measured 7.05 tok/s (85 rounds, 115 accepted), versus 6.92 for
the slice route. B=16 encoded measured 4.17 tok/s. This is a small exploratory
gain, not a stable result; full 3 × 200 interleaved A/B and clean footprint
measurements remain required. The runs printed the existing 86,720,512-byte
mlock warning; do not claim full drafter residency from those runs.

Rejected two chained blocks: one prompt, B=4, exact greedy output, 74 rounds,
129 accepted, 5.43 tok/s. The second block's added drafter work outweighed
saved target verification. Removed `KAPLUMBA_DFLASH_BLOCK_COUNT` branch.
Rejected separately encoded target-head path: slice-block + encoded-head was
6.54 tok/s; encoded-block + encoded-head was 6.69 tok/s, below the 6.92 and
7.05 tok/s respective single-block routes. Removed the head scratch and
`KAPLUMBA_DFLASH_GPU_HEAD` branch; the extra command submission/readback
erased any head compute savings. These one-prompt measurements also occurred
while the global swap state was elevated, so they are directional only.

Found `scripts/gpuq`'s swap status parser was selecting the first positional
value (`total`) instead of the field named `used`. It therefore printed 0 MiB
while `sysctl -n vm.swapusage` showed about 3.2 GiB used. Fixed the parser in
`scripts/gpuq_mem.sh` to find `used` and convert K/M/G values; missing/malformed
fields return `?`. `bash scripts/test_gpuq_mem.sh` passes. Coverage sabotage
changed the parser to read `total`; the tests failed with `expected 3296 MiB
got 4096`, `expected 1536 MiB got 4096`, and `expected ? got 4096`. Restoring
the real parser made the full script pass. This demonstrates the old metric
was wrong but does not identify when swap accumulated or attribute it to any
specific DFlash run. Repeat performance A/B only after recording stable
pre-run and post-run swap values.

Validation after removing failed branches: `cargo check -p kaplumba-engine
--lib`; `cargo test --release -p kaplumba-engine --test dflash_equivalence
--no-run`; `git diff --check`; and `bash scripts/test_gpuq_mem.sh` all passed.
The build emitted existing unused/deprecated-cfg warnings. The separate
GPUQ parser correction is recorded as T19g.5; it changes benchmark telemetry,
not engine behavior.

## 2026-10-14 — B=4 encoded route A/B and phase attribution

Ran the same exact 3 × 200 greedy prompts through the encoded B=4 route and
slice control. Encoded speeds were 7.08/7.01/7.16 tok/s (median 7.08); slice
speeds were 6.92/6.89/7.05 (median 6.92). Same-process plain decode was
12.32–12.35 tok/s. All 600 DFlash token IDs matched the sequential target.
These were grouped runs (encoded then slice), not a repeated A/B/A/B sweep, so
the 2.3% median difference is exploratory and is not enough to adopt the path.
`gpuq status` showed 17 GiB reclaimable before the profiled encoded run. The
run's `/usr/bin/time -l` peak footprint was 14,353,076,224 bytes (13.37 GiB)
and reported zero swapouts. It still logged the 86,720,512-byte mlock failure;
these measurements do not prove all drafter pages are pinned. Global swap
allocation was ~3.2 GiB before and after; `sysctl vm.swapusage used` is not the
same measurement as process swapouts.

Temporarily added `KAPLUMBA_DFLASH_PHASE_TIMING=1` around encoded block body,
tied head, `forward_verify`, and acceptance/retraction, then removed the hook.
Across 255 rounds the medians/means were: block body 60.038/60.082 ms; tied
head 37.970/37.998 ms; target verify 137.735/137.637 ms; acceptance/retraction
0.528/0.606 ms. The test remained exact for all 600 output tokens, with
7.01–7.16 tok/s across prompts. Combined draft+verify phase medians are about
236 ms, consistent with the outer test's ~237 ms bucket; the caller separately
spends ~81 ms in the anchor target forward and ~13.7 ms appending context.
Acceptance and tap read are negligible. This identifies verification, the
separate anchor forward, and proposal quality as the largest remaining costs.

The target-call lower bound rules out B=4 for the 30 tok/s gate: even emitting
all four positions from one B=4 block would require about 221 ms of anchor plus
verify time per round at these measured costs, a ceiling of ~18 tok/s before
any draft work. Larger trained blocks and stronger accepted prefixes are
required. The B=4 route's small relative gain does not establish a path to 30.
Exact profile command (log at `/tmp/dflash-t19g7-profile.log`):

```bash
scripts/gpuq try -- /usr/bin/time -l env \
  KAPLUMBA_TEST_MODEL="$HOME/.kaplumba/models/Gxma-repacked" \
  KAPLUMBA_DFLASH_DIR="$HOME/.kaplumba/models/Drafter" \
  KAPLUMBA_TEST_DFLASH_BLOCK_SIZE=4 KAPLUMBA_DFLASH_GPU_BLOCK=1 \
  KAPLUMBA_DFLASH_PHASE_TIMING=1 \
  target/release/deps/dflash_equivalence-2b02cd992899a095 \
  dflash_greedy_stream_matches_target_for_200_tokens_on_three_prompts \
  --ignored --exact --nocapture
```

## 2026-10-14 — encoded B=16 three-prompt result

Ran the trained B=16 block through `forward_block_gpu` on all three exact
200-token prompts. Speeds were 4.16/4.06/4.58 tok/s (median 4.16); plain
same-process decode was 12.32/12.33/12.32 tok/s. Rounds were 78/80/71 with
127/123/128 accepted drafts. Every output matched sequential greedy decode.
Per-round draft+verify was 520.97/520.15/519.68 ms; the separately processed
anchor was ~81 ms and context append ~14 ms. `gpuq status` reported 16 GiB
reclaimable before the run and ~3.28 GiB global swap allocation; the loader
again reported the 86.7 MB mlock failure. `/usr/bin/time -l` was not wrapped
around this run, so no footprint number is claimed. This confirms B=16 is
slower than B=4 with the encoded path and remains far below the target.

Exact command:

```bash
scripts/gpuq try -- env KAPLUMBA_TEST_MODEL="$HOME/.kaplumba/models/Gxma-repacked" \
  KAPLUMBA_DFLASH_DIR="$HOME/.kaplumba/models/Drafter" \
  KAPLUMBA_DFLASH_GPU_BLOCK=1 cargo test --release -p kaplumba-engine \
  --test dflash_equivalence \
  dflash_greedy_stream_matches_target_for_200_tokens_on_three_prompts \
  -- --ignored --exact --nocapture
```

## 2026-10-14 — interleaved B=4 A/B and known-token override rejection

Completed three prompt-0 A/B pairs in the order encoded/slice/encoded/slice/
encoded/slice. Encoded B=4 tok/s: 7.07, 7.05, 7.08 (median 7.07); slice:
6.96, 6.94, 6.95 (median 6.95). Pairwise differences were 0.11, 0.11,
0.13 tok/s; all six 200-token outputs were greedy-exact and acceptance stayed
85 rounds/115 drafts each. This confirms a small repeatable local improvement,
but both routes remain far below plain decode (~12.3 tok/s); the encoded route
stays opt-in. `gpuq status` was checked before every arm. Global allocated swap
varied from 3.24 to 3.42 GiB during the sweep; this is not process swapout
count. The earlier encoded `/usr/bin/time -l` run recorded 13.37 GiB peak
footprint and zero swapouts.

Tested a greedy proposal-zero override using `argmax(previous_logits)`, which
is the already-computed exact target token. A test assertion required the first
accepted candidate to equal that token. Coverage sabotage replaced the override
with a no-op; it failed at prompt 0, round 7 because no candidate was accepted.
With the override enabled, the 3-prompt exact test then failed at prompt 0,
output token 77: DFlash emitted 99382 while the sequential target emitted
84750. Earlier tokens matched. The old speculative sequence had rejected
before reaching this affected verifier row; forcing the first token advanced
the accepted prefix into the drift. Removed the override and its test assertion.

The T19i follow-up isolated that path. Recomputed the natural greedy prefix and
compared the exact B=4 shape (M=3) with sequential decode: rows producing output
tokens 75, 76 and 77 chose 563, 4512 and 84750 on both paths. At token 77 the
sequential top-two gap was only 0.015625 and max per-logit delta was 0.046875,
so this is a numerically sensitive row but not a mismatch from a clean cache.
After retracting the two M=3 tail rows, continuation also matched. Repeated
M=3 rounds accepting one ground-truth proposal and retracting two tails matched
through output token 79.

Then replayed the forced-override experiment's accepted-prefix pattern using
the ground-truth target proposals. That altered path again flipped at output
token 77 after start=73 accepted=3: its anchor chose 99382 vs sequential 84750,
with max logit delta 0.048828 and sequential gap 0.015625. The accepted target
cache had small accumulated K/V differences: 0.001953 at layer 0, growing to
0.074219 at layer 39. Padding the same operation to M=4 did not remove it. This
explains why the experimental override exposed a flip; it does not show a
failure in the unmodified DFlash stream. The override and the diagnostic test
that expected drift were removed. The remaining focused test covers natural
prefix M=3 rows, cache rollback continuation and repeated one-accepted rounds.

Coverage sabotage changed the M=3 positions by +1. The test failed with the
diagnostic `positions must begin at the current cache position` before logits
were compared. Restoring exact positions made the focused test pass in 33.98 s.
The test also logged output-77's close margin. No model change was justified.
Failed forced-override run: `/tmp/dflash-t19h-sabotage.log`; cache-drift replay
numbers came from the temporary accepted-pattern diagnostic, which was removed.

## 2026-10-14 — repeated rejection around the sliding-window boundary

Added ignored GPU test `repeated_rejections_preserve_live_cache_and_preflight_errors_are_atomic`
in `crates/engine/tests/dflash_verify_head.rs`. It prefills 1022 tokens, confirms
a non-contiguous verify-position error is rejected before any layer-0 cache
bytes or position slots change, then runs six four-row full rejections followed
by one target correction each. The sequence crosses position 1024. Correction
argmaxes matched an independently replayed sequential stream and max logit
delta stayed below 2.0. On layer 0, every live logical position (4 through 1027
at the final exclusive cursor 1028) had identical slot mapping, absolute position, K and V to the
sequential reference. This covers repeated stale-tail writes that T1b.2a's
single rollback cases did not.

Sabotaged `TextModel::retract_gpu_caches` to retract `tokens - 1`; the new test
failed with cursor 1023 vs expected 1022 immediately after the first rejection.
Restored the implementation and reran clean: 1 passed in 32.88 s. Build command:
`cargo test --release -p kaplumba-engine --test dflash_verify_head --no-run`.
GPU command:
`scripts/gpuq try -- env KAPLUMBA_TEST_MODEL="$HOME/.kaplumba/models/Gxma-repacked" cargo test --release -p kaplumba-engine --test dflash_verify_head repeated_rejections_preserve_live_cache_and_preflight_errors_are_atomic -- --ignored --exact --nocapture`.
Preflight failure is atomic; a post-dispatch device error is still not injected,
so parent task T1b.2 remains open. No model code changed.

## 2026-10-14 — T5 drafter permutation propagation

The DFlash buffer GEMM requests carry `QuantizedLinear::permutation_mode` at
all projection and sliced-head call sites. `buffer_perm_mode` checks that both
buffer entry points respond to that field; its earlier parameter-site sabotage
fails. Extended the real-checkpoint host oracle to toggle
`DflashModel::set_permutation_mode` on `fc`: default mode 0 matches independent
host `reconstruct_matrix`, mode 1 must differ and fail that same host bound,
and restoring mode 0 must reproduce the original output bit for bit.

```bash
scripts/gpuq try -- env KAPLUMBA_DFLASH_DIR="$HOME/.kaplumba/models/Drafter" \
  cargo test --release -p kaplumba-engine --test dflash_golden \
  dflash_exl3_decode_matches_host_on_every_linear -- --exact --nocapture
```

All 36 actual buffer linears matched host reconstruction at the loaded default;
worst absolute error was 0.003663 (layer 0 `o_proj`). The mode-1 `fc` result
failed the host bound. Setter sabotage hardcoded mode 0 inside
`DflashModel::set_permutation_mode`; the test failed by name because mode 1
still matched the host at 0.001942 error. Restored the setter and reran clean;
all 36 passed in 30.72 s. This closes T5. The independent default-layout
interpretation is recorded separately under T6. This test does not explain the
measured low accepted-prefix count or improve the 7.07 tok/s result.

## 2026-10-14 — T6 checkpoint layout/default

`Drafter/quantization_config.json` has no layout key; `QuantizedLinear::from_metal_buffer`
therefore supplies the mode-0 default. The host oracle's 36 independent EXL3
reconstructions all match actual mode-0 buffer GEMMs (worst error 0.003663).
Mode 1 on `fc` fails the same reference by more than the 0.013644 acceptance
bound; a separate temporary loader override to mode 1 failed at `fc` with
10.495664 max error. This selects mode 0 as the checkpoint's actual row-major
layout. The earlier mode-1/colmajor alpha result came from the invalid target
head oracle and is superseded. No DFlash weight or layout default changed.

T5/T6 result: the old colmajor claim was wrong; correcting it does not raise
acceptance. DFlash remains at 7.07 tok/s in the best measured B=4 route, so the
30 tok/s objective still requires a materially stronger drafter and/or a much
faster target verifier.

## 2026-10-14 — F1c same-target DFlash checkpoint compatibility

Reviewed `RedHatAI/gemma-4-31B-it-speculator.dflash` from its official Hugging
Face model card and fetched only `config.json` plus the first 128 KiB of the
Safetensors file to inspect its header. No checkpoint weights were downloaded.
The model card reports average accepted lengths from 3.9846 (summarization) to
6.6225 (math reasoning), block size 8 with 7 proposed tokens, taps
`[1, 17, 29, 47, 59]`, BF16, and 3B parameters. These are published external
results, not local benchmark evidence. The Safetensors model file is 5.08 GB;
the repository's 10.2 GB total includes a similarly sized optimizer state.

The artifact is not usable by the current path: its tensor header contains
dense BF16 `.weight` tensors (including `fc.weight [5376, 26880]`) and no EXL3
`.trellis`, `.mcg`, `.suh`, or `.svh` tensors. `DflashWeights::load` only builds
projections from `.trellis` plus MCG and input/output scales. Its config schema
uses `aux_hidden_state_layer_ids`, top-level `mask_token_id`, and nested
`transformer_layer_config`; `DflashConfig::load` currently expects top-level
`layer_types`, `num_hidden_layers`, and `dflash_config.target_layer_ids`.
Therefore config normalization alone cannot make the artifact load; a dense
BF16 DFlash projection path or an offline compatible quantization pipeline is
needed. The public config also selects a different draft attention geometry
(32 Q / 16 KV heads × 256 vs local 64 / 8 × 128), which must be validated against
the kernels if a future format path is designed.

Decision: reject this candidate for the current DFlash experiment; do not
download 5.08 GB or replace the existing 733 MB EXL3 drafter. Reopen only with
a concrete format/kernel or quantization plan and a measured footprint estimate.
No GPU test was appropriate because the existing loader cannot consume the
artifact. F1c is recorded complete as a read-only compatibility review; 30 tok/s
remains unmet.

Sources: <https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.dflash>
and its `config.json` / `model.safetensors` headers; local checks against
`crates/engine/src/dflash.rs` (`SUFFIX_RULES`, `DflashWeights::load`,
`DflashConfig::load`).

## 2026-10-14 — T11 validated batched 6-bpw verify head

T17's packed-offset fix made the existing `encode_gemm` route usable for the
target's 6-bpw tied head. `encode_lm_head_rows` now final-normalizes all hidden
rows together, calls the existing GEMM with `lm_head.metal_offset`, writes a
contiguous fixed-capacity logits buffer, then softcaps it. No kernel family was
added. `head_rows_matrix` returns only rows that fit the allocated buffer.

Correctness and sabotage:

- `batched_verify_head_matches_sequential_single_row` passed for M=1/2/4/8/16
  on three real prefixes. Greedy IDs matched every row; largest logit difference
  was 0.4375 (at M=16), within the existing diagnostic bound.
- `batched_verify_head_carries_signal_on_every_row` passed; adjacent rows differ
  in 4,190,784 of 4,194,304 logits.
- Temporarily forced `packed_offset: 0` in the batched head. The named greedy
  comparison failed at M=1 row 0 (14697 vs 236777). Restoring
  `self.lm_head.metal_offset` and rerunning passed.
- `cargo check -p kaplumba-engine` and release integration test compilation
  passed. The `dflash_head_ab` harness was formatted with `rustfmt`.

Paired B-A-B timing used the same target, prompt seeds, and one M=16 verify per
repeat. A was the previous committed rowwise head in a detached worktree with a
separate `CARGO_TARGET_DIR`; B was the new batched head in the main tree. Three
prompts × three repeats per run, short 32-token prefix; every row matched the
saved sequential greedy continuation:

```text
B1 batched median 368.240 ms (samples 366.708–368.567)
A  rowwise median 376.376 ms (samples 375.445–377.996)
B2 batched median 367.591 ms (samples 366.945–368.961)
```

The two batched medians are both about 8.5 ms (~2.3%) faster than A; B-A-B
guards against a single warm/cold timing. The gain is real but small and does
not close the DFlash throughput gap.

The full V0b test was rerun after the change with 3 prefixes × 4 contexts ×
M=2/4/8/16 × 3 interleaved repeats. All 144 verify rows matched the saved
target-greedy continuation. M=16 medians across the three prompt medians:

```text
context     old rowwise V0b   new batched median   ideal unprocessed ceiling
32          377.699 ms        369.422 ms           43.31 tok/s
1024        410.785 ms        402.254 ms           39.78 tok/s
2048        421.480 ms        413.015 ms           38.74 tok/s
8192        461.548 ms        452.821 ms           35.34 tok/s
```

The cross-run full-matrix deltas are descriptive; only the short-context
B-A-B above is the paired speed claim. The 8k run completed in 964.68 s. Swap
used was 3,051 MiB before and 3,704 MiB after the run; available memory returned
to 18.0 GiB after the process exited. This is an ideal-proposal ceiling, not
DFlash throughput: at 8k, 16 rows still consume 452.8 ms and leave only 80.5 ms
to emit 17 tokens at 30 tok/s (the historical unprocessed-anchor budget is
59.8 ms for 16 tokens). The existing DFlash draft/update work remains over the
budget, and 30 tok/s is not achieved.

Repro commands:

```bash
scripts/gpuq try -- env KAPLUMBA_TEST_MODEL="$HOME/.kaplumba/models/Gxma-repacked" \
  cargo test --release -p kaplumba-engine --test dflash_head_ab \
  m16_verify_head_is_greedy_exact_and_timed -- --ignored --exact --nocapture

scripts/gpuq try -- env KAPLUMBA_TEST_MODEL="$HOME/.kaplumba/models/Gxma-repacked" \
  cargo test --release -p kaplumba-engine --test dflash_verify_ceiling \
  ideal_target_continuations_bound_verify_throughput -- --ignored --exact --nocapture
```

## 2026-10-14 — F1d 6-bpw drafter candidate evaluation

Candidate: `eepos/gemma-4-31B-it-DFlash-exl3-6.0bpw`, local test copy at
`~/.kaplumba/models/Drafter-eepos-exl3-6bpw`; source payload SHA-256
`aeed2f2160e5cf1e412d3300b5e18aa3c5554366c6b0019424c341e08425001a`. The
test copy's safetensors header was padded by one JSON whitespace byte to make
packed payload offsets 4-byte aligned; tensor payload bytes are unchanged.
T20.2 rate inference and F1d's explicit-path failure check allowed the actual
candidate to reach the full host/GPU checks.

All 36 EXL3 linears passed host comparisons; maximum absolute GPU/host error
was 0.002741. The encoded block test passed and measured 70.41 ms vs 93.55 ms
for the slice path on that isolated operation. The 3×200-token end-to-end
greedy streams matched the target exactly:

```text
candidate 6-bpw B16: 3.92 / 3.72 / 4.04 tok/s (median 3.92), accepted 130/124/126
candidate 6-bpw B4 slice: 6.21 / 5.86 / 6.05 tok/s (median 6.05), accepted 119/115/115
candidate 6-bpw B4 encoded: 6.49 / 6.09 / 6.31 tok/s (median 6.31), accepted 119/115/115
candidate B2 p0: 5.27 tok/s, accepted 84; candidate B8 p0: 5.63 tok/s, accepted 131
same-prompt B4 encoded A/B/A/B: existing 4-bpw 6.23/6.24; candidate 6-bpw 6.50/6.51 tok/s
plain target decode in the same test process: 12.30–12.34 tok/s
```

The A/B/A/B comparison indicates a repeatable 4.3% improvement for this one
prompt, but the candidate is not promoted: it stays far below the 30 tok/s
goal and other prompts show no evidence that the larger drafter closes the
gap. B4 encoded is the best measured mode among those tested, at 6.31 tok/s
median across the three prompts.

Short-prompt process peak `phys_footprint` was 14.44–14.53 GB and process
swapouts were zero. These are not the required 2k-context memory results. In a
target-plus-drafter run, the target pinned 11.22 GiB of trellis data; then
`mlock` of the drafter failed on its first range (130,072,576 bytes for 6-bpw,
86,720,512 bytes for the existing 4-bpw drafter, both after zero drafter bytes
were locked). Thus the drafter is not guaranteed resident alongside the target
under the current global wire budget. This failure is shared by both drafter
rates and does not justify reducing target pinning or changing system limits.

Reproduction used the release `dflash_golden` ignored tests with
`--test-threads=1`, `scripts/gpuq try --`, and `/usr/bin/time -l` on the direct
test binary for process footprint. See task F1d for per-prompt streams and
candidate status. This candidate experiment is complete; the overall 30 tok/s
acceptance remains open.

## 2026-10-14 — F1e official 5-bpw candidate (in progress)

The Hugging Face tree for `turboderp/gemma4-31b-it-DFlash-exl3`, branch
`5.00bpw`, lists one 961,045,318-byte `model.safetensors` shard with LFS SHA-256
`5ae9ffe6f4c6250d7f08f0265b1aaff17bca6645b0879097e24aa447bd2f95ca`. Its
`config.json` matches the local five-layer DFlash geometry, block size 16, tap
IDs `[1, 12, 23, 35, 46, 57]`, 8 KV heads × 128, and declares 5 bpw. Download
to `~/.kaplumba/models/Drafter-turboderp-5bpw` was started after the F1e task
was added in `d155172`; no model test has run yet. The transfer is slow (~1.1
MB/s); verify full file size/hash and header alignment before loading. The
pre-run `gpuq status` was 16.0 GiB available with 3160 MiB system swap; rerun
the required guard immediately before any GPU test.

The in-progress shard already exposes a 16,558-byte Safetensors header. Its 36
trellis absolute starts are all 2 mod 4; the data section begins at byte 16,566.
After download, retain the original and make the isolated test copy with two
trailing JSON whitespace bytes (header length 16,560, data offset 16,568), then
verify the packed payload SHA-256 before testing the mmap/no-copy route.

The download completed at exactly 961,045,318 bytes and matched the expected
LFS SHA-256. The test copy is 961,045,320 bytes; the original remains in
`Drafter-turboderp-5bpw-source`. The packed-payload hash is identical before and
after normalization:
`4efc195cae66af89741db141cafe9f9a2c78b17704735e50c011b376cab8bbc6`.
The normalized header is 16,560 bytes; all 36 packed starts are now 0 mod 4.
Header rate dimension is 80 for all trellises (5 bpw), with 166 recognized
tensors and the expected tap/attention config.

`dflash_exl3_decode_matches_host_on_every_linear` passed all 36 linears; the
worst absolute GPU/host error was 0.002120. Loaded alone, the candidate mmap
loader pinned all 36 trellis tensors, 0.89 GiB. Under a target-plus-drafter
stream, the target pinned 11.22 GiB, then candidate `mlock` failed on the first
108,396,544-byte range after zero drafter bytes were locked. Existing 4-bpw
also failed at its first 86,720,512-byte range. The candidate and A/B controls
both had that residency limitation.

The complete 3 × 200 greedy tests matched the sequential target at every block
size. Results (tok/s, accepted drafts, and rounds in prompt order):

```text
B4:  6.18 / 5.84 / 6.00; accepted 118 / 115 / 115; rounds 82 / 87 / 84; median 6.00
B8:  5.57 / 5.29 / 5.53; accepted 131 / 125 / 124; rounds 74 / 78 / 75; median 5.53
B16: 3.92 / 3.80 / 3.91; accepted 130 / 126 / 124; rounds 75 / 77 / 75; median 3.91
plain target during these runs: 12.31–12.37 tok/s
```

`/usr/bin/time -l` at B4/B8/B16 reported peak `phys_footprint` of
14,533,208,336 / 14,538,041,640 / 14,552,345,400 bytes. Each process reported
zero process swapouts and zero block input operations; system swap moved from
3048 MiB before the B4 run to 3175 MiB after B8, so do not infer absence of
system-wide pressure from process counters alone.

Interleaved prompt-0 B4 encoded A/B/A/B used the existing 4-bpw drafter for A
and the official candidate for B. All four streams were greedy-exact:

```text
A (4 bpw): 6.22 / 6.24 tok/s, accepted 115 each
B (5 bpw): 6.46 / 6.46 tok/s, accepted 118 each
```

The 5-bpw candidate gains about 3.5% on that one prompt under the best encoded
B4 route. It does not improve B4 throughput across the three prompts in the
default slice route, gives no B8/B16 advantage, and remains far below 30 tok/s.
Do not promote it. The result reinforces the prior 6-bpw finding: small precision
or calibration gains do not solve the long-prefix survival and complete-round
cost limits. F1e is evaluated; AC5 remains unmet.

Commands, all through `scripts/gpuq try --` after `scripts/gpuq status`:

```bash
KAPLUMBA_DFLASH_DIR="$HOME/.kaplumba/models/Drafter-turboderp-5bpw" \
  target/release/deps/dflash_golden-fa25a1842fb94bb7 \
  dflash_exl3_decode_matches_host_on_every_linear --exact --nocapture

KAPLUMBA_TEST_MODEL="$HOME/.kaplumba/models/Gxma-repacked" \
KAPLUMBA_DFLASH_DIR="$HOME/.kaplumba/models/Drafter-turboderp-5bpw" \
KAPLUMBA_TEST_DFLASH_BLOCK_SIZE=4 KAPLUMBA_TEST_PROMPT_COUNT=3 \
  target/release/deps/dflash_equivalence-2b02cd992899a095 \
  dflash_greedy_stream_matches_target_for_200_tokens_on_three_prompts \
  --exact --ignored --nocapture --test-threads=1
```

For B8/B16, set `KAPLUMBA_TEST_DFLASH_BLOCK_SIZE` to `8`/`16`. The repeated
A/B/A/B run used the same command with `KAPLUMBA_DFLASH_GPU_BLOCK=1` and
`KAPLUMBA_TEST_PROMPT_COUNT=1`, alternating `Drafter` and the 5-bpw path. No
project code changed for F1e.

## 2026-10-14 — F1b Gemma-4 EAGLE-3 feasibility review

Read-only review selected `thoughtworks/Gemma-4-31B-Eagle3` as the concrete
Gemma-4-31B EAGLE-3 artifact. No model weights were downloaded. The Hugging Face
tree lists `model.safetensors` at 1,296,995,832 bytes, LFS SHA-256
`fff0c01d917611706321a74418b3000333d0e756fbcd2b47a00cee2c25030000`. Its
config specifies hidden size 5376, auxiliary taps `[2,29,56]`, one draft layer,
42 Q heads / 14 KV heads, head dimension128, 32,000 draft vocab, and maximum
position4096. The Safetensors header was read via HTTP Range for bytes0–65,535
only (header length1,520). It contains:

```text
fc.weight                         BF16 [5376, 16128]
midlayer.self_attn.{q,k,v,o}     BF16, one dense decoder layer
midlayer.mlp.{gate,up,down}_proj BF16, intermediate 16384
lm_head.weight                   BF16 [32000, 5376]
d2t                              I64  [32000]
t2d                              BOOL [262144]
```

This is not loadable by `DflashWeights`: there are no EXL3 trellis/MCG/scales,
and the model is an EAGLE3 decoder rather than a five-layer masked DFlash block.
The existing target tap API can capture the 5376-wide target layers, but feature
layer indexing and runtime parity with the checkpoint's SGLang training backend
need direct validation. Draft IDs must map through `d2t`; `t2d` identifies the
reduced-vocabulary subset. Exact sampled verification must preserve the complete
262,144-entry target p, use the mapped draft q only on its 32k support, and draw
the rejection residual from full target support. The target embedding is not a
checkpoint tensor; the shared target embedding lookup must map draft IDs through
`d2t`. Position4096 also leaves the target's 8192 context limit unsupported
without a new strategy.

Footprint estimate: the measured target + current 5-bpw DFlash short-prompt peak
is 14.55 GB; replacing its 961 MB draft shard by this 1.297 GB shard adds about
0.336 GB, or ~14.89 GB before draft-specific scratch/cache differences. This is
below20 GB as an estimate, not the required measured 2k result. Target-plus-draft
`mlock` already fails for a 0.96 GB DFlash file after target pins11.22 GiB, so
the 1.30 GB EAGLE3 weights would not have proven residency without a new measured
wire budget. No load was attempted.

The current kernel crate exposes EXL3 quantized GEMM but no dense BF16 GEMM
interface. EAGLE3 would need a dense path for `fc`, attention, MLP, and the
32k-output head, plus mapped embedding/cache/state code. This is substantial
new kernel and model work. The model card reports 1.05–1.72× on 8×H200 for
different datasets; applying1.72× illustratively to local plain12.3 tok/s gives
21.2 tok/s. This arithmetic is not a local prediction and gives no evidence
for30. Design conclusion: no weights/code, no implementation selected; F1b
feasibility review is complete but the overall 30 tok/s objective remains open.

Sources:

- Model config/header/card: <https://huggingface.co/thoughtworks/Gemma-4-31B-Eagle3>
- Publisher benchmark table and H200 context: <https://huggingface.co/thoughtworks/Gemma-4-31B-Eagle3/blob/main/README.md>
- SGLang EAGLE3 auxiliary hidden-state configuration:
  <https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner_components/spec_aux_hidden_state.py>
- SpecForge reduced-vocabulary teacher/token mapping:
  <https://github.com/sgl-project/SpecForge/blob/main/specforge/algorithms/eagle3/model.py>
