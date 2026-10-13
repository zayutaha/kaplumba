# spec/dflash-round-30tps — implementation notes

**Historical entries below retain their original observations and hypotheses.**
The 2026-10-13 documentation audit at the end supersedes stale task statuses,
round budgets, sampled-correctness claims, and head/ranking conclusions.
The subsequent `exllama_review.md` supersedes the earlier colmajor-layout
conclusion and overlap-based checkpoint judgments: their head read the wrong
weight offset. The exllamav3 checkout is present under the nested package path.

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

### D14 — the acceptance matrix, and the probe was sampling the wrong side of it

`dflash_alpha_probe` now prints the full `(row, t)` acceptance matrix instead of
only its maximum (`COLLECT = 6`, so `t` spans 0..5 — it was printing `best` over
a matrix it never showed). Real checkpoints, colmajor:

```
  [d14]        t:      0      1      2      3      4      5
  [d14] row   1       0.000  0.000  0.000  0.000  0.000  0.848
  [d14] row   2       0.000  0.000  0.000  0.000  0.000  0.848
  [d14] row   3       0.000  0.000  0.000  0.000  0.000  0.848
  [d14] row   4       0.000  0.000  0.000  0.000  0.000  0.848
  [d14] row   5       0.000  0.000  0.000  0.000  0.000  0.831
  [d14] row   6..11   0.000  0.000  0.000  0.000  0.000  0.848
  [d14] row  12..15   0.000  0.000  0.000  0.000  0.000  0.02..0.006
  [d14] per-row argmax offset (t - row): +5 +4 +3 +2 +1 +0 -1 -2 ... -10
```

Two things and they are not the same thing:

1. **The signal is column-constant at t=5, not diagonal.** Every row from 1 to
   11 lands on 0.83-0.848 against the target's distribution for `total + 5`,
   and on 0.000 for every other sampled position. A block whose row `r`
   predicts position `total + r` would show a *ridge* (high values along
   `t == r`); this is a *column*, so the drafter is predicting essentially the
   same token in every row — consistent with "4 distinct tokens of 16" and with
   `[d4] row 1 q top3: 496:0.9986 759:0.0005 614:0.0003`. The block's rows are
   not yet carrying distinct positions.
2. **The per-row argmax offset is a perfect arithmetic sequence that saturates
   at the window edge** (`+5, +4, +3, +2, +1, 0, -1, ... -10`). Every row's best
   column is the last one that was sampled, which means the comparison window
   `COLLECT = 6` is smaller than the displacement between the drafter's rows
   and the probe's assumed diagonal. `argmax_t(row) = 11 - row` outside the
   window. **The 0.1663 mean is a sampling artefact: it is an average over a
   diagonal that does not intersect the signal, taken across a window too short
   to reach it.**

So the honest state after D1+D2:

* the permutation bug is real and fixed (`alpha` 0.0000 -> 0.1663 on the
  probe's diagonal, and every mis-permutation signature gone);
* **0.848 exists in this data** — the drafter's distribution for its dominant
  token overlaps the target's by 0.85 — but nobody has yet measured the
  per-depth acceptance on the diagonal the round will actually use, because the
  probe's window is 6 deep and the block is 16 deep.

The next experiment is one line in this probe (`COLLECT`) and no new code:
sample `t` out to ~20-24 so the ridge is inside the window, and read whether the
row-1 acceptance on the real diagonal is ~0.85 or ~0. If it is 0.85, alpha is
real and the remaining question is the cheap-round work; if the column at t=5 is
coincidence, the drafter's rows are genuinely row-independent and block
speculation cannot pay at any round cost.

### D14 with COLLECT = 24 — MEASURED, and it is a clean negative result

One line (`COLLECT 6 -> 24`), no new code. Real checkpoints, colmajor:

```
  [d14] row   1   .000 .000 .000 .000 .000  .848 .000 .000 .000 .000 .000 .000
                                 .001 .015 .000 .000 .007 .001 .000 .000 .000
                                 .003 .000 .441
  [d14] row  11   .000 .000 .000 .000 .000  .848 .000 ... .003 .000 .441
  [d14] row  15   .000 .000 .000 .000 .000  .006 .000 ... .003 .000 .006
  [d14] per-row argmax offset (t - row): +5 +4 +3 +2 +1 +0 -1 -2 -3 +14 +13 -6 -7 +10 +9 -2
  [d11] M=1 head: mean alpha 0.0565, best 0.8479 (row 1 t 5), 4 distinct tokens of 16
```

**The 0.848 is one column, not a ridge.** Rows 1-11 all agree with the target at
`t = 5` and nowhere else, and the per-row offsets are no longer the arithmetic
sequence they were at `COLLECT = 6` (they were saturating the window edge; now
they scatter, which is noise, not structure). The rest of the 16 x 24 matrix is
0.000 apart from a column at t=23 worth 0.44.

So the earlier reading is refuted: it was not "the diagonal is shifted", it was
"the block's rows are the same distribution". `[d11] 4 distinct tokens of 16`
says the same thing directly, and `[d4] row 1 q top3: 496:0.9986 759:0.0005
614:0.0003` says the shape of it — **a point mass on one token, in every row.**

The block is `[anchor, mask x 15]`: rows 1..15 are the *same* input at fifteen
different positions, so every difference between them must come from RoPE and
the attention the positions induce. A point mass in every row means that
difference is not being made. **The drafter's masked block rows are not carrying
distinct positions**, and until they do, alpha stays ~0 at any round cost.

### Where that leaves the 30 t/s question

| | before this session | after |
|---|---|---|
| drafter tile permutation | unscored, hardcoded row-major | **scored and fixed: colmajor** |
| drafter logit rms | 12.59 (target's is 6.278) | **4.285** |
| softcap saturation | 68 logits pinned | **0 of 4 194 304** |
| mean diagonal alpha | 0.0000 | 0.0565 |
| best alpha | 0.0000 | 0.8479, at one column |

Two of the three things wrong with the drafter were fixed and measured in that
session snapshot. The then-standing third suspect was "position distinctness
inside the block". **Superseded 2026-10-13:** `dflash_block_rows` now shows the
masked rows stay distinct through all five layers; the unresolved issue moved to
head/ranking diagnostics.

**And alpha is still not sufficient on its own.** The probe's own decision rule
(`alpha < 0.919 -> slower than plain decode, DO NOT SHIP`) plus the round
arithmetic (31 drafts at ~837 ms -> 30 t/s needs alpha ~0.96-0.98) means the
feature ships only if the position fix produces a drafter *and* the round costs
come down. The cheap round wins (drafter off `gemv_tiled`, batched head) are
still unbuilt and still worth ~250 ms, but they lower the bar from 0.98 to
roughly 0.96, not to 0.9.

## 2026-10-13 — fresh rerun for spec sync (no code changes)

Re-ran three diagnostics on this branch with real checkpoints:

```bash
M=~/.kaplumba/models/Gxma-repacked
KAPLUMBA_TEST_MODEL=$M KAPLUMBA_DFLASH_DIR=~/.kaplumba/models/Drafter \
  KAPLUMBA_FORCE_LOAD=1 scripts/gpuq try -- \
  cargo test -p kaplumba-engine --release --test dflash_alpha_probe -- --ignored --nocapture --test-threads=1

KAPLUMBA_TEST_MODEL=$M KAPLUMBA_DFLASH_DIR=~/.kaplumba/models/Drafter \
  KAPLUMBA_FORCE_LOAD=1 scripts/gpuq try -- \
  cargo test -p kaplumba-engine --release --test dflash_block_rows -- --ignored --nocapture --test-threads=1

KAPLUMBA_TEST_MODEL=$M KAPLUMBA_DFLASH_DIR=~/.kaplumba/models/Drafter \
  KAPLUMBA_FORCE_LOAD=1 scripts/gpuq try -- \
  cargo test -p kaplumba-engine --release --test dflash_head_oracle -- --ignored --nocapture --test-threads=1
```

Results:

- `dflash_alpha_probe` reproduced the existing state exactly:
  - as-configured / config+1: mean α 0.0000
  - as-configured, colmajor: mean α 0.0565, best 0.8479 (row 1, t=5)
  - decision line unchanged: `alpha < 0.919 -> slower than plain decode, DO NOT SHIP`
- `dflash_block_rows`: still reports no post-RoPE collapse; block rows remain
  distinct through all 5 layers (`VERDICT: no stage after RoPE collapses the masked rows`).
- `dflash_head_oracle`: replay-vs-plain control still fails hard
  (`worst |diff| 40.5756..44.8307`), so replay-vs-draft numbers are not
  admissible for blame yet.

Net: the spec blocker chain stands — low α persists; block path is not the
collapse point; head oracle must be trusted before any `draft_logits` fix call.

## 2026-10-13 — documentation audit (A1)

Scope confirmed by user: update latest spec and flag stale older claims; no code
edits. Reviewed working tree at base `72d3d1a`, including existing uncommitted
spec rerun notes. Existing `crates/engine/src/sampler.rs` modification was not
edited or staged. This audit did not rerun GPU diagnostics or benchmark models.
Historical rerun results above remain historical observations.

### Findings and evidence

- `crates/engine/src/dflash.rs:974` selects argmax proposals; `:2444` calls it
  from `draft_block`; `:2333` computes p and q for sampled verification;
  `:2349` even accepts q_candidate<=0 unconditionally. The algorithm's sampled-q
  premise is violated by actual proposal creation. With p=(.6,.4), q=(.9,.1),
  argmax always proposes token 0, accepts it with probability 2/3, and otherwise
  residual sampling selects token 1. Emitted distribution is (2/3,1/3), not p.
  Sampling x~q instead gives (.6,.4) analytically. This demonstrates an
  algorithmic bug without editing or claiming test coverage of production code.
  z-lab `dflash/model.py:262-265` explicitly computes q then samples proposals;
  `:278-284` verifies against target probabilities. Reference path exists:
  `/private/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/zlab/dflash`.
- `crates/engine/tests/dflash_head_oracle.rs:274`: `.chunks(SLICE_COLUMNS)`
  over full `[rows,262144]` output makes chunk r belong to vocabulary columns
  r*512..(r+1)*512 of row 0 for r<512. The comparison expects columns 0..512
  of row r. All rows after row 0 are misindexed. `:309` divides host dot product
  by K, whereas `:353` does not. No justified consistent scaling contract is
  established. `:259` calls the same sliced buffer GEMM as `draft_logits`
  (`dflash.rs:2486`); this “plain engine” is not an independent target GEMV.
  Assertions at `:207` and `:242` only check shapes. Numerical failures are
  printed, so the cargo test can succeed while the control is red. Repair these
  before interpreting the 40.6–44.8 historical differences.
- `crates/kernels/src/linear.rs:461`: `perm_mode` already exists;
  `crates/engine/src/dflash.rs:1646`, `:1720`, `:2497` pass line/head modes.
  Historical commits: `9a4cf60` plumbing; `6622031` sweep work. Old T5/T6 status
  ignored these commits. Revised tasks distinguish implementation already done
  from default/reference/sabotage validation still required.
- `spec/dflash-correctness/implementation_notes.md:1219`: 156.71 ms is M=32
  fc+35 projections (139.94+16.774 ms), excluding attention/norm/head. The old
  round spec relabeled it a complete one-block cost including head, and reused
  it for B16. `:2037` calls 606 ms target verify layers, but this number's actual
  measurement boundary must be reconciled with complete-forward timing before
  adding head. No trustworthy complete round has been established by the table.
- `crates/engine/src/model.rs:4569`: `forward_verify` returns full head rows.
  `:5221` implements the head via sequential GEMVs because the existing GEMM
  route is not numerically equivalent. Historical M=2 difference 19.247559 at
  `spec/dflash-correctness/implementation_notes.md:1943`. The previous T11 was
  directing a switch to a path already known to fail this prerequisite.
- `crates/engine/src/dflash.rs:2385`: retract n−K, leaving correction unprocessed;
  its comment explicitly requires the caller to feed next to obtain subsequent
  logits. Old round budgets omitted this forward. At the illustrative old
  15.283 tok/s baseline it costs 65.432 ms; n31/1055ms ceiling becomes 28.560
  tok/s when added. This is arithmetic, not a fresh timing result.
- Block B includes anchor and B−1 masked proposals. Re-anchoring two B16 blocks
  gives 15+15=30 proposals; a single B32 gives 31. Earlier v1 B32 ceiling used
  33 tokens, although at most 32 newly emitted tokens are available.
- `crates/engine/src/model.rs:5705`: tap reads return current scratch prefix;
  a later forward overwrites it. `dflash.rs:2167` appends rows ending at the
  exclusive position argument. The round plan must preserve old-anchor taps,
  append accepted verify features once, and keep context ending before anchor.
  Rewinding rejected drafter features never appended would remove valid context.
- `crates/engine/src/gpu_cache.rs:421`: retract lowers cursor only;
  `dflash.rs:880`: rewind truncates surviving context after front eviction.
  Neither method itself restores overwritten/evicted old rows. Whether the
  complete attention path retains everything required after speculative ring
  wrap remains an explicit correctness gate, not a proven end-to-end failure.
- z-lab `dflash/model.py:37-42`: offset=1 indexes HF hidden-state list entries;
  engine `model.rs:5005` captures post-layer output directly. Incrementing tap
  layer IDs just because HF uses offset=1 is not a justified semantic fix.
- Distinct post-RoPE hidden rows rule out identical-row collapse in the recorded
  run, but do not prove correct positions, attention, head, or checkpoint quality.
  Single-pair off-diagonal overlap .8479 is not sequential acceptance evidence.

### Validation performed in this audit

Read-only checks: `git status --short`, `git log -8 --oneline`, targeted `rg`
searches and source reads, then `git diff --check`. No model run, no project
code/test changes, no production coverage claim, and no sabotage of project
code. The required future sabotage cases remain unchecked in tasks.md.

Standalone Ruby calculation, via a shell heredoc without creating project files:

```ruby
def emitted(a, n)
  (0..n).sum { |j| a**j }
end

def threshold(n, cost, target)
  return nil if emitted(1.0, n) / cost < target
  lo, hi = 0.0, 1.0
  80.times do
    mid = (lo + hi) / 2
    emitted(mid, n) / cost < target ? lo = mid : hi = mid
  end
  hi
end

[[15, 0.898], [31, 1.055], [31, 0.935]].each do |n, cost|
  puts [n, cost, emitted(1.0, n) / cost,
        emitted(0.95, n) / cost, threshold(n, cost, 15.283),
        threshold(n, cost, 30)].inspect
end
```

The executed heredoc additionally asserted a=0
and a=1 boundaries, the three break-even roots within 1e−12, one-block
impossibility at 30 tok/s, two-block proposal count, correction-forward ceiling,
and the two-token proposal counterexample. All assertions passed.

Results: break-even roots .979133724663, .949988513730, .939526669817 for the
old 898/1055/935 ms assumptions; required acceptance for 30 tok/s: impossible,
.999289330242, .991329576290. These correct the arithmetic without validating
those assumed costs. For measured acceptance use accepted-prefix survival,
and measure overall tokens/time rather than averaging round rates.

### Resulting plan changes

Correct proposals and cache/tap contracts gate experimental wiring; repaired
oracle and independent checkpoint semantics gate head blame; correctly measured
B16/B32 gate speculative speed claims. Head batching is guarded by the existing
numerical failure. Greedy near-tie exemptions removed. Performance failure stays
unmet, rather than being accepted as proof of impossibility. No fifth kernel
family is authorized; no implementation task was completed by this review.

## 2026-10-13 — T1a implementation scope authorized
User explicitly steered from docs-only review to implement and test the sampled-proposal defect. Amended spec/plan/tasks before code edits. Scope: `dflash.rs` sampled candidate draws from and retains each drafter q; deterministic greedy remains; empty/non-finite proposal distribution fails; deterministic no-model tests. Product loop, cache ring rollback, head oracle, and performance gates remain open. Existing uncommitted `crates/engine/src/sampler.rs` belongs to user and must not be modified.

## 2026-10-13 — T1a sampled proposal fix (`2f35201`)

`crates/engine/src/dflash.rs` now samples each non-greedy proposal from the
masked row's `sampler_distribution`, retains each exact q map, and passes that
same q to acceptance/residual verification. Greedy mode continues to use the
existing argmax selections. Empty/zero-mass/non-finite distributions and
uniforms outside `[0,1)` now return a shaped error; zero probability entries are
skipped even for a zero uniform, and sampled candidate q mass must be positive.
The acceptance comparison is `uniform < p/q` (with unconditional accept when
p≥q). This fixes proposal creation only; there is still no product caller for
`draft_and_verify`, and cache rollback/taps/head-oracle gates remain open.

Added unit cases cover 10,000 deterministic quantiles of q≈(.75,.25), retaining
the exact q and skipping an anchor row deliberately favoring the opposite token;
empty logits, top_k=0, NaN/infinite/out-of-range uniforms, and zero-mass support.
Existing greedy draft row, round position/retract, residual and acceptance
quadrature checks were rerun.

Validation:

- `cargo test -p kaplumba-engine --lib dflash:: -- --nocapture` — **12 passed**,
  0 failed, 127 filtered out. No model load or GPU benchmark.
- Sabotage: replaced sampled draw in `sample_draft_tokens` with `argmax_row`;
  `cargo test -p kaplumba-engine --lib dflash::sampler_tests::sampled_drafts_follow_the_retained_q_and_skip_the_anchor_row -- --nocapture`
  failed as expected: `one q draw per masked row`, left 0, right 10000.
  Restored sampling and reran the 12-test suite successfully.
- `git diff --check` passed.
- `cargo fmt --all -- --check` reports extensive pre-existing formatting
  differences across unrelated workspace files. `rustfmt --edition 2024 --check`
  also reports pre-existing formatting differences elsewhere in `dflash.rs`;
  changed implementation/test hunks were inspected and aligned, without
  reformatting unrelated code.

No acceptance-rate or throughput conclusion follows from these unit tests; they
prove the sampled proposal helper uses q and that replacing it with argmax is
detected. Existing uncommitted `crates/engine/src/sampler.rs` was not modified.

## 2026-10-13 — renewed implementation authorization and validation

User expanded authorization from T1a to the full experimental DFlash path and
explicitly requested a 30 tok/s result. Updated `spec.md`, `plan.md`, and
`tasks.md` to reflect this. Re-ran
`cargo test -p kaplumba-engine --lib dflash:: -- --nocapture`: 12 passed.

No real-model run was safe at this point: `scripts/gpuq status` reported 1.3 GiB
available, 7.8 GiB in use, and no swap. More critically, inspection confirms
`GpuKvCache::retract` in `crates/engine/src/gpu_cache.rs` only moves
`next_position`; it does not restore K/V or position tags overwritten by
speculative writes to a sliding ring. `draft_and_verify` invokes this cursor
retraction after verification. A generation loop cannot be considered correct
until T1b implements and tests the rollback transaction. No 30 tok/s result is
claimed.

## 2026-10-13 — forced-load alpha probe rerun

Command, after `scripts/gpuq status` reported no owner, queue 0, 3.4 GiB
available and 0 swap:

```bash
KAPLUMBA_TEST_MODEL="$HOME/.kaplumba/models/Gxma-repacked" \
KAPLUMBA_FORCE_LOAD=1 scripts/gpuq try -- \
  cargo test -p kaplumba-engine --release --test dflash_alpha_probe -- \
  --ignored --nocapture
```

Exit 0, 1/1 passed, probe time 38.67 s. Results on its 78-token prompt,
temperature 1/top-k 64, block size 16:

- As-configured permutation 0 and config+1 taps: mean diagonal overlap 0.0000.
- Colmajor permutation 1: mean diagonal overlap 0.0565; best isolated pair
  0.8479 at draft row 1 vs target position `total+5`.
- Colmajor generated only 3 distinct argmax draft tokens across 15 proposals;
  logits changed across contexts but drafts were identical for the sampled
  comparison.

These are the probe's distribution-overlap statistics, not sequential accepted
prefix survival from a product round. Its legacy `DECISION RULE` still prints
fixed 0.919/0.977 thresholds derived from old budgets; those are not valid
acceptance criteria per this spec. The head replay reports 100% relative error
because its target final-norm row is all zero (`max|v|=0`), consistent with the
known broken oracle; do not use it to assign blame to the target head. No full
round timing or end-to-end throughput was emitted. After process exit,
`scripts/gpuq status` showed owner none, queue 0, 1.8 GiB available, 0 swap.

## 2026-10-13 — repair strategy and alternative proposal sources

Follow-up investigation for the user's request for options. Proposed experiments,
not completed implementation tasks or speed claims:

1. Measure target-only verification at M=2/4/8/16 against sequential target
   forwards on identical token sequences. Use known target continuations as
   ideal proposals to establish an upper bound before spending more time on a
   drafter. Measure full head, target body, and wall time; verify logits/output
   equality and restore caches. A synthetic ideal-proposal rate is a ceiling,
   not real speculative generation throughput.
2. Repair T0c's independent head oracle; isolate the first divergent drafter
   stage using identical saved inputs. Verify original checkpoint/target/tokenizer
   compatibility, EXL3 reconstruction and layout, taps, fc, normalization, RoPE,
   attention, and head. Comparing two engine paths sharing the same bug is not
   independent evidence. Quantization or feature mismatch remains a hypothesis.
3. Consider unprocessed-anchor rounds after amending C1/C2 and their tests:
   z-lab `dflash/model.py:230-301` verifies `[anchor, proposals]` with the target
   cache ending before the anchor, then crops and carries accepted target taps.
   This avoids the current separate correction forward. It does not repair q.
4. Alternative: prompt/history n-gram lookup as deterministic proposals,
   validated against exact target sampling (q is a point mass). Low draft cost
   and no second model; expected benefit depends on repetition/copying. It still
   needs a fast verifier and transactional target caches.
5. Alternative: RedHatAI/gemma-4-31B-it-speculator.eagle3. Its config declares
   one transformer layer, draft vocab 32000 vs target vocab 262144, taps
   [2,30,57], and a distinct untied draft head. Its model card reports k=3
   acceptance lengths 3.10 on HumanEval and 2.22 on QA, from vLLM, with training
   on non-reasoning responses. These are external results, not M4/EXL3 evidence.
   It requires a new loader/state contract and measured memory/acceptance; a
   replacement drafter cannot solve slow target verification by itself.

For 30 tok/s, a round averaging 4/8/12 newly emitted tokens must cost at most
133.33/266.67/400 ms respectively, including all work. Select proposal length
from measured accepted-prefix survival and complete cost, rather than forcing
B32 or assuming near-perfect acceptance. Kernel scope restrictions remain in
force; a new kernel family requires an explicit scope amendment.

Primary sources inspected:
- Existing local z-lab `dflash/model.py:157-170`, `:230-301`, `:340-464`,
  `:550-600`: mask, unprocessed-anchor loop, Qwen3 drafter blocks and head.
- https://huggingface.co/z-lab/gemma-4-31B-it-DFlash : public B300 greedy
  acceptance lengths 4.23 (MT-Bench), 8.00 (HumanEval), 8.59 (Math500).
- https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.eagle3 and its
  `raw/main/config.json`: alternative model architecture and reported lengths.
- https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md :
  prompt/history lookup and EAGLE-3 support.

The AGENTS.md claim that `reference/exllamav3` is present does not match this
checkout: its referenced DFlash files are absent. Do not cite that path as
inspected evidence until the reference is restored.

## 2026-10-13 — draft only: verifier gate and conditional fallback plan

Latest user instruction: "Just draft the spec no impl yet." This supersedes
earlier implementation authorization for remaining tasks. Updated existing
`spec.md`, `plan.md` and `tasks.md`; no new spec folder, code, harness, model
load or checkpoint download. Existing `sampler.rs` modification remains outside
this documentation change.

Draft adds V0a/V0b (sequential-vs-batched verification and optimistic ceiling at
M=2/4/8/16), T1c (CLI sampling policy), T16 (unprocessed-anchor rounds), T15
(length selection from complete measured cost), and conditional F1a/F1b
(lookup/EAGLE-3 evaluation). Native B16 precedes speculative B32/two-block work.
The round entry is explicitly changed from the existing processed-anchor API:
target/drafter caches end before emitted anchor P; verify rows P..P+n; after K
acceptances commit rows through P+K, append those features, emit bonus at P+K+1
and leave it unprocessed. Tests must detect duplicate feeds, row shifts, tap
overwrites, ring restoration and missing old keys during early verify queries.

No diagnosis of "bad target" or "bad drafter" is established. The .0565 probe
overlap is not production accepted-prefix survival. Independent checkpoint and
stage comparisons must distinguish implementation/conversion errors, feature
mismatch and actual drafter quality. Numeric tolerances and alternate checkpoint
selection remain explicit unresolved evidence/design decisions.

Documentation validation: `git diff --check`; reviewed requirement/task mapping,
historical-vs-current authorization, round positions/counting, native block size,
sampling defaults and conditional alternative scope. No executable tests run
for this documentation draft; no implementation completion claim.

A2 documentation committed as `697dcc2`. Follow-up records that hash in tasks.

## 2026-10-13 — A3 exllamav3 review and authorized diagnostic tests

User requested exllama inspection, then "test it too". Implementation remains
held per the earlier draft-only instruction. Read the existing reference at
`3a513ac9f91aae777f317ca2341b8e445b62c64e`; it was present all along. Correct
Python-package paths contain `reference/exllamav3/exllamav3/`, so the earlier
absence claim was an incorrect path lookup.

Full findings, source `path:line` references, commands and raw text evidence are
in `exllama_review.md` and `evidence/exllama-2026-10-13/`. Temporary Rust diagnostics
were built outside the project, with a separate target directory. No product
code or reference checkout was changed. Diagnostic source is preserved as fenced
documentation text, and temporary sources/binaries/build targets were cleaned up.

Concrete results:

- Buffer GEMM fp16/fp32 wrappers validate `packed_offset` but bind weights at
  zero (`linear.rs:1459`, `:1569`, helper `:4669`). Real head offset 5321075520,
  size 1056964608 bytes. On 4 nonzero rows, current sliced buffer head disagrees
  with production decode GEMV on every argmax; capped max error 2.829250.
  Initial identical-input assertion failed with a named message. Copying only
  the same head bytes to offset 0 restores all argmaxes with capped max error
  .000488. This is an isolated addressing defect, not a checkpoint-quality test.
- Full 16-way layout/tap/pre-scalar/embedding sweep, original head: reproduces
  colmajor/native-tap mean .056493. With only temporary correct-head addressing,
  row-major/native/post-scalar/raw yields mean .116777, first overlap .553835;
  colmajor/native yields .000003 mean. Previous colmajor inference and .8479
  off-diagonal evidence were contaminated by the wrong head. All cases still
  reject the first greedy proposal on this prompt. No general quality claim.
- Existing forced-load `dflash_verify_head` suite: 1 passed, 1 failed; exit 101,
  3.10 s. M=2 full-forward max error 19.247559 at row 1/column 929.
- Temporary all-layer target comparison at identical `[1,3824]` input: layer 0
  max error 10.048828, relative RMS .210203; last layer relative RMS .636517.
  Divergence is already in the target body; exact operation and correct reference
  behavior still need isolation. This is separate from the DFlash head offset.
- Direct ring primitive: overwrite positions 4/5 in capacity 4, retract 2;
  key at position 1 (required by next query at 4) remains 1001 instead of 101.
  Named restoration assertion fails. The same temporary suite's independent
  packed-stream/fragment-scatter fixture matches the integer reader at 3/4/6 bpw.
  This is not full round coverage. `unpack_integer` already scatters, so applying
  the reference tensor-core permutation again would be a diagnostic mistake.
- Exllamav3 retains extra ring history (two 256-position pages by default),
  updates accepted draft context from target features, runs native B16 before
  prefix truncation, and keeps tensors/KV on GPU. Our host-vector wrappers allocate,
  transfer and wait per operation. A staging/wait speedup was not measured.

Proposed repairs T17/T18 are documented and held. T19 is conditional GPU ownership
work after correctness. Updated current spec/plan/tasks to remove stale head-only
blame and invalidated overlap-based layout assumptions. The sampled helper fix
T1a remains historical implemented work; no new code fix was made here.

Every model command had `gpuq status` beforehand, used `gpuq try`, and retained
the user's force-load authorization. Runs were serial and all model processes
exited. Final observed status: no holder/queue, 1.4 GiB available, 0 swap.
No 30 tok/s, interleaved performance median, full-round equivalence, or memory
acceptance is established. `git diff --check` validates documentation formatting.

A3 reference review, test evidence and proposed repair tasks committed as
`59a6f4b`; follow-up records the commit in the task status. Working tree retains
only the pre-existing `crates/engine/src/sampler.rs` modification.

## 2026-10-13 implementation resumed

User renewed implementation/testing authorization and explicitly retained the
30 tok/s goal. Spec/plan/tasks authorization status recorded in `47c8db5`.

T17 completed in `fda508d`:

- `encode_groups` now binds its first buffer at the caller's packed offset for
  fp16 and fp32 buffer GEMM. Slice-backed prefill still passes offset zero.
- Added `gemm_chunk_buffer_f32_sliced` to provide fp32 the same exact region-size
  contract as fp16. Existing non-sliced entry retains its buffer-end behavior.
- `validate_buffer_shape` still checks the exact EXL3 tensor byte count and now
  also rejects overflow or a region ending past the Metal buffer.
- `linear_metal::buffer_gemm_honors_packed_offset_for_fp16_and_fp32`: four
  activation rows; 4 and 6 bpw; canary prefix/trailing bytes; fp16/fp32; exact
  equality against offset-zero reference; short length and out-of-range rejection.
  Focused run passed 1/1. Full `linear_metal` run passed 8/8 before the final
  boundary assertions; focused run passed after them.
- Sabotage replaced the offset with zero. Focused test failed, exit 101, at the
  named fp16 rate-4 prefix-canary assertion. Restored implementation and reran
  successfully. Existing `buffer_perm_mode` passed 3/3.
- `cargo fmt --all -- --check` reports extensive formatting differences in
  unrelated baseline files, including CLI, kernels and integration tests; did
  not apply whole-workspace formatting.

Next: T18 isolates target batch/decode divergence. The uncommitted sampler change
predates this work and was not staged or modified. 30 tok/s has not been measured.

## T18 — target batch/decode repair (2026-10-13)

Forced-load target: `~/.kaplumba/models/Gxma-repacked`; every run used
`scripts/gpuq status` then `KAPLUMBA_FORCE_LOAD=1 scripts/gpuq try -- ...`.

- M=1 QKV fused kernels bind one input-sign vector and apply it to Q/K/V. The
  checkpoint's vectors differ, so M=1 K/V was wrong. The fused gate/up MLP had
  the same assumption for gate/up. Fast fusion now requires exact vector
  equality; fallback GEMVs each apply their own input transform.
- Cross-layer finalize→next input norm was not safe on this checkpoint. Baseline
  layer-1 cached K max difference was 0.3287964 and V was 9.848267; full logits
  differed by 12.320312. Disabling this fusion reduced layer-1 cache differences
  to K 0.0009766 / V 0.0078125. It is now opt-in with
  `KAPLUMBA_USE_CROSS_LAYER=1`; `KAPLUMBA_NO_CROSS_LAYER=1` still disables it.
- Three token sequences × M=1/2/4/8/16 compare whole-model batched logits to
  sequential target forwards. Every row's greedy token matches. Maximum logit
  drift was 1.6464844 (prompt 0, M=16, row 14, non-argmax column) on the first
  run and 0.2695313 on the final rerun; prompt 1/2 maxima on the first run were
  0.1445313/0.2617188. This run-to-run discrepancy is unexplained and remains
  a repeatability risk. The 2.0 bound is a diagnostic ceiling to catch the
  pre-fix >12 failure; greedy identity is the correctness gate.
  This does **not** establish distribution-exact sampling. Keep speculative
  verification greedy-only until sampled target distributions are proven.
- All-layer M=2 cache/tap test with cross-layer fusion default-off: maximum K
  diff 0.0030518 (layer 26), maximum V diff 0.0507813 (layer 45), maximum
  post-layer hidden diff 0.375 (final rerun, layer 42). Guards: K <
  0.01, V < 0.1, hidden < 0.5. These tolerances catch the observed catastrophic
  path, not exact floating-point equality.
- Coverage sabotage: replacing projection-sign comparison with unconditional
  `true` failed `shared_projection_transforms_require_identical_signs` at the
  mismatch assertion. Removing the cross-layer opt-in gate failed the target
  layer-state test with layer-1 K 0.3287964 / V 9.848267. Both sabotages were
  restored and the production code rebuilt.
- `linear_mrhs::mrhs_rows_match_the_actual_decode_gemv_at_model_width` compares
  M>1 against the production `encode_gemv` path, including perm modes 0/1; it
  avoids the previous invalid control that compared two batch routes.

T18 closes only target greedy row equivalence. T0c/V0a (head/oracle controls,
positions, ring boundaries), V0b (ideal ceiling), stochastic equivalence,
rollback, and the product round loop remain open. No throughput run was made;
30 tok/s remains unmet. Implementation commit: `6dade81`.

## T1b.1 — sliding append history (2026-10-13)

- Root cause: Gxma sliding cache capacity equalled its 1,024-token attention
  window. `forward_verify` writes every batch row before attention. With prefix
  length 1,023, row 1 at absolute position 1,024 overwrote ring slot 0 before
  row 0 at 1,023 had read key position 0. Restoring the old row after verify
  would not repair the already-computed logits.
- `DecoderLayer::new` now allocates sliding capacity
  `sliding_window + prefill_chunk()` (2,048 at the default chunk), while the
  attention mask remains the configured 1,024. KV reserve accounting and its
  independent layout formula include the physical slack. Gxma reserve changes
  from 3,524,730,880 to 4,363,796,480 bytes, +839,065,600 bytes (0.781 GiB).
- Added ignored forced-load test
  `batched_verify_preserves_keys_at_sliding_window_boundary` in
  `crates/engine/tests/dflash_verify_head.rs`. It pre-fills 1,023 tokens, writes
  positions 1,023/1,024 together, asserts position 0 remains in the first
  sliding-layer cache, then compares greedy outputs/logits with sequential
  decode. Fixed-path maximum logit delta: 0.136719; greedy rows match; cursor
  reaches 1,025.
- Coverage sabotage: changed physical capacity back to `sliding_window`. The
  new test failed at the named assertion: `batched row at position 1024 evicted
  key position 0 before row 0 attended`. Restored capacity and reran.
- Tests: `cargo test -p kaplumba-engine --lib` = 138 passed, 2 ignored;
  checkpoint-backed reserve test = 1 passed; all four ignored
  `dflash_verify_head` tests, forced-load and serial = 4 passed. Direct timed
  boundary test: 26.06 s; `/usr/bin/time -l` reported max RSS 10,124,951,552 B,
  peak footprint 2,745,147,456 B, and 0 swaps. The max-RSS and footprint fields
  are distinct OS metrics; do not substitute RSS for footprint.
- The default checkpoint directory used by the reserve test was stale
  (`Gxma/Gxma` did not exist); changed its fallback to the present
  `Gxma-repacked` checkpoint. First test invocation with the stale path failed
  before model load; reran successfully with the actual local checkpoint.
- Rollback after rejection/error, zero/partial/full acceptance, and stale-row
  exclusion after cursor retract remain T1b.2. This subtask does not complete
  transaction semantics. No 30 tok/s throughput measurement was made.
