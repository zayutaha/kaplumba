# B3 — the host top-k selection: `select_nth_unstable_by` -> two-pass counting select

**Verdict: shipped.** `top_k_order` (`crates/engine/src/sampler.rs`) no longer builds a
2 MiB index vector and no longer gathers two `f32`s per comparison. It counts the rank-`k`
threshold over a monotone `u32` key instead, and returns a `Vec<usize>` that is **byte-identical
to the selection it replaced** on every input tested — tie-dense rows, adversarial orderings,
signed zeros, `f32::MIN`/`MAX`/`NEG_INFINITY`, all-equal full-width rows, `k = 1`, `k >= len`,
and fp16-derived distributions.

Measured, from ONE binary, 262,144-wide fp16-derived row, `top_k = 64`, median of 41
**interleaved** A/B samples (`--release`):

| row | `select_nth_unstable_by` (was) | counting select (now) | |
|---|---:|---:|---:|
| uniform (random) | 2.04 ms [1.99..2.35] | **0.27 ms** [0.23..0.32] | 7.6x |
| tailed (realistic) | 1.82 ms [1.69..2.12] | **0.25 ms** [0.22..0.34] | 7.2x |
| gaussian head | 1.12 ms [1.05..1.47] | **0.26 ms** [0.23..0.34] | 4.4x |
| ascending (adversarial) | 0.23 ms | 0.66 ms | **0.3x** |
| descending (adversarial) | 0.30 ms | 0.66 ms | **0.4x** |
| all equal | 0.40 ms | 0.40 ms | 1.0x |
| two values | 0.39 ms | 0.40 ms | 1.0x |
| **worst single sample, any row** | **2.35 ms** | **0.91 ms** | **2.6x** |
| whole `sample()`, tailed | 1.89 ms | **0.33 ms** | 5.7x |

Peak allocation per call: **2,097,152 B -> 263,160 B**, and unchanged when the row is doubled.

The ratio is the claim, not the absolute (see §6).

---

## 1. Choice: counting (D) over a bounded heap (A)

B1 measured both candidates and recommended A ("11-14x faster on realistic rows"). **This
change ships D instead**, on the tail:

| selection | tailed | uniform | ascending | worst |
|---|---:|---:|---:|---:|
| A — f32 bounded heap | 0.13 | 0.14 | **4.95** | **4.95** |
| D — counting select (B1's variant) | 0.32 | 0.32 | 0.56 | 0.61 |
| shipped partition | 1.82 | 2.04 | 0.23 | 2.35 |
| **this (two-pass counting, 16/16)** | **0.25** | **0.27** | **0.66** | **0.91** |

A is 11-14x faster than the partition on realistic rows and **2.1x slower than the
partition's own worst case** on an ascending row. A sift-down runs `log k` times per late
arrival, so A's cost is a function of *where in the row* the best tokens sit. That is not
only an adversarial input: logits are indexed by token id, so a fine-tune whose strongest
continuations are high-id tokens puts the top-64 at the end of the row and pays 4.95 ms
inside a 69.6 ms step. Trading a possible 2.1x production regression for 0.12 ms on the
typical row is the wrong side of the ledger for a decode loop.

D's worst case is bounded by construction, and this implementation improves on B1's D: B1's
candidate collected *every* candidate above the cut and sorted them, which on a degenerate row
meant re-sorting the row. Here the candidate set is `top_k + (bucket width <= 512)` indices and
nothing wider, which is why `all_equal` and `two_values` cost the same as the partition rather
than more.

**One honest regression:** on **presorted** rows the partition wins, because Introselect is a
single pass on presorted data (0.23 ms vs 0.66 ms ascending). A logit row is not presorted. The
spread between the two on that input is 2.9x, against the heap's 36x (0.13 -> 4.95 ms).

## 2. The algorithm

```text
order_key(f32) -> u32      monotone in value; -0.0 folded onto +0.0
pass 1  histogram of key >> 16                      (65,536 u32 = 256 KiB)
        descend from digit 65535 until the running count reaches `need`
        -> `above` winners in strictly higher buckets, `within` in the chosen bucket
        if within <= 512: stop (this is the usual case -- ONE counting pass)
pass 2  histogram of the low 16 bits, over that bucket alone -> the EXACT key (16+16 = 32)
gather  one pass: keys >= bucket_end -> `order`; keys in the bucket -> `bucket`
        exact case: take the FIRST `need` of `bucket` (all tied, so the index tie-break
                    decides, and the Vec stays `need` long)
        wide case: sort `bucket` with the shipped comparator, truncate to `need`
        sort `order` with the shipped comparator, concatenate
```

Invariants the implementation depends on, each of which is a trap that was closed:

1. **`order_key` is order-preserving and tie-compatible with `partial_cmp`.** Flipping every
   bit of a negative float and only the sign bit of a positive one turns sign-magnitude into
   two's-complement order. **`-0.0` must be folded onto `+0.0`** (`value == 0.0` is true for
   both), because `partial_cmp` calls them equal and the shipped comparator then breaks the tie
   by INDEX. Without the fold, any row whose `top_k` boundary falls inside the zeros returns a
   different SET and `sample()` returns a different token for the same `--seed`. Verified by
   sabotage S1 below; a whole row of `-0.0` selected index 1 instead of 0.
2. **`bucket = key >> 16` is monotone**, so every key in a higher bucket beats every key in a
   lower one. That is what makes "count down to `need`" a correct rank-`k` search. Verified by
   the exhaustive f16 key sweep and by sabotage S5 (dropping the mask's `| 0x8000_0000`).
3. **`skipped <= need` at the walk's exit**, so `need -= skipped` cannot underflow.
4. **`u32` histogram counts.** `debug_assert!(logits.len() < u32::MAX)`.
5. **`prefix << shift` must include the digit just chosen.** The first draft omitted the
   assignment and the pre-existing `top_k_selection_matches_a_full_sort` gate failed on
   `len=2, top_k=1` immediately. This is the one bug this change wrote itself.
6. **`u64` for the gather bounds.** `end` is `low + (1 << shift)` and reaches 2^32 for the top
   bucket; a `u32` would wrap and a `>> 32` would panic in debug.

## 3. Gates shipped with the change (not after)

`crates/engine/tests/probe_topk_equality.rs` (10 tests) — the oracle is the **deleted
`select_nth_unstable_by`, verbatim**, and a verbatim `Sampler::sample` that uses it, so the
`--seed` claim is tested end to end and not only at the vector:

| gate | what it would miss |
|---|---|
| `counting_select_matches_the_shipped_select_on_full_width_rows` | 36 (row x k) combos at 262,144 |
| `tie_dense_rows_match_the_shipped_select` | >300 (width x tie density x k) combos |
| `boundary_top_k_values_match` | `k = 0`, `k = 1`, `k >= len` |
| `signed_zeros_straddling_the_boundary_match` | the `-0.0` fold, on a boundary that actually straddles |
| `extreme_f32_values_match` | `MIN`/`MAX`/`+/-inf`/subnormals, scattered and alone at full width |
| `random_f32_rows_match_the_shipped_select` | 3,000 random rows over the whole `f32` bit domain |
| `seeded_sampling_produces_the_identical_token_sequence` | the user-visible `--seed` claim, 4 seeds x 24 rows |
| `seeded_sampling_matches_across_the_config_grid` | 48 (T, top_p, top_k) combos x 6 steps |
| `greedy_is_still_argmax_and_never_touches_the_selection` | that `--temperature 0` was not touched |
| `probe_top_k_selection_timing` | worst-case ordering (asserts the counting worst < the partition worst) |

Plus, in `sampler.rs` itself: `order_key_is_order_preserving_over_every_fp16_pattern`
(**exhaustive** over all 65,536 fp16 patterns — the domain the logit row actually has, since
`generate.rs:326-330` widens fp16 to `f32` exactly),
`order_key_agrees_with_partial_cmp_on_the_f32_specials` (all 22x22 pairs, `Equal` demands
`Equal`), `order_key_is_monotone_across_a_broad_f32_sweep` (2^20 bit patterns, for the 13
mantissa bits fp16 never exercises).

Plus `crates/engine/tests/probe_topk_alloc.rs`: a `#[global_allocator]` budget, in its own
binary so no other test can allocate inside the measurement window. Peak must stay under 4 MiB
and must not grow with the row width.

## 4. Sabotage log (AGENTS.md 3.4) — every gate observed failing, then reverted

Each break was applied to `sampler.rs`, the whole gate set was run in `--release`, and the
observed failures recorded. `sampler.rs` was restored byte-identically afterwards (md5 checked
by the driver).

| # | break | gates that failed | observed failure |
|---|---|---|---|
| S1 | drop the `-0.0` fold in `order_key` | `order_key_agrees_with_partial_cmp_on_the_f32_specials`, `signed_zeros_straddling_the_boundary_match` | `partial_cmp says -0e0 == 0e0, so their keys must be equal` / keys `2147483647` vs `2147483648`; row returned `[1]`, oracle `[0]` |
| S2 | invert the index tie-break when ranking the bucket | 8 gates incl. both seeded-sampling gates | `T=2 top_p=0.95 top_k=64 step=5: left 534, right 1926` — a different **token** |
| S3 | one winner too few (`need -= skipped - 1`) | 8 gates incl. both seeded-sampling gates | `returned 0 candidates ... returned 1`; then `index out of bounds` inside `sample` |
| S4 | walk the histogram **up** from digit 0 (count what is below the threshold) | 10 gates incl. the timing gate | `returned 4096 candidates at k=1 (len=4096) where the shipped selector returned 1` |
| S5 | forget `\| 0x8000_0000` in the key's sign transform | 9 gates incl. all three key tests | `order_key inverted: -inf keyed 0x007fffff but -0e0 keyed 0x00000000` |
| S6 | make the gather's lower bound exclusive (`key > low`) | 7 gates | `returned 0 candidates at k=1` (the bucket's own minimum can never win) |
| S7 | **revert the function to `select_nth_unstable_by`** | **only** `probe_topk_alloc` — all 11 lib tests and all 10 equality gates PASS | `peak 2097152 B`, and `doubling the vocabulary raised the peak from 2097152 B to 4194304 B` |

**S7 is why the allocation gate exists.** The realistic regression for this change is a revert,
and a revert is *correct*, so every equality gate stays green while the 2 MiB per-token
allocation comes back. A gate that can only fail on wrong answers cannot catch it.

**One sabotage did not fire, and that turned out to be right.** Changing the descend walk's
`<` to `<=` (`skipped + within <= need`) passes every gate. On analysis it is output-equivalent:
the loop invariant still gives `skipped <= need`, and the extra bucket it admits is absorbed by
the final `sort_by` + `truncate`. It costs a wider candidate sort and nothing else. Recorded
because "we tried to break it and nothing happened" is only informative once you know why.

## 5. Two pre-existing tripwires this change trips, and the fix for each

Both live in `crates/engine/tests/perf_probe_host_ops.rs`, which this task is not permitted to
touch. Left for whoever owns that file:

1. **`sampling_a_full_width_row_costs_milliseconds_per_token` now FAILS**, by design. It asserts
   `sample_ms > 0.5` with the comment "even the *current*, already-optimised sampler cannot be
   under a millisecond on this width". It now measures **0.445 ms**, and the failure message is
   the right one: *"the report's sampling-penalty figure would need re-measuring"*. Fix: drop the
   `> 0.5` floor (or lower it) and update the comment, which is now false.
2. **`sampler_allocates_per_call` now passes for the wrong reason.** It asserts `top_k_order`'s
   body `contains("(0..logits.len()).collect::<Vec<_>>()")`. The counting path has no such line,
   but the `keep >= logits.len()` branch must materialise every index — the function's contract
   is to return them all — so the check is now satisfied by a line that is unreachable from the
   shipped `top_k = 64` path. This is exactly the failure mode the test's own doc comment warns
   about. Fix: scope the assertion to the main path, or replace it with `probe_topk_alloc.rs`
   (which is what actually pins the budget, and which S7 proves can fail).

`no_device_side_sampler_exists` (`perf_probe_concurrency_dependency_graph.rs`) and
`the_device_token_handoff_exists_and_the_row_follows_the_pick` were already failing before this
change: they trip on the other agent's in-flight `elements.metal` / `model.rs` work, which this
change does not touch.

## 6. Method, and what would falsify this

- Probes: `probe_topk_equality.rs`, `probe_topk_alloc.rs`. Both CPU-only: no checkpoint, no
  Metal dispatch, no GPU lock taken, so nothing here disturbs another agent's benchmark.
- **No in-situ decode measurement was taken, deliberately.** A real decode run needs the ~12 GiB
  model resident, and AGENTS.md 10.2 records that two concurrent instances have already destroyed
  benchmark numbers on this box. With another agent's kernels in the dirty tree, spending the
  GPU here risked invalidating *their* measurement. The host-side work `sample()` does is
  identical in-situ and out, and the B1 caveat stands: its warm-loop figure read 1.93 ms where
  in-situ was ~2.87 ms, so **apply the ratio (5.7-7.2x on realistic rows), not the absolute**
  (0.25 ms). If in-situ `sample()` was ~2.87 ms, 5.7x predicts ~0.5 ms and the 7.3 ms
  temperature-1.0-vs-0 gap should shrink by roughly 2.4 ms, i.e. 14.4 -> ~17.8 tok/s equivalent.
  That last figure is arithmetic, not a measurement, and should be re-measured before it is
  believed.
- `--release` throughout. `cargo test` at opt-level 0 is 7-8x slower (B1) and would have shown
  neither implementation's real cost.
- Median, min **and** max of 41 interleaved samples, order flipped every iteration; three
  consecutive runs agreed to within 4% on every cell.
- **What would falsify it:** an in-situ measurement showing `sample()` was not ~1.9 ms on the
  real decode path, which would mean the selection was never the bottleneck and the win shrinks
  proportionally. The gates would not be affected — they are about equality, not speed.

## 7. Not done, on purpose

- **The fp16 widen is still there.** B1's D also absorbed `generate.rs:326-330` (0.065 ms) by
  taking `&[u16]`; that file was out of scope here, so the counting select reads the widened
  `f32` row. A future caller that hands it `&[u16]` would save the widen *and* a 1 MiB
  allocation per token.
- **`RANK_DIRECT = 512` is a guess**, not a tuned constant. It trades a wider bucket sort for a
  second counting pass. On fp16 rows a bucket holds ~16 distinct values, so `within > 512` is
  reachable with a peaked row and the second pass fires (measured at `top_k >= 4096`: 0.44 ms).
  Sweeping it is cheap and was not done.
- **The same-bucket serialisation that costs 0.66 ms on a smooth row** (consecutive increments to
  one histogram slot serialise on store-to-load forwarding) is a known, unexploited 2.5x on
  adversarial input. Two interleaved histograms by index parity would break the dependency
  chains at the cost of a second 256 KiB table; not attempted, because it only helps inputs a
  model does not produce and it taxes the path that does.

## 8. Reproduce

```bash
cargo test --release -p kaplumba-engine --lib sampler
cargo test --release -p kaplumba-engine --test probe_topk_equality
cargo test --release -p kaplumba-engine --test probe_topk_equality -- --nocapture probe_top_k_selection_timing
cargo test --release -p kaplumba-engine --test probe_topk_alloc -- --nocapture
```

Timings print; the assertions are the equality gates. The one timing assertion is the worst-case
ordering (`counting worst < partition worst`) — deliberately not a millisecond threshold, which
would be a flake generator and would have to be deleted to go green when the code got faster.