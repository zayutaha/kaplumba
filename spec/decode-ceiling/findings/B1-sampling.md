# B1 — the cheapest sufficient sampler: is the 2.87 ms host cost algorithmically reducible with **no GPU work at all**?

**Verdict: yes, and it is not close.** ~98.5% of `Sampler::sample` is one call to
`top_k_order`, and that call is replaceable by a bounded heap that is **bit-identical** to the
shipped selection and **11-14x faster** on realistic rows. A host-only change recovers
**~1.76 ms of the 7.3 ms gap** at essentially zero quality risk and zero GPU involvement.
The device work in flight is still worth its 5.5 ms (it is the only thing that removes the
barrier), but the host path must be fixed anyway because it remains the fallback for
`top_k > MAX_DEVICE_TOP_K`.

Two framing corrections, both measured:

- **The top-k is not a full sort.** It was one, and it was already fixed. `top_k_order`
  (`sampler.rs:69`) is `select_nth_unstable_by` — O(n) Introselect — and its doc comment
  records full sort 14.71 ms -> partition 1.89 ms. The question is not "sort vs heap"; it is
  *why the O(n) partition still costs 1.9 ms*.
- **The 64-element tail is not 1% of the cost, it is 0.0065%.** The softmax, top-p walk and
  inverse-CDF are 125 ns. Optimising them is pointless; the brief's Q3 is answered.

---

## 1. Decomposition of `Sampler::sample` (measured, this box, CPU only)

Probe: `crates/engine/tests/probe_sampling_phases.rs`. Real `sampler::top_k_order` and real
`Sampler::sample`; 262,144-wide row; `top_k=64, T=1.0, top_p=0.95`; median of 41 samples.

> **Run it with `--release`.** `cargo test` uses the `dev` profile (opt-level 0) and reports
> the same code at **7-8x** these numbers. Measured both ways; the table is the release build,
> matching the workspace's release profile (`Cargo.toml` has no `[profile.release]`, so cargo
> defaults apply: opt-level 3, no LTO, codegen-units 16).

`row_tailed` is the realistic case (peaked + heavy tail, values in a post-softcap range).
`row_uniform` is `examples/bench_sampler.rs`'s row, so the two are comparable.

| phase | code | tailed | uniform | share of `sample()` |
|---|---|---|---|---|
| fp16 -> `Vec<f32>` widen | `generate.rs:326-330` | 0.065 ms | 0.065 ms | *(outside `sample`)* |
| `is_finite` validity guard | `sampler.rs:178` | 0.070 ms | 0.068 ms | 3.6% |
| **top-k selection** | `sampler.rs:69-88` | **1.899 ms** | **2.043 ms** | **98.5%** |
| 64-element tail | `sampler.rs:187-215` | 0.000125 ms | 0.000125 ms | 0.0065% |
| **`Sampler::sample` total** | | **1.928 ms** | **2.062 ms** | 100% |

The columns do not sum to `sample()` because the phases are cache-coupled: inside `sample` the
`is_finite` pass warms the 1 MiB row for the selection, so the measured whole is *less* than the
sum of the parts measured cold. The tail figure is measured directly (1000 calls, divided),
not by subtraction.

**Why the O(n) partition costs 1.9 ms.** `top_k_order` does
`(0..logits.len()).collect()` — a **2 MiB `Vec<usize>`** (8 B x 262,144) — then partitions it
in place with a comparator that dereferences *both* indices into the 1 MiB `f32` row on every
comparison:

```rust
order.select_nth_unstable_by(keep, |left, right| {
    logits[*right].partial_cmp(&logits[*left])...   // two indirect gathers
});
```

So each of ~2n comparisons is two **random** accesses into a 1 MiB array. Introselect's own
bookkeeping is negligible; the cost is the gather. It is also why `select_nth` looks *fastest*
on the adversarial sorted rows (0.24 ms) — ordered input makes its partitioning trivially
predictable — and slowest on random rows, which is what a real logits row is.

**Magnitude caveat.** The brief quotes 2.87 ms for `sample()`; I measure 1.93-2.06 ms in a warm
tight loop from the verbatim committed source. The gap is most likely the working set: in situ
the 1 MiB row has just been DMA'd from the GPU and is cold, and `bench_sampler.rs` reuses one
resident row. If the in-situ cost really is ~2.9 ms, the savings below scale by the same
factor (~2.4 ms) — the *ratios* are what this probe establishes.

---

## 2. The top-p walk is free (Q3) — confirmed, and it is even smaller than "over 64"

`sampler.rs:110-128` (now `:186-215`) truncates to the top-`k` candidates *before* the top-p
walk. Confirmed by reading and by timing the whole tail: **0.125 us**.

Stronger than that: the walk does not iterate 64 times, it iterates until the cumulative
reaches `top_p`, and at `T=1.0, top_p=0.95` that is **1 to 7 candidates**:

```
top-p=0.95 keeps, over 200 realistic rows:
  1 -> 56 rows,  2 -> 76,  3 -> 39,  4 -> 17,  5 -> 9,  6 -> 1,  7 -> 2
```

The shipped default's effective support is ~2 tokens. Any plan that spends GPU work moving
softmax/top-p/inverse-CDF to the device is moving **125 ns** of work.

---

## 3. Candidates, timed and proved equivalent (Q1, Q5)

All three return the **exact `Vec<usize>` the shipped `top_k_order` returns**, verified over
42 (row x `top_k`) combinations including tie-dense rows, exact ties straddling the `top_k`
boundary, and a `+/-0.0` pair placed exactly on that boundary. Same signature, same input,
same output — so `--seed` reproducibility is preserved *by construction*: `sample` is a pure
function of (order, logits, config, RNG state), and an identical order forces an identical
token and an identical RNG position.

| selection | uniform | **tailed** | ascending | descending | worst case |
|---|---|---|---|---|---|
| **shipped** `select_nth_unstable_by` | 2.043 | **1.899** | 0.236 | 0.312 | 2.04 |
| **A** bounded heap on `f32` | 0.148 | **0.139** | **5.427** | 0.130 | **5.43** |
| **B** bounded heap on raw fp16 keys | 0.247 | **0.238** | 4.192 | 0.267 | **4.19** |
| **D** 4096-bucket counting select on fp16 | 0.330 | **0.321** | 0.595 | 0.605 | **0.61** |

End-to-end host cost per token, including the widen and the validity guard that each option
either keeps or absorbs (`tailed` row):

| | widen | validity | selection | **total** | **recovered** |
|---|---|---|---|---|---|
| shipped | 0.065 | 0.070 | 1.899 | **2.034** | — |
| A (keep widen, heap on f32) | 0.065 | 0.070 | 0.139 | **0.274** | **1.760 ms** |
| B (fp16 keys, no `Vec<f32>`) | — | — | 0.238 | **0.238** | **1.796 ms** |
| D (bucket select, no `Vec<f32>`) | — | — | 0.321 | **0.321** | **1.713 ms** |

B and D absorb the widen and the NaN check entirely, so they recover ~0.14 ms more than A.

**The honest catch: A and B have a data-dependent worst case.** A bounded heap is
`O(n + m log k)` where `m` is the number of elements that beat the running threshold. On a
row whose logits increase with vocab index, `m = n` and A is **23x slower than what it
replaces** (5.43 ms vs 0.24 ms). Such a row requires the embedding matrix to be sorted by
output magnitude, which is not a thing that happens — but "not a thing that happens" is
exactly the class of claim AGENTS.md warns about, so:

- **D is the only bounded option.** 0.61 ms worst case, 5x better than shipped's worst case,
  and it never allocates the 2 MiB index vector. It is the recommendation for anything that
  must not have a cliff.
- **A is the fastest thing available** (11-14x on realistic rows) and its cliff is reachable
  only by a sorted logits row. If A is shipped, the cliff is closable for ~10 lines: count
  heap updates and, past `n/4`, redo with D. Worst case then provably never exceeds shipped
  by more than ~0.3 ms, while the fast path stays at 0.139 ms.

---

## 4. Does sampling need to see all 262,144 logits? (Q4) — yes, and the *rule* is load-bearing

Two separate questions, both answered.

### 4a. Top-k by raw logit vs top-k by softmax probability

In exact arithmetic these are **identical**: `sample` computes `w_i = exp((l_i - m)/T)` with
`m` the maximum and `T > 0`, both independent of `i`, so `i -> j => w_i < w_j`. Ties survive
the transform as ties and both paths then break them on ascending index. There is no
probability gap to quantify.

**In f32 they are not identical, and the divergence is not a small one.** `exp` underflows to
exactly `0.0f32` once its argument passes about -103, so at low temperature thousands of
distinct logits collapse onto one weight, the weight ordering degenerates into an index-order
tie among thousands of zeros, and a top-k taken on the computed probability picks an
essentially arbitrary set:

| temperature | rows where the two orders differ | mean set difference | weights underflowed to 0.0 |
|---|---|---|---|
| 0.05 | **100 / 100** | **41.7 of 64** | 4053 of 4096 |
| 0.5 | 0 / 100 | 0.0 | 0 |
| **1.0 (shipped default)** | **0 / 100** | **0.0** | 0 |
| 2.0 | 0 / 100 | 0.0 | 0 |
| 10.0 | 0 / 100 | 0.0 | 0 |

So a device reduction **must rank by the raw order-preserving logit key**, exactly as
`top_k_order` does. "Top-k by probability, it's close enough" is not an approximation — at
`T=0.05` it replaces **two thirds of the candidate set**. Rank by logit and the two paths are
distribution-identical at every temperature. Pinned in
`top_k_by_logit_and_by_softmax_probability_diverge_only_via_f32_underflow`.

### 4b. The 256-byte hand-off (Q2) — feasible, bit-exact, and not worth a kernel

The arithmetic works out exactly:

- 64 candidates x 4 B (`argmax_key`, the packed value+index) = **256 B**, versus the 512 KiB
  read now.
- Everything else stays host: softmax over <=64, the top-p walk over ~2, the xorshift draw,
  the inverse-CDF. That is the **125 ns** measured above. It is free, and keeping it there is
  what makes `--seed` bit-exact — the RNG never moves, and an identical order forces an
  identical token. Necessary and sufficient, no caveats.

Three things the implementation must respect, all found in `elements.metal`:

1. **`value_key` is only safe through `argmax_key`.** `sortable16` (`:527`) computes `~bits` on
   a **32-bit** `uint`, so for a negative input the result has bits 16-31 set and lands at
   `>= 0xFFFF0000` — *above every positive key*. `argmax_key` (`:545`) immediately does
   `value_key(bits) << 16`, which discards exactly those bits, so the existing greedy path is
   fine. **Any kernel that consumes `value_key` without packing through `argmax_key` ranks every
   negative logit above every positive one and returns confident garbage.** Flagged because a
   device top-64 is precisely that shape of consumer.
   (`value_key_upper_bits_are_garbage_for_negatives`)
2. **A u32 total order over the full row does not exist.** 16 value bits + 18 index bits
   (vocab 262144) = 34. The existing two-level 65536-slot structure is what makes the u32 key
   exact, because `index & 0xFFFF` *is* unique inside a slot. So k=64 must extend that
   structure — 4 slots x 64 = 256 candidates into a final stage over 256 — and must not be a
   flat u32 reduction. (`a_u32_key_cannot_total_order_the_full_row`)
3. **`sortable(x) == 0` iff `x` is NaN** (verified exhaustively over all 65,536 patterns).
   That identity is what lets a key-based reduction reproduce the host's `InvalidLogits`
   rejection at one compare per element. Without it, a NaN row silently samples from the
   bottom of the distribution instead of erroring. (`key_zero_is_exactly_the_nan_set`)

**And `argmax_slots` (`:562`) does not extend to k=64 by adjusting a constant.** Its final step
is `atomic_fetch_max` into `best[slot]`, which is k=1 by construction; 64 rank-accumulator
atomics per element is not viable. A per-slot top-64 needs a shared-memory structure per
threadgroup. Also note `argmax_keys_to_token` (`:586`) compares only `(key >> 16)` across slots
and discards the index tie-break — harmless at k=1 with unique values, but it is a place where
the tie-break has to be re-established for k=64.

**The decisive point: a half-sampler does not remove the barrier.** It still needs a
device->host sync and a readback; it just shrinks the payload. The 0.15 ms is latency-dominated,
not bandwidth: 512 KiB / 0.15 ms is **3.4 GB/s, 1.3% of the 273 GB/s budget**. Going to 256 B
saves ~none of it. So the half-sampler buys roughly the same ~1.8 ms as option A — for a new
kernel, a new staging buffer, a new sync, and no removal of the stall.

---

## 5. Ranked options: ms recovered / risk

| # | option | ms recovered | risk | notes |
|---|---|---|---|---|
| **1** | **A — swap `top_k_order`'s body for a bounded heap** (optional: count updates, fall back to D past `n/4`) | **~1.76** | **~0** | One function body. No Metal, no API change, no caller change. Order proved identical over 42 row x k combos incl. tie and signed-zero boundary rows. Ships today. |
| **2** | **D — fp16 bucket-counting select; drop the `Vec<f32>`** | **~1.71** | low | Also removes the 0.065 ms widen and folds the NaN check in. Strictly bounded (0.61 ms worst). Touches `generate.rs:326-330` to pass `&[u16]`. |
| 3 | B — fp16 bounded heap | ~1.80 | low, unbounded | Fastest, but 4.19 ms on a sorted row. Only with a stated, argued precondition. |
| 4 | device **half**-sampler (256 B hand-off, host keeps sampling) | ~1.8 | medium | Same win as #1 for a kernel + a sync. Does **not** remove the barrier, so it does not touch the 3.83 ms. Strictly dominated by #1. |
| 5 | **full device sampler + pipelined forward** (in flight) | **~7.3** | high | The only option that removes the barrier (+3.83), the readback (+0.15), the host sampler (+2.87) and the widen (+0.45). `HeadPick::Sample` + `forward_step_pipelined` already exist in the dirty tree — the greedy path (`generate_gpu_greedy` -> `model.last_token()`) is the working proof that this architecture pays. |

**Recommendation.** Do #1 (and #2 if you want the bounded variant) now — they are additive to
the device work, not competing with it, and they cost one afternoon. Two reasons they are not
throwaway even if the full device sampler lands:

- `MAX_DEVICE_TOP_K = 1024` (`sampler.rs:102`): a config above that **falls back to the host
  sampler on the non-pipelined loop** by design. That fallback gets ~1.8 ms cheaper for free.
- The device path is distribution-identical to `Sampler::sample` only by construction
  (`sampler.rs:99-101`). Keeping the host reference *fast and exact* is what makes that
  claim checkable, and the in-tree oracle (`reference_top_k`) is the thing any future device
  kernel is diffed against.

**Do not skip #4 on the theory that it is a cheap partial version of #5.** It is strictly
dominated: same milliseconds, strictly more machinery.

---

## 6. Method, reproducibility, and what would falsify this

- Probe: `crates/engine/tests/probe_sampling_phases.rs` (7 tests, CPU-only, no Metal
  dispatch, no checkpoint). Built with an isolated `CARGO_TARGET_DIR` per AGENTS.md 10.4 so it
  could not rebuild the binary another agent was benchmarking.
- Numbers were **also** produced from the verbatim committed `sampler.rs`
  (`git show HEAD:crates/engine/src/sampler.rs`) in a standalone crate, because the other
  agent's in-flight `model.rs` edit (`model.rs:3179`, missing `SAMPLE_CANDIDATES`) did not
  compile at the time. Both agree to within 5%, which is the cross-check that the in-repo probe
  is measuring the shipped code. The standalone crate was pinned to the workspace release
  profile after an initial run used `lto=thin, codegen-units=1` and was ~5% optimistic.
- **Three real bugs were written and caught while producing this**, all of the same species and
  all invisible on a tie-free row: an inverted index tie-break (the comparator put the *higher*
  index first), a max-heap built where a min-heap was meant (keeps the best element instead of
  evicting it), and the `value_key` upper-bit hazard. The first two were found *only* by the
  tie-dense rows, and the third by the exhaustive 65,536-pattern check. That is the argument
  for shipping the equality gates with any replacement rather than after it: a wrong selection
  produces plausible text and no test failure.
- **What would falsify the headline:** an in-situ measurement (KAPLUMBA_STEP_TRACE, sampling
  path, same binary) showing `sample()` materially below ~1.9 ms, which would mean the selection
  is not the bottleneck and the win shrinks proportionally. The ratios, not the absolutes, are
  the claim.