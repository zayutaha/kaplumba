# spec/device-sampler — implementation notes

Running log. Appended as work proceeds, not at the end (AGENTS.md §5).

## Environment facts

- Box: M4 Pro, 20 GPU cores, 273 GB/s, 24 GiB unified. `scripts/gpuq status`
  reports 1.8 GiB available / 8.2 GiB in use by processes at session start,
  which its own note says is usually enough.
- Model in use: `~/.kaplumba/models/Gxma/Gxma`. Also present: `Gxma-4-bit`,
  `Gxma-bpw4-shim`, `Gxma-repacked`.
- Pre-change binary preserved at
  `$TMPDIR/opencode/dsampler/kaplumba-before` (`863bce5`). A/B is
  before-binary vs after-binary, interleaved, plus a before-vs-before control
  arm so the noise floor is measured in the same session.
- `scripts/sampler_ab.sh` hashes **generated text only** — never the
  `--metrics` line, because `load_ms`/`ttft_ms`/`decode_tok_s` vary per run.

## Baseline

Single sanity run at `--temperature 1.0 --max-new-tokens 48 --seed 1234`:

```text
prompt_tokens=22 generated_tokens=48 load_ms=16323.8 ttft_ms=2025.0
prefill_tok_s=10.880 decode_tok_s=11.636
```

Note this is below the programme's cited 14.37 tok/s at temperature 1.0.
Same binary, same prompt shape — which is exactly the §10.5 hazard, and the
reason the control arm exists. Do not treat any absolute figure in this
session as comparable to the programme's; only the interleaved pairs are.

### Control arm, `--temperature 1.0`, before-vs-before, 6 pairs

```text
pair 1: 13.311  67cf2352832a   |   13.355  67cf2352832a
pair 2: 13.208  67cf2352832a   |   13.261  67cf2352832a
pair 3: 13.256  67cf2352832a   |   13.281  43497ed89847
pair 4: 13.207  67cf2352832a   |   13.279  67cf2352832a
pair 5: 13.272  67cf2352832a   |   13.278  67cf2352832a
pair 6: 13.332  67cf2352832a   |   13.273  67cf2352832a
```

min 13.207, max 13.355 — **1.1 % spread**, not the 35 % that AGENTS.md §10.5
records for this box. Two things follow:

1. A `+1.5 tok/s ≈ 11 %` effect is, in principle, resolvable here. It will
   still be reported with spread and a control arm rather than as a single
   reading.
2. **Pair 3's control arm produced different text from the other five.** The
   *host* sampler at `--temperature 1.0` with `--seed 1234` is **not**
   bit-reproducible run-to-run on this box, before any change of mine. The
   sampler itself is deterministic given the row it is handed, so the row must
   be moving — i.e. the decode forward is not bit-reproducible run-to-run.
   This is recorded because it bounds what any seed-parity claim in this
   session can mean: the *pre-existing* host path does not have it either.

### Where the temperature-1.0 idle actually shows up

`KAPLUMBA_DECODE_PROFILE=step KAPLUMBA_STEP_TRACE=1`, `--temperature 1.0`,
12 tokens:

```text
GPU_STEP_TRACE: forward_buffer=180.09ms   <- first step, cold
GPU_STEP_TRACE: forward_buffer=73.40ms    ... 73.04 73.36 73.80 73.05 73.42 74.17 74.37 73.91 73.91
dp_span_ms_per_step_median=73.700 dp_span_ms_per_step_min=73.292 dp_span_ms_per_step_max=73.713
dp_wall_ms_per_token=24.679      <- meaningless here; see below
decode_tok_s=12.001
```

Two things this says, and one it does not:

- **The GPU is busy for the whole step.** `dp_span` (73.700 ms, the step's
  duration on the GPU timeline) equals the wall time of the `forward_buffer`
  call (73.4 ms). So the 2.4 ms of host sampling is *not* inside the step —
  it happens between steps, in `generate_gpu`'s loop, which is why
  `decode_profile`'s brackets never see it.
- **The step's GPU duration is unchanged by anything in this spec.** So if
  wall/token falls while `dp_span_ms_per_step` stays at ~73.7, the difference
  *is* GPU idle, removed. That is the mechanism metric, and it needs no new
  instrumentation. It is what T5 reports.
- `dp_wall_ms_per_token=24.679` is **wrong and must not be quoted**.
  `end_step` is handed a wall time measured from *after* the encoder finished
  encoding (`model.rs:3874`), so it covers wait-only, not the step. Recorded
  so the next reader does not "fix" a number that is already reported with a
  distortion note.
- The 180 ms first step is why `--max-new-tokens 12` reports 12.0 tok/s while
  96 tokens reports 13.3: the artifact is inside `decode_duration` and only
  amortises with length. All A/B numbers here use `--max-new-tokens 96`.

## T1 — `HeadPick`, and the row a pick reads

`skip_softcap` was `(next_token_out.is_some() || host_token_out.is_some())
&& !KAPLUMBA_FORCE_SOFTCAP`. Now `pick.skips_softcap(out.is_some())`, i.e.
`staging && matches!(self, HeadPick::Argmax) && !KAPLUMBA_FORCE_SOFTCAP`.

Behaviour preserved exactly for every existing caller: `HeadPick::Argmax` is
only ever passed together with `out = Some(..)`, so `Argmax + staging` is the
old predicate.

### The dispatch-count test is the sabotage-verifiable form

`encode_softcap` is one dispatch and nothing else in the head varies with
`pick`, so `dispatches_encoded()` after a step **is** whether the softcap ran.
One decode step measured on `Gxma/Gxma`:

```text
HeadPick::Argmax  -> 1499 dispatches
HeadPick::Sample -> 1500 dispatches   (the softcap)
```

Reverting `reads_raw_row` to `staging` makes both 1499 and the test fails with
`argmax=1499 sample=1499 ... reading the RAW logits`. The unit test
`only_argmax_with_staging_reads_the_raw_row` fails on the same sabotage. Both
were run; neither is a claim from reading the code.

### `KAPLUMBA_ABLATE=head_softcap` is the *end-to-end* probe, and it lands in T3

`encode_softcap` early-returns when ablated, so `scratch.capped` is not
written. With `HeadPick::Sample` wired (T3) the sampler then reads a **stale**
capped row, so ablating the softcap changes sampled output; if the predicate
were reverted, `Sample` would skip the softcap entirely and the ablation would
have no effect on that path, so the two outputs would become identical and the
test would fail. That is the required direction and it is why the flag works as
a probe — but it needs a *running sampler*, so the test itself is a T3 item.

Note the flag does perturb `--temperature 0` as well, and for an unrelated
reason: the **prefill** runs the head with `out = None`, so its argmax reads
`scratch.capped`, and ablating the softcap leaves that buffer unwritten. So
"`KAPLUMBA_ABLATE=head_softcap` leaves temperature 0 bit-identical" (A4 §7(b))
is asserted **without** the flag, and the flag test is a temperature > 0 test.
Recorded because the two claims are easy to conflate.

### Temperature-0 byte-identity (A1)

`863bce5`'s binary vs the T1 binary, `--temperature 0 --seed 7/99`, generated
text only (never the `--metrics` line):

| prompt | runs | before | after | equal |
|---|---|---|---|---|
| `Count from 1 to 30, one number per line.` (64 tok) | 3 | `5f83e016789476f0` ×3 | `5f83e016789476f0` ×3 | yes |
| `Write a haiku about the sea.` (48 tok) | 2 | `1a4c5af6ddc7f001` ×2 | `1a4c5af6ddc7f001` ×2 | yes |
| `What is 17 times 23? ...` (48 tok) | 2 | `9698c777c4c1b5c9` ×2 | `9698c777c4c1b5c9` ×2 | yes |

### Incidental findings

- **`cargo fmt -p kaplumba-engine` reformats 86 files.** The tree is *not*
  rustfmt-clean, so running `cargo fmt` produces ~4,800 lines of unrelated
  churn. I ran it once, reverted everything, and re-applied only the intended
  edits by hand. Do not run `cargo fmt` on this crate. The diff is now 6 files
  / +414 / -46.
- **`crates/engine/examples/attn_bench.rs` does not compile at `863bce5`.**
  `decode_splits(Q_HEADS, variant.key_count)` is missing its third argument.
  Pre-existing, unrelated to this spec, confirmed by stashing this spec's work
  and rebuilding. It makes `cargo test --workspace --no-run` fail, so the
  per-test-target invocations used here are deliberate.
- **`tests/perf_probe_dispatchsel_columns.rs::the_colmajor_contig4_kernel_is_not_on_the_decode_path`
  fails at `863bce5`** — also pre-existing, confirmed by stashing.
- **`tests/concurrent_stream_economics.rs` failed once** during a
  `cargo test --workspace` run and passed 3/3 when run alone afterwards. It is
  a bandwidth-ratio test on the GPU; running it concurrently with other tests
  is a plausible cause. Not reproducible here, recorded rather than dismissed.
## T2 — two bugs in the device sampler's `top_k` bound, found before any GPU run

Neither was found by a benchmark; both were found by reading the host validator
against the kernel geometry and sweeping them against each other. Both are in
code that was already written and compiling.

### 1. The bound was stated twice, and the kernel's copy was stricter

`SamplerParams::validate` rejected `top_k` above `MAX_DEVICE_TOP_K` (1024). The
kernel's own `ElementKernels::sample_geometry` accepts only

```text
count * k_padded <= SAMPLE_STAGE * SAMPLE_CANDIDATES   (4096 * 4096)
```

which at this model's 262,144-wide vocabulary is `k_padded <= 64`. Measured
with a standalone repro of the shipped function:

| top_k | host `validate` | kernel geometry |
|---|---|---|
| 64 | ok | ok (per_block 4096) |
| **65** | **ok** | **reject (per_block 8192 > 4096)** |
| 128 | ok | reject (per_block 8192) |
| 1024 | ok | reject (k_padded 1024) |

So `--top-k 128` was accepted by the host and then failed as a `KernelError`
partway through a decode step, rather than taking the documented "fall back to
the host sampler on the non-pipelined loop" path. The doc comment on
`MAX_DEVICE_TOP_K` promised a fallback; the code could not deliver one.

**Fix:** `device_top_k_limit(vocab)` asks `sample_geometry` and `validate` uses
that. The kernel is the single owner of the number. `MAX_DEVICE_TOP_K` survives
only as a sanity ceiling on the probe, so that raising it cannot silently raise
what the staging arrays accept. It is probed once per model load, not per token —
`device_sampler_supported` is what keeps it off the hot path.

**Gate:** `the_host_bound_and_the_kernel_geometry_agree_at_every_top_k` sweeps
6 vocabularies (4096 … 524288) x ~26 `top_k` probes and asserts `host.is_ok() ==
kernel.is_ok()` everywhere. Pointed probes are not enough: the two functions are
in different crates and nothing else makes them agree.

**Sabotage S1** (revert the bound to `MAX_DEVICE_TOP_K`):

```
vocab=4096 top_k=512: host says true, kernel geometry says false
(host: None, kernel: Some(InvalidInput("device sampler: top_k 512 is above the
256 the staging arrays accept")))
```

### 2. `next_pow2` overflowed to 0 and span an infinite loop

```rust
let mut k_padded = 1usize;
while k_padded < top_k { k_padded <<= 1; }
```

At `top_k = usize::MAX`, `k_padded` walks up to 2^63, shifts left to 0, and
`0 < top_k` is then true forever. This is an **infinite loop, not a wrong
answer**: a large `top_k` hangs the process instead of returning the error the
caller is waiting for. Confirmed by repro before fixing.

It was reached from `top_k = usize::MAX`, which the sweep gate probes — the gate
*hung* rather than failed, which is how it was found.

**Fix:** reject `top_k > SAMPLE_TOP_MAX` on the **input**, before padding.

Bounding the loop instead would **not** have been enough, and this is the part
worth remembering: stopping at the cap lets `top_k = 1000` through as
`k_padded = 256`, which samples from a narrower candidate set than the config
asked for and reports success. That is precisely the silent quality regression
AGENTS.md §8 forbids, and it would have passed every "does it not hang" check.
The cap is on the input; the loop bound is defensive only.

**Sabotage S2** (remove the cap check): the gate hangs, killed at 25 s. This is
the only gate in the set whose failure mode is a hang, so it is worth knowing
that it fails that way — a CI timeout, not an assertion message.

### Test-running notes on this box

- `cargo test --release -p kaplumba-engine --lib sampler` **exceeds 600 s** even
  with everything cached — the link step dominates. Build once with
  `cargo build --release -p kaplumba-engine --lib --tests`, then run the binary
  directly: `./target/release/deps/kaplumba_engine-<hash> sampler`. The hash is
  stable while the lib is unchanged and the whole suite runs in 0.06 s.
- There is **no `timeout` binary** here. To observe a hang without stalling the
  session, run the test in a subshell, poll `kill -0 $P`, and `kill -9` after N
  seconds. That is how S2 was captured.

## T2 — the device RNG was wrong on 100% of draws, and looked fine

`sample_rng_next` in `elements.metal` reproduced the host's `next_f64() as f32`
by hand-rolling the 53-bit -> 24-bit rounding in integer arithmetic: split into
a 24-bit `hi` and a 29-bit remainder, compare the remainder against a tie point,
add the carry, scale by a power of two.

**It was wrong twice, and the two errors cancelled into something much worse
than either.** The 29-bit remainder needs a `2^28` tie point, not `2^24`; and the
scale was `2^-53` applied to a `hi` that had already absorbed the `2^29` split.
Measured against the host over 3,000,000 draws:

```text
hand-rolled: 100% of draws differ, mean unit = 2e-9   (host: 0.5)
```

So `unit` was ~2e9 times too small. The draw is
`target = unit * selected_total`, then an inverse-CDF walk — with a `unit` that
tiny, `target` is below the first partial sum on essentially every token and
**every draw collapses onto the top-ranked token.** The device sampler would have
been a greedy decoder wearing a sampler's name, emitting entirely plausible text,
with no error, no NaN, and a distribution test that would have needed 1e5 draws
to notice.

**Fix: delete the arithmetic.** `float(bits >> 11) * 0x1p-53f` — Metal's
`ulong` -> `float` conversion already rounds to nearest-even, which is exactly
what the host's `f64` demotion does. **0 differences in 3,000,000 draws.**

The lesson is the one worth carrying past this spec: *"reproduce the host's
rounding in integer arithmetic so we never touch the GPU's float conversion"* is
a worse plan than using the conversion, because it replaces a well-specified
instruction with an unspecified argument. The original comment even argued the
hand-rolled version was safer. It was not safer; it was wrong.

**Gates** (`sampler.rs`, `device_rng_tests`, CPU-only, no GPU):

| gate | what it catches |
|---|---|
| `device_rng_matches_the_host_bit_for_bit` | 3,000,000 draws x 6 seeds, **bitwise**; also pins the xorshift advance and `unit ∈ [0,1)` |
| `device_rng_stays_uniform_and_in_range` | mean within 0.005 of 0.5, every decile within 0.01 |
| `a_zero_state_would_stall_and_is_substituted` | `xorshift64(0)` is a fixed point, so `seed_state(0)`'s substitution is load-bearing |

Bitwise, not approximate — and note that "close enough" would **not** have caught
this bug, which was not close. That is the argument for the exact comparison.

**Sabotage S3** (restore the original hand-rolled version): both value gates
fail, first draw:

```
device unit 9.833068e-11 (0x2ed83b40) != host 5.2790873e-2 (0x3d583b3e)
mean 0.0000000009315009616178704 is not 0.5
```

### A trap in writing this gate

The first version of `host_rng_step` advanced the xorshift **twice** — once in
the host step and again by reading the already-advanced state into the device
step — which made 100% of draws differ for a reason that had nothing to do with
the kernel. It read exactly like the bug under test. The state-advance assertion
in the gate is what makes that class of harness error impossible: if the two
advances disagree, the failure says so instead of masquerading as a rounding
difference. Two of my three intermediate repros were wrong in this way before the
harness was right; in both cases the mean was the thing that gave it away
(2e-9, then 32803, then 0.5 once the scale was right).

## T2, part 2 — two silent GPU-only defects, both found by gates rather than reading

Neither faulted, neither produced a NaN, and both would have shipped as
"working". They were caught only because the gates compare against an oracle
rather than checking for a crash.

### `sample_draw` read its parameters from a null binding

The signature declared

```metal
constant SampleDrawParams &params [[buffer(1)]],
device ulong *rng [[buffer(2)]],
```

with a `None` placeholder on the host side to leave a gap. But
`MetalEncoder::dispatch_groups` binds the parameter struct at
**`bindings.len()`** (`encoder.rs:453`) — i.e. after the trailing buffers,
always. Every other kernel in the file follows that convention; this one
assumed an in-the-middle gap was possible.

So `params` was never bound at index 1. The kernel read a null `constant`
struct, i.e. all zeros — and a zero `top_k` means the ranked-output loop
`for (i = 0; i < top_k; ++i)` **writes nothing**. The symptom was a
silently all-zero ranked row, not a fault.

**Fix:** params moved to `buffer(5)`, the `None` placeholder removed. This is
now commented in the signature, because "add an optional output buffer" is
exactly the edit that reintroduces it.

**Sabotage S5** (params back at `buffer(1)`, `None` placeholder restored):
`device_top_k_matches_the_host_set_and_order_on_adversarial_rows` fails.

### The merge kept the worse half of the sorted pair

```metal
sample_bitonic_pairs(mv, mi, 2u * k, lid, tgdim);
if (lid == 0u) {
    for (uint i = k; i < 2u * k; ++i) { mv[i - k] = mv[i]; mi[i - k] = mi[i]; }
}
```

The network leaves the best `k` of the concatenation in `0..k`, descending.
The "compaction" loop copied `k..2k` **over** `0..k` — keeping the worse half
and discarding the better. It is the shape of code that reads like tidying up a
merge and is in fact an in-place reversal of one.

On an all-equal row it returned indices `131072..` instead of `0..63`, because
the discarded half was the only half holding the low tie-break-winning indices.
Zero overlap with the host's answer on a realistic row.

**Found without a GPU round-trip.** The device's `all_equal` output
(`131072, 131073, ...`) was reproduced exactly by transcribing the sort and
merge into a standalone host program — which then showed the copy loop as the
sole difference. Worth remembering: a 262,144-wide GPU kernel is a ~8-second
cycle per hypothesis, and this class of bug is decidable in microseconds of
scalar Rust.

**Fix:** delete the loop.

**Sabotage S4** (restore the loop): the top-k gate fails.

### A test bug the gate caught, worth recording

The `extremes` row originally included `NaN`, and the host returned a `NaN`
element at **rank 0**. That was not a device bug: `top_k_order` documents that
it requires finite logits, and its comparator's `partial_cmp(..).unwrap_or(Equal)`
is not a strict weak ordering under `NaN`, so **the host has no defined answer
there**. The test was asking a function a question outside its contract, and the
"failure" was the harness's.

The device's behaviour is separately pinned by `nan_never_wins_on_the_device`:
`value_key` folds `NaN` to 0, below every real key, so it cannot be selected.
That is asserted directly, because there is no oracle to compare against.

Worth separating: *the gate failed* and *the gate was wrong*. Both are real, and
the second is the one that would have shipped a permanent red test that people
learn to ignore.

### TV threshold: measured, not guessed

Five independent 1e5-draw trials on this box:

```text
tv = 0.0344 / 0.0357 / 0.0336 / 0.0256 / 0.0346     (top prob 0.0159, full top-64 support)
```

That matches theory — 64 bins at mean `p ~ 1/64`, per-bin sd ~4e-4, summed to
~0.025 — so this is the sampling noise floor for a *correct* sampler, not bias.
The threshold is **0.06**, about 1.7x the worst observed trial.

The first draft asserted `< 0.05`, which sits *inside* the noise band and would
have flaked roughly half the time. And the margin it was protecting against is
enormous: a sampler collapsed onto the argmax has TV ~ 0.98. Correct-vs-broken
is ~30x apart, so the threshold should be loose, not tight. A tight threshold
here buys nothing and costs a coin flip.

### A process note: commit kernel edits immediately

The merge fix was applied, verified on GPU (probe showed `0,1,2,...` matching
the host), and then **lost** when the session restarted — it was uncommitted. The
next run rebuilt cleanly, passed a compile, and failed with byte-identical
output to the pre-fix run, which read exactly like a stale binary and was not
one.

Two lessons, both cheap:
- `md5 -q` the file after a kernel edit, not just the test result. The test said
  "failed for a reason you think you already fixed"; only the md5 said the fix
  was not in the file.
- Commit `elements.metal` in the same step as the fix. An uncommitted kernel
  edit is one session restart from gone, and the failure mode is
  indistinguishable from a build problem.
