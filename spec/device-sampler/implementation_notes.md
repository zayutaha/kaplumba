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