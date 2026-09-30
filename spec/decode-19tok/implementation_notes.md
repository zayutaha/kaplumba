# Implementation notes: decode 8.1 to 19 tok/s

Running log for `spec/decode-19tok/`. Appended as work proceeds, not at the end
(AGENTS.md §5). Factual and dense; prefer paths, numbers and commit hashes over
narrative.

## 2026-09-29: spec created from a day of research

Three subagents were running when this spec was written (decode timing
instrument `ses_f14815069ffef7HavNB2d5a3T5`, streaming concurrency
`ses_f147351b1ffe7nZ4YVMbwqvHMq`, and a decode-attribution agent that never
started). Their results are NOT in this file — they were still running. What
follows is the state as of spec creation.

### The headline, and why it moved twice in one day

| time | claim | why it changed |
|---|---|---|
| morning | roofline 22.6 tok/s, from 273 GB/s ÷ 12.06 GB | 273 GB/s is arithmetic (256-bit x LPDDR5X-8533), not a measurement |
| midday | ceiling 248.5, then 250.7 GB/s | `bw.rs` reported min-of-passes; on a shared box that selects the pass that ran while a competitor was idle |
| now | **~220 GB/s median** -> 16.7 tok/s | repaired probe, quiet box, median of 3: 220.8 / 218.1 / 219.7 at 6 GB, 228.6 at 2 GB, spread 0.8-0.9x |

The 248.5/250.7 figures sat *above* what NVIDIA's GB10 — an identical
256-bit LPDDR5X-8533 part — measures (229-230 GB/s), and above four generations
of Apple STREAM results. That should have been the tell. The independent
literature reconciliation also put Apple silicon DRAM efficiency at 82-86%, and we
measure 80-81%.

**19 tok/s therefore needs 248 GB/s, which is not available.** The spec is
written to discover whether ~220 is a wall rather than to assume it, because if
it is a request-rate wall the target moves back into reach. That is T2, and it
is the cheapest high-value measurement available.

### What is solidly measured

- **Decode: 8.1 tok/s = 123 ms/token.** Clean box, n=5 at N=64:
  7.841 / 8.176 / 8.062 / 8.050 / 7.947. Flat from N=64 to N=256, so 123 ms is
  steady state, not startup-inflated.
- **Startup amortises over ~64 tokens**: 5.221 / 5.674 at N=16, 6.868 / 6.990 at
  N=32, 8.076 / 8.041 at N=64, 8.139 / 7.980 at N=128, 8.095 / 8.140 at N=256.
  Short generations read ~40% slow, which has skewed some comparisons.
- **13.145 GB/token**, from the safetensors headers: down 2.872, gate 2.872, up
  2.872, o 1.230, q 1.230, k 0.527, v 0.485, head 1.057. 60 layers, 2422 tensors.
- **KV is not 50% of a decode token.** That claim came from the broken profiler
  (below). The bytes are right — 553 MB at 614 context, 1479 MB at 8k, matching
  `config.json` (50 sliding x 16 kv x 256, ring clamped to 1024; 10 full x 4 kv x
  512) — but isolated timing put the kernel at ~14 ms of a ~125 ms step (~11%),
  and ~48 ms was unaccounted for *outside* the kernel body.

### The two measurement failures that shaped this spec

**1. `KAPLUMBA_GPU_PROFILE=1` measures prefill, not decode.** `sum of per-kernel
GPU own_ms` is flat at ~1270 ms across N=16/32/64/128:

| N | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| own_ms | 1263.4 | 1276.7 | 1269.8 | 1281.3 |

It cannot be per-token decode time. The plausible mechanism is that decode's
timestamp queries resolve to command-buffer boundaries inside the load+prefill
window, so prefill's work is attributed to decode kernels. **Every per-kernel
decode table in this project is really a prefill profile.** Void: "KV attention
is 9% at 614 / 50% at 8k", "the 7 matvecs are 98.6% of a decode token", and the
per-kernel table in this session. The last is also arithmetically impossible —
it implies 265% of peak bandwidth. Diagnosis with `path:line` is T1.

**2. `bw.rs` never wrote its buffer.** It allocated multi-GB and streamed
untouched zero pages; untouched pages are not a memory-system measurement. Its
55 GB/s became the "achievable bandwidth" denominator for an entire round of
experiments, including the brief I wrote for ten agents. Repaired in `9906610`,
which also switched the report from min-of-passes to median/best/worst/spread.
An agent had independently watched the same binary swing 79.9 -> 230.5 GB/s in
10-second intervals while load *fell* — the selection effect, in the wild.

### The trap table (reproduced in plan.md — every harness was wrong first)

| harness | defect | false result |
|---|---|---|
| `bw.rs` | never wrote its buffer | 55 GB/s |
| `bw.rs` | min-of-passes | 248.5 / 250.7 GB/s |
| load-width probe | wrong CPU reference twice (`uchar` wrap; `uint2`/`uint4` output stride) | invalid |
| KV attention harness | scanned 7814 keys on a 1024-slot ring against an 7814-key count | 20,340 GB/s |
| KV attention harness | read a single L2-resident buffer (48 MB L2) | 1,592 GB/s |
| attention tests | a z/w lane swap in the QK dot product PASSED; hardcoded `splits=8` vs shipped 16 passed until data varied | no teeth |

### Dead ends, so they are not re-tread

- **Occupancy is not the limit.** 8 accumulators instead of 16 measured 0.944x
  (slower). The premise failed: a decode/prefill dispatch is 1680 threadgroups on
  20 cores = 84/core, so any 3-vs-4 cap is oversubscribed ~20x and cannot bind.
  A probe found no knee at 3 or 4 and no sensitivity to +-32 dummy registers.
  `maxTotalThreadsPerThreadgroup` = 1024 even with 64 live float4.
- **Little's law is the wrong model here.** Bandwidth is flat from 0.082M to
  0.655M loads in flight — 8x concurrency, 1.06x bandwidth. The part saturates
  near 0.164M and refuses the rest, so the ceiling is request-throughput, not
  latency x concurrency. The old sweep only went to 20,480 threads; T2 goes past
  it.
- **The dequant is free in prefill.** Removing all dequant math made the kernel
  4.4% *slower*; the "46.6% of kernel time" figure was a dead-coded-away trellis
  *fetch*, not arithmetic. This does **not** transfer to decode, which is not
  compute-bound.
- **No bytes to reclaim by repacking.** 410 of 411 tensors are exactly 3.000
  bits/value; `lm_head` is 6.000. The kernel calls `decode_state(s & 0xffff, …)`
  on the full 16-bit window, so 14,490 distinct states -> 7,103 distinct values
  (12.51 bits of real content), but that is entropy already paid for in 3 bits.
- **Not page faults** (20/token, 0.003%), **not compression** (18G wired, ~450
  MiB compressor, weights pinned), **not power or thermal** (sustained 200.5 beat
  burst 117.5; `pmset -g therm` clean before and after 180 s), **not the 411
  buffers** (already 23 x 512 MB arena segments), **not host encode** (0.4% of a
  token, ~250x headroom), **not full residency** (rejected on measurement).
- **The `uint4` change was aimed at a wrong threshold.** It was briefed to raise
  bytes-in-flight from 12 to 30 B/thread, but the load-width sweep found 8 B/thread
  already saturates. The change may still be worth something on its own merits,
  but not on that reasoning, and it was killed before landing.

### Live state of the two running agents

- `wt-tm` (timing instrument) was running a **sabotage test** at last look:
  rewriting its span to `open.own_ms` so the step's work falls outside the
  measured interval, to confirm the instrument fails when it should. That is the
  right thing to be doing.
- `wt-conc` (concurrency) was running `conc grid 11`. It then **leaked 3.8 GB of
  RSS in one process without freeing between configurations**, driving the box to
  3.7 GiB free and into **573 MB of swap** (was 0). I killed the probe; memory
  recovered to 6.8 GiB. This is why every task in this spec says probes must
  free per configuration and why a leaked probe is grounds to kill it: swapped
  measurements are worse than no measurements.

### The memory gate was wrong, and it blocked two agents for an hour

Both running agents were told to run the model only when `gpuq status` reported
**>= 14 GiB available**. Neither ever got there, so **neither ever validated its
work against the real model.** The concurrency agent could not test; the timing
instrument's 123 ms cross-check is explicitly still unverified.

The gate was too strict. Verified directly: a model **loads and runs correctly at
6.7 GiB "available", with 0 MiB swap**, producing the correct answer. The reason
is that the ~12 GiB model sits in **clean, reclaimable file cache** after being
read all day, and macOS evicts that on demand before swapping. `gpuq`'s own
parser counts reclaimable cache as available, and after clearing ~81 abandoned
worktrees out of `/tmp` it reported **15.0 GiB available, 2.4 GiB in use, 0 swap**.

`scripts/gpuq` now says so explicitly when it warns, so the next agent does not
sit idle waiting for a condition that is not required. The real constraint is
unreclaimable anon+wired, not the headline "available" number.

The `/tmp` cleanup was worth doing on its own: **85 stale worktree dirs** had
accumulated across sessions, several 340-440 MB cargo target dirs. That cleanup
is what took the box from 3.7 GiB free and 573 MB of swap to 15.0 GiB and 0 swap.

**Lesson for the spec's own rules:** the ">= 14 GiB" gate written into
`tasks.md` was inherited from a guess, not a measurement, and it prevented
exactly the measurement it was meant to protect.

### The decode profiling gate was the real bug (not timestamp aliasing)

`KAPLUMBA_GPU_PROFILE=1` was **not** mismeasuring decode — it was never measuring
it. `GpuProfile` is constructed inside `forward_buffer_into` (`model.rs:2590`),
once per call, and a decode step is a one-token call. The gate at
`model.rs:2588-2589`:
```rust
let profile = env_flag("KAPLUMBA_GPU_PROFILE")
    && (token_ids.len() > 1 || env_flag("KAPLUMBA_GPU_PROFILE_DECODE"));
```
is **false for every decode step** unless `KAPLUMBA_GPU_PROFILE_DECODE` is also
set. No boundary is stamped, `GpuProfile::stamp` returns immediately, nothing
accumulates. The single table a decode run prints comes from the *prefill's* one
call (`generate.rs:457`), and its `own_ms` sum therefore depends only on the
prompt — which is exactly the observed flatness. My timestamp-aliasing hypothesis
was wrong; this was a one-line gate that had been silently off.

Two encoder defects blocked measurement on top of that, both fixed in the agent's
worktree: `commit_and_wait` took `self`, and GPU timestamps are undefined until
completion, so **no caller could ever read one**; and `commit_stamped` re-armed a
buffer that `Drop` then released uncommitted, which accumulates and **hangs**
`new_command_buffer` (the agent's first test hung indefinitely). Added
`commit_stamped_and_wait` and `finish`.

The instrument is `KAPLUMBA_DECODE_PROFILE=step`, two stamped boundaries per step
covering the ~42 rolled buffers between them. Evidence lives in
`tests/decode_profile_gpu.rs` and `tests/gpu_profile_decode_gate.rs`.

| property | result |
|---|---|
| scales with work | 16.3 / 37.9 / 81.5 / 168.8 ms over 16/32/64/128 steps = 10.3x for 8x work (flat would be 1.0) |
| excludes prefill | 38.08 ms clean vs 37.99 ms after a 466 ms prior block = 0.2% |
| resolves injected cost | +6 dispatches moved the span 1.153 ms vs host 1.161 ms, 0.7% apart, 82x the 0.014 ms noise |
| sabotage | all three modes fail diagnosably: widened scope -> 67x; span = open buffer only -> injection invisible; prefill inside brackets -> 38 to 13069 ms |

**A real bug the injection test found:** `spin_burn` read `iters` from
`buffer(1)` while `dispatch_1d` packs params at `bindings.len()` = 2, so 24
dispatches x 200,000 iterations injected **0 ms and reported success**. Fixed,
pinned at kernels level. That is precisely the AGENTS.md §3.4 case: a harness
that reported success while measuring nothing.

**Not yet verified against the model**, for the gate reason above. The agent's
own instruction stands: do not attribute any decode figure until the 123 ms
cross-check is done. `scripts/decode_run.sh` is ready and self-gating.

Noise floor, synthetic quiet box: per-step span p10-p90 is **1.2-4.6%** of a
step. Host wall exceeds span by a **constant 0.176-0.203 ms** regardless of step
size — `wait_until_completed` wake-up latency, not GPU time. MDE on a real 123 ms
step is unknown, but the ratio is the reliable signal, not the raw number.

### Decisions taken that the plan did not predict

- **The spec is written to discover whether 220 GB/s is a wall, not to assume
  16.7 is the answer.** The plan would have set 16.7 as the target. Keeping 19
  as the stated goal with an explicit gate (T2) is more honest than silently
  lowering the target, and T2 is cheap.
- **Bytes-per-token was promoted to a full requirement (R4) rather than a
  postscript**, because 13.145 GB/token is a property of the design and not a
  law. It is the only lever that does not depend on T2's answer.
- **Speculative decoding is named explicitly in T4.** If T2 returns "wall" and
  the byte-trimming rows are exhausted, weights-not-changing-between-tokens is
  the only structural move left, and it would be easy to lose.

### Environment facts worth keeping

- One model instance takes 11.7-13.0 GiB of a 24 GiB box, so **exactly one fits**
  and a second will swap. `gpuq status` must read >= 14 GiB before a model run.
- `getrusage().ru_maxrss` under-reports by ~5.5 GiB here (6.23 vs 11.69 real at
  614 tokens) because Metal buffer pages are not counted as resident. The field is
  now `peak_resident_pages_bytes` (`ee14f62`) and must never be read as a
  footprint. Use `footprint -p <pid>`, which quantises to 3 significant digits.
- Test sentence is 6.01 tokens: `*100` = 614, `*1300` = 7814. Over 8192 is
  rejected *before* loading and looks like a silent failure.
- `scripts/gpuq holder()` used to name dead pids as the GPU holder (`waiters()`
  pruned, `holder()` did not) — fixed in `ee14f62`.


## 2026-10-01: the decode split count, from a workgroup budget rather than a thread count

Branch `impl/attn-splits`. Answers this spec's open question "does the split
count matter?" and the split-count paragraph in `plan.md` §R2.

### 1. The diagnosis, and it is not the one the note assumed

`crates/kernels/src/attention.rs` derived the split count from a **thread**
target and was never handed `kv_heads`:

```rust
let wanted = ATTENTION_DECODE_THREADS_TARGET / (32 * q_heads.max(1)); // 16384 / 1024 = 16
```

A decode workgroup is `32 * (q_heads / kv_heads)` threads and the grid is
`splits * kv_heads` workgroups. So holding *threads* at 16384 lands both of
Gxma's shapes on 16 splits while giving them grids that differ 4x:

| layer | shape | workgroup | grid at 16 splits | per core (20 cores) |
|---|---|---|---|---|
| 50 sliding | 32 q / 16 kv / d256 | 64 threads | **256** workgroups | 12.8 |
| 10 full | 32 q / 4 kv / d512 | 256 threads | **64** workgroups | 3.2 |

A scheduler packs *workgroups* onto a core, and a 256-thread workgroup occupies
about 4x the per-core resources of a 64-thread one, so the thread target was
implicitly assuming a residency the wide shape does not have. That is the whole
mechanism; nothing about the number of keys or the partials is involved.

### 2. What the one-wave budget is, measured

Swept every split count 1..=16 for three shapes that all use a **256-thread**
threadgroup and differ only in `kv_heads`, so `splits * kv_heads` walks multiples
of 4, 5 and 6 and the one-wave/two-wave transition is bracketed from both sides.
The 4/5/6 shapes at 4096 keys, us per split+combine pair, 9 interleaved rounds:

| grid (workgroups) | 40 | 45 | 48 | 50 | **52** | **54** | **55** | **56** | 60 | 64 | 65 | 72 | 80 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `40q/5kv` best | 40 | | | **50** | | | 55 | | 60 | | 65 | | **80** | |
| `48q/6kv` best | | | 48 | | | **54** | | | 60 | | | 72 | | 96 |
| waves | 1 | 1 | 1 | 1 | **1** | **1** | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |

**54 workgroups of 256 threads is one wave and 55 is two.** So the budget is 54,
which is `18 * 3` rather than `20 * 3`; the constant is stated as the
measurement because which of those it is changes no answer, and picking a core
count to make the arithmetic tidy would be inventing a fact.

The rule `budget / kv_heads` then reproduces the *measured* optimum of all three
shapes without being told what it was — `4 -> 13`, `5 -> 10`, `6 -> 9` — which
is the evidence that it is a rule and not a fit to the 4-kv-head number. A
budget chosen to be tidy instead (60, from "20 cores x 3") would say 15, 12 and
10, and 15 and 12 are both **measured worse than the 16 that ships today**
(1.34x and 1.17x slower respectively at 4096 keys). `attention.rs`'s
`the_budget_reproduces_the_measured_optimum_of_every_shape_it_was_measured_on`
is that check.

The 64-thread width's budget is 288, and that is a **lower bound**: the sliding
curve is still improving at the 256 workgroups the `ATTENTION_SPLITS_MAX` cap
allows. 288/16 = 18 splits, above the cap, so the cap decides and **the shipped
16 is unchanged for 50 of 60 layers**. That is structural, not luck, and it is
asserted against the old formula rather than a transcribed number
(`sliding_layers_are_bit_identical_in_dispatch_to_the_legacy_rule`, which
compares against the pre-change expression itself).

### 3. The result: split count and bandwidth, before and after

9 interleaved rounds, GPU clock, one command buffer per arm, medians. **Taken
while another agent had a model resident on the box**, so absolute GB/s is
understated and the *ratios* are the trustworthy part; the within-round ratio
column is the estimator that survives contention (both arms of a ratio are
measured inside the same round, ~0.5 s apart).

| shape | keys | before splits / wgroups | waves | us | GB/s | after splits / wgroups | waves | us | GB/s | speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| sliding 32q/16kv/d256 | 512 | 16 / 256 | 1 | 57.8 | 145.1 | 16 / 256 | 1 | 57.8 | 145.1 | **1.00x, unchanged** |
| sliding 32q/16kv/d256 | 1024 | 16 / 256 | 1 | 127.3 | 131.8 | 16 / 256 | 1 | 127.3 | 131.8 | **1.00x, unchanged** |
| full 32q/4kv/d512 | 512 | 16 / 64 | 2 | 85.0 | 49.3 | 8 / 32 | 1 | 76.5 | 54.8 | 1.11x |
| full 32q/4kv/d512 | 1024 | 16 / 64 | 2 | 134.0 | 62.6 | 13 / 52 | 1 | 120.8 | 69.5 | 1.11x |
| full 32q/4kv/d512 | 2048 | 16 / 64 | 2 | 226.3 | 74.1 | 13 / 52 | 1 | 180.4 | 93.0 | 1.25x |
| full 32q/4kv/d512 | 4096 | 16 / 64 | 2 | 428.7 | 78.3 | 13 / 52 | 1 | 303.9 | 110.4 | 1.41x |
| full 32q/4kv/d512 | 8192 | 16 / 64 | 2 | 835.2 | 80.3 | 13 / 52 | 1 | 622.5 | 107.8 | **1.34x** |

At 512 keys the *keys-per-split* floor decides, not the wave budget
(`ceil(512/64) = 8`), so the full shape is 8 splits there however many cores
there are.

Paired within-round ratios against 12 splits, which is how 12 and 13 were
separated (they are ~6% apart, measured through 20-140% per-arm spread):

| shape | keys | 13 | 15 | 16 |
|---|---|---|---|---|
| sliding | 512 | 1.008 | 0.906 | 0.780 |
| sliding | 1024 | 0.989 | 1.003 | 0.896 |
| full | 512 | 0.971 | 1.337 | 1.139 |
| full | 1024 | 0.926 | 1.235 | 1.085 |
| full | 2048 | 0.904 | 1.343 | 1.236 |
| full | 4096 | 0.917 | 1.338 | 1.244 |
| full | 8192 | 0.936 | 1.336 | 1.168 |

Read that as: **13 beats 12 by 6-7% on every full-shape key count**, and both beat
16 by 17-27%. The sliding rows are the control — 16 is 11-22% *faster* than 12 or
13 there, so the rule must not touch them, and it does not.

**Per token**, 10 full-attention layers: at 8192 keys, 212.7 us/layer x 10 =
**2.13 ms/token**; at 1024 keys, 13.2 us x 10 = **0.13 ms/token**. Both are lower
bounds for the reason in the caveat above, and lower than the 2.8 / 0.39 ms the
task brief predicted from the reference harness. Under a contended memory system
a 1.34x ratio compresses, so the honest reading is "at least 1.34x on those 10
layers at 8k, and worth re-measuring on a quiet box".

### 4. Partials are not the cause, as claimed

At 8192 keys, 16 splits move 2.0078 MiB of partials (written by the split pass,
read by the combine) against 64.0 MiB of KV — 3.14%. 13 splits move 1.631 KiB less
than 16 does, about **0.9 us at 235 GB/s**, against a 212.7 us difference. So
99.6% of the win is parallelism and 0.4% is partial traffic, which is the
"97% parallelism" claim confirmed rather than assumed.

### 5. BIT-EXACTNESS: it fails, and the change is default-off because of it

`KAPLUMBA_DECODE_SPLIT_RULE=workgroups` selects the new rule. **The default is
unchanged** and stays the thread target.

**The control first**, as required — the golden hash reproduces under unchanged
behaviour, so the harness is sound and the refactor did not perturb the default:

```bash
BIN=$CARGO_TARGET_DIR/release/kaplumba
M=~/.kaplumba/models/Gxma-repacked          # NOT the Gxma/Gxma default
PP="Explain the theory of relativity in simple terms, and then explain why it matters."
scripts/gpuq try -- "$BIN" --model "$M" --prompt "$PP" \
    --max-new-tokens 128 --temperature 0 --show-ids > out.txt
shasum -a 256 out.txt
# 2fff2c970b611f64c4a485d8a3a3dc719a0f03293a3f48c167f4dff5466795b2   MATCH
```

**And the one-wave arm reproduces the same hash too** — which is a result worth
having, because it shows the recorded gate is *blind* to this change:

```text
arm A (default)        2fff2c97...6795b2
arm B (workgroups rule) 2fff2c97...6795b2     identical
```

The reason is structural, not luck: 128 generated tokens over a 28-token prompt is
~156 keys, and at 156 keys the *keys-per-split floor* caps the full shape at
`ceil(156/64) = 3` splits in both arms. The wave budget wants 13, the floor says
3, and `min` gives 3 either way. **A 128-token gate cannot see a change that only
bites above 13 x 64 = 832 keys.** Anyone using that hash as evidence of
bit-exactness for this change would be wrong, and would be wrong *silently*.

So the gate was re-run where the rule actually bites — the same 34 000-character
prompt (`"Explain the theory of relativity. "` x 1000, ~6010 tokens) and the same
128 greedy tokens, `--show-ids`:

| | arm A (default, 16 splits) | arm B (workgroups, 13 splits) |
|---|---|---|
| sha256 | `94713d903a338fc89357a3b698bd42ac65a89f448cb93dd803409963ad4a543a` | `3c65eb18bce3746a56b35b3fc6a6b7d496f634a454b044ee4b578c5d5a5c0597` |
| tokens | 128 | 128 |
| differing positions | — | **43 of 128** |
| first divergence | — | **token 86** (1-based), i.e. after 85 identical tokens |

**The size of the numerical change**, measured on the kernel rather than inferred
(`attention_single_pass_bit_exact.rs::the_one_wave_split_rule_costs_one_fp16_ulp_and_is_not_bit_exact`):

| full 32q/4kv/d512 | differing fp16 elements (of 16384) | max ULP |
|---|---|---|
| @1024 | 3 | 1 |
| @2048 | 4 | 1 |
| @4096 | 4 | 1 |
| @8192 | 3 | 1 |

**1 fp16 ULP on 3-4 elements out of 16384 is the fp16 rounding floor.** It is
0.02% of the output, four orders of magnitude below the 2.16e-3 the project
already accepts from prefill's fp16 operand rounding — and it still flips a third
of the sampled tokens, because greedy decode is a discontinuous function of the
logits. That gap between "numerically negligible" and "changes the answer" is the
entire argument for the numerically-frozen classification, and it is why this
lands default-off with the decision left to the owner.

Two things that are worth stating because they are easy to get wrong:

  * **1 ULP is not a guard.** Verified by sabotage: comparing **1** split against
    16 also stays within 1 ULP (1-18 elements differ, depending on key count).
    So "the deviation is at most 1 ULP" is true of *every* split count and
    discriminates nothing. The test therefore asserts the **exact** element
    counts, which do discriminate: comparing 16 against 16 fails (0 vs 3), and
    comparing 14 against 16 fails (1 vs 4).
  * **The deviation is not monotone in the split-count difference.** 13-vs-16
    moves 3-4 elements; 1-vs-16 moves 1-18. The fp16 store absorbs the
    reordering almost entirely, which is the same finding as
    `split_above_one_is_not_bit_exact_but_fp16_absorbs_most_of_it`.

### 6. Sabotage evidence (AGENTS.md §3.4) — every one reverted, suite green after

| what was broken | observed failure |
|---|---|
| `256 => Some(54)` -> `60` (the tidy "20 cores x 3") | 3 tests fail: `the_budget_reproduces_...: left 15, right 13`; `the_reported_split_table: full @1024 split count moved, left 15 right 13`; `full_layers_fit_one_wave: left 60` (budget) |
| `256 => Some(54)` -> `36` | same 3 tests fail (`left 9, right 13`) |
| `64 => Some(288)` -> `128` (**the sliding regression**) | 3 tests fail, and the message names the requirement: `sliding @1023 keys: the rule changed the split count, and 50 of 60 layers depend on it not doing so`, `left 8, right 16` |
| the original bug, reintroduced: threadgroup width from `kv_heads` instead of `q_heads / kv_heads` | 3 tests fail, `left 16, right 13` |
| keys-per-split floor dropped | 5 tests fail, incl. `full @1: 15 splits exceeds the 1 the key count can afford` |
| GPU test: compare 16 against 16 | fails with `0 of 16384 fp16 elements differ, measured 3` |
| GPU test: compare 14 against 16 | fails with `1 of 16384 fp16 elements differ, measured 4` |
| GPU test: compare 1 against 16 (attempted) | **PASSED — recorded as a non-guard**, see §5 |

The fourth row matters most: the suite fails loudly on the *sliding* regression
that the brief forbids, not on a proxy for it.

### 7. Three harness defects, all of which produced a plausible wrong number

The sweep was rebuilt from scratch three times. Every one of these produced
numbers that looked like measurements:

1. **A `MetalEncoder` per arm.** `commit_stamped_and_wait` re-arms a fresh
   command buffer, and dropping the encoder releases that buffer *uncommitted*.
   64 warm-up arms took **15 minutes** and the sweep never reached round 0 — the
   queue blocking in `new_command_buffer`, which is the exact trap
   `MetalEncoder::finish`'s doc comment describes. One encoder for the whole run.
2. **Rolling left on (the default is 50 dispatches).** `GPUStartTime`/
   `GPUEndTime` describe *one command buffer*, and the dispatch counter is
   cumulative across arms, so from the second arm onwards a 48-dispatch
   measurement straddles a roll and the stamp describes a fraction of the work.
   Measured directly: the same arm read **0.000 ms** at 100+ iterations and
   **721 us against a true 170 us** at 1-10; the sliding shape came out at
   **0.2 us per pair**, i.e. 42 TB/s for an 8.4 MiB read. Every number before
   `set_max_dispatches(1_000_000)` was noise. This is worth knowing about
   *any* GPU-clock measurement in this repo that batches dispatches.
3. **`(0..ATTENTION_SPLITS_MAX)` instead of `(1..=ATTENTION_SPLITS_MAX)`** in the
   rotating order: the arm at `splits == 0` does not exist, the run died at round
   0 with a bare `.expect("arm")`, and 2.5 minutes were spent wondering why.

The harness itself (`crates/engine/examples/attn_splits_sweep.rs`, plus an
`attn_clock_diag.rs` used for defect 2 and an `attn_splits_ulp.rs` used for §5)
was **deleted rather than committed**, per AGENTS.md §10.6. The method is
reproducible from this note: build the two real shapes plus the 5- and
6-kv-head probes, sweep `splits` 1..=16, one command buffer per arm with rolling
disabled, warm every arm once first, rotate the order by round, and compare
within-round ratios.

### 8. The comparison this repo has already made, which cuts against the obvious argument

The argument "it is only 1 fp16 ULP, and this project already accepts 1 ULP" is
**false as stated**, and the repo contains the counter-example already.
`spec/decode-19tok-repacked-250gb` §21.2-21.4 measured a *larger* perturbation at
a different site:

| site | perturbation | layers | tokens changed (128 greedy) |
|---|---|---|---|
| gate/up GEMV fusion (§21.2) | 1 ULP on **0.78%** of activations (168 of 21504) per layer | **all 60** | **0** — byte-identical output |
| decode attention split count (here) | 1 ULP on **0.02%** of attention outputs (3-4 of 16384) per layer | **10** | **43**, first at token 86 |

The smaller perturbation did much more damage. **1 ULP is therefore not a unit of
comparison across sites**, and "we already accept 1 ULP" cannot authorise this
change without a measurement. A mechanism is plausible — the gate/up
perturbation is a rounding difference in a value that then meets a residual add,
while an attention-output perturbation is multiplied by attention probabilities
and re-read by every later layer's projections — but that is a hypothesis and
nothing here tests it. The table is what is measured.

### 9. Not fixed here

- `pipeline_audit::tests::every_pipeline_construction_goes_through_the_audit`
  fails on this branch at `linear.rs:399` and **fails identically on the clean
  tree** (verified by stashing this change and re-running). Not ours; not fixed.
- The end-to-end decode-step number (ms/token for a real 60-layer forward) was
  **not** measured. Everything here is the isolated kernel, on a box with another
  agent resident, and the brief's own numbers (2.8 ms/token at 8k) come from a
  harness that is not in the tree. A quiet-box `--metrics` A/B of
  `KAPLUMBA_DECODE_SPLIT_RULE` is the missing acceptance measurement, and the
  128-token golden hash is **not** a substitute: §5 shows it is structurally
  blind to this change.

### 10. The decision this leaves to the owner

`KAPLUMBA_DECODE_SPLIT_RULE=workgroups` is in the tree, measured, tested,
default-off. Flipping the default is a one-line change to
`decode_split_rule()` and buys 1.34x on 10 of 60 layers at 8k of context, at the
cost of a token stream that is valid, differently-worded, and not identical.
`spec.md`'s "no quality loss" and the §8 precedent both point at leaving it off
until someone decides on the record that a 1-ULP-on-3-elements change to ten
layers' attention output is acceptable. That is a judgement about the model
rather than about the kernel, which is why it is not made here.
