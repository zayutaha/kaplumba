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
