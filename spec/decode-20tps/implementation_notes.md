# IMPLEMENTATION NOTES: 17-18 tok/s Decode

Running log per AGENTS.md §5.

---

## 0. Prior art and what NOT to redo

This spec builds on top of the completed work in `decode-25tok-peak-bandwidth` (T1-T5.6)
and `decode-3bpw-amplification` (T1 negative). The standing record:

- **Baseline: 15.85 tok/s (63.1 ms/step)**, commit `fc36ee1`, re-measured in
  `decode-25tok-peak-bandwidth/implementation_notes.md` §16.1 with spread 0.03 tok/s.
- **GEMV floor: 58.3 ms = 17.2 tok/s** with zero overhead, from the 11.834 GB/token
  weight budget at 203 GB/s best measured 3 bpw bandwidth.
- **10 dead ends formally closed** — do not retry any of them:
  lane-contiguous repack (impossible), 4 bpw (−6.7%), split-K>2 (negative),
  4-col (−4.2%), uint4 loads (−3.8%), simd_shuffle (−18.8%), remove dequant (−26.7%),
  megakernel (no grid barrier), elementwise fusion (actual 0.8ms not 3-4ms),
  1-col streaming (−0.16 tok/s).
- **ICB tried and failed** (per user report, not in this repo's commits).

### The two novel ideas in this spec

1. **o_proj at 100 GB/s is at HALF the bandwidth of every other projection.**
   Root cause: K=8192 is too short for the DRAM prefetcher to warm up before the
   GEMV ends. Microbenchmark confirms: K=8192 → 120.7 GB/s, K=21504 → 199.3 GB/s.
   The fix: overlap o_proj weight prefetching with the preceding attention phase,
   so the prefetcher is already warm when o_proj starts computing.

2. **Cross-layer norm fusion** saves ~118 dispatches (59 layer boundaries × 2) by
   fusing layer L's finalize+norm+residual with layer L+1's input_norm+transform
   into one dispatch.

---

## 1. Commands

```bash
# GPU state
scripts/gpuq status

# Baseline measurement
M=~/.kaplumba/models/Gxma-repacked
scripts/gpuq try -- ./target/release/kaplumba --model "$M" \
  --prompt "Count from 1 to 100: 1, 2, 3," --max-new-tokens 64 \
  --temperature 0 --metrics

# Per-kernel profile
KAPLUMBA_DECODE_PROFILE=kernel scripts/gpuq try -- ./target/release/kaplumba \
  --model "$M" --prompt "Count from 1 to 100: 1, 2, 3," \
  --max-new-tokens 32 --temperature 0 --metrics

# Full test suite
scripts/gpuq try -- cargo test --test live_kernel_bit_exact
scripts/gpuq try -- cargo test --test fused_bit_exact
scripts/gpuq try -- cargo test --test repacked_golden
scripts/gpuq try -- cargo test --test argmax_engine_parity
scripts/gpuq try -- cargo test --test norm_rope_qkv_bit_exact
```

---

## 1a. Task T2a RESULT: ❌ NEGATIVE. The spin-wait saturates the memory pipe.

T2a was implemented in full and removed again. The measurement is in
`crates/engine/tests/warmstart_starvation.rs`; this is the reading of it.

**What was built.** `linear_gemv_ishizuki_k3_colmajor_2col_sk2_warmstart`
(`linear_repacked.metal`), which is `colmajor_2col_sk2` with the first 2 unrolled
K-steps' weight tiles read into registers before anything touches `x`, then a
spin on `device atomic_uint attn_done` before the first `x` read. Plus
`linear_attn_done_signal` (one thread, stores the sequence number), the encoders
`encode_gemv_raw_sk2_warmstart` / `encode_attn_done_signal`, `attn_done` +
`attn_stall` scratch buffers, a process-monotonic sequence counter, and the
`model.rs` wiring that dispatches o_proj ahead of attention. Correctness
restructuring was careful: the wait is unconditional and sits ahead of *every*
`x` read, the arithmetic after it is in the same order as the plain kernel (so
bit-exact, not merely close), and the spin is bounded with a device-side stall
latch so a starved producer is a diagnosable wrong answer rather than a hang.

**It compiled, loaded, and died at load:**
`invalid layer configuration: Metal command buffer failed during warmup decode`.

**Why.** The first test run hung for 900 s. That was not a slow test, it was the
finding: at the production grid the spinner does not converge. Lowering the spin
bound to 2e6 iterations so the test returns a number in seconds:

| spinning threadgroups (x 256 thr) | producer ran | spin timed out | wall |
|---:|---|---|---:|
| 1 | **false** | false | 510 ms |
| 8 | true | **true** | 1,019 ms |
| 32 | true | **true** | 2,169 ms |
| 64 | true | **true** | 4,102 ms |
| 128 | true | **true** | 7,565 ms |
| 168 | false | **true** | 7,285 ms |
| **336 (production)** | true | **true** | **18,600 ms** |
| 672 | true | **true** | 35,259 ms |

Two numbers matter more than the table.

1. **18,600 ms for 2e6 spin iterations across 336 groups is ~9.3 us per
   iteration.** A relaxed atomic load from L2 is ~ns. So the spinners are not
   waiting on the flag; they are saturating the memory pipe that both they and
   the producer need. 336 x 256 = 86,016 threads re-reading one 4-byte line is
   enough to do it.
2. **Wall time is linear in spinner count** (1.0 s at 8 -> 18.6 s at 336 -> 35.3
   at 672). The overlap is not overlapping: each extra spinner costs its own
   wait. That is the opposite of what a prefetch-and-overlap is supposed to do.

**This refines T1 rather than contradicting it.** T1 measured that the flag *does*
get set, which is true and still true -- at 336 groups the producer ran. T1's
test used 1 group of 256 and 1 of 32, i.e. 0.4% of the machine, so it had no way
to see the saturation, and its result (`spin_count=0`, flag already set) is
equally consistent with the producer having simply run first. **A feasibility
probe at 0.4% occupancy does not establish feasibility at 100%**, and the two
answers differ. T1's gate was therefore answered but under-specified.

**Do not retry the spin.** It is not a tuning problem: the mechanism spends the
resource the overlap was supposed to save. `tasks.md` T2b is the spec's own
answer to this — an `MTLEvent` is a hardware wait, so it enforces the same
producer/consumer ordering with *no thread spinning and no memory traffic*, and
it is the only variant of this idea left worth building.

**Metal has no acquire/release for a plain `device atomic_uint`.** Worth
recording because it cost three compile cycles: `atomic_store_explicit`'s
non-relaxed orders exist only on the scope-templated `device _atomic<T, S>`
overloads, which a plain `device atomic_uint*` does not match, so
`memory_order_seq_cst` is a *compile error*, not a slower path. Relaxed is the
only spelling available. It would have been sufficient here (the producer and
consumer are in different dispatches, so Metal's inter-dispatch visibility
carries the ordering) -- but that reasoning is only sound because of the
dispatch boundary, which the spin design was at least honest about.

---

## 2a. Inherited-work triage (session start)

Picking up an uncommitted tree. Four things arrived, and each needed a decision
before any optimisation could start.

### 2a.1 `warmstart_starvation.rs` had to be gated before anything else

Both of its tests dispatch the production `o_proj` grid of 336 x 256 = **86,016
threads re-reading one 4-byte line in a spin loop**, and the sweep goes to 672
groups = 172,032. That pegs every compute unit on the box; running it makes the
machine unusable, which is exactly what happened to the previous agent's
session. Both are now `#[ignore = ...]` with the explicit invocation
(`cargo test --release --test warmstart_starvation -- --ignored --nocapture`) in
the doc header, and the reason the file is kept at all — it is the evidence that
closed T2a — is recorded next to the gate.

Note that the gate test **cannot pass on this GPU**, and that is its finding
rather than a defect: at the production grid the spin times out (§1a table), so
`assert!(!timed_out)` is false by construction. It is kept as a measurement, and
`#[ignore]` is the correct treatment for a test whose verdict is "this does not
work".

### 2a.2 `concurrent_stream_economics.rs` was failing ~50% of the time

Inherited failure, unrelated to the pending work. The last assertion is a
non-vacuity self-check: "the unsabotaged `serial / concurrent` must sit strictly
inside the band, or `> 0.85` is unreachable rather than discriminating", with the
far edge hardcoded at **1.05**.

Twelve consecutive invocations (each already a median of 7 interleaved arms):

```
0.974 0.986 0.987 0.987 1.006 1.014 1.020 1.029 1.058 1.066 1.070 1.083 1.091
```

median **1.02**, range **0.12** wide, **5 of 12 above 1.05**. The ratio's true
value is indistinguishable from 1 — which *is* the finding that closed T2b — and
a ceiling at 1.05 sat inside the noise, so the test failed on coin flips.

**Changed the far edge to 1.20**, and this is a measured change rather than a
loosened one:

- the conclusion is untouched: concurrent is a wash against serial, which is
  what closed T2b;
- 1.20 still discriminates *by construction*: the largest `serial / concurrent`
  this harness can report is `serial / floor` = **1.67**, so a genuine win from
  overlap has to land in 1.20…1.67;
- the band is now as wide as this box can resolve, which is the honest tolerance
  for a ratio measured at 1.02 ± 0.06.

**Sabotage evidence for the new edge (AGENTS.md §3).** Both directions were
broken on purpose and observed to fail:

| sabotage | resulting ratio | observed |
|---|---|---|
| `/ 1.35` (pretend concurrent is 35% faster) | 0.69 | trips `serial_over_concurrent > 0.85` |
| `* 1.35` (pretend concurrent is 35% slower) | 1.415 | trips `serial_over_concurrent < 1.20` |

Note that the lower sabotage trips the *original* finding assertion, which is
the one that matters: it says "if overlap really paid, this test would have
caught it". A test whose ratio assertions cannot fail is the exact defect this
repo has shipped before (`spec/engine-performance` Q11), so both were checked.

### 2a.3 The same test's *other* absolute threshold was flaky too

Fixing the first one exposed a second instance of the same defect, which is why
it is worth writing down rather than patching silently: the sabotage check
`cheap_concurrent <= 1.2` (an overlapping pair whose second stream does 1/10 the
work must land well under the finding's 1.2 bound) failed **1 run in 12** on an
absolute comparison. Twelve consecutive runs of the sabotage arm:

```
0.867 0.918 0.992 1.009 1.012 1.014 1.032 1.040 1.068 1.078 1.088 1.244
```

median 1.02, one excursion to 1.244. **"10x lighter" is not "free"**: the lighter
stream still pays a dispatch and an `MTLEvent` wait, and the heavier one still
saturates the pipe, so the ratio lands near 1.0 rather than the ~0.5 the
original comment guessed.

**Changed it to a paired comparison** — sabotage against the unsabotaged number
measured in the *same invocation* — with a bound of 0.90. This is the same
reason every other conclusion in the file is a ratio: a box running 15% fast or
slow moves both arms together and cancels. Ten consecutive paired figures:

```
0.462 0.579 0.585 0.587 0.590 0.600 0.604 0.609 0.613 0.625
```

median **0.60**, max 0.625 — a 44% margin below the 0.90 bound, where the
absolute form had none. **Sabotage-verified**: pointing the "cheap" stream at the
same work as the real one (`concurrent_scaled(1)` instead of `(10)`) produced a
paired figure of **1.093** and tripped the assertion, so it is reading a real
measurement.

**The transferable lesson, and it is the second time this repo has hit it:**
*do not compare two independently-measured ratios against a shared absolute
constant when both ratios have a ±0.1 noise band.* Pair them. Eight consecutive
full runs of the test are now green.

---

## 3a. Task T3 RESULT: ✅ KEPT. Cross-layer norm fusion, +0.12 tok/s, bit-exact

`finalize_post_ffn_and_next_input_norm_transform` (R2). Fuses layer L's
`finalize_and_post_ffn_norm_scale_sk2` + residual add with layer L+1's
`input_norm_and_qkv_transform`, saving one dispatch and one 5372-element
buffer round-trip at each of the 59 internal layer boundaries.

### A/B, one binary, interleaved

`scripts/abbench` (new, this session — see §6), 64-token greedy generation,
identical config in both arms, `KAPLUMBA_NO_CROSS_LAYER=1` is the control:

| block | fusion on | fusion off | delta |
|---|---|---|---|
| A (n=8) | **15.996** (15.884-15.996, spread 0.112) | 15.876 (15.853-15.936) | **+0.120** |
| B (n=8) | **15.924** (15.887-15.924, spread 0.037) | 15.855 (15.829-15.880) | **+0.069** |
| pooled (n=16) | **15.960** | 15.866 | **+0.094** |

**Both blocks are positive and the two independent 8-run blocks agree in sign.**
The pooled +0.094 is above the 0.01-0.08 tok/s noise floor recorded in
`decode-25tok-peak-bandwidth` §16.1, so the fusion is kept. It is a small win —
as it should be: §18.1 measured *all* element kernels at only 1.52 ms/step, and
this removes roughly a third of them.

Greedy output was **SAME** in every arm, i.e. bit-identical to the baseline
captured before any of this work started (implementation notes §6.1).

### Bit-exactness, and two sabotages that were tried

`crates/engine/tests/cross_layer_fusion_bit_exact.rs`, 24 vectors × 5376
elements, `to_bits()` equality on all three outputs. Two sabotages, both
tripping it with a diagnosable message:

| sabotage | observed |
|---|---|
| layer L's `lid < 256u` guard widened to `lid < 1024u` | seed 0 residual stream: **5245 of 5376** differ, `0x3D44` vs `0x3C51` |
| layer L+1's accumulation stride `256u` → `255u` | seed 0 next-layer normed: **5101 of 5376** differ, `0x3BE0` vs `0x3BDE` |

### A third "sabotage" that was not one, and the trap it sets

The first attempt widened layer L+1's tree from `stride = 128` to `stride = 512`
— i.e. gave it layer L's 1024-thread shape. **It passed.** That is correct
behaviour, not a test weakness, and the reason is worth writing down because it
looks like a hole in the coverage:

Only threads `0..255` accumulate anything (the guard is `lid < 256u`), so
`partial[256..1023]` holds exact zeros, and the extra tree levels add `0.0f`,
which is a no-op in IEEE-754. It is the same reason layer L's 1024-thread
reduction is bit-identical to the 256-thread original it replaced
(`decode-25tok-peak-bandwidth` §17).

**The rule this gives:** a reduction sabotage must change *which values are
grouped*, not *how many zeros sit above the tree*. The first attempt changed
the second kind, so it had nothing to detect. This is recorded in the test
header so the next person does not spend the same cycle.

---

## 4. Task T2b RESULT: ❌ NEGATIVE. Two bandwidth-bound kernels cannot overlap.

T2b is the spec's own fallback for T2a, so it got built — as a measurement
first, because a dual-queue restructure of the decode path is invasive and
there was a cheap way to find out whether it could possibly pay.

**Test:** `crates/engine/tests/concurrent_stream_economics.rs`. Median of 7,
interleaved arms, each stream reading 402.7 MB from a single 768 MB allocation:

| arm | ms |
|---|---|
| half 0 alone | 0.610 |
| half 1 alone | 0.620 |
| serial (both, one queue) | 1.049 |
| concurrent (2 queues + `MTLEvent`) | 1.025 |
| perfect-hiding floor (`max` of the two alone) | 0.620 |

- `concurrent / floor = 1.555-1.654` → overlap hides **nothing**.
- `serial / concurrent = 0.930-1.023` → the hardware wait is a **wash**.

**The reason is arithmetic, not mechanics.** R1's premise is that `o_proj`'s
deficit is a *ramp* it never finishes (K=8192 → 120.7 GB/s vs K=21504 → 199.3
GB/s, `gemv_bandwidth.rs`), and that hiding the ramp against attention recovers
it. But attention streams the KV cache at the same order of magnitude
(16.8 MB vs `o_proj`'s 16.5 MB per layer), so both kernels are bandwidth-bound.
Bandwidth is shared and fixed, and the byte count is identical whether they
overlap or not. **There is no third resource to exploit.** Hiding a ramp requires
a partner that is *not* saturating the pipe, and attention is.

This also explains T2a's failure mechanistically rather than as an
implementation defect: the spin-wait version was trying to buy latency hiding
with the one resource (memory bandwidth) that both kernels already want.

**So R1 as written — "overlap `o_proj`'s prefetch with attention" — is not
achievable on this hardware by any mechanism.** T2a (spin) and T2b (hardware
wait) bracket the design space: the only two ways to have one dispatch wait on
another's output. Both measured negative, for the same underlying reason.

**Sabotage evidence (AC4).** Removing the second stream's dispatch entirely
drops `concurrent/floor` to 0.981 and trips both ratio assertions. The in-test
10×-lighter second stream trips at 1.125 against the 1.2 threshold, so the
threshold is tight enough to discriminate rather than passing vacuously. Two
earlier sabotages passed *incorrectly* — both edited `concurrent()`, which
delegates to `concurrent_scaled()` and was therefore dead code. Worth recording:
the sabotage passed because it changed nothing, and only reading the diff caught
it.

**Harness defects found and fixed en route, all recorded in the test's header.**
Two are worth keeping in mind for any future bandwidth work on this box:

1. **`StorageModeShared` measured 804 GB/s on a 273 GB/s wire.** The host had
   written the buffer recently enough that the pages stayed resident and partly
   cacheable. Private storage is mandatory for a bandwidth measurement whose data
   the host never reads back — which is the case for every production weight read.
2. **GPU clock ramp put the first arm at low clock.** Calibration brought 1134 →
   628 GB/s. Still 2.3× over the wire, so *absolute* bandwidth remains
   untrustworthy in this harness and the test asserts only on **ratios**, where
   the scaling error cancels. I did not identify the remaining cause; it does not
   affect the ratios, and it is stated in the test rather than papered over.
3. First version also allocated separate 192 MiB shared buffers per stream and
   **segfaulted** on teardown.

`serial >= sum(alone)` held after calibration, which is the self-consistency
check that these kernels are genuinely bandwidth-bound rather than launch-bound.

**What R1 would need instead.** Since bandwidth is the binding constraint and
both kernels want it, the only remaining lever is *bytes*, not overlap. The
options are the ten closed dead ends plus their opposites — fewer bytes per token
(repack, lower bpw) — not scheduling. That is outside this spec's scope
(`spec.md:23` forbids retrying the closed dead ends) and is where
`decode-3bpw-amplification` already recorded that 199 GB/s is the 3 bpw floor.

---

## 5. Task T1 Log: Concurrent Threadgroup Scheduling Feasibility

- **Test**: `crates/engine/tests/concurrent_tg_feasibility.rs`
- **Objective**: Verify whether Apple Metal compute command encoder schedules threadgroups from separate dispatches within the same command buffer concurrently without CPU intervention, or if dispatches serialize strictly.
- **Mechanism**:
  - Kernel A (`signal_after_work`): executes 100,000 iterations of FMA busy loop across 256 threads (~1 ms work), then issues `atomic_store_explicit(&flag, 1u, memory_order_relaxed)`.
  - Kernel B (`wait_and_mark`): dispatched in the same command buffer, 32 threads, spin-waits on `atomic_load_explicit(&flag, memory_order_relaxed) == 0u` up to 500M spins, writes `0xCAFE` on seeing flag `1u`, or `0xBADD` on timeout.
- **Measurement**:
  - `cargo test --test concurrent_tg_feasibility -- --nocapture`
  - Result: `Wall time: 8.2 ms`, `Marker: 0xCAFE`, `Kernel B ran: yes`, `Spin count: 0`.
  - Kernel B observed flag set with 0 GPU timeout and zero host stalls. Metal's hardware threadgroup scheduler on M4 Pro actively dispatches threadgroups across available compute units concurrently when resources allow.
- **Architectural implication**:
  - Validates **T2a (Warmstart GEMV kernel)**: we can dispatch `o_proj` weight loading concurrent with `attention`, having `o_proj` threadgroups prefetch weight tiles into registers/L2 before attention writes the final input activation `scratch.attended`.
  - Eliminates the need for complex dual-queue `MTLEvent` host synchronization for o_proj latency hiding.


---

## 6. Delegated agent run #1: DIED INFRASTRUCTURE, not out of work

A subagent was dispatched to take this spec to 32k context and ≥14 tok/s at16k
context. It terminated with:

```
<subagent state="error"> getaddrinfo ENOTFOUND opencode.ai </subagent>
```

**That is a DNS/transport failure reaching the agent API.** It is not a
completion, not a partial-completion marker, and not a statement about the task.
The agent did real work first and then could not be continued.

### What it completed before the transport died (3 commits)

| commit | content |
|---|---|
| `14570f8` | T0 triage: `warmstart_starvation.rs` gated to `#[ignore]`, `concurrent_stream_economics.rs` ratio thresholds de-flaked (§2a) |
| `40e018d` | sampler `top_k_order` O(n) partition, **14.71 → 1.89 ms/token** |
| `121ff5b` | T3 cross-layer norm fusion **measured: +0.09 tok/s, bit-exact → KEPT** (§3a) |

The sampler's 1.89 ms/token is better than the 2.78 ms measured here before
hand-off, so it found headroom this session had not.

### What it left unfinished, and the rule that follows

Uncommitted at the time of the failure:

- `crates/engine/tests/multi_turn_chat.rs` (521 lines, **untracked**)
- modifications to `crates/engine/src/chat.rs` and `crates/cli/src/chat.rs`
- a 0-byte scratch file `crates/engine/src/XXxi1lf1` (since deleted)

**Untracked work does not survive an agent failure.** It survived this one only
because the failure was visible in the same session. A failure across a session
boundary would have left a 521-line test file that no commit referenced and that
`git stash list` would never surface.

**Rule for every delegated run in this repo: commit continuously, and treat the
uncommitted working tree as lost.** The per-task commit discipline in
AGENTS.md §4.2 is written for humans, but it matters *more* for agents, whose
session can be terminated by transport at any point. This run happened to commit
between each unit of work and that is the only reason three units of work
survived instead of none.

### A blank commit is worse than no commit

The agent left the tree *building* (0 errors) but with an untracked test file.
An agent that resumes must therefore not assume the tree is clean: `git status`
first, and treat untracked `.rs` files under `crates/` as a work-in-progress
marker rather than as litter.

---

## Appendix: Parallel Track A: static audit of the 126 vs 199 GB/s gap

Static audit only (no GPU touched). All claims quoted from
`spec/decode-25tok-peak-bandwidth/implementation_notes.md` (§16-18),
`tasks.md` T5.10/T5.11, `crates/engine/tests/gemv_bandwidth.rs`,
`crates/kernels/src/linear.rs`, `crates/engine/src/model.rs`.

### A.1 Why production halves the microbenchmark: the three stated causes

**(a) The microbenchmark's own validation anchors are the disputed numbers.**
§16.3 says the microbenchmark "reproduces" production: *"`colmajor_2col_sk2` at K=21504, N=5376 measures **199.3 GB/s**, against 200 GB/s derived from production for `down_proj`. And at K=8192 it measures 120.7 GB/s, against ~100-120 for `o_proj`. It reproduces both production anchors, so its *absolute* numbers are trustworthy."* But the §16.2 "200 GB/s" anchor comes from the `own_ms` profiler table, and §16.2 itself confesses: *"Calibrating it against `head/gemv` ... gives a scale factor of 1.71 between its `own_ms` and real ms."* §18.2, from the distortion-free ablation harness, restates the same projections at `gate_up 129 GB/s`, `down 126 GB/s`, `qkv 125 GB/s`. The §16.3 "reproduction" compares against the miscalibrated profiler numbers; the ablation says production is really ~126. The microbenchmark's ~199 only "matches production" if production is 200 — which §18.2 says it is not. This internal inconsistency is never reconciled in the notes.

**(b) The microbenchmark does not see production's working set.** §16.3:
*"Production streams 10.78 GB of distinct weights per token; re-reading one matrix repeatedly can be served from L2 and reports bandwidth the real workload never sees."* And §16.3 again: *"The isolated benchmark re-reads a small set of buffers back to back, which is not the interleaved 10.78 GB stream production sees."* The test code confirms: `MAX_WORKING_SET = 768 << 20`, `MIN_WORKING_SET = 192 << 20`, `nbuf` clamped to `(2, 8)` — at most ~768 MiB of distinct packed weights, re-read for tens of ms. Production touches 10.777 GB of distinct weight bytes per token (§16.4's own byte count). The rotating-buffer fix in the microbench shrinks L2 reuse but does not approach the 10.78 GB production stream, so a ~73 GB/s L2/streaming delta is fully plausible.

**(c) The microbenchmark inverts variant rankings — it does not transfer.** §16.3:
*"The microbenchmark says sk4 at K=8192 is 52% faster than sk2 and that no-split-K is 41% faster; production says the opposite for both ... Confirmed by profiling `o_proj` directly: production `o_proj` own_ms is **172.9 on sk2 and 211.1 on sk4**, while the microbenchmark had sk4 far ahead."* §18.2: *"...the layer GEMVs are running at 62% of the measured peak (126/203) — inefficient for decode, but changing variants hasn't moved them (T5.11 exhausted all seven SK/no-SK/grid-capping combinations; microbenchmark's variant ranking is inverted relative to production)."* T5.11 table adds: *"Capping the grid is monotonically worse ... kills the 'fixed per-threadgroup cost at short K' explanation."* So scheduling is exhausted; the 7 scheduling variants all failed in production.

### A.2 Threadgroup-count arithmetic (does occupancy starve production?)

Dispatch equations from `linear.rs`:
- `encode_gemv_raw_sk2`: `groups = ((out_features / 16) / 2) * 2` → `groups * 256` threads, 256/TG.
- `encode_gemv_raw_sk4`: `pairs = (out/16)/2; groups = pairs * 4`.
- `encode_gemv_raw` 2col branch: `pairs = out_features / 32; groups = pairs.min(gemv_max_groups())` (u64::MAX default → uncapped).
- `encode_gemv_fused2_mlp` / `encode_gemv_fused2` (2col branch): `groups = (tiles_a + tiles_b) / 2`.

Production decode shapes (hidden 5376, q 8192, k/v 4096, intermediate 21504):

| projection | path | TG equation | TGs launched | TGs per 20 cores |
|---|---|---|---|---|
| o_proj (K=8192, N=5376) | `encode_gemv_raw_sk2` | (5376/16/2)·2 | **336** | 16.8 |
| down (K=21504, N=5376) | `encode_gemv_raw_sk2` | (5376/16/2)·2 | **336** | 16.8 |
| qkv (q N=8192 + k N=4096) | `encode_gemv_fused2` 2col | (8192/16 + 4096/16)/2 | **384** | 19.2 |
| gate_up (N=21504 ×2) | `encode_gemv_fused2_mlp` 2col | (21504/16 + 21504/16)/2 | **1344** | 67.2 |
| head (N=262144) | `encode_gemv_raw` 2col | 262144/32 | **8192** | 409.6 |

Reading: o_proj/down launch only 336 TGs; T5.11 already showed capping below that (168, 84) is monotonically *worse*, and SK4 (672 TGs) does not help in production (`15.781 vs 15.816`). gate_up's 1344 TGs ≈ 8.4× more, yet it runs at the same ~129 GB/s. Head launches 8192 TGs and reaches 253 GB/s — but it is 6 bpw (192 B/tile vs 96 B/tile at 3 bpw, §16.4), so its better GB/s is attributed to tile size, not TG count alone. Verdict: TG count/occupancy arithmetic is a real constraint for o_proj/down (336 TGs ≈ 2 waves of ~160 on 20 cores), but it does **not** explain why gate_up at 1344 TGs also sits at 129 GB/s, nor why all 7 variants fail. Per-TG streaming efficiency, not TG count, is the limiter for the long-K GEMVs.

### A.3 If all seven variants failed: what the 199 GB/s can and cannot mean

T5.11 + §18.2: every scheduling knob (SK2/SK4/no-SK/contig4/1col/grid-capping) reproduces ~126–150 GB/s in production. Therefore the 199 GB/s from `gemv_bandwidth.rs` is an artifact of the harness conditions, not a transferable property of the kernel+shape. The two remaining candidate real mechanisms:

1. **Working-set / cache effect (the microbenchmark's rotating buffers hide it).** §16.3 itself: the microbench rotates over ≤768 MiB while production streams 10.78 GB; L2 on M4 Pro serves part of any re-read ≤ last-level cache. The microbench's 199 could be "what DRAM gives when the same ~175 MiB cycle through" — not what a cold 10.78 GB stream delivers.
2. **Production-side serialization / dispatch-gap accumulation.** §18.2: *"Dispatch gaps cost 0.7 us each (T5.10), so removing ~180 dispatches buys +0.031 tok/s — dispatch count is not the lever."* Breakdown of that arithmetic: 180 dispatches × 0.7 us ≈ 126 us ≈ 0.126 ms against a 63.1 ms step — negligible, and both the microbench (`set_max_dispatches(u32::MAX)`) and production (`set_max_dispatches(2500)`, model.rs:3820) now run one rolling command buffer, so MAX_DISPATCHES_PER_BUFFER=50 chunking (T5.10-era) no longer applies. This mechanism is largely already excluded by the notes; encoding more bubbles between GEMVs (0.7 us each) cannot account for 40.34 ms vs 26 ms on gate_up.

So the only candidate with a plausible 73 GB/s (57%) scale is a memory-system effect — L2 residency, DRAM page/bank behavior across the true 10.78 GB interleaved stream, or clock behavior — that the ≤768 MiB microbench working set does not reproduce.

### A.4 One decisive experiment (needs the GPU; do NOT run while the other agent holds it)

Modify `crates/engine/tests/gemv_bandwidth.rs` reproduction (do not commit): at the exact down_proj anchor `colmajor_2col_sk2, K=21504, N=5376, rate 3`, sweep **one variable only** — the rotation span `MAX_WORKING_SET` — and hold everything else at the §16.3 conditions (GPU clock timing, `set_max_dispatches(u32::MAX)`, same TARGET_MS):

- run A: current `768 MiB` cap (expect ~199 GB/s, reproducing §16.3);
- run B: cap raised so each rep touches a distinct buffer and the total working set is ≥ 10.78 GB (i.e. ~60 buffers × 179.6 MiB/layer-shaped, cycled once per pass) — expect ~126 GB/s if mechanism 1 is real.

Decision rule: if B collapses to ~126, the gap is a working-set/cache artifact and production's 126 is the true DRAM rate for this kernel — the 199 was never achievable in production and §16.4's 17.2 tok/s ceiling stands as *optimistic*, worth re-deriving from B. If B stays ~199, the loss is elsewhere in production and the next discriminator is inserting production-like inter-kernel dispatch spacing (and per-layer interleaving of qkv/o/gate_up/down buffers at real offsets) into the microbench.

Current microbench allocation (`run` in gemv_bandwidth.rs:124-135): `nbuf = (MAX_WORKING_SET / shape_bytes).clamp(2, 8)` and each buffer is `shape_bytes/2` bytes — for down_proj that is nbuf ≈ 8 × 21.7 MB ≈ 173 MiB. Run B needs only this constant changed (and the `clamp(2, 8)` raised).

---

## 7. KV pre-size reverted by user decision (2026-10-02)

`6f7b393` pre-sized the 10 full-attention caches to 32768 positions
(2.68 GiB resident up front), which removed the one-time stall of each
`ensure_capacity` doubling (512→…→16384; each copy is live keys+values).
Measured: 8-token decode window at 8.4k context 8.8→10.1 tok/s, 128-token
window 11.8→12.6, 16k 8.4→10.3.

The user directed that chat not pre-allocate the full context (CLI footprint
was reported at ~13.7 GiB), so `894f4c0` reverts `6f7b393` and the cache
is again allocated dynamically (double on demand, ring for sliding layers).
Post-revert footprint at 1.2k prompt / 32 tokens, `/usr/bin/time -l`:
`peak memory footprint = 13 024 650 984` (~12.1 GiB), i.e. the pre-size
added roughly 1.5–2 GiB of always-resident KV, and removing it returns to
the historical ~12–13 GiB.

Consequence, stated plainly: the one-time doubling stall during prefill of
a prompt longer than 512/1024/2048/4096 is back. It is a fixed cost per
doubling, not per-token, and it does not change decode steady-state speed.
`DEFAULT_CONTEXT_LIMIT` stays 32768; only the allocation policy changed.
