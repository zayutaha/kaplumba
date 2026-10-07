# implementation_notes — `spec/decode-3bpw-gap-19t`

Append as work proceeds, newest at the bottom. **No implementation has
started.** This entry is the discovery pass, and it is written down because
three of its findings changed the previous spec's conclusions.

## 0. Why this spec exists

`spec/decode-alu-fuse-18t` set a ceiling of ~17.2 tok/s and attributed the
remaining gap to a bit-locked integer ALU pool. That attribution came from an
instrument that could not work. With the instrument repaired, the gap is
kernel-local and open, and the ceiling is higher than that spec concluded.

## 1. The measurement that overturned it

`spec/alu-boost/findings.md §4` reported `p2_int*` scaling as n²
(4→26.7 ms, 8→106.8, 16→428.6, 32→1759) and correctly declared the whole
integer half of that probe untrustworthy. Two independent agents since cited
"the int pipe is unmeasurable" as a closed question.

The defect was in the generated body, `crates/engine/examples/alu_boost.rs`
`p2_kind()`:

```metal
su[c] = su[c] * 0xCBAC1FEDu;                    // (1)
su[c] = 0x3b603b60u ^ (su[c] & 0x8fff8fffu);     // (2) needs (1)'s result
```

A **dependent** pair: one chain, two ops, serialised. So the arm measured
`chain_latency × chains`, which is why it was super-linear. Nothing about the
hardware was being measured.

**The first repair failed, and the gate I added is what caught it.** Splitting
into two operand arrays (`su` and `sv`) does break the dependency, but 16
chains × 2 live `uint`s spills, and the arms scaled as n² again at **5.7 G
warp-instr/s flat across 4→32 chains**. Flat rate-per-instruction while total
instructions quadruple is the signature of local-memory traffic — and spilling
looks exactly like a slow machine.

**The working repair:** two separate `#pragma unroll` loops over the *same*
array.

```metal
su[c] = su[c] * 0xCBAC1FEDu;                 // loop 1: all chains independent
su[c] = 0x3b603b60u ^ (su[c] & 0x8fff8fffu);  // loop 2: all chains independent
```

Within each loop every chain is independent — that is the ILP the probe is
for — and only `nin` registers are live at a time. Two loops over one array
beat two arrays.

Also fixed: `instrs_per_iter` for the int arms was `chains * 2.0`; the body is
3 ops per chain (`mul`, `lop3`, and the join), so the old count would have
**overstated every int rate by 50%** the moment the dependency was fixed.

| pipe | rate | linearity |
|---|---:|---|
| `p2_int16` | **91.0 G warp-instr/s** | 4→8 `1.02×`, 8→16 `1.01×`, 16→32 `1.00×` |
| `p2_fp16` | 118.7 G warp-instr/s | as before |

`linearity` is a **printed gate**, not a comment: if `r16/r8 < 0.7` the probe
prints `VERDICT: integer arms still super-linear -- ignore every integer
number above`. It fired on the first repair, which is the only reason the
second one exists. This is `AGENTS.md §3.4` applied to an instrument rather
than to a test, and it is the third time in this project that a probe's own
inputs or body have produced a number that could not fail.

## 2. What the repaired number refutes

Layer GEMVs execute **169 G warp-instr/s** against a **91 G** integer pipe
ceiling — 186% of the ceiling. An issue-bound kernel cannot exceed its issue
ceiling, so the layers are **not** integer-issue bound. Therefore:

* the "3 bpw is ALU-bound at 9.375 instr/code" premise is false;
* the "bit-locked `decode_state_mcg` pool" is not the binding constraint;
* the ~30% micro-vs-production bandwidth gap is kernel-local and attackable.

What the repaired probe does **not** show is that the layers are
*bandwidth*-bound. It removes one candidate and leaves the rest open. The next
question is which of parallelism / contiguity / K / bits-per-code it is, and
the tree already contains the experiment.

## 3. The experiment already exists and has never been run

`crates/engine/tests/perf_probe_head_vs_layers.rs` (≈1200 lines) poses exactly
this question with four candidates — (a) parallelism, (b) contiguity, (c) K,
(d) bits-per-code — and one control per candidate:

* **Control R** `head_shape_at_three_bpw` (`:1098`) — head shape, kernel
  family, grid and row-major layout, at 3 bpw. **One variable.** Pre-registered
  criterion at `:1137`: `(bw_ratio - 0.5).abs() < 0.10`.
* **Control S** `layer_shape_at_six_bpw` (`:1152`) — `down_proj`'s shape at 6
  bpw, via the existing `linear_gemv_ishizuki_k6_colmajor_contig4_bench`
  (`linear_repacked.metal:3815`).
* **Control G** `head_grid_thickness_sweep` — head at 6 bpw, grid forced
  16384 → 336 groups.

**It has never been run.** `grep -rn bw_ratio spec/` returns nothing. Same
failure class as `alu_boost.rs`: a complete instrument, cited by nobody,
absent from the branch that needed it. Per `AGENTS.md §7`, a path was cited
before it was checked.

The kernel it needs at rate 6 exists because someone anticipated this exact
comparison — the comment block above the `_k6` bench kernel derives why.

## 4. The production numbers this rests on

`KAPLUMBA_DECODE_PROFILE=kernel`, `Gxma-repacked`, AC1 prompt (29 tokens),
`--max-new-tokens 128 --temperature 0`, `own_ms` per token, bytes from the
safetensors header count (exact to 0 B, `crux.md §1.2`):

| kernel | GB/token | own_ms | GB/s |
|---|---:|---:|---:|
| `gate_up_gelu` | 5.2022 | 25.609 | 203.1 |
| `down_proj` | 2.6011 | 13.090 | 198.7 |
| `qkv_proj` | 1.2386 | 8.651 | **143.2** |
| `qk_proj` | 0.3716 | 1.961 | 189.5 |
| `o_proj` | 1.1561 | 6.246 | 185.1 |
| `head/gemv` | 1.0570 | 4.152 | **254.6** |
| `attention` | (KV) | 2.163 | — |
| 9 element kernels | — | 1.95 | — |

Unprofiled step, 3 runs: **15.808 / 15.809 / 15.837 tok/s** — spread 0.03, a
clean baseline.

## 5. Two numbers that look like headroom and are not

* **The 8.96 ms "gap" between span and own.** `dp_span_ms_per_token` 72.783 vs
  `dpk_own_ms_per_token` 63.819 reads as 12.3% of the step unaccounted for. It
  is not: kernel `own_ms` sums to **63.86 ms** against a **63.29 ms** measured
  step. There is nothing to recover, and the span/own split misleads.
* **Dispatch.** `KAPLUMBA_DECODE_MAX_DISPATCHES_PER_BUFFER` = 50 / 250 / 2500 →
  15.93 / 15.88 tok/s. Identical, and consistent with `MEMORY.md`'s ~3.5 µs
  of latency for ~0 of work. Re-confirmed today rather than inherited.

## 6. Commands

```bash
scripts/gpuq status                      # must be idle before ANY rate
M=~/.kaplumba/models/Gxma-repacked

# T1, the gate (model files read; no second model loaded)
KAPLUMBA_TEST_MODEL="$M" scripts/gpuq try -- cargo test --release \
  -p kaplumba-engine --test perf_probe_head_vs_layers -- --ignored --nocapture

# the repaired int-pipe ceiling, model-free, ~40 s
cargo build --release -p kaplumba-engine --example alu_boost
scripts/gpuq try -- ./target/release/examples/alu_boost

# production attribution
KAPLUMBA_DECODE_PROFILE=kernel scripts/gpuq try -- \
  ./target/release/kaplumba --model "$M" \
  --prompt "Explain the theory of relativity in simple terms, and then explain why it matters." \
  --max-new-tokens 128 --temperature 0 --metrics

# AC1 throughput
scripts/gpuq try -- ./target/release/kaplumba --model "$M" \
  --prompt "<AC1 prompt>" --max-new-tokens 128 --temperature 0 --metrics
```

## 7. Environment at T0

`scripts/gpuq status` reported a model loaded by the 16k-chat agent
(`pid 99002`, 0.1 GiB reclaimable). **T0/T1 were therefore not run**: a
bandwidth probe allocating hundreds of MiB would evict its pages, and
`crux.md §2.5` records exactly that voiding two runs (own control −29%,
interleaved control +7.8%). Compute-bound work continues; measurement waits.

## 8. Log

(append per task: command, median ± spread, control drift, verdict, commit)
## 9. T0 — box ceiling re-confirmed (2026-10-06)

`crux_bandwidth`, GPU free after the chat agent was cancelled:

```
best median across every arm : 254.5 GB/s   (93.2% of wire)
top-half median of usable    : 249.4 GB/s   <- the number to use  (91.4%)
median of all arm medians    : 239.5 GB/s   (87.7%)
CONTROL_repeat drift         : -0.0%, noise floor +0.03%
PhysMem end: 11.0 GB unused, 0 kaplumba-shaped processes
```

Consistent with the historical 245–253.6 GB/s (`crux.md §2.6`), so the box
is in the state every rate in this spec assumes. T1 may proceed.

## 10. T1 — the gate: Control R (head shape at 3 bpw)

Pre-registered decision rule, restated before the run: `bw_ratio ≈ 0.5` →
bits-per-code is the whole story (T5 is the work); `bw_ratio ≈ 1.0` → the
head's advantage is shape (T5 dead).

## 11. T1 — Control R: six defects, then a physically impossible number

The probe has now been *run*. It had never been executed, and it was broken in
six independent ways. All six are in commit `d765774`.

| # | defect | symptom | fix |
|---|---|---|---|
| 1 | `assert_live` read `scratch.output` as `u16` and ran `half::f16::from_bits` on each half | reinterpreted f32 as fp16; a 5-bit all-ones exponent hits ~1/32 per half, so with 262144 outputs `finite=false` **always** | read as `f32` |
| 2 | `Scratch::new` sized `output: n * 2` (fp16) while every epilogue writes `device float *y` | half the buffer the kernel writes | `n * 4`; `raw`/`raw_b` two lines above already used `* 2 * 4`, which is what made it visible |
| 3 | `dispatch_row` used `row.bits()` (declared rate) while `load_row` overrode the buffer's rate | `decodes 3 bpw but the request is 6 bpw` — the control **could not run at all** | `Loaded.bits`, the overridden rate |
| 4 | kernel name built as `linear_gemv_ishizuki_k6` | that kernel does not exist; the lookup found nothing and the "timed" dispatch did nothing | `..._k6_colmajor_contig4_bench`, and the 3 bpw arm named `_k3_colmajor_contig4` |
| 5 | `Row::bytes()` used `shape.weight_bytes()`, i.e. the declared rate | 3 bpw arm credited 6 bpw bytes → **411.4 GB/s = 1.51x the 273 GB/s pin rate** | `Row::loaded_bits` + `weight_bytes_at(shape, bits)` |
| 6 | liveness checked `output` for `Route::Named` rows, which accumulate into `raw` and never finalize | "half written" — actually `output` was never the destination | `Scratch::liveness_buffer(row)` |

Defect 6 was the expensive one: it produced a symptom indistinguishable from a
kernel defect, and I chased the kernel for two rounds before isolating it with
`probe_k6_half_output.rs`, which poisons a buffer with `-1.0e30` and dispatches
`_k6_colmajor_contig4_bench` directly:

```
k6 direct: n=262144 groups=16384 written=262144 nonzero=262144
first untouched index: None      last written index: Some(262143)
```

**The kernel covers every output slot.** All four symptoms were the harness.

### The one thing still wrong, handed to an agent

```
ctl:head @6bpw   4.114 ms   256.9 GB/s   342.5 Gcodes/s   B/code=0.750
ctl:head @3bpw   2.569 ms   411.4 GB/s   548.6 Gcodes/s   B/code=0.375
Control R: bw_ratio 1.602   code_ratio 3.203
```

`B/code` is now correct and the 6 bpw arm (256.9 GB/s) sits exactly where
`crux.md` says the head should be. But **411.4 GB/s exceeds the wire**, so the
3 bpw arm is still not measuring what it claims.

Leading hypothesis, **unverified**: `_k3_colmajor_contig4` strides
`WORDS = (256*3)/32 = 24` u32 = 96 B/tile while
`_k6_colmajor_contig4_bench` strides `48` u32 = 192 B/tile, and lm_head is
**row-major** (`perm_mode` 0) while both dispatched kernels are the **colmajor**
ones. If the production head does not use a colmajor kernel, both arms of
Control R are measuring the wrong kernel and the control's premise is wrong —
which would be a finding, not a failure.

Decided by measurement, not argument: poison the buffer, dispatch, and map
which words the kernel consumed. That answers "what does it actually read"
without an arithmetic dispute.

## 12. Self-correction: a 2-point fit I nearly built a plan on

Separating each measured shape into `streaming + fixed per-dispatch` costs two
points per kernel family. Fitting fused2 on `qkv` and `gate_up` gave a tidy
`244.0 GB/s + 0.0715 ms/dispatch`, predicting both inputs *exactly* — as it
must, being a 2-equation solve for 2 parameters — and implying
**4.29 ms/step** of recoverable fixed cost, +1.15 tok/s.

`qk_proj` is a third fused2 shape. It is off by **+8.4%**, so a single fixed
cost does not describe the family. **The +1.15 tok/s is not established and no
task was built on it.** Recorded because the failure mode is seductive: two
points, two parameters, a perfect fit, and a physically plausible number.

## 13. T2 — the gap is NOT a kernel defect, and my headline number was wrong

The agent's one-variable pair (row-major `k3` vs `k6`, same grid, same
addressing, same 3 words/lane, same shape — **these are what production runs**):

| | run 1 | run 2 | mean |
|---|---:|---:|---:|
| 6 bpw | 4.275 ms, **247.2 GB/s** | 4.235 ms, 249.6 GB/s | 248.4 |
| 3 bpw | 3.210 ms, **164.7 GB/s** | 3.190 ms, 165.6 GB/s | 165.2 |
| `bw_ratio` | **0.666** | **0.664** | 0.665 |

Halving the bit rate at fixed request count cuts runtime **24.8%, not 50%**.
Neither pre-registered reading survives: not `(d)` bits-per-code (0.5), and not
the pure-shape reading (1.0). **~75% of the 6 bpw runtime is rate-independent
per-code work.**

### 🔴 RETRACTION — the +3.63 tok/s I quoted was arithmetic on the wrong rate

I told the user, more than once, that the layer GEMVs run at 194 GB/s where the
head runs 254.6, and that closing that was worth **+3.63 tok/s to 19.43**.

**254.6 GB/s is the SIX bpw rate.** It is not a 3 bpw ceiling, and the layers are
3 bpw. At 3 bpw this box, this kernel family, this grid, reaches **165.2 GB/s**.
The measured layer shapes run **185–203 GB/s — already above the 3 bpw
reference.** The layers are not underperforming and the +3.63 does not exist.

The error was comparing a 3 bpw measurement against a 6 bpw one, which is the
same class as the `alu_boost` n² arms and the four `crux.md` byte-count
disagreements: a number quoted without checking which configuration produced it.

### Corrected 16k budget (measured inputs only)

`attention` achieves 256 GB/s (0.5533 GB / 2.1625 ms), so at 16k its 2.181 GB
costs **8.52 ms**. With head 4.15 ms and elementwise 1.95 ms:

| layers at | step | tok/s @16k |
|---|---:|---:|
| 165.2 (3 bpw reference) | 81.22 ms | 12.31 |
| 185.1 (measured low) | 74.06 ms | 13.50 |
| **194.0 (measured mean)** | **71.33 ms** | **14.02** |
| 203.0 (measured high) | 68.82 ms | 14.53 |

Measured single-shot at 16,484 tokens today: **13.45–13.51 tok/s** (3 runs),
i.e. the layers are sitting near the bottom of their own 185–203 range.

**14.0 tok/s at 16k needs the layers at 193.7 GB/s** — inside the measured
spread. So the target is *not* excluded by the corrected physics; it is a
~1 ms/step question about which end of that range the layers occupy. That is a
far smaller and more tractable ask than the +3.63 I described.
