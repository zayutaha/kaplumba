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