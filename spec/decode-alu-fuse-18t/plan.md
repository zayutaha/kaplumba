# plan — `spec/decode-alu-fuse-18t`

How: technical design, touchpoints, sequencing, risks. Every path below was
read in this session before being written down; §5a of `spec.md` lists what
changed as a result.

---

## 0. Ground rules for every measurement in this plan

* `scripts/gpuq status` before any model run; `scripts/gpuq try --` (exit `75`
  = busy, do other work, retry). Never `run` when something else is useful.
* Worktree + its own `CARGO_TARGET_DIR` (`AGENTS.md §6`, `§10.2.4`).
* ≥16 **interleaved** A/B pairs (`A B A B`), report median **and spread** and a
  same-config control arm. Decode on this box has been measured at 3.7 / 3.9 /
  4.3 / 4.8 / 6.0 / 6.4 tok/s from the *same binary* minutes apart — a single
  reading is not a result.
* Microbenchmarks: CPU-computed checksum + a `CONTROL_repeat` arm (`crux.md §6`).
  This is not ceremony: the 248.5 / 250.7 / 55 / 205 GB/s ghost numbers all came
  from probes that skipped it.
* Memory: `phys_footprint` via `/usr/bin/time -l`, **never**
  `--metrics peak_resident_pages_bytes` (`ru_maxrss`, understates by 5.5 GiB).
* **Sabotage before a coverage claim** (`AGENTS.md §3.4`): break the thing, watch
  the named test fail with a diagnosable message, revert, watch it pass. Record
  the observed failure text.
* Every tok/s number is `--metrics decode_tokens_per_second` **and** the wall-clock
  `decode_wall_tok_s` once it exists (R3) — the `--metrics` bracket at
  `generate.rs:383` **excludes the sampler**, which is exactly why the shipped
  default reads `14.37` while the figure everyone quotes is `16.05`.

## 1. Branch and baseline pin (T0, blocks everything)

HEAD is `mtp-speculation` `517bc4e`. `33e8ead` (the R1 instrument) exists only on
`agent/tensor-ops`; `master` `fc36ee1` is the tree `MEMORY.md` flags as broken
(though at HEAD `pipeline_audit.rs` and `sweep.rs` do exist, so that note is
stale for HEAD — re-verify rather than trust).

* Confirm HEAD builds **before** anything else. If it does not, the first task is
  a build fix on its own commit, not a perf change.
* Work on a branch off HEAD. Record: branch name, commit, `Gxma-repacked` path,
  and the frozen greedy output hash for the AC1 prompt.
* **Re-pin the baseline in the same session as the A/B.** Do not A/B against a
  number from another session, another hour, or another agent — the spread above
  makes cross-session comparison meaningless.
* Recover `alu_boost.rs`: `git show 33e8ead:crates/engine/examples/alu_boost.rs`
  (781 lines). Note `AGENTS.md §6`: concurrent agents touching one file is the
  hazard; this file is unowned.

## 2. Sequencing

```
T0  pin branch, build, recover instrument, re-pin baseline
     │
T1  re-measure arm_decode_u4            ── gate: no reproduction → R1 dead, stop
     │
T2  port _tl4 into the kernels that run  ── + GEMV bandwidth A/B
     │
T3  end-to-end greedy A/B, ≥16 pairs    ── AC1 (≥17.2) or record honest number
     │
T4  input_transform fusion (R2)         ── bit-exact or revert
     │
T5  device-sampler gate (R3)  ──┐
T6  rewrite the 2 gate tests    ─┘
     │
T7  attention split sweep (R4, waiver-gated, default OFF)
     │
T8  residual fuse budget (R5)
     │
T9  defects (R6.1 confirmed; R6.2 reproduce-or-drop; R6.3 note)
```

T1 is a **gate, not a task**: it can kill R1, and that is a legitimate outcome
recorded in `findings/`, not a failure of the plan. T5–T6 are one commit —
landing the gate without the test rewrite would leave a red tree and, worse,
would leave the *old* assertion protecting the *new* behaviour's absence.

## 3. Touchpoints (verified this session)

### T2 — the kernels that actually run

From `KAPLUMBA_TRACE_GEMV=1` (`crux3 §2`):

| kernel | path | who dispatches it |
|---|---|---|
| `..._k3_fused3_2col` | `linear_repacked.metal:1432` | sliding q/k/v |
| `..._k3_fused2_2col` | `:434` | mlp gate+up; full q/k |
| `..._k3_colmajor_2col_sk2` | `:2429` | **o_proj, down_proj** |
| `..._k3_colmajor_contig4` | `:1992` | `linear.rs:2091` — "the kernel the engine actually reaches" |

`linear_repacked.metal` is spliced into `linear.metal` by `combined_source()`
(`linear.rs:514`) and is **outside** the generator's rewrite window (which runs
`linear_gemv_exl3_k1(` → `kernel void linear_gemv(`, ~8200 lines) — AC6 exists
because running the generator once deleted the repacked kernels **live** and the
dispatcher then silently fell through to the row-major kernel on column-major
weights (`linear_repacked.metal:11-19`). Every kernel is built by
`ishizuki_kernel_names` (`linear.rs:542`) because the repacked names are
`linear_gemv_ishizuki_k3_*`, which `repacked_kernel_names` (`linear.rs:557`,
matching `linear_gemv_repacked_`) **does not match**.

`_tl4` variant = `_colmajor` / `_fused2_2col` with the tile loop unrolled 4-deep
**at the same tile assignment and the same per-thread accumulation order**. Any
change to the reduction order makes it non-bit-exact and out of scope
(`alu-boost §3`: repartitioning tiles is `1.04–1.11×` and *not shippable* for
exactly this reason). **Do not add a threadgroup staging buffer** — that is what
`_u4` does and it loses (`probe_reduce_epilogue.rs:246`).

### T4 — `linear_input_transform` fusion

`linear.metal:475` (`transform_simd:402`) runs a 4-point Hadamard
(`simd_shuffle_xor` 1,2,4..64) over `input`+`input_signs` → fp16 `output`;
the GEMV then re-reads `output`. Two hard constraints from the source comments:

* `raw` stays **f32** — it holds the shuffle-reduced GEMV output and the very
  next dispatch runs `transform_simd` over it; narrowing would round *before* the
  Hadamard, the same 1-ULP class that changed 43 of 128 greedy tokens
  (`attention.rs:176-182`, `perf_probe_activations_traffic.rs:74-82`).
* The GEMV reads **`transformed`, never `input`**, when `pretransformed = true`
  (`linear.rs:2716-2725`, `:2513-2522`). Fusing means computing the Hadamard in
  the GEMV's `x` path — the **same** `simd_shuffle_xor` order — or the bit-exact
  property is gone.

Bit-exact proof: keep the shuffle order identical; compare full fp16 rows.
Bit-exactness is the *whole* claim, so it is checked before any A/B.

### T5/T6 — device sampler gate

One-line semantic change at `crates/engine/src/chat.rs:388`, plus `generate.rs:971`
(`let pipelined = !margins.enabled` — a *different* gate; read it before
touching), plus the engine already-present pieces (`model.rs:4004`, `:4057`,
`:4071`, `:4183`, `:4192`).

Two tests **assert the current gate** and must be rewritten to assert the new
meaning, not deleted:
* `tests/perf_probe_host_ops.rs:373 temperature_above_zero_disables_the_pipelined_path` (`:377` greps `let pipelined = self.greedy`; `:1099` sibling in the concurrency probe).
* `tests/multi_turn_chat.rs:229 pipelined_and_barrier_decode_agree_across_a_multi_turn_session` — currently runs **both** arms at temperature 0 (`chat_session(&turns, 30, false)`), which is exactly the configuration that cannot discriminate the gate. It must be generalised off temperature 0 to become a real test of R3, with a fixed seed.

Correctness bar: at `--temperature 0` output is byte-identical (≥3 runs, text
only, T2 established that bar). At `--temperature 1 --top-k 64` the device and
host samplers must agree **statistically** (`device_sampler.rs` uses a TV-distance
threshold of `0.06`, where a correct sampler reads `0.026–0.036`) — not
bit-identical, because one draws from device RNG state. Claiming bit-identity at
temp 1 would be the same class of error as claiming it at temp 0.

### T7 — attention split sweep

Sweep `splits {4,8,12,13,16}` on one real layer, one session, with a control arm
(resolves Q2: `1.34×` isolated vs `+2.17%` end-to-end, `§10.6`). Ship **only**
behind `KAPLUMBA_DECODE_SPLIT_RULE=workgroups`, default OFF. It is **not
bit-exact** (`434/476` elements differ by 1 fp16 ULP at 8204), which is why it is
opt-in and why `spec.md §2` names the waiver rather than hiding it.

## 4. Risks

| risk | why real | mitigation |
|---|---|---|
| **R1 does not reproduce** | instrument absent from HEAD; `alu-boost`'s own harness ceiling (`arm_seq` `217.1`) is *below* shipped `239.8`, so a +5–9% on a slower arm may vanish | T1 gate; Q5. Fall back to T4–T6 and record honestly |
| **micro→production gap** (`~1.6×`, `203` vs `125–150` GB/s, `~+30%` if closed) | the single largest unmeasured question in the tree, ~10× any instruction win; **nobody has done it** | not a task here — listed as the follow-on spec if T2 stalls |
| **`_tl4` register pressure** | 4-deep unroll may spill; `alu_boost` int-pipe instrument is untrustworthy so no ceiling can be predicted | record register/occupancy diagnostics; revert on any spill |
| **bit-exactness lost silently** | this repo's documented failure mode (`AGENTS.md §3.4`); 1 ULP changed 43/128 tokens before | greedy hash compared every run; sabotage before claiming coverage |
| **box drift masquerades as a win** | 3.7–6.4 tok/s spread from one binary | interleaved pairs + control arm, always |
| another agent holds the GPU | `AGENTS.md §10` — two model instances already cost 3.9 GiB of swap and collapsed to 1.4 tok/s | `gpuq try`, exit `75` = do T0/T9 work instead |
| **AC1 unreachable** | every legal lever summed is `+1.2–2.5`, and `17.2` sits at the top of that | that is a **result**, recorded with the spread — not a reason to keep tuning |
| `attended` resize changes memory | `22,016 B/layer/token` of padding is currently *free* (page-granular); sizing it exactly adds real bytes | AC5 re-measures `phys_footprint`; the fix must not push past `16.5 GiB` |

## 5. Requirement → task map

| req | tasks | gate |
|---|---|---|
| R1 unroll | T1 → T2 → T3 | T1 must reproduce `0 differ` **and** a ratio above noise |
| R2 transform fuse | T4 | fp16 rows identical before any A/B |
| R3 sampler gate | T5 + T6 (one commit) | temp-0 byte-identical; temp-1 TV < `0.06`; both gate tests rewritten |
| R4 split waiver | T7 | opt-in, default OFF, 1-ULP waiver recorded |
| R5 residual fuse | T8 | each candidate A/B'd alone or not landed |
| R6 defects | T9 | `attended` test green; `_mul1` reproduced or claim dropped |
| R7 measurement | **all** | §0 |

## 6. What this plan will not do

* No speculative decoding, no MTP, no draft model, no n-gram/lookup reuse — the
  user ruled this out explicitly, and it is also the only route that would have
  cleared the pins (`1.63` accepted/pass → `20 @16k` at today's `81.3 ms` step).
* No bpw change (`2–4 bpw` is a `62.3 ms` plateau; bytes/rate cancels), no KV
  quantization (`AGENTS.md §8`), no head two-stage (`≤0.158 GB` of `1.057`).
* No occupancy / split-K / 4col / `uint4` / `simd_shuffle` / dequant-removal /
  megakernel / threadgroup-size / prefetch retuning — each has a sabotage-verified
  negative in the tree, listed in `spec.md §6`.
* No Python in engine or product code (`AGENTS.md §8`).