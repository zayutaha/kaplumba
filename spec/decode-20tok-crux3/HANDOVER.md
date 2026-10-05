# HANDOVER — session of 2026-10-05

**This folder previously did not exist.** Commit `77cbca2` ("spec/decode-20tok-crux3:
T1-T2 -- the decode GEMV is NOT at a memory wall") added nine files — examples,
tests, a probe `.metal`, two scripts — and **no spec document at all**. The entire
finding lived in a commit message. This file is the record that was missing.

Read this first, then:

- `spec/decode-20tok-crux/crux.md` — the original roofline proof (branch `agent3/decode-20tok-crux`)
- `spec/decode-20tok-crux2/crux2.md` — the adversarial re-derivation (branch `adversary/decode-20tok-crux2`)
- `spec/prefill-24k-90tps/` — the other live workstream (`NEXT.md`, `pragma_peak_report.md`,
  `fixer_report.md`, `adversary_report.md`, `adversary_report2.md`)
- `MEMORY.md` §AGENT LOOP CONVENTION, §QUALITY GATE, §NEXT GOAL

---

## 1. What we were focused on

Two goals, in this order.

**(a) Finish `spec/prefill-24k-90tps`.** 90 tok/s prefill and 14 tok/s decode at
24k. Neither headline target was ever met. **23 commits** landed on `mtp-speculation`
in this session. Net throughput change in the default build: **approximately zero.**

**(b) 20 tok/s pure decode at 16k** — a new goal the user set after (a). Hard
constraints, restated verbatim because they are not negotiable: **no speculative
decoding, no drafters, no MTP, no Medusa/Eagle, no lookahead, no n-gram or prompt
lookup, no caching of generated text, no quality loss.**

---

## 2. State of each workstream

### prefill — very likely unreachable, and that is a target-change conversation

At 24,590 tokens: **GEMM+other 223.3 s + attention 127.5 s + inter-dispatch gap
16.2 s = 367.0 s → 67.0 tok/s**, against a 273.2 s budget for 90 tok/s. **88% of
the budget is gone before attention starts.** Attention must go **3.8×**, which
against the *corrected* 8.30 TFLOPS peak is **~86% of tensor peak** (it was
believed to be 46% — see §4.2). The FFN GEMM sits at 81% of peak, so there is
almost nothing there either.

The one lever still standing is **§9.1b's key-split flash decode — unbuilt, not
refuted.** Its original refutation used a load figure 4× too small (§12 established
the "86 s bandwidth floor" was wrong by ~350×), so the refutation does not hold.

### decode — 20 tok/s is impossible; the real ceiling is ~17

See §3. **This is not a target anyone should keep pushing at 20.**

---

## 3. The 20 tok/s verdict (two adversarial rounds, it survived)

```
bytes/token = 12.0580 (weights) + 2.1810 (KV) = 14.2390 GB
20 tok/s needs 284.8 GB/s
  = 104.3% of the 273 GB/s wire peak
  = 112.3% of the best arm ever measured (253.6)
  = 116.2% of sustained (245)
```

`system_profiler` confirms a 20-GPU-core M4 Pro pinning at **273 GB/s**. **No
kernel outruns the connectors.** The impossibility does not depend on any
measurement in this repo.

**bytes/token was reproduced to zero bytes** from three sources sharing no code:
Σ`data_offsets` over 2,422 tensors, Σ file size − header, and
`quantization_config.json`'s per-tensor `n_bytes`. Every layer tensor is exactly
**3.0000 bpw**, the head exactly 6.0000. This also **settled the 13.145-vs-11.834
disagreement**: 11.834 wrongly used `head_dim` 256 for all 60 layers (the 10
full-attention layers project to 32×512=16384) plus a phantom `v_proj`.

**Every byte-reduction route is closed, with numbers:**

| route | verdict |
|---|---|
| address-stream entropy | **dead.** 5 stride permutations, −1.3% to +1.1%, straddles the ~1% noise floor. Burst length 1→48 flat. No access pattern beats a raster scan. |
| `lm_head` | at most **0.1584 GB** (exclusion bound 200.2 vs config `final_logit_softcapping: 30.0`). Even a *completely free* head needs 263.6 GB/s = 96.6% of wire. Dead on exactness **and** arithmetic. |
| KV | `k_eq_v` looks like 0.671 GB of duplicated V in full layers, but V is unweighted-normed/unroped while K is weighted-normed/roped — all 512 dims differ. |
| lower bpw | no sub-3.0 checkpoint exists; building one needs the Python EXL3 quantizer (`AGENTS.md` §8 forbids); **the model is 4-bit-QAT, so 3.0 bpw is already below its training target.** |

**True maxima at zero overhead: ~18.2 tok/s at 8k, ~17.3 at 16k.** Today is 12.3
@16k. So the honest prize is **+40%: 12.3 → ~17**, not 20.

The user's original arithmetic was **250/11 = 22.5**. The 250 GB/s was *right* —
it is reachable (246–253 measured). The error was the denominator: omitting KV.
250 / 14.239 = 17.6, not 22.5.

---

## 4. The crux3 finding — SPLIT VERDICT, read this before trusting it

### 4.1 What is verified

`KAPLUMBA_ABLATE=attention` **never reached the decode path.** Every
"everything ablated" residual in this repo silently contained **60 live attention
kernels**. Fixed with two early returns. `probe_budget_reconcile`'s inverted test
now fails when a gate goes missing (verified by deleting each gate). **This part
is real and is a genuine bug.**

Also landed in `77cbca2`: `attn_bench.rs` `KV_DEPTH` 8192 → 16384, so attention
is measurable at 16k.

### 4.2 What is NOT established

The headline "**the decode GEMV is NOT at a memory wall**", and its table:

| shape | production | 2-op | 1-op | gain |
|---|---|---|---|---|
| down K21504 N5376 | 203.0 | 239.5 | 238.0 | +17.9% |
| q_slid K5376 N2048 | 156.5 | 184.8 | 200.8 | +18.1% / +28.3% |
| o_proj | 180.5 | 217.9 | 217.4 | +20.7% |
| gateup | 164.2 | 182.0 | 182.2 | +10.8% |

The mechanism claimed is that the 3 bpw decode GEMV is partly bound by
**`decode_state_mcg`**, worth 11–21% of GB/s, as a **latency** cost at the head of
a serial chain (`IMUL→AND→XOR→HADD2`) rather than a throughput cost. The evidence
for "latency, not throughput" is genuinely good: the 2-op and 1-op arms land within
1–10% of each other **despite the 1-op arm having half the arithmetic**, which an
op-count model does not predict. Methodology was disciplined — 9 interleaved
rounds, spread 1.00–1.04, clock checks 1.001–1.004×, and **five sabotages all
caught** (arm B made identical to A → reports 0.998 instead of 1.179; chain
collapse `v[7]=v[6]` caught; arm C stops reading `wb` caught; probe kernel named
in `linear.rs` caught).

**But the arms live in a transcription that production never runs.** Running its
own test suite (`cargo test -p kaplumba-engine --test probe_crux3_decode_algebra`
→ 5 passed) shows the test *names* give it away:

```
test the_macro_is_a_transcription_and_says_so ... ok      <-- asserts it is a copy
test no_probe_kernel_is_reachable_from_the_crate ... ok  <-- not in the prod path
```

Arm A's equivalence to the production kernel rests on
`scripts/crux3_armA_equiv.py`, whose own execution is unverified here.

**This is the third time in one session that a standalone transcription produced a
number that does not transfer** (see §5.1, §5.3). Treat `+17.9%` as a **strong
hypothesis**, not a result. **No end-to-end decode tok/s was ever measured** — the
agent was cancelled first.

### 4.3 The one experiment that settles it

Put arm C's algebra **into the real `linear.metal` decode path** and measure
end-to-end decode tok/s at 16k, with the quality gate applied.

- If decode moves **12.3 → ~15–16 tok/s with byte-identical greedy output**, the
  finding is real and it is the biggest win of the session.
- If it does not, it is transcription noise and **203 GB/s stands as the wall** —
  in which case the only remaining route to 20 tok/s is speculation, which the user
  has forbidden.

This is cheap and decisive. **Do it before anything else in this folder.**

Note the payoff if real: crux2's roofline used the *synthetic* sustained figure
(~245 GB/s), so removing a real-path inefficiency gets the path to the ceiling
crux2 already computed. 14.239 GB ÷ 240 GB/s = 16.9 tok/s. It does **not** reach
20 — it closes the 12.3 → ~17 gap.

---

## 5. Discoveries that will cost a newcomer hours

### 5.1 The transcription trap — three instances, same shape
A hand-written copy of a kernel produces a large, clean, reproducible number that
does **not** transfer to the shipping kernel.
- `crux3` arms B/C (§4.2) — `no_probe_kernel_is_reachable_from_the_crate`
- The `#pragma unroll` "2.3× win" — **refuted**. The 2.3× came from a standalone
  transcription. Measured on shipping `attention.metal`: every unroll variant is
  byte-identical, ratios 0.993–0.998, and `unroll(32)` is 3.5% *slower*. Shipping
  was already optimal.
- `mma_peak.rs` (below) — inflated peak by 2×.
**Always confirm a large effect on the shipping kernel, in one build, with every
arm compiled from the same source.**

### 5.2 `mma_peak.rs` measured a phantom 2× peak
It accumulated into **16** accumulators and **stored 8**. The eight unstored chains
are dead stores; the compiler deleted them. Real ceiling **8.30 TFLOPS**, not
15.02. **Every percentage in `implementation_notes.md` §20–§23 was wrong:**
attention 12%→23% of peak, AC2's ask 46%→**86%**, the FFN GEMM 43%→81%, the
1.7×-GEMM alternative 52%→**98%**, and §23.6's "0.8% margin" → **none** (its
90.7 tok/s row is 213% of peak). §22.3's own table reaches 110.8% — not an error,
because QK and PV share the tensor pipe as dependent chains.
Fixed: stores all 16, keeps `dead8` as a control, asserts it can fail.

### 5.3 cargo's freshness test is **mtime-based** — §10's ablation raced it
§10's decisive arm (`if (tid < 256u) { return; }`) read 964.79 ms; a truly empty
body costs **0.19 ms**. The flash path dispatches **256 threads per threadgroup**,
so that arm *was* an empty kernel. The wrongness was **not** `include_str!` (cargo
does track `.metal` edits — proved by making a `scale×1.5` edit fail a test through
the ordinary build path). The cause was **§10's own `git checkout`-per-arm loop
racing cargo's mtime-based freshness test**. §10's table proves the edits *were*
landing: its threadgroup-spin arm moved 5×.
**Lesson: a git-checkout-per-arm ablation loop is unsound on this repo. Compile one
binary with all arms and select at runtime.**

### 5.4 Bandwidth on this box moves by 2× silently
Two runs returned 138.6 and 231.2 GB/s against a 250.5 control, with `gpuq status`
and `ps` both reporting idle — no trace in `PhysMem`, `ps`, or load average. **Nine
bandwidth instruments have now misreported in this project.** The last author found
**seven defects in its own two instruments**, five of which would have produced
plausible wrong numbers (one credited 80× the bytes moved; one read 3 GB past a
buffer end and SIGSEGV'd; `--gb 8` silently measured 6 GB while printing its
settings).
**A uniform factor cancels in every ratio, so only a positive control catches it,
and only an assertion against something outside the measurement.** Every reported
bandwidth or tok/s number needs a positive control in the same run.

### 5.5 This box's throughput is not repeatable to better than ~10% without controls
Decode has been measured at 3.7, 3.9, 4.3, 4.8, 6.0 and 6.4 tok/s from the **same
binary** minutes apart. Single-round pairs carry 3.7% spread at 4k and up to ±20%
at 24k. Interleave A/B and report medians **with** spread.

### 5.6 `AC5`'s GEMM bound could not fail
`MMA_TOLERANCE` was **0.02** — ~10× looser than AC5's own 2.16e-3 limit, so the
acceptance criterion was unenforceable. Tightened to **3e-3**. The actual worst
normalised GEMM error is **4.55e-4 at production K=8192**, does **not** grow with K
(0.90× over a 64× range), and is 4.7× inside budget — so nothing was hiding under
the hole.

### 5.7 Short-context quality evidence is worthless for context-dependent changes
The `workgroups` split-rule flip measured **"1 differing token in 1,536"** across
three contexts — and then changed **434 of 476 greedy tokens from token 39** at
8,204 tokens, reproduced to the token in two runs, stopping 36 tokens early. Under
`--temperature 0.7` the same arm did not diverge at all, so the two legs disagree
and the rule's effect is **not characterised**. **The quality gate rejected the
flip; `DecodeSplitRule::ThreadTarget` remains the default** (`crates/engine/src/attention.rs:543`).
Always test quality at the context that matters.

### 5.8 The 24k decode measurement is structurally impossible on this box
The 24k run's own `phys_footprint` is 18.35 GiB of 24 GiB. It evicts the desktop
and the compressor then drives decode to **0.055 tok/s** — a 99× pessimisation. A
measurement gate cannot fix this; it can only refuse to report a compressed sample.
**A pair did fall out of a sweep at identical context and prompt**: 4.732 tok/s
beside 12.375 / 12.393, with `compressor_max` 9.55 vs 8.42/7.76 GiB — the
degraded run is *higher* on both fields and the memory gate **passed** it. Had that
been a control, the headline would have read "+161%". Discard any 24k sample with
>2% CPU in compression.

### 5.9 Stale branches are still in this repo and are not to be merged
`decode-host-overhead` (2026-09-29) and `decode-matvec-k3` (2026-09-28) predate
this work and **conflict** with the main tree in `sampler.rs`, `encoder.rs`,
`generate.rs`, `model.rs`. They were deliberately left unmerged. ~30 prunable
worktrees also exist under `/private/tmp/wt-*` and the opencode temp dir.

---

## 6. Quality gate (hard requirement, from the user)

No quality loss. Specifically:

1. **Byte-identical greedy text vs base `f2b2da1`** for the default configuration —
   not merely "identical between two arms of the same build", which cannot catch a
   change that moved the numerics for both arms.
2. **≥512-token generations** at temperature 0 **and** at a sampling temperature,
   checked for **repetition collapse** and loop-fixation explicitly.
3. **No NaN/Inf** in logits or hidden state.
4. Worst normalised GEMM error **≤ 2.16e-3**.
5. Every numeric perturbation labelled **exact** or **bounded-ulp**, **never both**.
   (`spec.md` once called N1 both "bit-exact" and "one fp16 ulp on ~1% of outputs".)

`scripts/refprompts.sh` is 5 prompts at `--max-new-tokens 96` — necessary but
**not sufficient**; five 96-token answers to "what is the capital of France"
cannot detect degeneration.

---

## 7. Commands

```bash
M=~/.kaplumba/models/Gxma/Gxma

# prefill (PP is ~24k tokens)
PP=$(python3 -c "print('Explain the theory of relativity. '*100)")
scripts/gpuq run -- ./target/release/kaplumba --model "$M" --prompt "$PP" \
    --max-new-tokens 1 --temperature 0 --metrics

# per-kernel GPU attribution (the profiler itself distorts: 0.082 vs 5.44 tok/s clean)
KAPLUMBA_GPU_PROFILE=1 ./target/release/kaplumba --model "$M" --prompt "$PP" \
    --max-new-tokens 1 --temperature 0 --metrics

# the crux3 decode arms
scripts/crux3_ab.sh
cargo test -p kaplumba-engine --test probe_crux3_decode_algebra
```

**GPU discipline:** one model process at a time, enforced by a kernel `flock` in
`crates/engine/src/gpu_lock.rs` — **exit 75 = busy, back off**. `scripts/gpuq`
deliberately takes **no** lock (an earlier version did and deadlocked: parent waits
for child, child waits for parent). **`scripts/gpuq try --` is the default; never
sit in a long blocking call.** Never leave a benchmark running when you finish —
a stray benchmark holds the GPU invisibly and has already broken one measurement
here. Use `/usr/bin/time -l` → `phys_footprint`; **never** `ru_maxrss` as a
footprint (it misses Metal pages entirely and understated real usage by 5.5 GiB).

---

## 8. Dead ends — do not retry

- **Query blocking / `BQ=2`**: −2.0% @8k, −6.2% @24k. `KAPLUMBA_FLASH_BQ=1` measured
  −0.9% E2E while the bench claimed 19% — a fifth ranking failure.
- **`#pragma unroll` on the QK loop**: shipping already optimal (§5.1).
- **Accumulator chain count**: not a lever at any count (compile-time 2 and 4
  chains ×1.001 / ×0.994; 16 chains 2.30×, 2 chains 2.28×, 24 chains worse).
- **Prefill key-range passes (N1)**: neutral at 16k and now also at 24k with a valid
  control (66.829 vs 67.051). Ships **off** (`KAPLUMBA_PREFILL_PASS_KEYS=0`) because
  the inter-boundary gap cost **+48.7%** for a neutral own_ms.
- **Decode split count**: settled at **13, not 16** (+2.17% @16k, measured −1.5%
  cliff at 14, exactly the 52→56 workgroup wave boundary). Reproduced twice
  independently. **Do not re-sweep.** But it is **opt-in** and the default was
  rejected on quality (§5.7).
- **The microbench can attribute but cannot rank.** Five documented ranking
  failures. Keep those two properties separate.
- **`o_proj` bandwidth anomaly** (from `spec/decode-20tps`): **does not exist.**
  All seven projections sit at 41–55% of peak; o_proj is at 98% of down_proj's
  efficiency. That spec's lead was an `M = 1` GEMV measurement and does not transfer.
- **The "86 s attention bandwidth floor"**: wrong by ~350× (§12). Do not re-open the
  GEMM (OQ4/OQ6 withdrawn).

---

## 9. Where to go next, in order

1. **§4.3 — the decisive crux3 experiment.** Arm C's algebra into the real
   `linear.metal`. This is the only thing standing between "biggest win of the
   session" and "transcription noise".
2. **Then the decode overhead work**, which is where the remaining room is: at 8k
   the miss is only 9.7% and **the binding constraint is overhead, not the
   connector**. `spec/decode-20tps` measured **749 dispatches/step at ~3.5 µs**
   (~2.6 ms of a ~60 ms step); `spec/decode-20tok` found ~560 ms of an 850 ms step
   was two kernels serialising to a single thread at `rows == 1`. Measure the
   current dispatch census (`perf_probe_dispatch_census`, `probe_pipeline_topology`)
   and fuse element-wise chains. Nobody has done this exhaustively.
3. **Then §9.1b's key-split flash decode** on the prefill side — unbuilt, not refuted.
   The kernel's measured limiter is **operand issue**: 98.7% MMA-issue-plus-loads,
   of which **52.4% is fragment loads** with the MMA count untouched, and **two
   fragment loads per MMA at `BQ=1`**. That is a concrete, cheap target.
4. **Report to the user** that AC2 and 20 tok/s both need a target decision. Do not
   change an acceptance target unilaterally.