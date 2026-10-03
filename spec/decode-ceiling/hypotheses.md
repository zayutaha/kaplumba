# Decode Performance Hypotheses — Register

Every performance idea gets an ID here **before** anyone implements it, so that a
later idea cannot quietly absorb credit for an earlier one's result. Each is then
tested by **exactly one implementation agent at a time, with sole GPU access**.

## The gate every hypothesis must pass first

Measured noise floor on this box:

| | |
|---|---|
| Same-binary, same-config spread | **35%** (`decode-25tok-peak-bandwidth` notes) |
| Same binary, minutes apart | **3.7 – 6.4 tok/s** (AGENTS.md §10.5) |
| Typical baseline, 3 bpw colmajor | **16.05 – 16.10 tok/s**, 62.3 ms/step |
| Workload | 37-token prompt + 32 generated tokens, `--temperature 0` |

Consequence: **a single run proves nothing.** Three false fixes have already been
retracted in this repo for exactly this reason. No hypothesis may be reported as a
win without interleaved A/B and a stated confidence interval.

## The arithmetic that frames everything

| quantity | value | source |
|---|---|---|
| bytes read per token | **12.04 GB** | 10.983 GB layer trellis + 1.057 GB lm_head |
| achieved | **193 GB/s** | 12.04 GB / 62.3 ms |
| best 3-bpw kernel ever measured | **203 GB/s** | `decode-25tok-peak-bandwidth` §16.3 |
| practical copy ceiling | ~250 GB/s | AGENTS.md §10 |
| wire | 273 GB/s | AGENTS.md §8 |

| target | ms/step | sustained BW needed | % of wire | verdict |
|---|---|---|---|---|
| **16.05 (now)** | 62.3 | 193 GB/s | 71% | measured |
| 18 | 55.5 | 217 GB/s | 80% | plausible ceiling |
| **20** | 50.0 | **236.7 GB/s** | **87%** | **above the best kernel ever built here** |
| 21.5 | 46.5 | 259 GB/s | 95% | outside envelope |
| 23 (wire bound) | 43.4 | 273 GB/s | 100% | impossible with any overhead |

**Therefore: 20 tok/s cannot be reached by making the existing kernels faster
alone.** 20 requires 236.7 GB/s against a 203 GB/s measured best. Any hypothesis
that claims to close that gap purely through kernel speed is falsified before it
is written, and is recorded as such rather than quietly pursued.

Two escape routes exist, and they are the only two:

- **R1 — fewer bytes per token.** (mixed precision, head precision, layout that
  reads less). The 3 bpw layers are 91% of bytes.
- **R2 — a fundamentally higher code-decode rate.** Evidence: GB/s scales exactly
  inversely with bit-width at ~330–556 Gcodes/s, so the loop is **code-throughput
  bound, not bandwidth bound**. 20 tok/s needs 660 Gcodes/s; the fastest kernel in
  this tree does 541.

## Hypotheses

Status key: `PROPOSED` → `TESTING` → `CONFIRMED` / `REFUTED` / `INCONCLUSIVE`.

| ID | Angle | Attacks | Status | Result |
|---|---|---|---|---|
| P01 | EXL3 per-code instruction cost | R2 | PROPOSED | |
| P02 | Dispatch & inter-kernel gap overhead | R2 | **REFUTED** | `findings/P02-dispatch-overhead.md` — step is AT the 203 GB/s measured peak |
| P03 | Why `lm_head` hits 253 GB/s vs layers' 125–150 | R2 | PROPOSED | |
| P04 | DRAM layout / contiguity / page hits | R2 | PROPOSED | |
| P05 | Split-K & drain for long-K projections | R2 | PROPOSED | |
| P06 | CPU per-token host work; temp>0 **non-pipelined** path | R2 | PROPOSED | |
| P07 | Attention + element-kernel cost; T5.3/T5.7 fusions | R2 | **partly REFUTED** | `findings/P07-attention-elements.md` — T5.3 already shipped; T5.7 worth +0.005 tok/s; **16k context costs ~1 tok/s** |
| P08 | Reduce bytes/token (head precision, 2-stage head) | **R1** | PROPOSED | |
| P09 | Measurement trustworthiness / power analysis | gate | PROPOSED | |
| P10 | Kernel fusion census | R2 | PROPOSED | |
| P11 | Hadamard + MCG codebook per-code cost | R2 | PROPOSED | |
| P12 | Occupancy & threadgroup shape | R2 | PROPOSED | |
| P13 | K-loop memory pipelining / burst loading | R2 | PROPOSED | |
| P14 | Activation dtype & non-weight DRAM traffic | R2/R1 | PROPOSED | |
| P15 | KV layout & sliding-window wrap | R1 (long ctx) | PROPOSED | |
| P16 | Concurrency / multi-queue overlap | R2 | PROPOSED | |
| P17 | Dispatch-selection audit (is the best kernel running?) | R2 | PROPOSED | |
| P18 | Metal SIMD & arithmetic intensity | R2 | PROPOSED | |
| P19 | Mixed precision across layers | **R1** | PROPOSED | |
| P20 | Critical-path latency analysis | R2 | PROPOSED | |

## Testing protocol — one agent at a time, sole GPU access

AGENTS.md §6 forbids parallel agents on one working tree; §10 forbids two model
processes. So testing is strictly serialised:

1. **One** implementation agent at a time. It owns the working tree and the GPU.
2. Before it starts: baseline re-measurement **in the same session**, ≥6
   interleaved A/B pairs, medians **and** spread, with a same-config control arm
   to measure the noise floor live.
3. The agent implements exactly one hypothesis. No opportunistic extras.
4. Per AGENTS.md §3.4: **sabotage the fix, show the test failing, revert the
   sabotage.** No coverage claim without an observed failure.
5. Required regression gate, every time: 3 bpw must stay **bit-identical** at
   `--temperature 0` over ≥3 runs. Compare the generated **text only** — never a
   hash of output that includes the `--metrics` line, whose timing fields vary
   every run and produce false divergence. (I hit that false positive myself.)
6. Verdict recorded here as CONFIRMED / REFUTED / INCONCLUSIVE with the interval.
   **A refuted hypothesis is a result.** Six of nine reference techniques mined in
   `engine-performance` did not pay off, and that list is only useful because it
   was written down.

## Open decisions for the user

- **D1.** 20 tok/s is above the measured best kernel. If the honest ceiling comes
  back at 18, do we accept 18, or change the goal to bytes/token (mixed precision)?
- **D2.** Anything in the R1 column trades quality. AGENTS.md §8 forbids quality
  loss. That is a user decision, not an engineering one.

---

# Round 2 — memory bandwidth (Apple Silicon) and kernel optimisation

Agents write their findings to `spec/decode-ceiling/findings/<ID>.md`. **Every
agent must produce a text report as well as a file** — two attempts (M01, K03)
returned nothing at all, so both were relaunched with the full brief.

Clean negatives are results. Six of nine techniques mined in `engine-performance`
did not pay off, and that list is only useful because it was written down.

## M — Apple Silicon memory bandwidth

| ID | Angle | Attacks | Status | Findings |
|---|---|---|---|---|
| M01 | TLB pressure; 16 KB VM pages vs `PAGE_ALIGN = 4096` | R2 | relaunched | `findings/M01-tlb-page-size.md` |
| M02 | Unified memory: host DRAM traffic competing with GPU streaming | R2 | PROPOSED | |
| M03 | Metal storage modes (Shared/Managed/Private) and compression eligibility | R2 | PROPOSED | |
| M04 | Cache pollution + load width / prefetch hints | R2 | PROPOSED | |
| M05 | True achievable DRAM peak; code-rate vs bandwidth reconciliation | gate | PROPOSED | |

## K — kernel optimisation

| ID | Angle | Attacks | Status | Findings |
|---|---|---|---|---|
| K01 | `linear_gemv_exl3_*` arm vs `ishizuki` arm | R2 | PROPOSED | |
| K02 | `_nod` and `_w2` variants; dead generated code | R2 | PROPOSED | |
| K03 | Column tiling `_2col`/`_4col`/`_co`/`_co4` | R2 | relaunched | `findings/K03-column-tiling.md` |
| K04 | Reduction tree: shuffle width, barriers, atomics | R2 | PROPOSED | |
| K05 | Vector loads (`ushort2`/`ulong2`) + threadgroup tile staging | R2 | PROPOSED | |
| K06 | `permutation_mode` / coherence probe / colmajor gain validity | R2 | PROPOSED | |
| K07 | MCG vs Mul1 codebook arithmetic (quality trade) | R1/R2 | PROPOSED | |
| K08 | Attention split-KV, `decode_splits`, long-context cliff | ctx | PROPOSED | |
| K09 | Shape specialisation; tile divisibility and tail waves | R2 | PROPOSED | |
| K10 | Activation vector reads and read amplification in the K-loop | R2 | PROPOSED | |
| K11 | `exl3_code_at` vs `exl3_decode8` — is any fused kernel on the slow path? | R2 | PROPOSED | |
| K12 | Threadgroup size retune (128/256/512 threads) | R2 | PROPOSED | |
| K13 | K-loop unroll and ILP; per-code branch elimination | R2 | PROPOSED | |
| K14 | Dispatch layer: `format!`, HashMap lookups, dead bindings, spin-waits | R2 | PROPOSED | |
| K15 | Fused vs separate MLP decision tree; unset env flags | R2 | PROPOSED | |

## Note on two leads already visible in the code

- **K13** — the per-code extract is `return (r <= 64 ? a >> (64-r) : c >> (96-r)) & 0xffff`. At ka=3, r is always <= 64, so that branch is evaluated 8x per lane and always takes the same side. Potentially free.
- **K05** — the unaligned path does `packed[i*2] | packed[i*2+1] << 16`, i.e. two separate 16-bit loads on data that IS 2-byte aligned. A single `ushort2` load would halve it. 199/411 of the 4 bpw tensors and the 6 bpw head take this path.