# P02 — Dispatch & gap overhead — **REFUTED**

Verdict: **this angle cannot close the gap.** Decode is already at the measured
bandwidth roofline. Agent report retained in full below.

## The two headline corrections

**1. 546 was never the dispatch count.** It is the count of `GpuProfile::stamp`
boundaries — 9 per layer plus 6 in the epilogue. Those exist **only** under
`KAPLUMBA_GPU_PROFILE=1`; in production `GpuProfile::stamp` returns on its first
line (`model.rs:830-832`). The real count is `MetalEncoder::dispatches_encoded()`,
and reading every `encode_*` body gives **750 dispatches per decode step**
(prior measurement: 749).

**2. The step is already at the roofline.**

| | |
|---|---|
| measured step | 62.3 ms |
| weight bytes/token | 11.834 GB |
| **implied achieved bandwidth** | **203.8 GB/s** |
| measured 3-bpw GEMV peak in this tree | **203 GB/s** |

They are the same number. The step is **at** the peak the hardware has ever
demonstrated in this codebase.

| target | ms/step | required GB/s | vs measured 203 peak | vs 273 wire |
|---|---|---|---|---|
| 16.05 (now) | 62.3 | 203.8 | **100%** | 75% |
| 20 | 50.0 | **253.9** | **125%** | **93%** |

**20 tok/s requires 253.9 GB/s = 125% of the best GEMV this tree has ever
measured, and 93% of theoretical wire with zero overhead.** It is not reachable
by this or any dispatch-level change.

## 🔴 A coverage hole that invalidates §18.1 of the previous spec

`ablated("attention")` appears **exactly once**, at `attention.rs:911`, inside
`encode_online_attention`. **The decode path never calls it** — `model.rs:1864`
calls `encode_online_attention_split` → `encode_decode_split_with_splits`, and
neither consults the flag.

**Therefore `KAPLUMBA_ABLATE=attention` removes nothing on a decode run.**

Consequence: the headline *"7.32 ms/step residual with EVERY kernel-side role
ablated"* — the number several optimisation programmes have been measured
against — **still contains 60 real attention kernels streaming ~860 MB of KV per
step**. At 860 MB / 7.32 ms that is 117 GB/s, i.e. **most of the "residual" is
attention KV bandwidth, not overhead.**

Likewise the *"+0.21 tok/s for kv_norm_rope+attention+head_softcap+head_argmax"*
row is really only `kv_norm_rope` (60 dispatches) + `head_argmax` (1), because
`head_softcap` is skipped on the pipelined path (`model.rs:4444`).

**Fix (2 lines, no production effect since the flag is unset):** add the
`ablated("attention")` early-return to `encode_online_attention_split` and
`encode_decode_single_pass`, then re-run T5.12. **Do this before spending
anything else — every number in §18.1 is suspect until it is.**

## Dispatch census (750/token)

Per layer: `input_norm_and_qkv_transform` 1 · `gemv_fused3` (qkv) 1 · `finalize_simd3` 1
· `norm_rope_qkv_cache` 1 · attention split 1 · attention combine 1 · `input_transform` (o_proj) 1
· `gemv_raw_sk2` (o_proj) 1 · `finalize_o_proj_and_pre_ffn_transform_sk2` 1 · `gemv_fused2_mlp`
(gate/up) 1 · `finalize_gated_gelu_and_down_input` 1 · `gemv_raw_sk2` (down) 1 ·
`cross_layer_finalize_and_norm` 1 = **~12.3/layer**, plus 9 in the epilogue.

**124 single-threadgroup dispatches**, 121 layer-side.

Trap worth recording: `encode_gemv_fused3` and `encode_gemv_fused2_mlp` are both
called with `pretransformed = true` (`model.rs:1528, 2212, 1584`), so **neither
issues an `input_transform`** — reading their bodies suggests 3/layer, there is
exactly 1 (the explicit o_proj call).

## Ranked findings

| # | source | ms/step | change | bit-exact | verdict |
|---|---|---|---|---|---|
| 1 | `attention` ablation is a no-op on decode | 0 | add early-return x2 in `attention.rs` | yes | **measurement fix — do first** |
| 2 | 16 command buffers + 7 waits per token | **0.1–0.3** | `decode_max_dispatches_per_buffer()` → `u32::MAX` | yes | **yes** |
| 3 | `decode_splits` live `env::var` (60x/token, `attention.rs:199`) | 0.02 | `OnceLock` | yes | yes |
| 4 | `ablated()` Mutex+hash, 662x/step | 0.03 | read-only `OnceLock<HashSet>` | yes | fold into #1 |
| 5 | 121 single-threadgroup row reductions | **1.27** | 2-pass grid | **NO** | **no — forbidden** |
| 6 | attention combine + o_proj input transform | 0.09 | fuse | yes | no — below noise floor |
| 7 | 860 MB KV read per token | **3.2** | quantise KV | — | **no — forbidden** |

Items 5 and 7 are the only ones above 0.5 ms, and both are barred:
- #5 changes the association order of a 5376-wide row reduction, which
  `attention.rs:169-187` records as moving **43 of 128 greedy tokens for 1 ULP**.
- #7 is KV quantisation, forbidden by AGENTS.md §8.

## Why the rolling window is now pointless

`roll()` commits every 50 dispatches → **15 rolls → 16 command buffers and 7
`wait_until_completed` per step**, for no benefit: `generate.rs:970-1007` is
already a **1-step-deep software pipeline**, so the GPU is a full step behind by
construction and there is nothing left to overlap. The rolling was designed for a
non-pipelined host that had to finish encoding before the GPU saw anything —
**that host no longer exists on the decode path.**

The 1-deep depth is structural, not an oversight: step N+1 needs step N's argmax,
and there is no device-side hand-off (`model.rs:3695-3700`: `encode_greedy_pick`
refuses destination index 0, the index the next step's embedding reads).

## Honest ceiling

Everything achievable in this angle totals **0.2–0.4 ms = +0.05–0.10 tok/s**,
landing at ~16.2 tok/s. The previous spec's "physically achievable 16.5–17.5" is
correct, and this angle's contribution to it is **+0.1 tok/s, not +4**.

The 12 ms/step gap is **not** in dispatch overhead, host overhead, or launch
latency. It is not anywhere in the CPU-side code.

## Test delivered

`crates/engine/tests/perf_probe_dispatch_census.rs` — 10 static tests (no GPU) plus
2 `#[ignore]`d real-weights tests. It encodes the census as a pure function of
layer shape and flags, asserts the breakdown rather than a magic total (so a
future fusion that moves a dispatch fails loudly), asserts 546 != 750, and
includes sensitivity tests for the four switches that change the count.

**Not verified on real weights** — the real-weights tests were compile-checked but
not run, because they need `KAPLUMBA_TEST_MODEL` and the GPU. Treat 750 as a
prediction and 749 as the prior.