# P07 — Attention & element-kernel cost — **partly REFUTED, and 16k context is the real finding**

## Corrections to the brief (recorded so it is not repeated)

1. **`decode_splits` is at `attention.rs:191`, not `linear.rs:191`.** `linear.rs:191`
   is the `packed` field of `LinearBufferRequest`.
2. **`crates/engine/examples/attn_bench.rs:374` does not compile** (brief said 372).
   Only the *reporting table* is stale — `ab_splits` measures through
   `encode_online_attention_split`, so the measurements are valid. But the only
   per-shape attention harness in the repo **cannot currently be re-run**.
3. The brief's eight kernel labels (`cache_append`, `gated_gelu`,
   `post_attn_norm_res`, …) **do not exist on the shipping path.** `cache_append`
   only exists when `KAPLUMBA_NO_NORM_ROPE_CACHE` is set (`model.rs:1787`);
   `gated_gelu` and `post_attn_norm_res` only on the non-`permutation_mode == 1`
   branch. On the shipping path they are `kv_norm_rope_cache`, `gate_up_gelu` and
   `post_attn_pre_ffn_transform`.

## 🔴 The 7.8 ms figure in the brief is ~12% a profiler artefact

`KAPLUMBA_DECODE_PROFILE=kernel` calls `encoder.commit_stamped(label)` per label
(`encoder.rs:677-685`), which **commits a command buffer and re-arms**. So the
instrument turns each stamp into a command-buffer boundary: **546 buffers under
the profiler versus 1 on the shipping path.** That serialisation, not kernel
execution, dominates the `own_ms` figures — which is why the spec had to
calibrate the profiler by **1.71x** and why production ablation disagrees with it
by ~9x.

| instrument | element+attention layer |
|---|---|
| profiler `own_ms`, brief's labels | 7.8 ms |
| profiler ÷ 1.71 calibration | ~4.6 ms |
| **production interleaved ablation A/B (T5.10 — the reliable one)** | **0.84 ms** |
| **bandwidth floor (arithmetic)** | **0.25 ms** |

**The real figure is 1.3–1.6% of the step, and 0.4% of that is bandwidth.**

## Attention cost vs context — the actual finding

KV bytes/token = `50·min(P,1024)·16384 + 10·P·8192` (K and V are separate
buffers and differ: `attention_k_eq_v` makes V the *raw* K projection, before
norm+RoPE, `model.rs:1625-1628`).

| context | sliding | full | total | @273 GB/s | @203 GB/s |
|---|---|---|---|---|---|
| 512 | 419.4 MB | 41.9 MB | **461.4 MB** | 1.69 ms | 2.27 ms |
| 1 024 | 838.9 MB | 83.9 MB | 922.7 MB | 3.38 ms | 4.54 ms |
| 8 192 | 838.9 MB | 671.1 MB | 1 510 MB | 5.53 ms | 7.44 ms |
| **16 384** | 838.9 MB | **1 342.2 MB** | **2 181 MB** | **7.99 ms** | **10.74 ms** |

**Attention equals GEMV weight streaming at P ≈ 13 100.** Above that it is more
than half the step. The 50 sliding layers grow 2x then stop; the 10 full-attention
layers grow **32x**.

**Consequence for a 16k step: ≥ 58.3 ms (GEMV alone) + 7–11 ms (KV) ≈ 65–69 ms
= 14.5–15.4 tok/s**, before a single element kernel. **16k context costs ~1 tok/s
and is dominated by KV streaming, not by the model weights.**

## T5.3 — already implemented, three times over

| T5.3 asks for | shipped as | site |
|---|---|---|
| fuse input RMSNorm with GEMV input transform | `input_norm_and_qkv_transform` | `model.rs:1429` → `linear.rs:3126` |
| fuse post-attention norm | `..._o_proj_and_pre_ffn_transform_sk2` | `model.rs:1999` → `linear.rs:3042` |
| fuse post-FFN norm | `finalize_post_ffn_and_next_input_norm_transform` | `model.rs:2317` → `linear.rs:3169` |

**The "~500 small elementwise dispatches per token" the task names do not exist.**
Today's count for those roles is **3/layer = 180/token**, of which 120 were already
removed by `pretransformed` sharing. Also stale: `tasks.md:155` counts
`input_transform` at ×180; on HEAD it is **×60**.

**Verdict: close T5.3 as superseded.** Remaining recoverable: **< 0.2 ms**.

## T5.7 — the stated 1.8 ms is 20–45x the ceiling

Full kernel design is in the agent report (`attention_decode_combine_o_transform_256/512`,
fusing the combine's epilogue with `linear_input_transform`). Bit-exactness is
achievable and the load-bearing subtlety is that the combine's **own fp16 store
happens before the signs** — holding the value in fp32 "looks equivalent" and is
not.

What it removes: 60 dispatches/token and 2.29 MB/token. What it does **not**
remove: the partials round trip, **36.95 MB/token**.

**Expected: 0.04–0.09 ms = +0.005 tok/s.** Below the harness noise floor
(spread 0.027–0.076 tok/s; "anything above 0.1 tok/s is signal"). Worth shipping
only if it comes free with T5.3 cleanup; **do not spend a task on it.**

## Single-threadgroup kernels — blocked by bit-exactness, not bandwidth

Exactly three dispatch shapes are `groups = 1`: the two fused 5376-wide norms
(`1 × 1024` threads) and `input_norm_and_qkv_transform` (`1 × 256`).
**121 dispatches/token, 14.89 MB/token, every byte on 1 of 20 cores.**

Re-shaping is barred: the 256-lane strided accumulation **is** the reduction
order, and `implementation_notes.md` §20.5 records
`partial[lid] = (lid==0) ? sum : 0` changing **8 192/8 192 elements with a worst
error of 9 266 ULP**. There is no two-stage layout that reproduces `partial[0]`
exactly, and Metal has no grid-wide barrier for a one-dispatch two-phase reduction.

T5.6 already took the one bit-exact widening: 256 → 1024 threads with threads
256–1023 contributing exact zeros, for **+0.20 tok/s**.

## Where the money actually is

**`o_proj` runs at roughly half the bandwidth of everything else** —
16.52 MB/layer × 60 = **992 MB/token at ~100 GB/s instead of ~200**, a **~6 ms
deficit**, seven times the entire measured element layer. T5.11 exhausted seven
scheduling variants for it. This is the only line item in this neighbourhood with
the right order of magnitude.

## Ceiling

| | ms/step | tok/s |
|---|---|---|
| measured baseline | 62.3 | **16.05–16.10** |
| GEMV weights alone at the best measured 203 GB/s | **58.3** | **17.2** |
| element layer, measured marginal | 0.84 | |
| element layer, bandwidth floor | 0.25 | |

**20 tok/s needs 12.3 ms. Zeroing the entire non-GEMV layer returns 0.84 ms — 7%
of the gap.** Unreachable by ~8 ms regardless of element kernels. **Realistic
ceiling 16.5–17.0 tok/s**, and at 16k context, 14.5–15.4.

## Test delivered

`crates/engine/tests/perf_probe_attn_o_transform_fusion.rs` — 12 tests,
clippy-clean, `cargo test --no-run` succeeds. **Not executed** (no GPU authorised).
The gate is `combine_then_input_transform_is_bit_exactly_the_fused_host_model`.
Notable design point: **the Hadamard level-order control cannot be done on the
GPU** — two level orders differ by ~1e-7 relative in f32 and `half()` absorbs that
for all but ~1 output in 5 000 — so it is asserted on the **host** with `to_bits()`.
A fused kernel that reordered its butterfly levels or skipped the combine's fp16
store would produce fp16 output that no GPU-to-GPU comparison in this repo could
catch.