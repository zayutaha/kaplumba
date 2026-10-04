# Plan: 90 tok/s prefill and 14 tok/s decode at 24k

## Architecture touchpoints

| path | current behaviour | intended change |
|---|---|---|
| `crates/engine/src/model.rs` | decode loop, KV append, attention dispatch | depends on R1 diagnosis |
| `crates/engine/src/gpu_cache.rs` | KV ladder 1.25× from 512; full layers grow to 28457 at 24k; sliding capped at 1024 | only if R1 says memory |
| `crates/kernels/src/attention.metal` | scalar `attention_gqa_body`; partitioned MMA `attention_gqa_mma_body` (T2) | query-blocking, or extend MMA |
| `crates/kernels/src/attention.rs` | shape-aware QK selection; decode split+combine | split grid / selection |
| `crates/kernels/src/linear.metal` | `regpipe` MMA GEMM (shipped default) | only if R4 finds a win |

## R1 — decode-at-24k diagnosis (the gate)

One `KAPLUMBA_GPU_PROFILE=1` run at 24k with 4 decode tokens, plus
`/usr/bin/time -l` and `sysctl vm.swapusage`. Discriminate:

- **If decode kernel `own_ms` is small and the wall/gap is large** → host or
  memory. Check `swaps` in `getrusage` and the system swap delta; if the
  footprint is the cause, reduce it (e.g. KV ladder top rung, lm_head mapping,
  scratch) before touching kernels.
- **If the decode attention/compute kernels themselves are seconds** → a kernel
  whose work scales wrongly. Compare per-layer kernel times against the 8k
  profile; the full-attention layers should scale ~3×, sliding stay flat.
- **If a non-attention kernel (cache_append, positions, norm) grows with L** →
  a per-token O(L) cost to remove.

Evidence so far: 8k decode 12.27 tok/s, 24k 0.30–0.43. O(L) predicts ~4.2 tok/s
at 24k (full attention 10 layers × 8192→23954 = 2.9×), so ~11× is unexplained.

## R2 — decode fix

Decision tree from R1. Preferred order (cheapest, highest-confidence first):

1. If a specific kernel is hot at 24k and cool at 8k → fix that kernel.
2. If it is memory pressure → cut footprint (KV ladder, head trellis mapping,
   scratch) OR change the access pattern so the working set is the active KV,
   not the whole resident set.
3. If it is host-side → remove the O(L) host step.

## R3 — prefill attention at 24k

Two candidate levers, in order:

1. **Query blocking.** Today one threadgroup = one query × one KV head, so K and
   V are re-read once per query. Blocking `Q_BLOCK` queries in one threadgroup
   amortises the K/V loads and half→float converts across `Q_BLOCK` queries,
   which is exactly the instruction-bound part the spec blames (57% `cvt.f32.f16`).
   `Q_BLOCK=2` halves K/V load+convert instructions.
2. **Extend the T2 MMA.** The partitioned MMA already wins on `groups >= 4`.
   At 24k the full-attention layers are most of the cost; publish 64 keys per
   barrier (8 sub-tiles across 8 simdgroups) instead of 32 to engage all 8 and
   halve barrier frequency.

Both must keep the online-softmax recurrence and the fp16 store, and must not
change the scalar path's bit-exactness (pin the scalar arm where required, as in
`spec/prefill-120tps` T2).

## R4 — GEMM at 24k

GEMM per token is context-independent; verify, do not re-optimise. The
`spec/prefill-120tps` T3 (weight repack) and the register-pressure ceiling
remain the only open GEMM items and are **out of scope** unless R5 shows GEMM
regressed at 24k.

## Requirement → task mapping

| requirement | tasks |
|---|---|
| R1 decode diagnosis | T1 |
| R2 decode fix | T2 (shape depends on T1) |
| R3 prefill attention | T3 (query-block), T4 (MMA 64-key publish) |
| R4 GEMM at 24k | T5 (verify only) |
| R5 report | T6 |

## Risks

1. **Decode is memory-bound and unfixable by kernel work.** If 17.1 GiB footprint
   forces paging, 14 tok/s at 24k may need the model weights or KV to be smaller.
   That collides with "no model memory increase" (it allows decreases).
2. **Query blocking changes register pressure.** At `head_dim=512`, `Q_BLOCK=2`
   doubles accumulators; if it spills, `Q_BLOCK=1` stays.
3. **24k runs take 8–9 minutes each.** Measurement budget dominates; interleave
   and cache the baseline.
