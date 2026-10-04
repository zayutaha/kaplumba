# Implementation notes — 90 tok/s prefill / 14 tok/s decode at 24k

Appended as work proceeds.

---

## 0 — Starting point (branch `mtp-speculation`, merged `spec/prefill-120tps`)

Measured with the shipped binary (`target/release/kaplumba`, `regpipe` GEMM +
shape-aware MMA attention):

| | 614 | 8k | 16k | 24k |
|---|---|---|---|---|
| prefill tok/s | 100.56 | 80.52 | 63.96 | 52.84 |
| decode tok/s | ~14.0 | 12.27 | not yet | 0.30–0.43 (T5) |
| peak `phys_footprint` | 12.66 GiB | 14.15 GiB | 15.54 GiB | 17.12 GiB |

## 1 — Decode-at-24k is a collapse, not a slope

8k decode is 12.27 tok/s (measured this session, `--max-new-tokens 16`). 24k is
0.30–0.43 (T5, `--max-new-tokens 8`). That is **32× for 3× the context**.

The cheap model: decode attention reads `k`+`v` for every attended key. Sliding
layers are capped at 1024 keys (verified: `gpu_cache.key_count(full=false) =
next_position.min(capacity)`, and sliding `capacity == 1024`), so they are
context-independent. The 10 full-attention layers read all keys, so their cost
scales 8192 → 23954 = **2.9×**. Predicted 24k decode ≈ 12.27 / 2.9 ≈ **4.2
tok/s**. Measured 0.38 — **~11× worse than the model**. Something else is going
on, and it is R1's job to name it.

Ruled out already (code read, not guessed):

- **Sliding layers scanning full context** — no: `key_count` caps at 1024.
- **Single-pass decode scanning all keys in one workgroup** — no:
  `ATTENTION_SINGLE_PASS_MAX_KEYS = 64`, so 24k always uses split+combine.
- **Split count exploding** — no: `decode_splits` clamps to
  `ATTENTION_SPLITS_MAX = 16` and to `affordable = ceil(key_count/128)`; at 24k
  that is 16. Grid is `splits × kv_heads` = 64 workgroups (full) / 256 (sliding).

Remaining hypotheses, in order of prior:

1. **Memory pressure.** 17.12 GiB footprint on a 24 GiB box; T5 saw the system
   swap grow to ~3 GiB during 24k runs (engine's own `swaps` line was 0). If the
   OS evicts/compresses weight or KV pages, every decode token re-reads them
   from disk. This predicts a *large wall/gap* in the profile with modest kernel
   `own_ms`, and it is the single most likely cause.
2. **A kernel with context-dependent work that should not be.** E.g. a norm or
   append reading the full positions buffer.
3. **Host-side per-token O(L) work.**

Diagnostic in flight: `KAPLUMBA_GPU_PROFILE=1`, 24k prompt, 4 decode tokens,
`/usr/bin/time -l`, log at
`/private/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/p120-int/decode24k-prof.log`.

## 2 — Prefill-at-24k: the arithmetic (why 90 is an attention problem)

At 24k, 52.84 tok/s = 18.93 ms/token. From the T0 profile, GEMM is
context-independent at ~9.6 ms/token-equivalent; the rest (~9.3 ms) is
attention + elementwise.

If **attention were free**, 24k would be 1000/9.6 ≈ **104 tok/s**. So 90 tok/s
needs the non-GEMM part cut from ~9.3 ms to ~1.5 ms — i.e. **≈6× less
attention**, not a GEMM change. That is a hard target but not a GEMM wall; it is
the same attention problem as AC5 in `spec/prefill-120tps`, amplified by 24k's
larger key counts.

## 2b — T1 RESULT: the 24k decode "collapse" was a profiler artifact

The clean (unprofiled) 24k decode run finished:

```
/private/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/p120-int/decode24k-clean.log
prompt_tokens=23954 generated_tokens=8
load_ms=10544.097  ttft_ms=420559.374
prefill_tok_s=56.960  decode_tok_s=5.440
peak_resident_pages_bytes=7953481728
  434.47 real  0.86 user  20.09 sys
  236971 page faults   48222 voluntary ctx switches   187378 involuntary   0 swaps
  18353943224 peak memory footprint   (18.35 GiB)
```

**Decode at 24k is 5.44 tok/s, not 0.30–0.43.** The 0.5–0.4 figure in T5 and
in `spec.md`'s opening table came from the `KAPLUMBA_GPU_PROFILE=1` run, whose
instrumentation distorts the run (`decode_tok_s=0.082` on the profiled 24k run,
vs 5.44 clean — a **66×** artifact). The profiled run also reported
`real 548 s / sys 68.6 s / 1,585,905 voluntary ctx switches`, i.e. the profiler
itself is what was thrashing, not the decode loop.

**OQ1 is answered: no memory cliff.** Engine `swaps 0`; system `sys` time 20 s
over 434 s; 18.35 GiB peak footprint under a 24 GiB box. Decode is **compute/
latency bound in the attention kernels**, not paged out.

**OQ-superseded:** the "~11× worse than the O(L) KV model" framing is dead. With
5.44 tok/s measured and a ~4.2 tok/s O(L) prediction, decode is at or slightly
*better* than the naive model — it is a slope, not a collapse. 14 tok/s at 24k
is therefore a **2.6× kernel-speedup** ask, not a bug hunt. T2 changes shape.

### Decode budget at 24k (184 ms/token)

| term | ms/token | source |
|---|---|---|
| GEMM (context-independent) | 9.4 | 5.75e10 FLOP / 6.13 TFLOPS |
| K+V read for 10 full layers | 7.2 | 4 kv_heads × 23954 keys × 512 dims × 2 (K,V) × 2 B / 273 GB/s |
| K+V read for 50 sliding layers | 3.1 | 16 kv_heads × 1024 keys × 256 × 2 × 2 B × 50 / 273 GB/s |
| **measured total** | **184** | 1000 / 5.44 |

So ~174 of 184 ms is unaccounted for by GEMM + KV bandwidth. The attention
kernels are **latency-bound**, not bandwidth-bound: the decode grid is only
`ATTENTION_SPLITS_MAX(16) × kv_heads(4) = 64` workgroups for the full layers
(256 for sliding), each serially scanning its split with 8 simdgroups of 32
lanes. 512 resident warps on a box that wants ~2–4k in flight.

## 2c — T1 RESULT: prefill budget at 24k, and the exact AC2 arithmetic

From the 24k profile (`decode24k-prof.log`, `KAPLUMBA_GPU_PROFILE=1`), the
kernel sums are cleanly separable even though the run's *wall* is distorted:

| bucket | ms | share of kernel sum |
|---|---|---|
| GEMM (q,k,v,o,gate,up,down) | 213,690 | 52.1% |
| **attention** | **192,984** | **47.1%** |
| everything else (norms, rope, append, gelu, head) | 3,189 | 0.8% |
| kernel sum | 409,863 | |

Per-layer totals separate the two attention shapes exactly (GEMM is identical
across layers):

- **50 sliding layers** (head_dim 256, kv_heads 16, groups 2, window 1024): 3,869–3,942 ms each.
- **10 full layers** (head_dim 512, kv_heads 4, groups 8, no window): 21,519–21,684 ms each.
- Full-minus-sliding ≈ 17.7 s/layer → **full attention ≈ 176.7 s**, sliding ≈ 16.3 s.

The 24k target is 23,954 / 90 = **266 s**. GEMM alone is **213.7 s = 80% of
that budget**. Therefore:

> AC2 requires attention to fall from ~193 s to **≤ 52 s (3.7×)** with GEMM
> completely unchanged. Full attention is the whole problem.

Full-attention FLOPs: causal (query,key) pairs `L²/2 = 2.87e8` × 4 kv_heads ×
8 q_heads × 512 dims × 2 (QK+PV) × 2 FLOP = **1.88e14 FLOP per layer**, ×10
layers = **1.88e15 FLOP over 176.7 s = 1.06 TFLOPS**.

That is **12% of the 8.76 TFLOPS SIMT fp32 peak and 7% of the 15.6 TFLOPS
tensor peak** measured by `examples/mma_peak.rs`. At 6 TFLOPS attention would
cost 31 s and the run would land at 214 + 31 + 16 = 261 s = **91.8 tok/s**. So
AC2 is *arithmetically* reachable, but only through a genuine MMA flash-attention
rewrite; no incremental change to the scalar PV loop or a 4→8 sub-tile publish
gets there.

### Why the current full-attention kernel is slow (read from the source)

`attention_gqa_mma_body` (attention.metal:1128) for HEAD_DIM=512, NSG=8:

1. **QK is MMA but only 4 of 8 simdgroups engage it.** Sub-tiles are owned by
   `sub = sg; sub < 4; sub += NSG`, so `sg ∈ {4,5,6,7}` run zero MMAs and only
   publish/read `sstage`.
2. **PV is fully scalar** and each of the 8 simdgroups re-reads and re-converts
   the *whole* 512-dim V row for every key, because every q-head's output needs
   all dims. 8× redundant device reads and 8× redundant `cvt` per key.
3. The Q fragment is reloaded from `qhstage` on every one of the 64 `d8` steps
   per key-tile, though it is loop-invariant across key-tiles.

All three are fixed by the same change: a flash-style kernel with 64-key tiles
(eight 8-key m-tiles, one per simdgroup) for QK and an **MMA PV**
(`P[8 heads × 64 keys] × V[64 keys × 512 dims]`, V loaded straight from device
with `simdgroup_load` at stride `kv_stride`). That is T3/T4, now the only path
to AC2.

## 4 — T3: the flash prefill body is in, and it is numerically sound

**Files:** `crates/kernels/src/attention.metal` (`attention_prefill_flash_body`
+ `KAPLUMBA_FA_FLASH` + `KAPLUMBA_FA_FLASH(online_attention_flash_512_8, 512, 8)`),
`crates/kernels/src/attention.rs` (`AttentionQk::Flash`, `flash_supported`,
`attention_qk_for(groups, head_dim)`, `attention_flash_pipeline`).

**Design.** One threadgroup per `(query, kv_head)`, 256 threads, same grid as the
MMA arm. Per 64-key tile:

1. **QK** -- eight 8-key m-tiles, `sub == sg`, so all 8 simdgroups issue MMAs
   (the MMA body engaged only 4 of 8). `sth[64][8]` publishes
   `dots.thread_elements()[0/1]` at `(sg*8 + fm)*8 + fn`.
2. **Softmax** -- simdgroup `sg` owns head `sg` and two keys per lane
   (`kt0 = 2*lane`), so the per-head max is one `simd_max` over 64 keys and the
   running sum one `simd_sum(w0 + w1)`. `m` and `s` stay in registers; only the
   per-head rescale `msb[sg]` is published, because the PV accumulators carry all
   8 heads in one 8x8 fragment and need `msb[fm]` per row.
3. **PV as MMA** -- `P[8 heads x 64 keys] x V[64 keys x 512 dims]`, with `V`
   loaded straight from the device by `simdgroup_load(.., kv_stride)`. Each
   simdgroup owns 8 of the 64 n-tiles (`nt = sg + j*NSG`) and holds 8
   `simdgroup_float8x8` accumulators. This is what removes the MMA body's 8x
   redundant V read + `cvt` across the GQA group (item 2 of the T1 read).
4. The straddling K sub-tile and V k-step are staged into one shared 8x512-half
   `tail` buffer (dead rows zeroed), reusing llama.cpp's guarded-tail shape. The
   reuse is safe because the K straddle is consumed before the softmax barrier
   and the V straddle only after it.

Threadgroup memory at 512/8: 8 KB `qstage` + 2 KB `sth` + 1 KB `wq` + 8 KB `tail`
+ 64 B = **19 KB**, the same as the MMA body, under the 32 KB cliff.

**Precision change, stated explicitly.** The MMA body kept `w` in fp32; the flash
body stores `P` as fp16 in `wq`, because `simdgroup_multiply_accumulate` needs a
half operand. That is the standard flash-attention trade and it is why the
assertion below is a tolerance, not `to_bits()`.

**Measured (GPU, `attention_qk_mma_matches_scalar`)** -- new test
`attention_flash_matches_scalar_on_the_full_shape`, 5 full-attention cases
(keys 7 / 31 / 32 / 129 / 1000, up to 3 queries, scale 1.0 and 0.04419):

```
flash full 32q/4kv/d512 (NSG=8) keys=7    -> diff=0.000488 nans=0
flash full 32q/4kv/d512 (NSG=8) keys=31   -> diff=0.000488 nans=0
flash full 32q/4kv/d512 (NSG=8) keys=32   -> diff=0.000488 nans=0
flash full 32q/4kv/d512 (NSG=8) keys=1000 -> diff=0.000488 nans=0
flash full 32q/4kv/d512 (NSG=8) keys=129  -> diff=0.000488 nans=0
worst |flash - scalar| fp16 abs diff: 0.000488 (tol 0.005)
```

0.000488 is one fp16 ulp at these magnitudes -- the fp16 `P` store, as
predicted, and two orders below the 5e-3 the MMA arm already ships with.
`the_default_prefill_path_is_shape_aware` was updated to expect `Flash` on the
full shape (it failed exactly there before the update, which is itself evidence
the selection took effect rather than silently staying on MMA).

**No-op hazard, recorded because it bit this session.** A Metal compile error
fails the *whole* library, so `AttentionKernels::new()` returns `Err`, and every
test in this file takes the `let Some(fx) = ... else { return }` path and
*silently passes*. `attention_flash_matches_scalar_on_the_full_shape` therefore
`panic!`s rather than returning when `flash_attention_available()` is false.

## 3 — Background: the GEMM ceiling (from `spec/engine-performance`)

Relevant to R4 (verify-only) and to why 120 tok/s @614 was reported as a wall in
`spec/prefill-120tps`:

- tensor peak is chain-count dependent (`examples/mma_peak.rs`): 1 chain 5.75,
  8 chains 7.65, **16 chains 15.49–15.64**, 32 (array) 0.74 TFLOPS.
- The shipped GEMM uses 16 **named** accumulators (an array spills) and reaches
  6.13 TFLOPS = 40% of peak. The gap is issue-slot contention between the 16
  MMAs, 10 fragment loads and the ~12-op dequant per step — not MMA throughput.
- Rejected, do not retry (`spec/engine-performance` §14): 64×64 tile, KCHUNK=32,
  staged `astage`, 32 accumulators, 8→4 simdgroups, 18 KB threadgroup, codebook
  LUT, `extract_bits`, gate/up concat, `AGX_RELAX_CDM_CTXSTORE_TIMEOUT`,
  resident weights. `spec/prefill-120tps` T4 adds 4sg and regpipe_k32 to that
  list; `regpipe` is the winner and ships.
