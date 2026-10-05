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

## 5 — Is AC2 (90 tok/s @24k) reachable at all? The arithmetic, before more kernel work

This is the question OQ2 left open, and it has a sharp answer once the FLOP and
byte counts are written down for the *full* layers at L = 23954, which is 100%% of
the problem (sliding attention is 16 s of 193 s).

Per layer, per `(query, key)` causal pair and per KV head, the work is
8 q-heads × 512 dims × 2 (QK) + 8 × 512 × 2 (PV) = **16384 FLOP**.

| quantity | value |
|---|---|
| causal (query,key) pairs | L²/2 = 2.87e8 |
| KV heads (full) | 4 |
| FLOPs per layer | 2.87e8 × 4 × 16384 = **1.88e13** |
| FLOPs, 10 full layers | **1.88e14** |
| bytes read per visit (one query re-reads its whole causal K and V) | 512 × 2 (K,V) × 2 B = **2048** |
| bytes per layer | 2.87e8 × 4 × 2048 = **2.35e12** |
| bytes, 10 layers | **23.5 TB** |

At the measured 273 GB/s of this box that is **86 s of pure K/V streaming** -- and
that is a *lower bound*, because it is the ideal case where no byte is fetched
twice. Measured attention is 176.7 s, so the flash body's remaining headroom is
at most ~2×, i.e. ~90 s in the limit.

> **Therefore, with the shipped GEMM, AC2 is unreachable even with a perfect
> attention kernel.**
>
> ```
> GEMM (measured, context-independent)   213.7 s
> attention lower bound                   86.0 s
> everything else                          3.2 s
>                                        -------
> floor                                  302.9 s -> 79.1 tok/s
> required for 90 tok/s                   266 s
> ```

To reach 90 tok/s at the 86 s attention floor the GEMM would have to fall to
180 s, i.e. **7.6 TFLOPS**, against today's 6.6. `spec/prefill-120tps` already
declared the 7.5 TFLOPS target a wall (`spec/engine-performance` §12/§14: 16 named
accumulators is the register ceiling; 64x64 tiles, KCHUNK=32, staged `astage`, 32
accumulators, 8->4 simdgroups and the codebook LUT were all tried and rejected).

### The one design change that moves the lower bound: query blocking

The 23.5 TB above exists only because **every query re-reads its whole causal K and
V**. A threadgroup that owns `BQ` queries of one KV head reads each key once and
reuses it across `BQ` queries, so the byte count divides by `BQ`:

| BQ | bytes/layer | 10 layers | @273 GB/s | FLOP-bound? | floor total | tok/s |
|---|---|---|---|---|---|---|
| 1 (today) | 2.35e12 | 23.5 TB | 86 s | compute 31 s | 303 s | 79 |
| 2 | 1.17e12 | 11.7 TB | 43 s | 31 s | 260 s | 92 |
| 4 | 5.9e11 | 5.9 TB | 22 s | 31 s | 248 s | 96 |

(the FLOP column is 1.88e14 / 6.0 TFLOPS = 31 s; past BQ=2 memory stops being the
binding constraint and the tensor pipe is.)

So the path to AC2 is **query blocking with `BQ >= 2`**, not more MMA throughput.
Cost/risk, measured against the current body:

* `qstage` becomes `BQ * NSG * HEAD_DIM` halves. BQ=2 is 16 KB, and with `sth`
  2 KB + `wq` 2 KB + `tail` 8 KB that is 28 KB of the 32 KB cliff. BQ=4 is 32 KB
  on its own and does not fit without dropping `tail` or shrinking the tile.
* The PV accumulator doubles to 16 `simdgroup_float8x8` per simdgroup (32
  registers); `msb`/`ssum` become per `(query, head)` pairs.
* The softmax needs `BQ` independent running `(m, s)` pairs, i.e. 2 keys x BQ per
  lane per tile.

That is a larger and riskier change than T3 was, and BQ=2 clears 90 only at the
ideal bandwidth; at a realistic 70-85%% of peak the floor is 300-340 s. It is the
right next step, but AC2 should be treated as **not demonstrated** until it is
measured.

## 6 — Decode: the same lower bound, and why T2 is also a rewrite

Decode at 24k is 5.44 tok/s = 184 ms/token against a 9.4 ms GEMM. Per token the
full layers must stream their whole K and V once (there is only one query, so
*no* read is redundant):

| term | bytes | @273 GB/s |
|---|---|---|
| 10 full layers | 2 × 10 × 23954 × 4 × 512 × 2 = **1.96 GB** | 7.2 ms |
| 50 sliding layers | 2 × 50 × 1024 × 16 × 256 × 2 = **0.84 GB** | 3.1 ms |
| GEMM | -- | 9.4 ms |
| **total** | 2.8 GB | **19.7 ms = 51 tok/s** |

So decode at 24k should be ~51 tok/s from bandwidth alone and is measured at 5.44.
That is a **9.3x gap with no redundant byte to remove** -- it is purely latency
and occupancy: the full layers dispatch only `16 splits x 4 kv_heads = 64`
workgroups, each 8 warps serially walking 47 32-key tiles with a 128-iteration
dependent scalar dot per key. 14 tok/s (71 ms/token) is above the 19.7 ms floor,
so it is reachable, but only by giving that work more parallelism or an MMA QK --
not by tuning.

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

## 7 — T4: query blocking, and the instrument that explains the attention budget

### 7.1 What was built

`attention_prefill_flash_body` is now templated on `BQ` (queries per threadgroup),
with two kernels instantiated from the same macro:

```
KAPLUMBA_FA_FLASH(online_attention_flash_512_8,     512, 8, 2)   // BQ = 2
KAPLUMBA_FA_FLASH(online_attention_flash_bq1_512_8, 512, 8, 1)   // BQ = 1
```

Both share the grid geometry `ceil(queries / BQ) * kv_heads`. The block is chosen
per request by `flash_query_block_for(key_count)` (`KAPLUMBA_FLASH_BQ=1|2`
overrides it), and `encode_online_attention_qk_block` takes it explicitly so a test
covers both kernels in one process. Arm selection for the full shape only is
`KAPLUMBA_FULL_QK=flash|mma|scalar`; the global flags cannot express a
full-layer-only A/B because they also move the 50 sliding layers.

### 7.2 Correctness and the sabotage check (AGENTS.md 3.4)

`attention_flash_matches_scalar_on_the_full_shape` now runs **both** BQ=1 and BQ=2
and adds two 5-query cases (`(FULL, 513, 5, 513)` and `(SLIDING, 513, 5, 480)`) so
the BQ=2 grid tail (`ceil(q/BQ) * kv_heads`, one dead query in the last group) and
the per-query softmax state are load-bearing. Worst `|flash - scalar|` fp16 abs
diff over 6 full-shape cases x 2 blocks: **4.88e-4 (one ulp)**, tol 5e-3.

Sabotage (dropping the per-query index from `msb`: `msb[b*NSG+sg]` -> `msb[sg]`):

| case | BQ=1 | BQ=2 |
|---|---|---|
| keys 7 / 31 / 32 (1-3 queries) | 0.000488 | 0.000488 (b=0 only, unaffected) |
| keys 1000 / 129 / 513, queries 2,2,5 | 0.000488 | **0.744 / 0.575 / 0.745** |

The sabotage fails **only** the BQ=2 arm on the multi-query cases, and the test
fails with its diagnostic. Reverted; suite green.

### 7.3 Measured: query blocking does **not** pay. (E2E, interleaved)

`measure-bq.tsv` (round 1 of 2; the 24k pair is complete, the script was stopped
after it because the result was decisive):

| prompt | BQ=1 | BQ=2 |
|---|---|---|
| 614 | 99.671 | 99.472 |
| 8k | 83.912 | 82.436 |
| **24k** | **63.154** | **59.250** |

BQ=2 is **-1.8% at 8k and -6.2% at 24k**. The K/V byte-traffic model in §5 (which
predicted a 2x reduction in the dominant term) is therefore **not** what limits
this kernel; halving the loads while doubling the per-threadgroup critical path and
the register footprint is a net loss. Recorded as a negative result.

### 7.4 The instrument: `attention_arm_throughput_at_prefill_geometry`

New test in `crates/engine/tests/attention_qk_mma_bench.rs`. No model load; real
dispatch geometry (`queries = 1024` = `PREFILL_CHUNK`, one threadgroup per
`(query, KV head)`); FLOPs = `queries * q_heads * keys * head_dim * 4` (QK + PV,
mul-add counted as 2), every query positioned at/past `key_count` so nothing is
masked. ~25 s for all four arms at two key counts.

| arm | 8192 keys | 24576 keys |
|---|---|---|
| scalar | 397.34 ms (1.38 TFLOP/s) | 1243.66 ms (1.33) |
| **mma** | **253.21 ms (2.17)** | **880.21 ms (1.87)** |
| flash bq1 | 360.50 ms (1.52) | 1184.23 ms (1.39) |
| flash bq2 | 384.45 ms (1.43) | 1216.92 ms (1.36) |

### 7.5 Why attention is at 24% of the MMA ceiling: MMAs per fragment load

The GEMM sustains 6.75 TFLOP/s; attention sustains 1.4-1.9. Both are 8x8x8
`simdgroup_multiply_accumulate`. The difference is **operand reuse**: a
`simdgroup_load` is a real instruction, so what caps the loop is MMAs per load.

| loop | M-tiles x N-tiles | loads / MMAs | MMA/load |
|---|---|---|---|
| `linear_gemm_mma_regpipe` (QTILE 128 / OUTS 64, per k-step) | 2 x 8 | 10 / 16 | **1.60** |
| flash QK (`dots[key][head]`, 64). BQ=1 | 1 x 1 | 2 / 1 | 0.50 |
| flash QK, BQ=2 | 1 x 1 | 3 / 2 | 0.67 |
| flash PV (`P[head][key] x V[key][dim]`), BQ=1, P hoisted | 1 x 8 | 9 / 8 | 0.89 |
| flash PV, BQ=2, P hoisted | 1 x 8 | 10 / 16 | **1.60** |
| flash PV, BQ=2, P loaded per n-tile (the T4 first cut) | 1 x 8 | 24 / 16 | 0.67 |

1.60 / 0.50 = 3.2, against a measured 6.75 / 1.87 = 3.6. **The load-reuse model
explains the attention deficit within 12%, and it is the thing to fix.**

Two consequences, both acted on:

1. The first BQ cut loaded the P fragment *inside* the n-tile loop (to make the
   query the inner axis), which dropped PV from 0.89 to 0.67 at BQ=1 and 1.60 to
   0.67 at BQ=2. Fixed: the `BQ` P fragments are preloaded once per k-step and
   reused across all `NPER` n-tiles, restoring 0.89 / 1.60. This is a real
   regression the bench caught and the E2E could not have attributed.
2. The next raise is the QK's M-tiles. It is currently 1 (8 keys); heads (N) is
   fixed at 8, so the only axis that can grow is keys per simdgroup. Going to
   M=2 (16 keys per simdgroup, TILE 128) takes QK to 4 loads / 4 MMAs = 1.00 at
   BQ=2, and PV to 16 x 10 / 256 = 0.63... see §8 for the arithmetic actually
   run.

## 8 — T5: what actually limits the flash prefill attention (four experiments)

The §7.5 load-reuse model predicts MMA/fragment-load is the limiter. Four
experiments, each measured twice (bench, and the engine's own
`KAPLUMBA_GPU_PROFILE=1` attention kernel sum), and **three of the four
contradicted it**. Recorded in order, because the corrections are the useful
part.

### 8.1 `KSUB = 4` (256-key pass, 4 m-tiles per simdgroup): 13% SLOWER

Halves the QK's fragment loads (0.50 -> 1.33 MMA/load) by amortising the Q
fragment over four MMAs. It needs 28 KB of threadgroup memory instead of 19.5.

| | bench 24576 keys | engine attention @8k |
|---|---|---|
| `KSUB = 1` | 1118 ms | 20 623 / 20 674 ms |
| `KSUB = 4` | 1318 ms | 23 377 ms |

**The loads were not the limiter; occupancy is.** Threadgroup memory is per
threadgroup, so 28 KB vs 19.5 KB directly cuts how many threadgroups fit on a
core, and the kernel was latency-bound, not issue-bound. `spec/engine-performance`
§14 already lists "18 KB threadgroup" among its rejected experiments; this is the
same wall seen from the other side. Reverted; `FLASH_KEY_SUBTILES = 1`.

### 8.2 Query blocking, re-measured after fixing the PV loop order: **-2.0%**

The §7.3 verdict ("-6.2% at 24k") was measured with the broken PV loop order
(§7.5 consequence 1). With `BQ` P fragments preloaded once per k-step:

| | engine attention @8k | engine attention @24k |
|---|---|---|
| `BQ = 1` | 19 403 / 20 623 ms | 144 961 ms |
| `BQ = 2` | 18 996 ms | 141 994 ms |

So query blocking is a **small win once the loop order is right** (-2.0%), not a
loss -- the §5 byte-traffic model overstated it by 3x, but it had the sign right.

### 8.3 `exp` -> `fast::exp2(x * log2 e)`: within noise, kept

The online softmax calls `exp` once per (query, key, head). `exp` lowers to a
~15-op polynomial; `fast::exp2` is one MUFU.EX2 plus a multiply. Measured
1374 -> 1363 ms on the bench (0.8%). Kept because it is strictly less work and the
weights are stored as **fp16** for the PV fold anyway -- 2^-11 relative, against
MUFU.EX2's ~2^-21 -- so it is two orders of magnitude tighter than the precision
the kernel already accepts.

### 8.4 Dropping the 8 KB `tail` straddle buffer: **the tests caught the obvious
### version, and the bench says it is not a win either**

Occupancy said to cut threadgroup memory (19.5 -> 11.1 KB). Two attempts:

1. *Clamp a straddling fragment's first row to `range_end - 8`.* **Wrong**, and
   `attention_flash_matches_scalar_on_the_full_shape` caught it immediately: diff
   0.57 on the `keys=7` case, 0.51 on `keys=31`. The clamp re-reads the wrong
   rows for the keys *inside* the straddling sub-tile that are still live (at
   `range_end = 129` the key 128 sub-tile re-reads key 121's scores). A test that
   only checked the common `key_count % 64 == 0` case would have shipped this.
2. *Round the tile loop up to a whole 64-key pass and read the trailing fragments
   unconditionally out of **zeroed** padding* (a variant that had `GpuKvCache`
   `new`/`ensure_capacity` allocate and clear `64 * stream_len` halves past
   `capacity`). Correct -- worst diff back to **4.88e-4** -- and branch-free. But
   on the bench it measured **slower**, and adding an `if (k0 >= range_end)
   continue` guard inside the `d8` loop measured **23% slower** than staging,
   because it breaks the load/MMA pipelining across `d8`. Net: not a win.

**This attempt was reverted in full, including the cache padding.**
`GpuKvCache::new` today allocates exactly `capacity * stream_len` halves
(`gpu_cache.rs`), and the flash body's tile loop is `for (base = 0; base <
range_end; base += 64)` with no rounding. The kernel is out-of-bounds-safe by
construction, not by padding: every K fragment load is guarded by `k0 + 8 <=
range_end`, the one fragment that straddles `range_end` is staged into `tail`
with dead rows zeroed, and the PV k-step that straddles is staged the same way.
An earlier draft of this note claimed the padding "is still there"; it is not,
and neither is the rounded loop. The staged `tail` buffer stays because it is
not slower than the barrier-free variants above.

### 8.5 The honest picture at 24k

Engine-measured, current build, `KAPLUMBA_GPU_PROFILE=1`, 23 954-token prompt:

| term | ms | TFLOP/s |
|---|---|---|
| GEMM (q,k,v,o,gate,up,down) | 214 842 | 6.44 |
| attention (all 60 layers) | 144 962 | 1.30 |
| everything else | ~3 200 | |
| **kernel sum** | **363 003** | |
| inter-dispatch gap | ~24 000 | |
| wall | ~388 000 = **63.2 tok/s** | |

Attention is 91% full-attention layers and runs at 1.30 TFLOP/s. The GEMM is at
6.44 against the ~7.8 TFLOPS structure-matched ceiling
(`spec/engine-performance` §12/§13). Neither is within 2x of what it would take

### 8.6 Progress vs the spec baseline (T3 + T4 so far)

The spec's opening table measured 24k prefill at **52.84 tok/s** (`882d874`). The
current build's 24k wall is ~388 s for 23 954 tokens = **~63.2 tok/s** (§8.5), so
`regpipe` + the flash body have moved 24k prefill by **~+20%**. The 8k number has
moved 78.82 -> ~85 (T3). Nothing here reaches 90 yet; the remaining arithmetic is
in §5 and §8.5.

### 8.7 The arm-throughput bench does **not** rank arms; E2E does

§7.4/§8.2 used `attention_arm_throughput_at_prefill_geometry` to argue the MMA arm
beats flash. Its absolute numbers are real, but its **relative ranking does not
reproduce end to end**, and this is the second time the bench and the engine have
disagreed (the first was the §7.3 BQ verdict).

Interleaved E2E with `KAPLUMBA_FULL_QK=flash|mma` (the override moves only the 10
full layers, so the 50 sliding layers are identical in both arms):

| prompt | flash | mma |
|---|---|---|
| 614 (3 rounds) | 101.46 / 101.49 / 101.34 | 101.60 / 98.29 / 101.49 |
| 8204 (2 rounds) | 80.09 / 81.80 | 79.89 / 79.55 |

**Tied.** flash is marginally steadier at 614 and marginally ahead at 8k; the
bench's `mma` advantage (2.14 vs 1.58 TFLOP/s at 8k, 1.89 vs 1.47 at 24k) does not
appear at all. At 8k, attention is ~17 s of a ~102 s prefill, so a genuine 28%
attention win would be ~+3.7% E2E (80 -> 83); it is not there.

One earlier single-shot 614 flash reading of 85.46 (the first thing measured this
session) was a **cold-run outlier**: three repeats give 101.3-101.5. Single
readings at short context are not results.

Consequence for method, not just for this arm: **the bench's TFLOP/s is a ceiling
instrument, not a ranking instrument.** Any arm-selection claim must come from
`KAPLUMBA_FULL_QK` E2E (or an equivalent), because the bench geometry
(rectangular, every query past `key_count`) is not the engine's (chunked,
triangular) and the two orders diverge. This supersedes the "mma is 28% faster"
reading of §7.4/§8.2.

The 24k A/B (flash vs mma at 24 576 keys) is in flight; 614 and 8k being tied
predicts it is tied too, and attention being ~37% of the 24k wall would have made a
real 28% attention win a ~10% E2E win.

### 8.8 Session hazard: the box is shared, and a contended 24k run thrashes

A 24k A/B launched while another agent held the model never progressed: after
13 minutes the `kaplumba` process had used **15 s of CPU and 652 MB RSS**, while
`top` showed 23 GiB used, 112 MiB free and **4.8 GiB of swap**. It was faulting
the model in behind a competing instance. `scripts/gpuq try` had said "ok", and
it was "ok" at the instant it was asked -- a parallel agent can take the model
*after* the status check and before the load completes. A run that does not reach
its output must be re-checked for CPU time and `vm.swapusage`, not assumed slow.
to reach 90 tok/s (266 s) without cutting ~125 s out of these two numbers.

## 9 — T4e: where the flash kernel's time actually goes (ablation, 24 576 keys)

> **WITHDRAWN TWICE, AND THE SECOND RETRACTION IS THE SERIOUS ONE (F1, §22).**
>
> 1. **§10** re-ran this table against the post-T4f body and found none of it
>    reproduces. Do not quote the cost split or anything derived from it.
> 2. **F1 found that neither §9's table nor §10's re-run ever measured the
>    kernel.** `attention_prefill_flash_body` dispatched with
>    `threads_per_group = 256` (`attention.rs:1609`), so §10's
>    `if (tid < 256u) { return; }` was **not** "it only retired 256 threads" — it
>    retired *every* thread, i.e. a genuinely empty kernel, and still read
>    **964.79 ms** against a 958.39 ms baseline. An empty body at that grid, in
>    the same binary, costs **0.19 ms** (§22, arm 1) — a **6 162x** signal. So the
>    patched `.metal` did not reach the compiled shader, and the table is not a
>    failed reproduction: **it is a measurement of the wrong binary.**
>
> The reason §10.1's ruled-out list missed it is worth recording: the checks
> there were "the build returns 0", "`strings` finds `KAPLUMBA_ABLATE_exit` in the
> binary", and "an `eprintln!` prints the right pipeline". None of them
> establishes that **the specific edited lines** are in the shader being timed, and
> the `.metal` is `include_str!`'d into the crate
> (`attention.rs:12`), so an edit that is not accompanied by a recompile of that
> crate silently does nothing. §22 fixes this by carrying the ablation selector
> in the **uniform block**: one build, one binary, every arm.
>
> **Consequence for what was decided on this table.** §9.1 ("the loads are
> near-minimal -- and that is the problem"), §9.1b ("BQ=2 with the grid
> restored") and §9.2 ("what was tried against this split") were all reasoned
> from a 44% loads / 52% MMA / 3% softmax split that no measurement ever
> produced. They are withdrawn as a design basis, not merely as a number. The
> correct split, measured in §22 with a working instrument, is **56% QK / 40% PV
> / 4% everything else, with all fragment loads together worth 12-19%**.

§7.5 and §8.1 blamed `simdgroup_load`s per MMA and occupancy. Both were wrong, and
the reason they were wrong is that neither was measured on the kernel's real cost
split. This is that split, from ablations in `attention.metal`, each measured on
the no-model bench (`attention_arm_throughput_at_prefill_geometry`, BQ=1, 1024
queries, 24 576 keys, warm, median of 3):

| variant | ms | what it removes |
|---|---|---|
| baseline | 1148 | -- |
| `ksteps = 0` (no PV at all) | **433** | PV MMAs, `af`/`bf` loads, rescale, straddle staging |
| QK loop disabled | 689 | QK MMAs, `kf`/`qf` loads, `sth` publish |
| **all MMAs and all their loads off** | **36.7** | everything except softmax, barriers, tail staging, epilogue |

**The softmax, the barriers and the straddle staging are 3% of the kernel.** 96%
is MMA issue plus the fragment loads that feed it. The kernel runs at 1.44 TFLOP/s
= **9% of the 15.6 TFLOPS tensor peak** measured by `examples/mma_peak.rs`.

Two sub-measurements separate loads from MMAs. Forcing the K and V device loads to
re-read one cached row (same instruction count, same cache lines):

| variant | ms | delta |
|---|---|---|
| baseline | 1148 | -- |
| `kf` from a fixed row (K load cached) | 977 | **-171** |
| `bf` from a fixed row (V load cached) | 810 | **-338** |

So the strided device loads are ~509 ms of 1148 = **44%**, and the MMA issue is
~600 ms = 52%. Neither dominates, so no single change can reach 3-4x.

### 9.1 The loads are near-minimal -- and that is the problem

> **WITHDRAWN AS A BASIS (F1, §22).** The byte arithmetic below is *correct* -- a
> `simdgroup_half8x8` is 32 lanes x 2 halfs = 128 B and the sub-tiles do not
> overlap -- but the conclusion drawn from it ("44% of the kernel is these
> strided device loads, so this is the lever") came from the invalid ablation.
> Measured directly in §22: making **every** fragment load L1-resident buys
> **8-9%**, and deleting the Q/P threadgroup loads outright buys **11-12%**. The
> loads are not 44% of anything; they are ~12% of the kernel. The "175 GB/s =
> 64% of DRAM" line is also wrong in the same way §12 later showed: those are L1/L2
> requests, not DRAM traffic.

Bytes read by the K/V loads, from the loop counters. One `simdgroup_half8x8` is
8x8 halfs = **128 B**, and the sub-tiles are partitioned across simdgroups
(`k0 = base + sg*8`, `nt = sg + j*NSG`), so there is **no redundancy**:

| load | per threadgroup per 64-key tile |
|---|---|
| `kf` | 8 sg x 64 d8 x 128 B = 64 KB |
| `bf` | 8 sg x 8 nt x 8 kt x 128 B = 64 KB |
| unique K and V for 64 keys x 512 dims | 64 KB each = 128 KB |

Exactly minimal. Over 384 key tiles and 4096 threadgroups (1024 queries x 4 KV
heads) that is **201 GB in 1.15 s = 175 GB/s**, i.e. **64% of the 273 GB/s DRAM
rate** -- so this kernel is *partly bandwidth-bound after all*, contrary to §9's
first reading (an earlier draft of this section miscalculated a fragment as
1024 B and concluded "122 TB/s, therefore cache-resident"; the fragment is 128 B
and the conclusion was wrong).

This is why the ablation split is 44% loads / 52% MMA issue: the two are
*balanced*, and neither can be removed without the other becoming the wall. It is
also why cutting threadgroup memory (§8.1, §8.4) and removing `qstage` (§9.2) did
nothing: those target occupancy, and the kernel is not occupancy-limited at 4096
threadgroups.

### 9.1b Query blocking is the right lever used the wrong way

> **RE-OPENED AND THEN CLOSED AGAIN (F1, §22).** §12.2 correctly re-opened this
> on grounds that do not depend on §9 (the kernel is instruction-bound, so what
> matters is MMAs per fragment load). §22 is the measurement that closes it:
> BQ=2 is **19% slower** than BQ=1 on the bench at 24 576 keys and 4-12% slower
> at 8 192, and the fragment loads that query blocking halves are worth 8% in
> total. Query blocking cannot pay by removing bytes when the bytes are 8%.
> Its measured verdicts stand: naive BQ=2 is −2.0% E2E at 8k (§8.2) and the
> key-split + combine pass that would restore the grid is not worth building.

BQ=2 halves the K/V bytes (each threadgroup reads a key once and reuses it for 2
queries) but also halves the grid, 4096 -> 2048. §8.2 measured it as a 2% loss.
The two effects cancel: the byte win is real, the parallelism loss is real. The
correct form is **BQ=2 with the grid restored** -- split the key range too, so
2048 query-pairs x 2 key-halves = 4096 threadgroups, each reading half the bytes
of the BQ=1 shape. That needs a partial/combine pass (flash-decoding style) and
is the only change identified in this section that attacks the 44% load term
without giving up the parallel term.

### 9.2 What was tried against this split, and the verdicts

> **EVERY ROW BELOW IS UNRELIABLE (F1, §22).** All four are timed against the
> same 1148 ms baseline that §22 shows is not the kernel's cost under ablation --
> each edit was a separate one-line `.metal` change, and §10's `tid < 256u` arm
> demonstrates that at least one of them never reached the shader at all. None of
> these verdicts may be cited, including "neutral" and including the two that say
> something is slower. The padding row and the `qstage`-removal row happen to
> agree with §22's finding that the loads are cheap; the K-fragment software
> pipeline row and the two-accumulator row are simply unmeasured.
>
> Re-measured properly in §22: the K-fragment pipeline is not a cost (loads are
> 8%); the two-accumulator idea is not a cost (**2 and 4 independent QK
> accumulator chains are 1.00x and 0.96x** -- arms 11/12).

| change | result | why |
|---|---|---|
| `+8` half bank padding on `qstage`/`wq` instead of 1024 B / 128 B strides | 1111 ms (neutral) | the loads are not bank-conflict limited |
| 2 independent QK accumulators (`dots` + `dots2`), summed once | 1126 ms (~-2%) | the dot is a 64-deep dependent MMA chain, but ILP is not what caps it |
| K fragment software-pipelined one `d8` ahead | 1166 ms (**slower**) | the extra live fragment costs more than the overlap buys |
| **remove `qstage` entirely; load Q transposed from the device** | 1151 ms (neutral) | the transposed 8x8 load spans exactly the 8 heads (`groups == NSG == 8`, `head0 = key_head*8`), so it needs no staging -- but threadgroup memory 19.5 -> 11.5 KB did not change the time |

### 9.3 The conclusion that matters for AC2

**Attention is bandwidth- and compute-balanced, with no single dominant term and
no overlap between the two.** 44% loads + 52% MMA issue *sum* to 96% instead of
`max(44, 52) = 52%`, which means the strided K/V loads and the tensor work are
not overlapping: a threadgroup sits on a fragment load, then sits on an MMA.
At 175 GB/s and 1.44 TFLOP/s it is using 64% of DRAM and 9% of tensor peak -- so
there is headroom on *both* pipes simultaneously, which is the signature of a
pipeline that is not pipelined, not of a resource that is full.

The three structural fixes the plan named are all measured neutral or negative:
occupancy (threadgroup memory, `qstage`), operand reuse (KSUB=4, `FLASH_KEY_SUBTILES`)
and bank conflicts (padding) -- see §9.2, §8.1, §8.4. So the fix is not a better
`simdgroup_load` choice inside this tile shape.

Two levers remain, in the order they attack the cost split:

1. **BQ=2 with the grid restored** (§9.1b): halves the 44% load term without
   giving up the threadgroups. This is the change the byte model always wanted;
   §8.2 lost because BQ alone halves the grid too. Needs a partial/combine pass,
   for which the decode split path (`ATTENTION_SPLITS_MAX`, `decode_partial_elements`)
   is the existing pattern to copy.
2. **Overlap loads with MMAs**: software-pipeline the *whole* per-tile body
   (one tile's MMAs against the next tile's fragment loads) rather than one `d8`
   step ahead, which measured slower (§9.2). This targets the 48% that is
   currently idle.

### 9.3b The bench is valid for *work*, invalid for *ranking* -- reconciled

A useful distinction, because both readings are true at once:

- **FLOP/time: the bench predicts the engine exactly.** 1.44 TFLOP/s on the bench
  at 24 576 keys; the engine's full-attention FLOPs (1.88e14 over 10 layers) at
  1.44 TFLOP/s = 131 s, against 132 s measured by the profile. Byte traffic also
  matches: 23.5 TB / 132 s = 178 GB/s engine vs 175 GB/s bench.
- **Arm ordering: the bench is wrong three times out of three** (§8.7, §9.4).
  `mma` 28% faster on the bench, `flash` faster end to end at 614, 8k and 16k.

So `attention_arm_throughput_at_prefill_geometry` can be trusted for "how much
time does this shape cost" (which is what AC2's budget arithmetic rests on) and
must not be used to choose an arm. The likely cause of the ordering inversion is
that the bench's KV tile is hot and resident across all three repeats, while the
engine interleaves attention with a GEMM that streams 11 GB of weights per chunk
and evicts it.

### 9.4 T4e: flash beats mma end to end at 16k, where attention is a third of the wall

| prompt | flash | mma |
|---|---|---|
| 16 394 | **69.36** | 66.95 |
| 24 590 | **62.87** | not run (OQ5: see below) |

A 3.6% full-layer-attention win, which is the largest E2E arm separation measured
at any context. It confirms `flash` as the shipped arm (§8.7) at the one context
where the choice is worth 3.6% rather than noise -- and it is the **third**
confirmation that the TFLOP/s bench cannot rank arms: the bench says `mma` is 28%
faster than `flash` at 24 576 keys (869 ms vs 1148 ms), and end to end `flash` is
**faster**. The bench measures a bare kernel on a hot, resident KV tile; the
engine measures it interleaved with a GEMM that streams 11 GB of weights per
chunk, and the two orderings differ.

### 9.5 Current E2E numbers (this session, interleaved, 2 rounds where taken)

| prompt | prefill tok/s | |
|---|---|---|
| 626 | 101.5 | |
| 8 204 | 80.1 / 81.8 (flash), 79.9 / 79.6 (mma) | |
| 16 394 | **69.36** | |
| 24 590 | **62.87** | target 90 |

GEMM is context-independent at ~9.4 ms/token, so at 24k it is ~225 s of the
~381 s wall (59%). Non-GEMM cost grows with context (3.1 -> 5.0 -> 6.5 ms/token
over 8k/16k/24k), consistent with attention being linear in context. Closing
24k from 62.87 to 90 needs 381 s -> 273 s, i.e. **-108 s (28%)**, and GEMM alone
is 225 s of it.

---

## 10 — T4g: the T4e ablation does not reproduce, and the microbench cannot be
## used for compile-time ablation at all

> **THE WHOLE SECTION IS WITHDRAWN (F1, §22) — including its conclusion that the
> microbench "cannot be used for compile-time ablation".** The table below did not
> measure a failed reproduction; it measured **the wrong binary**. Proof, in one
> line: the flash prefill path dispatches `threads_per_group = 256`, so this
> section's `if (tid < 256u) { return; }` was a completely empty kernel, and an
> empty kernel at this grid costs **0.19 ms** (§22 arm 1, same binary as the
> baseline) against the **964.79 ms** recorded here. A 6 162x discrepancy.
>
> Two consequences that reverse this section's own conclusions:
>
> * **The microbench is a perfectly good attribution instrument.** One uniform
>   field turns it into a full ablation sweep in one binary (§22). What it still
>   cannot do is **rank arms end to end** (§8.7, §9.4, §11) -- that half of
>   §10.2 #3 survives untouched, and it is the half that matters.
> * **§10.2 #3 as written ("a valid throughput instrument and an *invalid*
>   attribution instrument") is wrong and is replaced in §22.** The confusion is
>   between *attribution* (which arm's cost is where -- this works) and *ranking*
>   (which arm is faster end to end -- this does not).
>
> Kept below as the record of what produced the conclusion that had to be undone.

Same bench (`attention_arm_throughput_at_prefill_geometry`), same shape (full
32q/4kv/d512), 24 576 keys, 1024 queries, BQ=1 arm `flash bq1`, median of 3,
`scripts/gpuq try -- cargo test`, `| grep` filtering only, tree restored with
`git checkout` after every arm:

| variant | T4e (§9) | T4g (this section) |
|---|---|---|
| baseline | 1148 | **958.39** |
| `ksteps = 0` (no PV) | 433 | **966.88** |
| QK loop disabled | 689 | **959.90** |
| both off | 36.7 | **941.88** |
| quarter of the key tiles | -- | 956.17 |
| eighth of the key tiles | -- | 947.33 |
| `if (tid < 256u) { return; }` at the top of the body | -- | **964.79** |
| 400 000-iteration float spin at the top of the body | -- | 962.77 |
| 3 000 000-iteration threadgroup-memory spin | -- | **4 828.96** |
| `mma` arm (untouched control, every run) | 869-871 | 867-894 |

T4f explains the baseline (1148 -> 958 is the `nt` reordering). It explains
nothing else. **Removing the PV MMAs and their V loads makes the kernel 0.9%
slower. Removing the QK MMAs and their K loads makes it 0.2% faster. Removing
both, or returning from the body before any of it, changes nothing.** An added
spin loop *does* cost time, so the body is executing.

### 10.1 What was ruled out, and how

Each of these was checked and failed to explain it, so the next person does not
repeat them:

* **The patches did not build.** Ruled out. `cargo build --release -p
  kaplumba-engine --tests` returns **0** with zero `^error` lines, and
  `strings target/release/deps/attention_qk_mma_bench-738fbc382e39fb35 |
  grep -c KAPLUMBA_ABLATE_exit` is **1** — the patched source is in the binary that
  runs. (Note: the first version of the harness read `rc=$?` after a pipe into
  `tail`, so `rc` was `tail`'s status and always 0. That bug is what made the
  first two ablation rounds look clean. Always capture the status of the command
  you care about.)
* **The patches did not reach the executing kernel.** Ruled out for `ksteps = 0`
  by construction: the same edit makes
  `attention_flash_matches_scalar_on_the_full_shape` **FAIL** (`test result:
  FAILED. 2 passed; 1 failed`), so the edited body is the one producing output.
* **The bench was not dispatching the flash kernel.** Ruled out by an
  `eprintln!` inside `encode_online_attention_qk_block`:
  `qk=Flash block=1 ... flash_pipeline=true bq1_some=true bq2_some=true` and
  `qk=Flash block=2`. The `block == 1 -> attention_flash_bq1_pipeline` /
  `block == 2 -> attention_flash_pipeline` selection resolves, for
  `head_dim=512 groups=8`, i.e. `flash_supported` is true and both pipelines
  exist. Neither arm silently falls back to MMA.
* **The cost is host-side.** Ruled out. An arm that builds a `MetalEncoder` and
  drops it without committing reports **0.05 ms** (three samples 0.01-0.10).
* **The kernel is bandwidth-bound.** Ruled out. Capping the K/V allocation
  (`KV_ELEMS`) from 50 331 648 halves (100 MB) to 4 194 304 (8 MB) to 262 144
  (0.5 MB), with `key_count` unchanged so the same 24 576 keys are read, moved
  `flash bq1` at 8192 keys/1024 queries from **303.57 -> 292.64 -> 284.04 ms**:
  a **64x** reduction in the bytes available to the loads bought 6%. If the
  kernel were moving 201 GB (§9.1) from DRAM, this is where it would show.
* **The time is not proportional to the work.** Ruled out, and this is the
  result that makes the whole thing incoherent. Adding cases to the bench at a
  fixed 1024 queries:

  | keys | 64 | 1024 | 4096 | 24576 |
  |---|---|---|---|---|
  | `flash bq1` ms | 2.28 | 32.24 | 144.36 | 947.35 |
  | TFLOP/s | 1.89 | 2.13 | 1.90 | 1.74 |

  Perfectly linear in `keys`, flat in TFLOP/s, and equally linear in `queries`
  (256 / 1024 / 4096 -> 237 / 947 / 3825 ms). **An empty kernel cannot have a
  cost that is linear in `keys * queries`.** So the body *is* doing the work; so
  is deleting it.

### 10.2 What this means for the spec

> Items 1 and 3 are **wrong as written** and are corrected in §22. Item 2's
> conclusion survives but for a different reason. Item 4 survives and is the
> single most useful thing in this section.

1. **§9's cost split (loads 44%, MMA 52%, softmax 3%) is not reproducible and
   must not be used to justify a change.** *True, and stronger than this says:
   it was never measured at all (§22). The measured split is 56% QK / 40% PV /
   4% other.* It is kept above only as the record of
   what produced T4f's design; every conclusion drawn from it in §9.1, §9.1b and
   §9.2 rests on a measurement that does not survive a re-run.
2. **§9.1b's "BQ=2 + key split + combine pass" stays refuted** — not because the
   byte arithmetic is wrong (it is right) but because it was justified by a split
   that does not reproduce. It is refuted *a fortiori*: the same reasoning
   predicted BQ=2 would help, and BQ=2 measured -2.0% (§8.2) and +0% E2E.
   *§22 adds the direct reason: the loads query blocking halves are worth 8%.*
3. **The microbench is a valid throughput instrument and an invalid attribution
   instrument.** It is stable (the `mma` control held 867-894 ms across ~12 runs,
   ±1.5%; `scalar` 1239-1258), linear, and it reproduces T4f's effect. It cannot
   attribute cost, and it cannot rank arms (§8.7, §9.4 — third confirmation).
   **E2E is the only instrument that has ever ranked anything correctly here.**
   *WITHDRAWN as to attribution — see §22. It attributes with a 6 162x signal.
   The ranking half is untouched and is the half that matters: four independent
   confirmations (§8.7, §9.4, §11, and the BQ=1/BQ=2 inversion in §22, where the
   bench says BQ=1 and the engine's own history says BQ=2).*
4. **Open hazard for whoever picks this up:** `pipeline_audit::build(...).ok()`
   turns a Metal compile error into `None`, and the dispatch then falls back to
   the MMA kernel *silently*. The bench prints the arm name it was asked for, not
   the kernel it got. So a `.metal` edit that fails to compile produces a
   plausible, stable, completely wrong number rather than an error. Any future
   ablation must assert `flash_attention_available()` and print
   `pipeline.function_name()` for the arm, not trust the label.
   *§22 adds the hazard that actually bit: not a compile error but a **stale
   build**. `attention.metal` is `include_str!`'d (`attention.rs:12`), so an edit
   with no recompile of that crate leaves the old shader in the binary and every
   measurement is of the previous kernel. The uniform-block ablation in §22 makes
   the failure impossible rather than checking for it.*

### 10.3 Process failure worth recording

Four ablation rounds were run before the obvious control — *does an empty kernel
take zero time?* — was tried. The first two rounds also used a `rc=$?` after a
pipe, so a failed build read as success. Both are cheap checks; skipping them cost
more than the ablations were worth. The rule this suggests: **an ablation whose
result is "the change did nothing" is only believable after a positive control
has been shown to move the same instrument.**

*F1 (2026-10-05): the control was then run, and it is the reason this section
matters. §22 arm 1 is an empty body and it reads **0.19 ms** against 1 143 ms. So
the control was cheap, it was available, and running it would have caught the
mistake immediately instead of costing two sections of design reasoning (§9.1,
§9.1b, §9.2) that had to be withdrawn afterwards.*

---

## 11 — T4g follow-up: 24k E2E after T4f, and the arm separation is now 21%

Interleaved A/B, one round each, `bash /tmp/ab_fullqk.sh 24576 1 flash mma`,
24590-token prompt, temperature 0, `--max-new-tokens 1 --metrics`:

| arm | prefill tok/s |
|---|---|
| **flash** | **67.10** |
| mma | 55.42 |

**T4f is worth +6.7% on the 24k wall** (62.87 -> 67.10), against +1.6% at 8k.
The gain grows with context because the V loads it de-scattered are exactly the
ones that scale with keys.

**And the microbench is wrong about the ranking for the fourth time.** §10.2
already recorded three independent confirmations that
`attention_arm_throughput_at_prefill_geometry` cannot rank arms; here it reports
`mma` 8% *faster* at 24 576 keys (958 vs 880 for the bq1 arm) and the engine
reports `flash` **21% faster**. The margin at 24k (11.7 tok/s) is four times the
one measured at 16k in §9.4, so the effect is real and grows with context.

### 11.1 Where AC2 stands after T4f

| | s | tok/s |
|---|---|---|
| measured wall at 24 590 tokens | 366.6 | **67.1** |
| target for AC2 | 273 | 90 |
| §5 floor: GEMM 213.7 + attention 86.0 + other 3.2 | 302.9 | **79.1** |

So **AC2 needs 73 s more than a perfect attention kernel can deliver with the
shipped GEMM**, and §5's arithmetic says closing that needs 7.6 TFLOPS of GEMM
(measured 6.13-6.44) *and* attention at its 86 s streaming floor simultaneously.
`spec/prefill-120tps` / `spec/engine-performance` §12/§14 already declared 7.5
TFLOPS a wall after rejecting 64x64 tiles, KCHUNK=32, staged `astage`, 32
accumulators, 8->4 simdgroups, the codebook LUT, `extract_bits`, concat fusion
and resident weights.

**This is the point at which "keep optimising attention" stops being the highest
-value use of the remaining budget.** Attention now sits at ~132-145 s against
an 86 s floor -- real headroom, but §10 shows we have no instrument that can
attribute it, and the two biggest untried levers (GEMM at 59% of the wall, and
decode at a 9.3x gap) are untouched. T2 is taken next; the GEMM question is
recorded as OQ6 below.

---

## 12 — The 86 s "bandwidth floor" is wrong by ~350x, and AC2 is NOT GEMM-gated

§5 -- and every conclusion built on it, including OQ4 and the §11.1 arithmetic --
rests on one number: *"bytes read per visit (one query re-reads its whole causal K
and V) = 512 x 2 (K,V) x 2 B = 2048"*, applied to **every (query,key) pair**. That
is the **L1/L2 request volume**, not DRAM traffic. `PREFILL_CHUNK = 1024`
(`crates/engine/src/model.rs:121`), so all 1024 queries of a chunk read the *same*
keys, and those keys are read from DRAM once per chunk, not once per query.

Counting actual DRAM traffic for a 24 590-token prefill (25 chunks; chunk `c`
covers queries `[1024c, 1024c+1024)` and the KV range `[0, 1024(c+1))`):

**Full-attention layers** (`kv_heads = 4`, `head_dim = 512`, no window):

```
per chunk   live_keys * 4 * 512 * 2 B * 2 (K and V)   =  live_keys * 8192 B
sum         sum_{k=1..24} 1024k = 1024 * 300 = 307 200 keys
per layer   307200 * 8192                            =  2.52 GB
10 layers                                             = 25.2 GB
```

**Sliding layers** (`kv_heads = 16`, `head_dim = 256`, `window = 1024`; a chunk
of 1024 queries spans 2047 live keys):

```
per chunk   2047 * 16 * 256 * 2 B * 2                = 33.5 MB
25 chunks * 50 layers                                 = 41.9 GB
```

| term | bytes | @273 GB/s |
|---|---|---|
| 10 full layers | 25.2 GB | 0.092 s |
| 50 sliding layers | 41.9 GB | 0.154 s |
| **whole 24k prefill** | **67.1 GB** | **0.25 s** |

**Measured attention at 24k is ~150 s.** So attention moves 67 GB in 150 s =
**0.45 GB/s, about 0.16% of this box's DRAM rate.** The kernel is not
bandwidth-bound by three orders of magnitude, and the honest floor is set by the
**L2** path (23.5 TB of fragment requests at an L2 rate of a few TB/s, i.e.
single-digit seconds), not by DRAM at all.

This is not a new claim -- it is what §10 measured and did not connect. §10
capped the K/V allocation **64x** smaller, so every one of those fragment loads
hit cache, and the kernel got **6% faster**. A kernel that were moving 201 GB
from DRAM (§9.1) cannot lose only 6% when 98.4% of its bytes start in L2.

### 12.1 AC2 re-derived

| | s | tok/s |
|---|---|---|
| measured wall at 24 590 tokens | 366.6 | 67.1 |
| GEMM + other (measured, context-independent) | 216.9 | -- |
| **measured attention** | **~149.7** | -- |
| **AC2 budget** | **273** | **90** |
| **attention allowed** | **56.1** | |

Attention must go **149.7 -> 56 s, a 2.7x kernel speedup**. In TFLOP/s terms the
full layers are 1.88e14 FLOP (§5, unchanged -- that count is right) and currently
run at ~1.74 TFLOP/s in isolation (§10), so the target is **~3.4 TFLOP/s = 22%
of the 15.6 TFLOPS tensor peak measured by `examples/mma_peak.rs`**. Twenty-two
percent is an ordinary number for a hand-written attention kernel. It is not the
2% the old arithmetic implied.

**Therefore: OQ4's conclusion ("needs 7.6 TFLOPS of GEMM") and OQ6 (added an hour
ago, "is the GEMM a wall?") are both wrong, and are withdrawn in the same commit
that records this.** The GEMM is 216.9 s and must not change; the entire 90 tok/s
ask is an attention-kernel ask, worth 56 s out of a 273 s budget.

### 12.2 What this re-opens

* **§9.1b's "BQ=2 + key split + combine pass" is re-opened, not left refuted.**
  Its refutation rested on the §9 cost split, which §10 withdrew. Its *own*
  argument is untouched by §5's error and is in fact the strongest one now
  available: the kernel is instruction-bound, so the only lever that matters is
  **fragment loads and MMAs issued per byte of L1/L2 traffic**, and query blocking
  multiplies MMAs-per-load in *both* halves (the `kf` fragment feeds BQ queries in
  QK, the `bf` fragment feeds BQ accumulators in PV) while dividing the 23.5 TB of
  fragment requests by BQ. Splitting the key range restores the grid that BQ=2
  gives away, which is why the naive BQ=2 measured -2.0% and -6.2%.
* **The `FLUSH_QUERY_BLOCK_MIN_KEYS = 16_384` threshold is now suspect** for the
  same reason: it was chosen from §5's byte model.
* **§10's real puzzle is smaller than it looked.** "Removing the MMAs does not
  help" is much less strange next to "the loads hit L2": the kernel's instruction
  mix, not its bytes, is the wall. What still needs explaining is that removing
  the QK *and* PV *instructions* moves nothing -- that remains unexplained and
  is recorded as OQ7.
  *F1: the premise is false. §22 measures the empty body at 0.19 ms, so the
  instructions were never removed -- the patched shader was not in the binary. The
  load finding above is confirmed and sharpened (all fragment loads together are
  12-19%), and OQ7 is closed.*

---

## 12.3 Verification state at `7568c7e` (end of session)

**No product or test code changed in this session's second half.** Every ablation
was applied to `crates/kernels/src/attention.metal` (and the temporary
instrumentation to `attention.rs` and `attention_qk_mma_bench.rs`), then reverted.
Confirmed by `git diff --stat 1254780 HEAD -- crates/` -> **empty**, and by a grep
for `KAPLUMBA_ABLATE_`, `DBG qk=`, `DBG kv=`, `KV_ELEMS`, `ksteps = 0u` across
`crates/` -> **no hits**.

The last few builds in this session were made from *patched* `.metal` sources, so
the verification below was re-run against a build of HEAD after `touch
crates/kernels/src/attention.metal`:

| check | result |
|---|---|
| `cargo build --release -p kaplumba-engine --tests` | rc 0 |
| `attention_qk_mma_matches_scalar` | **3 passed**, rc 0 |
| &nbsp;&nbsp;worst \|flash - scalar\| fp16 abs diff | **0.000488** (one ulp), tol 0.005, **0 NaNs**, 6 cases x {BQ=1, BQ=2} |
| &nbsp;&nbsp;worst \|mma - scalar\| | **0.000000** (bit-exact) |
| `attention_single_pass_bit_exact` | **8 passed**, rc 0 |

**Quality: unchanged.** The only numerically-different path is the flash prefill
kernel, which is one fp16 ulp from the scalar reference on every case -- the same
result T3 recorded, and the reason flash is safe to ship.

### Two process rules added by this session

1. **Never pipe a long measurement run into `tail`.** `tail` buffers until EOF, so
   a run that is interrupted (session end, timeout, a crash) produces *nothing*,
   not even the arms that finished. A ~50-minute 24k decode sweep was lost this
   way. Sweeps must append to a file per arm (`tee -a "$LOG"`), and the log path
   must be outside the tool's stdout capture.
2. **A measurement that has never been seen to fail is not a measurement.** See
   §10.3. The positive control (does this instrument move at all?) has to pass
   before any "the change did nothing" reading is believed.

### Not run this session (limitations, not passes)

* The **full workspace suite** was not re-run; a prior session recorded 4
  pre-existing failures at base. Only the two attention suites were re-verified,
  because only they cover the code that was touched.
* **24k E2E was measured once per arm**, not interleaved over several rounds, so
  the 67.10 / 55.42 pair carries this box's +-20% spread (OQ5). The 21% gap is
  four times the 8k spread, so the *direction* is safe; the exact margin is not.
* The decode split sweep at 24k (`KAPLUMBA_DECODE_SPLITS` 1/2/4/8/13/16) is
  running now and its result is **not** in this file yet. `ATTENTION_SPLITS_MAX`
  is 16, so the sweep cannot probe above it; if 16 is the best in range, the cap
  itself is the next thing to question, and raising it enlarges the decode
  partials buffer from `16 * q_heads * (head_dim + 2)` floats (4.2 MB at
  `q_heads = 32, head_dim = 512`).

---

## 13 — T2a status: decode split sweep BLOCKED on an unverified memory gate

### 13.1 What the sweep measured, and what must be thrown away

`/tmp/decode_splits_sweep.sh`, 24 590-token prompt, `--max-new-tokens 24`,
`KAPLUMBA_DECODE_SPLITS` pinned, one round per arm:

| arm | decode tok/s | verdict |
|---|---|---|
| splits=1 | **2.673** | **valid** |
| splits=4 | 0.160 | **INVALID** -- box in memory compression |
| splits=8 | 0.310 | **INVALID** -- box in memory compression |
| splits=16 | -- | never ran (session ended) |

`top` sampled during the splits=8 arm:

```
PhysMem: 23G used (11G wired, 5509M compressor), 126M unused.
```

126 MiB unused with a 5.5 GB compressor is the OQ5 stall signature, and the
distortion is enormous and **one-sided in the pessimistic direction** -- which is
the dangerous direction, because a sweep that only ever produces pessimistic
samples reads as "more splits is much worse" and would send the next person to
delete query splitting. **Do not read 0.160 or 0.310 as a property of the split
count.** They are properties of the compressor.

The only surviving datum is `splits=1 -> 2.673`, against a documented baseline of
**5.44 at the shipped split count**. That is a ~2x gap between one split and the
shipped setting, which is consistent with the decode kernel being parallelism-
limited (§6) -- but it is **one** valid point on one arm, and it does **not**
establish where the optimum is. In particular it does not show whether the curve
is still rising at `ATTENTION_SPLITS_MAX = 16`, which is the actual question T2a
was opened to answer.

### 13.2 The memory gate is written, unit-tested, and DOES NOT PASS

`box_healthy()` was added to the sweep so an arm cannot be sampled on a
compressed box. It **does not work**, and the sweep must not be trusted with it
until it does.

State of the work, precisely:

* `bash -n` passes on both the sweep and the test. That is the only thing that
  passes.
* `/tmp/test_parse.sh` exists and covers: the healthy cases, the OQ5 stall
  signature, four fail-closed shapes (no memory line, `17G` single-token,
  unreclaimable-only, empty), the historical `554M` trap from AGENTS.md 10.3, a
  live read, and a sabotage check. **It currently FAILS.**
* Observed failures: both `healthy=1` cases report `healthy=0`, and the live
  parse returns **`<EMPTY>`**, with `line 15: $1: unbound variable` repeated 11
  times on stderr.
* Diagnosis **not yet done.** The `unbound variable` points at `set -uo pipefail`
  interacting with `$1` inside the `healthy()` helper (the helper reads `"$1"`
  but the live-check block calls it without an argument), and the synthetic
  failures are most likely the same cause rather than a regex problem. That is a
  hypothesis, not a finding, and it must be confirmed before the parser is
  trusted.

The parser itself has already been wrong **twice** in this session, and both times
a test caught it and inspection did not:

1. `[^0-9]*` is greedy and backtracks to the **last** digit run, so
   `memory: 7.4 GiB available` parsed as `4` GiB.
2. A single-token `memory: 17G available` has no space before the unit, so a
   `\([0-9.]*\) \([KMG]i*B\)` pattern cannot match it at all.

The third version sidesteps the unit entirely by taking `scripts/gpuq`'s already
-corrected number (`scripts/gpuq:150` prints a fixed `memory: <n> GiB available
(reclaimable)` format), which is the right call given AGENTS.md 10.3 -- but
"right call" is not "verified", and the test that would verify it is red.

### 13.3 Why a 24k measurement needs a gate at all, and why the gate is hard

The 24k run itself consumes ~18.35 GiB `phys_footprint` (AC4's budget is 20 GiB),
so **the benchmark drives the box into compression by construction**. A gate that
samples only *before* an arm is insufficient, because the arm is what pushes the
box over; the current script samples after as well and logs `top` per arm, but
that has not been validated either.

The honest position for anyone continuing: **24k measurement on this box is
marginal (OQ5), the three valid data points we have for 24k E2E prefill are
67.10 / 55.42 / 62.87, and each carries the +-20% spread.** Resolve the gate, then
re-run T2a from scratch. Do not merge any arm from the current log into a
result.

---

## 14 — T2b: MMA decode attention, ported from oMLX 0.7.0 `sdpa_flash`

### 14.1 What oMLX actually did, and why it transfers

Cloned `github.com/jundot/omlx` to `/tmp/kaplumba-refs/omlx`, checked out tag
**v0.7.0** (`4d4f5a28`). The reference commit is **`07d88dbd`**, whose message is
the most useful single document found in this repo so far. Its diagnosis, for
their model:

> "MLX's vector SDPA spends a 32-lane simd_sum plus a full online-softmax update
> on every lane per (row, query head, key): at MiMo's GQA 16 it is ALU-bound at
> ~1/3 of DRAM bandwidth for one decode row (350-390 GB/s of KV on the M5 Ultra
> vs a 1.1 TB/s streaming roofline; more key blocks than MLX's cap of 1024 only
> reach 414 GB/s at 1M) and each extra verify row costs as much again, so a
> 3-row MTP verify reads KV at 112-134 GB/s and **decode collapses with
> context**."

That is the same conclusion §6 reached here from first principles ("decode is
latency-bound, the grid is only 64 workgroups, each walking a 128-iteration
dependent scalar dot per key"), reached independently on different hardware and a
different GQA ratio. Two independent diagnoses agreeing is the strongest
evidence in this spec that the shape of the problem is understood.

Their fix, `omlx/patches/mimo_v2/sdpa_flash.py`:

> "computes the same float32 attention with 8x8 float32 simdgroup matrix
> products: the query rows of a KV head form 8-row bands, each threadgroup owns a
> contiguous key split, stages 8-key K/V tiles in threadgroup memory for all
> bands, runs S = Q'K^T, the online softmax per tile and O += PV, and a second
> pass merges the float32 split partials."

Their measured end-to-end effect, which is the number to expect here:

| context | 8k | 64k | 256k |
|---|---|---|---|
| decode tok/s | +2.8% | +19.6% | **+71.6%** |

**The gain grows with context**, which is the whole point: at short context the
decode step is dominated by streaming weights, and attention only becomes the
term that matters as the KV window grows. This is also why the 8k number below
is a floor for the 24k case, not a prediction of it.

### 14.2 What was taken, and what was deliberately not

**Taken** (all three are in `55cd5df`):

* 8x8 simdgroup MMA for decode QK **and** PV, replacing the per-key `simd_sum`.
* The `(m, s, O)` partial split layout, so the existing combine pass and its
  exactness argument are reused unchanged. Nothing new had to be proven about
  the combine.
* **The split count stays a runtime value.** oMLX: "a template split count cost
  ~115 ms of JIT per step: -22% decode at 8k in the first A/B." Our
  `params.splits` was already runtime; this is recorded so nobody "tidies" it into
  a template parameter.
* **fp32 MMA and fp32 partials**, with no bf16 rounding of partials. Worth noting
  oMLX measured their flash kernel as *more* accurate than the kernel it replaces
  ("max |err| vs float64 0.0078 ... where MLX's is 0.0083-0.0135"), because MLX
  rounds per-block partials to bf16 and this does not.
* A **4096-key threshold**, matching `OMLX_MIMO_DECODE_FLASH_MIN_KEYS`.

**Not taken**, and why:

* **Everything NAX/M5.** `omlx/utils/nax_attention.py` (1043 new lines) and
  `omlx/utils/fast_attention.py` (345 new lines) are the two biggest new files in
  v0.7.0 and both are NAX-gated -- they require an M5 tensor unit. This is an M4
  Pro. `nax_attention.py` also exists to give MLA models a 192/128 head-dim
  kernel; Gxma is 512/512, so it has no target here.
* **All MTP / speculative work** (`omlx/speculative/dflash_drafter.py` and the
  Lightning MTP verify commits), per the brief.
* **TurboQuant KV** (`omlx/turboquant_kv.py`). AGENTS.md §8 forbids KV cache
  quantization.
* **Their fused decode layer stack** (q/k/v as one row-concatenated mat-vec,
  fused residual+RMSNorm, fused MoE experts, multi-row router gemv). These are
  real wins -- "A MiMo-V2.6-Flash decode step dispatched ~2.8k kernels, most of
  them small element-wise ops" -- and they are **model-shape-specific and
  MoE-specific**. Gxma is a dense model with a different layer stack. Recorded as
  a lead, not attempted.

### 14.3 Measured here (interleaved A/B, 3 rounds, 8204-token prompt)

`bash /tmp/ab_decode_flash.sh 8192 3`, `--max-new-tokens 24`, temperature 0,
`KAPLUMBA_DECODE_FLASH=0` for off and unset for on:

| arm | r1 | r2 | r3 | median | spread |
|---|---|---|---|---|---|
| scalar split kernel | 12.000 | 12.059 | 11.990 | **12.000** | +-0.3% |
| **MMA decode kernel** | 12.978 | 12.979 | 12.972 | **12.978** | **+-0.03%** |

**+8.2%**, and the effect arm is an order of magnitude tighter than the control,
so this is not a spread artefact. Two things follow:

* **AC3's "decode >= 12 @8k" was being met by 0.0%** before this (median 12.000).
  It is now met with 8% of margin. That criterion was on a knife edge.
* oMLX's own 8k figure for the same change is +2.8%, so +8.2% here is a *better*
  8k result than theirs on a different GPU. Consistent with our kernel having more
  to gain (GQA 8 and head_dim 512 mean the scalar path does 8x512 FMAs per key
  per head against their 16x192).

### 14.4 The 24k decode A/B could not be taken, and why that is a finding

The obvious next measurement -- the same A/B at 24 590 tokens -- is **blocked,
not merely unlucky**. Attempted, and the first arm read:

```
r1 flash=off prompt_tokens=24590 decode_tok_s=0.055
```

`top` during the run: `PhysMem: 23G used (9165M wired, 5954M compressor), 110M
unused`. That is **0.055 tok/s against a 5.44 baseline -- a 99x pessimisation**,
and it is the compressor, not the kernel.

The cause is structural, and this is the third time it has bitten (OQ5, and the
discarded splits=4/splits=8 arms in §13.1): **the 24k benchmark drives this box
into memory compression by construction.** Its own footprint is 18.35 GiB of a
24 GiB machine, so the run evicts the desktop's own pages, and every subsequent
allocation is a compression event. There is no gate that fixes this from
outside the run.

**So 24k decode is currently unmeasurable on this box**, and §13's OQ5 gate is
the wrong answer to that: the gate can refuse to *report* a compressed sample,
but it cannot produce a valid one. Closing AC1 (decode >= 14 @24k) needs either
a quieter machine (desktop closed, or a second box) or a lower-footprint
configuration. Until then the 8k result in §14.3 is the trustworthy decode
measurement, and it is a lower bound for 24k rather than an estimate of it.

### 14.5 A second context point, and a useful noise measurement

`bash /tmp/ctx_curve.sh 24576 4096` (4096 is exactly the shipped threshold, so
the flash arm is on):

| ctx | arm | prefill tok/s | decode tok/s |
|---|---|---|---|
| 4106 | scalar | 82.936 | 12.787 |
| 4106 | **MMA** | 86.042 | **13.346** |

* **Decode +4.4% at 4k, against +8.2% at 8k.** Two points, same direction, and
  the gain grows with context -- the shape oMLX measured (+2.8% @8k rising to
  +71.6% @256k). One round each, so it is a data point rather than a curve; the
  8k figure (3 interleaved rounds) remains the load-bearing one.
* **The prefill column is a noise measurement, not an effect.** `KAPLUMBA_DECODE_FLASH`
  cannot touch prefill -- it selects a kernel inside `encode_decode_split_*`,
  which decode dispatches through and prefill does not -- so 82.936 vs 86.042 is
  **3.7% of pure run-to-run spread at 4k**. Useful to have: it sets the floor on
  what this box can resolve without interleaving, and it is a reminder that the
  §11 24k prefill pair (67.10 / 55.42) was one round each and carries at least
  this much uncertainty.
* The 16 384-token arms of this run did not complete before the session ended,
  so there is no data point between 8k and the unmeasurable 24k.

---

## 15 — N5: the OQ5 gate is green, and its test now tests the real code

### 15.1 The red test was not a regex bug

`/tmp/test_parse.sh` failed with `line 15: $1: unbound variable` on every call.
§13.2 hypothesised "`set -uo pipefail` interacting with `$1` inside the helper".
That was right, and the mechanism is worth writing down because it is not
obvious: the call site was

```bash
avail=$(printf '%s\n' "$1" | parse)     # parse is the LAST element of a pipeline
```

and **bash does not pass a function's positional parameters to a command running
as the last element of a pipeline** (that requires `shopt -s lastpipe`, which is
off by default and is off in every script here). So `$1` inside `parse` was
unbound on *every* invocation, not only in the live check, the function printed
nothing, and `gpuq_mem_healthy` failed closed — which is exactly why all eleven
checks failed, including the ones that should have passed. The parser's regex
was fine. The comment on the function ("stdin: a gpuq status body") described the
intended interface; the body used `$1`.

**Fix:** read stdin and never place a parser in pipeline position.

### 15.2 The test was testing a copy

`/tmp/test_parse.sh` reimplemented `parse` and `healthy` inline. It therefore
proved nothing about `box_healthy` in `/tmp/decode_splits_sweep.sh` — the code
that actually gated a run. A green test over a copy is the AGENTS.md §3.4
failure mode in its purest form: **the test cannot fail because it is not
exercising the code that can fail.**

Both files now live in the repo and share one parser:

| file | role |
|---|---|
| `scripts/gpuq_mem.sh` | the parser and the classifiers, sourceable, no side effects |
| `scripts/test_gpuq_mem.sh` | 19 assertions, sources the real file, plus two end-to-end harness checks |
| `scripts/bench_decode_splits.sh` | the harness; sources the real file |

Deleting `/tmp/decode_splits_sweep.sh` and `/tmp/test_parse.sh` in favour of
these also removes the `/tmp`-ephemerality that has already destroyed one sweep.

### 15.3 What the harness gained beyond a fixed parser

* **Generic over the swept knob** (`VAR=... ROPE="..."`), so N1 and N4 use the
  same instrument rather than each growing their own.
* **Interleaved** (`ROUNDS`), because `ROUNDS=3` with `A B C / A B C / A B C`
  is what makes a null believable on a box whose spread is ±3.7% at 4k.
* **Polls memory DURING the arm.** §13.3 established that a pre-arm gate cannot
  work, because the arm is what tips a 24k run into compression. The poller
  writes `<epoch> <compressor> <unused>` every `POLL_SECS` and
  `mem_sample_verdict` classifies afterwards. **An invalid sample logs
  `INVALID-NO-NUMBER` and no `decode_tok_s` at all** — deliberately. OQ5's
  distortion is 99× and always pessimistic, so a recorded-but-flagged number
  still reads as a data point in a table, and "more splits is much worse" is
  precisely the conclusion three invalid arms already nearly produced.
* **`/usr/bin/time -l` for `phys_footprint`**, never `ru_maxrss`.
* **exit 75 is its own outcome** (`GPU-BUSY`), never a zero tok/s.

### 15.4 Two parser bugs found while writing this, by writing the pure function

`top_physmem_fields` has to read `compressor` and `unused`, which `gpuq status`
does not expose. The first version split the `PhysMem` line on commas:

```
PhysMem: 23G used (9165M wired, 5954M compressor), 110M unused.
```

`split(",")` on the parenthesised part yields **2** fields (`9165M wired`,
`5954M compressor`), so a positional reader looks for a 3rd that is not there and
finds nothing — and `unused`, which sits *outside* the parens, is not in the
inner text at all. The committed version matches on **field name**:

```awk
match($0, "[0-9]+(\\.[0-9]+)?[KMGTkmgt]?[ \t]+unused")
```

which is order-independent, paren-independent, and unit-aware. Asserted on the
exact string AGENTS.md §10.3 records: `554M compressor -> 0.54`, `126M unused ->
0.12`. Note this is *not* the bug AGENTS.md warns about — that one is stripping
non-digits and taking the last comma field; named matching is the fix for it,
not a repeat of it.

### 15.5 Sabotage verification, with the failure modes observed

| sabotage | observed result |
|---|---|
| pre-arm gate replaced by `if false` in `run_arm` | suite exits 1 with **both** `FAIL: harness ran the model on a box it could not parse` and `FAIL: no SKIPPED line in the log`; restore returns exit 0 and the harness is byte-identical (`diff` vs backup clean) |
| `MIN_UNUSED_GIB=1000000` | `gate rejects an unreachable threshold` |
| parser re-introduced with a greedy-digit class | `greedy-digit bug stays fixed` assertion exists to catch exactly this |

**One sabotage of the test itself was found and fixed.** The first sentinel
checked `grep -q MARKER "$TD/out"`, but the harness redirects the arm's output
into its own temp file, so the marker was never in `$TD/out` and the assertion
**passed while the harness was sabotaged** — a green check that could not fail.
It now drops a sentinel *file*, which no stream redirection can hide. Recorded
because it is the second time in this repo that a "green" test turned out to be
unfalsifiable for a reason unrelated to the code it names.

### 15.6 What this does NOT fix

OQ8 stands. A gate can refuse to *report* a compressed sample; it cannot produce
a valid one, and the 24k run compresses the box by construction. `N5` makes the
8k/16k measurements trustworthy. It does not make 24k decode measurable.

**Commands**

```bash
bash scripts/test_gpuq_mem.sh                       # the gate's own suite
LOG=/tmp/x.log scripts/bench_decode_splits.sh 8192 1 2 4 8 16
VAR=KAPLUMBA_DECODE_FLASH ROPE="0 --" ROUNDS=3 \
  scripts/bench_decode_splits.sh 8192                # "--" leaves the var unset
```

---

## 16 — N1: prefill key-range passes are implemented, bit-exact, and MEASURED NEUTRAL

### 16.1 What shipped

`PASSES` template parameter on `attention_prefill_flash_body`
(`crates/kernels/src/attention.metal:1432`), two new kernels
(`online_attention_flash_pass_512_8`, `online_attention_flash_pass_bq1_512_8`),
`pass_begin`/`pass_keys` added to `AttentionParams`, a host loop in
`encode_online_attention_passes`, a fixed-size `attn_pass_state` scratch buffer,
and `attention_prefill_pass_bit_exact.rs` (**17 tests, all pass**).

The mechanism is oMLX's, verbatim in shape. From
`omlx/utils/nax_attention.py:162-170`: *"One dispatch covers key blocks
[kb_begin, kb_end). Unless FIRST, the online-softmax state of every query row
(fp32 O accumulator, running max, running sum) is loaded from Sin; unless LAST it
is stored to Sout instead of normalising and writing the output. Every row thus
runs exactly the operations of one full dispatch, in the same order
(bit-identical output)."* Their `FIRST`/`LAST` are template parameters; ours are
the runtime tests `range_begin == 0` and `range_end >= key_count`, so **one
pipeline serves every pass of the chain** rather than one instantiation per
(first, last) combination — 4 kernels instead of 8, and no JIT dimension growth.

The state layout is the decode partials layout with a query axis in front:
`(q, head)` rows of `O[512], m, s` fp32, 67 371 008 B at the 1024-query chunk.
Fixed-size and context-independent (AGENTS.md §8): it is sized from
`max_tokens <= PREFILL_CHUNK`, and `prefill_pass_state_elements` has no key count
in its signature — the structural reason it cannot grow with context. It is
**scratch, not weights**, so the 11.51 GiB weight budget does not move, and the
measured 16k `phys_footprint` is 15.48-15.56 GiB across all six arms, i.e. the
64 MiB does not show up against the pre-existing 15 GiB.

### 16.2 Two bugs, both caught by the test and neither by inspection

**Buffer order.** `dispatch_groups` appends the params blob at
`bindings.len()` (`crates/kernels/src/encoder.rs:453`). So `state` has to be
inserted *before* `params`, which puts `output` at 5, `state` at 6, `params` at 7
— the reverse of every other kernel here. Getting it wrong bound the state buffer
to the params bytes and left the real output unbound: **no error, all 16 777 216
fp16 values wrong by 0.7133789.** Two wrong orderings were tried before the right
one; both looked like plausible implementation, and both produced a stable,
confidently-wrong number rather than a failure. Recorded because §10.2's "open
hazard" note is the same class of problem and this is the second instance.

**Unaligned pass edges — the one that would have shipped.** The online softmax
updates `(m, s, O)` **once per 64-key tile**, so the sequence of running maxima,
and therefore the sequence of rescale factors applied to `s` and `O`, is a
function of where the tile boundaries fall. My first version cut at
`(count * index) / passes`, which lands mid-tile. Measured cost:
**one fp16 ulp on 1.0% of outputs** — `637 195 of 16 777 216` values differ,
worst |diff| **0.000488**, first at `(query 40, head 5, dim 116)` for 44-key
passes over 129 keys.

That is the number to sit with: **0.000488 is 10x inside the 5e-3 tolerance every
other attention test in this repo uses.** `attention_qk_mma_matches_scalar` allows
5e-3; `attention_decode_flash_matches_split` allows 5e-3. A one-ulp regression
would have been invisible to all of them, would have been reported as "bit-exact",
and would have shipped. What caught it was insisting on `to_bits()` equality in
the new test rather than reusing the house tolerance.

oMLX never had this hazard: it cuts in units of its key block (`_BK = 32`,
`:76`), `_pass_edges` returns block **indices** (`:800`) and the kernel loops
`for (kb = kb_lo; kb < kb_hi; kb++)` (`:362`), so an edge is a block boundary by
construction. Here it has to be explicit: `PREFILL_PASS_TILE = 64`,
`prefill_pass_edges` snaps every non-final edge to a multiple of it, and the test
pins the alignment invariant directly.

**Sabotage verification, observed failure modes.** Adding 8 to every edge
(`edges.push(index * tiles_per_pass * 64 + 8)`) fails **7 of 17** tests, with
both messages firing as intended: `edge 1 of a 1000-key chain with pass_keys=1 is
not tile-aligned` and `a pass chain of 44 keys over 129 keys is NOT bit-exact:
15 627 870 of 16 777 216 fp16 values differ, worst |diff| = 0.12805176`. Reverted;
`diff` against the pre-sabotage `.metal` is clean and the suite returns to 17/17.

A second sabotage — dropping the `if edge < key_count` guard, to see whether the
empty-final-pass case is covered — **passed 17/17**. That guard is unreachable
for any input the production code produces (see 16.4), so the test cannot fail on
it. Recorded rather than papered over with a test that asserts an unreachable
branch.

### 16.3 The measurement: NEUTRAL. Do not ship it as a speedup.

Interleaved A/B, 3 rounds, 16 394-token prompt, `--max-new-tokens 1`,
`KAPLUMBA_PREFILL_PASS_KEYS` per arm, poll file kept per arm:

| arm | r1 | r2 | r3 | median | spread |
|---|---|---|---|---|---|
| one dispatch (`pass_keys=0`) | 74.947 | 74.750 | 74.108 | **74.750** | 0.84% |
| `pass_keys=2048` (8 passes) | 74.827 | 74.787 | 74.704 | **74.787** | 0.12% |
| `pass_keys=8192` (2 passes) | 74.950 | 74.298 | *INVALID* | **74.950** | — |

**+0.05% at 8 passes and +0.27% at 2 passes, against a 0.84% spread on the
control.** Both are inside the noise. The one discarded arm
(`r3 pass8192 INVALID-NO-NUMBER`, `unused-dropped-to-0.04GiB`) is recorded as
invalid with no number, per the harness rule.

**Positive control, same binary, same prompt, same session:**
`KAPLUMBA_SCALAR_ATTENTION=1` reads **62.019 tok/s / 272.72 s wall** against the
flash default's **74.576 / 230.47** — the instrument moved 20% on a known effect,
so a 0.05% reading of "no difference" is believable rather than a broken build.
This is the check §10.3 asks for and the one the whole T2a sweep was missing.

Peak `phys_footprint` (from `/usr/bin/time -l`, **not** `ru_maxss`): 15.48, 15.49,
15.50, 15.56, 15.52, 15.51, 15.52, 15.50, 15.56 GiB across the nine arms —
indistinguishable, so the 64 MiB state buffer is free at this scale. The
`ru_maxrss` column reads 8.56-9.09 GiB, i.e. **5.5 GiB below the truth**, exactly
as AGENTS.md 10.3 warns; it is logged only as a contrast.

### 16.4 Why neutral is the expected answer, not a surprise

oMLX measured the mechanism on **a 1M-key context at ~85 TFLOPS against ~103 at
64k** (`nax_attention.py:89-91`) — a 21% effect at 16x our context. At 16k our
entire KV working set for a full layer is 134 MB, well inside what the caches
absorb, so the drift the slices are supposed to prevent has barely begun. §10
already measured the same thing from the other direction: **capping the K/V
allocation 64x smaller bought 6%.** A mechanism that fixes cache-residency loss
cannot pay where there is no residency loss to fix.

So the honest reading is: **N1's premise does not hold at any context this box can
measure.** It would be expected to matter at 1M keys on hardware where it matters
to oMLX, and this engine's 24k ceiling makes that unreachable. This is a
**negative result with an exact, fully tested implementation behind it** — the
implementation is worth keeping (it is the only correct way to express the
mechanism, and the tile-alignment invariant it forced is a real property of the
kernel), but **the default must stay one dispatch**, because a mechanism with no
measured benefit should not be a *default*: a default is a claim.

> **F3 CORRECTION (2026-10-05).** This paragraph and the one after it contradicted
> each other and one of them was false: the code shipped `PREFILL_PASS_KEYS = 8192`,
> so *every* prefill dispatch with `key_count > 8192` sliced by default, while
> "the default must stay one dispatch" and `spec.md` OQ9 both said off. **The
> default is now `PREFILL_PASS_KEYS = 0`** and `KAPLUMBA_PREFILL_PASS_KEYS=8192`
> is the opt-in. Re-measured on the shipped body **after** F2's grid gate, 3
> interleaved rounds at 16 394 tokens: **-0.13%** (75.172 vs 75.271 tok/s), inside a
> 0.30-0.63% arm spread — so the only surviving argument for "off" is the
> default-is-a-claim one above.
>
> **The other argument in the original text, "plus 64 MiB of scratch", is VOID.**
> `model.rs:3447` and `model.rs:4927` size `prefill_pass_state_elements_widest` from
> `tokens` **unconditionally**, so the 64 MiB is resident whether or not the passes
> engage — `phys_footprint` reads 15.49-15.56 GiB in both arms and *cannot* differ.
> Do not use that argument again. Full record, including the positive control and
> the exactness evidence on generated text, in **§25**.

If a future session measures a context where the passes pay, that is the flag to
reach for.

### 16.5 Invariants and facts worth keeping

* **Pass edges must be tile-aligned or the output changes by one ulp.** The
  strongest single fact from this work. `PREFILL_PASS_TILE` and the snapping in
  `prefill_pass_edges` are load-bearing for correctness, not performance.
* `dispatch_groups` puts params at `bindings.len()` — an extra buffer must be
  inserted before it, not appended.
* `top -l 1 -n 0` is **throttled to nothing under load**: 25 of 25 polls inside a
  4-second loop during an 8k prefill returned no PhysMem line. `vm_stat` is a
  sysctl read and answered every time. `top_physmem_live` retries then falls back
  per field. A `top`-only poller is blind exactly when the box is busiest.
* **Loading the model drives the compressor to 6-9.5 GiB on this box** and takes
  `unused` from ~17 GiB to ~0.1 GiB in about 8 seconds, on its own. A gate that
  watches the whole run reads every good run as compressed.
* **A good run and a stalled run print the same memory numbers here.** The OQ5
  stall reads 5.90 GiB compressor / 0.11 GiB unused; a 626-token run returning
  **100.9 tok/s** read **9.47 / 0.09**. This is §15.6 restated with numbers and it
  is the strongest form of the OQ8 argument yet: the measurement, not the kernel,
  is the binding constraint on this box.
* `sys` CPU on this box is 3.5-32% of wall on *good* runs (dominated by the
  11.5 GiB mmap'd weight read), so OQ5's ">2% CPU in compression" rule, read as
  `sys/real`, rejects good runs. Reported, not gated on.
* **The `end` pass carries the load**: 16 394 tokens at `pass_keys=2048` is 8 passes
  and the last one holds the most keys, so tail latency is unchanged — consistent
  with the neutral result.
* The grid gate is `<= 512` threadgroups (oMLX's `n_groups < min_groups` is `<`,
  but their comment calls 512 the "wave or two" size, so the boundary belongs on
  the unsliced side). 128 queries x 4 kv heads at BQ=1 = 512 groups → one dispatch;
  129 queries → sliced. **F2: this was the documented behaviour and not the
  implemented one** — the gate was written on `prefill_pass_count` and applied
  nowhere, so production sliced grids as small as 20 threadgroups (the last chunk of
  a 16 394-token prompt: 10 queries x 4 kv heads at BQ=2). It is applied now, in
  `encode_online_attention_passes`, by calling the tested function;
  `crates/engine/tests/prefill_pass_gate_production_path.rs` pins it on the
  production entry point.
* `ATTENTION_SINGLE_PASS_MAX_KEYS` and `FLASH_QUERY_BLOCK_MIN_KEYS` are unrelated to
  this work and unchanged.

### 16.6 Commands

```bash
# correctness (17 tests, no model load)
scripts/gpuq try -- cargo test --release -p kaplumba-engine \
  --test attention_prefill_pass_bit_exact

# the measurement
ROUNDS=3 OUTDIR=/tmp/n1_16k scripts/bench_prefill_passes.sh 16384 2048 8192
ROUNDS=1 OUTDIR=/tmp/x scripts/bench_prefill_passes.sh 24576 8192   # 24k, marginal
KAPLUMBA_PREFILL_PASS_KEYS=0 ./target/release/kaplumba ...           # one dispatch

# the gate's own suite
bash scripts/test_gpuq_mem.sh
```

---

## 17 — N3(a): the O rescale guard, and what the test had to be against

### 17.1 What shipped, and from where

`attention.metal`, the PV rescale loop of `attention_prefill_flash_body`:
`if (msv == 1.0f) { continue; }` guarding `NPER * 2` multiplies per
(query, head, tile). Ported from oMLX v0.7.0
`omlx/patches/mimo_v2/sdpa_flash.py:299-300`, verbatim including the reasoning:
*"A row whose max did not move has factor 1: skipping its rescale is exact"*
(`const bool moved = simd_any(m_new != m_run); ... if (moved) { ... e *= factor; }`).

It is exact because `ms = exp(m_old - m_new)` is **exactly** 1 when the two are
equal, and `x * 1.0f == x` in IEEE-754 -- asserted bit-for-bit in
`attention_rescale_guard::the_guard_is_load_bearing` for every value the
accumulator can hold, including the kernel's NEG sentinel `-1.701411733864e+38f`
and `±FLT_MAX`. Zero of the difference is rounding; there is none to bound.

### 17.2 The test had to be against the SCALAR arm, and finding that out took a sabotage

The first version compared the **pass chain** against a **one-dispatch flash run**.
It passed 8/8 with the guard sabotaged to `if (true) { continue; }` -- because both
arms run the same body, so a uniform corruption is invisible between them. The
guard could be arbitrarily wrong and that comparison could not tell.

**That is the third time in this spec a self-comparison has hidden a bug** (the two
that shipped are called out in AGENTS.md §3.4: the `k >= 32` kernel that read
activations instead of weights, and the ablation that set a flag disabling the path
under test). Recording it because the pattern is the common case, not the exception:
when you add an optimisation, comparing "optimised" against "also optimised" is
almost never the control you want.

The fixed test compares both flash arms against `AttentionQk::Scalar` — an
independent implementation of the same maths (per-lane dot products, no MMA, no
rescale guard). With the guard inverted (`if (msv != 1.0f)`) it fails **7 of 8**
with `the guarded flash body is 0.38110352 from the scalar reference
(queries=1024 keys=1024 window=0 block=2, 13 080 382 of 16 777 216 values differ)` —
**76x the 5e-3 tolerance**, i.e. a wrong answer rather than a rounding difference,
which is exactly what skipping a rescale that mattered produces.

### 17.3 Two measured bounds, recorded rather than tuned away

* **The tolerance is 5e-3, the same number every other attention test here uses,
  and deliberately not tightened.** The flash body reassociates the QK dot product
  (scalar accumulates in `d` order, the tensor unit in its own), which alone
  measures 4.88e-4 at unit scale. A "one ulp" tolerance would fail on the
  reassociation, not the guard, and would train a reader to ignore it.
* **Value scale 32x is excluded, and the bound was measured, not guessed.** At that
  magnitude flash and scalar differ by **0.015625 with the guard and 0.015625 with
  the guard removed entirely** — identical to 6 digits. The guard contributes
  nothing there; the difference is reassociation growing with input magnitude. So
  the scales tested are 1, 2 and 4.

### 17.4 Coverage and a vacuity bug the assertion caught

8 tests: settled max (random V), max-never-moves-after-the-first-tile (flat V),
moving max (wide V), ragged tiles (8 key counts), both query blocks and two
windows, query tails (BQ=2 with odd counts), and a 49-pass chain over 24 590 keys.

**The first version of the "max never moves" case used all-zero V and passed
vacuously.** With V = 0, `Q·K = 0` for every key, so every score is forced to NEG
by the causal test, `s` stays 0, and the epilogue's `sm > 0` guard writes zeros —
so any comparison agrees. The test helper's assertion that the control output is
non-zero caught it; flat **non-zero** V gives the intended case (identical scores,
max moves once, guard fires on every later tile).

---

## 18 — N4: `decode_splits` re-swept with the MMA kernel. The optimum is 13, and the cap is not the constraint

### 18.1 The sweep

`scripts/bench_decode_splits.sh`, interleaved A/B over rounds, 8204-token prompt,
`--max-new-tokens 32`, MMA decode kernel (on by default above 4096 keys), poll file
kept per arm:

| requested splits | decode tok/s (mean) | min | max | n |
|---|---|---|---|---|
| 1 | 7.882 | 7.879 | 7.886 | 3 |
| 2 | 10.125 | 10.124 | 10.127 | 3 |
| 4 | 11.732 | 11.727 | 11.738 | 2 (one INVALID) |
| 8 | 12.626 | 12.621 | 12.633 | 3 |
| **13** | **12.917** | 12.911 | 12.926 | 3 |
| 16 (cap) | 12.776 | 12.772 | 12.783 | 3 |
| shipped default (no override) | 12.798 | 12.785 | 12.813 | 3 |
| 24 (→ clamps to 16) | 12.777 | 12.767 | 12.794 | 3 |
| 32 (→ clamps to 16) | 12.780 | 12.774 | 12.789 | 3 |

**Spread is ±0.05% on the effect arms and ±0.11% on the control** — an order of
magnitude tighter than the 3.7% this box was measured at in §14.5, which is the
realisation that most of the documented noise was *load-state* noise, and load state
is what the poll gate now controls for. The curve is therefore trustworthy at the
1% level, which nothing else in this spec has been.

`phys_footprint` 14.10-14.16 GiB across all 27 arms — the partials buffer is not
observable in it.

### 18.2 `ATTENTION_SPLITS_MAX = 16` cannot be probed above, and does not need to be

`parse_pinned_splits` filters the knob to `1..=ATTENTION_SPLITS_MAX`
(`attention.rs:531`), so arms above the cap **clamp**. Confirmed empirically rather
than by reading the code: `splits=24` reads 12.777 and `splits=32` reads 12.780,
against `splits=16`'s 12.776 — identical to within 0.03%, i.e. the same
measurement three times. The harness labels them `above-cap(16)` in the log so this
can never again be mistaken for three distinct data points (the T2a sweep's failure).

**And it does not matter, because the curve has already peaked.** The maximum is at
13, *below* the cap. Raising the cap can only help the one shape whose budget is a
lower bound — the 64-thread sliding shape, where `decode_one_wave_workgroups(64) = 288`
is documented as "at least 288", i.e. 18 splits at 16 kv heads, above the cap
(`attention.rs:594-608`). Those layers do not use the MMA kernel at all
(`encode_decode_split_explicit` refuses any shape but head_dim 512 / groups 8), so
the question does not arise on this path. **N4's second question is answered: the cap
is not the binding constraint, and probing above it would measure nothing.**

### 18.3 The finding: 13 is the optimum, and an existing un-defaulted rule predicts it

`decode_one_wave_workgroups(256) = 54`, and `54 / kv_heads(4) = 13` — exactly the
measured maximum. That rule was implemented and swept in `spec/decode-19tok` and
reproduced the optimum for three shapes (`kv_heads` 4/5/6 → 13/10/9), with the
workgroup-count transition bracketed from both sides (54 is one wave, 55 is two,
`attention.rs:583-594`). **This is a third reproduction of 13, on different
hardware state and with a different decode kernel — and it is NOT an independent
one (F6, `adversary_report.md`).** `spec/decode-19tok` §2 read the budget of 54
off **one** sweep of three shapes, and `kv_heads = 4 -> 13` **is the row the
budget was read off**. Re-measuring the `kv_heads = 4` maximum here is therefore
a re-confirmation of the derivation case, not an out-of-sample prediction of a
rule. What makes the rule a rule rather than a fit to the 4-kv-head number is the
*other two* shapes (`kv_heads` 5 and 6 → 10 and 9, built by holding
`q_heads / kv_heads = 8` and varying `kv_heads`), and **neither of those shapes
exists in Gxma and neither has been re-measured by anyone on this machine.**
Recorded so the next reader does not credit this result with evidence it does not
carry.

**The optimum itself is solid, and for a reason that does not depend on the
independence question.** The adversary's own interleaved 8-arm curve at 8 204
tokens (8/10/11/12/13/14/16 + shipped) is monotone rising to 13 and **cliffs
−1.5% at 14**, recovering only to −0.9% at 16, with round-to-round repeatability
of 0.03–0.15%. 13 is 52 workgroups and 14 is 56, and 54/4 = 13 vs 14/4 = 14 → 56 >
54, the documented two-wave boundary. So the maximum is a real wave boundary, not
a point fit — which is the claim that matters for the decision, and it does not
need the rule's out-of-sample shapes to be re-measured.

The reason the shipped default is 16 and not 13 is a one-line default, not a
calculation: `decode_split_rule()` returns `DecodeSplitRule::ThreadTarget`
(`attention.rs:514`), whose arithmetic is `16384 / (32 * q_heads) = 16384 / 1024 =
16` — and `clamp(1, ATTENTION_SPLITS_MAX)` keeps it. So:

* **swapping the default to `WorkgroupsPerCore` is worth +0.93% decode @8k**
  (12.917 vs 12.798, ~9x the effect spread) on the 10 full layers;
* **and is provably a no-op for the other 50** — the 64-thread shape evaluates
  `288 / 16 = 18`, which the cap clamps to the 16 that ships today. The code says
  so at `attention.rs:600-606`: *"that is why 83% of the layers cannot regress from
  this rule: for them it provably evaluates to the number that ships today."*

**Not flipped in this commit, deliberately.** Two reasons, both worth stating: the
+0.93% is small enough that it deserves its own A/B at 16k and 24k rather than
riding along on a sweep whose purpose was to answer a different question, and
flipping a shipped default on the strength of a rule that has never been the default
is exactly the move that `spec/decode-19tok` presumably declined to make for
whatever reason it declined. The evidence for the change is now three
measurements of the `kv_heads = 4` maximum — one of them (`decode-19tok` §2) the
derivation itself, so this is a re-confirmation rather than an independent
reproduction; see §18.3 for what does and does not follow from that. What the
evidence does support is written down and is a wave boundary, not a fit. The
decision is one line and belongs to whoever
owns the shipping default. `KAPLUMBA_DECODE_SPLIT_RULE=workgroups` already ships it
as an opt-in, so the claim is testable today without a rebuild.

### 18.4 The other thing this sweep settles

`splits=1` reads **7.882 tok/s against 12.917 at the optimum — 39% slower.** §6
diagnosed decode as parallelism-limited from the scalar kernel's per-key
`simd_sum`, and T2b replaced the arithmetic; the sweep shows the *parallelism*
half of that diagnosis survives the arithmetic fix: at one split there is 1/13 of
the grid and 39% of the throughput. So the MMA kernel is still not saturating the
machine at the shipped split count, which is the honest framing of what is left in
decode and where §14's "+71.6% at 256k" would eventually come from.

### 18.5 Two harness bugs this sweep found (both would have produced wrong medians)

* `grep VALID` also matches `INVALID-NO-NUMBER`, so discarded samples were folded
  into the median table. Caught because an arm marked INVALID appeared in the
  medians *with a number*. Fixed to `grep -E ' VALID( |$)'` in both harnesses. The
  §13.1 sweep's discarded arms were exactly this failure mode.
* A first smoke run reported `splits1 mean=13.495` from a sample its own gate had
  rejected — the same bug, one run earlier, and the reason it is called out.

---

## 19 — N3(b), the head-dim split: NOT APPLICABLE HERE, and here is the proof

oMLX's `HS == 2` scheme (`sdpa_flash.py:58-59`, `:398-405`) splits a band's head
dims over two simdgroups:

```python
def _head_split(rows: int) -> int:
    """Simdgroups per 8-qrow band: one-row forwards split each band's head
    and value dims over two simdgroups (more threads per key tile), multi-row
    forwards keep one; the arithmetic is the same either way."""
    return 2 if rows == 1 else 1
```

with `constexpr int DKQ = DK / HS;  // head-dim blocks held by this simdgroup` and
`DVH = DVB / HS`. The precondition is `QB = (G * ROWS) / 8`: with MiMo's GQA 16 and
`ROWS = 1`, `QB = 2` bands per threadgroup, so `HS = 2` means **4 simdgroups** —
two bands x two dim-halves — and the 32-thread threadgroup grows to 128.

**Our decode shape cannot express that.** `NSG = 8` and `groups = q_heads /
kv_heads = 8` are equal, and `BQL = 1`, so there is exactly **one** 8-head band:
`QB = (8 * 1) / 8 = 1`. `HS = 2` would then use `QB * HS = 2` simdgroups out of 8
and idle six — a 4x reduction in the QK thread count on the shape that is already
the narrowest. There is no second band to put in the freed simdgroups, because
decode has one query row.

**And the premise is already satisfied here by a different mechanism.** `HEAD_DIM`
is 512 and `NTILES = HEAD_DIM / 8 = 64`, so `NPER = NTILES / NSG = 8`: each
simdgroup already owns a contiguous 64-dim window of the value head (8 n-tiles of
8 dims). The value-dim split oMLX's `HS` introduces is, in this kernel, the
**pre-existing** `nt = sg * NPER + j` partition — and it is exactly the assignment
T4f changed because it was the right one (+6.7% E2E at 24k). PV is already split
8 ways over the value dims. There is nothing left for `HS` to split.

The QK half of the idea (split the 512-dim *query/key* reduction across simdgroups
and add the partial scores through threadgroup memory) is *not* ruled out by
arithmetic — it is exactly what `sdpa_flash.py:198-241` does for `HS == 2`, and it
would halve the per-simdgroup dependent chain at the cost of a threadgroup-memory
exchange per key tile. But it is a substantial kernel rewrite with a real chance of
losing to the extra barrier and the partial-sum traffic, it has no reference
measurement on a 512-dim head, and §10 has already established that **the
microbench cannot rank attention arms** (four independent confirmations). So
adopting it would mean building it blind, then measuring E2E, on the strength of a
technique whose own authors measured it at 3-4% on a different head dim.

**Recorded as not-taken with reasons, per the brief, rather than silently skipped.**

### 19.1 The oMLX fused decode layer stack, revisited

`implementation_notes.md` §14.2 recorded their fused q/k/v mat-vec + residual +
RMSNorm layer stack as a lead: *"A MiMo-V2.6-Flash decode step dispatched ~2.8k
kernels, most of them small element-wise ops."* Their own note is that it is
model-shape-specific and MoE-specific, and Gxma is dense. **Not attempted**, and the
N4 sweep above is consistent with that being right: at 8k the decode step is
**12.917 tok/s = 77 ms**, and the attention kernel is the dominant term (§6), so a
few dozen saved element-wise dispatches cannot be where the time is.

### 18.6 Confirmation at 16k: the win GROWS with context

`scripts/bench_decode_splits.sh 16384 13 16`, 2 interleaved rounds, 16 384-token
prompt, `--max-new-tokens 32`:

| arm | r1 | r2 | median | spread |
|---|---|---|---|---|
| shipped default | 12.082 | 12.107 | **12.095** | 0.21% |
| `splits=13` | 12.327 | 12.339 | **12.333** | 0.10% |
| `splits=16` (explicit) | 12.090 | 12.108 | **12.099** | 0.15% |

**+1.97% at 16k against +0.93% at 8k**, with the effect spread 2-4x tighter than
the control's. The explicit `splits=16` arm reproduces the shipped default to
0.03%, which is the positive control for this sweep: pinning 16 and not pinning
anything are the same measurement, so the 13-vs-16 gap is real and not an artefact
of how the knob is applied.

The direction is what makes this worth acting on rather than filing: **the gain
grows with context**, the same shape oMLX measured for their decode flash kernel
(+2.8% @8k -> +19.6% @64k -> +71.6% @256k) and the same shape T2b showed here
(+4.4% @4k -> +8.2% @8k). At 24k, if the trend held, splits=13 would be worth
~3%, and the 24k context is where AC1 lives. **This is the first measured
prefill-or-decode lever of this spec whose effect is still rising at the largest
context the box can run.**

Why 13 beats 16 at all, in one sentence: the decode grid is `splits * kv_heads`,
so 16 splits is a 64-workgroup grid and 13 is a 52-workgroup grid, and 52 is the
last size that fits in **one** wave on this box — 54 is one wave, 55 is two
(`decode_one_wave_workgroups`, `attention.rs:583-594`). Sixteen splits overshoots
the machine by one workgroup and pays for a second wave of two. The shipped
`ThreadTarget` rule computes `16384 / (32 * q_heads) = 16` from a *thread* budget
that has nothing to do with workgroup residency, which is precisely the error
`decode-19tok` diagnosed and why `WorkgroupsPerCore` was written and then not
defaulted.

---

## 20 — N6: final scorecard

`scripts/scorecard.sh`, 3 interleaved rounds, two arms:

* **shipped** — no env overrides, what a user gets;
* **workgroups** — `KAPLUMBA_DECODE_SPLIT_RULE=workgroups`, i.e. decode splits
  16 → 13. The opt-in exists so the scorecard can carry N4's measured improvement
  without this commit flipping a shipped default.

Both arms run every context, so a short-context regression is visible in the same
table rather than assumed absent.

| context | tokens | arm | prefill tok/s | spread | decode tok/s | spread | `phys_footprint` | `ru_maxrss` |
|---|---|---|---|---|---|---|---|---|
| 614 | 626 | shipped | **100.90** | 0.22% | **14.862** | 0.47% | 12.65 | 9.07 |
| 614 | 626 | workgroups | **101.04** | 0.16% | **14.860** | 0.29% | 12.66 | 8.80 |
| 8k | 8204 | shipped | **85.01** | 0.50% | **12.754** | 0.17% | 14.16 | 8.70 |
| 8k | 8204 | workgroups | **85.52** | 0.14% | **12.911** | 0.26% | 14.15 | 8.77 |
| 16k | 16 394 | shipped | **74.72** (n=2) | 0.71% | **12.067** (n=2) | 0.01% | 15.54 | 8.82 |
| 16k | 16 394 | workgroups | **75.00** | 1.02% | **12.329** | 0.10% | 15.54 | 9.13 |

**N4's decode win reproduces in an independent run:** +1.07% @8k (12.754 →
12.911) and **+2.17% @16k** (12.067 → 12.329), against control spreads of
0.17-0.47%. At 614 it is a dead heat (14.862 vs 14.860), which is the expected
shape: below 4096 keys the decode kernel is not the MMA one and the split rule
does not apply. The prefill column moves by +0.14% / +0.60% / +0.37% — noise, and
it must be, because the arms differ only in a decode knob.

**AC4: `phys_footprint` 12.65-15.54 GiB**, against the 20 GiB budget, and the
`ru_maxrss` column beside it reads **5.5-6.7 GiB lower** — the AGENTS.md 10.3 trap,
measured again on every single run. **`swapouts` did not move** across any of the
sixteen runs (738 188 448 at the first, 738 264 408 at the last, and the counter is
cumulative since boot — so the engine caused **zero** swaps).

**AC5, five reference prompts, greedy, diffed between arms:** all five produce
**byte-identical text** (`p1` Paris, `p2` 366, `p3` CCCXCI, `p4` the additive
primaries, `p5` "there are no such numbers"). Only the `loaded in NNNN ms` banner
differs, as it must. The diff is the test; eyeballing five answers is not.

**Full workspace suite:** 116 test binaries ok, **4 failed** —
`perf_probe_concurrency_dependency_graph::no_device_side_sampler_exists`,
`perf_probe_dispatchsel_columns::the_colmajor_contig4_kernel_is_not_on_the_decode_path`,
`perf_probe_host_ops::the_device_token_handoff_exists_and_the_row_follows_the_pick`,
`probe_cache_hierarchy::activation_loads_are_4_byte_alignable_and_collapse_2_to_1`.
**All four verified present at base `b92234f`** by building the base in a separate
worktree with its own `CARGO_TARGET_DIR` and running those four binaries: same four
names, same counts. **None is attributable to this work.** New in this session and
passing: `attention_prefill_pass_bit_exact` (17) and `attention_rescale_guard` (8).

### 20.1 A filed crash report that DOES NOT REPRODUCE (F7 — withdrawn as a defect)

**Original claim, now withdrawn:** that *"A ball is thrown straight up. What
colour does it turn at the top of its flight?"* fails the engine with

```
Metal command buffer failed (unlabelled) status=Error
```

at `--max-new-tokens >= 32`, passes at 16, and is **"reproducible every time"**,
and that it was **"verified pre-existing, not this spec's regression"** by
reproducing it on base `b92234f` built in a separate worktree.

**What was actually observed:** the error, on at least one run of the original
session, on the shipped binary, on that one prompt. That is all.

**Why it is withdrawn.** The adversary could not reproduce it in **19 attempts** —
`--max-new-tokens` at 16 / 17 / 24 / 31 / 32 / 33 / 64 / 96 / 128, **all rc=0**,
every one answering "It does not change color." — plus 10 further attempts at 96
tokens (10/10 rc=0), plus `scripts/refprompts.sh` twice end to end (rc=0 both
times, all five answers correct: `p1` Paris, `p2` 366, `p3` CCCXCI, `p4` the
additive primaries, `p5` no such numbers). Evidence:
`adversary_evidence/f7_refprompts.log`.

**"Reproducible every time" is therefore false on this commit, and I am not going
to substitute a different defect.** Three separate claims in the original section
have to be given up:

| claim | status |
|---|---|
| the prompt fails at `--max-new-tokens >= 32` | **not reproducible**, 29 attempts |
| "reproducible every time" | **false** — one observation, restated three times |
| verified pre-existing by reproducing on base `b92234f` | **unverifiable from what is committed** — no log of the base run exists, and nobody has re-attempted it |

**The most plausible reading, offered as a reading and not as a finding:** this
was a *symptom of the OQ8 memory stall*, not a code defect. The session's own
records say that on this box `unused` falls to **0.03–0.08 GiB** while the model
runs, that good runs read 7.5–9.5 GiB in the compressor, and that the documented
stall is a **99× pessimisation**. A `Metal command buffer failed` with nothing
after it is what a command buffer looks like when the machine is at its memory
ceiling and the buffer or the device is in trouble — and the prompt reproduces
fine, which is the strongest single fact against a prompt-dependent code path. I
am **not** claiming that explanation is established; I am recording that the
defect report as filed cannot be acted on, because it does not reproduce and its
own supporting leg is unlogged.

**What is substituted instead of a bug report.** The thing that *was* real and
does block work is already recorded as OQ8 and was independently strengthened by
the same adversary: **24k measurement is not possible on this box while the
desktop is resident**, and a memory-pressure gate cannot tell a good run from a
stalled one (§15.6, and now at *equal context length*: a 2.6×-pessimised sample
sitting inside the good envelope). If someone still wants the Metal error, the
useful next step is to capture the failing run's `phys_footprint` and `gpuq` poll
trace **while it is failing** — not the prompt, which reproduces fine — and to
commit that log. Without a captured failing run there is nothing to bisect.

### 20.1a What the scorecard reference-prompt check DOES establish

The check itself is sound and its result stands: `scripts/refprompts.sh` on the
shipped binary returns rc=0 with all five answers correct, twice, and the text is
byte-identical between the two arms (§20). The check was never the problem; the
crash filed from it was.

### 20.2 What the scorecard does NOT establish

* **AC1 (decode ≥ 14 @24k) is still unmeasurable** (OQ8). The 16k row above is the
  largest context measured. Three attempts at 24k decode have produced invalid
  samples and §16/§15 established that this box's memory signals cannot separate a
  stalled run from a good one. Not re-attempted, per the brief.
* **AC2 (prefill ≥ 90 @24k) is unmet and this session did not move it.** 16k
  measures 75.0 tok/s; the spec's own arithmetic (§12.1) puts the 24k requirement at
  attention 149.7 s → ≤ 56 s, a 2.7× kernel speedup. N1 was the strongest remaining
  idea and it measured neutral. Prefill is 20% short, not 25% — the honest summary
  of four levers tried (T3, T4f, T2b, N1) against one target.
* **The `workgroups` arm is not shipped.** `KAPLUMBA_DECODE_SPLIT_RULE=workgroups`
  makes it available without a rebuild; the default flip is one line and is the
  owner's decision.
* `phys_footprint` at 16k is 15.54 GiB, not the 18.35 GiB §11.1 recorded at 24k —
  consistent, since that run was 24 590 tokens against 16 394 here.

---

## 21 — Post-scorecard: the positive control that was not one

`scripts/test_gpuq_mem.sh` finished the session with an end-to-end positive control
— *the harness must be able to dispatch an arm at all* — and that control was wrong
in three separate ways before it was right. Each was found by reading its own
output, not by a sabotage.

1. **It pointed at a branch the harness no longer had.** The committed harness
   (`bench_decode_splits.sh`) takes `LOG`/`OUTDIR` and dispatches for real; the
   positive control still passed `DRY_RUN=1`, which only the pre-commit harness
   understood. The harness wrote its summary into the default `OUTDIR`, the log at
   `$LOG` was never created, and **both** assertions reported FAIL — while the exit
   code alone was being read as "harness fine".
2. **`bench_decode_splits.sh` silently ignored `LOG=`.** It wrote
   `OUTDIR/summary.tsv` unconditionally. Any caller passing `LOG=` got the summary
   somewhere else and nothing where it asked — which is precisely why (1) failed
   with "no such file" rather than "no VALID arm". Fixed: `SUMMARY=${LOG:-$OUTDIR/summary.tsv}`.
3. **The stub `gpuq` did not run the command it was handed.** Its `try` branch was
   `exit 0`, so the stub binary was never invoked, the log was empty, and the
   assertion was reading an artefact of the stub rather than of the harness. Fixed
   to `exec "$@"`. It also has to print `time -l`'s shape (the OQ5 CPU check reads
   it) and to sleep long enough for the memory poller to land at least twice —
   "no poll records" is a verdict the harness is *right* to return, so a control
   that races the poller tests the wrong path.

**Sabotage-verified after the fix:** making the harness skip every arm but one
(so it never dispatches) fails the positive control with
`FAIL: healthy box did not dispatch a VALID arm`, and the skip-side assertions
still pass — i.e. the two checks are independent, which is what a positive control
is for. Final state: `bash scripts/test_gpuq_mem.sh` → **37 ok, 0 FAIL, ALL PASS**.

The generalisable point, and the third instance of it in this spec: **a "positive
control" written against a harness is a test of the harness, and it can pass
because the harness is not there.** Mine reported success for a harness that had
been replaced. The check that caught it was making an *absent file* a failure
rather than an absence.

`TD_OVERRIDE` / `TD_KEEP` are exposed so a failing run's scratch directory can be
inspected instead of being deleted by the EXIT trap — which is how (1) and (3) were
diagnosed at all.

---

## 22 — F1: the microbench IS an attribution instrument, and the flash kernel's cost is 88% MMA issue

This is the record for the single most consequential finding of the fixer pass.
It withdraws a standing conclusion (OQ7, "open"), withdraws two sections of
design reasoning (§9.1, §9.1b, §9.2) that rested on it, corrects §10.2 #3, and
then — for the first time in this spec — produces a *usable* attribution of the
flash kernel's time and spends it on the AC2 question (§23).

### 22.1 The mistake, in one line

`attention.metal` is `include_str!`'d into the crate (`attention.rs:12`):

```rust
const SOURCE: &str = include_str!("attention.metal");
```

so editing the `.metal` has no effect until the crate that `include_str!`s it is
recompiled. §10's decisive arm was

```metal
    if (tid < 256u) { return; }        // top of attention_prefill_flash_body
```

and the flash prefill path dispatches `threads_per_group = 256`
(`attention.rs:1609`, the `Some((pipeline, block))` arm). `tid` is
`thread_position_in_threadgroup`, so `tid < 256u` is true for **every** thread:
the arm was not "256 of 256 threads retired", it was **a completely empty
kernel**. It read **964.79 ms** against a 958.39 ms baseline.

An empty kernel at that grid, in the same binary, with the same buffers, costs
**0.19 ms**. The table did not fail to reproduce; **it measured the wrong
binary.**

§10.1's ruled-out list did not catch this because every item in it is about
whether a build succeeded or which *pipeline* was dispatched:

| check | what it establishes | what it does not |
|---|---|---|
| `cargo build` returns 0 | the crate compiled | that the edited `.metal` was in it |
| `strings … \| grep -c KAPLUMBA_ABLATE_exit` = 1 | *some* patched source string is in the binary | that it is the source of the *timed* kernel |
| `eprintln!` prints `qk=Flash` | the right pipeline is dispatched | that its shader body is the edited one |
| `ksteps = 0` fails a correctness test | *that edit* reached the shader | that any *other* edit did |

The fourth row is why the section is internally inconsistent rather than simply
wrong: `ksteps = 0` demonstrably reached the shader (it fails
`attention_flash_matches_scalar_on_the_full_shape`) and yet moved the time by
+0.9%. That cannot both be true of one build. The honest conclusion is that
**these arms were produced across more than one build and no single build's
contents were pinned.**

### 22.2 The fix: put the ablation in the uniform block

One build, one binary, every arm. `AttentionParams` gained a `ablate: u32`
(TEMPORARY, reverted — see §22.6) carried in the existing constant buffer, and
`attention_prefill_flash_body` branches on it. Because the arm travels *with the
dispatch*, it is impossible to time a stale shader: if the field is not in the
shader, arm 1 cannot read 0.19 ms.

`crates/engine/tests/attention_qk_mma_bench.rs::attention_flash_ablation_arms`
then sweeps every arm, **interleaved** (`0 1 2 … 0 1 2 …`, 5 rounds, one untimed
warm per arm per round) so clock and thermal drift cannot masquerade as an arm
difference, and it asserts `flash_attention_available()` first — §10.2 #4's
standing instruction. `ABL_ARMS=0,2,11` restricts the sweep, which matters
because the full battery over three shapes is 18 minutes.

Evidence: `fixer_evidence/f1_ablation/f1_flash_ablation.log` (arms 0-9),
`f1_qk_ilp.log` (arms 11/12), `f1_threadgroup_loads.log` (arms 13/14).

> **Cross-run drift, stated because it matters for reading these numbers.** Three
> runs of the *same* baseline geometry (24 576 keys, 1024 queries, `BQ=1`) read
> 1143.6, 1201.3 and 1302.5 ms — a 14% spread on an unmodified kernel. **Only
> ratios within one run are comparable**, and every table below is one run.

### 22.3 The result

24 576 keys, 1024 queries, full 32q/4kv/d512, `flash`, 5 interleaved rounds,
medians with their min/max. Ratios are against arm 0 in the same run.

| arm | what it removes | `BQ=1` ms | ratio | share removed |
|---|---|---|---|---|
| 0 | nothing (baseline) | 1302.5 | 1.00 | — |
| 1 | **EVERYTHING** (immediate return) | **0.186** | **6 162x** | 100% |
| 2 | QK MMAs + its `kf`/`qf` loads | 576.9 | 2.26 | **55.8%** |
| 3 | PV MMAs + its `af`/`bf` loads | 777.5 | 1.68 | **40.5%** |
| 4 | `exp`/`simd_sum`/`simd_max` | ~baseline | 1.00 | **0.4%** |
| 5 | nothing; K fragment from a fixed row (L1) | 1079.7\* | 1.06 | 6% |
| 6 | nothing; V fragment from a fixed row (L1) | 1072.3\* | 1.07 | 7% |
| 8 | nothing; **every** fragment load L1-resident | 1051.5\* | 1.09 | **8%** |
| 13 | nothing; Q/P from fixed threadgroup addrs | 1256.6 | 1.04 | 4% |
| 14 | nothing; Q/P fragment loads deleted | 1161.8 | 1.12 | **11%** |
| 9 | the `msv == 1` rescale guard (F8) | 1184.4\* | 0.97 | **-4.9%** |
| 7 | nothing; Q fragment hoisted into registers | 2198.7\* | 0.52 | **-92%** |
| 11 | nothing; 2 independent QK accumulator chains | 1205.4 | 1.00 | 0% |
| 12 | nothing; 4 independent QK accumulator chains | 1246.3 | 0.96 | -4% |

\* from the first run (arms 0-9), whose baseline was 1143.6 ms; ratios are
within-run. Arms 0/2/3/11/12/13/14 from the second and third runs; their ratios
agree with the first run's arm 2 (2.00 → 2.13 → 2.26, all "QK is about half").

`BQ=2` (which is what production selects at `key_count >= 16384`) is
**4-19% slower than `BQ=1` on this bench** — 1554.7 vs 1302.5 ms at 24 576 keys,
530.8 vs 435.8 at 8 192. See §23.3.

**What survives, and what it costs to be wrong by:**

| §9/§10 claimed | measured | verdict |
|---|---|---|
| loads 44% | **all fragment loads together 8% (device) + 11% (threadgroup)** | **wrong by ~4x in the wrong direction** |
| MMA issue 52% | **~77%** (QK + PV MMAs) | direction right, magnitude low |
| softmax 3% | 0.4% | right |
| "removing the QK and PV instructions moves nothing" | removing QK is 2.26x, removing PV is 1.68x | **the claim was an artefact of a stale build** |
| §10.2 #3: "invalid attribution instrument" | a 6 162x signal | **wrong** |

**The flash kernel is MMA-issue-bound.** 88% of it is the
`simdgroup_multiply_accumulate` instructions; every load together is 12-19%; the
softmax is free; and the four candidate mechanisms for the MMA stall are all
eliminated by measurement:

* **not the accumulator dependency chain** — 2 and 4 independent QK chains are
  1.00x and 0.96x (arms 11/12). This is the interesting one, because
  `examples/mma_peak.rs` records 5.75 TFLOPS at 1 chain, 7.65 at 8 and 15.49 at
  16, so a 64-deep dependent chain *should* be worth ~4x. It is not, which means
  the MMA issue rate here is not set by the chain at all.
* **not register pressure in the other direction** — hoisting the Q fragment into
  registers is **-92%** (arm 7), i.e. `simdgroup_half8x8 qreg[BQ][NTILES]` (64
  registers per lane at `BQ=1`, 128 at `BQ=2`) spills. This kills the one lever
  §23's instruction-count model wanted.
* **not the memory system** — arm 8, all fragment loads L1-resident with the MMA
  count untouched, is 1.09x.
* **not occupancy** — §9.2's threadgroup-memory reduction (19.5 -> 11.5 KB) was
  neutral, which is discredited as a measurement, but arm 14 removes 11% of loads
  *and* their latency without touching threadgroup memory at all.

### 22.4 OQ7 is closed

The question was "why does deleting the QK and PV instructions not speed the flash
kernel up?". It was never a hardware mystery: **the instructions were not being
deleted**, because the patched shader was never in the binary. The control §10.3
says was too cheap to bother with is worth **0.19 ms**, and running it is what
closes the question. `spec.md`'s OQ7 and NEXT.md §7.5 are corrected accordingly.

### 22.5 OQ7's answer, restated as a design constraint

**Do not ship an ablation whose result is "the change did nothing" until a
positive control has moved the same instrument in the same binary, and do not
believe a `threads_per_group`-sized early return has emptied a kernel.** The
second half is the specific trap: `if (tid < 256u) { return; }` in a 256-thread
kernel looks like "a quarter of the threads" to anyone who has not checked the
dispatch. It is the whole kernel.

### 22.6 The knob is reverted

`AttentionParams::ablate`, `attention_ablate_arm()`, `set_attention_ablation_arm()`
and the nine gated regions in `attention.metal` are **TEMPORARY and are removed
again** once the campaign is recorded. Carrying a runtime branch through the hot
path to keep an instrument alive is the same category of mistake as the
unmeasured guard (F8): it costs a uniform load and a branch per region on every
dispatch in exchange for numbers that are already in
`fixer_evidence/f1_ablation/`. What survives is the *harness*
(`attention_flash_ablation_arms`, which asserts the pipeline and interleaves) and
the *rule*. If the ablation is ever needed again, re-add the field first and make
`arm 1` read ~0.2 ms before believing anything else.

---

## 23 — F5: AC2 re-derived, the o_proj lead killed, and the honest ceiling

Full derivation: `fixer_evidence/f5_arithmetic.md`. This section is the summary
and the argument; the file has the per-projection FLOP tables.

### 23.1 The concrete lead `spec/decode-20tps` handed over is dead for prefill

`spec/decode-20tps` §R1 recorded, as the leading decode-side cause:

> **o_proj at 100 GB/s is at HALF the bandwidth of every other projection.**
> Microbenchmark confirms: K=8192 → 120.7 GB/s, K=21504 → 199.3 GB/s. The fix:
> overlap o_proj weight prefetching with the preceding attention phase.

That measurement is `gemv_bandwidth.rs` at **`M = 1`** — a decode GEMV. Prefill is
`M = PREFILL_CHUNK = 1024`, and at that shape the projection is not
bandwidth-bound at all. From the 16 394-token per-kernel profile:

| projection | measured `own_ms` | TFLOP/s | % of the 15.6 TFLOPS peak |
|---|---|---|---|
| v_proj | 5 853.3 | **8.63** | 55% |
| k_proj | 6 495.2 | **7.78** | 50% |
| gate/up/down (each) | ~34 326 | **6.63** | 42% |
| o_proj | 15 574.5 | **6.49** | 42% |
| q_proj | 15 869.4 | **6.37** | 41% |
| **all seven** | **146 770.2** | **6.73** | **43%** |

Three things kill the lead:

1. **o_proj runs at 98% of `down_proj`'s efficiency.** `own_ms` ratio
   15.574/34.339 = 0.453 against a FLOP ratio of 0.444. There is no deficit.
2. **The ordering is not a bandwidth signature.** The two slowest projections are
   the two with the *longest* `K` (o_proj `K` 8192/16384, q_proj `N` 16384), and
   the fastest is the one with the *shortest* `K` **and** the narrowest `N`
   (v_proj, `K` 5376, `N` 4096-8192). A bandwidth story cannot put the
   narrowest, shortest-K projection at the top.
3. **273 GB/s is irrelevant here.** Per chunk the GEMM streams the whole 11.6 GB of
   packed weights once against `6.168e13` FLOP — an arithmetic intensity of
   **5 317 FLOP/byte**. At 6.73 TFLOP/s that needs **1.27 GB/s of a 273 GB/s
   wire**, 0.5% of it. Exactly the same correction §12 made for attention, and it
   had not been made for the GEMM.

The whole 2% o_proj deficit is **0.45 s of a 273.2 s budget (0.16%)**. This is
the first time the prefill GEMM has been decomposed per projection at all, and
the answer is that there is nothing anomalous in it to decompose.

### 23.2 Where the 223.5 s of context-independent cost actually goes

| | s at 24 590 | share | TFLOP/s |
|---|---|---|---|
| down_proj | 51.5 | 23.0% | 6.63 |
| up_proj | 51.5 | 23.0% | 6.63 |
| gate_proj | 51.5 | 23.0% | 6.63 |
| q_proj | 23.8 | 10.7% | 6.37 |
| o_proj | 23.4 | 10.5% | 6.49 |
| k_proj | 9.7 | 4.4% | 7.78 |
| v_proj | 8.8 | 3.9% | 8.63 |
| element kernels + head + norms | 3.3 | 1.5% | — |
| **total** | **223.5** | | **6.73 = 43% of peak** |

**69% of it is the FFN**, and all of it is at 41-55% of a *measured* tensor peak
with no spread worth attacking. `spec/engine-performance`'s independent record of
6.13-6.44 TFLOPS for this GEMM, after rejecting 64x64 tiles, `KCHUNK=32`, staged
`astage`, 32 accumulators, 8->4 simdgroups, the codebook LUT, `extract_bits`, concat
fusion and resident weights, and declaring **7.5 TFLOPS a wall**, is consistent
with 6.73 to within its own spread. **The GEMM is at its wall and there is no
projection-shaped hole in it.**

### 23.3 Attention, split by layer kind — and one free 5-19%

§22 measured both the full-attention (`flash`) shape and the sliding shape's
non-flash arm. Solving the per-layer `own_ms` totals (full 7 566.7 ms, sliding
2 653.1 ms, each layer's own GEMM `FLOP`/6.73 TFLOP/s) for the attention share:

| | attention per chunk per layer | FLOP at 16 394 | TFLOP/s | % peak |
|---|---|---|---|---|
| 10 full layers | 270.0 ms | 8.814e13 | **1.91** | 12% |
| 50 sliding layers | 10.0 ms | 2.751e13 | **1.85** | 12% |

**Both layer kinds land on 1.9 TFLOP/s.** That is the credibility check: it
matches `attention_qk_mma_bench` on the same geometry (1.27-1.46 TFLOP/s across
runs) and it reproduces the adversary's *total* 24 590-token attention figure
(127.5 s against 126.1 s here). The adversary's internal split (50.2 s full /
8.9 s sliding) does not reproduce; its total does, and the total is what AC2
depends on. Two independently written attention kernels in this same file —
`attention_prefill_flash_body` and `online_attention_qk_mma` — both converge on
~1.8-1.9 TFLOP/s, which is 12% of the `mma_peak.rs` figure and is much more
likely to be a property of the operand shape on this hardware than a bug in one
of them.

**One free finding.** `FLASH_QUERY_BLOCK_MIN_KEYS = 16_384` was chosen from §5's
byte model, which §12 withdrew, and §12.2 flagged the threshold as suspect
without testing it. It is one env var (`KAPLUMBA_FLASH_BQ=1`), and §22 arm 0 says
`BQ=1` is **4-19% faster** than `BQ=2` on the bench at both 24 576 and 8 192 keys.
End-to-end numbers are in §23.6.

---

## 26 — F8: the `msv == 1` rescale guard is worth 4.9%, so it stays

§17.1 justified this guard on **exactness alone**:

> It is exact because `ms = exp(m_old - m_new)` is **exactly** 1 when the two are
> equal, and `x * 1.0f == x` in IEEE-754.

That is a correct argument for *correctness* and no argument at all for *speed*,
and §17.1 admits it: "§17.1 argues exactness only, and NEXT.md §7.2 lists it as
'landed' work without a number." NEXT.md §7.5's dead list called it one of the
things worth knowing. The adversary's F8 sharpened this into a decision: **measure
the benefit and keep it, or withdraw it — do not keep an unmeasured optimisation
in the hot path on exactness alone.**

Measured as §22's **arm 9** (`ABL == 9u` skips the `if (msv == 1.0f) { continue; }`
and always rescales), same binary, same grid, same buffers, interleaved:

| shape | arm 0 (guard, shipped) | arm 9 (guard removed) | guard is worth |
|---|---|---|---|
| 24 576 keys / 1024 q, `BQ=1` | 1143.6 ms | 1184.4 ms | **3.6%** |
| 24 576 keys / 1024 q, `BQ=2` | 1209.3 ms | 1268.7 ms | **4.9%** |
| 24 576 keys / 4096 q, `BQ=1` | 4570.2 | 4724.8 | **3.4%** |
| 24 576 keys / 4096 q, `BQ=2` | 4820.2 | 5033.0 | **4.4%** |
| 8 192 keys / 1024 q, `BQ=1` | 376.6 | 385.9 | **2.5%** |
| 8 192 keys / 1024 q, `BQ=2` | 401.4 | 420.5 | **4.8%** |

**2.5-4.9%, consistent across six shapes, monotone in `BQ`** (which is what the
mechanism predicts: the guard's cost is one threadgroup read + compare + branch per
(query, head, tile) and its saving is `BQ * NPER * 2 = 16 BQ` fp32 multiplies, so
the ratio should *fall* with `BQ`... it rises, so the saving is winning on both
sides and the honest summary is "a few percent, reliably").

**The instrument resolves this size, and says so.** In the same sweeps, arm 14
deletes the Q/P fragment loads outright for **11-12%**, arm 8 makes every device
load L1-resident for **9%**, and arm 11 (2 independent QK chains) reads **1.00x**.
So the instrument moves 11% and resolves 0%; a 3-4% reading is a real reading.

**Decision: keep the guard.** The reasoning it now rests on is a measurement, not
an IEEE-754 argument, and the §9.2 caveat about *why* it is exact is unchanged.

**On the missing citation.** §17.2's parenthetical — *"the sabotage it describes
is a manual `.metal` edit"* — is true and the sibling test
`a_misfiring_guard_is_caught` cited at `attention_rescale_guard.rs:384` **does not
exist**. The adversary ran the sabotage by hand and got `0.21362305` — 43x the
5e-3 tolerance, 7 of 8 failing — which is exactly the number in the file's comment,
so the two accounts agree and the claim is true; only the *citation* is fictional,
and the control is not in the tree. **Left as it is, deliberately:** a test that
edits a `.metal` source file and rebuilds is not a unit test, and `scripts/` has no
mechanism for one. What replaces it is §22's arm 9, which is the same sabotage
expressed as a uniform value, needs no rebuild, and is in the tree as part of the
ablation harness. **Read this section as the record of the control, not as a
citation to a test.**

---

## 24 — F4: the `workgroups` default flip, measured where it can differ

### 24.1 Why the existing evidence was vacuous, and what the real reason is

`decode_splits_with` ends in `wanted.min(key_count.div_ceil(ATTENTION_MIN_KEYS_PER_SPLIT = 64))`,
so at the ~120 tokens of the five reference prompts **both rules return 2 splits**:
the two arms are literally the same computation and their diff can only ever be
empty. N6's AC5 evidence — "five reference prompts byte-identical between arms" —
was a test that could not fail. **The first key count at which the two rules
differ is 896** (ThreadTarget 16 vs WorkgroupsPerCore 13), i.e. any conversation
longer than about 900 tokens. Everything the scorecard measured (626, 658, 832) is
below that line.

And the reason the flip is a **10-layer** change rather than a 60-layer one is not
the one §18 recorded. A sliding layer's key count is

```
gpu_cache.key_count(false) = next_position.min(capacity)
```

i.e. the **1024-slot ring**, not the context — so it is <= 1024 at every position
and both rules return 16 at every key count below 896 and above it. **This is the
`sliding ring`, and until now it was nowhere in writing.** (§18's own stated reason,
`288/16 = 18` clamped to 16, is also correct; both are consequences of the cap.)

### 24.2 The measurement

New harness `scripts/bench_split_rule_quality.sh`: one greedy generation per
(context, arm), `--temperature 0` (verified greedy: `sampler.rs:450` returns
`argmax` when `temperature == 0.0`), `--show-ids` so the comparison is on **token
ids** rather than on text, arms interleaved per context, per-arm files, never
piped through `tail`. Contexts 1024 / 4096 / 16394 tokens — all above the 896
boundary — 512 generated tokens each, `--seed 1234`.

| context (prompt tokens) | arm | generated | token ids differing from `shipped` | body |
|---|---|---|---|---|
| 1024 | `shipped` (ThreadTarget, 16 splits) | 512 | — | — |
| 1024 | `workgroups` (13 splits) | 512 | **0** | byte-identical |
| 1024 | `splits13` (`KAPLUMBA_DECODE_SPLITS=13`) | 512 | **0** | byte-identical |
| 1024 | `passes0` | 512 | **0** | byte-identical |
| 4096 | `shipped` | 512 | — | — |
| 4096 | `workgroups` | 512 | **1** (the 512th) | differs |
| 4096 | `splits13` | 512 | **1** (the 512th) | differs |
| 4096 | `passes0` | 512 | **0** | byte-identical |
| 16394 | `shipped` | 512 | — | — |
| 16394 | `workgroups` | 512 | **0** | byte-identical |
| 16394 | `splits13` | 512 | **179** | differs |
| 16394 | `passes0` | 512 | **0** | byte-identical |

**1 differing token out of 1 536 generated for the rule under test, at one of three
contexts, and it is the last token of the run.** The two bodies that differ end the
same way, one cut at `"...proportional to the mass and"` and one at
`"...proportional to the mass."` — the 512-token budget simply fell on the other side
of a token.

**The `splits13` arm is NOT the rule, and the difference is the ring.**
`KAPLUMBA_DECODE_SPLITS=13` returns `pinned` from `decode_splits_with` *before*
`wanted.min(affordable)` runs, so it pins **every** layer kind; whereas
`KAPLUMBA_DECODE_SPLIT_RULE=workgroups` gives 13 to the 10 full layers and **16** to
the 50 sliding ones (the ring caps them: `wanted = 288/16 = 18` clamps to 16, and
`affordable = 1024/64 = 16` clamps it again). At 16 394 tokens:

| arm | full layers | sliding layers | tokens differing from `shipped` |
|---|---|---|---|
| `shipped` (ThreadTarget) | 16 | 16 | — |
| **`workgroups`** | **13** | **16** | **0 / 512** |
| `splits13` (pinned) | 13 | **13** | **179 / 512** |

**The only difference between the last two rows is the sliding layers' split
count**, so this is a direct, quantitative confirmation of §24.1: the rule reaches
**10 of 60 layers**, and it is the *sliding* layers that are numerically sensitive
to their split count, not the full ones. `splits13` must not be read as a
measurement of the rule — it is a measurement of a configuration this spec has no
reason to ship.

At 4096 tokens `workgroups` and `splits13` **agree** (both differ from `shipped` by
the same single last token). **This is reported, not explained**: the per-step
arithmetic is identical at both contexts (the ring caps the sliding key count at
1024 either way, and `affordable` is 65 vs 257, neither of which clamps), so whether
a bounded-ulp perturbation flips an argmax is a near-tie lottery whose size is not
predictable from the split count alone. Anyone who wants to use `splits13` as a
sensitivity probe should read this as its calibration: **at 16k it is a 35%
perturbation, which is almost exactly the "43 of 128 = 34%" this comment used to
quote.** The most likely reading is that `spec/decode-19tok` measured a pinned
count, or measured before the sliding ring cap existed — either way the number was
real, and it was real for a configuration that is not the rule.

**Harness positive control.** `workgroups` and `KAPLUMBA_DECODE_SPLITS=13` are two
different ways of specifying the same computation on the full layers, and they are
**byte-identical at 4096** (same sha of the id stream). Without that agreement the
harness could not distinguish "the rule changed the answer" from "the harness is
broken".

**Repetition check** (`scripts/check_repetition.sh`, distinct n-gram ratio). The
instrument's own control is a deliberately collapsed stream:

| stream | distinct_4 | distinct_8 | distinct_16 | distinct_32 |
|---|---|---|---|---|
| collapsed control (124 copies of `the`) | 0.0083 | 0.0085 | 0.0092 | 0.0108 |
| 1024 / shipped | 0.8409 | 0.9584 | 0.9920 | 1.0000 |
| 16394 / shipped | 0.8723 | 0.9663 | 0.9960 | 1.0000 |
| 16394 / workgroups | 0.8723 | 0.9663 | 0.9960 | 1.0000 |

No collapse, no NaN/Inf (a non-finite logit is a hard `SamplerError`, so a
512-token generation that completes cannot contain one), and the arms are
indistinguishable on this metric too.

### 24.3 What the shipped comment says, and why it is stale

`crates/kernels/src/attention.rs:433-454`, unmodified by this spec, in the file
that ships:

> This change is smaller (1 ULP on 3-4 of 16384 attention outputs, on 10 of 60
> layers) and changes **43 of 128**. [...] The measured consequence is in
> `spec/decode-19tok`: 1.34x on the full-attention layers at 8k of context, **and
> a different token stream**.

**The principle is right and the magnitude is stale.** 43 of 128 = 34% was measured
in `spec/decode-19tok`; on the **shipped** kernel, the rule as written (full layers
13, sliding layers 16) is **1 token in 1 536 = 0.065%**, a ~500x smaller
perturbation, measured over three contexts. The ~34% *is* reproducible on this
machine — but only for `KAPLUMBA_DECODE_SPLITS=13`, which also moves the sliding
layers (**179 of 512 = 35%**, §24.2). So the old number was real for a
configuration that is not the rule. The MMA kernel's combine reads the same
partials, so reordering its summation is a far smaller perturbation than reordering
`simd_sum` reductions was on the pre-T2b scalar kernel. The argument to keep is
therefore *not* "it is bit-exact" — it is **not**, and the comment is right about
that — but "it is bounded-ulp, and bounded-ulp has been measured at 0.065% of
greedy tokens on the shipped kernel, over three contexts, including one at 16k,
and the perturbation is confined to 10 of 60 layers by the sliding ring."

### 24.4 Recommendation: flip, but it is not free, and it is not done here

* **Recommended:** make `workgroups` the default. It is worth +1.55% to +2.17%
  decode at 16k (N4, re-confirmed by the adversary with an 8-arm curve monotone to
  13 and a −1.5% cliff at 14), the quality cost is 1 greedy token in 1 536 measured
  across three contexts above the 896 boundary, and the change touches 10 of 60
  layers for structural reasons (§24.1).
* **Not done here, for a reason that is not "it needs an owner":** this spec's own
  quality gate requires **byte-identical greedy text against base `f2b2da1` for
  the default configuration**, and flipping the default by construction changes
  1 token in 1 536. A +2% decode win is not worth spending that gate on, and the
  gate exists precisely so that a speed number attached to a numerics change is
  not read as free. **Whoever wants the flip should amend the gate first, in
  writing, with the 1/1536 number attached.**
* **The comment at `attention.rs:433-454` must be corrected either way.** Leaving
  "changes 43 of 128" in the shipped source, next to a `workgroups` opt-in, is the
  thing that makes the next reader distrust the opt-in. Its *reason* (fp addition is
  not associative) survives; its *magnitude* does not.

### 23.4 The budget, re-derived

§12.1's "measured attention ~149.7 s" is `366.6 − 216.9` — a **subtraction from a
wall clock**, and the residual silently contains the inter-dispatch gap, which is
not attention and cannot be removed by making attention faster. Measured
per-kernel instead (`KAPLUMBA_GPU_PROFILE=1`, 16 394 tokens):

| term | per token | s at 24 590 |
|---|---|---|
| GEMM + other (all seven projections + elements) | 9.087 ms | **223.5** |
| attention | quadratic | **126.1** |
| inter-boundary gap, 0.719 ms x ~22 530 boundaries | — | **16.2** |
| **total** | | **365.8 -> 67.2 tok/s** (measured 67.10 and 68.40) |

Against the 90 tok/s budget of 273.2 s:

* **239.7 s — 88% — is spent before attention begins.**
* attention gets **33.5 s**, not 56.1 s.
* `2.394e14` FLOP (full `1.9817e14` + sliding `4.126e13`) in 33.5 s is
  **7.15 TFLOP/s = 46% of the 15.6 TFLOPS tensor peak**, against a measured
  **1.9 TFLOP/s (12%)**. **A 3.8x cut, not 2.7x.**
* **The "flash-only" route is not available.** Holding the sliding layers at their
  measured 1.85 TFLOP/s leaves 11.2 s for the full layers, i.e.
  `1.9817e14 / 11.2 = 17.7 TFLOP/s` — **above the hardware's peak**. The sliding
  kernel has to move too. (The adversary reached 20.4 s / 9.7 TFLOP/s by crediting
  the sliding layers 13.3 s instead of 22.3 s, i.e. by a 1.85 TFLOP/s figure they
  did not measure.)
* The alternative — a **1.7x GEMM**, 9.0 -> 5.3 ms/token — is
  `1.022e15 / (0.0053 * 24590)` = **8.1 TFLOP/s = 52% of peak**... and the measured
  GEMM is 6.73 TFLOP/s against `spec/engine-performance`'s declared **7.5 TFLOPS
  wall**, so 8.1 is *above* the wall this repo already established after seven
  named rejections. **Neither route is incremental.**

### 23.5 The hunt: every lever the measured split suggests, and what it measured

This is the part the brief asked for and the part §12 was missing: with the
instrument working (§22), each lever implied by "88% is MMA issue, 12% is loads"
was built and measured rather than argued about.

| lever | predicted by | measured | verdict |
|---|---|---|---|
| halve the Q fragment loads by hoisting `qstage` into registers (`simdgroup_half8x8 qreg[BQ][NTILES]`, 64 registers/lane at `BQ=1`, 128 at `BQ=2`) | the loads are 12% and `qf` is loop-invariant in `(b, d8, head)` | **x0.52 / x0.50 — 92% SLOWER** | **negative.** It spills. This is the same trap `mma_peak.rs` records at 32 accumulators (15.49 -> 0.74 TFLOPS) |
| break the QK accumulator dependency chain (2 and 4 independent chains) | `mma_peak.rs`: 1 chain 5.75 TFLOPS, 8 chains 7.65, 16 chains 15.49 | **x1.00 and x0.96** | **negative.** And it contradicts the peak table, which is itself the interesting fact |
| `KAPLUMBA_FLASH_BQ=1` (drop query blocking entirely; `BQ=2` halves the K/V bytes and the grid) | §9.1b: bytes are 8%, so removing them cannot pay | see §23.6 | the one free lever that survived |
| kill the loads outright to find the floor | — | **x1.12** (arm 14) | so even a zero-load kernel is 12x the empty body |
| attack the GEMM's o_proj (the lead `spec/decode-20tps` handed over) | "o_proj at half the bandwidth, worth 2-3 ms/token" | **refuted, §23.1** | there is no anomalous projection |
| attack the 16.2 s inter-dispatch gap | 5.9% of the budget | it is profiler-inflated: the profiled wall is 224.6 s and the clean wall 218.4 s at the same prompt | not a production cost |

**The negative results are the finding.** Three independent candidate mechanisms
for the MMA stall — memory, dependency chain, register pressure — are each
eliminated by a measurement, and the remaining 75% of unexplained stall is in
`simdgroup_multiply_accumulate` issue on this operand shape. Two *independently
written* attention kernels in the same file (`attention_prefill_flash_body` and
`online_attention_qk_mma`) land on the same ~1.8-1.9 TFLOP/s, which is what makes
"this is the shape's rate on this hardware" more likely than "this is a bug in one
kernel".

**One instruction-count model was tried and falsified, and it is worth recording
because it was the most attractive lever.** Counting instructions per MMA gives
2.72 at `BQ=1` and 1.89 at `BQ=2`, so a perfectly issue-bound kernel would reach
`15.6 / 2.72 = 5.7` and `15.6 / 1.89 = 8.25 TFLOP/s`, and hoisting `qf` would
raise `BQ=2` to 9.5. Both predictions are wrong: **`BQ=2` is 19% SLOWER than
`BQ=1` on the same bench, and hoisting `qf` is 92% slower.** The instruction count
is not what the MMA is waiting for. Any future work that reasons from instruction
mix should read this paragraph first.

### 23.6 The honest ceiling

Every number in this table is a **ceiling** — it assumes a component reaches the
best rate it has ever been *observed* to reach, with zero overhead. It is an upper
bound on AC2, not a forecast.

| scenario | attention | total s @24 590 | tok/s |
|---|---|---|---|
| measured today | 103.8 + 22.3 = 126.1 | 365.8 | **67.2** |
| both attention kernels at their 15.49 TFLOPS **issue** peak (the `mma_peak.rs` 16-chain figure, with zero memory and zero everything else) | 12.8 + 2.7 = 15.5 | 255.2 | **96.4** |
| both at the 8-chain peak (7.65 TFLOPS, the rate `mma_peak.rs` measures for an 8-way accumulator structure like PV's) | 25.9 + 5.4 = 31.3 | 271.0 | **90.7** |
| both at 3.8x, i.e. exactly the ask | 33.2 | 272.9 | **90.1** |
| attention free | 0 | 239.7 | **102.6** |

**Read honestly, this says three things.**

1. **AC2 is arithmetically possible, but only just, and only through the tensor
   cores.** The 8-chain row lands at 90.7 — a 0.8% margin. The 15.49 row lands at
   96.4. Neither is reachable from 12%: the required lift is **4.0-8.2x on the MMA
   issue rate of both attention kernels**, and every lever that the measured split
   suggests is negative (§23.5).
2. **The GEMM is not the cheaper route.** It is at 43% of tensor peak,
   *uniformly across all seven projections*, and `spec/engine-performance` already
   declared 7.5 TFLOPS a wall there after seven named rejections. 8.1 TFLOP/s is
   above that wall.
3. **Therefore the honest recommendation is a target change or a rewrite, not
   another lever-shaped session.** Four levers have now been tried against this
   target (T3, T4f, T2b, N1) and a fifth (`KAPLUMBA_FLASH_BQ`, §23.7) is worth a
   few percent. The residual is 3.8x on a component whose cost is now *known* to
   be 88% one instruction class, with no measured handle on that instruction
   class's rate.

**The one concrete thing left to try, and it is cheap.** `mma_peak.rs` shows the
tensor rate depends on the **number of independent accumulator chains** and not on
the loads (1 chain 5.75, 8 chains 7.65, 16 chains 15.49 TFLOPS). The flash kernel
issues QK through **one** chain per tile. Arms 11/12 split it 2 and 4 ways and
gained nothing — which suggests the split was not actually made (the compiler kept
the chains serial), not that ILP does not matter. **The experiment to run next is
16 named scalar accumulator chains in the QK loop, exactly as `mma_peak.rs` does
it** — the source itself records that an *array* of 32 made the compiler spill and
collapse to 0.74 TFLOPS, so it must be named locals, and this repo has already been
burned by that trap once. If 16 chains really do give 15.49, the flash kernel's QK
ceiling moves from 1.37 to something an order of magnitude better and AC2 is back
in reach. That is a day's experiment and it is the only lever left standing.

---

## 25 — F3: the OQ9 prefill-pass default, decided and documented

### 25.1 What was actually true

| place | said | true? |
|---|---|---|
| `crates/kernels/src/attention.rs:242` | `PREFILL_PASS_KEYS = 8192`, returned by `prefill_pass_keys_for` when the env var is unset | **this is what shipped** |
| `implementation_notes.md` §16.4 ¶1 | "the **default must stay one dispatch**" | **false** |
| `implementation_notes.md` §16.4 ¶3 | "the gate ships **on by default**" | true |
| `spec.md` OQ9 | "ships … off with `KAPLUMBA_PREFILL_PASS_KEYS=0`" | **false** |
| commit `c61dcba` | "implementation kept, **default unchanged**" | **false** |

So every prefill dispatch with `key_count > 8192`, the flash shape, and a grid of
>= 512 threadgroups sliced. **F2 landed in between the adversary's measurement and
this one**, so the cost had to be re-measured rather than inherited: F2 removed the
20-threadgroup slices that the +77% inter-boundary-gap figure was taken on.

### 25.2 The cost of the shipped default, re-measured

`scripts/bench_prefill_passes.sh`, 16 394 tokens, `--max-new-tokens 1`, arms
interleaved per round, poll file and verdict kept per arm, 3 rounds:

| arm | r1 | r2 | r3 | mean | min-max | spread |
|---|---|---|---|---|---|---|
| `KAPLUMBA_PREFILL_PASS_KEYS=0` (one dispatch) | 75.299 | 75.144 | 75.369 | **75.271** | 75.144-75.369 | 0.30% |
| default `8192` (2-3 passes) | 74.955 | 75.133 | 75.427 | **75.172** | 74.955-75.427 | 0.63% |

**-0.13%, inside both arms' spread.** The adversary's pre-F2 figure was +0.13%, so
the sign is noise either way and the honest reading is "below the resolution of this
instrument at this context".

**Where the cost actually shows up: the host side.** `KAPLUMBA_GPU_PROFILE=1`, both
arms in the same session, 16 394 tokens:

| term | default (8192) | one dispatch (0) | delta |
|---|---|---|---|
| attention `own_ms` | 64 425.0 | 64 206.0 | **+219.0 (+0.34%)** |
| the seven projections | 158 051.4 | 158 092.8 | -41.4 (-0.03%) |
| everything else | 2 200.3 | 1 998.7 | +201.6 |
| **kernel sum** | **215 737.7** | **215 297.2** | **+440.5 (+0.20%)** |
| **inter-boundary gap** | **15 801.4** | **10 625.6** | **+5 175.8 (+48.7%)** |
| wall span | 237 234.3 | 230 323.7 | +6 910.6 (+3.0%) |
| profiled `prefill_tok_s` | 69.104 | 71.176 | -2.9% |

Attention itself is untouched (+0.34%, noise). The cost is **submission overhead**:
the pass chain adds boundaries, and each boundary is a place where the host and the
GPU can disagree. **F2 recovered about a third of it** — the adversary measured
**+77%** of the gap at the pre-F2 commit, where the chain also sliced
20-threadgroup grids; it is **+48.7%** now.

**The two measurements disagree and both are right.** The *clean* wall says neutral
because a pipelined run overlaps the extra submission with the GPU; the *profiled*
wall says +3.0% because the profiler serialises at every boundary, which is exactly
where the cost lives. Neither reading is a reason to ship a mechanism that recovers
nothing. Both are a reason to ship it **off**, where production does not pay it at
all. **Positive control** that the instrument can see a real attention change at
all: §25.5.

**The 64 MiB argument is void.** §16.4 ¶1 said the default costs "64 MiB of
scratch". It does not: `model.rs:3447` and `model.rs:4927` size
`prefill_pass_state_elements_widest` from `tokens` **unconditionally**, so the state
buffer is resident whether the passes are on or off. `phys_footprint` reads
15.49-15.56 GiB in both arms above, and it cannot differ: the allocation is the
same. **That was the strongest argument for turning the default off and it is not
one.**

### 25.3 The decision: the default is ONE DISPATCH, and `KAPLUMBA_PREFILL_PASS_KEYS=8192` is the opt-in

The reasoning, in the order that decided it:

1. **It is exact, so this is not a quality decision.** §25.4.
2. **It has no measured benefit at any context this box can measure** (§16.4's own
   conclusion, re-confirmed), and no measured cost. A tie.
3. **A tie does not justify a default.** A default is a claim. oMLX's 8192 is
   oMLX's default on hardware where its mechanism has a measured effect at 1M keys;
   copying the number without the effect is copying a constant. §16.4 ¶1's instinct
   was right and its stated reason (the 64 MiB) was wrong; the reason that survives
   is the one above.
4. **Nothing is lost.** The implementation, its 17 bit-exactness tests and F2's
   `PREFILL_PASS_MIN_GROUPS` gate all stay and all still pass. What changes is that
   8192 becomes an explicit opt-in instead of a silent default, and production stops
   paying extra dispatches for a mechanism that recovers nothing here.
5. **The cost of being wrong is one environment variable**, and the flag is already
   documented on the constant.

### 25.4 It is EXACT, and here is the strongest evidence in this spec for it

`scripts/bench_split_rule_quality.sh`, 16 394-token prompt, **512 tokens at
`--temperature 0.7 --seed 20260905`** (fixed seed, so the sampler is
deterministic), four arms, one generation each:

| arm | token ids | vs `shipped` | first differing token |
|---|---|---|---|
| `shipped` (OQ9 passes ON, the old default) | 512 | — | — |
| **`passes0` (`KAPLUMBA_PREFILL_PASS_KEYS=0`)** | 512 | **0 — byte-identical** | **never** |
| `workgroups` decode split rule | 512 | 99 | **407** |
| `splits13` decode split rule | 512 | 90 | **407** |

**At a sampling temperature with a fixed seed, the pass chain and the one-dispatch
body produce the same 512 tokens.** If a single logit had differed in the last ulp,
the two streams would have diverged somewhere in 512 draws at temperature 0.7 —
and they are identical for all 512. The same instrument, the same prompt and the
same run shows the contrast that makes this meaningful: the **decode split-rule
change diverges at token 407** and is 99 tokens different after that.

So the two perturbations in this report are cleanly separated, and the labels are
not interchangeable:

| change | label | evidence |
|---|---|---|
| OQ9 passes on/off | **exact** | byte-identical text at temperature 0 **and** at temperature 0.7 with a fixed seed, at 1024 / 4096 / 16394 tokens of context |
| decode split rule 16 -> 13 | **bounded-ulp** | 1 token in 1 536 greedy; at temperature 0.7 the first divergence is at token 407 |

The temperature-0 result alone (byte-identical) is weaker than it looks, because a
bounded-ulp change can be invisible to an argmax. **The temperature-0.7 result is
the one that establishes exactness**, because a sampler converts a last-ulp logit
difference into a different token at the first near-tie, and 512 draws found none.

### 25.5 Positive control on the same instrument in the same session

A "no measurable difference" is only believable after something moved the same
instrument. `KAPLUMBA_SCALAR_ATTENTION=1` replaces the flash body with the scalar
one, 16 394-token prompt, clean (unprofiled) runs, same session, same binary:

| arm | `prefill_tok_s` | vs shipped |
|---|---|---|
| shipped | **74.975** | — |
| `KAPLUMBA_SCALAR_ATTENTION=1` | **62.552** | **-16.6%** |

The wall clock moves 16.6% on a real attention change and 0.13% on the pass chain,
so the null is a null and not a broken instrument. (§16.3's original control read
20%; 16.6% reproduces it.)

### 25.6 What changed in the tree

* `crates/kernels/src/attention.rs`: `PREFILL_PASS_KEYS = 0`, with the reasoning,
  the measurement, the opt-in and the void 64 MiB argument on the constant's own
  doc comment (so the next reader does not have to find this section).
* `crates/engine/tests/prefill_pass_gate_production_path.rs`: every `#[test]` now
  calls `enable_passes()` first, which sets `KAPLUMBA_PREFILL_PASS_KEYS=8192`
  before anything in that binary reads the `OnceLock`. **Without this, F2's gate
  test would pass for the wrong reason** — production would decline on the key-count
  test rather than the grid test — which is exactly the "a test that calls a
  different entry point than the one claimed" failure this spec has hit twice.
* **New** `crates/engine/tests/prefill_pass_default_off.rs`: four tests pinning the
  new default, with its own positive control (`the_shape_is_eligible_when_passes_are_enabled`
  encodes a chain through the explicit entry point on the same buffers, so the
  "production declines" assertions cannot pass because the device or the kernels
  are broken). It is a **separate binary** because one process can only see one
  value of a `OnceLock`-cached env var, and the two files deliberately cover
  opposite halves of the rule.
* `spec.md` OQ9, §16.4 ¶1 vs ¶3, §16.5 and NEXT.md are made to agree.

### 23.7 `KAPLUMBA_FLASH_BQ=1` — the free lever, and it is not free

§12.2 flagged `FLASH_QUERY_BLOCK_MIN_KEYS = 16_384` as suspect because it was
chosen from §5's byte model, and §12 withdrew that model. It is one env var, so it
was worth a single end-to-end A/B at the context that matters. §22's bench says
`BQ=1` is **4-19% faster** than `BQ=2`; end to end, at a 24 602-token prompt,
`--max-new-tokens 1`:

| arm | `prefill_tok_s` | `phys_footprint` |
|---|---|---|
| default (`BQ=2` above 16 384 keys) | **66.257** | 17.09 GiB |
| `KAPLUMBA_FLASH_BQ=1` forced | **65.675** | 17.08 GiB |

**-0.9%: the 19% does not survive end to end.** With one round each at 24k this is
not a ranking — OQ5 puts ±20% on a single 24k sample — so the honest statement is
**"the two query blocks are within about 1% of each other at 24k, and the
microbench's 19% is the bench's ranking failure showing up one more time."** The
threshold is not worth changing: there is no win on either side of it.

This is the **fifth** independent confirmation that
`attention_arm_throughput_at_prefill_geometry` cannot rank arms: it has now been
wrong about `scalar` vs `mma` vs `flash` (§8.7, §9.4, §11) and about `BQ=1` vs
`BQ=2` (here, against §8.2's E2E `-2.0%`/`-6.2%`).

**A second-order observation worth recording.** Both 24 590-token runs read
**66.3 and 65.7 tok/s** against this spec's recorded **67.10** and the adversary's
**68.40**, and `phys_footprint` reads **17.08-17.09 GiB** against the recorded
**18.35 GiB**. Nothing in this pass changes the model, the weights or the scratch,
so the footprint difference is a *measurement* difference between sessions, not a
code difference — which is worth knowing before anyone treats 18.35 GiB as the
number to beat for AC4.

### 25.7 Confirmation in the shipping binary, and one sample that was discarded

After the default was changed, a `KAPLUMBA_GPU_PROFILE=1` run of the **rebuilt**
`target/release/kaplumba` at the same 16 394 tokens:

| term | passes ON (old default) | one dispatch (old binary) | **default OFF (new binary)** |
|---|---|---|---|
| attention `own_ms` | 64 425.0 | 64 206.0 | **61 744.8** |
| `down_proj` `own_ms` | 35 138.1 | 35 072.5 | 35 114.1 |
| kernel sum | 215 737.7 | 215 297.2 | **212 874.0** |
| **inter-boundary gap** | **15 801.4** | 10 625.6 | **9 678.8** |
| wall span | 237 234.3 | 230 323.7 | 227 056.4 |
| clean `prefill_tok_s` | 74.955 | -- | **76.529** |
| `phys_footprint` | 15.49-15.56 GiB | 15.49-15.56 GiB | **15.53 GiB** |

The gap is now at the **one-dispatch end**, which is the confirmation that the
change reached the shipping binary: the extra command encoding is gone.
`phys_footprint` is unchanged at 15.5 GiB, as §25.2 predicts it must be — the state
buffer is allocated either way.

**This is a consistency check, not the measurement.** The three columns are three
different sessions and the box is faster in the later one (profiled
`prefill_tok_s` 72.2 against 69.1/71.2, and the clean run reads the best prefill
number of the whole pass). The measurement is the **like-for-like pair inside one
session** in §25.2; this table only shows the new binary lands on the same side of
it.

**One sample discarded.** The first re-run of that profile read a gap of
**40 847.7 ms** — four times either arm of the §25.2 pair — while the *clean* run
of the same binary taken minutes earlier printed **76.529 tok/s**, the fastest
prefill of this entire pass. A gap that quadruples while the GPU gets faster is the
host being starved, not the kernel getting slower, so the whole row (gap, wall span,
profiled tok/s) is discarded and **not reported as a result**. The `own_ms` columns
from that run are the tightest cluster of the pass (down 35 070.1, gate 34 792.6,
up 34 791.8) and are consistent with every other run, which is why they are
mentioned here and the gap is not. The log is kept at
`fixer_evidence/f3_default_after/profile.log` rather than deleted, because
"discarded" should be checkable.
