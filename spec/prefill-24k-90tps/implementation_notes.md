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
