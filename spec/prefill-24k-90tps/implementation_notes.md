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

> **SUPERSEDED BY §10 — the ablation table below does not reproduce.** Re-run
> against the post-T4f body, `ksteps = 0` gives 966.88 ms against a 958.39 ms
> baseline (here: 433 against 1148), and so does every other arm. Do not quote
> the cost split or anything derived from it. Kept as the record of what produced
> T4f.

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

BQ=2 halves the K/V bytes (each threadgroup reads a key once and reuses it for 2
queries) but also halves the grid, 4096 -> 2048. §8.2 measured it as a 2% loss.
The two effects cancel: the byte win is real, the parallelism loss is real. The
correct form is **BQ=2 with the grid restored** -- split the key range too, so
2048 query-pairs x 2 key-halves = 4096 threadgroups, each reading half the bytes
of the BQ=1 shape. That needs a partial/combine pass (flash-decoding style) and
is the only change identified in this section that attacks the 44% load term
without giving up the parallel term.

### 9.2 What was tried against this split, and the verdicts

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

**Read this before quoting any number in §9.** T4g re-ran the §9 ablation against
the post-T4f body (`1254780`). None of it reproduces.

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

1. **§9's cost split (loads 44%, MMA 52%, softmax 3%) is not reproducible and
   must not be used to justify a change.** It is kept above only as the record of
   what produced T4f's design; every conclusion drawn from it in §9.1, §9.1b and
   §9.2 rests on a measurement that does not survive a re-run.
2. **§9.1b's "BQ=2 + key split + combine pass" stays refuted** — not because the
   byte arithmetic is wrong (it is right) but because it was justified by a split
   that does not reproduce. It is refuted *a fortiori*: the same reasoning
   predicted BQ=2 would help, and BQ=2 measured -2.0% (§8.2) and +0% E2E.
3. **The microbench is a valid throughput instrument and an invalid attribution
   instrument.** It is stable (the `mma` control held 867-894 ms across ~12 runs,
   ±1.5%; `scalar` 1239-1258), linear, and it reproduces T4f's effect. It cannot
   attribute cost, and it cannot rank arms (§8.7, §9.4 — third confirmation).
   **E2E is the only instrument that has ever ranked anything correctly here.**
4. **Open hazard for whoever picks this up:** `pipeline_audit::build(...).ok()`
   turns a Metal compile error into `None`, and the dispatch then falls back to
   the MMA kernel *silently*. The bench prints the arm name it was asked for, not
   the kernel it got. So a `.metal` edit that fails to compile produces a
   plausible, stable, completely wrong number rather than an error. Any future
   ablation must assert `flash_attention_available()` and print
   `pipeline.function_name()` for the arm, not trust the label.

### 10.3 Process failure worth recording

Four ablation rounds were run before the obvious control — *does an empty kernel
take zero time?* — was tried. The first two rounds also used a `rc=$?` after a
pipe, so a failed build read as success. Both are cheap checks; skipping them cost
more than the ablations were worth. The rule this suggests: **an ablation whose
result is "the change did nothing" is only believable after a positive control
has been shown to move the same instrument.**

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
**negative result with a bit-exact, fully tested implementation behind it** — the
implementation is worth keeping (it is the only correct way to express the
mechanism, and the tile-alignment invariant it forced is a real property of the
kernel), but the *default must stay one dispatch*, because "no measurable
difference" plus 64 MiB of scratch plus 2x the dispatch count is a net loss on
anything that is not bandwidth-starved.

The gate ships **on by default** because it is measured neutral and bit-exact;
`KAPLUMBA_PREFILL_PASS_KEYS=0` restores a single dispatch. If a future session
measures a context where it pays, that is the flag to reach for.

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
  129 queries → sliced.
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
