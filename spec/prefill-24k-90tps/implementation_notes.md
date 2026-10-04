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
