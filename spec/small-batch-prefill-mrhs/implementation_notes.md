# Implementation Notes: Small-Batch High-Bandwidth Prefill (Multi-RHS GEMV)

Append as work proceeds. Factual and dense.

## 0. Hardware & Environment Facts
- **Machine:** Apple M4 Pro (20 GPU cores, 24 GiB unified RAM, 273 GB/s peak bandwidth).
- **OS:** macOS 15+ (Darwin kernel 27.0).
- **Model:** Gemma 4 31B dense (`~/.kaplumba/models/Gxma-repacked`), EXL3 ~3.0 bpw.
- **Weights Residency:** 12.04 GiB wired via `LlamaMlock` + `MTLResidencySet`.
- **Target:** Time-To-First-Token $\le 200\text{ ms}$ on typical interactive chat inputs ($M \le 32$ delta tokens).

## 1. Measured Baseline & Root Causes
- Baseline TTFT measured at $M=14$ tokens: **1312–1315 ms** (`spec/chat-ttft-300ms/` Task 5).
- Breakdown from `KAPLUMBA_GPU_PROFILE=1`:
  - `down_proj`: 336.8 ms
  - `gate_proj`: 275.1 ms
  - `up_proj`: 273.4 ms
  - `o_proj`: 153.4 ms
  - `q_proj`: 141.9 ms
  - `k_proj`: 70.3 ms
  - `v_proj`: 61.5 ms
  - Sum of linear projections across 60 layers: **1312 ms**.
- Root cause: `linear_gemm_mma` uses a fixed $128 \times 64$ tile with threadgroup barriers and low occupancy (1.6 threadgroups/core at $M=14$), dropping memory bandwidth to 9.1 GB/s.
- Target bandwidth with Multi-RHS GEMV: ~185 GB/s (decode GEMV rate), projecting ~65–85 ms across all 60 layers.

## 2. T3: Sabotage verification results (crates/kernels/tests/linear_mrhs.rs)

Test `mrhs_parity_against_single_row_gemv` covers M in {1, 2, 4, 7, 14, 16} × perm_mode {0, 1}.
Four sabotages were run against the compiled pipelines; each failure mode observed:

1. **Row-major epilogue zeroed rows m > 0** (`linear.metal` MRHS): caught —
   `MRHS divergence on perm_mode=0, M=2, row=1: max_diff=30.875 exceeds 1e-2`.
2. **Column-major epilogue zeroed rows m > 0** (`linear_repacked.metal` MRHS): caught —
   `MRHS divergence on perm_mode=1, M=2, row=1: max_diff=38.625 exceeds 1e-2`.
3. **Off-by-one overrun** (`m <= valid_m` in colmajor epilogue, writes one row past the
   last valid query row): initially **NOT caught**. The original test only checked a canary
   in the *finalized* `output` buffer, but the MRHS kernel writes the intermediate `raw`
   buffer — `linear_finalize_rows` never reads row `queries`, so the output canary is
   structurally blind to a raw overrun. The test already seeded a `0xDEAD_BEEF` canary in
   `raw` at row `queries` but never asserted on it. **This is a false-coverage claim in
   waiting** (same class as `spec/engine-performance` Q11). Fixed by asserting the raw
   canary after the run; the same sabotage now fails with
   `MRHS canary overrun in raw buffer at offset 0 for queries=1`.
   Lesson: a canary must live in the buffer the kernel under test actually writes.

Environment note: `LinearKernels::new()` fails if the Metal source fails to compile. An
earlier revision of the test silently `return`ed on that error, so the whole parity test
had been skipping on a box where the MRHS kernels did not compile (threadgroup memory is
not allowed inside a `static inline` template in MSL — it must be passed as a parameter
from the kernel entry point). The test now panics with `LinearKernels::new failed: {e}`
instead of skipping, which is what surfaced the compile error. Never `return`-skip a GPU
parity test.

## 3. T4: routing in `encode_linear` (model.rs)

Routing extracted into a pure function `route_linear(tokens) -> LinearRoute`
(enum `DecodeGemv | SmallBatchMrhs | LargeBatchGemm`), so the table is
unit-testable without a checkpoint on the GPU — the alternative was a test that
silently skips on a box without a Metal device, which is exactly the failure mode
T3 recorded above.

Routing table: `0|1 -> DecodeGemv`, `2..=32 -> SmallBatchMrhs`, `33+ -> LargeBatchGemm`.
`MRHS_MAX_QUERIES = 32` is public and documented; the boundary matches the compiled
m4/m8/m16/m32 register buckets (`mrhs_repacked_pipeline` / `mrhs_exl3_pipeline`
round queries up into the next bucket).

`encode_gemv_mrhs` keeps its own eligibility gate (rate/codebook must be MCG 3 bpw,
`KAPLUMBA_DISABLE_MRHS`), so the engine-side arm only means "prefer MRHS".
`ForwardScratch::raw` is `(max_tokens * inter).max(vocab)` f32, already large enough
for `queries * out_features` at any routed batch; no scratch change was needed.

Sabotage verification (all caught, then reverted):
- boundary lowered to 16: `small_batches_take_mrhs` FAILED (21 tokens),
  `boundaries_match_the_compiled_mrhs_buckets` FAILED (17 -> m32 bucket).
- lower boundary moved to include 2 rows on the decode path:
  `small_batches_take_mrhs` FAILED (2 tokens).

`cargo test --package kaplumba-engine --lib`: 112 passed, 2 pre-existing ignores
(require KAPLUMBA_TEST_MODEL).

## 4. T5 (partial): first end-to-end measurement, and the register-spill wall

Interleaved A/B, `--max-new-tokens 1 --temperature 0 --metrics`, median of 3,
`Hi` (14 template tokens):

| arm | TTFT ms | prefill tok/s |
|---|---:|---:|
| MRHS (default) | **392.4** | 35.7 |
| GEMM (`KAPLUMBA_DISABLE_MRHS=1`) | **1307.8** | 10.7 |

3.33x, but AC4 wants <= 200 ms. `KAPLUMBA_GPU_PROFILE=1` at M=14:
every projection at ~16-19 GB/s of weight bytes while the same tensors stream
at ~185 GB/s in decode.

`examples/mrhs_shape.rs` (new probe, cold 1.6 GB working set, 5 interleaved
rounds, 5376x21504 @3bpw) isolates why -- time per projection, median ms:

    arm                     med ms     GB/s   vs decode
    decode gemv (perm1)     0.4347     99.7    1.000x
    mrhs m1  (m4 bucket)    0.4397     98.6    0.988x
    mrhs m4  (m4 bucket)    0.4468     97.0    0.973x
    mrhs m8  (m8 bucket)    0.6223     69.7    0.699x
    mrhs m16 (m16 bucket)   1.5179     28.6    0.286x
    mrhs m32 (m32 bucket)   6.5805      6.6    0.066x
    (same at perm0: 83.5 / 80.7 / 57.5 / 34.9 / 10.3 GB/s)

Marginal per-row cost GROWS with the bucket: (m8-m4)/4 = 0.044 ms/row,
(m16-m8)/8 = 0.112 ms/row, (m32-m16)/16 = 0.316 ms/row. A bandwidth-limited
kernel would show a *flat* per-row cost near zero and a *constant* time; this is
the signature of register spilling — m32 needs 64 accumulator floats plus the
weight window and spills to private memory, and every reload in the K-loop grows
the per-row cost. This also explains the earlier end-to-end cliff at the bucket
boundary: 14/15/16-token prompts (m16) ran ~392 ms while 19-22/31-token prompts
(m32) ran 1461-1701 ms.

Consequence for the design: accumulate at most 4 rows **per simdgroup** (8
accumulator floats, the measured m4 profile) and cover a 32-row batch by giving
each of the 8 simdgroups a disjoint 4-row slice. Weight tiles stay disjoint per
simdgroup (the decode `tk += SG` stride is unchanged), so DRAM traffic and
memory-level parallelism are unchanged; only the accumulator count per thread
drops. Buckets become R = ceil(M/8) rows per simdgroup, with the existing
same-rows m4 kernel kept for M <= 4 (it uses all 8 simdgroups on <= 4 rows and
measures 97-99 GB/s).

## 5. T6: the three changes that closed most of the gap, and the two dead ends

All measurements interleaved, `examples/mrhs_shape.rs` (5376x21504 @3bpw,
cold 1.6 GB working set, 5 rounds) unless marked end-to-end (Gemma 4 31B,
`--prompt ... --max-new-tokens 1 --temperature 0 --metrics`).

| kernel change | m4 | m8 | m16 | m32 |
|---|---:|---:|---:|---:|
| baseline (T1 design) | 98.6 | 70 | 28.6 | 6.6 GB/s |
| + explicit `fma()` chains | 98.0 | 67.8 | **37.3** | 10.7 |
| + x loads as 2 uint loads/row/tile | 98.4 | 66.3 | 36.7 | (retired) |
| + 17-32 rows as 8-row blocks of m8 | -- | -- | -- | end-to-end 1.0-1.1 s -> 0.41-0.59 s |

End-to-end TTFT (median of 3, interleaved A/B against `KAPLUMBA_DISABLE_MRHS=1`):

| turn tokens | GEMM ms | MRHS ms | speedup |
|---|---:|---:|---:|
| 14 ("Hi") | 1320.2 | **313.2** | 4.2x |
| 20 ("What is 2+2?") | ~1461 | **405.9** | 3.6x |
| 22 | ~1514 | **453.4** | 3.3x |
| 31 | ~1701 | **591.6** | 2.9x |

Multi-turn `--chat` (R4): turn 1 = 14 prefill tokens, TTFT 313 ms, ctx 23;
turn 2 = 21 DELTA tokens (no re-prefill), TTFT 508 ms, ctx 51; turn 3 = 17
delta, TTFT 463 ms, ctx 241; decode 15.03 tok/s session average.

### What each change was
1. **Explicit `fma()` contraction.** The kernels wrote
   `acc += p00*v[0] + p01*v[1] + p02*v[2] + p03*v[3]`, which the Metal
   compiler left as 4 mults + 4 adds. Writing the 4-deep `fma()` chain took
   m16 from 28.6 to 37.3 GB/s and m32 from 6.6 to 10.7. Reassociated the dot
   product (documented for the decode kernels as "not bit-identical in f32,
   bit-identical after fp16 rounding"); the end-to-end greedy A/B below
   confirms text is unchanged.
2. **Vectorised x loads.** `x[r0], x[r0+1]` and `x[r0+8], x[r0+9]` are two
   4-byte-aligned words, so 4 half loads became 2 uint loads
   (`mrhs_load_tile_x`). No change at m4/m8/m16 (confirms they are FMA-bound,
   not LSU-bound), but it cut the m32 register pressure enough to matter
   before m32 was retired anyway.
3. **Row blocks for 17-32 rows** (`MRHS_SINGLE_PASS_MAX_ROWS = 16`). The m32
   kernel needs 64 accumulator floats per thread; measured end-to-end it ran
   ~1.0-1.1 s TTFT per turn. `ceil(rows/8)` passes of the m8 kernel with the
   activation/raw pointers advanced per block run the same rows at 0.41-0.59 s.
   Byte offsets are ELEMENTS x size, because Metal's `set_buffer` offset is in
   bytes; the parity probe catches the mix-up at M=17 row=8.

### Dead ends, recorded so they are not retried
- **Rows-per-simdgroup (R = ceil(M/8) rows per sg).** Two variants tried and
  both are wrong-or-slow:
  (a) Keep the decode `tk += SG` stride (disjoint weight tiles per sg) and give
      each sg its own rows: each row then only receives 1/8 of its K sum and
      the cross-simdgroup `red[g][lid]` reduce mixes DIFFERENT rows. Caught by
      the parity probe (M=7 row=1, max_diff 30.9).
  (b) Full K per simdgroup (correct: `red[sg][lid]` is then the finished
      column, read back by the same simdgroup): correct after two fixes
      (`lid < 16` must be `lane < 16` for the read-back, and the write address
      must use `lane`, not `lid` -- the raw canary caught that one), but SLOW:
      the 8 simdgroups issue identical weight streams and the L1, not DRAM,
      becomes the wall: 22.3 GB/s at m8, flat across every bucket.
- **x-load vectorisation as a speed fix**: no effect at m4-m16; they are
  FMA-throughput-bound (~1.6 TFMA/s, about half of the M4 Pro's fp32 peak),
  not LSU-bound. Kept anyway for the register relief it gave m32.
- **`KAPLUMBA_MAX_IN_FLIGHT` from 4 to 128**: no effect (307-308 ms). The
  rolling encoder already overlaps host and GPU; in-flight depth is not the
  bottleneck.
- **The GPU profile under-reports prefill**: `KAPLUMBA_GPU_PROFILE=1` stamps
  one command buffer per boundary, which SERIALISES host and GPU; the same
  14-token prefill reads 139.5 ms of own time / 156 ms wall span under the
  profiler while the unprofiled TTFT is 313 ms and the host encode alone is
  ~71 ms (2149 dispatches, 33 us/dispatch measured by
  `examples/prefill_probe.rs`, which now prints the split). Do not read the
  profiled span as the prefill's GPU time.
