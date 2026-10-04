# Plan: prefill ≥120 tok/s at 16k–24k context

## Current architecture

Prefill processes prompt tokens through 60 transformer layers, each with:
- **7 linear projections**: q/k/v/o (attention) + gate/up/down (FFN)
  - q: [B, 5376] → [B, 8192] (32 heads × 256)
  - k: [B, 5376] → [B, 4096 or 2048] (16×256 sliding / 4×512 full)
  - v: same as k
  - o: [B, 8192] → [B, 5376]
  - gate/up: [B, 5376] → [B, 21504]
  - down: [B, 21504] → [B, 5376]
- **Attention**: online-softmax GQA with scalar QK dot product, scalar PV

The two code paths:
- **GEMM** (`linear_gemm_mma`, `linear.metal:10173`): 128×64 MMA tile, 8
  simdgroups, 16 named accumulators, lane-parallel EXL3 dequant, bank-padded
  `bstage`. Achieves 5.31 TFLOPS (35% of 15.18 TFLOPS peak).
- **Attention** (`attention_gqa_body`, `attention.metal:738`): one query per
  threadgroup, GQA-batched (NSG simdgroups per KV head), scalar QK loop with
  hoisted Q float conversion, online softmax via `simd_max`/`simd_sum`, scalar PV.

### Where time goes (estimated at ~78 tok/s, 614 tokens)

| component | ms/token | fraction |
|---|---|---|
| GEMM (all projections) | ~7.5 | ~58% |
| attention | ~0.7 | ~5.4% |
| norms + RoPE + activation + dispatch | ~4.6 | ~36% |

At **24k tokens**, attention grows quadratically:
- GEMM stays ~7.5 ms (independent of context)
- Attention grows to ~15–20 ms (40–60% of total)
- This is why attention MMA is the biggest lever at long context

---

## Optimization vectors, in priority order

### Vector 1: Attention MMA (R2, R5) — biggest lever at 16k+

**Problem:** `attention_gqa_body` computes QK dot products with a serial scalar
loop:
```metal
for (int i = 0; i < HEAD_DIM / 4; ++i) {
    dot += qv.x * kh0.x;
    dot += qv.y * kh0.y;
    dot += qv.z * kh1.x;
    dot += qv.w * kh1.y;
}
```
At head_dim=256, this is 256 FMAs per key, 57% of which are `half→float`
conversions (`spec/engine-performance` §16). At 24k keys × 256 FMAs × 32 heads,
attention is ~200M FMAs per token.

**Approach: MMA-based QK.**
Instead of one scalar dot product per key, batch **8 keys per MMA step**:

1. Load Q fragment: 8×8 half matrix from the query's head_dim slice
   (8 consecutive dimensions)
2. Load K fragment: 8×8 half matrix from 8 keys' same dimension slice
   (transposed: 8 dims × 8 keys)
3. `simdgroup_multiply_accumulate` → 8×8 partial score matrix
4. Accumulate HEAD_DIM/8 = 32 steps → full 8-key score row

**Benefits:**
- Eliminates the per-element `half→float` conversion (MMA does it in hardware)
- 8 keys scored per MMA step instead of 1 key per scalar loop iteration
- The 15.18 TFLOPS tensor pipe replaces the 8.76 TFLOPS SIMT pipe

**Challenges:**
- Only **1 query** per threadgroup in the current prefill dispatch (the Q
  dimension of the MMA tile is 8 but we only have 1 query). Must either:
  - Use a 1×8 score tile (1 query, 8 keys) — wastes 7/8 of the M dimension
  - Tile multiple queries per threadgroup — but this changes the dispatch grid
    and the online softmax state management
  - **Recommended:** tile **Q_TILE=1** (one query), batch 8 keys. The waste
    in the M dimension is acceptable because the reduction is in the K
    (head_dim) dimension, not the output. Each 8×8 MMA produces one partial
    of the 1×8 score vector, and HEAD_DIM/8 = 32 MMAs complete it.
- For **PV**: the current scalar loop multiplies a single weight × value vector.
  MMA-based PV would batch 8 keys × 8 dims, but the weight operand is diagonal
  (one weight per key row). This makes the B fragment a scaled identity, which
  MMA handles but may not be efficient. **Measure before committing to PV MMA.**
- **Register pressure:** 1 score accumulator (8×8 float) + 1 output accumulator
  per dimension block. At HEAD_DIM=256: 32 output blocks × 8×8 = impossible.
  Must tile the dimension and process in chunks.

**Expected gain:** 2–3× attention speedup. At 24k context where attention is
~60% of time, this is a **30–40% overall improvement** (i.e., from ~60 tok/s
to ~80–85 tok/s at 24k).

### Vector 2: GEMM efficiency (R3) — biggest lever at short context

**Problem:** The MMA pipe achieves 5.31 TFLOPS (35% of 15.18 TFLOPS peak).
The staging-stripped diagnostic caps the current tile at 76.7 tok/s. The
remaining headroom is ~11% staging/dequant.

**What has been exhaustively tried and rejected** (do NOT retry):
- 64×64 tile, 32 accumulators, 512 threads, 128q×128n, KCHUNK sweep, staged
  astage, threadgroup staging for trellis, codebook LUT, `extract_bits`, gate/up
  concat, `AGX_RELAX_CDM_CTXSTORE_TIMEOUT`, fp8, wider MMA fragments (8×8 is
  the only hardware instantiation). See `spec/engine-performance` §4.3 and §14.

**Remaining candidates not in the rejected list:**

1. **Multi-query tiling in the GEMM dispatch.**
   The current GEMM dispatches one tile column per threadgroup (128q × 64n).
   At 614 tokens: `ceil(614/128) = 5` tiles along y, `N/64` tiles along x.
   For gate_proj (N=21504): 5 × 336 = 1680 threadgroups. On 20 cores, that's
   84/core — well oversubscribed. But at PREFILL_CHUNK=1024 and N=2048 (k_proj
   for full-attn layers): only 8 × 32 = 256 threadgroups = 12.8/core. The
   small shapes don't fill the GPU.
   
   **Fix:** for small grids, increase PREFILL_CHUNK beyond 1024 for that
   specific dispatch, or use a 256q tile that doubles the y dimension.
   Already partially addressed by raising PREFILL_CHUNK from 256→1024 (+10%).
   Further raising to 2048 was rejected (743 MB scratch, no gain once grid
   is large enough). But shape-aware chunking has not been tried.

2. **Persistent threadgroups across output column tiles.**
   Each threadgroup currently processes one 128×64 output tile and exits.
   A persistent threadgroup could stay resident, processing multiple column
   tiles sequentially, amortizing the dequant/barrier startup. The notes
   estimate a 0.53% ceiling — but that was for a different tile shape and
   the grid was already oversubscribed. At small shapes (k_proj), it could
   matter more.

3. **Weight repack for MMA tile locality.**
   The trellis is stored in tile-major order `[K/16, N/16, tile_words]`. The
   MMA kernel reads NTILES=4 column tiles per K-chunk, so the reads stride
   across N/16 tiles. Repacking so that the 4 column tiles are contiguous
   would improve L2/L1 locality. This doesn't change total bytes (just the
   order), so it's compatible with the "no memory increase" constraint.
   **This has NOT been tried for the prefill MMA kernel.**

4. **Reduced threadgroup memory.**
   Current: `bstage[KCHUNK * BSTRIDE]` = 64 × 72 = 4608 halfs = 9216 B.
   If KCHUNK could be halved to 32 with the same throughput, threadgroup
   memory drops to 4608 B, potentially improving occupancy. KCHUNK=32 was
   tried and was –2%, but that was before the regpipe variant. Re-measure
   with regpipe.

### Vector 3: Context extension (R4) — enabling 16k–24k

**Changes needed:**
1. `HOST_TOKEN_SLOTS`: 8192 → 32768 (or dynamic)
2. KV cache validation at 24k (total ~2.66 GiB, see spec.md R4)
3. `PREFILL_CHUNK` interaction: 24 chunks at 1024, 12 at 2048
4. Attention partial buffer sizing: currently 16 splits. At 24k with 16 splits,
   each split handles 1500 keys. May need more splits for parallelism.
5. Memory budget verification: 11.51 GiB weights + 2.66 GiB KV + 0.37 GiB
   scratch = 14.54 GiB. Fits in 24 GiB with 9.46 GiB for OS + Metal overhead.

### Vector 4: Dispatch & scheduling optimization

1. **Rolling command buffer window.** Currently 8 in flight, 50 dispatches/buffer.
   Validated as near-optimal (`spec/engine-performance` §16). No change needed
   unless 24k context changes the picture.

2. **Fused elementwise kernels.** `gated_gelu` between gate and up projections
   was rejected pre-measurement. At 24k, the intermediate (21504 × tokens)
   grows; re-evaluate if it becomes significant.

3. **RoPE optimization.** Currently evaluates `powf` 2,097,152 times for 32,768
   distinct values (`spec/engine-performance` §7). At 24k with larger query
   counts, this could become measurable. A precomputed RoPE table (in constant
   memory, not scratch) would eliminate it.

---

## Architecture: attention MMA kernel

The most impactful change. Here is the detailed design:

### QK MMA attention body

```
Template: attention_gqa_mma_body<HEAD_DIM, NSG>

Per threadgroup: 1 query row, 1 KV head, NSG query heads (GQA)
Per simdgroup: 1 query head

Inner loop (per 8-key tile):
  1. Load Q slice: 8 consecutive dims from the query → half8x8 fragment
     (row 0 = this query's dims, rows 1-7 = padding or repeated)
  2. Load K slice: 8 keys × 8 dims → half8x8 fragment (transposed)
  3. simdgroup_multiply_accumulate(score_acc, Q_frag, K_frag)
  4. Repeat for HEAD_DIM/8 = 32 steps → score_acc has 1×8 partial scores
  5. Extract 8 scores from score_acc using lane mapping
  6. Online softmax update (same simd_max/simd_sum as current)
  7. PV: load 8 value vectors, multiply by weights (scalar, same as current)
```

**Key design decisions:**
- Q_TILE=1 (one query per threadgroup). Tiling multiple queries would need
  independent softmax states per query and would conflict with the current
  dispatch grid. This is how llama.cpp does it (`fa_common.metal`).
- K_TILE=8 (MMA fragment width). 32 lanes × 8 keys = 256 key-fragments per
  warp, which is the natural batch.
- The PV step stays scalar initially (V1). If attention is still dominant after
  QK MMA, add PV MMA (V2).
- Online softmax is updated every 8 keys instead of every 1 key. This means
  the rescale `acc *= exp(m_old - m_new)` runs 8× less often, which is a
  secondary win.

### Memory layout (unchanged)
- Q: `[queries, q_heads, head_dim]` as device half
- K: `[key_count, kv_heads, head_dim]` as device half
- V: same layout as K
- No new buffers needed.

---

## Requirement → task mapping

| requirement | tasks |
|---|---|
| R1 baseline measurement | T0 |
| R2 attention MMA | T1, T2 |
| R3 GEMM efficiency | T3, T4 |
| R4 context extension | T5 |
| R5 PV MMA | T6 (gated on T2 results) |
| R6 measurement | T7 |
| R7 no regression | every task |

## Risks

1. **Attention MMA register pressure.** If 32 score-accumulator MMAs per key
   tile cause spilling, the whole approach fails. Mitigation: tile the head_dim
   into chunks (e.g., 4 chunks of 64 dims = 8 MMAs per chunk), flush partial
   scores between chunks. OQ1.

2. **GEMM is already near its ceiling.** The staging-stripped diagnostic at
   76.7 tok/s may be the real limit for this tile shape + format combination.
   If R3 produces no gain, the GEMM stays at ~78 tok/s and the overall
   target depends entirely on attention + context extension.

3. **24k context memory pressure.** 14.54 GiB minimum (weights + KV + scratch)
   leaves 9.46 GiB for OS + Metal. If Metal overhead grows with context
   (driver-side graphics regions), the system may page. Mitigation: measure
   with `/usr/bin/time -l` and `sysctl vm.swapusage`.

4. **The nondeterminism bug interacts with long context.** At 24k with
   PREFILL_CHUNK=1024, there are 24 chunk boundaries. Each is a potential
   race. The benchmark must use `KAPLUMBA_MAX_IN_FLIGHT=1` for deterministic
   measurements.
