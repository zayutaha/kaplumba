# Consolidated Implementation & Engineering Notes

## 1. Commit History & Architectural Progression

| Commit | Subsystem | Description & Measured Impact |
|---|---|---|
| `f20f301` | Memory | Mapped `embed_tokens` & `lm_head`; resident RAM dropped 15.12 → 11.51 GiB. |
| `8c3ea95` | Decode | GPU greedy sampling & command pipelining; decode reached 6.99 tok/s. |
| `89eacc8` | Analysis | Corrected M4 Pro ALU roofline; proved bandwidth (22.6 tok/s) is the ceiling. |
| `b811675` | Decode | Fixed benchmark calibration; FMA peak measured at 4.35e12 FMA/s. |
| `49814d4` | Decode | Matvec lane-invariant hoisting and loop unrolling; decode reached 7.43 tok/s. |
| `e1d72f2` | Decode | Coalesced decode attention reduction across key splits; decode reached 7.52 tok/s. |
| `22c6325` | Decode | Hoisted shared input transforms across attention heads; decode reached 7.87 tok/s. |
| `63f94e5` | Decode | Added grid-stride matvec kernel (paused unswept to focus on prefill). |
| `9619b81` | Prefill | Created initial prefill 200 tok/s specification. |
| `143937c` | Prefill | Resolved KV cache allocation size (1.41 GiB at 8192 context). |
| `11aef59` | Prefill | Register accumulator in `linear_gemv_tiled`; prefill rose 9.67 → 9.86 tok/s. |
| `500a902` | Prefill | Specialized prefill attention on `HEAD_DIM=256` and `512`; prefill rose to 10.26 tok/s. |
| `724fa4d` | Prefill | Restored generic prefill attention fallback alongside specialized kernels. |
| `12fa4d8` | Prefill | Threadgroup attention (`online_attention_tg2_256/512`) with register accumulators; prefill reached 11.44 tok/s. |
| `ba9263f` | Prefill | Tested 64×64 tile with KCHUNK=64/16; degraded to 8.27 / 2.58 / 6.90 tok/s; reverted cleanly to 11.38 tok/s. |

---

## 2. Experimental Post-Mortems (What Failed and Why)

### 2.1 The Global Atomic Accumulator Myth (+2% actual gain)
- **Hypothesis:** `linear_gemv_tiled` used atomic `output[idx] += val` in global memory across all $k$ chunks. We calculated that this generated 554 GB of DRAM traffic, theorizing it was the sole cause of the 0.56 TFLOPS prefill bottleneck.
- **Implementation:** Thread-local register accumulation walking $k$ strictly in order (`11aef59`).
- **Result:** Prefill improved from 9.67 to only 9.86 tok/s (+2%).
- **Root Cause:** The atomic read-modify-writes were hitting the M4 Pro GPU L1/SLC cache hierarchy, not saturating DRAM bandwidth. The true bottleneck lay elsewhere.

### 2.2 The 64×64 Tile Failure (Theoretical 12.6 TF vs Actual 8.27 tok/s)
- **Hypothesis:** A 16×16 tile has an arithmetic intensity of only 13.4 FLOP/byte (capping at 3.2 TFLOPS). Widening to 64×64 yields 53.8 FLOP/byte, raising the theoretical ceiling to 12.6 TFLOPS.
- **Implementation:** 256 threads per group; each thread owned 16 output queries and 1 column. Trellis staged into threadgroup memory.
- **Results:**
  - 64×64, `KCHUNK=64`: **8.27 tok/s**
  - 64×64, `KCHUNK=16` with dynamic loop: **2.58 tok/s**
  - 64×64, `KCHUNK=16` with `#pragma unroll`: **6.90 tok/s**
  - 16×16 restored: **11.38–11.44 tok/s**
- **Root Causes:**
  1. *Local memory spilling:* Guarding the loop with a runtime `nvalid` trip count prevented unrolling, causing the compiler to spill the 16-register accumulator to local thread memory (the 2.58 tok/s reading). Unrolling with compile-time 16 recovered to 6.90 tok/s.
  2. *Occupancy collapse:* 64×64 with `KCHUNK=64` allocated 20 KB of threadgroup memory per group (8 KB weights, 8 KB inputs, 4 KB trellis), restricting the GPU to 1 threadgroup per core and killing latency hiding.
  3. *Weight decode overhead:* Each thread performed 16 decodes per chunk, each requiring ~10 ALU ops with integer division and modulo (`c / 16`, `c & 15`, MCG multiplication). The ALU time spent decoding weights eclipsed any benefit from the wider tile.

### 2.3 One-Thread-Per-Dimension Attention (8.0 tok/s)
- **Hypothesis:** To avoid `float accum[512]` register pressure in attention, assign one thread per head dimension ($D=256$).
- **Result:** Throughput dropped from 10.26 tok/s to **8.00 tok/s**.
- **Root Cause:** In this layout, every single thread recomputed the full dot-product score across the entire dimension independently. This multiplied arithmetic and key-fetch memory bandwidth by 256×.

---

## 3. Critical Profiling & Ablation Data

### 3.1 Breakdown at 600 Prompt Tokens (Commit `12fa4d8`)
Total prefill time per token: **87.4 ms** (11.44 tok/s).

- **Attention Cost:** **~44.7 ms/token** (~51%)
  - Phase 1 (score dot-products): ~12 ms
  - Phase 1b (serial online-softmax on thread 0): ~25 ms
  - Phase 2 (value accumulation): ~8 ms
- **Linear GEMM Cost:** **~42.7 ms/token** (~49%)
  - Staging & decoding weights: ~28 ms
  - FMA accumulation: ~14 ms

### 3.2 The M4 Pro Hardware Bounds
- Apple M4 Pro (20 GPU cores) @ ~1.45 GHz:
  - Peak SIMT fp32 FMA: **4.38e12 FMA/s = 8.76 TFLOPS**.
  - Gemma 4 31B GEMM: 2 × 28.74e9 = **5.75e10 FLOPs/token**.
  - **Absolute scalar fp32 ceiling:** $8.76 \times 10^{12} / 5.75 \times 10^{10} = \mathbf{152.3 \text{ tok/s}}$ (GEMM only, zero time for attention).
  - Apple Metal `simdgroup_matrix` (tensor cores): Capable of 30–50+ TFLOPS, but computes sums in SIMD hardware order rather than strict sequential order.

---

## 4. Key Developer Gotchas & Verification Commands

1. **Metal Shader Recompilation Trap:**
   - Metal shaders are embedded into the CLI binary via `include_str!`.
   - Running `cargo build --release --workspace --examples` does **not** update `kaplumba-cli`.
   - Always compile with:
     ```bash
     cargo build --release -p kaplumba-cli
     ```
2. **End-to-End Prefill Measurement:**
   ```bash
   M="/Users/zayaantaha/.kaplumba/models/Gxma/Gxma"
   PP=$(python3 -c "print('Explain the theory of relativity. '*100)")
   ./target/release/kaplumba --model "$M" --prompt "$PP" --max-new-tokens 8 --temperature 0 --metrics
   ```
3. **Reference Output Integrity:**
   ```bash
   for q in "What is the capital of France?" "Count from one to five." "Name three primary colors." "Hi" "What is 17 times 23?"; do
       ./target/release/kaplumba --model "$M" --prompt "$q" --max-new-tokens 24 --temperature 0
   done
   ```
   *Expected golden answers:*
   - Paris
   - 1, 2, 3, 4, 5
   - red, blue, and yellow
   - Hello! How can I help you today?
   - 17 times 23 is 391

---

## 5. Session 2026-09-27 — T0 landed; fractional prefill gap found

### 5.1 T0: the encoder bug was in `attention.rs`, not the test file

`plan.md` §4.1 and `tasks.md` T0 both name
`crates/engine/tests/attention_metal.rs`. That file contains **no Metal encoder** —
it calls `AttentionKernels::online_attention`. The real fault was in
`crates/kernels/src/attention.rs`, and it was two bugs:

1. `online_attention` opened the encoder and *then* resolved the pipeline with
   `encoder.set_compute_pipeline_state(match head_dim { 256 => .., 512 => ..,
   _ => return Err(..) })`. Any other `head_dim` returned with a live encoder
   open → `Command encoder released without endEncoding`.
2. The generic `online_attention` kernel was compiled and loaded into
   `online_attention_pipeline` but **never dispatched**. The 256/512
   specialisation (commits `500a902`, `724fa4d`) dropped the documented fallback
   for any other `head_dim` up to `MAX_HEAD_DIM`.

Fix: resolve the pipeline before creating the encoder; fall back to
`online_attention_pipeline` instead of erroring.

Verification: `cargo test --release -p kaplumba-engine --test attention_metal`
→ **4 passed; 0 failed** (previously failing on `head_dim` 16 and 32).

### 5.2 Prefill GEMM had no fractional path (fixed; model unaffected)

`cargo test --release --workspace` (ignored tests excluded) is green except:

- `linear_metal::metal_prefill_tiles_match_cpu_across_chunks_and_variants`
- `linear_metal::metal_prefill_tiles_match_non_square_shapes`

Both fail **only on `bits=1.5`** (fractional). Integer rates 1..=8 and all three
codebooks pass. First mismatch: `bits=1.5 index=0 expected=0 actual=-154.125`.

Root cause: `LinearKernels::gemm_chunk_with_stats` (`crates/kernels/src/linear.rs:566`)
unconditionally dispatches `gemv_tiled_pipeline`. That kernel decodes through
`exl3_code_at_tg` (`linear.metal:2166`) and `exl3_decode8` — both **integer-only**
(`ring = 256*ka`, eight codes per lane). The only fractional-capable decoder is
`packed_state` → `fractional_window` (`linear.metal:250`), reached solely from the
non-tiled `linear_gemv` (one thread per output, single query). So prefill
silently returns wrong numbers for fractional rates rather than erroring.

Not a regression from this session (diff touched only `attention.rs`).

**Fixed.** `linear.metal` gained `fractional_window_tg`, a threadgroup-resident
 twin of the global `fractional_window` used by the non-tiled `linear_gemv`, and
 `linear_gemv_tiled` now derives the correct tile word count and decode path for
 a fractional rate:

```c
const bool fractional = params.rate >= 16u;              // encoded(frac) = 16 + bits
const uint bits = params.rate - 16u;
const uint word_count = fractional ? (8u * bits + 4u)   // packed_words()/2
                                   : (256u * params.rate / 32u);
state = fractional ? fractional_window_tg(tw + t * word_count, word_count,
                                          lr * 16u + lc, bits)
                   : exl3_code_at_tg(tw + t * word_count, word_count, lr, lc, params.rate);
```

`word_count` was `256 * rate / 32` unconditionally, which for a fractional rate
gave `128 + 8*bits` words instead of `8*bits + 4` — the tile stride and the ring
wrap were both wrong. `index = lr * 16 + lc` is the plain row-major identity that
`decode_tile` uses (`tile[row * 16 + column]`), so `perm_mode = 0` is correct and
no permutation is needed. Fractional only admits bits 1..3 (`BitRate::from_rate`),
so the largest fractional `word_count` is 28 and `threadgroup uint tw[TILES * 64]`
still suffices. No new buffer, no scratch: the fix decodes from the same staged
threadgroup words the integer path already used, so working set is unchanged.

Verification: `cargo test --release -p kaplumba-engine --test linear_metal`
→ **6 passed; 0 failed** (was 4 passed, 2 failed).

**Why it does not affect the 150 tok/s target:** the checkpoint is integer-only.
`quantization_config.json` for `Gxma/Gxma` carries `bits: 3.0` and a `head_bits`
for `lm_head`; there are no 1.5/2.5/3.5 tensors. `model.rs` constructs
`Exl3Rate::Integer(3)` / `Integer(6)` exclusively.

### 5.3 Short tests that cover decode and prefill quality

The ignored tests are the model-gated ones; the cheapest that exercises **both**
prefill and decode on the real checkpoint is:

```bash
export KAPLUMBA_TEST_MODEL=/Users/zayaantaha/.kaplumba/models/Gxma/Gxma
cargo test --release -p kaplumba-engine --test real_forward_buffer \
  -- --ignored --exact real_model_buffer_forward_prefill_then_decode
```

Measured: **1 passed in 10.07 s** (14-token prefill + 4 decode steps; model load
included). Note it asserts shapes/sizes and drives argmax, not a golden token —
strict text equality lives in `output_identity.rs` (five greedy decodes).

Fast, model-free correctness (all green):
`cargo test --release --workspace` → 29+4+1+5+6+2+3+1+6 tests pass across
`linear_metal` (minus §5.2), `attention_metal`, `attention_shapes`,
`elements_metal`, `exl3_format`, `exl3_metal`, `forward_buffer_parity`,
`kv_cache`, `metrics`, `tokenizer_sampler`.

---

## 6. Session 2026-09-27 (cont.) — scratch made context-independent

### 6.1 Decision record

- **Bit-exact is the hard constraint.** 120-150 tok/s was requested and then
  withdrawn in favour of bit-exactness; it is not achievable bit-exactly (see
  §6.4). Do not plan to 120-150 without an explicit decision to use tensor cores.
- **Q3 (attention strategy) is now inconsistent with Q1.** Q3 was answered "port
  the oMLX multi-simdgroup merge", but that merge is the *relaxed-float* option
  (`plan.md` §2.2) and Q1 was answered "stay bit-exact". Recorded as an open
  question in `spec.md` §5; not silently resolved. The bit-exact option is the
  prefix-max scan (`plan.md` §2.1).

### 6.2 Chunked prefill: scratch no longer scales with the prompt

`forward_buffer_into` ran the entire prompt through all 60 layers in one pass and
called `ensure_scratch(tokens)`, sizing `ForwardScratch` by prompt length. At 8192
tokens that is ~2.5 GiB of activation scratch — precisely the
context-length-dependent scratch AGENTS.md §8 forbids.

Fix: `PREFILL_CHUNK` (in `model.rs`) plus a wrapper that splits
`forward_buffer_into` into chunks over a new private `forward_chunk`.

Why chunking is semantically inert, checked before doing it:
- positions are absolute and `positions[0] == gpu_next_position()` is re-checked
  per chunk;
- `GpuKvCache::advance` runs during each chunk's encode (`model.rs:766`), so
  `next_position` is correct for the following chunk and it is host-side;
- `encode_lm_head` normalises exactly 1 row, so only the last chunk's logits are
  read back;
- all chunks share one `MetalEncoder`, so it stays a single command buffer.

### 6.3 Choosing the chunk size (measured, not estimated)

A 256-row chunk was tried first and failed the budget:

```
scratch is 160707332 bytes, over the 100 MB budget
```

The hand estimate had been ~88 MB, off by 2x. Recomputing exactly from
`ForwardScratch::new` reproduced **160,707,332** to the byte, which located the
surprise:

| buffer | 256 rows | 64 rows | scales with chunk |
|---|---|---|---|
| `raw` (f32, `chunk*inter`) | 22.02 MB | 5.51 MB | yes |
| `transformed` (u16, `chunk*inter`) | 11.01 MB | 2.75 MB | yes |
| `gate`+`up`+`activated` (u16, `chunk*inter`) | 33.03 MB | 8.26 MB | yes |
| `q` (u16, `chunk*q_max`, q_max=16384) | 8.39 MB | 2.10 MB | yes |
| `k`+`v` (u16, `chunk*kv_max`, kv_max=4096) | 4.19 MB | 1.05 MB | yes |
| `hidden_a/b/normed/attended/proj` | 13.76 MB | 3.44 MB | yes |
| **`attn_scores` (f32, `q_max*1024`)** | **67.11 MB** | **67.11 MB** | **no** |
| `logits`+`capped`+fixed | 1.20 MB | 1.20 MB | no |
| total | 160.71 MB | **91.40 MB** | |

`PREFILL_CHUNK = 64` gives **91,401,476 bytes** measured, under the 100 MiB cap.
At 128 rows it would still be ~114 MB, so 64 is the largest power of two that
fits while `attn_scores` stays oversized.

**Finding worth 67 MB: `attn_scores` is ~512x larger than needed.** It is
allocated `f32_elements(q_max, 1024)` where `q_max = q_heads * head_dim` (16384),
i.e. `q_heads * head_dim * 1024`, but it is only passed to
`encode_online_attention_split` on the `tokens == 1` decode path, whose contract
is `q_heads * key_count` floats (32 * 8192 = 262,144 = 1.05 MB). The over-allocation
is why it has never overflowed. Right-sizing it (task T0d) drops total scratch to
~24 MB at a 256-row chunk.

### 6.4 Why 120-150 tok/s is not reachable bit-exactly

- M4 Pro SIMT fp32 peak (measured, `examples/alu_peak.rs`): 8.76 TFLOPS.
- Gemma 4 31B GEMM: 5.75e10 FLOP/token.
- Ceiling **with attention at 0.00 ms**: 8.76e12 / 5.75e10 = **152.3 tok/s**.

150 tok/s leaves 0.1 ms/token for all attention; 120 tok/s leaves 1.76 ms at 100%
ALU efficiency, against a measured ~7.5% today. Both are physically impossible
without tensor cores, which change the last float bits.

### 6.5 Commands for the new test

```bash
export KAPLUMBA_TEST_MODEL=/Users/zayaantaha/.kaplumba/models/Gxma/Gxma
cargo test --release -p kaplumba-engine --test real_forward_buffer \
  -- --ignored --exact real_model_prefill_scratch_is_independent_of_prompt_length --nocapture
# ~35 s; prints weights=11001026276 kv=881029120 scratch=91401476
```

---

## 7. Tensor-core peak, measured (T3 done)

### 7.1 T3: the probe bug was the leading dimension

`metal_probe.rs` called `simdgroup_load(ta, a, 0)`. The third argument is the
**leading dimension — elements per row**, not a row stride, element stride or
count. Passing `0` made every row load from the same base address, which is
exactly why the identity test matched 8 of 64 elements. With `LD = 8`:

```
A=identity, B known: 64/64 elements correct -> PASS
```

### 7.2 The spec's "30-50+ TFLOPS" estimate is wrong

`examples/mma_peak.rs` (new) measures pure ALU tensor throughput: 2048 simdgroups
running dependent/independent 8x8x8 MMAs with no memory traffic.

| chains per simdgroup | measured |
|---|---|
| 1 | 5.75 TFLOPS |
| 8 | 7.65 TFLOPS |
| **16** | **15.64 TFLOPS** |
| 32 via `simdgroup_half8x8 acc[32]` | 0.74 TFLOPS |

Two implementation facts worth keeping:

1. **One accumulator measures latency, not throughput.** 1 -> 16 chains is a
   2.7x difference. Any future MMA benchmark using a single accumulator will
   under-report by that factor.
2. **An array of simdgroup matrices spills.** `simdgroup_half8x8 acc[32]` with an
   unrolled index loop fell to 0.74 TFLOPS; the accumulators went to local memory.
   Named scalar locals only. The committed example therefore uses 16 named
   accumulators.

### 7.3 What this means for the throughput target

| resource | peak | GEMM-only ceiling (attention = 0 ms) |
|---|---|---|
| SIMT fp32 | 8.76 TFLOPS | 152.3 tok/s |
| tensor fp16 (`simdgroup_multiply_accumulate`) | **15.64 TFLOPS** | **272 tok/s** |

Real prefill is `GEMM + attention`. At a realistic 50% GEMM efficiency the tensor
path gives 136 tok/s with **attention still costing nothing**; attention is ~51%
of wall clock today. So:

- **150 tok/s needs >86% GEMM efficiency plus a ~3 ms attention budget.** Not
  reachable — at 50% GEMM efficiency the GEMM alone (7.4 ms) already exceeds the
  6.67 ms/token budget.
- **120 tok/s needs ~60-70% GEMM efficiency and attention down to a few ms from
  44.7 ms.** The edge of plausible, and only on the relaxed-precision path.
- The tensor route is only ~1.8x the SIMT ceiling, not the 4-6x the spec assumed.
  It is not a magic bullet, and it costs the last bits of every float.

### 7.4 Reproduce

```bash
cargo run --release -p kaplumba-engine --example metal_probe   # 64/64 PASS
cargo run --release -p kaplumba-engine --example mma_peak      # 15.64 TFLOPS
```

---

## 8. Parallel prefill softmax + attn_scores fix; and a measurement warning

### 8.1 The parallel softmax is in — and it changed nothing at 614 tokens

`online_attention_parallel_body<HEAD_DIM>` (`attention.metal`) replaces the serial
phase: the old kernel ran `if (tid == 0u) { for t in 0..n { ... exp() ... } }`, one
serial `exp()` per key while 255 threads waited. The new body scores in parallel,
finds the chunk max with a simdgroup-shuffle + 8-way tree, advances the running max
**once per 256-key chunk** (`alpha = exp(m - m')`), evaluates every weight in
parallel, and reduces the sum the same way. 5 barriers/chunk; 1 KB `sbuf` + 32 B
`red`. Both `online_attention_tg2_256/512` now call it; `online_attention_tg2_body`
is left in the tree as dead code (the A/B harness for the next patch).

A/B at 614 prompt tokens, everything else identical:

| kernel | prefill |
|---|---|
| old serial softmax | 10.394 tok/s |
| new parallel softmax | 10.373 tok/s |

**Within noise — zero measurable gain.** That directly contradicts notes §3.1,
which attributes ~25 ms/token to "phase 1b (serial online-softmax on thread 0)".
Either that attribution was wrong, or `n` is too small at 614 tokens (3 chunks of
<=256 serial iterations each) for it to matter. It should matter at 8192 context,
where the serial walk is 32x longer — **that is the experiment that would settle
it, and it has not been run.**

### 8.2 `attn_scores` right-sized (T0d done): 67.11 MB -> 1.05 MB

`ForwardScratch::new` now takes an explicit `attn_scores` element count; both call
sites compute `max(q_heads * gpu_cache.capacity())` in the loop that already walks
the layers. `encode_online_attention_split` now asserts
`scores.length() >= q_heads * key_count * 4` — the guard that was missing precisely
because the 512x over-allocation made an undersized buffer impossible to notice.

| state | scratch @ 256 rows | `PREFILL_CHUNK` |
|---|---|---|
| before | 160,707,332 B | **64** (forced) |
| after | **93,729,540 B** (89.4 MiB) | **256** (restored) |

### 8.3 WARNING: the 11.44 tok/s baseline is not reproducible, and the phase profiler lies

Two measurement problems that invalidate naive before/after comparisons:

1. **Decode reads 7.08-7.19 tok/s today against 7.87 in notes §3.3**, and decode
   code was not touched by anything in this section. A ~10% machine-level shift
   (thermal, background load, differing resident set) is present. Any A/B must be
   run back-to-back in the same session, not against the recorded numbers.
2. **`KAPLUMBA_PROFILE_PHASES=1` is not trustworthy for prefill.** It reported the
   three prefill chunks as `3.00 ms + 1.65 ms + 7.85 ms` = 12.5 ms total, while
   the same run reported `ttft_ms=58707`. A 4700x disagreement. Do not use the
   PHASE lines to locate prefill cost until that is fixed.

### 8.4 Reproduce

```bash
M=/Users/zayaantaha/.kaplumba/models/Gxma/Gxma
PP=$(python3 -c "print('Explain the theory of relativity. '*100)")
./target/release/kaplumba --model "$M" --prompt "$PP" --max-new-tokens 4 --temperature 0 --metrics
# 614 prompt tokens; prefill ~10.4 tok/s, decode ~7.1 tok/s
```
