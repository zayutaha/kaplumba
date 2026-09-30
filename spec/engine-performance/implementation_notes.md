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

## 2026-09-28 — reference mining (omlx + llama.cpp) and the first three fixes

Ten research agents read `/tmp/kaplumba-refs/omlx` (Apache-2.0) and
`/tmp/kaplumba-refs/llama.cpp` (MIT) plus our own prefill path. Everything
below is cited; nothing is inferred.

### Fixes landed in this commit

| change | why | effect |
|---|---|---|
| removed `linear_zero_raw` (410 dispatches/chunk) | `linear_gemv_tiled` ends in a plain store (`linear.metal`), not an accumulate, and its grid covers every (query,output) pair. The zeroing outlived its reason when 11aef59 made the accumulator a register. 4.77 GB of dead writes per 256-token chunk. | 10.55 → 10.58 tok/s |
| rolling command buffers: 50 dispatches/buffer, 8 in flight | MLX uses `max_ops_per_buffer_ = 50` on this arch class (`mlx/backend/metal/device.cpp`) and `MAX_ACTIVE_TASKS = 10` (`mlx/transforms.cpp`); omlx ships 16 (`omlx/cluster/deployment.py`). Our one-buffer prefill carried ~8000 dispatches. | 10.58 → 10.98 tok/s |
| commit the prefill in the greedy path | `forward_buffer_into` encoded the prefill and the encoder was **dropped un-ended** — never submitted. `prefill_tokens_per_second` therefore measured host encode, not the GPU. | measurement now honest |
| rms_norm threshold 32 → 8192 rows | prefill (256 rows) was taking the one-thread-per-row kernel, handing a 20-core GPU 256 threads. `examples/norm_sweep.rs`: 0.7595 → 0.0253 ms, 30x. | (in 4d55bbf) |

Verified: all five reference prompts byte-correct; `cargo test --release
-p kaplumba-engine` all green. Prefill 10.98 tok/s at 614 tokens.

### The finding that invalidates my own earlier diagnosis

I had spent this session claiming prefill attention's "phase 1b runs the
online-softmax recurrence on one thread while 255 wait" was ~half the cost,
and had queued "parallelise it" as the top item. **That premise is wrong for
the current kernel.** `online_attention_tg2_body` (`attention.metal:539-655`)
is dead code with no entry point; the live kernel is
`online_attention_tg2_256/512` → `online_attention_parallel_body`, which
already reduces the chunk max and chunk sum in parallel via `simd_shuffle_xor`
plus an 8-way tree, and advances `running_max` once per 256-key chunk on all
256 threads. The recorded A/B was 10.394 (serial) vs 10.373 (parallel) —
within noise. Lesson: I optimised the "phase 1b" bottleneck for a code path
that does not run.

### What the references actually say, ranked

**1. The GEMM re-reads the whole weight matrix once per 16-query block.**
`linear_gemv_tiled` gives every thread exactly one accumulator, so each weight
tile is fetched from DRAM and dequantised again for each 16-query block:
~16x weight-traffic amplification, ~204 GB per 256-token chunk. **This is the
mechanical reason the 64x64 tile experiment failed** — it widened N while
leaving Q=16, so the amortisation factor did not move. The lever is register
blocking on Q, not a bigger output tile.
omlx's answer (`bonsai/csrc/quantized.h:1595`): 32x32 tile, 128 threads, one
decode feeding 32 MACs; and `qwen35_verify_qmm.py:403-413` is the shape to
copy — 8 simdgroups each own one K chunk and dequantise into a *private*
threadgroup buffer, so **there is no barrier in the K loop at all** and the
partials reduce once at the end.

**2. The correct simdgroup fragment layout** — the missing piece that made our
MMA kernel emit garbage. Five independent implementations in omlx agree
(`glm_moe_dsa/.../mma_dsa_indexer_score.h:68-75`, `bonsai/.../quantized.h:1636`,
`deepseek_v41/kernels.py:97`):

    qid = lane >> 2
    fm  = (qid & 4) + ((lane >> 1) & 3)          // M offset, 0..7
    fn  = ((qid & 2) << 1) + ((lane & 1) << 1)   // N offset, in {0,2,4,6}
    // element[0] at (fm, fn), element[1] at (fm, fn+1)

The trap: **A's pair is (row m, adjacent k) and B's pair is (fixed k, adjacent
n)** — the shared index is the row in both, but it means a different axis. Copy
the NVIDIA m8n8 assumption and you are off by a transpose *and* an axis swap.
Cross-lane reductions use masks **2, 4, 16** (`kernels.py:263-268`), not 32/64.
Accumulator is always `simdgroup_matrix<float,8,8>`; operands half; one
multiply covers 8 k. omlx stages dequantised weights as **[k][n] (n
contiguous)** so each lane's B fragment is one `vec2` load
(`mma_dsa_indexer_score.h:18-23`).

**3. llama.cpp has the attention fix I was groping for**, and it is a
structural transpose, not a tuning knob (`kernels/fa_common.metal:411-454`):
lane index -> KV index, simdgroup index -> query row, `M`/`S` as per-lane
replicated registers, and the recurrence is two shuffles:

    M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));
    S[jj] = S[jj]*ms + simd_sum(vs2[0] + vs2[1]);

Result is uniform across 32 lanes, so the O rescale `so4[...] *= ms` is a
disjoint per-lane write with no barrier (`:447`). The O accumulator *must* live
in threadgroup memory (`so`, `:98`) because the simdgroup holding `ms` does not
own that row's O columns. 3 threadgroup barriers per 64-key tile, none of them
a softmax reduction. `Q=8` queries, `C=64` keys, `nsg = 4` for dk=256 / 8 for
dk=512.
**Do not port the mask** — llama.cpp materialises a 128 MiB f16 mask at L=8192
(`src/llama-graph.cpp:977`), which violates our §8 fixed-scratch rule. Derive
the SWA bound arithmetically from `(q_abs, k_abs, window)`; it is two scalar
comparisons on values already in registers.
Also: llama.cpp is structurally *weak* at GQA — one threadgroup per query head
re-reads the same K/V (`fa_common.metal:131-132`). We re-read 512x. Batching
q-heads per KV head is a place we can beat them.

**4. Command buffers — llama.cpp's own workaround is one line.**
`ggml-metal.cpp:1025-1029` sets `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` as a
documented workaround for giant-command-buffer interactivity stalls (issue
#20141). Cheap to try. Note also omlx measured that **blocking waits beat
completion-handler pipelining by 22%** (`docs/experimental/qwen35_ane_prefill.md:372`),
so the instinct to make submission async is wrong.
llama.cpp's whole design is the counter-example: 1 global queue, `n_cb=1`
(2 buffers per graph), and `n_ubatch=512` (`common/common.h:453`) so no buffer
is ever large. Our equivalent of `n_ubatch` is our prefill chunk size.

**5. Weight decode: kill the divide and the modulo.** omlx has zero in any hot
path. For small alphabets it uses a `select()` chain over hoisted register
constants, explicitly *rejecting* a dynamically-indexed table:
`quantized.h:612-613` "no dynamic array index which would spill to device
memory on Apple GPU". So the plan to add a 256-entry float LUT for our EXL3
codebook is probably wrong on this hardware; a 2-3 `select()` mux is likely
faster. Where a LUT *is* right, omlx keeps it in `constant` address space and
only for radix conversion, not for the float value (`quantized.h:1301-1311`).
Also `dq_a8_decode.h:86-91`: `metal::extract_bits` is one instruction where a
shift+mask is two.

**6. Threadgroup memory cliff, measured.** omlx: 27 KB collapses occupancy to
1 threadgroup/core and costs **-18%** (`dsa_indexer.metal:48-50`); their own
t5 GEMM sits at 16.9 KB. Our 64x64 attempt used 20 KB and lost — consistent.
llama.cpp's entire prefill GEMM budget is 6 KB (`ggml-metal-device.cpp:844-851`).

**7. Dead work in our prefill, still to remove** (from the code audit, all
cited in the audit agent's report): 120 dead `linear_input_transform`
dispatches (prefill overwrites the buffer with `linear_input_rows`); the
262144-wide LM head + softcap + greedy pick encoded for *every* chunk
(~2.1 GB dead per 614-token prefill); `key_count` returning `capacity` (1024)
for every sliding layer so PV loads masked keys — 87.5% of PV work wasted at
256 tokens; `q_norm` at exactly 8192 rows falling off the coalesced path
because the guard is `rows < 8192`; `float values[128]` in all four FWHT
kernels spilling to local memory; `rope` evaluating `powf` 2,097,152 times for
32,768 distinct values.

### Still unsolved

Prefill is 10.98 tok/s, not 100. The 614-token prefill is ~56 s wall of which
the GPU is busy only a few seconds, so **the host still dominates and I still
cannot account for it**: encode-only was 58.06 s and encode+commit+wait 57.86 s,
which says the cost is in encoding, yet 8000 dispatches cannot cost 7 ms each
on the CPU. Two rounds of in-process instrumentation (per-dispatch timers, then
`new_command_buffer` counters) produced no output at all, despite the format
string being present in the binary — the same failure as the earlier session's
`ApiTimer`. That instrumentation approach is untrustworthy here and needs a
different method (external `sample`/DTrace, or a timing harness that cannot be
defeated by the same bug).

---

# HANDOVER — 2026-09-28: prefill 9.67 → 68.8 tok/s, and everything that was wrong on the way

Commits in order: `ff46bdd`, `4d55bbf`, `60ae014`, `d678a79`, `c056b48`, `cb0cde5`, `1dd4da9`.
State: **68.8 tok/s** at 614 prompt tokens (was 9.67). 20 suites / 81 tests green,
5/5 reference prompts byte-identical.

## 9. Dead work removed (three separate faults, all in the prefill dispatch list)

| removed | was | why it was dead | cost it was burning |
|---|---|---|---|
| `linear_zero_raw` dispatch | 410 dispatches / 256-token chunk | `linear_gemv_tiled` ends in a plain **store**, not an accumulate, and its grid covers every (query, output) pair, so every element of `raw` is written before `finalize` reads it. The justifying comment stopped being true in `11aef59`, when the accumulator became a register. | **4.77 GB** of writes per chunk |
| `linear_input_transform` in prefill | 120 dispatches / chunk | `encode_gemm` first runs `linear_input_rows`, which rewrites all `queries × in_features` of `transformed` from the input and its signs. `PrefillEncodeRequest` has **no** `pretransformed` field precisely because prefill never reuses it. | 2 dependent tiny dispatches per layer per chunk |
| LM head on every chunk | 1.06 GB of weights read per invocation | only the last chunk's logits survive | **2.1 GB dead per 614-token prefill** |

- `linear_zero_raw` dispatch removed from `LinearKernels::encode_gemm`
  (`crates/kernels/src/linear.rs:913`; the old call site was `:999`). The kernel
  (`crates/kernels/src/linear.metal:2616`) and its `zero_raw_pipeline`
  (`crates/kernels/src/linear.rs:147`, loaded at `:237-241`) are still there and
  are still needed by the scalar tiled path — only the prefill dispatch went.
- `linear_input_transform` now gated on `let decode = tokens == 1;`
  `crates/engine/src/model.rs:1001`, applied at `model.rs:1022` and
  `model.rs:1265`. **Decode genuinely benefits** (7 transforms per layer → 4) and
  keeps them, so this is not a "prefill was slow" fix.
- `run_head: bool` added to the chunk entry point
  (`crates/engine/src/model.rs:2479`), documented at `:2459`; the whole head
  block is skipped at `model.rs:2650-2656`, returning `Ok(Vec::new())`. The
  non-final `greedy_pick` is **not** load-bearing: `take_greedy_tokens` runs
  after decode step 0 has rewritten slot 0, and the next chunk rewrites
  `token_ids` from the host.
- Net: **−374 dispatches per 614-token prefill, −5.5%**. head/gemv 35.4 → 20.6 ms,
  greedy_pick 14.3 → 4.1 ms.

**Also in this window, and load-bearing:** the greedy path **never committed its
prefill**. `forward_buffer_into` encoded it and the encoder was dropped
un-ended, so the work was never submitted, the KV cache was never populated, and
`prefill_tokens_per_second` was reporting *host encode time*, not GPU time. Every
prefill number in notes §1–§8 before `ff46bdd` is suspect for this reason.
Command buffers now roll at 50 dispatches with 8 in flight (MLX uses
`max_ops_per_buffer_ = 50` for arch `s`, `mlx/backend/metal/device.cpp`;
`MAX_ACTIVE_TASKS = 10`, `mlx/transforms.cpp`; omlx ships 16,
`omlx/cluster/deployment.py`).

## 10. The profiler was lying, and why that wasted three paths

`KAPLUMBA_PROFILE_PHASES=1` had three independent defects:

1. `OpTimer::lap` had **zero call sites**.
2. `layer_ms` was **never marked**, so the PHASE table omitted every layer —
   hence `layers=0.00ms`.
3. It measured **host encode time**, not GPU time.

It reported `layers=0.00ms` and 12.5 ms total against a 58.7 s wall — a 4700x
disagreement. Notes §3.1 and §8.1's cost attribution derive from it and are
**invalid**.

Replaced by `KAPLUMBA_GPU_PROFILE=1` (`crates/engine/src/model.rs:2413`), which
uses `MTLCommandBuffer` `GPUStartTime`/`GPUEndTime`. Attribution agrees with wall
clock to **91.8%** (17468 ms instrumented + 1168 ms inter-boundary gap vs
`ttft_ms` 20291 ms). Report rendered by `GpuProfile::render`,
`crates/engine/src/model.rs:667`.

- **The Objective-C selectors are CAPITALISED.** `GPUStartTime` / `GPUEndTime`.
  The Swift lowercase spelling (`gpuStartTime`) **aborts**. Documented at
  `crates/kernels/src/encoder.rs:94-100`; bindings at `encoder.rs:131-141`.
- Both values are **undefined until the command buffer has completed** — Metal
  returns garbage, not zero. Read only via `take_gpu_ms()`
  (`encoder.rs:605`) after `take_gpu_stamps` has waited.
- `commit_stamped(label)` at `encoder.rs:555`. Row-level mechanics tested in
  `crates/engine/tests/gpu_profile_table.rs`; the encoder's own regression test
  is `encoder.rs:814`.
- The profiler prints an **overhead floor** (empty commit / one trivial
  dispatch) and warns when `stamps_ingested != boundaries`. Both exist because
  a profiler that silently drops work is worse than no profiler.
- Profiling costs **1.15 s of wall** (32.11 → 30.31 tok/s) and reports 0.99 s of
  it. **Never A/B a production change against a profiled run.**

**The "inter-boundary gap" of 1168 ms was an artefact of this**, not hidden GPU
work. Unprofiled, `tick()` rolls at 50 dispatches with 8 in flight: ~45 rolls
per chunk, not 906.

## 11. I was wrong twice. Both recorded.

**(a) The attention "phase 1b" bottleneck does not exist.**
I claimed the serial online-softmax on one thread was ~half of prefill cost and
queued "parallelise it" as the top item. `online_attention_tg2_body`
(`crates/kernels/src/attention.metal:539-655`) is **dead code with no entry
point**. The live kernel already reduces the chunk max and chunk sum in parallel
via `simd_shuffle_xor` plus an 8-way tree. The recorded A/B was 10.394 (serial)
vs 10.373 (parallel) — within noise. The prefix-max scan in `plan.md` §2.1 is
**not worth building**. The real attention win came from somewhere else
(§12), a structural transpose from llama.cpp.

**(b) The 58 s prefill was not host-side paging.**
My argument: a per-layer byte count divided by a per-layer time matched mmap read
speed. That is a **numerology fit**, not a measurement. `KAPLUMBA_RESIDENT_WEIGHTS=1`
puts every weight back in the arena and was **13% SLOWER** — which disproves
paging as the cause, because genuine paging pressure would be *reduced* by
residency. The real cause was the tensor GEMM. No production change was made on
the basis of the fit.

## 12. The wins, each with its mechanism

**12.1 `bstage` array overflow → MMA path correct → 11.3 → 24.5 tok/s** (`60ae014`)

`crates/kernels/src/linear.metal:2337` (at the time) declared
`threadgroup half bstage[QTILE * OUTS]` = 32×32 = 1024 halfs, but the dequant
loop wrote **64×32** and the fragment loads address `bstage + k8*OUTS` with `k8`
up to 56. The upper half **aliased `astage`**, which activation staging then
overwrote. Every `k >= 32` weight read activations. Not a wrong fragment layout —
an array-sized-too-small.

Why it shipped: `8f1a863` claimed "`linear_metal` 6/6 against a CPU reference",
but `gemm_chunk` (`crates/kernels/src/linear.rs:579`) **hardcodes
`gemv_tiled`**, so the MMA path had **zero** test coverage and the 6/6 was
counting the other kernel. Same failure mode as Q11 below.

Also in that commit, from two parallel audits:
- `key_count` returned `capacity` (1024) for every sliding layer forever, so
  attention scanned and accumulated over 1024 keys regardless of live count. Now
  `next_position.min(capacity)`, which collapses back to `capacity` once the ring
  wraps, so no separate wrapped path is needed. +3.8%.
  `crates/engine/src/gpu_cache.rs`.
- Latent NaN in the PV loop: masked keys had weight 0 but their **value was
  still loaded**, and `0.0f * NaN` is NaN, so unwritten cache slots could poison
  the accumulator. The load is now skipped, matching `attention.metal:624`.
  Live-key arithmetic and order untouched.

**12.2 64-query MMA tile, output tile aliased onto `astage` → 32.0 tok/s** (`c056b48`)

Each weight fragment now feeds 16 query rows per simdgroup instead of 8, so
multiplies-per-fragment-load went 0.80 → 1.33. Fitting that in the same 14336 B
required noticing the fp32 output tile does not need to exist: the accumulators
live in `simdgroup_matrix` registers for the whole K loop, so `astage` is dead by
the time outputs are staged, and the two blocks are the same size by
construction. **Aliasing the output tile onto `astage` holds the budget flat
instead of pushing to 22 KB.** Worst error bit-identical to the 32-row kernel at
2.16e-3 — the per-element accumulation order is unchanged.

**12.3 GQA-batched attention: attention 1324 → 338 ms, 3.9x** (`c056b48`)

llama.cpp puts one threadgroup per **query** head, so each of our 32 query heads
re-reads the same K/V — a re-read factor of **512**. Every query head sharing a
KV head now sits in **one** threadgroup and reads K/V out of one L1: re-read
factor **1**. Threadgroups 19648 → 9824 (sliding) / 2456 (full).

`attention_gqa_body<HEAD_DIM, NSG, PARTIALS>`,
`crates/kernels/src/attention.metal:725`; the transposed mapping is at
`attention.metal:755-762` (lane → KV index, simdgroup → query row) and follows
llama.cpp `fa_common.metal:411-454`. Threads became `32*NSG` with `NSG` = the GQA
ratio, so the online softmax is `simd_max`/`simd_sum` with **1 barrier per 32
keys** instead of two five-step shuffle ladders and 5 barriers per 256 keys.
Threadgroup memory fell 1056 → 512 B (sliding) / 2048 B (full). The SWA mask
stays **arithmetic** on `(k_abs, q_abs, window)` — llama.cpp materialises a
128 MiB f16 mask at L=8192, which would violate the §8 fixed-scratch rule.
Pipelines: `online_attention_gqa_{256,512}_{1,2,4,8}`, indexed by
`gqa_index(head_dim, nsg)` at `crates/kernels/src/attention.rs:132`.

**12.4 128×64 tile + lane-parallel dequant + two buffers deleted → 32.2 → 68.8 tok/s** (`1dd4da9`)

`linear_gemm_mma`, `crates/kernels/src/linear.metal:2360`, header commentary at
`linear.metal:2290-2359`. The dominant cost was **never** the
multiplies-per-load ratio — it was the **dequant**.

1. **Lane-parallel dequant.** `exl3_code_at_tg` derived the trellis window per
   weight, so 16 weights/thread cost three threadgroup loads and a 64-bit window
   setup **each**: ~400 instructions against 64 multiplies, and the kernel ran at
   13% of tensor roofline. One thread now decodes the **eight** codes a trellis
   lane owns from **one** window setup, exactly as `exl3_decode8`
   (`linear.metal:234`) does in the decode path: 16 weights/thread cost ~40
   instructions. Window setup at `linear.metal:2424-2442`, decode at
   `linear.metal:2468-2493`. **The codes are bit-identical** — same window, same
   two overlapping 64-bit windows, same 16-bit slices. Pure instruction-count
   win, zero numerical change.
2. **Tile 128q × 64n** on 256 threads (8 simdgroups, `QTILE=128`, `OUTS=64`,
   `KCHUNK=64`, `BSTRIDE=72`), `linear.metal:2367-2375`. Sixteen accumulators
   share ten fragment loads per step = 16 multiplies per load, against 8 per 6.
   Weights are streamed `ceil(queries/128)` times per output block instead of
   `ceil(queries/64)`, halving both weight DMA and activation re-reads. Eight MMA
   steps at `linear.metal:2521-2552`. Sixteen **named** accumulators at
   `linear.metal:2400-2417` — an array of chains spills to threadgroup memory
   and collapses 15.49 → 0.74 TFLOPS (`examples/mma_peak.rs`).
3. **`BSTRIDE = OUTS + 8 = 72`** (`linear.metal:2372`). The lane-parallel dequant
   scatters one 16-byte piece into each of eight rows; at a 128-byte stride all
   eight land on threadgroup bank 0 — an 8-way conflict on every store. At 144
   bytes the row bases fall on banks 0,4,…,28 and the eight pieces are
   conflict-free.
4. **`astage` DELETED.** Each simdgroup owns a disjoint set of query rows, so a
   staged `astage` was read exactly once by exactly one simdgroup: 32 global
   loads plus 32 threadgroup stores per thread per chunk bought coalescing and
   nothing else. A fragments now load straight from the activation buffer
   (`linear.metal:2390-2391`, row pointers clamped with `min(..., queries-1)`
   because rows past the query count would read past the buffer).
5. **`tw` DELETED**, replaced by a software-pipelined global→register prefetch
   one chunk ahead (`linear.metal:2500-2517`). A staged 1.5 KB trellis block
   needs a barrier whose only job is to publish that copy, and it measured
   **15%** — more than the dequant arithmetic it fed.
6. **Outputs leave the accumulators directly** via a row-guarded 4-byte store
   (`KAPLUMBA_MMA_STORE`, `linear.metal:2560-2568`). A 128×64 f32 staging tile is
   32 KB, far past the 16 KB ceiling, and `simdgroup_store` has no per-row mask,
   so each lane recomputes its own (row, column) from its lane id
   (`linear.metal:2394-2396`).

**Threadgroup memory therefore FELL 14336 → 9216 B**: 64×72 halfs of dequantised
weights and nothing else.

Result: **5.31 TFLOPS** on the projections (was 2.11), **35% of the 15.18 TFLOPS
`mma_peak`** measured on this machine today. down/gate/up GPU time
4447/3725/3708 → 1665/1514/1512 ms. **A staging-stripped diagnostic caps this
tile shape at 76.7 tok/s**, so the remaining dequant is ~11% of headroom and the
rest needs a different tile.

**12.5 Decode: deleted the O(L²) fp32 score matrix** (`1dd4da9`)

`attn_scores`, `attn_row_max`, `attn_row_sum` and the **five kernels** that used
them are **deleted**. Decode now runs the **same** online-softmax body as prefill
via a `PARTIALS` template parameter (`attention.metal:724`, contract documented
at `attention.metal:707-724`, threadgroup mapping at `:755-769`, partial
write-out at `:868`), emitting an unnormalised `(m, l, O)` per split;
`attention_decode_combine_{256,512}` merges with `exp(m_s − M)`
(`crates/kernels/src/attention.rs:189-190`, dispatch at `:387`, partials length
assert at `:354`).

Partial buffer: `16 splits × 32 heads × 514 × 4 B = 1,052,672 B`, **fixed size
and independent of context** (`crates/engine/src/model.rs:38`, sized at
`model.rs:1953-1968`). Scratch is now sized from **layer shapes**, never from
`gpu_cache.capacity()`. Threadgroup memory 2 KB worst case.

This fixed a **hard failure**: `attention score buffer is 131072 bytes, needs
508800 for 32 heads x 3975 keys` — 131072/32/4 = exactly 1024 keys, so decode
worked at 914 and died at 1215. It was exactly the O(L²) scratch `AGENTS.md` §8
forbids; prefill was already tiled, decode was not. **2414 tokens of context,
which previously died, now prefills at 75.1 tok/s and decodes at 4.96 tok/s.**

## 13. The test that lied (Q11 in `spec.md`)

`mma_prefill_gemm_matches_cpu_reference` set `KAPLUMBA_DISABLE_MMA_PREFILL=1`,
which **DISABLES the path under test**, so a deliberately sabotaged
`linear_gemm_mma` still passed. Identical failure mode to the original "6/6"
claim in `8f1a863` (§12.1). Two false coverage claims in one project.

Fixed: `crates/engine/tests/linear_metal.rs:900` now calls
`remove_var("KAPLUMBA_DISABLE_MMA_PREFILL")`; query count raised 33 → 65 to
exercise a full 64-row block plus a partial one. The test now has **teeth**:
sabotaging the kernel makes it fail with `worst normalised error inf`. The
128×64 kernel added query counts 1/127/128/129/255/256/257 for the 128-row tail
plus a canary row past the output buffer, and a sabotaged row guard was verified
to fail the test. Worst error is 2.1603336e-3, **bit-identical to the 32.2
tok/s kernel** — same weight bits, same `KCHUNK`, same per-element accumulation
order.

**Still open, not fixed because it is a comment in a source file:**
`crates/engine/tests/linear_metal.rs:745-753` says `mma_gemm_rows` is called
"with `KAPLUMBA_DISABLE_MMA_PREFILL=1` set by the caller so the dispatch selects
`linear_gemm_mma`" — backwards — and that the gate requires
`in_features % 32 == 0` / `out_features % 32 == 0` when
`crates/kernels/src/linear.rs:976-987` actually gates on `% 64`. Anyone relying
on that comment will be misled.

## 14. Rejected experiments — do not retry

| experiment | measured | why |
|---|---|---|
| 64×64 output tile | 8.27 tok/s (2.58 with a dynamic loop bound) | widened **N**, left Q=16, so the weight-reuse amortisation factor never moved; plus a 20 KB threadgroup. `ba9263f` |
| `KCHUNK=32` (128×64 shape) | −2% |
| staged `astage` (128×64 shape) | −5% | simdgroups own disjoint query rows; see §12.4.4 |
| 32 accumulators/simdgroup (32q row block) | ratio 2.67 but **10% slower** | 64 accumulator registers per thread |
| 8→4 simdgroups with `KCHUNK=32` | no change | |
| 18 KB threadgroup (padded) | 21.1 tok/s | the 16 KB occupancy cliff is real and binding |
| 256-entry float codebook LUT | worse | **our codebook has only 8 distinct values**, so a `select` chain needs 7 selects and beats `decode_state`'s ~6 ops. And oMLX explicitly rejects dynamically-indexed tables on Apple: `omlx/custom_kernels/bonsai/csrc/quantized.h:612-613`, "no dynamic array index which would spill to device memory on Apple GPU" |
| `metal::extract_bits` (one instr vs shift+mask) | n/a | **our `0x8fff8fff` window is non-contiguous**, so extract_bits cannot express it |
| oMLX gate/up concat fusion (`qwen35_packed_linear.py` `project()`) | **−2.3%** | |
| `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` (llama.cpp `ggml-metal.cpp:1025-1029`, issue #20141) | **null** | no command buffer is giant any more |
| fusing `gated_gelu` into the `up_proj` epilogue | rejected pre-measurement | removes a 21504-wide intermediate worth ~0.2% of runtime; gate+up's 7.4 s is arithmetic that only a better inner loop moves |
| `KAPLUMBA_RESIDENT_WEIGHTS=1` | **13% slower** | killed the host-paging theory (§11b) |

## 15. Open, with numbers

- **152 tok/s at 8192 is not reached and, on this tile shape, is not reachable.**
  The measured scaling fit (164–3974 tokens, free memory held 15.1–15.4 GiB) is
  `t = 1567 + 27.93·L + 0.000533·L²` ms, log-log exponent **+0.066** — nearly
  flat. Context is **not** the constraint; GEMM efficiency is. Attention's
  quadratic term is only 7% of per-token cost at 3974 and ~14% at 8192. The fit
  was taken at 32.2 tok/s and **predates `1dd4da9`**; re-measure before quoting.
- Peak RSS is flat at 7.81–8.06 GiB (projected 8.20 at 8192) because KV is
  preallocated. **8192 was not run** — free memory fell below the 16 GiB gate.
- **A staging-stripped diagnostic caps the current tile shape at 76.7 tok/s**, so
  ~11% remains in staging/dequant and the rest needs a different tile. This is
  now the whole remaining project.
- **`lm_head.trellis` (1.06 GB) is mapped by default despite a code comment
  saying that is "actively harmful."** `crates/engine/src/weights.rs:530-533`
  records the measurement (12.99 → 14.14 GiB, "actively harmful. Left
  resident"), and `weights.rs:552` computes
  `let offload_head_trellis = offload_head && name == "lm_head.trellis";`
  which **maps it**, because `offload_head = !force_resident`
  (`weights.rs:544`). **RESOLVED 2026-09-30: the comment was the error, not the
  code — keep mapping it.** The 12.99 GiB it used as a baseline was a run that
  had already failed (`lm_head.suh has 0 values`) and never allocated KV or
  scratch, so it was artificially low and the comparison inverted the sign of
  the result. Real A/B, same session, 2500 tokens: **11.98 GiB mapped vs
  15.57 GiB all-resident**, a 3.59 GiB saving equal to the 3.609 GiB offload
  pair, with `time -l` `swaps 0` and `vm.swapusage` 0 MiB at every 5 s sample.
  The comment has been corrected in place; no code change was needed. A
  throughput cost was **not** established: a first pair suggested ~24 %
  (9.124 vs 11.992 tok/s), but a repeat of the *same* default config returned
  12.338, a 35 % same-config swing that swamps the effect — `decode-20tok` Q6
  stays open. `phys_footprint` by contrast reproduced to 196 488 bytes across
  two identical runs. Full data in
  `spec/resident-memory-12gib/implementation_notes.md` (session 2026-09-30).
- **Decode is 5.3–6.9 tok/s at 614**, degrading to 1.6–2.1 at 4814. The
  degradation is the expected cost of rescanning a long context, not a failure —
  but it is below the 7.87 tok/s in notes §1–§3 and the difference is not
  explained. Notes §8.3 already warned that the 7.87 baseline is not
  reproducible; a ~10% machine-level shift (thermal, background load, differing
  resident set) is present. **Any A/B must run back-to-back in one session.**

## 16. Process and environment

- **A `flock` on `~/.kaplumba/gpu.lock` now serialises model loads inside the
  binary** (`d678a79`). Two concurrent loads exceed unified memory; this repo
  has already measured that cost — `crates/engine/src/weights.rs:508-511`:
  "file-backed pages get evicted, which measured 3.9 GiB of swap and ~1.4
  tok/s". A wrapper script is not sufficient, because a parallel agent fleet
  runs the binary directly.
  Implementation `crates/engine/src/gpu_lock.rs`: `acquire()` at `:63`,
  `acquire_lock_file()` at `:81`, `libc::flock(fd, LOCK_EX|LOCK_NB)` at `:105`,
  static `HELD: OnceLock<File>` at `:34`. Called from
  `crates/engine/src/model.rs:2076`. Released by the kernel when the fd closes,
  so normal exit, abort and `kill -9` all release it — a crashed holder cannot
  wedge it. The `File` lives in a `OnceLock` that is never dropped; threads
  inside one process serialise on a `Mutex` first, because `OnceLock::get_or_init`
  cannot propagate an error. A second process **waits** and prints who it is
  waiting for plus current free memory — waiting, not failing, is deliberate.
  Verified: second instance printed `GPU is busy, waiting for the lock. Free
  memory now 1.4 GiB…`, then produced 9.873 tok/s against the first's 9.836.
- **`MetalEncoder` has a `Drop` behind an `ended` flag with a single private
  `end()` choke point**, so a double `end_encoding` is impossible. Without it
  any error path propagating out of `forward_buffer_into` dropped an un-ended
  encoder and Metal SIGTRAPped. Verified the test has teeth: with `Drop`
  neutered it dies with `Command encoder released without endEncoding`.
- `commit_and_waitless` → **`commit_terminal`**, with `debug_assert!` on every
  `dispatch_*`. It previously left a dead encoder and a stale dispatch counter.
- **metal-rs returns unretained autoreleased pointers and the process has no
  autorelease pool**, so every command buffer leaked. Switched to the crate's own
  owned handles (`foreign_type` `ToOwned=retain`, `Drop=release`). Measured
  1896 B leaked per buffer/encoder pair before, 0 after: 260 KB per prefill,
  79.6 KB per decode token. **Draining the pool is still open** and is the
  correct follow-up.
- `prefill_tok_s` always reported 0.000 on the pipelined path — a shadowed
  `prefill_duration` hid the real value.
- In-process timer instrumentation is **untrustworthy here**. Two separate
  approaches (per-dispatch timers, then `new_command_buffer` counters) produced
  **no output at all** despite the format string being present in the binary —
  the same failure as the earlier session's `ApiTimer`. Use
  `KAPLUMBA_GPU_PROFILE=1`, or external `sample`/DTrace.

### Commands

```bash
# shaders are include_str!'d into the CLI; --workspace does NOT rebuild it
cargo build --release -p kaplumba-cli

# end-to-end, 614 prompt tokens
M=/Users/zayaantaha/.kaplumba/models/Gxma/Gxma
PP=$(python3 -c "print('Explain the theory of relativity. '*100)")
./target/release/kaplumba --model "$M" --prompt "$PP" --max-new-tokens 4 --temperature 0 --metrics

# real per-kernel GPU time. Selectors inside are CAPITALISED; the env var is not.
KAPLUMBA_GPU_PROFILE=1 ./target/release/kaplumba --model "$M" --prompt "$PP" ...

# A/B the tensor path against the scalar tiled kernel
KAPLUMBA_DISABLE_MMA_PREFILL=1 ./target/release/kaplumba ...

# 5 reference prompts; expected: Paris / 1,2,3,4,5 / red, blue, and yellow /
# "Hello! How can I help you today?" / "17 times 23 is 391"
for q in "What is the capital of France?" "Count from one to five." \
         "Name three primary colors." "Hi" "What is 17 times 23?"; do
    ./target/release/kaplumba --model "$M" --prompt "$q" --max-new-tokens 24 --temperature 0
done

# cheapest test that exercises BOTH prefill and decode on the real checkpoint
export KAPLUMBA_TEST_MODEL=$M
cargo test --release -p kaplumba-engine --test real_forward_buffer \
  -- --ignored --exact real_model_buffer_forward_prefill_then_decode   # ~10 s
```

**Reference-prompt caveat, recorded so it is not re-investigated:** 4/5 are
byte-identical across repeated runs. #5 ("What is 17 times 23?") is fluent but
reworded, and it diverges **identically on the pre-change binary**, so the
divergence predates this work. Worst GEMM error is unchanged at 2.16e-3.

## 2026-09-28 — 8k context measured: 69.7 tok/s, and the "model won't load" red herring

**8k result, the number the whole spec has been targeting:**

| context | prefill tok/s | TTFT | peak RSS |
|---|---|---|---|
| 614 | 71.5 | 8.6 s | 8.6 GB |
| **7814** | **69.7** | **112.1 s** | **7.7 GB** |

Only 2.5% slower than 614, so throughput is essentially context-independent and
the flat scaling law holds. RSS is flat because the KV cache is preallocated.
Reproduce with `*1300` repetitions of the test sentence = 7814 tokens, i.e.
**6.01 tokens per repetition** — get that constant wrong and the prompt is
silently rejected:

    PP=$(python3 -c "print('Explain the theory of relativity. '*1300)")

At 1365 repetitions the prompt is 8656 tokens, over the 8192 limit, and the
binary prints "prompt plus generation exceeds context limit 8192" and exits
WITHOUT LOADING. This cost three separate debugging detours in one session:
the error went to a pipe whose head was not being read, so it looked exactly
like a silent hang, a stuck queue, and a broken model load. Before concluding
"the model is not loading", run the binary with `2>&1 | head -5` and a short
prompt, and confirm the token count first.

**Still open at 69.7, and it is not context.** Two independent kernel-level
analyses converged on the same conclusion:

- `linear_gemm_mma` is **~90% tensor-pipe bound** (deleting all 16 MMAs leaves
  2.34 ms of 23.82), and the dequant is now **free** (deleting the entire EXL3
  decode arithmetic moves 23.91 -> 23.84 ms, ranges overlapping) after the
  `ka==3 && codebook==1` arm folded MCG to 4 instructions.
- Wider MMA fragments **do not exist on this hardware**: the compiler header
  `/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/.../metal_simdgroup_matrix`
  carries `static_assert(Cols == 8 && Rows == 8)`, and 8x8 is the only
  instantiation. 16x8/8x16/32x32 all fail to compile.
- Exhausted and reverted, do not retry: 32 accumulators/simdgroup (64 registers,
  spills, 0.82x), 256 rows over 16 simdgroups/512 threads (0.64x), 512q (0.54x),
  128q x 128n 4x4 fragments (0.993x), register staging of B (B is shared by all
  8 simdgroups), persistent threadgroup across query blocks (0.53% ceiling, and
  it needs 160 regs/lane), split-K (re-associates the k-sum, reduces zero MMA
  work), #pragma unroll, fragment prefetch, unpad BSTRIDE 64, KCHUNK=128.

So the honest position: the prefill GEMM is at ~90% of a hardware pipe that has
no wider form, and the remaining headroom is small. Reaching 120 tok/s from here
requires either a warm ceiling above the measured ~7.8 TFLOPS, or a way to cut
work rather than cut overhead. The load-bearing number to re-verify is that
~7.8 TFLOPS ceiling: `examples/mma_peak` prints 15.49 for the same instruction
and cannot be reproduced, and one agent's own 8x8 probe printed 111 TFLOPS
before `loop_ratio=0.359` showed the compiler had folded the loop away.

## 14. `PREFILL_CHUNK` 256 -> 1024: the tail is not the problem, dispatch
## fragmentation is. Plus a pre-existing nondeterminism bug in chunked prefill

### 14.1 The premise was wrong, and the profile says so in one line

The idea under test was "a 614-token prompt ends in a 102-token chunk that is
only 40% of a 512-row tile, so raise `PREFILL_CHUNK`". The tile count refutes it:
`linear_gemm_mma` dispatches `ceil(queries / 128)` threadgroups along y
(`linear.rs:1043`), so **the number of 128-row tiles is a function of the token
count, not the chunk count**. 614 tokens is `ceil(614/128) = 5` tiles whether the
host slices it 256+256+102, 512+102, or 614. All four chunk sizes below issue
the same 5 tiles and waste the same 26 of 640 rows (4.06%).

`KAPLUMBA_GPU_PROFILE=1` was extended with a per-chunk row count
(`by_chunk`, keyed `<rows>:<kernel>`) so a multi-chunk prefill can be split;
counts come out exactly 60 per chunk per kernel, no drain smear, at
`DRAIN_EVERY = 15`. GEMM own_ms at 614 tokens, grouped by tiles per dispatch:

| `PREFILL_CHUNK` | tiles/dispatch | ms per 128-row tile | GEMM total |
|---|---:|---:|---:|
| 128 | 1 | 1354.0 | 6783.5 |
| 256 | 2, 2, 1 | 1271.7 | 6429.6 |
| 512 | 4, 1 | 1222.8 | 6235.9 |
| 1024 | 5 | 1204.1 | 6020.7 |

**The loss is 6.4% of GEMM, and it is grid size, not tail occupancy.** A small
grid does not fill 20 GPU cores. The 128 row is the control that proves it: same
5 tiles, five dispatches of one, slowest of all four. Sub-128-row tiles cannot
help either -- 8x8 is the only MMA the hardware header instantiates (notes §13),
and at 128-row granularity the waste is `(-tokens) mod 128` by construction.

### 14.2 What shipped, and what it costs

`PREFILL_CHUNK = 1024`. Interleaved A/B, real checkpoint,
`--max-new-tokens 1 --metrics`, medians of 5, spreads 0.0-1.7%:

| | 614 tok/s | rss @614 | 7814 tok/s | rss @7814 |
|---|---:|---:|---:|---:|
| 256 (before) | 70.62 (69.75-70.93) | 7.85 GiB | 69.34 (69.08-69.42) | 8.15 GiB |
| 512 | 71.97 (71.52-72.21) | 7.90 GiB | -- | -- |
| **1024 (after)** | **77.79 (77.55-78.23)** | **8.01 GiB** | **74.57 (74.30-74.59)** | **7.91 GiB** |

+10.2% at 614, +7.5% at 7814. Scratch 93 MB -> 372 MB, and `peak_resident_pages_bytes` is
flat-to-+160 MB, so the 12 GiB budget is untouched: the extra scratch displaces
file cache, not the 15.5 GB of weights. 2048 was rejected -- 743 MB of scratch
for no measured gain once the grid is already 1344 threadgroups.

Per-element accumulation order is untouched (a chunk boundary splits QUERY rows,
never K), `linear_metal` is 7/7 and its worst error is still 2.1603336e-3, and
all five reference prompts are correct. **Sabotage check, as §3.4 demands**:
retargeting `y7` to the wrong column chunk makes only
`mma_prefill_gemm_matches_cpu_reference` fail, with `worst normalised error
8.416497e-2 exceeds 2e-2`; removing the `if (r < params.queries)` store guard
fails 7/7. So the suite really does cover the MMA path -- and note that only
ONE of the seven tests does; the other six exercise `linear_gemv_tiled`.

### 14.3 BUG FOUND, pre-existing and worse than the thing I was measuring

Greedy output on a 614-token prompt is **nondeterministic at HEAD**, at either
chunk size. Same binary, same prompt, three runs at `PREFILL_CHUNK=256`:

```
The theory of relativity is not
Albert Einstein's theory of
Since you asked many times,
```

and at 1024 two samples differed too. A short prompt that is 2 chunks at 256 and
1 at 1024 flipped the other way (256 -> "Albert Einstein's", 1024 -> "Since you
asked"), so it is not a function of chunk size. One-chunk prompts are always
stable and correct.

Cause: `forward_chunk` memcpys `scratch.token_ids` and `scratch.query_positions`
from the host once per chunk, into `StorageModeShared` buffers, with **no
barrier between chunks**, while `MetalEncoder::roll` keeps up to
`MAX_IN_FLIGHT + 1 = 9` command buffers outstanding -- 450 dispatches of slack.
A 256-row chunk is ~900 dispatches, and attention is stamp 8 of ~15 per layer, so
the host reaches chunk N+1's write while the GPU is still in chunk N's middle
layers, and the remaining layers attend non-causally (`q_abs` reads as
256+query while only 256 keys exist). The hazard is documented for the *decode*
path (`forward_buffer_into_step`, "The caller must hold a barrier across the
next call") and was never applied to the prefill chunk loop.

Repro, GPU-free to check, decisive -- any of these serialises and all four chunk
sizes then agree on "Albert Einstein's theory of":

```bash
KAPLUMBA_MAX_IN_FLIGHT=1 KAPLUMBA_PREFILL_CHUNK=256 ...   # 128/256/1024 identical
KAPLUMBA_PROFILE_PHASES=1 ...
KAPLUMBA_GPU_PROFILE=1 ...
```

`KAPLUMBA_MAX_IN_FLIGHT` is a new measurement override in
`crates/kernels/src/encoder.rs` (1 = wait on every rolled buffer). It exists
because a 9-deep window is a licence to race, and without it there is no way to
tell a real numerical difference between two chunk sizes from scheduling noise.

Consequences: AGENTS.md §8's "no quality loss / greedy-identical" does not
currently hold for any prompt longer than `PREFILL_CHUNK`, and every prefill
benchmark on a multi-chunk prompt has been measuring a model whose output is a
function of scheduling. **Not fixed here** -- it is a separate change with its
own performance cost, and it deserves its own spec. What `PREFILL_CHUNK = 1024`
does buy is exposure: 2 chunk boundaries at 614 tokens instead of 2 (0), 30
instead of 7 at 7814, and 0 for any prompt up to 1024 tokens.

### 14.4 Environment facts worth keeping

- The engine's `flock` is what makes measurements trustworthy under contention:
  every sample that appears in a table above ran with the GPU to itself, and
  every sample lost to another agent's run was exit 75 and discarded. A second
  agent was interleaving 8k benchmarks in `/tmp/wt-fuse-target` throughout; the
  A/B interleaving absorbed it, but roughly a third of attempted runs were 75.
- `scripts/ab.sh` / `scripts/dec.sh` / `scripts/prof.sh` / `scripts/bisect.sh` in
  the throwaway worktree implement "on 75, discard the cell and carry on".
  `scripts/prof.sh <chunk> <reps> <size>` is the one to reuse: it prints
  `chunks=`, `prefill_tok_s`, `peak_resident_pages_bytes`, summed GEMM ms, and the per-chunk
  GEMM breakdown in one line per run.

---

## 2026-09-29: the search converged, and it converged against 120 tok/s

Prefill went 77.8 -> **81.3** at 614 and 73.7 -> **75.0** at 8192 this session.
More importantly, six research angles came back negative and the reasons are now
recorded, because they close the largest open questions in the project.

### Landed

- `1b5de3c` — register-pipelined dequant (`linear_gemm_mma_regpipe`).
  `linear_gemm_mma` alternated dequant / barrier / MMA / barrier inside one
  threadgroup, and both barriers are load-bearing for the single 9216 B `bstage`,
  so the ALU and tensor phases never coexisted in time. A second `bstage` cannot
  fix that: `simdgroup_multiply_accumulate` is a 32-lane collective, so all 8
  simdgroups must enter together. Each thread already decodes its OWN 16 weights
  per chunk and those fit in 16 registers, so writing them to registers instead
  of `bstage` gives one barrier per chunk and puts `{ MMAs; dequant }` in one
  basic block with no memory dependency. Bit-identical (SHA f3508d3a31537325);
  `bstage` unchanged so occupancy is unaffected. Defaults on for folded-MCG
  (rate 3 + Mcg, i.e. every projection here); `KAPLUMBA_NO_REGPIPE=1` falls back.
  **Caveat on the win:** an ablation attributes only 1.8% of it to the actual
  overlap and ~3.6% to the barrier reduction. The overlap is not the win.
- `fc474ff` — hoisted Q's half->float conversion out of the per-key attention
  loop. `qrow` is loop-invariant (one query row per threadgroup) but was reloaded
  and reconverted for every key, which was half the loop's instructions. All 32
  lanes want the same row, so it is converted once into threadgroup memory and
  read back as floats. Bit-exact by construction. +2% at 8k, nothing at 614,
  which is what the cost model predicts (attention is 5.4% of prefill at 614 and
  26.1% at 8k).

### Closed negatively — do not re-litigate these

- **8 accumulators instead of 16: 0.944x, slower.** The premise was wrong, not
  just the result. At 614 tokens the dispatch is 1680 threadgroups on 20 cores =
  **84 per core**, so any 3-vs-4 residency cap is oversubscribed ~20x and can
  never bind. A probe found no knee at 3 or 4 and no sensitivity to +-32 dummy
  registers. The `+92.6%` for 16->8 chains came from a microbenchmark with 84
  groups/core of nothing else to compete with. Occupancy was never the limit.
  Tail-waste premise was also inverted: a 64-row tile never wastes more than a
  128-row one (both round 614 up to 640; at 7814 the 64-row tile is better).
- **`KCHUNK` sweep 16/32/64/128/256: 64 is optimal.** 16 and 32 are ~2.5-4.5%
  slower, 128 is free at 614 but -7.5% at 8k (1 group/core), and 256 is
  impossible: `bstage` would be 36 864 B against a 32 768 B device maximum.
  `OUTS` 32 is -8%, `OUTS` 128 is +1.4% at 8k only. Question closed.
- **The MCG dequant is free; "46.6%" was an artifact.** Isolated properly, the
  entire decode is 0.303 ms and the fetch-kept-alive-no-decode arm is 0.298 ms.
  The 0.18 ms that "removing" the decode appeared to save was the *trellis word
  fetch* being dead-coded away, not arithmetic. In the real kernel, deleting all
  dequant math makes it **4.4% slower**. Real ratio is ~1.4 ALU/MMA, and +8
  injected ops per weight costs only 4.5%. The codebook is folded: there is no
  select and no table.
- **No host submission overhead exists to remove.** The 1.42 s "stall" is not a
  stall. Per-dispatch host cost is **0.60 us**; 99.78% of wall is
  `wait_until_completed` on the 9th rolled buffer, which is correct
  back-pressure. Raising the in-flight window from 8 to 16 changed nothing
  (20.9-22.3 vs 22.1-23.1, i.e. noise), which is the decisive test: if there were
  really 19% dead time a deeper queue would have absorbed it. The overlap is
  already maxed. An earlier "GPU idle 18.5-19.9%" was a measurement artifact.
  Batching into fewer, larger command buffers is 0.913x -- the stall grows with
  work per buffer.
- **The MMA ceiling is not 14.79 TFLOPS.** That number (from a
  structure-matched 16-chain issue probe) disagrees by ~19x with another
  structure-matched probe reading 0.79 on the real tile shape, and A13's
  ablation puts MMAs at 99.5% of kernel time. Three agents had converged on 7.93
  (a low-chain-count artifact) before this. **The 14.79 is the number that
  produced the "~123 tok/s ceiling" and it does not survive contact.** Treat the
  prefill GEMM as near its achievable issue rate and stop promising 120 from
  this shape.
- **fp8 does not exist on this M4 Pro.** `float8_t_e4m3`/`e5m2` fail as
  *undeclared identifier* as scalar, matrix, 16x16, mixed accumulator, and even
  `as_type<char4>` — the types are absent from the stdlib, not an operand-width
  error. No 2x is reachable via precision. `simdgroup_float8x8` exists but is a
  half alias. The 7.01 vs 7.93 penalty is a *narrow accumulator*, not a narrow
  operand, so half accumulators are not a register saving.
- **Repacking the weights can never reduce bytes.** 410 tensors sit at exactly
  3.000 bits/value (43 352 064 / 451 584 = 96 B/tile) and lm_head at 6.000. The
  3.04->3.35 effective-bpw gap is bf16 embed_tokens + scales + lm_head at 6 bpw,
  not padding. The kernel calls `decode_state(s & 0xffff, ...)` on the FULL 16-bit
  window, never masked to the rate: 14 490 distinct states -> 7 103 distinct
  values, 12.51 bits of real content against 3.000 bits paid. `exl3_decode8` has
  24/24 distinct words across 32 lanes, i.e. 100% of a tile with zero inter-lane
  overlap, so the 4x is intra-L1 within a 96 B working set and DRAM is already
  optimal.
- **Page faults during decode are irrelevant** (20/token, 0.003% of 12.06 GB;
  236 547 major faults are the one-off load). The layer trellis is *dirty*
  anonymous arena (11 GB, `SM=SHM PURGE=N`), so it is compressible and evictable
  but not free. Full residency is rejected on measurement: decode 8.70/8.93/8.99
  (default) vs 7.59/8.31/8.80 (resident), prefill 2.5-4.3x slower, +3.55 GiB.
- **Metal pipeline caches cost nothing.** All 68 pipelines plus 4 `include_str!`
  libraries total **6.97 MiB**; compilation is out-of-process
  (`MTLCompilerService`) with zero load-time transient (peak 9.4 MiB during
  load, 43+8+12 = 63 ms warm).
- **Widening the attention query tile does not work.** Q_TILE=2 is 0.87, =4 is
  0.88, =8 is 0.98 (spills). It trades K/V bytes for *more* Q bytes and more PV
  work. A persistent-tile estimate of 26.8 s -> 8-11 s (+15-18%) is not reachable
  this way. Attention at 8k is instead bound by the per-lane serial QK loop
  being instruction-bound on ~2 `cvt.f32.f16` per FMA (57% cvt / 29% FMA / 14%
  loads) — which is what `fc474ff` attacked.

### Measurement traps hit this session, all now documented

- **`peak_resident_pages_bytes` was wrong by 5.5 GiB** and is now `peak_resident_pages_bytes`
  (`ee14f62`). See AGENTS.md §10.3. This is what made a 1.5 GiB
  context-scaling question sound like there was several GB of slack.
- **Untouched buffers are free.** A 354 MiB `ForwardScratch` that is allocated
  but never written charges 0.016 MiB; at load the whole 840 MB KV cache was
  uncharged. Footprint only appears on first write, so no static audit can
  predict it and every "resident bytes" under-report is explained by this.
- **The 1.5 GiB context delta is fully accounted for** and is not a leak:
  11.696 GiB @ 622 tok -> 12.963 GiB @ 8126. 1038 and 1630 tokens are
  byte-identical in both allocated and footprint, so there is no hidden O(L)
  term beyond KV. Delta = KV growth (full layers 80->640 MiB) + sliding-window
  fill as context passes 1024 (~312 MiB) + scratch (213->354 MiB) + a residual
  ~0.25-0.5 GiB of driver-side Metal graphics regions that are not returned.
- **Sliding-layer KV allocated at the full 1024 window is NOT waste.** The
  sliding caches *ring* (`gpu_cache.rs:10,123-132`), so position 0 is evicted
  once the ring wraps and the buffer genuinely needs its full window from the
  start. Full-attention layers get grow-on-demand because they are not rings and
  hold the whole context; sliding layers never grow, so the asymmetry is correct
  and intentional. Combined with "untouched is free", the "333 MiB never
  written" figure is worth approximately nothing.
- **In-flight command buffers are not the residual.** `KAPLUMBA_MAX_IN_FLIGHT`
  2 vs 8 at 7814 tok: 12.93 vs 12.95 GiB, i.e. 23 MiB for a 4x change in window,
  at no speed cost (75.3 vs 75.6). ~5-7 MiB retained per buffer, consistent with
  the earlier 34.75 MB -> ~5 MB measurement. The bulk of the residual is
  committed-then-released buffers plus driver graphics regions, and A14 already
  showed that batching them is 0.913x. Characterized, not fixable without a
  speed regression.
- **The in-flight window is saturated at 8**: 1 -> 18.7, 4 -> 21.1, 8 -> 22.1-23.1
  tok/s on a short prompt, and 8 -> 12 -> 16 is noise. The 8 is right.
- **`footprint` prints 3 significant digits** (`10 GB`, `120 MB`), so big
  categories carry +-512 MiB of quantisation. An agent initially misread
  `11 GB` as 11264 MiB and invented a 280 MiB phantom. Do not do byte-exact
  cross-checks above ~1 GB from this tool.
- **`scripts/gpuq holder()` named dead pids** as the GPU holder indefinitely
  (`waiters()` pruned, `holder()` did not), so an agent reported the box busy
  against a pid that had exited 20 s earlier. Fixed in `ee14f62`; all three cases
  (stale / live / missing) verified.

### Open, in priority order

1. **Nondeterminism is a live AGENTS.md §8 violation.** 14 runs of the same
   binary gave 13 identical outputs and 1 outlier, and that is recorded in
   `1f11f47`'s own commit message. Greedy-identical is required. Not fixed.
2. **The quality gate cannot detect a 2e-3 change.** `output_identity.rs` is
   `#[ignore]`d, reads `lines().next()` for 32 tokens on 5 prompts (~160 tokens
   total), and `MMA_TOLERANCE` is 2e-2 against 2.16e-3 actual, so a 9x
   regression passes. Proposed replacement: a 16-prompt SHA-256 logit
   fingerprint (262144 f16 row + top-8 pairs + top1-top2 margin), 12 prompts x
   256 tokens exact-match, and 5x determinism per prompt in a fresh process. The
   determinism arm fails today, correctly.
3. **120 tok/s is not reachable from this implementation shape.** It needs less
   math or fewer operands, not faster execution of the same math. The two
   directions nobody has costed: exploiting the 12.51 bits of real per-weight
   value content, and narrowing the B operand — though fp8 is confirmed absent
   on this part, so that one is closed.
4. **Attention at 8k.** A QK `simdgroup_matrix` rewrite is estimated 2.4x on
   attention (+18% at 8k, +0.7% at 614) but is not bit-exact, and an 8x8
   fragment needs 8 query rows = 128 accumulator registers per lane. Unbuilt.
5. `decode-matvec-k3` is preserved on that branch (commit `e360769`, bit-exact,
   measured 8.7 -> 9.3 tok/s) but is unlanded and unverified against current HEAD.

### Corrections to earlier notes in this file

- The KV arithmetic in §-history that assumed 16 KB/token x 60 layers =
  0.98 MB/token is WRONG. The two layer types differ: sliding is 16 heads x 256 =
  16 KB/token but capped at the 1024 window (800 MiB total, fixed), and
  full-attention is 4 heads x 512 = 8 KB/token and grows (80 MiB -> 640 MiB at
  8k). KV total is 880 MiB at 614 tok, not the ~2.1 GB or 8.05 GB implied.
- `max_sequence_len` is a hardcoded 8192 (`model.rs:2284`), NOT the checkpoint's
  `max_position_embeddings` of 262144.
- The budget is **14 GiB** (AGENTS.md said 12 in several places; corrected in
  `ee14f62` and the `gpuq` warning).
