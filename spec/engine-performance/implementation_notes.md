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
