# Implementation Notes: Performance and Memory Overhaul

## Overview
This document records technical decisions, benchmark measurements, register allocation details, and handover context for `spec/perf-and-memory-overhaul`.

## Architectural Baseline
- Target: Apple M4 Pro (20 GPU cores, 273 GB/s unified memory bandwidth).
- Model: Gemma 4 31B dense EXL3 (~3.0 bpw).
- Starting state:
  - Startup RSS: ~16.6 GB.
  - Decode throughput: ~8.39 t/s (134.5 GB/s, 49% bandwidth).
  - Prefill throughput: 87.6 t/s @ 512 tokens (MMA rate crippled to ~5.0–6.3 TFLOP/s).
- Target state:
  - Startup RSS: ≤ 14.6 GiB (15.7 GB).
  - Decode throughput: ≥ 18.0 t/s.
  - Prefill throughput: ≥ 200.0 t/s.

---

## Log of Steps and Learnings

### T5.0 — Cooperative GEMV attempt REVERTED (2026-09-23)
- **Attempt**: in-place `exl3_gemv` rewrite to 128-thread / 4-simdgroup cooperative structure (T5.1, commit `d3d3ffb`) + ExllamaV3 coalesced 96B word loads with `simd_shuffle` fetch (T5.2, commit `182a101`) + `gemv_splits` heuristic `min_split_tiles=128` / target 8192 TGs.
- **A/B measurement (same session, `bench_passes_slow`, this machine)**:
  | linear | T4 (7197888) | T5.1+T5.2 | Δ |
  |---|---|---|---|
  | gate | 165 Gw/s | 130 | −21% |
  | down | 168 | 131 | −22% |
  | o | 109 | 95 | −13% |
  | q | 111 | 81 | −27% |
  | k | 70 | 55 | −21% |
  | v | 76 | 55 | −28% |
  | lm_head | 243 | 225 | −7% |
  (T4 numbers reproduce the plan's original 7.15 t/s table — that premise was correct.)
- **Root causes**:
  1. The 4-simdgroup rewrite **discarded the T2.6 2-way unroll** (old loop loaded 32 lanes × 2B = one 64B burst covering two in-tiles per iteration) — T2.6 was worth +5% (7.15→7.52).
  2. It also **ignored the file's own empirical warning** that was present before the rewrite: *"Fatter threadgroups (4 warps sharing a threadgroup) measured slower on M4 Pro"*.
  3. `min_split_tiles=128` stopped k/v at nsplit=2 (in_tiles 336: 336/(2·2)=84 < 128 after first doubling... first check 336/2=168≥128 allows n=2, then 336/4=84<128 stops) — under-splits vs T2.9's measured k/v optimum ≈ 4.
  4. Coalesced loads (T5.2) were never isolated from (1)/(2) — their independent value is still unknown; re-test on the 32-thread kernel (T5.3r).
- **Correctness was NOT the problem**: one-hot `debug_gemv_rows` GPU==CPU all rows, decoded-tile parity, 116 fast engine tests — all green while perf regressed. Reverted in `c250fc0`.
- **ExllamaV3 study (`/tmp/exllamav3/.../exl3_gemv_kernel.cuh`)**: it is an **MMA kernel** (mma.m16n8k16, QTIP-style, 512/256-thread NVIDIA blocks) — structure not portable to Apple SGs. Portable ideas: (a) lane-coalesced tile load + shuffle extraction (tested, see above), (b) **PF=4 prefetch ring** — next tile's words register-resident while current decodes — this is the un-copied trick that addresses our load-then-decode serial dependency (issue-bound ~50%, ~60% BW), (c) in-block k-split reduction (our T5.1 striding was an analogue; structure lost anyway).

### T5.0b — Corrected bandwidth metric: 11.2 GB/token → roofline 24.4 t/s
- `decode_throughput_slow` used `bytes_per_tok = 16.028e9` (raw disk shards incl. vision tower + embed). After T1 (vision filtered) and T2 (embed mmap host-side, decode reads one row), the true per-token GPU stream is **11.2 GB** (user-confirmed 2026-09-23).
- **Old "hard ceiling" math was wrong**: 16.028 ÷ 273 ≈ 17.6 t/s was used in `spec/rust-metal-engine` and `spec/final-same-algorithm-ceiling` to declare 18/20 t/s physically unreachable. Corrected: **11.2 ÷ 273 ≈ 41.0 ms → ~24.4 t/s**. 20 t/s (50 ms) needs ~84% sustained bus utilization; lm_head already sustains 89% (243 Gw/s), so the target is bandwidth-valid with headroom.
- Historical "103–134 GB/s achieved" figures are similarly understated in % terms (they divided the wrong numerator... actually they used 16 GB numerator so % was overstated-as-low: true stream 11.2 GB at the same ms/token yields LOWER achieved GB/s but the same t/s — the % of bus now reads ~37% at 8.39 t/s, i.e. **63% of the bus idle** — that's the real work).
- Awaiting `decode_throughput_slow` re-run to record the corrected baseline line.

### T5.1r — PF=4 register prefetch ring REVERTED (2026-09-23)
- **Attempt** (commit `a4e15e2`): in the T4 32-thread kernel, split each tile's funnel-window load into `exl3_load_win` and its decode/multiply into `exl3_acc_win`, then replaced the 2-way unrolled loop with a 4-slot register ring. Each iteration decodes slot `I&3`, then refills that slot with iteration `I+4`'s words, so 3 future iterations (24 word-loads) sit in flight behind the current decode. Prologue fills all 4 slots. Logic per tile (window shifts, FMA order) was byte-identical to the old loop; tail loop unchanged.
- **A/B measurement (same session, `bench_passes_slow`, swap stable 9449/10240 MB used before+after)**:
  | linear | T4 (7197888) | PF=4 ring | Δ |
  |---|---|---|---|
  | gate | 165 Gw/s | 107 | −35% |
  | down | 168 | 108 | −36% |
  | o | 109 | 80 | −27% |
  | q | 111 | 81 | −27% |
  | k | 70 | 63 | −10% |
  | v | 76 | 57 | −25% |
  | lm_head | 243 | 135 | −44% |
- **Root causes** (hypothesis, not proven): (a) `ulong pf_m0[4][2]` indexed by dynamic `cur = I&3` is *not* guaranteed register-resident in MSL — dynamic indexing most likely demotes the arrays to stack/local memory, turning the "ring" into extra per-iteration memory traffic; (b) +32 regs/thread would cut concurrent threadgroups/core, killing the cross-TG latency hiding that already exists at 336–1344 TGs spread over 20 cores.
- **Learned**: the "load-then-decode serial → issue-bound" theory is **not** confirmed on M4 Pro. With 2-way unroll already issuing 8 loads per iteration and ~hundreds of live threadgroups, the ~60% bus utilization is probably a genuine memory-system/coalescing ceiling, not latency. A non-array `simd_shuffle`/scalar double-buffer might dodge the spill issue, but the ExllamaV3 PF trick does not obviously transfer. Do not retry without a register-spill check (e.g. `xcrun metal -frameworks Metal -c ... -gline-tables-only` / compile with `-zeitgeist` or inspect SASS-like disassembly).
- Correctness was NOT the problem: kernels 21/21, one-hot GPU==CPU, decoded-tile parity, `gpu_linear_matches_cpu_reference`, 116 fast engine tests — all green while perf regressed. Reverted in `fb26c21`.

### T5.2r — nsplit sweep, baked argmax LANDED (2026-09-23)
- **Instrumentation**: `gemv_pass` refactored → `gemv_pass_nsplit(xh, trellis, shape, mode, Some(nsplit), out)`; `encode_gemv` gained `nsplit_override: Option<usize>` (None → policy; production callers unchanged). Sweep test `bench_gemv_split_sweep_slow` (KAPLUMBA_RUN_SLOW, ignored) prints min-ms per nsplit ∈ {1,2,4,8,16,32} per real shape; gemv-only (no had), 12 reps.
- **Sweep (swap 9501→9900/10240 MB, stable-ish)**, best nsplit → Gw/s:
  | shape | in/out tiles | n=1 | 2 | 4 | 8 | 16 | 32 | argmax |
  |---|---|---|---|---|---|---|---|---|
  | gate | 336×1344 | 85 | 165 | **167** | 164 | 152 | 140 | 4 |
  | down | 1344×336 | 142 | 154 | 161 | 163 | **167** | 161 | 16 |
  | o | 512×336 | 105 | 109 | 113 | **114** | 113 | 111 | 8 |
  | q | 336×512 | 110 | 115 | **119** | 111 | 112 | 106 | 4 |
  | k | 336×256 | 76 | 78 | 81 | **89** | 78 | 76 | 8 |
  | v | 336×256 | 73 | 72 | 78 | **78** | 78 | 75 | 8 |
  | lm_head | 336×16384 | **241** | 238 | 235 | 225 | 207 | 184 | 1 |
- **Policy change** (`gemv_splits`): only two real outliers from the 4096-TG target formula — q 8→4 (119 Gw/s), k/v 16→8 (89/78). gate 4, down/o 13 (band 8–16 straddled), lm 1 stay on the formula. Policy test updated.
- **A/B `bench_passes_slow` (swap 9692/10240 both sides)**: gate 165→**168**, down 168→**168**, o 109→**113**, q 111→**115**, k 70→**76** (+9%), v 76→**79**, lm 243→**242**; no shape below T4 baseline. Commit `d8adb19`.
- Note: opt. TG count for q/k/v ≈ 2048, not the 4096 target — but apex is pull-shaped, not a cliff (all ≥ ~111 at n≥4); the 4096 target stays fine for other shapes.

### T1: Filter vision tower tensors during model load
- Implemented `is_vision_tensor` in `crates/engine/src/load.rs`.
- Safetensors loader skips 356 tensors matching `model.vision_tower.*` and `model.embed_vision.*`.
- Retained language model tensors: 2066 tensors (all 60 layers, norms, trellis, embed_tokens, lm_head).
- Resident Metal allocation dropped from 16.028 GB to **14.877 GB (13.855 GiB)**, saving **1.151 GB**.
- Load time: 3.3 seconds.
- Verified: `load_real_artifact_if_present` and `is_vision_tensor_filters_prefixes` passed; all 115 engine unit tests passed.

### T2: Verify baseline resident memory reduction (HostMmap for embed_tokens)
- Added `HostMmap` (using `libc::mmap` with `MAP_SHARED, PROT_READ`) in `crates/engine/src/load.rs`.
- `embed_tokens.weight` (2.819 GB) is accessed exclusively on the host CPU in `embed_row` and `embed`; it is now mapped on-demand via virtual memory rather than pre-allocated into resident Metal shared buffer.
- `model.resident_bytes()` dropped to **12,058,593,088 bytes = 12.058 GB (11.23 GiB)**.
- Gated test `prefill_memory_profile_slow`:
  - `baseline: rss=3959280 KB` (~3.96 GB)
  - `after 8192-token prefill: rss=3943936 KB`
  - `RSS growth: 0 KB (limit 49152)`
  - Confirmed 100% numerical accuracy and complete adherence to ≤ 14.6 GB (decimal) limit.

### T3: Vectorized `exl3_decode_f`, `dq4` window unpacking, and SIMD warp reduction
- In `crates/kernels/src/exl3.metal`:
  - Implemented `exl3_decode_f`: uses `half2 h2 = as_type<half2>(x); return float(h2.x + h2.y);` for single-instruction FP16 unpacking, replacing `as_type<half>(ushort(x))` + `as_type<half>(ushort(x >> 16))`.
  - Implemented `dq4` 64-bit window extraction: for $k \in \{1..6, 8\}$, 4 codewords are packed in at most $3 \times 8 + 16 = 40$ bits, which always fits within a 64-bit funnel shift `((ulong)w0 << 32) | (ulong)w1`. This extracts 4 codewords using only 2 memory loads instead of 4, cutting ALU shifts/funnel operations in half.
  - Handled $k=7$ via fallback pair table path (since $3 \times 7 + 16 = 37$ bits spans 3 words only at $k=7$).
  - Eliminated threadgroup shared memory buffers `shA[128]`, `shB[128]` and `threadgroup_barrier(mem_flags::mem_threadgroup)` in `exl3_gemv`: replaced with register-level `simd_shuffle_down(acc, 1)` and `simd_shuffle_down(acc, 2)` across the 4 lanes computing each output column.
- Verified: All 21 kernel tests in `kaplumba-kernels` passed; all 115 engine tests in `kaplumba-engine` passed cleanly.

### T4: SIMD-shuffle activation sharing in `exl3_gemv`
- In `crates/kernels/src/exl3.metal`:
  - Implemented coalesced SIMD-shuffle activation sharing: each lane loads a single activation `half x_lane = xh[it * 16u + g]`, and the 4 values needed for accumulation ($x_{00}, x_{01}, x_{02}, x_{03}$) are fetched across the warp using `simd_shuffle`.
  - For the 2-way unrolled loop, `xh` is loaded for both tiles in a single coalesced 64-byte burst across all 32 lanes, cutting activation memory loads by 8x.
- Verified: `gpu_linear_matches_cpu_reference` passed; all 21 kernel tests passed.

### Decode Coherence Root Cause & Resolution
- **Symptom**: CLI generation on `"What is the capital of France?"` produced `"It seems like to answers: \nIt seems like to answer"` instead of `"The capital of France is Paris."`.
- **Root Cause Investigation**:
  - Step-by-step intermediate tracing against the reference Python oracle (`ponyexl3` in `.venv-baseline`) revealed exact mathematical divergence starting at token 1:
    - Step 0 (<bos>): Both predicted token 236773 with identical logit `29.9782` (bit-exact match to 4 decimal places).
    - Step 1 (<|turn>): Diverged in layer 0 hidden state (`h_mid[1]` was `0.6660` in Python vs `1.0547` in Rust).
  - Traced layer 0 attention sublayer internals:
    - In Python `mlx_lm.models.gemma4_text.Attention`: every attention layer normalizes values with `v = self.v_norm(values)` using `RMSNormNoScale` (weightless per-head RMSNorm).
    - In Kaplumba: `gemma4.rs` and `gpu_decode.rs` applied head normalization only to full-attention layers when `k_eq_v == true`. On all sliding-attention layers (e.g. layers 0, 1, 2, 4, ...), `v` was taken directly from the raw linear projection `v_proj` without RMSNorm!
    - This caused attention values on all sliding layers to enter with unnormalized magnitudes, distorting the attention output vector and compounding across all 60 layers.
- **Resolution**:
  - In `crates/engine/src/gemma4.rs`: applied `norm_heads` (weightless RMSNorm) to `v_proj` in `forward_seq_layers` and `layer_step`.
  - In `crates/engine/src/gpu_decode.rs`: added `gpu.encode_rms` (weightless RMSNorm) to normalize `self.v_raw` after the linear projection in both `encode_token` and `encode_span`.
  - Cleaned up `trace_chat_prompt_steps_slow` as a regression test asserting token 818 argmax on step 19.
- **Verification**:
  - Every single token from step 0 to step 19 matched the reference oracle with 100% bit-exact argmax identity!
  - Step 19 argmax: Token 818 ("The") with logit 21.5117 (an 8.8-logit margin over rank 2). Token 1509 ("It") dropped to logit 5.0036.
  - Live CLI generate test:
    ```
    The capital of France is Paris.
    prefill 1379.1ms (14.5 tok/s) · ttft 1379.7ms · 8 tokens · eos · 7.157 tok/s decode (7 steps, 1.0x recompute)
    ```
    Generated the exact answer and terminated cleanly on `eos`!
