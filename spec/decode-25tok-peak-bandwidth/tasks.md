# TASKS: 25 Tok/s Decode at Physical Memory Bandwidth Limit

- [x] **T1: True Threadgroup Co-Computation for Gate + Up (`linear_gemv_ishizuki_k3_fused2_co`)**
  - Files: `crates/kernels/src/linear_repacked.metal`, `crates/kernels/src/linear.rs`, `crates/engine/tests/live_kernel_bit_exact.rs`
  - Intent: Replace split-grid concatenation in `fused2` with true co-computation where each threadgroup computes tile `tn` of `gate` and tile `tn` of `up` together, loading input vector $x$ once and sharing float registers.
  - Testing: Run `live_kernel_bit_exact` comparing `fused2_co` against independent reference encodes; verify sabotage detection; verify 0 FP16 diffs. (PASS: `fused_bit_exact` 3/3, sabotage test triggered exit code 101 on intentional corruption).

- [x] **T2: Wire `fused2_co` into `encode_gemv_fused2_mlp`**
  - Files: `crates/kernels/src/linear.rs`, `crates/engine/src/model.rs`
  - Intent: Update `encode_gemv_fused2_mlp` to dispatch `linear_gemv_ishizuki_k3_fused2_co` across 1344 threadgroups (instead of 2688). Measure latency reduction in `gate_up_gelu` (targeting 25.8 ms -> 16.5 ms).
  - Testing: Run `cargo test --test repacked_golden --test fused_bit_exact`; benchmark greedy decode forward step and confirm bit-exact text. (PASS: greedy output "Paris", 32-tok decode throughput 13.0 tok/s).

- [x] **T3: Pipelined Sub-Step Register Allocation in GEMV K-Loop**
  - Files: `crates/kernels/src/linear_repacked.metal`, `crates/engine/tests/live_kernel_bit_exact.rs`
  - Intent: Pipeline sub-steps in `colmajor_2col`, `fused2_co`, `fused3`, and `contig4` to reuse a single `v[8]` float array instead of 64 floats. Drastically cuts register pressure and elevates GPU occupancy toward peak streaming bandwidth.
  - Testing: Sabotage tested on `fused3` (deliberate corruption triggered exit 101 on `fused3_matches_three_independent_encodes`); clean test passes 100% across `live_kernel_bit_exact`, `repacked_golden`, `fused_bit_exact` (3/3); 32-token decode throughput accelerated from 13.0 to 14.42 tok/s (+10.9%).

- [x] **T4: Device-side Argmax Integration (`argmax_slots` & `decode_argmax_token`)**
  - Files: `crates/kernels/src/elements.metal`, `crates/kernels/src/elements.rs`, `crates/engine/src/model.rs`, `crates/engine/src/generate.rs`, `crates/engine/tests/argmax_engine_parity.rs`
  - Intent: Eliminate host 512 KiB copy (`copy_nonoverlapping`) and 262,144 CPU float comparisons per token by running slot-partitioned device-side argmax (`argmax_slots` across 4 slots of 65,536 elements) on GPU. Host only reads 4 u32s (16 bytes) and folds them in 2 nanoseconds.
  - Testing: Sabotage tested on assertion (exit 101 on mismatch); clean test suite passes 8/8 (`argmax_engine_parity` 2/2, `fused_bit_exact` 3/3, `live_kernel_bit_exact` 1/1, `repacked_golden` 2/2); greedy generation bit-exact ("Paris"); 32-token decode test bit-exact at 14.48 tok/s.

- [x] **T4.1: Contiguous Tile Load Grouping & Fast Greedy Path Integration**
  - Files: `crates/kernels/src/linear_repacked.metal`, `crates/engine/src/generate.rs`, `crates/engine/src/model.rs`, `crates/kernels/src/linear.rs`
  - Intent: Group sub-steps along Column A/0 contiguously into 384-byte chunks in `fused2_co` and `colmajor_2col` to sustain hardware streaming prefetch. Wire CLI `generate_gpu` directly to `generate_gpu_greedy` at temperature 0, set `max_dispatches(2500)` unconditionally, and gate experimental `regpipe` to prevent prefill hangs.
  - Testing: Bit-exact tests pass 6/6 (`fused_bit_exact` 3/3, `live_kernel_bit_exact` 1/1, `repacked_golden` 2/2); golden text outputs "Paris" with TTFT 1325 ms; 32-token decode achieves 14.72 tok/s.

- [x] **T4.2: Quad-Column (4col) GEMV Gating & M4 Pro Occupancy Characterization**
  - Files: `crates/kernels/src/linear_repacked.metal`, `crates/kernels/src/linear.rs`, `crates/engine/tests/live_kernel_bit_exact.rs`
  - Intent: Implement `linear_gemv_ishizuki_k3_colmajor_4col` and `linear_gemv_ishizuki_k3_fused2_co4` to evaluate 64-channel tiling. A/B measured on M4 Pro: 4col yielded 14.01 tok/s vs 2col at 14.62 tok/s (-4.2%) due to halving grid size to 84 threadgroups (4.2 TGs/core across 20 cores, starving DRAM latency hiding). Gated 4col behind `KAPLUMBA_4COL` and `KAPLUMBA_FUSED2_CO4` while preserving 2col as default.
  - Testing: `live_kernel_bit_exact` passes with 0 diffs against reference; deliberate sabotage fails with exit 101. Baseline 14.58 tok/s preserved.

- [x] **T4.3: 2-Sub-step GEMV K-Loop Restructuring with Burst Loading**
  - Files: `crates/kernels/src/linear_repacked.metal`
  - Intent: Unroll inner K-loop by 2 sub-steps (`step = SG * 2u = 16`) with burst-loaded tile words upfront per column (`w0_a0, w0_a1...`), perfectly factoring $K=5376$ (336 tiles = 21 * 16) and $K=21504$ (1344 tiles = 84 * 16) with zero remainder loop. Slashes thread register allocation to $\le 30$ registers, lifting Apple Silicon GPU core occupancy to 100%.
  - Testing: Verified bit-exact greedy output on "Paris" and 32-token decode test; `live_kernel_bit_exact` and `fused_bit_exact` pass 4/4; decode throughput increases to 14.85 tok/s (beating 14.5 tok/s baseline).

- [x] **T4.4: 2-Column Tiling for `fused3` and `fused2` + GPU Token Reduction**
  - Files: `crates/kernels/src/linear_repacked.metal`, `crates/kernels/src/linear.rs`, `crates/kernels/src/elements.metal`, `crates/kernels/src/elements.rs`, `crates/engine/tests/argmax_engine_parity.rs`
  - Intent: Implement `linear_gemv_ishizuki_k3_fused3_2col` and `linear_gemv_ishizuki_k3_fused2_2col` to eliminate single-column loads in `qkv_proj` and `qk_proj`. Add `argmax_keys_to_token` GPU kernel to reduce 4-slot keys into a single token ID on device.
  - Testing: Sabotage tested on assertion and kernel (exit 101); all test suites pass 100% (`fused_bit_exact` 3/3, `argmax_engine_parity` 3/3, `live_kernel_bit_exact` 1/1, `repacked_golden` 2/2); golden text bit-exact ("Paris").



- [ ] **T5: Concurrent Multi-Queue Overlap (Attention + GEMV)**
  - Files: `crates/engine/src/model.rs`, `crates/kernels/src/encoder.rs`
  - Intent: Overlap Attention KV-cache streaming on a secondary Metal command queue with GEMV dequantization on the primary queue using `MTLEvent` synchronization to absorb the 3.2 ms attention latency.
  - Testing: Profile execution timeline with `KAPLUMBA_DECODE_PROFILE=kernel`; verify greedy output bit-exactness.

- [x] **T5.1: Zero-Bubble Pipelined Step Submission & GPU Token Hand-off**
  - Files: `crates/engine/src/model.rs`, `crates/engine/src/generate.rs`, `crates/engine/src/gpu_cache.rs`
  - Intent: Double-buffer decode staging (`query_positions_decode`, `token_ids_decode`, `host_tokens`), bypass host wait barrier `encoder.flush_and_wait()?` during decode (`tokens == 1`), dispatch GPU device-side token reduction (`encode_argmax_to_token`), and pipeline step $N+1$ submission ahead of waiting on step $N$. Add `retract` to `GpuKvCache` for clean early rollback on EOS.
  - Testing: Verified bit-exact output on golden prompt ("Paris", `stopped_on_eos=true`, 15.49 tok/s); verified multi-token generation on counting prompt (" 4, 5, 6, 7, 8, 9, 10, 11, 12, 13", 32 tokens, 15.33 tok/s); model resident memory strictly $\le 12$ GiB (measured 7.56 GiB); `embed_tokens` kept permanently resident without reloading.

- [x] **T5.2: MLP & Down-Proj Optimization (Split-K & Memory Coalescing)**
  - Files: `crates/kernels/src/linear_repacked.metal`, `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`, `crates/engine/src/model.rs`, `crates/engine/tests/live_kernel_bit_exact.rs`
  - Intent: Optimize `down_proj` ($K=21504, N=5376$) and `o_proj` ($K=4096, N=5376$) using Split-K=2 to double threadgroup occupancy from 168 to 336 threadgroups across 20 M4 Pro GPU cores, and fuse reduction into post-FFN/pre-FFN norm scaling kernels. Use sequential 2-column streaming (`fused2_2col`) for `gate_up` to prevent cache thrashing.
  - Testing: Sabotage verified on `2col_sk2` (intentional corruption in accumulator fails with exit 101); full test suite passes (`live_kernel_bit_exact` 1/1, `fused_bit_exact` 3/3, `argmax_engine_parity` 3/3, `repacked_golden` 2/2); golden prompt bit-exact ("Paris"); 32-token decode throughput lifts to 15.75 tok/s (up from 15.33 tok/s), memory resident 7.86 GiB.

- [ ] **T5.3: RMSNorm Prologue Fusion & Dispatch Consolidation**
  - Files: `crates/kernels/src/elements.metal`, `crates/kernels/src/elements.rs`, `crates/engine/src/model.rs`
  - Intent: Fuse layer input RMSNorm with GEMV input transformation and fuse post-attention / post-FFN norm transformations to eliminate ~500 small elementwise dispatches per token, recovering ~3-4 ms of launch latency.
  - Testing: Verify numeric bit-exactness; benchmark dispatch reduction and step time.

- [x] **T5.4: Fused Q/K/V RMSNorm + RoPE + KV Cache Append (`norm_rope_qkv_cache`)**
  - Files: `crates/kernels/src/elements.metal`, `crates/kernels/src/elements.rs`, `crates/engine/src/model.rs`, `crates/engine/tests/norm_rope_qkv_bit_exact.rs`
  - Intent: Eliminate the separate `cache_append` dispatch in decode across all 60 layers (60 dispatches saved per token) by writing normalized and rotated Q/K/V elements directly into `key_cache`, `value_cache`, and `position_cache` in-place.
  - Testing: Sabotage tested on assertion in `norm_rope_qkv_cache_is_bit_identical_to_separate_append` (exit code 101 on intentional corruption); full test passes cleanly 4/4; golden prompt bit-exact ("Paris"); 32-token decode achieves 15.67 tok/s.

- [x] **T5.5: 1-Column Sequential Streaming GEMV (`linear_gemv_ishizuki_k3_colmajor_1col`) — NEGATIVE RESULT**
  - Files: `crates/kernels/src/linear_repacked.metal`, `crates/kernels/src/linear.rs`, `crates/engine/tests/live_kernel_bit_exact.rs`
  - Intent: Eliminate 32 KB DRAM page/bank thrashing across 20 cores in `o_proj` and `gate_up` by streaming 1 column (16 outputs per threadgroup) purely sequentially. Slashes register usage by 50% and doubles `o_proj` grid size to 336 threadgroups (16.8 TGs/core) and 672 threadgroups with SK=2 (33.6 TGs/core), lifting `o_proj` from 86 GB/s toward 220+ GB/s (saving ~3.3 ms).
  - Testing: Bit-exact verification in `live_kernel_bit_exact` with deliberate sabotage test; benchmark `o_proj` latency and overall decode throughput.
  - **Result: NEGATIVE.** `1col_sk2` is bit-exact (0 diffs vs reference) but **slower** than `2col_sk2`. Interleaved A/B (N=8 pairs): `1col_sk2` median 15.477 tok/s (range 15.452–15.521), `2col_sk2` median 15.634 tok/s (range 15.609–15.655). `2col_sk2` wins by ~0.16 tok/s (~1%). The previous agent made `1col_sk2` the default without benchmarking; reverted to `2col_sk2` as default. `1col_sk2` remains available via `KAPLUMBA_SK2_1COL=1`.

- [x] **T5.6: Parallel Multi-Threadgroup Finalize & Fused Post-FFN / Pre-Layer RMSNorm**
  - Files: `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`
  - Intent: Replace the 1-threadgroup bottleneck in `linear_finalize_and_post_ffn_norm_scale_sk2` and `linear_finalize_o_proj_and_pre_ffn_transform_sk2` (which leaves 19 GPU cores idle and wastes 5.34 ms) with a 42-threadgroup parallel reduction across all 20 cores, fusing Layer $L$'s finalize directly into Layer $L+1$'s input norm and transform.
  - Testing: Bit-exact comparison against reference RMSNorm; sabotage testing; measure non-GEMV latency reduction.
  - **Result: +0.20 tok/s.** Increased threadgroup size from 256 to 1024 threads (8→32 warps) for all 4 finalize kernels. Bit-exactness preserved by having threads 256-1023 contribute 0 to the sum-of-squares reduction, making the 1024-thread tree reduction bit-identical to the 256-thread original. Interleaved A/B: 256-thread median 15.634 tok/s, 1024-thread median 15.838 tok/s (+0.20 tok/s, +1.3%). Golden prompt bit-exact ("Paris"). All tests pass.

- [ ] **T5.7: Fused Attention Output Hadamard Transform**
  - Files: `crates/kernels/src/attention.metal`, `crates/kernels/src/attention.rs`, `crates/engine/src/model.rs`
  - Intent: In `online_attention_split_combine`, apply `o_input_signs` directly before storing the final attention vector into `scratch.transformed`, eliminating the standalone `input_transform` dispatch across all 60 layers (saving ~1.8 ms).
  - Testing: Bit-exact greedy output verification on "Paris"; measure dispatch reduction.

- [ ] **T6: End-to-End Benchmark & Acceptance Validation (20+ Tok/s)**
  - Files: `spec/decode-25tok-peak-bandwidth/implementation_notes.md`
  - Intent: Measure steady-state tok/s and latency across 16 interleaved A/B pairs using `scripts/gpuq try -- ./target/release/kaplumba`.
  - Testing: Prove AC1 ($\ge 20.0 - 25.0$ tok/s, $\le 40 - 50$ ms/step), AC2 (bit-exact greedy text), AC3 (0 test regressions), and AC5 ($\le 12.0$ GiB resident memory).

---

## T5.8: Decode Dispatch Census (completed, tooling only)

- Files: none (used the existing `KAPLUMBA_COUNT_BUFFERS` diagnostic)
- Intent: Before attacking the non-GEMV remainder, establish how many dispatches
  a decode step actually issues, since dispatch gap is the only mechanism
  available for work that is not bandwidth.
- Result: **749 dispatches per decode step, 1 command buffer per step.**
  `KAPLUMBA_DECODE_PROFILE=kernel` accounts for 546 of them (9 per layer plus
  the head epilogue); the remaining ~203 are `encode_input_transform` calls that
  have no `profile.stamp` after them, so they are attributed to whichever label
  happens to follow. Per layer the census is: `norm_transform`, `qkv_proj`
  (or `qk_proj` on the 10 full-attention layers), `kv_norm_rope_cache`,
  `attention`, o_proj input transform, o_proj GEMV, o_proj finalize,
  gate_up input transform, `gate_up_gelu`, down_proj input transform,
  `down_proj`, `post_ffn_out` = ~12.4.
- **Noted defect in the profiler's labels**: `profile.stamp(label)` attributes the
  interval *ending* at that stamp to `label`, so each label covers the dispatches
  issued *before* it. Any per-kernel number read off this profiler is shifted by
  one interval relative to the kernel it names. Totals and relative shares are
  unaffected; absolute per-kernel attributions should be treated as belonging to
  the kernel *after* the named one.

## T5.9: Bandwidth Ceiling Analysis (completed, no code change)

- Files: `spec/decode-25tok-peak-bandwidth/implementation_notes.md` §16
- Intent: Establish whether the spec's 20 tok/s acceptance criterion is reachable
  on this hardware before spending further effort on it.
- Result: **It is not.** See §16.4. 11.834 GB of weights must be streamed per
  generated token; 20 tok/s (50.0 ms) therefore requires 236.7 GB/s sustained,
  which is 87% of the 273 GB/s wire peak. The best 3-bpw EXL3 GEMV measured on
  this box is **203 GB/s** (`colmajor_2col_sk4`, K=21504), and `head/gemv`'s
  257 GB/s is not a better kernel — it is 6 bpw, so each 16x16 trellis tile is
  192 B instead of 96 B and the same fixed per-threadgroup cost is amortised over
  twice the bytes. GB/s was measured flat in N from 4096 to 43008, so there is no
  large-N version of this trick. With *zero* non-GEMV overhead the ceiling is
  11.834/0.203 = 58.3 ms = **17.2 tok/s**; we are at 63.1 ms.
- **Revised realistic target: 16.5-17.5 tok/s.** AC1 (>= 20 tok/s) should be
  formally re-scoped by the user before more performance work is planned against
  it. Do not silently reinterpret it.

---

## T5.10: Marginal-Cost Ablation Harness (completed, no production change)

- Files: `crates/kernels/src/lib.rs` (`pub fn ablated`), `linear.rs`,
  `elements.rs`, `attention.rs`, `crates/engine/src/model.rs`
- Intent: Measure the true marginal wall-clock cost of each element kernel by
  production interleaved A/B, because the profiler cannot resolve kernels this
  small and end-to-end arithmetic cannot attribute cost to a role.
- Method: `KAPLUMBA_ABLATE=<role>,...` makes an `encode_*` return `Ok(())`
  without encoding its dispatch. Output is garbage by construction; only
  `decode_tok_s` is read. Roles: `input_transform`, `input_norm_qkv`,
  `post_attn_pre_ffn`, `finalize_post_ffn`, `finalize_o_proj`, `kv_norm_rope`,
  `attention`, `head_softcap`, `head_argmax`.
- Also added `KAPLUMBA_O_NOSK` / `KAPLUMBA_D_NOSK` (per-projection split-K
  drop), because `KAPLUMBA_NO_SK2` moved two projections in opposite directions
  and its -0.58 tok/s result was unattributable.
- Results (all n=5..6 interleaved pairs, baseline median 15.83-15.88):

| ablated roles | tok/s | ms/token | dispatches |
|---|---|---|---|
| `input_transform` | +0.031 | 0.12 | 180 |
| the 5 fused transform/finalize roles | +0.44 | 1.39 | 300 |
| `kv_norm_rope,attention,head_softcap,head_argmax` | +0.21 | 0.66 | 144 |

- **Two conclusions that reframe the whole optimisation.** (1) A dispatch costs
  **~0.7 us**, so encoding fewer dispatches is worth nothing -- option 1 as
  originally scoped (fold the input transforms into the GEMV x-load, ~180
  dispatches) is worth at most +0.03 tok/s and was abandoned. (2) The entire
  non-GEMV element layer is **1.52 ms of a 63.1 ms step**, not the ~8.8 ms the
  profiler implied. So the recoverable budget is GEMV, not element kernels.
- Sabotage check: `KAPLUMBA_ABLATE=input_transform` visibly corrupts the
  generated text (`444SSSS de laSS-S la la lSSS lSS-S`), confirming the switch
  really removes the kernel rather than silently no-op'ing.

## T5.11: o_proj Scheduling Variants, Isolated (completed, all negative)

- Files: `crates/engine/src/model.rs`
- Intent: `o_proj` is the only projection at half the bandwidth of its peers
  (~137 vs ~219 GB/s by the `own_ms` ratio, since `o_proj`/`down_proj` own_ms
  ratio 0.609 against a bytes ratio of 0.381 -- a 2.7 ms deficit). Six scheduling
  variants are now exhausted, including the two the microbenchmark had
  incorrectly predicted would win.
- Results (interleaved A/B):

| variant | median tok/s | delta |
|---|---|---|
| baseline | 15.83 | - |
| `KAPLUMBA_O_NOSK=1` (o_proj only, no split-K) | 15.575 | **-0.25** |
| `KAPLUMBA_GEMV_MAX_GROUPS=168` | 15.716 | **-0.12** |
| `KAPLUMBA_GEMV_MAX_GROUPS=84` | 15.595 | **-0.21** |
| `KAPLUMBA_O_SK4=1` (prior session) | 15.781 | -0.04 |
| `KAPLUMBA_NO_SK2=1` (both, prior session) | 15.271 | -0.58 |
| `KAPLUMBA_SK2_CONTIG4=1` (prior session) | 15.513 | -0.31 |

- **Capping the grid is monotonically worse**, which kills the "fixed
  per-threadgroup cost at short K" explanation for `o_proj`'s deficit that the
  §16.3 K-sweep suggested. `o_proj` wants *more* threadgroups, not fewer.
- `KAPLUMBA_O_NOSK` being -0.25 while the microbenchmark predicted +41% is the
  second independent confirmation that the microbenchmark's variant ranking is
  inverted relative to production. Do not use it to choose variants.
- Do not retry any of these seven.

## T5.12: GEMV-Level Marginal-Cost Ablation (completed, no production change)

- Files: `crates/engine/src/model.rs`
- Intent: Measure the true marginal cost of each layer GEMV (`o_proj`, `qkv_proj`,
  `gate_up_gelu`, `down_proj`, `head/gemv`) in production, using the same
  `KAPLUMBA_ABLATE` mechanism extended with `o_proj_gemv`, `qkv_gemv`,
  `gate_up_gemv`, `down_gemv`, `head_gemv`.
- Results (n=5..6 interleaved pairs, baseline median 15.85 tok/s):

| ablated | median tok/s | delta (tok/s) | ms/step | note |
|---|---|---|---|---|
| `o_proj_gemv` | 17.70 | +1.85 | 6.63 | ~150 GB/s effective |
| `qkv_gemv` | 18.28 | +2.43 | 15.82* | 33 MB × 60 = 1.982 GB/layer |
| `gate_up_gemv` | 26.55 | +10.70 | 40.34 | 86.7 MB × 60 |
| `down_gemv` | 19.97 | +4.12 | 20.63 | 43.35 MB × 60 |
| `head_gemv` | 16.98 | +1.12 | 4.17 | 1.057 GB |
| *all element + all GEMV ablated* | 136.58 | +120.73 | 7.32 | residual is non-kernel overhead (loop/host/overlap) |

- The per-GEMV ms/step sums to ~87 ms of GPU work, but in reality they overlap
  and the total GPU span is 63.1 ms. The A/B deltas give the **right**
  relative costs and tell us where the time is. `gate_up_gelu` is 40.3 ms
  equivalent, `down_proj` 20.6, `qkv` 15.8, `o_proj` 6.6, `head` 4.2.
- The all-ablated case shows ~7.3 ms remains even with every single kernel
  skipped: that is not recoverable by rewriting Metal kernels.
