# A3 — kernel selection: the 138-kernel census, and which of the four that ship could a different tiling beat

**Verdict: all four live kernels are the right choice for their shape, on every
axis that is pinned. Exactly one reachable 3-bpw variant changes an axis that
is *not* pinned — `linear_gemv_ishizuki_k3_fused2_4col`, behind
`KAPLUMBA_FUSED2_4COL=1`, on gate+up (43.2% of a token's bytes). My prior is
~30% win / ~30% loss / ~40% wash, and the honest headline is the census: of 138
kernels, four dispatch, 12 are one env var away, and 115 are built at load and
never named by anything a decode step runs.**

That experiment is **not** cheap and I am not recommending it blind.
`WHERE-THIS-STANDS.md:434-452` already named it, priced it (+0.3 median on a
−0.2…+0.65 range) and correctly declined it: **+0.3 tok/s is 1.9% and needs
>179 interleaved pairs ≈ 6.5 hours** at this box's recorded σ. §8 adds the two
things the census contributes — the experiment as specified is **confounded in
five variables, not one**, and its kernel is **missing T4.3's hoist** — and
proposes a per-kernel-GPU-time pre-check that costs an afternoon instead of a
day and a half.

Read-only. No model loaded, no GPU used, no `gpuq`, no microbenchmark.
Everything below is derived from `crates/kernels/src/linear.metal`,
`crates/kernels/src/linear_repacked.metal`, `crates/kernels/src/linear.rs` and
`crates/engine/src/model.rs` by
`crates/engine/tests/probe_kernsel_census.rs` (4 tests, green, every detector
with a negative control — §7).

---

## 0. Corrections to the brief, up front

| brief | reality | evidence |
|---|---|---|
| "gate,up (60) → `fused2_2col`" | **correct**, and the reason is arm *order*, not coincidence | `encode_gemv_fused2_mlp` tests `_4col`, then `_2col`, then `gate == up && _co` (`linear.rs:2578/2586/2594`). 21504 % 32 == 0 and `_2col` exists, so `_2col` always wins. `probe_mlpdispatch_tree.rs` already establishes this. |
| "the `_co` anomaly: `_2col` has 8 loads in flight, `_co` has 4" | **correct, and it is not a problem** — see §4 | `linear_repacked.metal:506-509` (four hoisted `const uint w…` before the first `extract_v_mcg_32`) vs `:829-830` (two). |
| "`fused3_2col` … is the only live GEMV with no split-K variant of its own kernel" | **correct**, and the reason is now visible in the source: `fused3_sk2` exists but is **1-column on the 64-bit shift chain**, so it is not a split-K variant of `fused3_2col` — it is a different kernel wearing a split-K flag. See §5. | census coverage matrix, §3 |
| "173 pipeline builds, 157 distinct kernels, 10.3 ms, all at load" | **correct, and it is not a throughput problem** — 10.3 ms once against a ~10.8 s load and a 62 ms step. The real cost of the 138 is *review surface*, not milliseconds. §6. | `spec/decode-19tok-repacked-250gb/implementation_notes.md:1385` |
| "`_nod` … measured *removing* decode made the kernel 26.7% slower at 3 bpw" | consistent; the census confirms all 8 `_nod` kernels are **row-major, 3-word, `nod_stub`**, i.e. a bandwidth probe on a layout this checkpoint does not use at all. | census rows |

Two claims in the wider findings set that this census **breaks**, both worth
more than anything in this document:

1. **`ROUND1-2-SUMMARY.md:30` / `WHERE-THIS-STANDS.md:53` rank #3, "drop the
   dead third ring word (`w2`, `j2`, both funnels) guarded by `KA <= 4`,
   +0.3–1.4 tok/s, bit-exact over 1,048,576 codes", points at
   `linear.metal:356-364` — which is `exl3_decode8_mcg`, the helper of the
   `linear_gemv_exl3_k*_mcg` family. That family is **unreachable for this
   checkpoint** (§6). The three live colmajor kernels already read exactly two
   ring words (`j0`, `j1`) and call `extract_v_mcg_32`; there is no `w2` to
   drop. `perf_probe_kloop_loads_in_flight.rs` already asserted
   `!body.contains("j2")` for those five kernels. **That ranked action is worth
   0.0 tok/s here, and it is the largest instruction-level item on the list
   after the sampler and the 1.6× gap.**
2. **`WHERE-THIS-STANDS.md:54` / `:202` name the flag `KAPLUMBA_O_SK2_1COL` for
   ranked action #4 (+0.73 tok/s). That flag does not exist.** The real one is
   `KAPLUMBA_SK2_1COL` (`linear.rs:2240`), it is **not** isolated to `o_proj`
   (it also switches `down_proj`), and `KAPLUMBA_O_NOSK` exists to isolate
   `o_proj` but `KAPLUMBA_O_SK2_1COL` does not. Anyone following that row
   measures nothing.

---

## 1. The dispatch table, re-derived (not re-read)

Every arm was re-read in source order and re-evaluated against this
checkpoint's facts. All four rows are confirmed; the "why" column is the guard
that makes each one win rather than merely be reachable.

| projection | entry point | kernel | guard that makes it win | TGs | bytes/TG |
|---|---|---|---|---:|---:|
| q,k,v (50 sliding) | `model.rs:1534` → `encode_gemv_fused3` | `…_k3_fused3_2col` | all three `N % 32 == 0` and `!KAPLUMBA_NO_FUSED3_2COL` (`linear.rs:2776`) | 512 | 64,512 |
| q,k (10 full) | `model.rs:1596` → `encode_gemv_fused2` | `…_k3_fused2_2col` | `N_a != N_b` (16384 vs 2048), so the `_co` arm is unreachable; then both `% 32 == 0` (`linear.rs:2430`) | 576 | 64,512 |
| gate,up (60) | `model.rs:2225` → `encode_gemv_fused2_mlp` | `…_k3_fused2_2col` | `_2col` is tested **before** `gate == up` (`linear.rs:2586` before `:2594`) | 1344 | 64,512 |
| o_proj, down_proj (60) | `model.rs:2015`/`:2314` → `encode_gemv_raw_sk2` | `…_k3_colmajor_2col_sk2` | `use_sk2` is an **opt-out** and all three shape guards hold, so it fires with no flag (`model.rs:2273`) | 336 | 129,024 / 49,152 / 98,304 |
| lm_head (1) | `model.rs:4420` → `encode_gemv` → `encode_gemv_raw` | `…_ishizuki_k6` | head is `perm_mode == 0`, so both `_2col` guards fail; `ishizuki_gemv_pipeline` skips its colmajor lookups and `_k6_w2` does not exist, so the rate-6 slot wins | 16,384 | 64,512 |

Byte model (recomputed from `config.rs`, and it lands on the published
figure exactly): **12,039,487,488 B/token = 12.039 GB**, which is the number
every tok/s in this repo is quoted against.

---

## 2. The complete census of 138 kernels

Derived by scanning `^kernel void NAME(` in both `.metal` files and reading
each body. Columns: **rate** (bpw), **layout**, **cols** (output tiles per
threadgroup; the fused arms own 2 *matrices* as their second dimension), **SK**
(split-K), **unr** (K-loop unroll depth), **ring** (ring words per lane per
tile), **alu** (shift-chain width: 32 = the §22 funnel-shift form, 64 = the
pre-§22 `ulong` form), **loads** (weight `uint` loads held across one unrolled
trip).

### 2a. `linear.metal` — 111 kernels

| kernel | rate | layout | cols | SK | unr | ring | alu | decode | loads |
|---|--:|---|--:|--:|--:|--:|--:|---|--:|
| `linear_input_transform` | – | – | 1 | 1 | 1 | n/a | – | – | 0 |
| `linear_finalize_simd` | – | – | 1 | 1 | 1 | n/a | – | – | 0 |
| `linear_finalize_simd_sk2` | – | – | 1 | 1 | 1 | n/a | – | – | 0 |
| `linear_finalize_simd_sk4` | – | – | 1 | 1 | 1 | n/a | – | – | 0 |
| `linear_finalize_simd3` | – | – | 1 | 1 | 1 | n/a | – | – | 0 |
| `linear_finalize_simd3_sk2` | – | – | 1 | 1 | 1 | n/a | – | – | 0 |
| `linear_finalize_gated_gelu_simd` | – | – | 1 | 1 | 1 | n/a | – | – | 0 |
| `linear_finalize_gated_gelu_simd_sk2` | – | – | 1 | 1 | 1 | n/a | – | – | 0 |
| `linear_finalize_gated_gelu_and_down_input_transform` | – | – | 1 | 1 | 1 | n/a | – | – | 0 |
| `linear_finalize_gated_gelu_and_down_input_transform_sk2` | – | – | 1 | 1 | 1 | n/a | – | – | 0 |
| `linear_finalize_and_post_ffn_norm_scale` (+`_sk2`, `_sk4`) | – | – | 1 | 1 | 1 | n/a | – | – | 0 |
| `linear_finalize_o_proj_and_pre_ffn_transform` (+`_sk2`, `_sk4`) | – | – | 1 | 1 | 1 | n/a | – | – | 0 |
| `post_attn_pre_ffn_transform` | – | – | 1 | 1 | 1 | n/a | – | – | 0 |
| `input_norm_and_qkv_transform` | – | – | 1 | 1 | 1 | n/a | – | – | 0 |
| `finalize_post_ffn_and_next_input_norm_transform` (T3 cross-layer) | – | – | 1 | 1 | 1 | n/a | – | – | 0 |
| `linear_gemv_ishizuki_k1`…`k8` (8) | 1–8 | row-major | 1 | 1 | 2 | 3 | 64 | `decode_state` (runtime CB) | 6 |
| `linear_gemv_exl3_k1`…`k8`, `…_u` (16) | 1–8 | row-major | 8 | 1 | 1 | n/a | – | runtime CB | 0 |
| `linear_gemv_exl3_k1_mcg`…`k8_mcg`, `…_u_mcg` (16) | 1–8 | row-major | 8 | 1 | 1 | n/a | – | `exl3_decode8_mcg` (folded) | 0 |
| `linear_gemv_exl3_k1_mcg_r2/r4/r8`…`k8` (24) | 1–8 | row-major | 8 | 1 | 1 | n/a | – | `exl3_decode8_mcg` | 0 |
| `linear_gemv_ishizuki_k1_w2`…`k4_w2` (4) | 1–4 | row-major | 1 | 1 | 2 | 3 | 64 | `decode_state` (runtime CB) | 4 |
| `linear_gemv_ishizuki_k1_nod`…`k8_nod` (8) | 1–8 | row-major | 1 | 1 | 2 | 3 | 64 | `nod_stub` — **garbage by design** | 6 |
| `linear_gemv` (scalar reference, 1 thread/output) | any | row-major | 1 | 1 | 1 | n/a | – | `decode_state` | 0 |
| `linear_finalize`, `linear_finalize_rows`, `linear_zero_raw` | – | – | 1 | 1 | 1 | n/a | – | – | 0 |
| `linear_input_rows` (prefill) | – | – | 1 | 1 | – | 3 | – | – | 3 |
| `linear_gemv_tiled` | – | row-major | 1 | 1 | 1 | n/a | – | `decode_state` | 0 |
| `linear_gemm_mma`, `linear_gemm_mma_regpipe` (prefill, simdgroup) | – | – | 1 | 1 | – | 3 | – | – | 0 |
| `linear_gemv_fast1`…`fast8` (8) | 1–8 | row-major | 1 | 1 | 1 | n/a | – | `decode_state` (runtime CB) | 0 |

### 2b. `linear_repacked.metal` — 27 kernels (all `k3` unless noted, all col-major)

| kernel | rate | cols | SK | unr | ring | alu | decode | loads | in the (cols × SK) grid? |
|---|--:|--:|--:|--:|--:|--:|---|--:|---|
| `…_k3_colmajor` | 3 | 1 | 1 | 1 | 2 | 64 | runtime CB | 8 | 1c/SK1 (64) |
| `…_k3_colmajor_simd` | 3 | 1 | 1 | 1 | 2 | 64 | runtime CB | 8 | no (different family) |
| `…_k3_colmajor_stream` | 3 | 1 | 1 | 1 | 2 | 64 | runtime CB | 8 | no (different family) |
| `…_k3_colmajor_u4` | 3 | 1 | 1 | 4 | 2 | 64 | `decode_state_mcg` | 2 | no (different family) |
| `…_k3_colmajor_sk4` | 3 | 1 | 4 | 2 | 2 | 64 | runtime CB | 4 | 1c/SK4 (64) |
| `…_k3_colmajor_contig4` | 3 | 1 | 1 | 4 | 2 | 32 | `extract_v_mcg_32` | 2 | no (different family) |
| `…_k3_colmajor_contig4_sk2` | 3 | 1 | 2 | 4 | 2 | **64** | `decode_state_mcg` | 8 | no (different family) |
| `…_k3_colmajor_contig4_sk4` | 3 | 1 | 4 | 4 | 2 | **64** | `decode_state_mcg` | 8 | no (different family) |
| `…_k3_colmajor_1col` | 3 | 1 | 1 | 2 | 2 | 32 | `extract_v_mcg_32` | 4 | 1c/SK1 (32) |
| `…_k3_colmajor_1col_sk2` | 3 | 1 | 2 | 2 | 2 | 32 | `extract_v_mcg_32` | 4 | 1c/SK2 (32) |
| `…_k3_colmajor_2col` | 3 | 2 | 1 | 2 | 2 | 32 | `extract_v_mcg_32` | 4 | 2c/SK1 (32) |
| **`…_k3_colmajor_2col_sk2`** | 3 | 2 | 2 | 2 | 2 | 32 | `extract_v_mcg_32` | 8 | **2c/SK2 (32) — LIVE (o/down)** |
| `…_k3_colmajor_2col_sk4` | 3 | 2 | 4 | 2 | 2 | 32 | `extract_v_mcg_32` | 8 | 2c/SK4 (32) |
| `…_k3_colmajor_4col` | 3 | 4 | 1 | 4 | 2 | 32 | `extract_v_mcg_32` | 2 | 4c/SK1 (32) |
| `…_k3_fused2` | 3 | 2 | 1 | 4 | 2 | 32 | `extract_v_mcg_32` | 8 | 2c/SK1 (32) |
| **`…_k3_fused2_2col`** | 3 | 2 | 1 | 2 | 2 | 32 | `extract_v_mcg_32` | 8 | **2c/SK1 (32) — LIVE (gate/up, q/k)** |
| `…_k3_fused2_4col` | 3 | 4 | 1 | 2 | 2 | 32 | `extract_v_mcg_32` | 2 | 4c/SK1 (32) — **reachable, never run** |
| `…_k3_fused2_co` | 3 | 2 | 1 | 2 | 2 | 32 | `extract_v_mcg_32` | **4** | dead (arm order) |
| `…_k3_fused2_co4` | 3 | 4 | 1 | 4 | 2 | 32 | `extract_v_mcg_32` | 2 | dead (arm order) |
| `…_k3_fused2_sk2` | 3 | 2 | 2 | 4 | 2 | **64** | `decode_state_mcg` | 8 | 2c/SK2 (64) |
| `…_k3_fused3` | 3 | 1 | 1 | 4 | 2 | 32 | `extract_v_mcg_32` | 2 | 1c/SK1 (32) |
| **`…_k3_fused3_2col`** | 3 | 2 | 1 | 2 | 2 | 32 | `extract_v_mcg_32` | 8 | **2c/SK1 (32) — LIVE (q,k,v)** |
| `…_k3_fused3_sk2` | 3 | 1 | 2 | 4 | 2 | **64** | `decode_state_mcg` | 8 | 1c/SK2 (64) |
| `gemv_reduce_sk2`, `gemv_reduce_sk4` | – | 1 | 2/4 | 1 | n/a | – | – | 0 | no dispatch site at all |
| `…_k4_colmajor_contig4_bench` | 4 | 1 | 1 | 4 | 2 | 64 | `decode_state_mcg` | 8 | bench only |
| `…_k6_colmajor_contig4_bench` | 6 | 1 | 1 | 4 | 3 | 64 | `decode_state_mcg` | 12 | bench only |

### 2c. Totals

| family | kernels | rate-specialised | produce a valid output | comment |
|---|--:|--:|--:|---|
| `epilogue` (finalize / transform) | 21 | 0 | 21 | all live; `…_sk2`/`_sk4` variants are live *only* under split-K flags |
| `exl3` (`linear_gemv_exl3_k*`) | 56 | 56 | 56 | **32 + 24 = 56; only 48 of them are built (the NR0 loop builds 24, the slot loop 48 with 16 names built twice)** |
| `fast` (`linear_gemv_fast*`) | 8 | 0 | 8 | row-major, runtime codebook; reached only from `gemv_buffer` |
| `ishizuki` (row-major 8 + `_w2` 4 + `_nod` 8 + repacked 25) | 45 | 45 | 35 | `_nod` = 8 deliberately-wrong probes |
| `other` (`linear_gemv`, `gemv_tiled`, `input_rows`, `gemm_mma*`, `finalize_rows`, `zero_raw`) | 6 | 0 | 6 | prefill + CPU reference |
| `reduce` (`gemv_reduce_sk{2,4}`) | 2 | 0 | 0 | explicitly "never dispatched" |
| **total** | **138** | 107 | 126 | 517,332 + 157,624 = **674,956 B of Metal source**, one `new_library_with_source` |

**A note on the `_w2` / `_nod` / `fast` classification, since three of the four
groups above are dead and it is worth saying why in one place.**
`linear_gemv_ishizuki_k{1..8}` are the rate-slot owners
(`ishizuki_gemv_pipelines[bits]`), reached from `ishizuki_gemv_pipeline` — and
for this checkpoint that function is entered **only by `lm_head`**, whose rate
is 6. So the rate-3 slot (`linear_gemv_ishizuki_k3`, the row-major 3 bpw form)
is dead too: a colmajor projection would have to miss the `_2col` arm to reach
it, and none does. `_w2` (4) is the two-word row-major form selected by
`ishizuki_gemv_pipeline` when `!KAPLUMBA_ISHIZUKI_3WORD`; only `_k1_w2`.._k4_w2
exist, so at rate 6 the lookup misses and falls to the 3-word `_k6`. `_nod` (8)
is deliberately-wrong output (`nod_stub`) kept as a bandwidth probe. `fast` (8)
is reached, but only from `gemv_buffer` (`linear.rs:884`, `:2207`) — the
readback/verification forward, not the GPU decode loop, for the same reason the
`_mcg` EXL3 family is (`encode_gemv_raw` returns at its `_2col` arm before
`decode_gemv_pipeline` is consulted; asserted in the probe). So all 8 are built
and none can run in a decode step.

---

## 3. Reachability, in three buckets (the part the brief asked for)

Derived by binding every `linear_gemv_*` literal **and** every
`format!("linear_gemv_ishizuki_k{bits}…")` template in `linear.rs` and
`model.rs` to its enclosing `fn`, and collecting the `env_flag("…")` literals
inside that `fn`.

**A. One env flag away from the production decode path (12 kernels, of which
exactly one is on the 32-bit shift chain):**

| kernel | flag | cols | SK | alu | loads |
|---|---|--:|--:|--:|--:|
| `…_fused2_4col` | `KAPLUMBA_FUSED2_4COL` | 4 | 1 | 32 | 2 |
| `…_fused2_co` | `KAPLUMBA_FUSED2_CO4` / `KAPLUMBA_NO_FUSED2_2COL` | 2 | 1 | 32 | 4 |
| `…_fused2_co4` | `KAPLUMBA_FUSED2_CO4` | 4 | 1 | 32 | 2 |
| `…_fused2` | any of the above | 2 | 1 | 32 | 8 |
| `…_fused3` | `KAPLUMBA_NO_FUSED3_2COL` | 1 | 1 | 32 | 2 |
| `…_fused2_sk2` | `KAPLUMBA_USE_SK` | 2 | 2 | **64** | 8 |
| `…_fused3_sk2` | `KAPLUMBA_USE_SK` / `KAPLUMBA_QKV_SK2` | 1 | 2 | **64** | 8 |
| `…_colmajor_2col` | `KAPLUMBA_4COL` | 2 | 1 | 32 | 4 |
| `…_colmajor_4col` | `KAPLUMBA_4COL` | 4 | 1 | 32 | 2 |
| `…_colmajor_1col_sk2` | `KAPLUMBA_SK2_1COL` / `KAPLUMBA_SK2_CONTIG4` | 1 | 2 | 32 | 4 |
| `…_colmajor_contig4_sk2` | `KAPLUMBA_SK2_CONTIG4` | 1 | 2 | **64** | 8 |
| `…_colmajor_contig4_sk4` | `KAPLUMBA_USE_SK` | 1 | 4 | **64** | 8 |

**B. No production dispatch reference at all — 115 of 138.** Built at load,
never named by anything that runs. Includes all 56 `exl3`, all 8 `fast`, all 8
`_nod`, both `gemv_reduce_*`, both `_bench`.

**One clarification about "unreachable", because it is load-bearing and easy to
overstate.** `linear_gemv_exl3_k*` and `linear_gemv_fast*` are *not* dead code
in the crate sense — they are dispatched by `gemv` / `gemv_buffer`
(`linear.rs:863`, `:884`, `:2207`), which `Model::run_linear` calls from the
**readback verification forward** (`model.rs:1165-1234`, `:2636-2718`). What
they cannot do is run inside a GPU decode step, because `encode_gemv_raw`
returns at its `_2col` arm (`linear.rs:2129`) before `decode_gemv_pipeline` is
consulted (`:2178`), and every layer projection in this checkpoint satisfies
that arm's guard. So: **unreachable in production decode, reachable in the test
and reference paths.** `KAPLUMBA_DECODE_NR0` and the 24 `_mcg_r*` kernels ride
the same arm and are dead in *both* senses — `encode_gemv_named`, their only
entry point, has no caller in `crates/engine/src` at all.

**C. Reachable only through the rate-slot selector `ishizuki_gemv_pipeline`,
which for this checkpoint is entered *only* by `lm_head`** (`perm_mode == 0`),
so only the rate-6 unsuffixed kernel survives:
`_k1_w2`, `_k2_w2`, `_k3_w2`, `_k4_w2`, `_k3_colmajor`, `_k3_colmajor_contig4`,
`linear_gemv_tiled`.

**So: of 138, four dispatch; 12 more are one flag away; 115 are constructed and
never named; 7 sit behind a selector that this checkpoint's shapes route around.**
Every prior claim in the brief is confirmed and, where it was an inference, now
has a line number.

**One asymmetry worth recording.** The twelve flag-gated kernels are *not*
equally trustworthy. `…_colmajor_2col`, `…_colmajor_1col_sk2`,
`…_colmajor_2col_sk4` and `…_fused2_4col` are on the 32-bit chain, so they are
plausibly competitive. `…_fused2_sk2`, `…_fused3_sk2`, `…_colmajor_contig4_sk2`,
`…_colmajor_contig4_sk4` and `…_colmajor_sk4` are on the 64-bit chain, i.e.
they predate §22, and §22 measured that rewrite at **−22.8% on qkv_proj**
specifically. A flag that swaps one of those in is not testing what its name
says.

---

## 4. Q3, answered: the `fused2_co` hoist gap is real and harmless

The metric that matters is how many weight-load **bursts** are issued before the
first decode consumes one, i.e. the hoisted `const uint w…` declaration groups
that precede the first `extract_v_mcg_32`:

```text
linear_repacked.metal:506-509   fused2_2col    w0_0_*, w1_0_*, w0_1_*, w1_1_*   4 bursts = 8 words
linear_repacked.metal:829-830   fused2_co      w0_a*,  w1_a*                    2 bursts = 4 words
```

Both arms do identical work per trip (2 columns × 2 K sub-steps, 2-word window,
`extract_v_mcg_32`, same accumulator and reduction tree). `_co` therefore holds
half the memory-level parallelism for the same work — the T4.3 8→4→8 hoist was
never carried into it. **And `_co` is dead**: `encode_gemv_fused2_mlp` tests
`!NO_FUSED2_2COL && gate%32==0` *before* `gate == up`, so `_2col` always wins
for 21504. The live gate/up kernel has **8**.

The live set is hoisted correctly everywhere:

| live kernel | bursts | words |
|---|--:|--:|
| `fused3_2col` | 4 | 8 |
| `fused2_2col` | 4 | 8 |
| `colmajor_2col_sk2` | 4 | 8 |
| `ishizuki_k6` (head) | 2 | 6 (3-word window at KA=6) |

**The correct reading of the anomaly is inverted from the brief's framing.** It
is not "the live gate/up kernel is under-hoisted"; it is "the live gate/up
kernel is the one that got the hoist, and the dead one did not". The only live
risk it carries is that `KAPLUMBA_NO_FUSED2_2COL=1` would silently select a
half-hoisted kernel — which is a reason to leave that flag unset, not a
performance finding.

---

## 5. Q4, answered: `fused3_sk2` is reachable, correct, and predicted to lose

**Correctness.** The K-split arithmetic is sound for every shape:
`k_per_split = ((tiles_k/SK)/32)*32 = ((336/2)/32)*32 = 160`, `k_end` for the
last split is `tiles_k = 336`, so split 0 covers `[0,160)` and split 1 covers
`[160,336)` — the union is exactly `[0,336)` with no gap and no overlap
(`linear_repacked.metal:1639-1641`). The reduction pairing matches:
`fused3_sk2` writes `y[(sk*N_curr) + tn*16 + lid]` and
`linear_finalize_simd3_sk2` reads `raw[base] + raw[local_n + base]`. Both
kernels exist and both are built.

**It has no bit-exactness coverage.** `live_kernel_bit_exact.rs` covers
`colmajor`, `colmajor_contig4`, `contig4_sk2`, `colmajor_2col`, `colmajor_2col_sk2`
— not `fused3_sk2`, not `fused2_sk2`, not `fused3_2col`. Enabling the flag
today would put an unverified kernel on 50 of 60 layers.

**Two dead guards in it, both evidence it has never been run.**
`fused3_sk2:1614` reads `if (sk >= SK) return;` immediately after
`const uint sk = tgid / total_tiles;`. `tgid` is bounded by the dispatched grid,
`(tiles_a + tiles_b + tiles_c) * 2` (`linear.rs:2736`), so `sk < 2 == SK` always
and the guard cannot fire. The sibling `if (tn >= tiles_c) return;` at `:1636`
is unreachable for the same reason. Neither is wrong — they are the residue of a
kernel written to be robust against a grid it was never actually given.

**Three static reasons to expect it to lose, in order of size:**

1. **It is on the pre-§22 64-bit shift chain.** `fused3_sk2` builds
   `const ulong A0 = ((w0a << 32) | w0b) << sh0`. §22 replaced exactly this
   with `extract_v_mcg_32` and measured **qkv_proj 11.28 → 8.71 ms (−22.8%)**,
   `gate_up_gelu` 27.44 → 25.84 ms, `down_proj` 14.01 → 13.22 ms. §22's note
   lists the three kernels it converted — `contig4`, `fused2`, `fused3` — and
   **every split-K kernel in the tree was left behind**: `fused2_sk2`,
   `fused3_sk2`, `contig4_sk2`, `contig4_sk4`, `colmajor_sk4`. So
   `KAPLUMBA_QKV_SK2=1` is not a split-K A/B. It is a
   *64-bit-vs-32-bit* A/B wearing a split-K flag, plus a 2-col→1-col change
   (the census above: `fused3_2col` = 2 cols, `fused3_sk2` = 1 col). The
   predicted sign of the confounded term is −20% or worse; the most split-K
   could possibly buy is the ~+5% the 120-cell sweep saw on its single best
   cell.
2. **The repo's own grid rule says the split is not needed.**
   `fused3_2col` already launches **512** threadgroups. Split-K was introduced
   exactly where the 2-col grid would have been **168**: `colmajor_2col` at
   N=5376 is 168 TGs, which is why `o_proj` and `down_proj` take `use_sk2` and
   land on 336. `fused3_2col` at 512 and `fused2_2col` at 1344 are both already
   inside (indeed far above) the 168–672 bracket the repo measured three ways.
3. **§21's 120-cell sweep found nothing there.** 10 shapes × 4 K-splits × 3
   threadgroup widths, median of 11 interleaved rounds, 4 GB working set: the
   engine's SK=1/TG=256 is the best cell on 7 of 10 shapes, and on the three
   QKV shapes specifically `sliding/q`, `sliding/k`, `sliding/v` all report gap
   **+0.0%**. `SK=8` is the worst or second-worst cell on every shape
   (−9.0% to −36.7%). (Caveat that cuts the other way: that sweep used
   `colmajor_sweep.metal`, not the production kernels, and the harness's
   variant ranking is known to be inverted — which is exactly why this is
   listed as a *reason not to spend the GPU time*, not as a measurement.)

**Verdict: do not run `KAPLUMBA_QKV_SK2=1`.** If a split-K A/B on QKV is ever
wanted, the kernel to write is `fused3_2col_sk2` — a mechanical composition of
`fused3_2col` (which has the grid/hoist/window shape) with `colmajor_2col_sk2`
(which is the one split-K kernel in the tree that is *on* the 32-bit chain) —
and the flag to add is a name suffix in `encode_gemv_fused3`, not
`KAPLUMBA_QKV_SK2`.

**The coverage matrix, from the census** (`--` = no kernel exists):

```text
single-matrix (o_proj/down)   SK=1/32:  1c  2c  4c     SK=2/32:  1c  2c  --
                             SK=4/32:  --  2c  --     SK=1,2,4/64: 1c only
fused2 (q/k full, gate+up)   SK=1/32:  1c  2c  4c     SK=2/32:  --  --  --
                             SK=2/64:  1c  --  --
fused3 (q/k/v sliding)       SK=1/32:  1c  2c  --     SK=2/32:  --  --  --
                             SK=2/64:  1c  --  --
```

The single-matrix family is the only one with a filled 2c×SK2 cell — and that
cell is the live `o_proj`/`down_proj` kernel. **The fused families have no
split-K on the 32-bit chain at all.** That is the gap, stated precisely.

---

## 6. Q5, answered: how much of the 138 is dead weight, and what does it cost

**In milliseconds: nothing.** `LinearKernels::new()` compiles
`combined_source()` — both `.metal` files, 674,956 B, **one**
`new_library_with_source` — and eagerly builds every pipeline through
`pipeline_audit::build`. Measured across the whole crate: **173 builds, 157
distinct names, 10.3 ms, `late_builds=0`** across a 614-token prefill and 48
greedy steps. 10.3 ms against a ~10.8 s load is 0.1%, and against a 62 ms step it
is zero. **Dead kernels in this library are not a throughput problem and should
not be argued about as one.**

The 173-vs-157 gap of 16 is fully explained by the EXL3 slot loop
(`linear.rs:484-500`): it iterates 3 codebooks × 2 alignments × 8 rates = 48
states, but `Exl3Codebook::Plain` and `Exl3Codebook::Mul1` both use the **empty**
family suffix, so 16 of those 48 builds carry a name that was already built.
That is also why `no _mul1 kernels exist` is a *naming* defect as well as a
reachability one: the mul1 family is not missing from the library, it is missing
from the *names*, so `linear.rs:485` builds the same 16 kernels twice and the
mul1 dispatch slot is indistinguishable from the plain one.

**In review surface: 134 of 138.** That is the real cost, and it is already
being paid — this document exists because the inventory had never been
decomposed, and two ranked actions in `WHERE-THIS-STANDS.md` turned out to point
at unreachable families (§0). A library where 97% of the surface has never run
will keep producing that failure mode. If someone wants to spend a day here, the
highest-value deletion is the **24 NR0 kernels**: one knob,
`KAPLUMBA_DECODE_NR0`, **zero reachable effect**, 24 generated kernels compiled
on every load.

---

## 7. The live-kernel review, one cell at a time

Ranked by share of the token's bytes. "B/TG" is bytes each threadgroup reads;
§16.3 measured bandwidth monotone in it (199.3 GB/s at 129,024 B/TG against
~175 at interleaved streams), so it is the one axis worth arguing about.

| cell | kernel | GB/tok | share | TGs | B/TG | is there a sibling that changes an unpinned axis? |
|---|---|--:|--:|--:|--:|---|
| gate+up (60) | `fused2_2col` | 5.202 | **43.2%** | 1344 | 64,512 | **`fused2_4col`: 672 TGs, 129,024 B/TG, `KAPLUMBA_FUSED2_4COL=1`, never run.** |
| down_proj (60) | `colmajor_2col_sk2` | 2.601 | 21.6% | 336 | **129,024** | no — this *is* the 129,024 cell the law was measured on |
| q,k,v sliding (50) | `fused3_2col` | 1.652 | 13.7% | 512 | 64,512 | `fused3` (1col, 2 TGs) via `KAPLUMBA_NO_FUSED3_2COL`; `fused3_sk2` is on the 64-bit chain (§5). No 4col, no 32-bit split-K. |
| lm_head (1) | `ishizuki_k6` | 1.057 | 8.8% | 16,384 | 64,512 | `k6_colmajor_contig4_bench` — but the head is `perm_mode 0`, so **all 25 colmajor kernels are structurally out of reach**. |
| o_proj (60) | `colmajor_2col_sk2` | 1.156 | 9.6% | 336 | 49,152 (sliding) / 98,304 (full) | `colmajor_2col` (168 TGs, 98,304/196,608 B/TG) via `KAPLUMBA_O_NOSK`; `colmajor_1col_sk2` (672 TGs, 24,576 B/TG) via `KAPLUMBA_SK2_1COL` |
| q,k full (10) | `fused2_2col` | 0.372 | 3.1% | 576 | 64,512 | same kernel as gate/up; nothing new |

**lm_head deserves its own note, because it looks bad and is not.** The live
`linear_gemv_ishizuki_k6` calls `decode_state(s, CB)` with a **runtime**
codebook and carries an `if (r <= 64u) ? A0 : B0` **per code**, inside an
8-iteration unrolled loop. The `linear.metal:328-331` header says that pattern
is the expensive one: "Removing the decode OPS on top of the specialization
makes it SLOWER (134.3 < 163.0): the arithmetic is free and the BRANCH is what
serialises the eight values per lane." The MCG folding that fixed it
(2.61×, 62.4 → 163.0 GB/s) exists **only** in the unreachable
`linear_gemv_exl3_k*_mcg` family; there is no `linear_gemv_ishizuki_k6_mcg`.
Two arithmetic facts, both asserted in the probe:

* At KA = 6 the highest bit any lane reads is `7*6 + 16 = 58 ≤ 64`, so **every
  read lands in `A0` and `B0` is dead code** — 2 shifts, an `or` and a select
  per tile, removable, bit-exact.
* `sh0 = (48*lane + 1526) % 32 ∈ {6, 22}`, so bit `63 + 22 = 85` of the 96-bit
  window is read and `w0`/`w1` only cover bits 0..63 — **`w2` is still
  load-bearing**, which is why `TWO_WORD_SAFE_RATES = (1,2,3,4)` and the head
  cannot use `extract_v_mcg_32`.

**And it still does not matter, because the head is at the DRAM wall.** Two
independent stream harnesses agree on 253 GB/s to 0.1% (M05), and production
head time 4.18 ms / 1.057 GB = 253 GB/s. At the wall, ALU is free. The honest
verdict on `lm_head` is therefore: *code is measurably wasteful, kernel is
measurably at the roofline, do not touch it without a repack* — and a repack of
the head to col-major is an **R1 (bytes) play**, not an R2 (kernel) play, exactly
as K03 already concluded.

---

## 8. The verdict, and the one experiment

**All four live kernels are the right choice for their shape.** The arithmetic:

* **Grid.** 512 / 576 / 1344 / 336 TGs. The repo's own measured resident
  bracket is 168–672 TGs; split-K exists in the tree *because* the 2-col grid
  for a 5376-wide output is 168, and both projections with that shape already
  took it. Nothing else is grid-starved. Note the direction of the argument: the
  two projections that *do* take split-K are exactly the two whose 2-col grid
  would be 168. `fused3_2col` at 512 and `fused2_2col` at 1344 are past the
  bracket already, so the split-K family has nothing left to buy here.
* **Ring window.** All three 3-bpw live kernels read exactly 2 words per lane
  per tile, which is the proven floor (the window spans 37 bits > 32).
  Nothing to trim.
* **Shift chain.** All three use `extract_v_mcg_32` (32-bit funnel shift).
  Nothing to convert.
* **Hoist.** 8 loads / 4 bursts on all three. Nothing to hoist.
* **Reduction.** K04 measured it at 0.664% of GEMV issue slots, 0.40 ms/token.
* **ALU.** The `_mcg` folding is 2.61× — and it is already applied to every live
  kernel (`extract_v_mcg_32` calls `decode_state_mcg`). Nothing left on that
  axis either. The one live kernel that *isn't* MCG-folded is the head, and §7
  explains why that is still correct.
* **Bytes.** 12.039 GB is fixed by the checkpoint; no kernel changes it.
* **Corroboration from an independent axis.** P19 (`WHERE-THIS-STANDS.md`)
  establishes that decode time tracks `lane_words(rate)`, not `rate` — bytes and
  time move together, so byte-shaving buys nothing. The live kernels read
  **2 ring words per lane per tile at 3 bpw**, which is `lane_words(3)` and the
  minimum for the 37-bit window. There is no axis left on this axis either.

One caveat on that clean negative, stated rather than buried: it is an argument
from the four cells' *shapes*, and the one place where a shape argument and a
measured result could disagree is **o_proj sliding at 49,152 B/TG**, which is
below every other cell. That is §8's runner-up.

**The one live kernel I would not call settled is `fused2_2col` on gate/up**,
because it is 43.2% of the token's bytes and its immediate sibling
`fused2_4col` moves the *only* axis §16.3 measured as monotone — bytes per
threadgroup, 64,512 → 129,024, the same value `down_proj` runs at today and the
same value §16.3 measured **199.3 GB/s** against, at 64,512.

### The experiment

Two steps, in this order. **Do not skip step 1** — it is the difference between
a measurement and a coin flip, and it costs an afternoon rather than 6.5 hours.

**Step 1 — hoist, then pre-check on the kernel's own GPU time.**

```text
(1a) port T4.3's hoist into linear_gemv_ishizuki_k3_fused2_4col
    - raise `t0_0[j0], t0_0[j1], … t1_3[j1]` into eight `const uint w…`
      declarations above the first `extract_v_mcg_32`, exactly as
      linear_repacked.metal:2477-2480 does for colmajor_2col_sk2.
    - bit-exact: each column's accumulation order and reduction tree are
      untouched. Verify with live_kernel_bit_exact.rs, extended to 4col.
(1b) KAPLUMBA_GPU_PROFILE=1 KAPLUMBA_GPU_PROFILE_DECODE=1, ~30 interleaved
    rounds, read the `gate_up_gelu` own_ms SHARE (not wall -- kernel mode
    inflates wall via its own wait_until_completed, decode_profile.rs:1-55).
    If the share does not move by more than its own spread, stop.
```

(1b) is the whole point: `gate_up_gelu` is ~28.6 ms of a 62.3 ms step, so a 5%
effect on the *kernel* is a 5% signal on the number that matters, rather than
1.9% on a whole-step tok/s that carries the sampler, the dispatch gaps and the
host encode. Same binary, same production dispatch, same kernel — not a
microbenchmark, so the brief's inverted-ranking warning does not apply.

**Step 2 — only if (1b) moves, the end-to-end A/B.**

```bash
# one binary, two arms, interleaved, on the production greedy path
KAPLUMBA_PROBE_ARMS='A=|B=KAPLUMBA_FUSED2_4COL=1' \
  <the existing interleaved-A/B harness from perf_probe_stats_noise_floor.rs>
```

```bash
# one binary, two arms, interleaved, on the production greedy path
KAPLUMBA_PROBE_ARMS='A=|B=KAPLUMBA_FUSED2_4COL=1' \
  <the existing interleaved-A/B harness from perf_probe_stats_noise_floor.rs>
```

`KAPLUMBA_FUSED2_4COL=1` is the *entire* runtime change step 2 requires — no env
var beyond it, no code change beyond the 1a hoist. The kernel exists
(`linear_repacked.metal:580`), the arm is `linear.rs:2578`, and 21504 % 64 == 0
so the guard passes.

**Step 2 is not new, and the reason it has not been run is already on record.** `WHERE-THIS-STANDS.md:434-452` names `KAPLUMBA_FUSED2_4COL=1` as "the
one zero-code experiment worth naming", prices it (**+0.57 tok/s** if
bytes-per-threadgroup is the driver, `_co4`'s **−0.19** if two-streams-per-SM is;
honest range −0.2…+0.65, median guess +0.3), and correctly declines to run it:
+0.3 on 16.05 is **1.9%**, which needs **>179 pairs ≈ 6.5 hours** at this box's
recorded σ. That analysis is right and I am not going to talk past it.

**What the census adds is that the experiment's *design* is confounded, and
fixing that is what makes it worth re-specifying.** `WHERE-THIS-STANDS.md:437`
frames `fused2_4col` as discriminating "bytes-per-threadgroup" against "two
interleaved streams", using `_co4` as the known negative. The census shows `_co4`
and `_4col` differ in **five** variables, not one:

| | cols | unroll | live 32-bit words | DRAM runs / TG | grid | B/TG |
|---|--:|--:|--:|--:|--:|--:|
| `fused2_2col` (live) | 2 | 2 | 32 | **1** — one contiguous 64,512 B run | 1344 | 64,512 |
| `fused2_co4` (measured −0.19) | 4 | **4** | **72** | **2** — two 64,512 B runs, two buffers | 672 | 129,024 |
| `fused2_4col` (the experiment) | 4 | **2** | **48** | **1** — one contiguous 129,024 B run | 672 | 129,024 |

So `_co4`'s loss cannot be attributed to streams alone: it moved grid, B/TG,
unroll depth, register budget and stream count together. The new kernel is the
clean cell — same grid and same B/TG as `_co4`, one stream, half the unroll, two
thirds of the registers. **That is a better experiment than the one on record,
not a different one**, and it still faces the same 6.5-hour power problem.

**Why the hoist is step 1a and not optional.** In `fused2_4col` the index
expression is *inlined* into each `extract_v_mcg_32(t0_0[j0], t0_0[j1], …)`
call rather than hoisted, so the kernel as written holds **1 burst** where
`_2col` holds 4 — the census counts it at 2 words in flight against `_2col`'s
8. That is a source property, not necessarily a compiled one (the Metal
compiler can hoist across calls, and there is no barrier), but it is a real
confound on a kernel whose entire thesis is memory-level parallelism per
threadgroup, and it is the exact regression T4.3 fixed in
`colmajor_2col_sk2`. **Without 1a the experiment measures the absence of a
hoist, not column tiling, and a loss would be uninterpretable.**

**What makes it a fair test, established from source not from a benchmark:**

* **It is bit-exact in f32, not merely fp16.** `fused2_4col` accumulates each
  of the four columns with the identical sequence of operations as
  `fused2_2col` and reduces with the identical tree (`red[0..3][sg][…]`,
  `simd_shuffle_xor` by 1 and 2, `lid < 64` store, `linear_repacked.metal:744-764`)
  — only *which* four columns a threadgroup owns changes. This is a stronger
  guarantee than the sibling kernels have: `colmajor_contig4` and `contig4_sk2`
  both *reassociate* (the file header, `linear_repacked.metal:30-44`, says so) and
  are only bit-identical after the fp16 rounding in the finalize. `fused2_4col`
  should be bit-identical in the accumulator itself. That has to be *verified*,
  not assumed — FMA contraction is not stable across recompiles, and
  `linear.metal:339-343` records a case where two variants each matched a
  baseline alone and differed when combined. So step 1a requires extending
  `live_kernel_bit_exact.rs` to 4col and **seeing it fail** on a reordered
  reduction before believing it.

**Two facts cut against it, and they are the reason the prior is ~30% win rather
than higher:**

* `_co4` measured **15.46 vs `fused2_2col`'s 15.65 tok/s, −1.2%** (notes §13) —
  but see the table above: that cell also quadrupled the unroll and moved to 72
  live words and two streams, so it is a weak negative for `fused2_4col`
  specifically.
* K03's mechanism table bounds the family: column tiling amortises the `x`
  loads, which are ~6 of ~50 instructions per lane per tile, so **C=4 can
  remove at most ~6% of the instruction stream**. If the 3-bpw kernel is
  genuinely issue-bound (WHERE-THIS-STANDS' open question — the
  `probe_dram_roofline` discriminator printed *both* verdicts on a calibration
  constant and now prints `WITHHELD`), then the expected value is ≈ 0 ± 3%. If
  it is bandwidth-bound at the *threadgroup* level rather than the chip level,
  B/TG is the lever and +0.57 is on the table. **The experiment's real value is
  that it settles that open question in production**, which is worth more than
  the 0.3 tok/s either way.

**Win criterion, and the power problem stated honestly.** The box has been
measured at 3.7, 3.9, 4.3, 4.8, 6.0 and 6.4 tok/s from the *same binary* minutes
apart (`AGENTS.md` §10.5), so a single reading is not a result. The prior
analysis in `WHERE-THIS-STANDS.md:450` puts the required sample at **>179
interleaved pairs ≈ 6.5 hours**, because +0.3 tok/s is 1.9%. Require **all
three** of:

1. median `B` − median `A` **> the noise floor measured by the same harness on
   the same day** (run `perf_probe_stats_noise_floor` first — that is what it is
   for), not > an absolute number;
2. the sign stable across rounds, i.e. the two arms' spreads overlapping by less
   than the gap — not carried by one lucky pair;
3. the five reference prompts byte-identical (this one is cheap and is not a
   statistical matter at all).

That is why step 1 exists: `gate_up_gelu` is ~28.6 ms of a 62.3 ms step, so a
5% effect on the *kernel* is a 5% signal on the number that actually has to
move, rather than 1.9% on a whole-step tok/s that also carries the sampler, the
1617 dispatch gaps and the host encode.

**What counts as a negative, which is a fine result:** `B` within noise of `A`
confirms, on the largest single cell in the model and by the only axis with a
measured monotone law, that **column tiling is exhausted** — and closes
`fused2_4col` as K03 left it open, with production numbers instead of a
mechanism table. A `B` worse by more than the noise floor closes 4-column
tiling on gate/up and, by the shared grid arithmetic, `fused2_co4` and
`colmajor_4col` with it.

### Runner-up, for whoever takes it

`KAPLUMBA_O_NOSK=1` — o_proj only (`model.rs:1974`), **168 TGs / 98,304 B/TG
instead of 336 / 49,152**, 9.6% of the token's bytes, flag already exists, and
**no isolated result for it is recorded anywhere in `spec/`**
(`KAPLUMBA_NO_SK2` was measured at −0.58 tok/s but switches o_proj *and*
down_proj together, which is why it cannot be attributed). It is a one-token
change to a running A/B and it is the only unmeasured cell on the split-K axis
that the repo's own rule points at. Do it second, not first.

Note the trap: `KAPLUMBA_SK2_1COL` looks like the same knob from the other side
(672 TGs, 24,576 B/TG) and **also switches `down_proj`**
(`linear.rs:2240` is read inside `encode_gemv_raw_sk2`, which both projections
call). The `WHERE-THIS-STANDS.md` row that proposes `KAPLUMBA_O_SK2_1COL`
confuses the two: that flag does not exist, the real one is global, and running
it would move 31% of the token's bytes for a change aimed at 9.6%. If someone
wants the isolated version it needs a new `KAPLUMBA_O_SK2_1COL` plumbed through
`encode_gemv_raw_sk2` — a small `model.rs` change, not a one-token env var, and
under AGENTS.md §2 a spec before the edit.

### What this census settles that nobody had written down

Three results that are not about speed and cost nothing to keep:

1. **`KAPLUMBA_DECODE_NR0` has zero live effect, confirmed structurally.** Its
   24 kernels are reachable only from `exl3_gemv_pipeline`, which
   `encode_gemv_raw` never reaches for this checkpoint (§3B). The prior note
   says "no reachable path"; this says *which* line of code makes that true
   (`linear.rs:2129` returns first) and asserts it.
2. **The `_mcg` EXL3 family is not dead code, it is off the decode path.**
   `gemv`/`gemv_buffer` still dispatch it, via `Model::run_linear`, for the
   readback verification forward. "Unreachable" was overstated; "unreachable in
   a decode step" is what the evidence supports.
3. **`linear_gemv_exl3_k3` and `linear_gemv_fast3` — the two kernels the
   `_mcg`/NR0 work was done for — cannot run at 3 bpw in this checkpoint at
   all.** The 3-bpw rate slot (`ishizuki_gemv_pipelines[3]`) holds
   `linear_gemv_ishizuki_k3`, which is *row-major*; a colmajor 3 bpw tensor
   takes the `_2col` arm before that slot is consulted. So the row-major 3-bpw
   kernel is unreachable *and* wrong-shaped for the checkpoint, which is why it
   has sat there looking like a fallback.

---

## 9. What this census cannot tell you

Stated so the next reader does not over-read it:

* **Nothing here is a timing.** Every number is source arithmetic. The brief's
  warning applies with full force: this repo's microbenchmark variant rankings
  were measured to be *inverted* vs production, so a census can say which cells
  exist and which kernels are structurally comparable; it cannot say which is
  faster. Where I lean on a measurement (§7's B/TG law, §8's `_co4` −1.2%, §5's
  §21 sweep) I name it and attribute it.
* **`loads_in_flight` is a source-level count, not a compiled one.** It measures
  hoisted declaration groups before the first extractor call. The Metal
  compiler may hoist or sink further. `perf_probe_kloop_loads_in_flight.rs`
  makes the same measurement and reaches 8 for the three live kernels, so at
  least the ordering (`_2col` 8 > `_co` 4) is corroborated by an independent
  implementation of the same idea — not by an independent *fact*.
* **`fused2_4col`'s 48 live words is transcribed from K03, not re-derived here.**
  `probe_coltile_registers.rs` derives it by scanning the Metal source and is
  the authority; I cite it rather than duplicating the scanner.
* **The census says nothing about prefill.** `linear_gemm_mma` /
  `linear_gemm_mma_regpipe` / `linear_input_rows` are 3 of the 138 and are
  multi-query kernels; they are counted and classified here because they are in
  the same library, and nothing more should be read into their rows.

---

## 10. Sabotage discipline (AGENTS.md §3.4)

`the_census_detectors_can_fail` feeds mutated bodies to the same detectors and
asserts each moves. Observed values:

| # | mutation | observed |
|---|---|---|
| 1 | delete both hoisted `w*_1_*` groups from `fused2_2col` | loads 8 → 4 (the pre-T4.3 `colmajor_2col` shape) |
| 2 | hoist `_co`'s col-B loads above col-A's first extract | loads 4 → 8 (this *is* the T4.3 hoist) |
| 3 | `const ulong A` → `uint A` in `fused3_sk2` | alu 64 → 32 |
| 4 | delete `constexpr uint SK = 2u` from `fused3_sk2` | SK 2 → 1 (the detector reads source, not the `_sk2` name suffix) |
| 5 | `red[2]` → `red[4]` in `fused2_2col` | cols 2 → 4 |
| 6 | insert a `j2` read into `fused2_2col` | ring 2 → 3 |
| 7 | classify `gemv_reduce_sk4` | rate `None`, not 4 |
| 8 | rename `fused2_co4` → `fused2_coX` / demote it to `static void` | inventory 27 / 27-with-old-name-gone / 26 |
| 9 | insert the EXL3 call above the `_2col` guard in a copy of `linear.rs` | `exl3_arm_follows_two_col` → false |

**Limitation, stated rather than hidden:** this probe is read-only by
instruction, so the detectors are sabotaged *in memory* rather than on disk. The
one claim with no on-disk negative control is the `138` inventory figure itself;
mutation #8 covers it as far as a string-level probe can.

---

## 11. Reproduce

```bash
cargo test -p kaplumba-engine --test probe_kernsel_census -- --nocapture
```

No Metal device is created, no checkpoint is opened, no `gpuq` is involved.
The full 138-row census, the reachability buckets, the coverage matrix and the
byte/grid table are all printed.