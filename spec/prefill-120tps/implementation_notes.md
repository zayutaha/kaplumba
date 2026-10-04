# Implementation Notes — prefill ≥120 tok/s

Running log. Appended as work proceeds, not at the end. Factual and dense.

---

## T0 — BASELINE MEASURED (2026-10-04), branch `mtp-speculation` @ 2091654

Binary: `target/release/kaplumba` built from `cargo build --release -p kaplumba-cli`.
All runs `KAPLUMBA_MAX_IN_FLIGHT=1` (nondeterminism hazard, notes §4).
Model `~/.kaplumba/models/Gxma/Gxma`. Harness: `spec/prefill-120tps/measure.sh`.

### Headline: the spec's baseline was too pessimistic at short context.

| ctx | prompt_tokens | prefill tok/s | ttft ms | spec estimate |
|---|---|---|---|---|
| 614 | 614 | **94.50 / 94.66 / 94.66** (median 94.66) | 6487 | ~78 |
| 8k | 7814 | **74.70 / 74.35** (median 74.52) | ~104 900 | ~75 |
| 16k | 15974 | **60.75** | 262 940 | not measured |
| 24k | 23954 | **51.12** | 468 624 | not measured |

24k **already runs** — `HOST_TOKEN_SLOTS=8192` does *not* block prefill at 24k
(T5's premise is partly wrong; see T5 section below).

### Per-kernel attribution — `KAPLUMBA_GPU_PROFILE=1`

The profiler prints **two separate reports** per run: first `tokens=2 chunks=1`
(a **load-time warmup forward**, emitted before `loaded in …`) and then
`tokens=614` / `tokens=7814` (the real prefill). Only the second is the
measurement. Confirmed by grepping `tokens=`; the warmup block appears
identically for different prompts (gate_proj 279.2 vs 278.8 ms).

**614 rows, 1 chunk, wall span 6544.5 ms, sum own 6490.3 ms, gap 38.7 ms:**

| kernel | total ms | share |
|---|---|---|
| down_proj | 1438.2 | |
| gate_proj | 1411.7 | |
| up_proj | 1410.1 | |
| o_proj | 645.1 | |
| q_proj | 639.2 | |
| k_proj | 263.5 | |
| v_proj | 238.4 | |
| **GEMM subtotal** | **6046.2** | **93.2%** |
| attention | 356.2 | 5.5% |
| norms+rope+gelu+cache+head | 87.7 | 1.3% |

**7814 rows, 8 chunks, wall span 108 984 ms, sum own 103 475 ms, gap 3686 ms:**

| kernel | total ms | share |
|---|---|---|
| **GEMM subtotal** | **75 061** | **72.5%** |
| attention | 27 353 | **26.4%** |
| norms+rope+gelu+cache | 1044 | 1.0% |
| inter-boundary gap (host) | 3686 | 3.4% of wall |

### The plan's attribution table is WRONG — this is the load-bearing T0 finding

`plan.md` §"Where time goes (estimated at ~78 tok/s)" claims GEMM ~58%,
attention ~5.4%, **"norms + RoPE + activation + dispatch ~4.6 ms/token ~36%"**.

Measured: the third row is **1.3%**, not 36%. There is no 4.6 ms/token of
elementwise/dispatch slack to harvest — it does not exist. GEMM is 93.2% at 614.

### Run-to-run variance: the host gap is the noisy term

Four 614-token runs gave prefill_tok_s 94.50 / 94.66 / 94.66 / 93.71 / 86.90.
The one 86.90 outlier had `inter-boundary gap = 261 ms` where the others had
38.7 ms. So **`own_ms` is stable and the host/dispatch gap is what swings**
(38 → 261 ms, i.e. up to 3.5% of the span). Always record gap beside tok/s;
a 7% tok/s regression can be pure host gap, not kernel time.

### Derived arithmetic (this governs the whole spec)

- GEMM ms/token: 6046.2/614 = **9.85 ms** @614; 75 061/7814 = **9.61 ms** @8k.
- GEMM TFLOPS now = 5.75e10 / 0.00985 = **5.84 TFLOPS @614**, 5.98 @8k
  (spec's 5.31 baseline was stale — we start at 5.84 = 38.5% of 15.18 peak).
- **AC1 (120 tok/s @614) ⇒ 5117 ms budget** vs 6545 ms wall ⇒ must save **1428 ms**.
- If attention were made **completely free** (0 ms instead of 356), we reach only
  614/(6545−356)/1000 = **99.2 tok/s**. Attention MMA alone cannot reach AC1.
- ⇒ **AC1 and AC4 are the same task.** To hit 120: GEMM must reach ≈7.5 TFLOPS
  (6046 → 4707 ms, −1339 ms) *and* attention must improve. GEMM ≥7.5 TFLOPS is
  therefore not optional — it is the whole short-context story.
- Attention share rises 5.5% (614) → 26.4% (8k), so R2/R5 still matter for AC2
  and for 16k/24k, but never for AC1.

### Attention cost model (from the two profiles, for R2 sizing)

total attention ∝ Σ over layers of keys attended. 50/60 layers are
sliding-window (cap 1024), 10 are full attention. At 614 all layers are below the
cap; at 7814 the sliding layers are capped and the 10 full layers go quadratic.
Measured 356 ms → 27 353 ms = 76.8× for 12.7× more tokens, i.e. between linear
and quadratic as expected. Per-dispatch 5.94 ms @614 → 57.0 ms @8k.

---

## Pre-existing state (2026-10-04)

### Baseline numbers (from spec/engine-performance, NOT re-measured yet)

| metric | value | source |
|---|---|---|
| prefill @ 614 | ~78 tok/s | PREFILL_CHUNK=1024, post-regpipe |
| prefill @ 8k | ~75 tok/s | measured |
| GEMM TFLOPS | 5.31 | 35% of 15.18 peak |
| tensor peak | 15.18 TFLOPS | mma_peak.rs, 16 chains |
| attention @ 614 | ~5.4% of total | profiler |
| attention @ 8k | ~26% of total | estimated |
| decode | ~15 tok/s | user-reported, not independently verified |
| worst GEMM error | 2.16e-3 | vs CPU reference |
| peak RSS @ 614 | 7.81–8.06 GiB | measured |
| HOST_TOKEN_SLOTS | 8192 | model.rs:2999 |
| PREFILL_CHUNK | 1024 | model.rs:121 |
| max_sequence_len | from max_position_embeddings (262144) | model.rs:3584 |

### Current HEAD

Branch: `mtp-speculation` (commit `2091654`). This is AHEAD of the `chat-cli`
branch where the performance work happened. Need to verify that the prefill
performance baseline is the same or better on this branch.

### Files that will be touched

- `crates/kernels/src/attention.metal` (1089 lines) — attention kernel
- `crates/kernels/src/attention.rs` — attention dispatch
- `crates/kernels/src/linear.metal` (11094 lines) — GEMM kernel
- `crates/kernels/src/linear.rs` — GEMM dispatch
- `crates/engine/src/model.rs` — HOST_TOKEN_SLOTS, chunking, dispatch

### Key constraints to remember

1. 16 named accumulators is the maximum before spilling (mma_peak: 15.49→0.74
   TFLOPS at 32 via array).
2. 16 KB threadgroup memory cliff is real and binding (18 KB → –18% per oMLX).
3. 8×8 is the ONLY simdgroup_matrix size on M4 Pro (compiler header has
   `static_assert(Cols == 8 && Rows == 8)`).
4. The nondeterminism bug (missing barrier between chunks) is NOT fixed and
   means multi-chunk benchmarks must use `KAPLUMBA_MAX_IN_FLIGHT=1`.
5. Build shaders: `cargo build --release -p kaplumba-cli` (shaders are
   `include_str!`'d; `--workspace` does NOT update the CLI binary).
6. Measurement: interleave A/B, report medians with spread. A single reading
   is not a result. Same-config spread is up to 35%.
7. `phys_footprint` from `/usr/bin/time -l` is the real memory figure.
   `peak_resident_pages_bytes` understates by ~5.5 GiB.

### Build and measure commands

```bash
# Build
cargo build --release -p kaplumba-cli

# Prefill measurement
M=~/.kaplumba/models/Gxma/Gxma
PP=$(python3 -c "print('Explain the theory of relativity. '*100)")    # ~614 tok
PP8=$(python3 -c "print('Explain the theory of relativity. '*1300)")  # ~7814 tok
PP16=$(python3 -c "print('Explain the theory of relativity. '*2660)") # ~15987 tok
PP24=$(python3 -c "print('Explain the theory of relativity. '*3990)") # ~23980 tok

scripts/gpuq try -- ./target/release/kaplumba --model "$M" --prompt "$PP" \
    --max-new-tokens 1 --temperature 0 --metrics

# GPU profile
KAPLUMBA_GPU_PROFILE=1 ./target/release/kaplumba --model "$M" --prompt "$PP" \
    --max-new-tokens 1 --temperature 0 --metrics

# Deterministic multi-chunk measurement
KAPLUMBA_MAX_IN_FLIGHT=1 ./target/release/kaplumba --model "$M" --prompt "$PP" \
    --max-new-tokens 1 --temperature 0 --metrics

# Reference prompts
for q in "What is the capital of France?" "Count from one to five." \
         "Name three primary colors." "Hi" "What is 17 times 23?"; do
    ./target/release/kaplumba --model "$M" --prompt "$q" --max-new-tokens 24 --temperature 0
done

# Memory measurement
/usr/bin/time -l ./target/release/kaplumba --model "$M" --prompt "$PP" \
    --max-new-tokens 1 --temperature 0 --metrics 2>&1 | grep -E 'phys|peak_res|prefill'
```

---

## T7 — COORDINATOR INTEGRATION + FINAL MEASUREMENT (2026-10-04)

Branches merged into `mtp-speculation`, in order:
`438ead9` (impl/ctx24k, T5) → `e1005e7` (impl/gemm-mma, T3/T4) →
`de5acfd` (impl/attn-mma, T1/T2). The three touched disjoint files, so all
three merges were clean. Then: `0e15434` build/census fixes, `c02b270` regpipe
default, `3ea15aa` T2 partitioned MMA.

### Build fixes the agents flagged (both red at base)

- `crates/engine/examples/attn_bench.rs:374` called 2-arg `decode_splits`;
  the 3-arg (q_heads, kv_heads, key_count) form has existed since `2297f19`.
  This is why `cargo test` without `--tests` never compiled.
- Census literals bumped for the two new `linear.metal` kernels:
  `perf_probe_simd_alu_roofline.rs` 111→113 / 138→140,
  `probe_kernsel_census.rs` 138→140 (×2), `probe_variants_window_forms.rs`
  138→140.

### T4 GEMM — regpipe wins, now the default (`c02b270`)

Full interleaved table in `/private/var/folders/tp/wz3877_s1z756p1kcsz3_…/opencode/p120-t4-measure.tsv`.
614-token GEMM (7 projections, ms): baseline 6035 / regpipe **5753** / 4sg 6246 /
k32 6295. TFLOPS 5.85 → **6.13**. 8k end to end 75.6 → 78.8 (+4.2%). 4sg and
k32 are measured losers (recorded negative results). regpipe is bit-identical
to `linear_gemm_mma` and MCG-3bpw gated; `KAPLUMBA_MMA_128X64` forces baseline.

### T2 ATTENTION — the T1 MMA QK was the bug, and the fix is a partition

T1's MMA body ran all four 8-key sub-tiles in **every** simdgroup. Because
`simdgroup_multiply_accumulate` is per-simdgroup, all NSG simdgroups computed
the identical 8×8 dot tile from identical fragments — NSG× redundant tensor
work *and* NSG× redundant strided K loads. Measured 1.7–5× **slower** than the
scalar QK it replaced.

T2 assigns sub-tile `sub` to simdgroup `sub % NSG`, publishes each 8×8 tile to
a 1 KB `sstage` shared buffer, and reads its own row back (one extra barrier per
full tile). MMA work 4·NSG → 4 sub-tiles. The final partial tile keeps T1's
all-simdgroup staging path (its barrier must be uniform), once per pass.

`attention_qk_mma_bench` ratio mma/scalar:

| shape | T1 | T2 |
|---|---|---|
| full d512 NSG8 keys=8192 | 1.700 | **0.833** (MMA 1.2× faster) |
| sliding d256 NSG2 keys=1024 | 1.813 | 1.588 |
| sliding d256 NSG2 keys=8192 | 4.967 | 3.296 |

The NSG=2 loss is structural: an 8×8 MMA always computes 8 query rows, but a
`groups == 2` shape uses only 2 — 6/8 of every MMA is waste. `groups == 8` uses
all 8. So `attention_qk_for(groups)` selects MMA only for `groups >= 4`
(`== 8` on Gxma); both env flags still force a global arm.

### End-to-end (merged binary, `KAPLUMBA_MAX_IN_FLIGHT=1`, interleaved)

| ctx | before | after | peak `phys_footprint` |
|---|---|---|---|
| 614 | 96.17 | **100.56** | 12.66 GiB (T5 run D) |
| 8k | 75.61 | **80.52** | 14.15 GiB |
| 16k | 60.75 | **63.96** | 15.54 GiB |
| 24k | 51.12 | **52.84** | 17.12 GiB |

`.tsv`s: `p120-int/measure-arms.tsv`, `p120-int/8k-run.log`, `p120-int/mem-run.log`.

### AC status

- **AC2 MET** (80.52 @8k). **AC6 MET** (17.12 GiB @24k < 20). **AC8 MET**
  (worst error 2.1603e-3, unchanged — regpipe is bit-identical). **AC9 MET**
  (HOST_TOKEN_SLOTS 32768; 24k prefill+decode runs).
- **AC1 (120 @614) / AC4 (7.5 TFLOPS) UNMET** — same wall. 5117 ms budget needs
  GEMM ≤4707 ms; regpipe is 5753. Attention=0 would give only ~101 tok/s. The
  remaining 6.13→7.5 is the tile-shape ceiling (T4 exhausted 4sg/k32; the
  staging-stripped diagnostic caps this shape near 76.7 tok/s of GEMM work).
  Reported as a measured wall per the spec's AC1 escape clause.
- **AC5 (attention ≥1.5× @8k) UNMET** — 1.2× on the full QK, ~+8% attention
  overall. The sliding (NSG2) arm cannot use an 8×8 MMA efficiently.
- **AC7 partly unmet** — 5 reference prompts correct (Paris / red-blue-yellow /
  391 / etc.), but decode measures **~14.0 tok/s**, below the 15 target. This is
  pre-existing (decode does not touch the MMA prefill path or regpipe) and was
  never independently verified before — T0 recorded it as "user-reported".

### Four pre-existing suite failures (NOT this spec's)

`cargo test --release --workspace --tests --no-fail-fast` → 111 targets ok,
4 failing. Each verified present at the base commit `43a4176` by inspecting the
base blobs directly:

| test | why it was already failing at 43a4176 |
|---|---|
| `perf_probe_host_ops::the_device_token_handoff_exists…` | the `host_tokens` allocation is split over two lines (base model.rs:3172-3173), the probe greps one line |
| `perf_probe_dispatchsel_columns::the_colmajor_contig4…` | base model.rs already names `ishizuki_gemv_pipeline` (probe asserts 0) |
| `perf_probe_concurrency_dependency_graph::no_device_side_sampler_exists` | `kernel void sample` already present in base elements.metal (device-sampler spec shipped) |
| `probe_cache_hierarchy::activation_loads_are_4_byte_alignable…` | base linear_repacked.metal:2121 is not the pinned line (kernel moved) |

They are stale source-text probes owned by other specs; left untouched (not
weakened) and reported as limitations, per AGENTS.md §4.3.

### Measurement hygiene notes worth keeping

- The 614 arm of the *final* A/B had one 82.64 tok/s outlier with
  `ttft=7430 ms` (box recovering from the 24k run); its r2 was 100.60. Report
  the median, and never a single amber reading.
- `KAPLUMBA_GPU_PROFILE` cannot be turned off with `=0` — it is presence-based
  (`env_flag` uses `var_os().is_some()`); use `env -u` (T5 §7).
- `/usr/bin/time -l` "peak memory footprint" is the budget number;
  `peak_resident_pages_bytes` understates by ~5 GiB (AGENTS §10.3).
```
