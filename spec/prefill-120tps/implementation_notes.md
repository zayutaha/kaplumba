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
