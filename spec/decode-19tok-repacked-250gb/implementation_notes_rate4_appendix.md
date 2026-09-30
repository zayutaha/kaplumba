# Appendix: is the 3 bpw colmajor GEMV slow because of the rate, or the kernel?

Separate file, not appended to `implementation_notes.md`, because that file is
under active edit by another agent in this working tree and staging it would
commit their uncommitted work with mine. Nothing here duplicates it.

Experiment: **2026-09-30**. Question asked by the orchestrating session:

> does our colmajor 3 bpw GEMV stall at 187.7 GB/s because of the EXL3 *rate*
> (amplification 2.667), or because the kernel has a defect that shows at every
> rate?

---

## Status: instrumentation + arithmetic complete and verified; the measurement
## itself is BLOCKED and was not taken

**The number this experiment exists to produce does not exist yet.** The GPU was
occupied for the entire session by another agent running a tight A/B loop of model
loads (`Gxma` and `Gxma-repacked`, ~9 `load` cycles per 100 s, holder pid changing
every ~10 s throughout). AGENTS.md 10.1/10.2 forbid a second GPU job, and 10.2
rule 3 records that a "just to check" second instance has already destroyed
measurements in this project. So the sweep was not run and no bandwidth figure is
reported below.

Everything needed to run it in one command is committed. See **How to run** below.

---

## What is settled (no GPU required, all committed and tested)

### 1. The amplification arithmetic is confirmed exactly

`crates/engine/tests/rate4_colmajor_bit_exact.rs`,
`lane_load_amplification_is_2_667_at_3_bpw_and_2_000_at_4_and_6`, enumerates the
32 lanes' real `j0`/`j1`/`j2` indices from the kernel's own arithmetic:

| rate | words/tile | lane-loads/tile | amplification |
|---|---:|---:|---:|
| 3 bpw | 24 | 64 | **2.6667** |
| 4 bpw | 32 | 64 | **2.0000** |
| 6 bpw | 48 | 96 | **2.0000** |

Compared as exact rationals, not with an epsilon. The spec's own fit reproduces:
`250.5 x 2.000 / 2.667 = 187.9 GB/s` against the measured `187.7`.

**Answer to "does a 4 bpw tile need 2 words/lane or 3?" — two.** `WORDS =
RING/32 = 32`, i.e. 64 u16 per tile, against the 48 u16 (24 words) that
`exl3::reorder_tensor_to_colmajor` hardcodes. That hardcode is why a 4 bpw
colmajor tensor cannot be built by the production repacker and why this
experiment needed a new repack.

### 2. DRAM line traffic is 1.000x at every rate — the amplification is
###    request-path only

`dram_line_traffic_is_1x_at_every_rate_so_the_amplification_is_requests_only`
takes the union of 128-byte lines across a whole colmajor column at all three real
`tiles_k` values (336, 1024, 1344) and gets **1.0000x** in all nine cases.

So the 2.667 at 3 bpw is **not** wasted DRAM traffic. The kernel reads every byte
of the matrix; it issues 2.667x as many lane-loads for those same bytes. The
limit is request throughput / load latency, not bandwidth. This is what makes the
amplification model able to predict the measured 3 bpw rate from a 6 bpw DRAM
ceiling with no term for wasted bytes, and `implementation_notes.md` previously
asserted it in one sentence without checking it.

### 3. Two words per lane is legal at KA 1..=4 and at nothing above

`the_two_word_window_is_safe_exactly_at_rates_one_through_four` measures it rather
than arguing it: recompute all 256 codes from real packed data at every rate 1..=8
with the two-word form and the three-word form, require equality where
`max_lane sh <= 48 - 7*KA` says it should hold, and require that it **does not**
hold at 5 and 6 bpw — otherwise the comparison could be passing vacuously.

**I got this wrong first and the hardware caught it.** The first version of the
6 bpw kernel was a literal copy of the 4 bpw one, justified by "`7*KA + 16 <= 64`,
so `KA <= 6`". That is condition (a) only. Condition (b) —
`max_lane sh <= 48 - 7*KA` — is the one that bites, and at KA = 6 max sh is 22
against 6, so every sh=22 lane decodes six bits of zeros as weight. Result:
**1023 of 1024 fp16 outputs wrong at 128x1024**.

The repo already had the correct derivation: `two_word_is_safe` in
`scripts/gen_exl3_gemv.py` states *both* failure conditions and
`TWO_WORD_SAFE_RATES` evaluates to `(1, 2, 3, 4)`. Worth noting that **3 bpw and
4 bpw sit exactly ON the boundary** (27 <= 27 and 20 <= 20) — the two-word form is
legal there only because the lowest bit a code reads coincides with the first
valid bit. Anyone adding a rate to the two-word family should read that derivation
before assuming it extends.

### 4. Latent bug found in `linear.metal`, not fixed (out of scope, not mine)

`linear_gemv_ishizuki_k{1..8}` compute

```metal
const uint j2 = j1 + 1u >= WORDS ? j1 + 1u - WORDS : j0;
```

where the `: j0` arm should be `j1 + 1`. It is **latent** at the rates the engine
dispatches, because Metal reports `j2` as `unused variable` there (nothing reads
`A0` below its own `sh` at those rates) — the compiler warning is the only trace
of it. It is **actively wrong** at any rate where `j2` is read. Left alone: fixing
a generated block is out of scope for a measurement experiment and would need the
golden re-verified. The new 6 bpw kernel uses the corrected form and says so.

---

## What was built

### `crates/kernels/src/linear_repacked.metal` (additive, 515 lines)

`linear_gemv_ishizuki_k4_colmajor_contig4_bench` and
`linear_gemv_ishizuki_k6_colmajor_contig4_bench`: the production decode kernel
`linear_gemv_ishizuki_k3_colmajor_contig4` with `KA` changed and nothing else.

**Neither is reachable from the engine, and the naming is the argument.**
`ishizuki_gemv_pipeline` picks a production kernel by exact HashMap key on
`k{bits}_colmajor_contig4`, `k{bits}_colmajor`, `k{bits}_w2`, or the sorted-first
kernel naming that rate. The first is built with `format!`, so a kernel named
`..._k4_colmajor_contig4` **would** be dispatched for any 4 bpw `perm_mode == 1`
tensor. The `_bench` suffix cannot match any of the keys, and sorting puts the
unsuffixed `linear_gemv_ishizuki_k4` ahead of it so the slot table is untouched
too. Both halves are asserted against the real `LinearKernels` registry.

### `crates/engine/tests/rate4_colmajor_bit_exact.rs` (6 tests)

CPU reference built from `exl3::decode_tile` (a different implementation of the
ring-window arithmetic from the Metal one), mirroring the whole dispatched pipeline
including both Hadamards and the fp16 rounding at each boundary.

The **3 bpw production kernel is the calibration arm**: 0 ULP at 128x1024 and
1024x1024, 1 ULP / 19-of-5376 at 5376x5376. So its deviation from the reference is
the reference's own f32 reassociation noise, and 4 and 6 bpw are required to be no
worse. Measured: 0/0 ULP and 1 ULP / 20-of-5376.

### `crates/engine/examples/bench_rate_sweep.rs`

11 shapes x 3 rates, chunked by shape (peak 2.3 GiB, not 5.3), every rate of a
shape interleaved within every round, arm order reversed on alternate rounds,
medians with best/worst and spread, byte-weighted aggregate, and a VERDICT block
that states which of the three readings the numbers support.

---

## Sabotage checks (AGENTS.md 3.4) — every one run and observed to fail

| sabotage | observed failure |
|---|---|
| `WORDS` hardcoded to 24 in the 6 bpw kernel | 1024/1024 fp16 wrong, max 4198 ULP |
| dropped `& 0xffffu` from the 4 bpw decode | 1022/1024 wrong at 1024x1024, max 2277 ULP — **and 0/1024 at 128x1024**, i.e. a narrower shape would have passed it |
| renamed to `..._k4_colmajor_contig4` | reachability claim fails |
| `per_lane = 2` for all rates in the audit | "6 bpw: words per lane is 2, expected 3" |

---

## Two measurement traps found in this repo's existing harnesses

**The running-LCG tile generator makes the colmajor repack the identity.**
`bench_residency`, `live_kernel_bit_exact` and friends build tile states with a
running LCG (`s = s * 1664525 + 1013904223`). Its low 3 bits have period 16
(the multiplier is 5 mod 8, so the low bits alternate with period 2 and close at
16) and a tile is 256 draws — so **every tile receives the identical 256-state
sequence** and the packed tensor is one tile repeated. Bandwidth is unaffected
(same bytes, same addresses) but any layout check is vacuous. Caught because
`the_colmajor_repack_is_a_pure_tile_permutation`'s `assert_ne!(mine, row)` — "the
repack actually transposed something" — failed. Fixed here with splitmix64 over
`(tile, code)`. **The three pre-existing harnesses still have it.**

**A Metal compile failure was a silent skip.** `LinearKernels::new()` returning
`Err` was `else { eprintln!("skipping"); return; }` — which turned a one-character
syntax error (a missing `)` in `[[buffer(0)]]`) into a **green test** and a
bandwidth number computed from a library that did not contain the kernel. Both GPU
tests now `expect()` it. That is the third instance in this repo of a fallback that
fails open.

---

## How to run (the deliverable that is missing)

On an **idle** box — `scripts/gpuq status` must show `gpu: none` and >= 14 GiB
available, per AGENTS.md 10.1/10.3:

```bash
git worktree add --detach /tmp/rate4 HEAD && cd /tmp/rate4
export CARGO_TARGET_DIR=/tmp/rate4/target          # AGENTS.md 10.4: never share
cargo build --release --example bench_rate_sweep
scripts/gpuq try -- ./target/release/examples/bench_rate_sweep --rates 3,4,6
```

Exit 75 means busy: retry later, do not bypass. Peak resident 2.3 GiB (lm_head at
all three rates). Expect a few minutes; weight synthesis is CPU-bound and the
measurement is ~12 s of GPU time.

Optional sanity first (no GPU, safe on a busy box):

```bash
./target/release/examples/bench_rate_sweep --check-determinism --rates 3,4,6
# run twice; digests must be identical. Verified: they are, and they are
# byte-identical across the two rewrites of this harness.
```

### What each outcome would mean (the decision rule)

| reading | conclusion |
|---|---|
| 4 bpw and 6 bpw ~220-250 while 3 bpw is ~188 | the **rate** is the constraint; the kernel is sound; the 3 bpw gap is a format property. Both arms sit at amplification 2.000 vs 3 bpw's 2.667, so this is the amplification model confirmed, and it is the same ratio the spec already fitted. |
| 4 and 6 bpw both stall near 188 | **rate-independent kernel defect**; real headroom exists. Report the per-shape ratio — anything != 1.0 localises it. |
| between | report the ratio per shape; a shape-dependent ratio localises the defect. |

Note the 6 bpw arm is *not* a lower-amplification point: at 2 words/lane it would
have been 1.333, but 6 bpw cannot use two words, so it is **2.000** — the same as
4 bpw. That makes it a same-amplification / different-rate control, and it is the
arm that separates "the requests are the limit" from "the bit rate is the limit".
Its in-situ comparator already exists: `lm_head` at 6 bpw measured **250.5 GB/s**
(row-major, `perm_mode = 0`). If 4 bpw colmajor lands at ~250 too, two different
rates at the same amplification agree, and amplification is the explanation.
