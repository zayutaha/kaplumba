# IMPLEMENTATION NOTES: 3 bpw Lane-Load Amplification

Running log per AGENTS.md §5. Appended as work proceeds, **not** at the end. Factual and
dense; paths, numbers and commands over narrative. A later implementer should not have to
rediscover anything below.

---

## 1. Environment facts (verify before trusting; this box drifts)

- **Apple M4 Pro, 20 GPU cores, 24 GiB unified, 273 GB/s datasheet.** Measured streaming
  ceiling in this repo: **~250.5 GB/s** (`decode-19tok-repacked-250gb/implementation_notes.md:29`).
- **~16.0 GiB is the working ceiling; the machine has 24 GiB.** Two model instances
  exceed physical memory and have already destroyed measurements here.
- **25 git worktrees are registered and all still exist on disk.** None is prunable.
  `git worktree list` before creating one; always give a new worktree its own
  `CARGO_TARGET_DIR` (AGENTS.md §10.4).
- **The Metal toolchain is ABSENT on this box**: `xcrun -sdk macosx metal` → "missing
  Metal Toolchain"; `air-dump`, `metallib`, `mtldis` all absent. **No register counts or
  disassembly can be obtained. Do not invent them**
  (`spec/decode-19tok-repacked-250gb/findings/02:134-141`).
- **`scripts/bench.py` has been removed**, not merely unused. `.gitignore` still carries
  `scripts/__pycache__/`.
- `scripts/gpuq` warns when available < **11.0 GiB** (`gpuq:152`), not the "~14 GiB"
  AGENTS.md §10.1 states. The script's own note records that a load succeeds at 6.7 GiB
  available with 0 swap, because the model sits in clean reclaimable cache.
- `gpuq status` frequently reports `none (stale lock, pid N is gone)`. The lock file is
  never cleaned; this is harmless because `gpuq` takes no lock and the engine's own
  `flock` is the real guard.

## 2. Baseline verified on hardware — 2026-10-01

`Gxma-repacked` (colmajor, 3 bpw), `scripts/gpuq try -- ./target/release/kaplumba`,
one process at a time:

| run | prompt | generated | decode tok/s |
|---|---|---:|---:|
| 1 | "The capital of France is" | 2 (EOS) | 15.033 |
| 2 | "The capital of France is" | 2 (EOS) | 15.389 |
| 3 | "The capital of France is" | 2 (EOS) | 15.409 |
| 4 | "Count from 1 to 100: 1, 2, 3," | 128 | 15.295 |
| 5 | "Count from 1 to 100: 1, 2, 3," | 128 | 15.338 |

**15.03 – 15.41, median 15.32 tok/s = 65.3 ms/step.** Long runs agree within ±0.03,
so the short 2-token runs are the noisy ones, not the long ones. Matches the
`decode-25tok-peak-bandwidth` figure of 65.2 ms. Output bit-exact; 0 MiB swap.

`peak_resident_pages_bytes` read 7.8–8.5 GiB across these runs. **That is `ru_maxrss`
and is not a footprint** (AGENTS.md §10.3: it understates real usage by 5.5 GiB). Use
`/usr/bin/time -l`.

## 3. The measurement this spec rests on — 2026-10-01

Built and ran the previously-blocked experiment from
`spec/decode-19tok-repacked-250gb/implementation_notes_rate4_appendix.md` (written
2026-09-30, never run because the GPU was busy for that whole session).

```bash
WT=/private/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/wt-rate4
git worktree add --detach "$WT" HEAD
export CARGO_TARGET_DIR="$WT/target"
cargo build --release --example bench_rate_sweep
scripts/gpuq try -- ./target/release/examples/bench_rate_sweep --rates 3,4,6
```

Reproducibility first (`--check-determinism`, no GPU touched): **33/33 digests
identical across two independent runs.** The harness is sound.

```
3 bpw:  199.4 GB/s   amp 2.667   24 words/tile   96 B
4 bpw:  249.0 GB/s   amp 2.000   32 words/tile  128 B
6 bpw:  251.4 GB/s   amp 2.000   48 words/tile  192 B
VERDICT 4/3 = 1.249x  (model predicts 1.333x, residual 0.937)
```

Three conclusions, all load-bearing:

1. **The kernel is not defective.** At amp 2.000 it reaches 249–251 GB/s = the measured
   ceiling. There is no headroom hiding behind a bug.
2. **The limiter is lane-load count.** 4 bpw and 6 bpw differ by 1.5× in both bytes and
   dequant ALU, yet land within 1% of each other. ALU- or DRAM-bound would separate
   them. Only amplification tracks bandwidth.
   *(This also contradicts the standing "ALU is 62.9% of the kernel, make `rad`
   compile-time for 3.6×" claim at `decode-19tok-repacked-250gb/implementation_notes.md:1241-1258`.
   That claim may still be true as ALU arithmetic, but ALU is **not** what is limiting
   bandwidth — otherwise 6 bpw would be far slower than 4 bpw. Treat the 3.6× as an
   unverified upper bound on a non-binding resource.)*
3. **4 bpw is a net loss, not a wash.** Bytes ×1.333, bandwidth ×1.249 ⇒ **time ×1.067,
   i.e. 6.7% slower.** 6 bpw likewise ×1.15. My earlier prediction of a "wash" was
   directionally right and specifically wrong on sign. `Gxma-bpw4-shim` and the identical
   copy in `~/Downloads` (byte-for-byte, 20,041,520,424 B) are **not** worth pursuing.

The harness's isolated 199.4 GB/s agrees with the ~196–207 GB/s measured in-engine, so
these numbers transfer.

## 4. Where the 2.667 comes from, and the derivation that replaces it

`crates/kernels/src/linear_repacked.metal`, shared by all four live 3 bpw kernels:

```
constexpr uint KA=3; RING=256*3=768; WORDS=768/32=24;      // :1794-1795
lo0 = (lane*8u+1u)*3u + RING - 16u ;  sh0 = lo0 % 32u       // :1804-1805
j0 = (lo0/32u) mod WORDS ;  j1 = (j0+1) mod WORDS            // :1808-1809
extract_v_mcg_32(t0[j0], t0[j1], sh0, rsh0, v)                // :1836
```

Each lane wants 8 codes × 3 bits = **24 bits** but loads a **2-word / 8-byte** window and
funnel-shifts it. 32 lanes × 2 = **64 lane-loads / 24 words = 2.667**.

`dram_line_traffic_is_1x_at_every_rate_so_the_amplification_is_requests_only`
(`rate4_colmajor_bit_exact.rs:746`) proves the 128-byte lines are shared, so DRAM bytes
are 1× — the waste is **requests**. Hence the fix must not add bytes. Padding each lane
to 4 bytes would give amp 1.333 but 128 B/tile, which **is** 4 bpw, which is −6.7%.

**The way through:** 32 lanes × 24 bits = 768 = exactly the tile, zero padding. Put lane
L's 8 codes at bit offset `24·L`:

| `L*24 mod 32` | 0 | 24 | 16 | 8 |
|---|---:|---:|---:|---:|
| 24 bits fit one word? | yes | **no** | **no** | yes (8+24=32) |

16 lanes straddle, 16 do not ⇒ 16×2 + 16×1 = **48 lane-loads / 24 words = amp 2.000**,
at **96 B/tile unchanged**. 48 is the value 4 bpw already proves sufficient. Bonus: code
*i* sits at bit `3i`, so every shift is a compile-time immediate and
`extract_v_mcg_32`'s 64-bit funnel shift leaves the hot path.

This is **hand-derived arithmetic, not a measurement** — which is exactly why T1 exists.

## 5. Corrections to the standing record (do not re-learn these)

- **`reference/omlx` is the only reference actually present** (Apache-2.0). Cite the repo
  copy, not the scratch `.../T/opencode/omlx`, which contains only `quantized.h`.
- **`exllamav3` is a HOLLOW directory — 0 files.**
  `exllamav3/modules/quant/exl3_lib/` = 0 entries, `exllamav3/loader/` = 0 entries.
  **`exllamav3_gemm_inner.cuh` does not exist.** AGENTS.md §7 lists it as the source of
  the `cp.async`/`int4` staging strategy; there is nothing to read. Re-clone first.
- **`ik_llama.cpp` is not cloned.** `/tmp/kaplumba-refs/` holds `llama.cpp`, `omlx`,
  `TensorFold` only.
- **`omlx/patches/qwen35_verify_qmm.py` is NOT a decode kernel.** `vk_eligible` is
  `3 <= M <= 6` (`:624`), `mma_eligible` is `7 <= M <= 24` (`:529`), and it explicitly
  falls through at M=1 (`:665`) — our decode case. AGENTS.md §7's description of it as
  "the 32×32 tile and no-barrier-in-K-loop shape" is wrong: the no-barrier property comes
  from **private per-simdgroup buffers** (`:465`), and the geometry does not transfer.
- **oMLX uses no second stream.** `grep mx.new_stream` over the whole tree = 0 matches;
  `engine_core.py:521-528` states all scheduler steps run on `_mlx_executor`
  (single-worker) "to guarantee that MLX GPU operations are never concurrent". Any
  multi-queue attention/GEMV overlap has **no precedent to port** — and oMLX, the closest
  working analogue, deliberately avoids it.
- **ishizuki is AGPL-3.0** and `spec/engine-performance/spec.md` Q9 (line ~270) is still
  **OPEN**. Citing its ideas is safe; porting `GGML_ACC8`/`GGML_W8` is not without the
  user closing Q9.
- **`contig4` is not the live decode kernel.** `encode_gemv_raw` picks
  `linear_gemv_ishizuki_k3_colmajor_2col` first because `out % 32 == 0` always
  (`linear.rs:1963`), so `ishizuki_gemv_pipeline`'s by-name `contig4` lookup
  (`linear.rs:1372-1383`) is never reached in this model. Live set and threadgroup counts
  are tabulated in `plan.md` §2.1. A subagent initially reported `contig4` as live; that
  was wrong and has been corrected.
- **`lm_head` is 6 bpw on the EXL3 arm**, not ishizuki: its trellis sits at offset
  `% 4 == 2`, which `ishizuki_gemv_pipeline` refuses (`linear.rs:1353`). It resolves to
  `linear_gemv_exl3_k6_u_mcg` with 16384 threadgroups. **Out of scope for a 3 bpw fix**,
  and the likely reason `lm_head` has always measured near the ceiling.
- **`linear_gemv_ishizuki_k3_colmajor_u4` is threadgroup-STAGED** (3072 B/tg, 2 barriers
  per 4 tiles, `:2500-2605`), which is why it lost (180.5 → 173.7). It is **not** a
  precedent for "wider direct loads". It is bench/test-only, never on the decode path.
- **`linear_gemv_ishizuki_k3_colmajor_stream`** (`:1544-1651`) already exists as a
  **barrier-free, no-threadgroup-reduction** variant: one output tile per simdgroup
  (`tn = tgid*SG + sg`, `:1571`), writes `y` directly (`:1646-1650`). Not on the decode
  path. This is the fallback morphology if the `straddle` branch proves expensive
  (`plan.md` §5 risk 1).
- **The running-LCG tile generator makes the colmajor repack the identity** (low 3 bits
  have period 16; a tile is 256 draws), so any layout check against it is vacuous.
  `bench_residency`, `live_kernel_bit_exact` and friends **still have it**. Use splitmix64
  over `(tile, code)`, as `rate4_appendix:155-166` does.
- **`LinearKernels::new()` returning `Err` used to fail OPEN** — `else { eprintln!("skipping"); return; }`
  turned a missing `)` in a `[[buffer(0)]]` into a green test and a bandwidth number from a
  library that did not contain the kernel. **Third instance of this in the repo.** Both GPU
  tests now `expect()` it. Any new harness must too.
- **The lossy bit-plane repacker `exl3.rs:687-726` has zero callers and must stay dead.**
  It keeps `s & 0x7` while `decode_state_mcg` consumes a 16-bit window of which 13 bits
  survive `0x8fff8fff`, and its `imul` mixes all 16 bits *before* masking — so 3 bits of
  input yield **different** dequant, not imprecise dequant. Barred by the no-quality-loss
  rule, not merely unpromising. This spec writes a **different** permutation alongside it.
- **`extract_v_mcg_fast` (`linear_repacked.metal:66-77`) has zero callers.** Dead.
- `repack_trellis_3bpw_tensor`/`_tile`, `gemv_reduce_sk2`/`_sk4` (built, never dispatched),
  `repacked_kernel_names()` (matches nothing), `_colmajor_simd` (never dispatched) are all
  dead. Listed so they are not mistaken for live paths.

## 6. Model directory map (verified 2026-10-01)

| path | total bytes | layout | bits | head | codebook | notes |
|---|---:|---|---|---:|---|---|
| `Gxma/Gxma` | 16,028,050,244 | none (row-major) | 3.0 | 6 | mcg | **original — must stay untouched** |
| `Gxma-repacked` | 16,028,388,994 | `colmajor` | 3.0 | 6 | mcg | **the 15.3 tok/s baseline** |
| `Gxma-bpw4-shim` | 20,041,520,424 | none | 4 | 8 | mcg | fixture, not runnable |
| `~/Downloads` (3 shards) | 20,041,520,424 | — | 4 | 8 | **mul1** | **byte-identical to the shim** |

Per-token bytes from the actual safetensors headers: MLP 130.05 MB/layer, attention
49.56 MB/layer, +norms ⇒ **179.73 MB/layer × 60 = 10.78 GB**, plus `lm_head.trellis`
1.06 GB. `embed_tokens` is BF16 2.82 GB (search only — one row per step, no bandwidth cost).

The 4 bpw shim is unusable as a checkpoint regardless of speed: `codebook: mul1` (the
engine's loader refuses any non-MCG codebook), no `embed_tokens` entry, no
`processor_config.json`, no `chat_template.jinja`, and its `quantization_config.json`
claims `bits: 3.0 / head_bits: 5` while the payload is genuinely 4/8 (trellis 3.468 GB
vs 2.601 GB = exactly 1.333×). **The JSON header is stale relative to the payload** — do
not trust it; read the tensor headers.

## 7. Commands

```bash
# GPU state (instant, read-only)
scripts/gpuq status

# decode throughput — ALWAYS one at a time, always report spread
M=~/.kaplumba/models/Gxma-repacked
scripts/gpuq try -- ./target/release/kaplumba --model "$M" \
  --prompt "Count from 1 to 100: 1, 2, 3," --max-new-tokens 128 \
  --temperature 0 --metrics

# which kernel each projection resolved to
KAPLUMBA_TRACE_GEMV=1 ./target/release/kaplumba --model "$M" --prompt hi \
  --max-new-tokens 2 --temperature 0

# per-kernel GPU attribution
KAPLUMBA_GPU_PROFILE=1 ./target/release/kaplumba --model "$M" --prompt "$PP" \
  --max-new-tokens 1 --temperature 0 --metrics

# real footprint (NOT peak_resident_pages_bytes)
/usr/bin/time -l ./target/release/kaplumba --model "$M" --prompt Hi \
  --max-new-tokens 8 --temperature 0

# rate sweep (T1 lives here)
scripts/gpuq try -- ./target/release/examples/bench_rate_sweep --rates 3,4,6
./target/release/examples/bench_rate_sweep --check-determinism --rates 3,4,6
```

`--prompt` over 8192 tokens is rejected **before loading** with "prompt plus generation
exceeds context limit 8192" — which looks exactly like a silent failure if you are only
reading the tail of a pipe (AGENTS.md §10.2 rule 7). The test sentence
"Explain the theory of relativity. "×100 is 6.01 tokens.

## 8. Corrected baseline ledger

`decode-25tok-peak-bandwidth/implementation_notes.md:18-20` labels 65.0–65.4 ms /
15.35 tok/s as "Baseline Measurements (Starting Point from Commit `d7140b2`)" — a
starting point, not a current figure, and later task results appear *above* it in the
same file. For the current build, **65.3 ms / 15.32 tok/s, re-measured 2026-10-01 (§2
above)** is the number to plan against. `decode-19tok-repacked-250gb/spec.md:11,204`
still carries 74.01 ms / 13.511 tok/s as its figure of record; four different "current"
baselines exist across the specs. Cite §2 of this file.
