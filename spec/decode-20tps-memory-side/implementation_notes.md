# spec/decode-20tps-memory-side — implementation notes

Dense, factual, append as work proceeds.

## Environment facts

- Checkpoint for kernel work: `~/.kaplumba/models/Gxma-repacked` (colmajor, 3 bpw,
  what the live kernel dispatches against). `Gxma/Gxma` is the row-major twin.
- `config.json`: hidden 5376, intermediate 21504, 32 heads × head_dim 256,
  kv heads 16, vocab 262144, 60 layers.
- GPU probes need sole GPU access; `scripts/gpuq try`, exit 75 = busy, retry.
- The probe is ~7 s of GPU and reads ~90 MiB of trellis. Cheap enough to iterate.
- Build with `cargo build --release -p kaplumba-engine --test <name>` then run
  `./target/release/deps/<name>-<hash>` directly. `cargo test` re-links and has
  exceeded 600 s here.

## Established numbers (measured this session, reproducible)

| item | value | source |
|---|---|---|
| arm (G), register-resident, zero memory | 633.4 / 637.2 / 634.0 Gcodes/s = 237.6 GB/s | `findings/F1-arm-g-runs.md` |
| arm (C), no weight loads | 566.8–567.9 Gcodes/s = 213 GB/s | F1 |
| production `_colmajor_2col`, gate_proj | 463.6–512.4 Gcodes/s = 174–192 GB/s | F1 |
| arm (E), extraction masks removed | +0.1% → masks are free | F1 |
| arm (D), constant shift amounts | −2.5% → variable shifts cost 2.5% | F1 |
| copy ceiling | 250.5 GB/s | `decode-19tok-repacked-250gb` |
| bytes/step (corrected) | 12.039 GB | two independent derivations |
| 20 tok/s | 631 Gcodes/s = 236.7 GB/s | arithmetic |

## Traps already paid for — do not re-pay

- **A verdict that flips on a calibration constant is not a discriminator.**
  `decode-ceiling` ran `code_rate_vs_bandwidth_discriminator` twice, changing one
  constant, and it printed *both* verdicts. Its ALU body is a serial dependency
  chain, so instruction count became latency. Register the criterion *and* show
  the instrument can produce each outcome before reporting either.
- **Silent wrong-read defects are the norm here.** Out-of-bounds and misaligned
  device reads are tolerated on AGX (measured to 2^30 elements past the end), so a
  bad read returns wrong data, not a fault. `MTL_DEBUG_LAYER=1` is what names
  them. This spec's loader bug was exactly that shape.
- **`data_offsets` are relative to the data section at `8 + header_len`.** Reading
  them as absolute returns header bytes. Cost one session already.
- **Uncommitted kernel edits get lost on session restart**, and the resulting
  failure looks exactly like a stale binary (`device-sampler` T2). `md5 -q` the
  file after an edit; commit it in the same step.
- **A `None` buffer placeholder in `dispatch_groups` does not create a binding
  gap** — the param struct is bound at `bindings.len()`, always. Caused a silent
  all-zero result in `device-sampler`.
- **Do not run `cargo fmt`** on this crate: it reformats 86 files.
- **Same-config spread is 35%**, and decode on this box has been measured at
  3.7–6.4 tok/s from one binary minutes apart. Interleave A/B; report medians with
  spread; a single reading is not a result.

## Dead ends already closed in this tree (do not spend GPU time)

Occupancy, threadgroup size, layout/contiguity, TLB, split-K on gate/up,
persistent kernels, 4col, NR0, activation dtype, element-kernel fusion,
multi-queue overlap, KV-quant-by-layout, storage modes, prefetch, multi-word
widening, per-code decode reduction, `uint4`/`ulong2` widening (no lane-uniform
residue class exists at KA=3), `ushort2` (requires `%4==0`, which already takes the
fast path), threadgroup staging for trellis (70 instr vs 64, plus barriers),
the `w2` trim (already 2 words/lane at 3 bpw).

Also closed: **bytes/token (R1)**. Time is set by `lane_words(rate)`, and
`bytes/rate` cancels exactly, so lowering precision moves bytes but not time.
Exhaustive over all 8^6 rate assignments, the 410 layer tensors are a plateau over
1–4 bpw. **So the memory-side route is the only route, which is why this spec
exists.**

## T1 result — per-shape bandwidth tracks N (files: `findings/F2-per-shape.md`)

Changed the loader to select one payload per **distinct (K,N) shape**. Selecting
"the 2 biggest" had returned two identical gate/up tensors, so the old "2 biggest"
sweep was really one shape measured twice. Added `trellis_shape(n,k)` as the
single owner of the 3-D tile-major convention (`[N/16, K/16, 16*bpw]` colmajor;
`[K/16, N/16, 16*bpw]` row-major) and `shape_inventory()`, a header-only byte
census that needs no GPU.

| shape | ×tensors | bytes | GB/s | Gcodes/s | grid |
|---|---:|---:|---:|---:|---:|
| K=5376 N=21504 | 120 | 5202 MB | 191.9 | 511.7 | 672 |
| K=21504 N=5376 | 60 | 2601 MB | 175.0 | 466.7 | 168 |
| K=5376 N=8192 | 50 | 826 MB | 167.5 | 446.6 | 256 |
| K=8192 N=5376 | 50 | 826 MB | 160.3 | 427.3 | 168 |
| K=5376 N=4096 | 100 | 826 MB | 136.1 | 362.8 | 128 |
| K=5376 N=16384 | 10 | 330 MB | 185.4 | 494.5 | 512 |
| K=16384 N=5376 | 10 | 330 MB | 172.1 | 458.9 | 168 |
| K=5376 N=2048 | 10 | 41 MB | 100.2 | 267.2 | 64 |
| **weighted** | **410** | **10.983 GB** | **176.5** | **470.7** | |

Byte-weighted: 10983/Σ(bytes/GB/s) = 62.23 ms, matching the measured ~62.3 ms
step. The ratio is monotone in N over 1.9×, and **Gcodes/s tracks GB/s**, so the
small shapes are *not* instruction-bound: the same instruction stream runs 1.9×
slower, which is the signature of insufficient parallelism. This is the T2
hypothesis, and it must be proven by varying grid at **fixed shape and fixed
bytes** before it is accepted.

## T2/T3/T5 result — F1/F2 measured a dead kernel (files: `findings/F3-live-kernels.md`)

**The P01 probe's "production" arm is not the live decode kernel.**
`KAPLUMBA_TRACE_GEMV=1` on a real decode run prints:

```
linear_gemv_ishizuki_k3_colmajor_2col_sk2    (o_proj, down_proj)
linear_gemv_ishizuki_k3_fused2_2col          (gate, up)
linear_gemv_ishizuki_k3_fused3_2col          (q, k, v)
linear_gemv_ishizuki_k<unknown>              (lm_head; reverse name lookup bug)
```

`linear_gemv_ishizuki_k3_colmajor_2col` — F1's "production" and the kernel F2's whole
table is built on — **is not in the list**. It is only reached from `encode_gemv_raw`'s
2col arm (`linear.rs:2130-2157`), which the live path does not take. Same class as the
loader defects in F1: the instrument timed an entry point the engine never calls.

**The live kernels are faster than F2 said.** `KAPLUMBA_DECODE_PROFILE=kernel`, 32 steps:

| kernel | own ms/step | share | bytes/token | GB/s |
|---|---:|---:|---:|---:|
| gate_up_gelu | 25.58 | 40.0% | 5.202 GB | 203.4 |
| down_proj | 13.10 | 20.5% | 2.601 GB | 198.6 |
| qkv_proj | 8.62 | 13.5% | 1.652 GB | 191.6 |
| o_proj | 6.22 | 9.7% | 0.991 GB | **159.4** |
| head/gemv | 4.11 | 6.4% | 1.057 GB | ~257 |
| attention | 2.34 | 3.7% | — | — |
| qk_proj | 1.95 | 3.1% | — | — |
| post_ffn_out / post_attn / kv_norm / norms | 2.0 | 3.1% | — | — |

Uninstrumented step **62.8 ms = 15.92 tok/s** (`decode_tok_s=15.917`, 128 tokens,
temp 0). Matvec share 93.2% ⇒ ~58.5 ms for ~12.04 GB = **~206 GB/s live matvec**.

**Ceiling re-measured today** (`probe_bw_250gb`, self-gating, 2/6/12 GB, uint4):
median **241–249 GB/s**, best **253.3**. The user's 250 GB/s is real for a 2 GB pure
stream; it is not what 12 GB of weights delivers into the kernel.

**The arithmetic cap — this is the headline:**

```
now         58.5 ms matvec + 4.3 ms rest = 62.8 ms  15.92 tok/s   206 GB/s
at 245      49.1 + 4.3                  = 53.4 ms  18.7 tok/s
at 237 (arm G instruction bound)        55.1 ms    18.1 tok/s
20 tok/s    needs matvec <= 45.7 ms  => 263 GB/s  > ceiling
```

**So the bandwidth route tops out at ~18–19 tok/s.** 20 tok/s needs fewer bytes/token or
more tokens per weight pass. §8 forbids precision loss, so the lossless route is
verify-gated speculative decoding / MTP — **and no such infrastructure exists in this
tree** (searched; `spec/mtp-40tps/` is a different 4-bit checkpoint, not a decoder).

Full JSON-safe repro commands:

```bash
M=~/.kaplumba/models/Gxma-repacked
PP=$(python3 -c "print('Explain the theory of relativity. '*10)")
# end to end
scripts/gpuq try -- ./target/release/kaplumba --model "$M" --prompt "$PP" \
    --max-new-tokens 128 --temperature 0 --metrics
# per-kernel
scripts/gpuq try -- env KAPLUMBA_DECODE_PROFILE=kernel ./target/release/kaplumba \
    --model "$M" --prompt "$PP" --max-new-tokens 40 --temperature 0
# ceiling
scripts/gpuq try -- ./target/release/examples/probe_bw_250gb
```

## T4.1/T4.2 — `fused2_4col` run at last, and it loses (files: `findings/F4-fused2-4col-negative.md`)

**T4.1** (`bb858c6`): hoisted the 16 ring-word loads in
`linear_gemv_ishizuki_k3_fused2_4col` above the first `extract_v_mcg_32`, matching
`_fused2_2col` (`linear_repacked.metal:491-495`) and the T4.3 hoist in
`_colmajor_2col_sk2` (`:2477-2480`). Without it the kernel holds 1 word pair in flight
against `_2col`'s 4, and A3 predicts the A/B would measure the missing hoist rather than
column tiling.

**T4.2**: bit-exactness first, then the A/B. One binary, `--temperature 0 --show-ids`:

```
armA (default, _fused2_2col)   4ce9ef31cfe358957ce89573aa72864e3be638123a6090778200504bdd782d4f
armB (FUSED2_4COL=1, 4col)     4ce9ef31cfe358957ce89573aa72864e3be638123a6090778200504bdd782d4f
```

128/128 ids identical => 4 columns per threadgroup is bit-exact here (stronger than the
`contig4` siblings, which only agree after fp16 rounding). Then 6 interleaved pairs:
**A median 16.021, B median 15.907, B slower in 6/6.**

**The box was unusually quiet** — within-arm spread 0.03 tok/s, not the recorded 35%.
That is why 6 pairs resolved a 0.7% effect; do not assume the same power on a busy box.

**Also noted:** the first arm-A run after a rebuild died with
`Metal command buffer failed (unlabelled) status=Error` and the immediate rerun was
clean. A leftover crashed process (the `perf_probe_splitk_gridshape` segfault, below)
appears to have left the driver in a bad state. Treat a single Metal failure as
suspicious-but-retryable, and never conclude from one run in either direction.

**`perf_probe_splitk_gridshape` segfaults** (exit 139) in `KAPLUMBA_PROBE_WEIGHTS` mode
after printing its header, in `split_factor_sweep`. Fix or delete it; do not cite its
absent rows. Its own header already disclaims its variant ranking, so the loss is small.

## Verified bytes/token — the number the ceiling rests on

Read from the `Gxma-repacked` safetensors headers directly (all shards, all `.trellis`):

```
total                                    12.0395 GB
  words_per_tile=48  (3 bpw)             10.9825 GB
  words_per_tile=96  (6 bpw, lm_head)     1.0570 GB

shape (1344,336,48) x120   5202.25 MB   gate/up
shape (336,1344,48) x 60   2601.12 MB   down
shape (336,16384,96) x  1  1056.96 MB   lm_head (row-major, K=5376 N=262144)
shape (256,336,48) x100     825.75 MB
shape (336,512,48) x 50     825.75 MB
shape (512,336,48) x 50     825.75 MB
```

So 12.039 GB is exact and the F3/F4 arithmetic is not resting on a stale figure.

## T6 — the wide lane window, analysed out (file: `findings/F6-wide-window-analysed.md`)

Recovered the bit→(output,input) map from `pack_integer_windows`
(`crates/engine/src/exl3.rs:330-353`) plus `extract_v_mcg_32` / `_fused2_2col`:

```
R(o,i) = 96*(o mod 8) + 24*g + 755 + 12*(o>=8) + 3*idx(i)
```

One output's two adjacent input groups are a contiguous 48-bit span, so 16 lanes x 48 bits
tile the 768-bit ring exactly — **no repack is needed**, the ring order is unchanged.
But the window start is `755 + 48L`, so `sh = (755+48L) mod 32 = 19 + 16L mod 32`, which is
**19 or 3, never 0**. At sh=19 a 48-bit span needs bits [19,67) = **3 words**:

```
current  2 words = 64 bits /  8 codes (24 useful) -> 2.67x
wide     3 words = 96 bits / 16 codes (48 useful) -> 2.00x
"ideal"  2 words = 64 bits / 16 codes             -> 1.33x   <- unreachable
```

And per code, `x` loads do not improve (0.50 B either way), so total load traffic per
code goes 1.50 -> 1.25 B, **-17%**, not -50%. Arm (c) — which removes *all* DRAM cost at
the same instruction count — is only +11%. **Declined.**

If anyone wants the cheap falsifier before discarding it: one probe arm, replica with a
single `uint2` instead of two `u32` at the same address. Odd-`j0` lanes cannot use
`uint2` (the recorded `uint4`/`ulong2` dead end), so it is a 50%-of-lanes experiment on
the exact axis.

## Live per-kernel numbers after the F5 correction

| kernel | own ms/step | bytes/token | GB/s |
|---|---:|---:|---:|
| gate_up_gelu | 25.58 | 5.202 GB | 203.4 |
| down_proj | 13.10 | 2.601 GB | 198.6 |
| qkv_proj | 8.62 | 1.652 GB | 191.6 |
| qk_proj | 1.95 | 0.372 GB | 190.4 |
| o_proj | 6.22 | 1.156 GB | 186.0 |
| head/gemv | 4.11 | 1.057 GB | ~257 |

186-203 GB/s across every 3 bpw projection. `KAPLUMBA_ABLATE=o_proj_gemv`:
16.001 -> 17.865 tok/s (+11.6%, 6 pairs), so o_proj's 6.2 ms is real and the profiler is
not mis-attributing. `KAPLUMBA_O_NOSK=1` -1.4% (6/6), `KAPLUMBA_O_SK4=1` -0.4%.

## Open, with the current best guess

- **Q1** the two-gate_proj 10% spread → placement, worth ~10% if model-wide.
- **Q2** how much of arm (G)'s 634 transfers → arm (C)'s 568 is the honest target.
- **Q3** does the synthetic 2.3× shape spread appear on real weights → T1.
- Latency vs pattern for the 11% → T2. Unknown, and it decides the fix.
