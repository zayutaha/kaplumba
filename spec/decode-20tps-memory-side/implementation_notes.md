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

## Open, with the current best guess

- **Q1** the two-gate_proj 10% spread → placement, worth ~10% if model-wide.
- **Q2** how much of arm (G)'s 634 transfers → arm (C)'s 568 is the honest target.
- **Q3** does the synthetic 2.3× shape spread appear on real weights → T1.
- Latency vs pattern for the 11% → T2. Unknown, and it decides the fix.
