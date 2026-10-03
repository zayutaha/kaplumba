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

## Open, with the current best guess

- **Q1** the two-gate_proj 10% spread → placement, worth ~10% if model-wide.
- **Q2** how much of arm (G)'s 634 transfers → arm (C)'s 568 is the honest target.
- **Q3** does the synthetic 2.3× shape spread appear on real weights → T1.
- Latency vs pattern for the 11% → T2. Unknown, and it decides the fix.
