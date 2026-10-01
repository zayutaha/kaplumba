# TECHNICAL PLAN: Decode at the EXL3 3 bpw Floor — 16 tok/s

Implements `spec/decode-3bpw-amplification/spec.md`. **Rewritten after T1 returned a
negative** — the layout/amplification programme it originally described is closed by
`spec.md` §3 and is not attempted here.

---

## 1. What changed and why

The previous plan was built on: *place lane L's 8 codes at bit offset `24·L`, reach
amp 2.000, 199 → 250 GB/s, ~+22%.* That is disproven. `decode_state_mcg` multiplies
before it masks, so each lane's 16-bit decode windows depend on **37** ring bits, not
24; the union over all 32 lanes is also 37, because every lane reads the same window at
a different rotation. Two words per lane is a floor. amp is `64/(8·KA)` — set by the bit
rate, not the layout.

So there is no new layout, no new kernel family, and no new checkpoint in this plan.
The remaining work is three small, independent changes to code that already exists, plus
acceptance.

## 2. Where the time actually goes

60-layer decode step, 65.3 ms, from `decode-25tok-peak-bandwidth/implementation_notes.md:190-200`
and `findings/07`:

```
59.1 ms  GEMV  (gate_up 25.50 | down 14.20 | qkv 8.58 | o 6.66 | head 4.20)
 4.8 ms  elementwise + norms   (~1167 dispatches, ~500 of them tiny)
 1.2 ms  attention KV streaming
```

The GEMVs are at their format floor (§3 of the spec) — not addressable. The addressable
mass is the **4.8 ms of elementwise/norm work**, and `findings/07` localises most of it
to a latency problem, not a bandwidth problem:

- 4 one-group norm kernels: **2.665 ms**, 56% of the non-GEMV tax
- `kv_norm_rope`: 0.8598 → 0.8622 ms across an unrelated change, i.e. **2.8× unexplained**
  and needing one `KAPLUMBA_NO_NORM_ROPE_QKV` A/B before anything is spent on that row
- a dispatch costs **3.5 µs of latency and ~0 of work** — which is why fusion
  arithmetic collapsed from 9.0 ms claimed to ~0.8 ms measured

The norm kernel's shape is the actual problem: 256 threads, `for (i = lid; i < 5376;
i += 256)` = **21 dependent loads at 512 B stride**, 8 `threadgroup_barrier`s, then a
second 21-load pass and 21 stores. 8 warps on one core have nothing to hide 21 round
trips behind: **21 × ~500 ns ≈ 10.5 µs against 11.2 µs of measured work.** Work
bandwidth runs 4.6–133 GB/s, 2–53% of peak. *"Latency problems wearing bandwidth's
clothes."*

## 3. File & module touchpoints

| path | current behaviour | intended change |
|---|---|---|
| `crates/kernels/src/elements.metal` | `rms_norm_weighted_tg` and siblings: 2-pass strided walk, 8 barriers, `threadgroup partial[]` | T2: hoist the second pass's `weight`/`residual` loads above the barrier. T3: fuse the four measured-safe pairs. |
| `crates/engine/src/elements.rs` | one `encode_*` per elementwise op | T3: fewer `encode_*` calls, one per fusion |
| `crates/engine/src/model.rs` | per-layer attention + GEMV + ~500 elementwise dispatches, all on one queue | T4: second `MTLCommandQueue` + `MTLEvent` for attention/GEMV overlap |
| `crates/kernels/src/encoder.rs` | owns one `CommandBuffer` + `ComputeCommandEncoder` + `&CommandQueue` (`encoder.rs:59-97`); `set_max_dispatches(2500)` per step so a decode step commits 2 buffers | T4: add a second encoder bound to the secondary queue. `max_dispatches`/`roll()` semantics unchanged. |

### 3.1 Explicitly not touched

- `linear_repacked.metal`, `linear.rs`, `exl3.rs` — the GEMV path is at its floor (§3 of
  the spec). No relayout, no repack, no new kernel.
- `decode_state_mcg` (`linear.metal:121-126`) — narrowing its 16-bit window to 3 bits is
  the only route past 199 GB/s and it changes every output value. Barred.
- `lm_head` — 6 bpw, EXL3 arm, already near ceiling.
- The `simd_shuffle` overlap idea — measured 180.5 → 146.5 GB/s, closed.

## 4. Sequencing

```
T1  amplification relayout ................ CLOSED, negative (spec §3)
     |
T2  norm second-pass hoist ............... 0 – 1.2 ms, bit-exactness-frozen axis
     |                                     measure before keeping; may be worth 0
T3  four remaining norm fusions ........... ~1.05 ms total, each A/B'd separately
     |
T4  attention/GEMV overlap ............... 0.6 – 1.5 ms, never attempted, no precedent
     |
T5  acceptance ............................ AC1 ≥ 16.0 tok/s, AC6 negatives recorded
```

T2 and T3 are GPU-light and can proceed while the box is busy elsewhere; T4 and T5 need
an idle GPU and are serialised per AGENTS.md §10. Each task commits independently, and
**each is allowed to fail** — a task that does not beat baseline is reverted and its
number recorded (AC6). None of them blocks the next.

| requirement | tasks |
|---|---|
| R1 bit-exactness | T2, T3, T4, T5 |
| R2 measure before believing | T2, T3, T4, T5 |
| R3 sabotage evidence | T2, T3, T4 |
| R4 no regression | T5 (AC6 baseline re-measured in-session) |
| R5 residency | T5 |

## 5. Risks and fallbacks

| # | risk | mitigation |
|---|---|---|
| 1 | **T2 is worth 0.** The record says so explicitly: "may be worth 0 — do not budget before measuring". | Measure first, keep only if it wins. The second pass's loads may already be covered by the barrier. |
| 2 | **T2 breaks bit-exactness.** The 256-lane strided partial + tree *is* the reduction order. `partial[lid] = (lid==0)?sum:0` changes 8192/8192 elements, worst 9266 ULP. | Do not change lane→element mapping, threadgroup width, or the two-stage structure. Hoisting loads only. Assert `to_bits()` on every element. |
| 3 | **T4 is a race that hides.** Attention→o_proj ordering. A 1-in-10⁴ race is worse than no overlap. | Synchronisation proven with an input-poison test over ≥ 1000 steps before any timing is believed. Note the o_proj fusion precedent: 60 dispatches removed for **+0.22%, t=1.32, not significant** (`findings/08:46-58`). |
| 4 | **A contended run fabricates a win.** `gpuq`'s lock is engine-only, so `gpuq try -- <bench>` runs happily beside a 12 GB model load. One sweep returned plausible medians at spread 0.46 with a fabricated **+34%**. | Gate on `gpuq status` showing `gpu: none`; interleave A/B; min-based estimator; report spread. |
| 5 | T4 needs a second queue and the encoder abstraction owns one. | `MTLCommandQueue` + a second `MetalEncoder`. `in_flight` is back-pressure, not concurrency, so nothing existing depends on there being one queue. |
| 6 | The whole programme lands under target. | AC1 is 16.0 tok/s, not 20 or 25. If T2–T4 deliver 1 ms, that is the honest result and gets recorded as such. |

### 5.1 Measurement traps already paid for in this repo

- `peak_resident_pages_bytes` is `ru_maxrss`; it missed 5.5 GiB of Metal pages. Use
  `/usr/bin/time -l` `peak memory footprint`.
- The running-LCG tile generator makes the colmajor repack the identity, so layout checks
  against it are vacuous. `bench_residency` and `live_kernel_bit_exact` still have it.
- `LinearKernels::new()` returning `Err` once failed **open** — `else { return; }` turned a
  missing `)` into a green test. Third instance. New harnesses must `expect()` it.
- A **1-element** value flip is below the fp16 noise floor and can never be used as a
  sabotage; even 256 1-ULP flips in one value row move 0 of 8192 outputs.
- `dp_wall_ms_per_token` read 27.568 against a 70.239 span — impossible for a step ending
  in a barrier. Do not quote `dp_wall_*` from a step-mode run.
- Unreconciled: 11.2 µs vs 16.7 µs for one dispatch of `rms_norm_weighted_tg` at the same
  shape. Reconcile before either number is used to justify anything.
