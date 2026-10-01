# IMPLEMENTATION NOTES: 17-18 tok/s Decode

Running log per AGENTS.md §5.

---

## 0. Prior art and what NOT to redo

This spec builds on top of the completed work in `decode-25tok-peak-bandwidth` (T1-T5.6)
and `decode-3bpw-amplification` (T1 negative). The standing record:

- **Baseline: 15.85 tok/s (63.1 ms/step)**, commit `fc36ee1`, re-measured in
  `decode-25tok-peak-bandwidth/implementation_notes.md` §16.1 with spread 0.03 tok/s.
- **GEMV floor: 58.3 ms = 17.2 tok/s** with zero overhead, from the 11.834 GB/token
  weight budget at 203 GB/s best measured 3 bpw bandwidth.
- **10 dead ends formally closed** — do not retry any of them:
  lane-contiguous repack (impossible), 4 bpw (−6.7%), split-K>2 (negative),
  4-col (−4.2%), uint4 loads (−3.8%), simd_shuffle (−18.8%), remove dequant (−26.7%),
  megakernel (no grid barrier), elementwise fusion (actual 0.8ms not 3-4ms),
  1-col streaming (−0.16 tok/s).
- **ICB tried and failed** (per user report, not in this repo's commits).

### The two novel ideas in this spec

1. **o_proj at 100 GB/s is at HALF the bandwidth of every other projection.**
   Root cause: K=8192 is too short for the DRAM prefetcher to warm up before the
   GEMV ends. Microbenchmark confirms: K=8192 → 120.7 GB/s, K=21504 → 199.3 GB/s.
   The fix: overlap o_proj weight prefetching with the preceding attention phase,
   so the prefetcher is already warm when o_proj starts computing.

2. **Cross-layer norm fusion** saves ~118 dispatches (59 layer boundaries × 2) by
   fusing layer L's finalize+norm+residual with layer L+1's input_norm+transform
   into one dispatch.

---

## 1. Commands

```bash
# GPU state
scripts/gpuq status

# Baseline measurement
M=~/.kaplumba/models/Gxma-repacked
scripts/gpuq try -- ./target/release/kaplumba --model "$M" \
  --prompt "Count from 1 to 100: 1, 2, 3," --max-new-tokens 64 \
  --temperature 0 --metrics

# Per-kernel profile
KAPLUMBA_DECODE_PROFILE=kernel scripts/gpuq try -- ./target/release/kaplumba \
  --model "$M" --prompt "Count from 1 to 100: 1, 2, 3," \
  --max-new-tokens 32 --temperature 0 --metrics

# Full test suite
scripts/gpuq try -- cargo test --test live_kernel_bit_exact
scripts/gpuq try -- cargo test --test fused_bit_exact
scripts/gpuq try -- cargo test --test repacked_golden
scripts/gpuq try -- cargo test --test argmax_engine_parity
scripts/gpuq try -- cargo test --test norm_rope_qkv_bit_exact
```

---

## 1a. Task T2a RESULT: ❌ NEGATIVE. The spin-wait saturates the memory pipe.

T2a was implemented in full and removed again. The measurement is in
`crates/engine/tests/warmstart_starvation.rs`; this is the reading of it.

**What was built.** `linear_gemv_ishizuki_k3_colmajor_2col_sk2_warmstart`
(`linear_repacked.metal`), which is `colmajor_2col_sk2` with the first 2 unrolled
K-steps' weight tiles read into registers before anything touches `x`, then a
spin on `device atomic_uint attn_done` before the first `x` read. Plus
`linear_attn_done_signal` (one thread, stores the sequence number), the encoders
`encode_gemv_raw_sk2_warmstart` / `encode_attn_done_signal`, `attn_done` +
`attn_stall` scratch buffers, a process-monotonic sequence counter, and the
`model.rs` wiring that dispatches o_proj ahead of attention. Correctness
restructuring was careful: the wait is unconditional and sits ahead of *every*
`x` read, the arithmetic after it is in the same order as the plain kernel (so
bit-exact, not merely close), and the spin is bounded with a device-side stall
latch so a starved producer is a diagnosable wrong answer rather than a hang.

**It compiled, loaded, and died at load:**
`invalid layer configuration: Metal command buffer failed during warmup decode`.

**Why.** The first test run hung for 900 s. That was not a slow test, it was the
finding: at the production grid the spinner does not converge. Lowering the spin
bound to 2e6 iterations so the test returns a number in seconds:

| spinning threadgroups (x 256 thr) | producer ran | spin timed out | wall |
|---:|---|---|---:|
| 1 | **false** | false | 510 ms |
| 8 | true | **true** | 1,019 ms |
| 32 | true | **true** | 2,169 ms |
| 64 | true | **true** | 4,102 ms |
| 128 | true | **true** | 7,565 ms |
| 168 | false | **true** | 7,285 ms |
| **336 (production)** | true | **true** | **18,600 ms** |
| 672 | true | **true** | 35,259 ms |

Two numbers matter more than the table.

1. **18,600 ms for 2e6 spin iterations across 336 groups is ~9.3 us per
   iteration.** A relaxed atomic load from L2 is ~ns. So the spinners are not
   waiting on the flag; they are saturating the memory pipe that both they and
   the producer need. 336 x 256 = 86,016 threads re-reading one 4-byte line is
   enough to do it.
2. **Wall time is linear in spinner count** (1.0 s at 8 -> 18.6 s at 336 -> 35.3
   at 672). The overlap is not overlapping: each extra spinner costs its own
   wait. That is the opposite of what a prefetch-and-overlap is supposed to do.

**This refines T1 rather than contradicting it.** T1 measured that the flag *does*
get set, which is true and still true -- at 336 groups the producer ran. T1's
test used 1 group of 256 and 1 of 32, i.e. 0.4% of the machine, so it had no way
to see the saturation, and its result (`spin_count=0`, flag already set) is
equally consistent with the producer having simply run first. **A feasibility
probe at 0.4% occupancy does not establish feasibility at 100%**, and the two
answers differ. T1's gate was therefore answered but under-specified.

**Do not retry the spin.** It is not a tuning problem: the mechanism spends the
resource the overlap was supposed to save. `tasks.md` T2b is the spec's own
answer to this — an `MTLEvent` is a hardware wait, so it enforces the same
producer/consumer ordering with *no thread spinning and no memory traffic*, and
it is the only variant of this idea left worth building.

**Metal has no acquire/release for a plain `device atomic_uint`.** Worth
recording because it cost three compile cycles: `atomic_store_explicit`'s
non-relaxed orders exist only on the scope-templated `device _atomic<T, S>`
overloads, which a plain `device atomic_uint*` does not match, so
`memory_order_seq_cst` is a *compile error*, not a slower path. Relaxed is the
only spelling available. It would have been sufficient here (the producer and
consumer are in different dispatches, so Metal's inter-dispatch visibility
carries the ordering) -- but that reasoning is only sound because of the
dispatch boundary, which the spin design was at least honest about.

---

## 2. Task T1 Log: Concurrent Threadgroup Scheduling Feasibility

- **Test**: `crates/engine/tests/concurrent_tg_feasibility.rs`
- **Objective**: Verify whether Apple Metal compute command encoder schedules threadgroups from separate dispatches within the same command buffer concurrently without CPU intervention, or if dispatches serialize strictly.
- **Mechanism**:
  - Kernel A (`signal_after_work`): executes 100,000 iterations of FMA busy loop across 256 threads (~1 ms work), then issues `atomic_store_explicit(&flag, 1u, memory_order_relaxed)`.
  - Kernel B (`wait_and_mark`): dispatched in the same command buffer, 32 threads, spin-waits on `atomic_load_explicit(&flag, memory_order_relaxed) == 0u` up to 500M spins, writes `0xCAFE` on seeing flag `1u`, or `0xBADD` on timeout.
- **Measurement**:
  - `cargo test --test concurrent_tg_feasibility -- --nocapture`
  - Result: `Wall time: 8.2 ms`, `Marker: 0xCAFE`, `Kernel B ran: yes`, `Spin count: 0`.
  - Kernel B observed flag set with 0 GPU timeout and zero host stalls. Metal's hardware threadgroup scheduler on M4 Pro actively dispatches threadgroups across available compute units concurrently when resources allow.
- **Architectural implication**:
  - Validates **T2a (Warmstart GEMV kernel)**: we can dispatch `o_proj` weight loading concurrent with `attention`, having `o_proj` threadgroups prefetch weight tiles into registers/L2 before attention writes the final input activation `scratch.attended`.
  - Eliminates the need for complex dual-queue `MTLEvent` host synchronization for o_proj latency hiding.

