# plan — ICB decode replay

## Architecture

```
decode token
  |
  v
[record mode, first token]          [replay mode, subsequent tokens]
  |                                    |
  v                                    v
encode_* functions                    encoder.execute_commands_in_buffer(icb[slot], range)
  |                                    ^
  v                                    |
Dispatch trait                        one CPU call replays all 906 kernels
  ^
  |
  +-- MetalEncoder (normal, unchanged behavior)
  +-- IcbRecorder (new, writes into ICB)
```

The core idea: abstract the dispatch calls behind a trait so the same `encode_*`
functions can write to either a normal command buffer or an ICB recorder. The
ICB recorder captures each dispatch into an ICB. After recording, replay is a
single `execute_commands_in_buffer` call.

## File touchpoints

| file | change | intent |
|---|---|---|
| `crates/kernels/src/icb.rs` | **new** | ICB record/replay. `IcbRecorder`, `IcbPlayer`, descriptor setup. |
| `crates/kernels/src/encoder.rs` | **additive** | Add `Dispatch` trait. Implement for `MetalEncoder`. No behavior change. |
| `crates/kernels/src/dispatch.rs` | **new** | The `Dispatch` trait definition (or in `icb.rs`). |
| `crates/engine/src/model.rs` | **modify** | Decode loop: record on first token, replay on subsequent. Slot alternation. |
| `crates/kernels/src/elements.rs` | **modify** | `encode_*` functions take `&mut impl Dispatch` instead of `&mut MetalEncoder`. |
| `crates/kernels/src/attention.rs` | **modify** | Same trait change for attention encode functions. |
| `crates/kernels/src/linear.rs` | **modify** | Same trait change for GEMV encode functions. |

## The Dispatch trait

```rust
pub trait Dispatch {
    fn dispatch_1d(
        &mut self,
        pipeline: &ComputePipelineState,
        buffers: &[Option<&Buffer>],
        params: &[u8],
        threads: u64,
        threads_per_tg: u32,
    ) -> Result<(), KernelError>;

    fn dispatch_2d(&mut self, ...);
    fn dispatch_groups(&mut self, ...);
    fn dispatch_1d_offset(&mut self, ...);
}
```

- `MetalEncoder` impl: identical to current behavior (encode into command buffer).
- `IcbRecorder` impl: write into ICB at the next command index.

The trait is the **only** change to existing code, and it is purely additive —
a new trait + a new impl for the existing encoder. No existing behavior changes.

## Recording process

1. Create ICB descriptor: `command_types = ConcurrentDispatch`,
   `inherit_buffers = false`, `inherit_pipeline_state = false`,
   `max_kernel_buffer_bind_count = N` (audit kernel signatures for max).
2. Create ICB: `device.new_indirect_command_buffer_with_descriptor(desc, 906, options)`.
3. Run the decode forward pass with `IcbRecorder` as the dispatch target.
4. For each dispatch, `IcbRecorder` calls:
   - `cmd = icb.indirect_compute_command_at_index(i)`
   - `cmd.set_compute_pipeline_state(&pipeline)`
   - `cmd.set_kernel_buffer(j, Some(&buffer), offset)` for each buffer
   - `cmd.concurrent_dispatch_threadgroups(tg_per_grid, threads_per_tg)`
5. After recording, the ICB is immutable and replayable.

## Replay process

1. On each decode token, call `encoder.execute_commands_in_buffer(&icb[slot], range)`.
2. One CPU call replays all 906 kernels.
3. The GPU executes them with zero per-launch CPU overhead.

## KV cache slot alternation

The decode path uses `slot % 2` for KV cache buffers. Token ID and position also
live in per-slot buffers. So:
- Record 2 ICBs: `icb[0]` for slot 0, `icb[1]` for slot 1.
- Each ICB points at its slot's token/position/KV buffers.
- Replay alternates: `icb[slot % 2]`.

This is the simplest correct approach. ICB memory cost is tiny (906 commands x
~100 bytes = ~90 KB per ICB).

## Bit-exactness

ICB replays the exact same kernels with the exact same arguments. The only
difference is the dispatch mechanism (ICB vs command buffer). Metal guarantees
that ICB replay produces identical results to normal dispatch. Verified by
digest comparison (R4).

## Risks

1. **Trait abstraction scope.** The `encode_*` functions are numerous and take
   `&mut MetalEncoder`. Changing them to `&mut impl Dispatch` touches many files.
   *Mitigation:* mechanical change, compiler-guided. The trait is additive.
2. **ICB API correctness.** `concurrent_dispatch_threadgroups` vs
   `dispatch_threadgroups` may have subtle differences. *Mitigation:* bit-
   exactness test (R4) catches any difference.
3. **Buffer bind count.** The ICB descriptor's `max_kernel_buffer_bind_count`
   must be >= the max buffer count of any kernel. *Mitigation:* audit kernel
   signatures, set conservatively high.
4. **Per-slot buffers.** If there are slot-dependent buffers beyond token,
   position, and KV cache, the 2-ICB approach breaks. *Mitigation:* audit the
   decode path (Q4).
5. **ICB memory.** 906 commands x 2 ICBs. *Mitigation:* measure during
   implementation. Expected ~180 KB total, negligible.
6. **The gap is smaller than measured.** The profiler distorts absolutes.
   *Mitigation:* T1 verifies the gap on an idle GPU before building ICB.

## Requirement -> task mapping

| requirement | tasks |
|---|---|
| R1 (verify gap) | T1 |
| R2 (ICB module) | T2, T3, T4 |
| R3 (2 ICBs) | T5 |
| R4 (bit-exact) | T6 |
| R5 (dispatch trait) | T7, T8, T9 |
| R6 (no prefill regression) | T10 |
| R7 (measurement) | T11 |

## Sequencing

1. **T1** (verify gap) — must pass before anything else. If gap < 15 ms, stop.
2. **T7-T9** (dispatch trait) — additive, no behavior change. Land first.
3. **T2-T4** (ICB module) — new module, depends on trait.
4. **T5** (2 ICBs + slot alternation) — depends on ICB module.
5. **T6** (bit-exactness) — depends on everything.
6. **T10** (prefill regression) — verify prefill unchanged.
7. **T11** (A/B measurement) — final proof.
