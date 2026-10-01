# tasks — ICB decode replay

## T1 — Verify the gap on an idle GPU
- [ ] Measure true CPU encode-submit cost without profiler distortion
- [ ] Command: `KAPLUMBA_PROFILE_PHASES=1 scripts/gpuq run -- ./target/release/kaplumba --model ~/.kaplumba/models/Gxma/Gxma --prompt "The capital of France is" --max-new-tokens 40 --temperature 0`
- [ ] Read the "ENCODE (all layers' host encode-submit cost" line
- [ ] **Gate:** if gap < 15 ms/token, ICB is not worth building. Record the number and stop.
- [ ] Files: none (measurement only)
- **Status:** pending

## T2 — Audit dispatch call sites and kernel buffer counts
- [ ] Enumerate all `dispatch_*` methods on `MetalEncoder` (dispatch_1d, dispatch_2d, dispatch_groups, dispatch_1d_offset, others?)
- [ ] Enumerate all `encode_*` functions that take `&mut MetalEncoder`
- [ ] Find the kernel with the most buffer arguments (for `max_kernel_buffer_bind_count`)
- [ ] Audit the decode path for slot-dependent buffers (Q4)
- [ ] Files: `crates/kernels/src/encoder.rs`, `crates/kernels/src/elements.rs`, `crates/kernels/src/attention.rs`, `crates/kernels/src/linear.rs`, `crates/engine/src/model.rs`
- **Status:** pending

## T3 — Define the Dispatch trait
- [ ] New file `crates/kernels/src/dispatch.rs` with the `Dispatch` trait
- [ ] Methods: dispatch_1d, dispatch_2d, dispatch_groups, dispatch_1d_offset (whatever T2 finds)
- [ ] Implement for `MetalEncoder` — identical to current behavior
- [ ] No behavior change to existing code
- [ ] Files: `crates/kernels/src/dispatch.rs` (new), `crates/kernels/src/encoder.rs` (add impl)
- **Status:** pending

## T4 — Refactor encode_* functions to use Dispatch trait
- [ ] Change `&mut MetalEncoder` to `&mut impl Dispatch` in all encode_* functions
- [ ] Compiler-guided: mechanical change
- [ ] Verify all existing tests pass (no behavior change)
- [ ] Files: `crates/kernels/src/elements.rs`, `crates/kernels/src/attention.rs`, `crates/kernels/src/linear.rs`
- **Status:** pending

## T5 — ICB module: IcbRecorder
- [ ] New file `crates/kernels/src/icb.rs`
- [ ] `IcbRecorder` implements `Dispatch` — writes into ICB at next command index
- [ ] Descriptor setup: ConcurrentDispatch, inherit_buffers=false, inherit_pipeline_state=false, max_kernel_buffer_bind_count from T2
- [ ] `device.new_indirect_command_buffer_with_descriptor(desc, 906, options)`
- [ ] For each dispatch: `indirect_compute_command_at_index(i)` + set pipeline/buffers + concurrent_dispatch_threadgroups
- [ ] Files: `crates/kernels/src/icb.rs` (new)
- **Status:** pending

## T6 — ICB module: IcbPlayer (replay)
- [ ] `execute_commands_in_buffer(&icb, range)` — one CPU call
- [ ] 2 ICBs: one per KV cache slot
- [ ] Slot alternation: `icb[slot % 2]`
- [ ] Files: `crates/kernels/src/icb.rs`
- **Status:** pending

## T7 — Wire ICB into the decode loop
- [ ] First decode token: record ICB (run forward pass with IcbRecorder)
- [ ] Subsequent tokens: replay ICB
- [ ] Slot alternation: record 2 ICBs, replay alternately
- [ ] Prefill: unchanged (normal dispatch path)
- [ ] Files: `crates/engine/src/model.rs`
- **Status:** pending

## T8 — Bit-exactness test
- [ ] Run 100+ tokens through normal path, record digest
- [ ] Run 100+ tokens through ICB path, record digest
- [ ] Digests must be identical
- [ ] Files: `crates/engine/tests/icb_bit_exact.rs` (new)
- **Status:** pending

## T9 — Prefill regression test
- [ ] Prefill TTFT unchanged (within noise)
- [ ] Files: none (measurement only)
- **Status:** pending

## T10 — A/B measurement
- [ ] Interleaved: normal vs ICB, 5 runs each
- [ ] Report median + spread
- [ ] Target: >= 18 tok/s
- [ ] Files: none (measurement only)
- **Status:** pending

## T11 — Record result
- [ ] If ICB reaches 18 tok/s: commit, update spec
- [ ] If not: record negative with numbers, track fusion fallback as separate spec
- [ ] Files: `spec/decode-icb/implementation_notes.md`
- **Status:** pending
