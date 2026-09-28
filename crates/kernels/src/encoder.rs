use std::ffi::c_void;
use std::marker::PhantomData;

use metal::{
    Buffer, CommandBufferRef, CommandQueue, ComputeCommandEncoderRef, ComputePipelineState,
    MTLCommandBufferStatus, MTLSize,
};

use crate::exl3::KernelError;

/// A buffer plus the byte offset of a tensor inside it.
///
/// Zero-copy weights are bound over a whole shard mapping, so a tensor can start
/// at an arbitrary (non-Metal-aligned) offset within the bound range.
#[derive(Clone, Copy)]
pub struct TensorRef<'a> {
    pub buffer: &'a Buffer,
    pub offset: u64,
}

impl<'a> TensorRef<'a> {
    pub fn new(buffer: &'a Buffer, offset: u64) -> Self {
        Self { buffer, offset }
    }
}

/// A reusable Metal compute command encoder.
///
/// The per-op execution path used during bring-up created, committed, and waited
/// on a command buffer for every kernel, which forced a full CPU/GPU sync per op
/// (~1200 per generated token). `MetalEncoder` lets a caller batch many dispatches
/// into a single command buffer and sync once, which is the dominant throughput
/// win for decode.
///
/// Dispatches encoded into one encoder execute in order and their memory writes
/// are visible to subsequent dispatches, so buffers may be reused as ping-pong
/// scratch within a single batch.
pub struct MetalEncoder<'q> {
    command_buffer: &'q CommandBufferRef,
    encoder: &'q ComputeCommandEncoderRef,
    /// Kept so `flush_and_wait` can start a fresh command buffer; a compute
    /// encoder cannot be used after `end_encoding`.
    queue: &'q CommandQueue,
    /// Committed-but-unfinished command buffers, oldest first, bounded by
    /// `MAX_IN_FLIGHT`. Waiting on the oldest when the window overflows is what
    /// applies back-pressure: the GPU is never starved, and the host is never
    /// allowed to queue without limit.
    in_flight: Vec<&'q CommandBufferRef>,
    /// Encoded dispatches in the current command buffer. A buffer is committed
    /// once it reaches `MAX_DISPATCHES_PER_BUFFER`.
    dispatches: u32,
    /// Committed, un-waited, labelled buffers kept only to read their GPU
    /// timestamps, in commit order. Populated by [`Self::commit_stamped`] and
    /// drained by [`Self::take_gpu_stamps`]. It is deliberately *not* bounded
    /// by `MAX_IN_FLIGHT`: a profiling caller submits a handful of phase
    /// boundaries per forward and reads them all at the end, and an unbounded
    /// list of already-committed buffers costs no GPU memory (a finished
    /// buffer releases its resources).
    stamped: Vec<(String, &'q CommandBufferRef)>,
}

/// `MTLCommandBuffer` GPU timestamps, which metal-rs 0.33 does not bind even
/// though the Metal SDK has exposed them since macOS 10.15. The Objective-C
/// selectors are `GPUStartTime` / `GPUEndTime` (upper-case `GPU`; Swift
/// imports them as `gpuStartTime` / `gpuEndTime`) and both return an
/// `NSTimeInterval`, i.e. a `double`.
///
/// Both are **undefined until the command buffer has completed** — before that
/// Metal returns garbage, not zero — so read them only after
/// `wait_until_completed`. The values are seconds on a fixed GPU timeline
/// (not a Unix epoch and not host wall clock), so only differences between two
/// timestamps are meaningful.
pub trait CommandBufferGpuTiming {
    /// Seconds on the GPU timeline at which this buffer began executing.
    fn gpu_start_time(&self) -> f64;
    /// Seconds on the GPU timeline at which this buffer finished executing.
    fn gpu_end_time(&self) -> f64;
}

// `msg_send!` expands through objc's `sel_impl!`, which contains a
// `cfg(feature = "cargo-clippy")` written for the objc crate's own lint setup.
// The check-cfg pass evaluates that against *this* crate, and for a cfg coming
// from an external macro the lint level of the macro's own crate wins, so no
// local `allow` silences it. Registering the selector through the runtime
// function and calling the same `__send_message` that `msg_send!` expands to
// keeps the FFI identical and the build clean.
mod gpu_time {
    use std::ffi::CStr;

    use metal::CommandBufferRef;
    use metal::objc::__send_message;
    use metal::objc::runtime::{Sel, sel_registerName};

    use super::CommandBufferGpuTiming;

    /// `NSTimeInterval` is a `double`, so the message returns one `f64`.
    fn seconds(buffer: &CommandBufferRef, selector: &CStr) -> f64 {
        let selector: Sel = unsafe { sel_registerName(selector.as_ptr()) };
        let (seconds,): (f64,) = unsafe { __send_message(buffer, selector, ()) }
            .expect("GPUStartTime/GPUEndTime message");
        seconds
    }

    impl CommandBufferGpuTiming for CommandBufferRef {
        fn gpu_start_time(&self) -> f64 {
            seconds(self, c"GPUStartTime")
        }

        fn gpu_end_time(&self) -> f64 {
            seconds(self, c"GPUEndTime")
        }
    }
}

/// One measured command buffer, from the GPU's own clock.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuStamp {
    /// The label passed to [`MetalEncoder::commit_stamped`].
    pub label: String,
    /// GPU-timeline seconds at which the buffer started executing.
    pub start_s: f64,
    /// GPU-timeline seconds at which the buffer finished executing.
    pub end_s: f64,
    /// `end_s - start_s`: time the GPU spent executing this buffer. This is
    /// the meaningful number — it is the buffer's own cost, measured on the
    /// GPU clock, independent of what the host was doing.
    pub own_ms: f64,
    /// `start_s - previous stamp's end_s`: time the GPU sat idle between this
    /// buffer and the previous one. Not a GPU cost; it is submission latency,
    /// i.e. how far the host ran ahead of (or behind) the GPU. 0.0 for the
    /// first buffer, which has no predecessor.
    pub gap_ms: f64,
}

/// Dispatches per command buffer. MLX uses 50 on this hardware class
/// (`mlx/backend/metal/device.cpp`, arch `'s'` -> `max_ops_per_buffer_ = 50`,
/// alongside a 50 MB cap) and omlx ships 16 for its multi-rank default
/// (`omlx/cluster/deployment.py`). Those are measured values on an M4-class
/// part, not guesses, and they are the direct fix for our symptom: one buffer
/// holding a whole prefill (~8k dispatches) is what made the host spend 58 s
/// wall on work whose GPU time is a few seconds.
pub const MAX_DISPATCHES_PER_BUFFER: u32 = 50;

/// Command buffers allowed in flight. MLX's `MAX_ACTIVE_TASKS = 10`
/// (`mlx/transforms.cpp`); past it the host blocks on a condvar. We measured
/// `new_command_buffer()` blocking ~175 ms once ~64 buffers were outstanding,
/// so staying well under that is deliberate.
pub const MAX_IN_FLIGHT: usize = 8;

impl<'q> MetalEncoder<'q> {
    pub fn new(queue: &'q CommandQueue) -> Result<Self, KernelError> {
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        Ok(Self {
            command_buffer,
            encoder,
            queue,
            in_flight: Vec::new(),
            dispatches: 0,
            stamped: Vec::new(),
        })
    }

    /// Encode a 1D dispatch of `threads` threads into the current command buffer.
    pub fn dispatch_1d(
        &mut self,
        pipeline: &ComputePipelineState,
        bindings: &[Option<&Buffer>],
        params: &[u8],
        threads: u64,
        threads_per_group: u64,
    ) -> Result<(), KernelError> {
        if threads == 0 || threads_per_group == 0 {
            return Err(KernelError::InvalidInput(
                "kernel thread and threadgroup counts must be non-zero".to_string(),
            ));
        }
        self.bind_and_dispatch(
            pipeline,
            bindings,
            params,
            MTLSize::new(threads, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        )
    }

    /// Encode a 2D threadgroup dispatch into the current command buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_2d(
        &mut self,
        pipeline: &ComputePipelineState,
        bindings: &[TensorRef<'_>],
        params: &[u8],
        groups_x: u64,
        groups_y: u64,
        threads_per_group_x: u64,
        threads_per_group_y: u64,
    ) -> Result<(), KernelError> {
        if groups_x == 0 || groups_y == 0 || threads_per_group_x == 0 || threads_per_group_y == 0 {
            return Err(KernelError::InvalidInput(
                "dispatch group and threadgroup sizes must be non-zero".to_string(),
            ));
        }
        self.encoder.set_compute_pipeline_state(pipeline);
        for (index, binding) in bindings.iter().enumerate() {
            self.encoder.set_buffer(
                index as u64,
                Some(&**binding.buffer),
                binding.offset,
            );
        }
        self.encoder.set_bytes(
            bindings.len() as u64,
            params.len() as u64,
            params.as_ptr() as *const c_void,
        );
        self.encoder.dispatch_thread_groups(
            MTLSize::new(groups_x, groups_y, 1),
            MTLSize::new(threads_per_group_x, threads_per_group_y, 1),
        );
        self.tick()?;
        Ok(())
    }

    /// Encode a 1D dispatch whose bindings may sit at non-zero byte offsets.
    /// Dispatch `groups` threadgroups of `threads_per_group` threads.
    ///
    /// Needed by kernels that reduce across a threadgroup, which cannot use the
    /// 1D form because the thread-to-row mapping is per-group rather than global.
    pub fn dispatch_groups(
        &mut self,
        pipeline: &ComputePipelineState,
        bindings: &[Option<&Buffer>],
        params: &[u8],
        groups: u64,
        threads_per_group: u64,
    ) -> Result<(), KernelError> {
        if groups == 0 || threads_per_group == 0 {
            return Err(KernelError::InvalidInput(
                "kernel group and threadgroup counts must be non-zero".to_string(),
            ));
        }
        self.encoder.set_compute_pipeline_state(pipeline);
        for (index, binding) in bindings.iter().enumerate() {
            self.encoder
                .set_buffer(index as u64, binding.map(|value| &**value), 0);
        }
        self.encoder.set_bytes(
            bindings.len() as u64,
            params.len() as u64,
            params.as_ptr() as *const c_void,
        );
        self.encoder.dispatch_thread_groups(
            MTLSize::new(groups, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        self.tick()?;
        Ok(())
    }

    pub fn dispatch_1d_offset(
        &mut self,
        pipeline: &ComputePipelineState,
        bindings: &[TensorRef<'_>],
        params: &[u8],
        threads: u64,
        threads_per_group: u64,
    ) -> Result<(), KernelError> {
        if threads == 0 || threads_per_group == 0 {
            return Err(KernelError::InvalidInput(
                "kernel thread and threadgroup counts must be non-zero".to_string(),
            ));
        }
        self.encoder.set_compute_pipeline_state(pipeline);
        for (index, binding) in bindings.iter().enumerate() {
            self.encoder.set_buffer(
                index as u64,
                Some(&**binding.buffer),
                binding.offset,
            );
        }
        self.encoder.set_bytes(
            bindings.len() as u64,
            params.len() as u64,
            params.as_ptr() as *const c_void,
        );
        self.encoder
            .dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(threads_per_group, 1, 1));
        self.tick()?;
        Ok(())
    }

    fn bind_and_dispatch(
        &mut self,
        pipeline: &ComputePipelineState,
        bindings: &[Option<&Buffer>],
        params: &[u8],
        threads: MTLSize,
        threads_per_group: MTLSize,
    ) -> Result<(), KernelError> {
        self.encoder.set_compute_pipeline_state(pipeline);
        for (index, binding) in bindings.iter().enumerate() {
            self.encoder.set_buffer(index as u64, binding.map(|value| &**value), 0);
        }
        self.encoder.set_bytes(
            bindings.len() as u64,
            params.len() as u64,
            params.as_ptr() as *const c_void,
        );
        self.encoder.dispatch_threads(threads, threads_per_group);
        self.tick()?;
        Ok(())
    }

    /// Count one dispatch and roll to a fresh command buffer once the buffer is
    /// full, bounding how much work any single buffer carries.
    fn tick(&mut self) -> Result<(), KernelError> {
        self.dispatches += 1;
        if self.dispatches >= MAX_DISPATCHES_PER_BUFFER {
            self.roll()?;
        }
        Ok(())
    }

    /// Commit the current buffer without waiting and continue on a fresh one,
    /// keeping at most `MAX_IN_FLIGHT` buffers outstanding. When the window is
    /// full the oldest is waited on; by then it has almost always already
    /// finished, so this throttles submission only when the GPU is genuinely
    /// behind. Ordering between buffers is implicit: a single Metal command
    /// queue executes them in commit order.
    fn roll(&mut self) -> Result<(), KernelError> {
        self.encoder.end_encoding();
        self.command_buffer.commit();
        self.in_flight.push(self.command_buffer);
        if self.in_flight.len() > MAX_IN_FLIGHT {
            let oldest = self.in_flight.remove(0);
            oldest.wait_until_completed();
        }
        self.command_buffer = self.queue.new_command_buffer();
        self.encoder = self.command_buffer.new_compute_command_encoder();
        self.dispatches = 0;
        Ok(())
    }

    /// End the current compute encoder, commit, and wait, without consuming self.
    ///
    /// Lets a caller profile phases of one logical forward pass by splitting it
    /// across several command buffers. `KAPLUMBA_PROFILE_PHASES` uses this.
    ///
    /// The rolling window is part of "this phase": `tick` rolls every
    /// `MAX_DISPATCHES_PER_BUFFER` dispatches, so the work encoded since the
    /// last flush is spread over `in_flight` plus the current buffer. Draining
    /// it explicitly, instead of `clear()`ing it, keeps the barrier honest
    /// rather than resting on the argument that a single Metal queue executes
    /// in commit order — which is what makes waiting on the newest buffer
    /// already imply that every older one finished. That argument does hold
    /// today (every buffer here was committed to `queue`), so the old
    /// `clear()` was not a live miscompile, but it is precisely the invariant a
    /// future multi-queue change would break silently. Draining oldest-first
    /// also applies back-pressure in submission order, and taking the buffers
    /// out of the vector means a panic mid-drain cannot drop the tail.
    pub fn flush_and_wait(&mut self) -> Result<(), KernelError> {
        self.encoder.end_encoding();
        self.command_buffer.commit();
        for buffer in std::mem::take(&mut self.in_flight) {
            buffer.wait_until_completed();
        }
        self.command_buffer.wait_until_completed();
        // A compute encoder is single-use, so continue on a fresh buffer. The
        // old reference is borrowed from `queue`, so it stays valid for `'q`;
        // dropping our pointer to it is not what keeps the object alive.
        self.command_buffer = self.queue.new_command_buffer();
        self.encoder = self.command_buffer.new_compute_command_encoder();
        self.dispatches = 0;
        Ok(())
    }

    /// End the encoder and drop the work without committing. Used on error paths
    /// so the encoder is never released without `end_encoding`.
    pub fn abandon(&mut self) {
        self.encoder.end_encoding();
    }

    /// Commit without waiting, so several command buffers can be in flight.
    ///
    /// # This is a terminal operation
    ///
    /// Unlike [`Self::flush_and_wait`] and [`Self::commit_stamped`] it does
    /// *not* start a fresh command buffer and does *not* reset `dispatches`,
    /// because the only current caller (`generate.rs`, the pipelined decode
    /// loop) drops the encoder into a list and calls [`Self::wait`] on it —
    /// which takes `&self` and so cannot re-arm. Encoding anything else after
    /// this call is a use-after-`end_encoding` bug and Metal will raise an
    /// Objective-C exception, i.e. abort the process, rather than return an
    /// error: the `encoder` field still points at the encoder this call just
    /// ended, so the next `dispatch_*` would encode into it, and a second
    /// `commit_and_waitless` would end and commit the same buffer twice. The
    /// stale `dispatches` counter is a second, quieter symptom — it stays at
    /// whatever the last roll left it at, so a subsequent dispatch can roll an
    /// empty buffer.
    pub fn commit_and_waitless(&mut self) -> Result<(), KernelError> {
        self.encoder.end_encoding();
        self.command_buffer.commit();
        self.in_flight.push(self.command_buffer);
        Ok(())
    }

    /// Wait for this command buffer without consuming it, so a caller that queued
    /// several can read their results in submission order.
    ///
    /// The `in_flight` loop is redundant — every buffer here was committed to
    /// the one queue in commit order, and a Metal command queue executes in
    /// commit order, so waiting on the newest transitively waits on the older
    /// ones. It is kept explicit anyway: the loop is a handful of pointers and
    /// it makes the barrier independent of that ordering argument surviving a
    /// future change (e.g. spreading work over several queues).
    ///
    /// Takes `&self`, so it cannot drain `stamped`; a buffer parked there by
    /// [`Self::commit_stamped`] is waited on by [`Self::take_gpu_stamps`].
    pub fn wait(&self) {
        self.command_buffer.wait_until_completed();
        for buffer in &self.in_flight {
            buffer.wait_until_completed();
        }
    }

    pub fn commit_and_wait(mut self) -> Result<(), KernelError> {
        self.encoder.end_encoding();
        self.command_buffer.commit();
        self.command_buffer.wait_until_completed();
        // Buffers rolled out mid-forward are committed but unfinished. On one
        // in-order queue the wait above already covers them, but drain them
        // anyway so the barrier does not depend on that.
        for buffer in std::mem::take(&mut self.in_flight) {
            buffer.wait_until_completed();
        }
        if self.command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(KernelError::CommandFailed);
        }
        Ok(())
    }

    /// Commit the work encoded so far under `label` and continue on a fresh
    /// command buffer, without waiting.
    ///
    /// This is [`Self::commit_and_waitless`] plus two things: the buffer is
    /// labelled, and it is parked in `stamped` so [`Self::take_gpu_ms`] can
    /// read `GPUStartTime` / `GPUEndTime` off it later. Because the queue
    /// serialises command buffers, the retained buffers form an ordered
    /// timeline even though none of them is waited on here.
    ///
    /// Does not touch `in_flight` and does not apply `MAX_IN_FLIGHT`
    /// back-pressure, so a caller that stamps every phase of every token
    /// without ever calling [`Self::take_gpu_ms`] would grow `stamped` without
    /// bound; drain it periodically (it is also what bounds GPU work in
    /// flight, since nothing here waits).
    pub fn commit_stamped(&mut self, label: &str) -> Result<(), KernelError> {
        self.encoder.end_encoding();
        self.command_buffer.set_label(label);
        self.command_buffer.commit();
        self.stamped.push((label.to_string(), self.command_buffer));
        self.command_buffer = self.queue.new_command_buffer();
        self.encoder = self.command_buffer.new_compute_command_encoder();
        self.dispatches = 0;
        Ok(())
    }

    /// Wait on every buffer retained by [`Self::commit_stamped`] and return its
    /// GPU measurement, in commit order.
    ///
    /// The timestamp is read only after `wait_until_completed`, which the Metal
    /// docs require: before completion `GPUStartTime` / `GPUEndTime` hold
    /// undefined values rather than zero. Waiting here also makes the returned
    /// timeline a real barrier, so it can be compared against host wall clock.
    ///
    /// This does not touch the current command buffer, so the encoder left open
    /// by the last [`Self::commit_stamped`] is still open afterwards: end it
    /// with a commit or [`Self::abandon`] before dropping `self`, exactly as
    /// for every other path, or Metal traps on the un-ended encoder's dealloc.
    pub fn take_gpu_stamps(&mut self) -> Vec<GpuStamp> {
        let retained = std::mem::take(&mut self.stamped);
        let mut stamps = Vec::with_capacity(retained.len());
        let mut previous_end: Option<f64> = None;
        for (label, buffer) in retained {
            buffer.wait_until_completed();
            let start_s = buffer.gpu_start_time();
            let end_s = buffer.gpu_end_time();
            stamps.push(GpuStamp {
                label,
                start_s,
                end_s,
                own_ms: (end_s - start_s) * 1e3,
                gap_ms: previous_end.map_or(0.0, |previous| (start_s - previous) * 1e3),
            });
            previous_end = Some(end_s);
        }
        stamps
    }

    /// Per-label GPU time in milliseconds, measured on the GPU clock, for the
    /// buffers retained by [`Self::commit_stamped`], in commit order.
    ///
    /// Each value is that buffer's own duration (`GPUEndTime - GPUStartTime`),
    /// which is the number to quote as the cost of that phase. Gaps between
    /// consecutive buffers are submission latency, not GPU work, so they are
    /// deliberately not folded in here; use [`Self::take_gpu_stamps`] when the
    /// gap matters.
    pub fn take_gpu_ms(&mut self) -> Vec<(String, f64)> {
        self.take_gpu_stamps()
            .into_iter()
            .map(|stamp| (stamp.label, stamp.own_ms))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use metal::{CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions};

    use super::*;

    /// Busy enough that a dispatch is measurable on the GPU clock, cheap enough
    /// that a few hundred of them stay inside a test's time budget.
    const BURN_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void burn(device float* out [[buffer(0)]],
                 constant uint& iters [[buffer(1)]],
                 uint gid [[thread_position_in_grid]])
{
    float acc = float(gid);
    for (uint i = 0; i < iters; ++i) {
        acc = fma(acc, 1.0000001f, 0.25f);
    }
    out[gid] = acc;
}
"#;

    const THREADS: u64 = 256 * 256;
    const THREADS_PER_GROUP: u64 = 256;
    const CHEAP_ITERS: u32 = 64;

    /// Compiles [`BURN_SOURCE`] at runtime so these tests need nothing from
    /// `build.rs`. Returns `None` on a machine with no Metal device, which
    /// turns each test below into a no-op instead of a failure.
    fn burn_context() -> Option<(Device, CommandQueue, ComputePipelineState, Buffer)> {
        let device = Device::system_default()?;
        let library = device
            .new_library_with_source(BURN_SOURCE, &CompileOptions::new())
            .ok()?;
        let function = library.get_function("burn", None).ok()?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .ok()?;
        let out = device.new_buffer(THREADS * 4, MTLResourceOptions::StorageModePrivate);
        let queue = device.new_command_queue();
        Some((device, queue, pipeline, out))
    }

    fn burn(
        encoder: &mut MetalEncoder<'_>,
        pipeline: &ComputePipelineState,
        out: &Buffer,
        iters: u32,
    ) {
        encoder
            .dispatch_1d(
                pipeline,
                &[Some(out)],
                &iters.to_ne_bytes(),
                THREADS,
                THREADS_PER_GROUP,
            )
            .expect("dispatch");
    }

    #[test]
    fn flush_and_wait_completes_every_rolled_buffer() {
        let Some((_device, queue, pipeline, out)) = burn_context() else {
            return;
        };
        let mut encoder = MetalEncoder::new(&queue).expect("encoder");
        for _ in 0..(MAX_DISPATCHES_PER_BUFFER as usize + 5) {
            burn(&mut encoder, &pipeline, &out, CHEAP_ITERS);
        }
        let rolled: Vec<&CommandBufferRef> = encoder.in_flight.clone();
        assert_eq!(
            rolled.len(),
            1,
            "the 50-dispatch boundary should have rolled exactly one buffer"
        );

        encoder.flush_and_wait().expect("flush");

        assert!(encoder.in_flight.is_empty());
        for buffer in &rolled {
            assert_eq!(
                buffer.status(),
                MTLCommandBufferStatus::Completed,
                "flush_and_wait returned with a rolled buffer still unfinished"
            );
        }
        encoder.abandon();
    }

    #[test]
    fn commit_and_waitless_then_wait_completes_every_buffer() {
        let Some((_device, queue, pipeline, out)) = burn_context() else {
            return;
        };
        let mut encoder = MetalEncoder::new(&queue).expect("encoder");
        for _ in 0..(MAX_DISPATCHES_PER_BUFFER as usize + 5) {
            burn(&mut encoder, &pipeline, &out, CHEAP_ITERS);
        }
        encoder.commit_and_waitless().expect("commit");

        encoder.wait();

        for buffer in &encoder.in_flight {
            assert_eq!(buffer.status(), MTLCommandBufferStatus::Completed);
        }
    }

    #[test]
    fn commit_stamped_keeps_working_after_the_first_stamp() {
        let Some((_device, queue, pipeline, out)) = burn_context() else {
            return;
        };
        let mut encoder = MetalEncoder::new(&queue).expect("encoder");
        burn(&mut encoder, &pipeline, &out, CHEAP_ITERS);
        encoder.commit_stamped("first").expect("stamp");
        assert_eq!(encoder.dispatches, 0);
        // Encoding into the ended encoder of the stamped buffer would raise an
        // Objective-C exception and abort, so getting here at all is the check.
        burn(&mut encoder, &pipeline, &out, CHEAP_ITERS);
        encoder.commit_stamped("second").expect("stamp");
        let stamps = encoder.take_gpu_stamps();
        assert_eq!(
            stamps.iter().map(|s| s.label.as_str()).collect::<Vec<_>>(),
            ["first", "second"]
        );
        // The buffer `commit_stamped` left open is still usable, and ending it
        // with nothing encoded is legal.
        encoder.commit_and_wait().expect("wait");
    }

    #[test]
    fn gpu_timestamps_are_plausible_and_scale_with_work() {
        let Some((_device, queue, pipeline, out)) = burn_context() else {
            return;
        };
        let mut encoder = MetalEncoder::new(&queue).expect("encoder");
        let started = Instant::now();
        burn(&mut encoder, &pipeline, &out, CHEAP_ITERS);
        encoder.commit_stamped("cheap").expect("stamp");
        for _ in 0..8 {
            burn(&mut encoder, &pipeline, &out, CHEAP_ITERS);
        }
        encoder.commit_stamped("heavy").expect("stamp");
        let stamps = encoder.take_gpu_stamps();
        let host_ms = started.elapsed().as_secs_f64() * 1e3;

        assert_eq!(
            stamps.iter().map(|s| s.label.as_str()).collect::<Vec<_>>(),
            ["cheap", "heavy"]
        );
        for stamp in &stamps {
            assert!(
                stamp.start_s > 0.0 && stamp.end_s > 0.0,
                "gpu timestamps read as zero/garbage: {stamp:?}"
            );
            assert!(
                stamp.own_ms > 0.0 && stamp.own_ms < 60_000.0,
                "gpu duration out of range: {stamp:?}"
            );
        }
        // One queue executes in commit order, so the two intervals cannot overlap.
        assert!(
            stamps[0].end_s <= stamps[1].start_s,
            "stamped buffers overlapped on the GPU timeline: {stamps:?}"
        );
        assert!(
            stamps[1].own_ms > stamps[0].own_ms,
            "eight dispatches measured no more GPU time than one: {stamps:?}"
        );
        assert_eq!(stamps[0].gap_ms, 0.0);
        let gpu_total_ms: f64 = stamps.iter().map(|stamp| stamp.own_ms).sum();
        // The GPU was serialised and the host blocked on it, so GPU time is
        // bounded by the host time of the same region, and within an order of
        // magnitude of it: a disagreement here is the bug this feature exists
        // to catch.
        assert!(
            gpu_total_ms <= host_ms + 1.0,
            "GPU time {gpu_total_ms} ms exceeds host time {host_ms} ms"
        );
        assert!(
            host_ms - gpu_total_ms < 5_000.0,
            "host spent {host_ms} ms while the GPU spent {gpu_total_ms} ms"
        );
        assert!(encoder.take_gpu_stamps().is_empty());
        encoder.abandon();
    }

    #[test]
    fn take_gpu_ms_returns_own_duration_in_commit_order() {
        let Some((_device, queue, pipeline, out)) = burn_context() else {
            return;
        };
        let mut encoder = MetalEncoder::new(&queue).expect("encoder");
        burn(&mut encoder, &pipeline, &out, CHEAP_ITERS);
        encoder.commit_stamped("a").expect("stamp");
        burn(&mut encoder, &pipeline, &out, CHEAP_ITERS);
        encoder.commit_stamped("b").expect("stamp");

        let measured = encoder.take_gpu_ms();

        assert_eq!(
            measured
                .iter()
                .map(|(label, _)| label.as_str())
                .collect::<Vec<_>>(),
            ["a", "b"]
        );
        assert!(
            measured.iter().all(|(_, ms)| *ms > 0.0),
            "expected non-zero GPU time per label: {measured:?}"
        );
        assert!(encoder.take_gpu_ms().is_empty(), "take_gpu_ms must drain");
        encoder.abandon();
    }
}
