use std::ffi::c_void;

use metal::{
    Buffer, CommandBuffer, CommandQueue, ComputeCommandEncoder, ComputePipelineState,
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
///
/// # Command buffers and encoders are owned handles
///
/// `-[MTLCommandQueue commandBuffer]` and `-[MTLCommandBuffer
/// computeCommandEncoder]` are *not* `new`-prefixed in Objective-C: they hand
/// back an object already added to the autorelease pool
/// (<https://github.com/gfx-rs/metal-rs/issues/128>, and the doc comments
/// metal-rs puts on those two methods). The `&CommandBufferRef` metal-rs returns
/// therefore owns nothing — the only thing keeping the object alive is that this
/// process never drains a pool. Storing the borrowed reference is a promise that
/// nothing ever will, so this type stores the *owned* handles instead
/// (`metal::CommandBuffer` / `metal::ComputeCommandEncoder`). metal-rs builds
/// those with `foreign_type! { fn clone = obj_clone }`, i.e. `Clone`/`ToOwned`
/// send `retain` and `Drop` sends `release` (`metal-0.33.0/src/lib.rs:180-182,
/// 620-627`), so the retain/release pair is the crate's own mechanism rather
/// than hand-rolled FFI. The `From<$owned> for $parent` impls do *not* retain,
/// correctly: they move a reference, they do not copy one.
///
/// Retaining is a precondition for ever draining the pool, not a substitute for
/// it: the pool still holds a pending release, so the bytes are only returned
/// when the pool is drained. What owning the handle buys is that draining can
/// no longer dangle a `&CommandBufferRef` this type is still using.
pub struct MetalEncoder<'q> {
    command_buffer: CommandBuffer,
    encoder: ComputeCommandEncoder,
    /// Kept so `flush_and_wait` can start a fresh command buffer; a compute
    /// encoder cannot be used after `end_encoding`.
    queue: &'q CommandQueue,
    /// Committed-but-unfinished command buffers, oldest first, bounded by
    /// `MAX_IN_FLIGHT`. Waiting on the oldest when the window overflows is what
    /// applies back-pressure: the GPU is never starved, and the host is never
    /// allowed to queue without limit.
    in_flight: Vec<CommandBuffer>,
    /// Encoded dispatches in the current command buffer. A buffer is committed
    /// once it reaches `MAX_DISPATCHES_PER_BUFFER`.
    dispatches: u32,
    /// Every dispatch encoded over this encoder's lifetime, across all command
    /// buffers. `dispatches` is reset by `rearm`, so it cannot answer "how many
    /// dispatches did this forward issue" -- which is the number a fusion is
    /// measured against, since each dispatch costs a launch plus an inter-
    /// dispatch bubble. Exposed by [`Self::dispatches_encoded`].
    dispatches_total: usize,
    /// Whether `encoder` has been ended. Every path that ends it goes through
    /// [`Self::end`], so this cannot drift from reality, and [`Drop`] uses it to
    /// end an encoder that an error path left open — which Metal traps on.
    ended: bool,
    /// Command buffers created over this encoder's lifetime, including the one
    /// currently being encoded into. Rolling bounds this at
    /// `dispatches_total / MAX_DISPATCHES_PER_BUFFER` regardless of how much
    /// work the caller encodes; [`Self::command_buffers_created`] exposes it.
    command_buffers: usize,
    /// Committed, un-waited, labelled buffers kept only to read their GPU
    /// timestamps, in commit order. Populated by [`Self::commit_stamped`] and
    /// drained by [`Self::take_gpu_stamps`]. It is deliberately *not* bounded
    /// by `MAX_IN_FLIGHT`: a profiling caller submits a handful of phase
    /// boundaries per forward and reads them all at the end, and an unbounded
    /// list of already-committed buffers costs no GPU memory (a finished
    /// buffer releases its resources).
    stamped: Vec<(String, CommandBuffer)>,
    max_dispatches: u32,
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

/// Report a non-`Completed` command buffer with the label it carries.
///
/// `KernelError::CommandFailed` is a bare "Metal command buffer failed": it names
/// neither which buffer failed nor when, and every non-completed status collapses
/// into it. A 32k-context prefill that dies after ~273 s produced exactly that one
/// line, which is enough to know *that* something failed and nothing about *what*.
///
/// The status is printed with it, because `-[MTLCommandBuffer status]` *is* bound
/// (`CommandBufferRef::status`) and needs no FFI either. It separates a genuine
/// execution error from the other non-completed cases: on the read-page-fault
/// campaign it reported `status=Error` every time, which is what rules out a
/// stall or an uncommitted buffer and points at the GPU-side fault instead.
///
/// Naming the phase costs nothing and needs no FFI: `-[MTLCommandBuffer label]`
/// *is* bound by metal-rs (`CommandBufferRef::label`), and callers can label the
/// work they encode. `-[MTLCommandBuffer error]` would carry the driver's own
/// `localizedDescription`, but it is unbound in metal-rs 0.33 and reaching it by
/// raw `msg_send` turned a clean `exit 1` into a segfault on the failure path —
/// strictly worse than the bare message, so the labels are the diagnostic.
fn command_failed(buffer: &CommandBuffer) -> KernelError {
    let label = buffer.label();
    let status = buffer.status();
    if label.is_empty() {
        eprintln!("Metal command buffer failed (unlabelled) status={status:?}");
    } else {
        eprintln!("Metal command buffer '{label}' failed status={status:?}");
    }
    KernelError::CommandFailed
}

/// Dispatches per command buffer. MLX uses 50 on this hardware class
/// (`mlx/backend/metal/device.cpp`, arch `'s'` -> `max_ops_per_buffer_ = 50`,
/// alongside a 50 MB cap) and omlx ships 16 for its multi-rank default
/// (`omlx/cluster/deployment.py`). Those are measured values on an M4-class
/// part, not guesses, and they are the direct fix for our symptom: one buffer
/// holding a whole prefill (~8k dispatches) is what made the host spend 58 s
/// wall on work whose GPU time is a few seconds.
///
/// The same value is now also the decode default, via
/// [`decode_max_dispatches_per_buffer`]. Decode needs it for the opposite
/// reason: a step is small enough to encode in ~5 ms, so with the whole step in
/// one uncommitted buffer the GPU receives nothing until the host has finished
/// encoding and then runs ~73 ms with the host blocked in the barrier. Commit
/// every 50 dispatches instead and the two overlap. The one barrier per step is
/// still there — the next token is the argmax of this step's logits — but it no
/// longer has to wait for the encode as well.
pub const MAX_DISPATCHES_PER_BUFFER: u32 = 50;

/// Command buffers allowed in flight. MLX's `MAX_ACTIVE_TASKS = 10`
/// (`mlx/transforms.cpp`); past it the host blocks on a condvar. We measured
/// `new_command_buffer()` blocking ~175 ms once ~64 buffers were outstanding,
/// so staying well under that is deliberate.
pub const MAX_IN_FLIGHT: usize = 8;

/// MEASUREMENT OVERRIDE: `KAPLUMBA_MAX_IN_FLIGHT=N` shrinks the in-flight
/// window, and 1 serialises completely (every rolled command buffer is waited
/// on before the next is encoded).
///
/// This exists because a window of 9 is a *licence to race*: `forward_chunk`
/// writes `scratch.token_ids` and `scratch.query_positions` from the host once
/// per chunk, and with 9 buffers outstanding the GPU may still be hundreds of
/// dispatches into the previous chunk when that happens. Setting this to 1 is
/// how to tell a genuine numerical difference between two chunk sizes apart
/// from the shared-memory race. Not a product setting.
fn max_in_flight() -> usize {
    static CACHE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("KAPLUMBA_MAX_IN_FLIGHT")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|value| (1..=MAX_IN_FLIGHT).contains(value))
            .unwrap_or(MAX_IN_FLIGHT)
    })
}

fn max_dispatches_per_buffer() -> u32 {
    static CACHE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("KAPLUMBA_MAX_DISPATCHES_PER_BUFFER")
            .ok()
            .and_then(|value| value.trim().parse::<u32>().ok())
            .unwrap_or(MAX_DISPATCHES_PER_BUFFER)
    })
}

/// Dispatches per command buffer for a **decode** step. Defaults to
/// [`MAX_DISPATCHES_PER_BUFFER`]; `KAPLUMBA_DECODE_MAX_DISPATCHES_PER_BUFFER=N`
/// overrides it.
///
/// Separate from [`max_dispatches_per_buffer`] on purpose. Prefill already
/// rolls at the default and is correct doing so; the decode path used to call
/// `set_max_dispatches(2500)`, which disabled rolling entirely and left a whole
/// step in one uncommitted buffer. Folding that into the one shared
/// `KAPLUMBA_MAX_DISPATCHES_PER_BUFFER` knob would make the A/B for the decode
/// threshold also move prefill's, which is exactly the confound §3 of the
/// measurement plan warns about. With its own variable, the A and B arms differ
/// in one number and nothing else.
pub fn decode_max_dispatches_per_buffer() -> u32 {
    static CACHE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("KAPLUMBA_DECODE_MAX_DISPATCHES_PER_BUFFER")
            .ok()
            .and_then(|value| value.trim().parse::<u32>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(MAX_DISPATCHES_PER_BUFFER)
    })
}

impl<'q> MetalEncoder<'q> {
    pub fn new(queue: &'q CommandQueue) -> Result<Self, KernelError> {
        let (command_buffer, encoder) = Self::pair(queue);
        Ok(Self {
            command_buffer,
            encoder,
            queue,
            in_flight: Vec::new(),
            dispatches: 0,
            dispatches_total: 0,
            ended: false,
            command_buffers: 1,
            stamped: Vec::new(),
            max_dispatches: max_dispatches_per_buffer(),
        })
    }

    /// Override the dispatch limit before rolling to a new command buffer.
    pub fn set_max_dispatches(&mut self, max: u32) {
        self.max_dispatches = max;
    }

    /// A fresh command buffer and the compute encoder on it, both retained.
    fn pair(queue: &CommandQueue) -> (CommandBuffer, ComputeCommandEncoder) {
        let command_buffer: CommandBuffer = queue.new_command_buffer().to_owned();
        let encoder: ComputeCommandEncoder =
            command_buffer.new_compute_command_encoder().to_owned();
        (command_buffer, encoder)
    }

    /// Move to a fresh command buffer and encoder, which re-arms dispatching
    /// after anything that ended the old pair.
    fn rearm(&mut self) {
        let (command_buffer, encoder) = Self::pair(self.queue);
        self.command_buffers += 1;
        self.command_buffer = command_buffer;
        self.encoder = encoder;
        self.dispatches = 0;
        self.ended = false;
    }

    /// `end_encoding` on the current encoder, exactly once.
    ///
    /// Metal raises an Objective-C exception — an abort, not an error return —
    /// both for a second `end_encoding` and for dropping an encoder that was
    /// never ended, so the single choke point here is what keeps either from
    /// being reachable.
    fn end(&mut self) {
        if !self.ended {
            self.encoder.end_encoding();
            self.ended = true;
        }
    }

    /// Encoding into an ended encoder is the same abort, and a silent one in
    /// release builds, so the misuse is caught wherever debug assertions are on.
    fn assert_live(&self) {
        debug_assert!(
            !self.ended,
            "dispatching into a MetalEncoder that has already been ended; \
             commit_terminal is terminal and flush_and_wait/commit_stamped are not, \
             so this is a use-after-terminal",
        );
    }

    /// Command buffers this encoder has created over its lifetime, including
    /// the one it is encoding into right now.
    ///
    /// The rolling scheme in [`Self::tick`] is unconditional: it applies to every
    /// `dispatch_*`, so a decode step that encodes thousands of dispatches
    /// commits them in `MAX_DISPATCHES_PER_BUFFER` slices exactly like a
    /// prefill. This counter is the evidence, and a decode step is expected to
    /// report tens, not one.
    pub fn command_buffers_created(&self) -> usize {
        self.command_buffers
    }

    /// Every compute dispatch encoded over this encoder's lifetime.
    ///
    /// This is the denominator a kernel fusion is measured on. A decode step
    /// issues ~1400 of them, and the non-GEMV ones are small enough that the
    /// inter-dispatch bubble, not the kernel, is a material share of the step:
    /// measured own time for the non-GEMV kernels is 7.45 ms against an
    /// uninstrumented step of ~81.5 ms, so ~5.4 ms of the non-GEMV tax is
    /// launch latency that only fewer dispatches can remove.
    pub fn dispatches_encoded(&self) -> usize {
        self.dispatches_total
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
        self.assert_live();
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
        self.assert_live();
        if groups_x == 0 || groups_y == 0 || threads_per_group_x == 0 || threads_per_group_y == 0 {
            return Err(KernelError::InvalidInput(
                "dispatch group and threadgroup sizes must be non-zero".to_string(),
            ));
        }
        self.encoder.set_compute_pipeline_state(pipeline);
        for (index, binding) in bindings.iter().enumerate() {
            self.encoder
                .set_buffer(index as u64, Some(&**binding.buffer), binding.offset);
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
        self.assert_live();
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
        self.assert_live();
        if threads == 0 || threads_per_group == 0 {
            return Err(KernelError::InvalidInput(
                "kernel thread and threadgroup counts must be non-zero".to_string(),
            ));
        }
        self.encoder.set_compute_pipeline_state(pipeline);
        for (index, binding) in bindings.iter().enumerate() {
            self.encoder
                .set_buffer(index as u64, Some(&**binding.buffer), binding.offset);
        }
        self.encoder.set_bytes(
            bindings.len() as u64,
            params.len() as u64,
            params.as_ptr() as *const c_void,
        );
        self.encoder.dispatch_threads(
            MTLSize::new(threads, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
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
        self.assert_live();
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
        self.encoder.dispatch_threads(threads, threads_per_group);
        self.tick()?;
        Ok(())
    }

    /// Count one dispatch and roll to a fresh command buffer once the buffer is
    /// full, bounding how much work any single buffer carries.
    fn tick(&mut self) -> Result<(), KernelError> {
        self.dispatches += 1;
        self.dispatches_total += 1;
        if self.dispatches >= self.max_dispatches {
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
        self.end();
        self.command_buffer.commit();
        self.in_flight.push(self.command_buffer.clone());
        if self.in_flight.len() > max_in_flight() {
            let oldest = self.in_flight.remove(0);
            oldest.wait_until_completed();
        }
        self.rearm();
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
        if self.dispatches == 0 && self.in_flight.is_empty() {
            return Ok(());
        }
        self.end();
        self.command_buffer.commit();
        for buffer in std::mem::take(&mut self.in_flight) {
            buffer.wait_until_completed();
        }
        self.command_buffer.wait_until_completed();
        // A compute encoder is single-use, so continue on a fresh buffer. The
        // pair is owned, so dropping the old one is what returns its memory.
        self.rearm();
        Ok(())
    }

    /// End the encoder and drop the work without committing.
    ///
    /// The automatic path is [`Drop`], which ends any encoder still open, so an
    /// error unwinding out of a caller no longer needs this. It stays because
    /// `model.rs` calls it to end the encoder *before* returning an error, and
    /// because ending at the failure point rather than at the end of the scope
    /// releases the bound resources sooner.
    pub fn abandon(&mut self) {
        self.end();
    }

    /// Commit without waiting, so several command buffers can be in flight.
    ///
    /// # This is a terminal operation
    ///
    /// Unlike [`Self::flush_and_wait`] and [`Self::commit_stamped`] it does
    /// *not* start a fresh command buffer, so the value is dead for dispatching
    /// afterwards: the `encoder` field still points at the encoder this call just
    /// ended, and the next `dispatch_*` would encode into it and abort on an
    /// Objective-C exception. The only things still legal on the returned value
    /// are [`Self::wait`] (which takes `&self` and so cannot re-arm) and
    /// dropping it, which thanks to [`Drop`] is now safe rather than a trap.
    ///
    /// Reachability of the misuse: the only caller is the device-pick decode
    /// loop in `generate.rs`, which calls `wait()` on the value and then lets it
    /// go out of scope. `dispatches` is deliberately left stale for the same
    /// reason — a fresh buffer would invite another commit. `debug_assert`s on
    /// every `dispatch_*` and here turn a future caller into a debug-build
    /// failure instead of a release-build abort.
    pub fn commit_terminal(&mut self) -> Result<(), KernelError> {
        debug_assert!(
            !self.ended,
            "commit_terminal is terminal: this encoder has already been ended",
        );
        self.end();
        self.command_buffer.commit();
        self.in_flight.push(self.command_buffer.clone());
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
        self.end();
        self.command_buffer.commit();
        self.command_buffer.wait_until_completed();
        // Buffers rolled out mid-forward are committed but unfinished. On one
        // in-order queue the wait above already covers them, but drain them
        // anyway so the barrier does not depend on that.
        for buffer in std::mem::take(&mut self.in_flight) {
            buffer.wait_until_completed();
        }
        if self.command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(command_failed(&self.command_buffer));
        }
        Ok(())
    }

    /// Commit the work encoded so far under `label` and continue on a fresh
    /// command buffer, without waiting.
    ///
    /// This is [`Self::commit_terminal`] without the terminality -- the encoder
    /// is re-armed -- plus two things: the buffer is labelled, and it is parked
    /// in `stamped` so [`Self::take_gpu_ms`] can
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
        self.end();
        self.command_buffer.set_label(label);
        self.command_buffer.commit();
        self.stamped
            .push((label.to_string(), self.command_buffer.clone()));
        self.rearm();
        Ok(())
    }

    /// Commit the work encoded so far under `label`, wait for it and for every
    /// rolled buffer, and park it in `stamped` for a GPU measurement.
    ///
    /// This exists because [`Self::commit_and_wait`] takes `self`, and a
    /// measurement needs the buffer *after* it has completed:
    /// `GPUStartTime` / `GPUEndTime` are undefined until then. So the only way
    /// to time a region of work with these two primitives was to commit it
    /// terminal and never look at it -- which is precisely how a decode step
    /// ended up with no GPU-clock measurement at all.
    ///
    /// Unlike [`Self::wait`] it does not touch the fresh buffer that
    /// `commit_stamped` re-armed: that buffer has never been committed, and
    /// waiting on a buffer Metal never enqueued is not a documented operation.
    /// Every buffer it does wait on is status-checked, so a failed dispatch is
    /// still an error rather than a plausible-looking number.
    pub fn commit_stamped_and_wait(&mut self, label: &str) -> Result<(), KernelError> {
        self.commit_stamped(label)?;
        for (_, buffer) in &self.stamped {
            buffer.wait_until_completed();
            if buffer.status() != MTLCommandBufferStatus::Completed {
                return Err(command_failed(buffer));
            }
        }
        for buffer in std::mem::take(&mut self.in_flight) {
            buffer.wait_until_completed();
            if buffer.status() != MTLCommandBufferStatus::Completed {
                return Err(command_failed(&buffer));
            }
        }
        Ok(())
    }

    /// Commit the work encoded since the last commit, wait for it, and leave
    /// the encoder terminal.
    ///
    /// `commit_stamped` re-arms a fresh command buffer. Dropping the encoder
    /// then *releases that buffer without ever committing it*, and a Metal
    /// queue that has accumulated enough uncommitted command buffers blocks in
    /// `new_command_buffer` rather than allocating another -- which is a hang
    /// that looks like nothing at all from the outside. Measured here: a loop
    /// that creates one encoder per "step" and drops it after
    /// `commit_stamped_and_wait` blocks indefinitely in `new_command_buffer`
    /// after a few hundred steps.
    ///
    /// So a caller that makes a command buffer per step has to end the step
    /// with this, not with a drop. The buffer committed here is normally empty,
    /// which costs a few microseconds.
    pub fn finish(&mut self) -> Result<(), KernelError> {
        self.end();
        self.command_buffer.commit();
        self.command_buffer.wait_until_completed();
        for buffer in std::mem::take(&mut self.in_flight) {
            buffer.wait_until_completed();
        }
        if self.command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(command_failed(&self.command_buffer));
        }
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
    /// with a commit or [`Self::abandon`] before dropping `self`, or let
    /// [`Drop`] do it.
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

impl Drop for MetalEncoder<'_> {
    /// End the encoder if it is still open, so dropping an encoder on an error
    /// path cannot raise.
    ///
    /// Metal traps (`SIGTRAP` in a debug build, an Objective-C exception that
    /// aborts otherwise) when a compute encoder is deallocated without
    /// `endEncoding`. Before this existed, that meant every `?` in a caller that
    /// encoded into a `MetalEncoder` — `generate.rs`'s prefill, for one — was a
    /// process abort on any error, not a returned `Err`, which is why callers had
    /// to hand-place `abandon()` on each path and one missed path was enough.
    /// `ended` is the only thing this consults, and every method that ends the
    /// encoder goes through [`MetalEncoder::end`], so a second `end_encoding`
    /// here is impossible — which matters, because Metal traps on that too.
    fn drop(&mut self) {
        self.end();
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
        let rolled: Vec<CommandBuffer> = encoder.in_flight.clone();
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
    fn commit_terminal_then_wait_completes_every_buffer() {
        let Some((_device, queue, pipeline, out)) = burn_context() else {
            return;
        };
        let mut encoder = MetalEncoder::new(&queue).expect("encoder");
        for _ in 0..(MAX_DISPATCHES_PER_BUFFER as usize + 5) {
            burn(&mut encoder, &pipeline, &out, CHEAP_ITERS);
        }
        encoder.commit_terminal().expect("commit");

        encoder.wait();

        for buffer in &encoder.in_flight {
            assert_eq!(buffer.status(), MTLCommandBufferStatus::Completed);
        }
    }

    /// The rolling scheme is not a prefill-only optimisation: every `dispatch_*`
    /// ticks, so a decode-sized step lands in tens of small command buffers
    /// rather than one giant one. This is the check that `tick` keeps applying
    /// after the per-call scheme was added.
    #[test]
    fn a_decode_sized_step_rolls_rather_than_using_one_buffer() {
        let Some((_device, queue, pipeline, out)) = burn_context() else {
            return;
        };
        // A decode-sized step of a 31B-class model at 1 query. The constant is
        // only a scale choice for this rolling test; the real figure is
        // 1617 dispatches/token, measured via `MetalEncoder::dispatches_total`
        // under `KAPLUMBA_COUNT_BUFFERS=1`.
        //
        // (This comment has been corrected twice and was circular both times:
        // it first claimed "measured: 42 command buffers per step", which is
        // exactly 2100/50 -- this constant divided by the default budget. It
        // was then corrected to "`forward_buffer_into_step` calls
        // `set_max_dispatches(2500)`, so it never rolls", which was true of
        // the callers, not of `tick`: the mechanism works, the threshold was
        // simply set above the step size. The callers now ask for
        // `decode_max_dispatches_per_buffer()`, 50 by default, and the step
        // rolls again.)
        const DECODE_DISPATCHES: usize = 2100;
        let mut encoder = MetalEncoder::new(&queue).expect("encoder");
        for _ in 0..DECODE_DISPATCHES {
            burn(&mut encoder, &pipeline, &out, CHEAP_ITERS);
        }

        let expected = 1 + DECODE_DISPATCHES / MAX_DISPATCHES_PER_BUFFER as usize;
        assert_eq!(
            encoder.command_buffers_created(),
            expected,
            "a decode-sized step should roll every {MAX_DISPATCHES_PER_BUFFER} dispatches, \
             not encode into a single command buffer",
        );
        assert_eq!(
            encoder.dispatches,
            (DECODE_DISPATCHES % MAX_DISPATCHES_PER_BUFFER as usize) as u32,
            "the trailing partial slice is the only work left in the current buffer",
        );
        encoder.commit_and_wait().expect("wait");
    }

    /// Dropping without ending used to be a Metal trap, which is why callers had
    /// to place `abandon()` on every error path. This is the regression check:
    /// if `Drop` stops ending the encoder, this test aborts the process instead
    /// of failing.
    #[test]
    fn dropping_an_unfinished_encoder_does_not_trap() {
        let Some((_device, queue, pipeline, out)) = burn_context() else {
            return;
        };
        let mut encoder = MetalEncoder::new(&queue).expect("encoder");
        burn(&mut encoder, &pipeline, &out, CHEAP_ITERS);
        drop(encoder);

        // Same again with nothing encoded at all: an encoder created and dropped
        // is the cheapest way to hit a missing `endEncoding`.
        let encoder = MetalEncoder::new(&queue).expect("encoder");
        drop(encoder);
    }

    /// `commit_terminal` leaves nothing dispatchable, and ending must not
    /// double-fire: the flag it leaves behind is what keeps `Drop` from calling
    /// `endEncoding` a second time, and a double `endEncoding` traps.
    #[test]
    fn terminal_commit_leaves_the_encoder_ended() {
        let Some((_device, queue, pipeline, out)) = burn_context() else {
            return;
        };
        let mut encoder = MetalEncoder::new(&queue).expect("encoder");
        burn(&mut encoder, &pipeline, &out, CHEAP_ITERS);
        encoder.commit_terminal().expect("commit");
        assert!(
            encoder.ended,
            "commit_terminal must leave the encoder ended"
        );
        // Dropping here is the double-end check: without the flag it would trap.
        drop(encoder);
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
