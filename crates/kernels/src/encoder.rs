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
}

impl<'q> MetalEncoder<'q> {
    pub fn new(queue: &'q CommandQueue) -> Result<Self, KernelError> {
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        Ok(Self {
            command_buffer,
            encoder,
            queue,
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
        Ok(())
    }

    /// Encode a 1D dispatch whose bindings may sit at non-zero byte offsets.
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
        Ok(())
    }

    /// End encoding, commit, and block until the batch has completed.
    /// End the current compute encoder, commit, and wait, without consuming self.
    ///
    /// Lets a caller profile phases of one logical forward pass by splitting it
    /// across several command buffers. `KAPLUMBA_PROFILE_PHASES` uses this.
    pub fn flush_and_wait(&mut self) -> Result<(), KernelError> {
        self.encoder.end_encoding();
        self.command_buffer.commit();
        self.command_buffer.wait_until_completed();
        // A compute encoder is single-use, so continue on a fresh buffer. The
        // previous one stays alive until this struct drops, which is what keeps
        // the reference valid.
        self.command_buffer = self.queue.new_command_buffer();
        self.encoder = self.command_buffer.new_compute_command_encoder();
        Ok(())
    }

    pub fn commit_and_wait(self) -> Result<(), KernelError> {
        self.encoder.end_encoding();
        self.command_buffer.commit();
        self.command_buffer.wait_until_completed();
        if self.command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(KernelError::CommandFailed);
        }
        Ok(())
    }
}
