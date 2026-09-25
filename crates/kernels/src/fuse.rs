//! Host side of the elementwise glue kernels (`fuse.metal`, T4.2).
//!
//! Encoder-oriented on purpose: the decode path encodes *every* pass of a token
//! into one command buffer, so these functions take a
//! [`ComputeCommandEncoderRef`] and device spans rather than host slices. The
//! `run_*` helpers that stage through scratch are for tests.
//!
//! Grid/threadgroup conventions (must match `fuse.metal`):
//! * `kr_rms`: grid `(rows)`, 128 threads, one threadgroup per normalized row;
//! * everything else: grid `(ceil(n / 256))` × 256 threads, flat element index.

use metal::{
    CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

use crate::exl3::{DeviceSpan, KernelError, check_span_fits};

const MSL: &str = include_str!("fuse.metal");

/// Elements per thread for the flat kernels.
const BLOCK: usize = 256;

#[repr(C)]
#[derive(Clone, Copy)]
struct MslRms {
    row_len: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MslRope {
    head_dim: u32,
    pos: u32,
    rows: u32,
    row_stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MslScale {
    n: u32,
    scalar: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MslRmsRope {
    head_dim: u32,
    pos: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MslRmsAddScale {
    row_len: u32,
    eps: f32,
    scalar: f32,
}

/// Pipelines + queue for the glue kernels.
pub struct Fuse {
    device: Device,
    queue: CommandQueue,
    rms: ComputePipelineState,
    rope: ComputePipelineState,
    rms_rope: ComputePipelineState,
    rms_add: ComputePipelineState,
    rms_add_scale: ComputePipelineState,
    copy: ComputePipelineState,
    gelu_mul: ComputePipelineState,
    add: ComputePipelineState,
    add_scale: ComputePipelineState,
}

impl Fuse {
    pub fn new(device: &Device, queue: &CommandQueue) -> Result<Self, KernelError> {
        // Precise math: these passes are verified against the CPU reference, and
        // fast-math would let `sqrt`/`tanh` drift.
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let lib = device
            .new_library_with_source(MSL, &opts)
            .map_err(|e| KernelError::Library(e.to_string()))?;
        let build = |name: &str| -> Result<ComputePipelineState, KernelError> {
            let f = lib
                .get_function(name, None)
                .map_err(|e| KernelError::Function(name.to_string(), e.to_string()))?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| KernelError::Pipeline(name.to_string(), e.to_string()))
        };
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            rms: build("kr_rms")?,
            rope: build("kr_rope")?,
            rms_rope: build("kr_rms_rope")?,
            rms_add: build("kr_rms_add")?,
            rms_add_scale: build("kr_rms_add_scale")?,
            copy: build("kr_copy")?,
            gelu_mul: build("kr_gelu_mul")?,
            add: build("kr_add")?,
            add_scale: build("kr_add_scale")?,
        })
    }

    /// RMSNorm over `rows` rows of `row_len`, writing f16 into `out`.
    ///
    /// `weight` is f32 of length `row_len` (shared by every row) or `None` for
    /// the weightless `k_eq_v` value path.
    pub fn encode_rms(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: DeviceSpan<'_>,
        weight: Option<DeviceSpan<'_>>,
        out: DeviceSpan<'_>,
        rows: usize,
        row_len: usize,
        eps: f32,
    ) -> Result<(), KernelError> {
        check_span_fits(&x, rows * row_len * 2, "rms x", 2)?;
        check_span_fits(&out, rows * row_len * 2, "rms out", 2)?;
        if let Some(w) = weight {
            check_span_fits(&w, row_len * 4, "rms weight", 4)?;
        }
        let p = MslRms {
            row_len: row_len as u32,
            eps,
        };
        enc.set_compute_pipeline_state(&self.rms);
        enc.set_buffer(0, Some(x.buffer), x.byte_offset as u64);
        match weight {
            Some(w) => enc.set_buffer(1, Some(w.buffer), w.byte_offset as u64),
            None => enc.set_buffer(1, None, 0),
        }
        enc.set_buffer(2, Some(out.buffer), out.byte_offset as u64);
        enc.set_bytes(
            3,
            std::mem::size_of::<MslRms>() as u64,
            &p as *const MslRms as *const _,
        );
        enc.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(128, 1, 1));
        Ok(())
    }

    /// Fused RMSNorm + residual add: `out = residual + rms(x, weight)`.
    pub fn encode_rms_add(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: DeviceSpan<'_>,
        weight: Option<DeviceSpan<'_>>,
        residual: DeviceSpan<'_>,
        out: DeviceSpan<'_>,
        rows: usize,
        row_len: usize,
        eps: f32,
    ) -> Result<(), KernelError> {
        check_span_fits(&x, rows * row_len * 2, "rms_add x", 2)?;
        check_span_fits(&residual, rows * row_len * 2, "rms_add residual", 2)?;
        check_span_fits(&out, rows * row_len * 2, "rms_add out", 2)?;
        if let Some(w) = weight {
            check_span_fits(&w, row_len * 4, "rms_add weight", 4)?;
        }
        let p = MslRms {
            row_len: row_len as u32,
            eps,
        };
        enc.set_compute_pipeline_state(&self.rms_add);
        enc.set_buffer(0, Some(x.buffer), x.byte_offset as u64);
        match weight {
            Some(w) => enc.set_buffer(1, Some(w.buffer), w.byte_offset as u64),
            None => enc.set_buffer(1, None, 0),
        }
        enc.set_buffer(2, Some(residual.buffer), residual.byte_offset as u64);
        enc.set_buffer(3, Some(out.buffer), out.byte_offset as u64);
        enc.set_bytes(
            4,
            std::mem::size_of::<MslRms>() as u64,
            &p as *const MslRms as *const _,
        );
        enc.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(128, 1, 1));
        Ok(())
    }

    /// Fused RMSNorm + scaled residual add: `out = (residual + rms(x, weight)) * scalar`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_rms_add_scale(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: DeviceSpan<'_>,
        weight: Option<DeviceSpan<'_>>,
        residual: DeviceSpan<'_>,
        out: DeviceSpan<'_>,
        rows: usize,
        row_len: usize,
        eps: f32,
        scalar: f32,
    ) -> Result<(), KernelError> {
        check_span_fits(&x, rows * row_len * 2, "rms_add_scale x", 2)?;
        check_span_fits(&residual, rows * row_len * 2, "rms_add_scale residual", 2)?;
        check_span_fits(&out, rows * row_len * 2, "rms_add_scale out", 2)?;
        if let Some(w) = weight {
            check_span_fits(&w, row_len * 4, "rms_add_scale weight", 4)?;
        }
        let p = MslRmsAddScale {
            row_len: row_len as u32,
            eps,
            scalar,
        };
        enc.set_compute_pipeline_state(&self.rms_add_scale);
        enc.set_buffer(0, Some(x.buffer), x.byte_offset as u64);
        match weight {
            Some(w) => enc.set_buffer(1, Some(w.buffer), w.byte_offset as u64),
            None => enc.set_buffer(1, None, 0),
        }
        enc.set_buffer(2, Some(residual.buffer), residual.byte_offset as u64);
        enc.set_buffer(3, Some(out.buffer), out.byte_offset as u64);
        enc.set_bytes(
            4,
            std::mem::size_of::<MslRmsAddScale>() as u64,
            &p as *const MslRmsAddScale as *const _,
        );
        enc.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(128, 1, 1));
        Ok(())
    }

    /// Neox RoPE for `rows` rows of `n_heads` heads of `head_dim`. Row `r` is
    /// roped at absolute position `pos + r` (prefill batches pass the position
    /// of row 0); `rows = 1` is the single-row decode case.
    pub fn encode_rope(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: DeviceSpan<'_>,
        cos: DeviceSpan<'_>,
        sin: DeviceSpan<'_>,
        out: DeviceSpan<'_>,
        n_heads: usize,
        head_dim: usize,
        pos: usize,
        rows: usize,
    ) -> Result<(), KernelError> {
        assert!(rows >= 1, "rope needs rows >= 1");
        let row_stride = n_heads * head_dim;
        let n = row_stride * rows;
        check_span_fits(&x, n * 2, "rope x", 2)?;
        check_span_fits(&out, n * 2, "rope out", 2)?;
        let half = head_dim / 2;
        if cos.byte_offset % 4 != 0 || sin.byte_offset % 4 != 0 {
            return Err(KernelError::Unsupported(
                "rope tables must be 4-byte aligned".into(),
            ));
        }
        // The table span must cover head_dim/2 floats at the last row's position.
        for (s, what) in [(cos, "cos"), (sin, "sin")] {
            let need = (pos + rows) * half * 4;
            if (s.buffer.length() as usize) < s.byte_offset + need {
                return Err(KernelError::Unsupported(format!(
                    "rope {what}: needs {need} bytes at offset {}, buffer is {}",
                    s.byte_offset,
                    s.buffer.length()
                )));
            }
        }
        let p = MslRope {
            head_dim: head_dim as u32,
            pos: pos as u32,
            rows: rows as u32,
            row_stride: row_stride as u32,
        };
        enc.set_compute_pipeline_state(&self.rope);
        enc.set_buffer(0, Some(x.buffer), x.byte_offset as u64);
        enc.set_buffer(1, Some(cos.buffer), cos.byte_offset as u64);
        enc.set_buffer(2, Some(sin.buffer), sin.byte_offset as u64);
        enc.set_buffer(3, Some(out.buffer), out.byte_offset as u64);
        enc.set_bytes(
            4,
            std::mem::size_of::<MslRope>() as u64,
            &p as *const MslRope as *const _,
        );
        enc.dispatch_thread_groups(
            MTLSize::new(n.div_ceil(BLOCK) as u64, 1, 1),
            MTLSize::new(BLOCK as u64, 1, 1),
        );
        Ok(())
    }

    /// Fused per-head RMSNorm + RoPE for rows=1 (decode, spec/bandwidth-saturate T6).
    /// Normalizes each head of length `head_dim` and applies Neox RoPE in registers,
    /// writing roped output directly to `out`. Eliminates separate kr_rms and kr_rope passes.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_rms_rope(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: DeviceSpan<'_>,
        weight: Option<DeviceSpan<'_>>,
        cos: DeviceSpan<'_>,
        sin: DeviceSpan<'_>,
        out: DeviceSpan<'_>,
        n_heads: usize,
        head_dim: usize,
        pos: usize,
        eps: f32,
    ) -> Result<(), KernelError> {
        let total = n_heads * head_dim;
        check_span_fits(&x, total * 2, "rms_rope x", 2)?;
        check_span_fits(&out, total * 2, "rms_rope out", 2)?;
        if let Some(w) = weight {
            check_span_fits(&w, head_dim * 4, "rms_rope weight", 4)?;
        }
        let half = head_dim / 2;
        if cos.byte_offset % 4 != 0 || sin.byte_offset % 4 != 0 {
            return Err(KernelError::Unsupported(
                "rope tables must be 4-byte aligned".into(),
            ));
        }
        for (s, what) in [(cos, "cos"), (sin, "sin")] {
            let need = (pos + 1) * half * 4;
            if (s.buffer.length() as usize) < s.byte_offset + need {
                return Err(KernelError::Unsupported(format!(
                    "rope {what}: needs {need} bytes at offset {}, buffer is {}",
                    s.byte_offset,
                    s.buffer.length()
                )));
            }
        }
        let p = MslRmsRope {
            head_dim: head_dim as u32,
            pos: pos as u32,
            eps,
        };
        enc.set_compute_pipeline_state(&self.rms_rope);
        enc.set_buffer(0, Some(x.buffer), x.byte_offset as u64);
        match weight {
            Some(w) => enc.set_buffer(1, Some(w.buffer), w.byte_offset as u64),
            None => enc.set_buffer(1, None, 0),
        }
        enc.set_buffer(2, Some(cos.buffer), cos.byte_offset as u64);
        enc.set_buffer(3, Some(sin.buffer), sin.byte_offset as u64);
        enc.set_buffer(4, Some(out.buffer), out.byte_offset as u64);
        enc.set_bytes(
            5,
            std::mem::size_of::<MslRmsRope>() as u64,
            &p as *const MslRmsRope as *const _,
        );
        enc.dispatch_thread_groups(MTLSize::new(n_heads as u64, 1, 1), MTLSize::new(128, 1, 1));
        Ok(())
    }

    /// Copy `n` f16 elements.
    pub fn encode_copy(
        &self,
        enc: &ComputeCommandEncoderRef,
        src: DeviceSpan<'_>,
        dst: DeviceSpan<'_>,
        n: usize,
    ) -> Result<(), KernelError> {
        check_span_fits(&src, n * 2, "copy src", 2)?;
        check_span_fits(&dst, n * 2, "copy dst", 2)?;
        let p = n as u32;
        enc.set_compute_pipeline_state(&self.copy);
        enc.set_buffer(0, Some(src.buffer), src.byte_offset as u64);
        enc.set_buffer(1, Some(dst.buffer), dst.byte_offset as u64);
        enc.set_bytes(2, 4, &p as *const u32 as *const _);
        dispatch_flat(enc, n);
        Ok(())
    }

    /// `out = gelu_tanh(gate) * up`.
    pub fn encode_gelu_mul(
        &self,
        enc: &ComputeCommandEncoderRef,
        gate: DeviceSpan<'_>,
        up: DeviceSpan<'_>,
        out: DeviceSpan<'_>,
        n: usize,
    ) -> Result<(), KernelError> {
        check_span_fits(&gate, n * 2, "gelu gate", 2)?;
        check_span_fits(&up, n * 2, "gelu up", 2)?;
        check_span_fits(&out, n * 2, "gelu out", 2)?;
        let p = n as u32;
        enc.set_compute_pipeline_state(&self.gelu_mul);
        enc.set_buffer(0, Some(gate.buffer), gate.byte_offset as u64);
        enc.set_buffer(1, Some(up.buffer), up.byte_offset as u64);
        enc.set_buffer(2, Some(out.buffer), out.byte_offset as u64);
        enc.set_bytes(3, 4, &p as *const u32 as *const _);
        dispatch_flat(enc, n);
        Ok(())
    }

    /// `out = a + b` (f16, exact).
    pub fn encode_add(
        &self,
        enc: &ComputeCommandEncoderRef,
        a: DeviceSpan<'_>,
        b: DeviceSpan<'_>,
        out: DeviceSpan<'_>,
        n: usize,
    ) -> Result<(), KernelError> {
        check_span_fits(&a, n * 2, "add a", 2)?;
        check_span_fits(&b, n * 2, "add b", 2)?;
        check_span_fits(&out, n * 2, "add out", 2)?;
        let p = n as u32;
        enc.set_compute_pipeline_state(&self.add);
        enc.set_buffer(0, Some(a.buffer), a.byte_offset as u64);
        enc.set_buffer(1, Some(b.buffer), b.byte_offset as u64);
        enc.set_buffer(2, Some(out.buffer), out.byte_offset as u64);
        enc.set_bytes(3, 4, &p as *const u32 as *const _);
        dispatch_flat(enc, n);
        Ok(())
    }

    /// `out = (a + b) * scalar`.
    pub fn encode_add_scale(
        &self,
        enc: &ComputeCommandEncoderRef,
        a: DeviceSpan<'_>,
        b: DeviceSpan<'_>,
        out: DeviceSpan<'_>,
        n: usize,
        scalar: f32,
    ) -> Result<(), KernelError> {
        check_span_fits(&a, n * 2, "add_scale a", 2)?;
        check_span_fits(&b, n * 2, "add_scale b", 2)?;
        check_span_fits(&out, n * 2, "add_scale out", 2)?;
        let p = MslScale {
            n: n as u32,
            scalar,
        };
        enc.set_compute_pipeline_state(&self.add_scale);
        enc.set_buffer(0, Some(a.buffer), a.byte_offset as u64);
        enc.set_buffer(1, Some(b.buffer), b.byte_offset as u64);
        enc.set_buffer(2, Some(out.buffer), out.byte_offset as u64);
        enc.set_bytes(
            3,
            std::mem::size_of::<MslScale>() as u64,
            &p as *const MslScale as *const _,
        );
        dispatch_flat(enc, n);
        Ok(())
    }

    // -- test/standalone helpers (stage through scratch, submit, wait) --------

    /// Allocate a shared f16 buffer holding `data`.
    pub fn half_buffer(&self, data: &[half::f16]) -> metal::Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            (data.len() * 2).max(2) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Allocate a shared f32 buffer holding `data`.
    pub fn f32_buffer(&self, data: &[f32]) -> metal::Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            (data.len() * 4).max(4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    pub fn buffer(&self, bytes: usize) -> metal::Buffer {
        self.device
            .new_buffer(bytes.max(2) as u64, MTLResourceOptions::StorageModeShared)
    }

    /// Run one encoded closure in its own command buffer and wait.
    pub fn submit(&self, f: impl FnOnce(&ComputeCommandEncoderRef)) -> Result<(), KernelError> {
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        f(enc);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let st = cb.status();
        if st != MTLCommandBufferStatus::Completed {
            return Err(KernelError::CommandFailed(st));
        }
        Ok(())
    }

    /// Read `n` f16 values back from a shared buffer.
    pub fn read_half(buf: &metal::Buffer, n: usize) -> Vec<half::f16> {
        unsafe { std::slice::from_raw_parts(buf.contents() as *const half::f16, n).to_vec() }
    }

    pub fn span(b: &metal::Buffer) -> DeviceSpan<'_> {
        DeviceSpan {
            buffer: b,
            byte_offset: 0,
        }
    }
}

fn dispatch_flat(enc: &ComputeCommandEncoderRef, n: usize) {
    enc.dispatch_thread_groups(
        MTLSize::new(n.div_ceil(BLOCK) as u64, 1, 1),
        MTLSize::new(BLOCK as u64, 1, 1),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    fn ctx() -> Option<(Device, CommandQueue)> {
        let d = Device::system_default()?;
        let q = d.new_command_queue();
        Some((d, q))
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn half(&mut self) -> f16 {
            f16::from_f32(((self.next() % 4001) as f32 / 1000.0) - 2.0)
        }
    }

    fn rms_row(x: &[f16], w: Option<&[f32]>, eps: f64) -> Vec<f16> {
        let mut ss = 0f32;
        for v in x {
            ss += v.to_f32() * v.to_f32();
        }
        let ms = ss / x.len() as f32 + eps as f32;
        let inv = ms.powf(-0.5);
        x.iter()
            .enumerate()
            .map(|(i, v)| {
                let mut o = v.to_f32() * inv;
                if let Some(w) = w {
                    o *= w[i];
                }
                f16::from_f32(o)
            })
            .collect()
    }

    fn max_rel(a: &[f16], b: &[f16]) -> f32 {
        let mut worst = 0f32;
        let mut scale = 0f32;
        for (x, y) in a.iter().zip(b) {
            worst = worst.max((x.to_f32() - y.to_f32()).abs());
            scale = scale.max(x.to_f32().abs());
        }
        if scale > 0.0 { worst / scale } else { worst }
    }

    /// `kr_rms` against the CPU reference, for both norm shapes (hidden row and
    /// per-head) and the weightless variant.
    #[test]
    fn rms_matches_cpu_reference() {
        let Some((device, queue)) = ctx() else {
            eprintln!("skip: no Metal device");
            return;
        };
        let fuse = Fuse::new(&device, &queue).expect("pipelines");
        let mut rng = Rng(0x1234_5678);
        let eps = 1e-6f32;

        // hidden-sized row (5376) with a weight, and a per-head shape (32 × 256),
        // plus the weightless variant used by k_eq_v.
        for (rows, row_len, weighted) in [
            (1usize, 5376usize, true),
            (32, 256, true),
            (2, 128, true),
            (32, 256, false),
            (1, 5376, false),
        ] {
            let x: Vec<f16> = (0..rows * row_len).map(|_| rng.half()).collect();
            let w: Vec<f32> = (0..row_len).map(|i| 1.0 + (i % 7) as f32 * 0.1).collect();
            let xb = fuse.half_buffer(&x);
            let wb = fuse.f32_buffer(&w);
            let ob = fuse.buffer(rows * row_len * 2);
            let wspan = weighted.then(|| Fuse::span(&wb));
            fuse.submit(|enc| {
                fuse.encode_rms(
                    enc,
                    Fuse::span(&xb),
                    wspan,
                    Fuse::span(&ob),
                    rows,
                    row_len,
                    eps,
                )
                .expect("encode rms");
            })
            .expect("submit");
            let got = Fuse::read_half(&ob, rows * row_len);

            let mut want = Vec::with_capacity(x.len());
            for r in 0..rows {
                let row = &x[r * row_len..(r + 1) * row_len];
                // Per-head rows share one weight vector; the reference is the same.
                want.extend(rms_row(row, weighted.then_some(w.as_slice()), eps as f64));
            }
            let err = max_rel(&want, &got);
            eprintln!("rms rows={rows} row_len={row_len} weight={weighted}: rel {err:.2e}");
            assert!(err < 1e-3, "rms rel err {err}");
        }
    }

    /// `kr_rope` against the CPU neox rotate, including into a strided
    /// destination (the KV cache row layout).
    #[test]
    fn rope_matches_cpu_reference() {
        let Some((device, queue)) = ctx() else {
            eprintln!("skip: no Metal device");
            return;
        };
        let fuse = Fuse::new(&device, &queue).expect("pipelines");
        let mut rng = Rng(0x9999);
        let max_pos = 16usize;
        for (n_heads, hd, pos) in [
            (32usize, 256usize, 0usize),
            (32, 256, 7),
            (32, 512, 3),
            (4, 256, 15),
        ] {
            let hh = hd / 2;
            // Tables exactly as `rope_cos_sin` builds them: [max_pos, hd/2].
            let mut cos = vec![0f32; max_pos * hh];
            let mut sin = vec![0f32; max_pos * hh];
            for p in 0..max_pos {
                for k in 0..hh {
                    let f = p as f32 * (1e4f32.powf(-2.0 * k as f32 / hd as f32));
                    cos[p * hh + k] = f.cos();
                    sin[p * hh + k] = f.sin();
                }
            }
            let x: Vec<f16> = (0..n_heads * hd).map(|_| rng.half()).collect();
            let xb = fuse.half_buffer(&x);
            let cb = fuse.f32_buffer(&cos);
            let sb = fuse.f32_buffer(&sin);
            let ob = fuse.buffer(n_heads * hd * 2);
            fuse.submit(|enc| {
                fuse.encode_rope(
                    enc,
                    Fuse::span(&xb),
                    Fuse::span(&cb),
                    Fuse::span(&sb),
                    Fuse::span(&ob),
                    n_heads,
                    hd,
                    pos,
                    1,
                )
                .expect("encode rope");
            })
            .expect("submit");
            let got = Fuse::read_half(&ob, n_heads * hd);

            let mut want = vec![f16::ZERO; x.len()];
            for h in 0..n_heads {
                for j in 0..hh {
                    let (c, s) = (cos[pos * hh + j], sin[pos * hh + j]);
                    let (x1, x2) = (x[h * hd + j].to_f32(), x[h * hd + j + hh].to_f32());
                    want[h * hd + j] = f16::from_f32(x1 * c - x2 * s);
                    want[h * hd + j + hh] = f16::from_f32(x1 * s + x2 * c);
                }
            }
            let err = max_rel(&want, &got);
            eprintln!("rope heads={n_heads} hd={hd} pos={pos}: rel {err:.2e}");
            assert!(err < 1e-3, "rope rel err {err}");
        }
    }

    /// T3.2: multi-row rope (rows ≥ 2, row r at absolute `pos0 + r`) must match
    /// the CPU neox rotate per row — the same reference the single-row test
    /// uses, so batched prefill ropes exactly what the token loop would.
    /// Also: a table that only covers `pos0 + rows - 1` must be **rejected**
    /// when the batch would run past its end.
    #[test]
    fn rope_multi_row_matches_cpu_reference() {
        let Some((device, queue)) = ctx() else {
            eprintln!("skip: no Metal device");
            return;
        };
        let fuse = Fuse::new(&device, &queue).expect("pipelines");
        let mut rng = Rng(0xBEEF);
        let (n_heads, hd, rows, pos0, max_pos) = (32usize, 256usize, 3usize, 4usize, 16usize);
        let hh = hd / 2;
        let mut cos = vec![0f32; max_pos * hh];
        let mut sin = vec![0f32; max_pos * hh];
        for p in 0..max_pos {
            for k in 0..hh {
                let f = p as f32 * (1e4f32.powf(-2.0 * k as f32 / hd as f32));
                cos[p * hh + k] = f.cos();
                sin[p * hh + k] = f.sin();
            }
        }
        let x: Vec<f16> = (0..rows * n_heads * hd).map(|_| rng.half()).collect();
        let xb = fuse.half_buffer(&x);
        let cb = fuse.f32_buffer(&cos);
        let sb = fuse.f32_buffer(&sin);
        let ob = fuse.buffer(x.len() * 2);
        fuse.submit(|enc| {
            fuse.encode_rope(
                enc,
                Fuse::span(&xb),
                Fuse::span(&cb),
                Fuse::span(&sb),
                Fuse::span(&ob),
                n_heads,
                hd,
                pos0,
                rows,
            )
            .expect("encode multi-row rope");
        })
        .expect("submit");
        let got = Fuse::read_half(&ob, x.len());

        let mut want = vec![f16::ZERO; x.len()];
        for r in 0..rows {
            let pos = pos0 + r;
            for h in 0..n_heads {
                for j in 0..hh {
                    let (c, s) = (cos[pos * hh + j], sin[pos * hh + j]);
                    let base = (r * n_heads + h) * hd;
                    let (x1, x2) = (x[base + j].to_f32(), x[base + j + hh].to_f32());
                    want[base + j] = f16::from_f32(x1 * c - x2 * s);
                    want[base + j + hh] = f16::from_f32(x1 * s + x2 * c);
                }
            }
        }
        let err = max_rel(&want, &got);
        eprintln!("rope rows={rows} pos0={pos0} heads={n_heads} hd={hd}: rel {err:.2e}");
        assert!(err < 1e-3, "multi-row rope rel err {err}");

        // pos0 + rows must fit the table: 14 + 3 = 17 > max_pos 16 → reject.
        let err = std::cell::Cell::new(None);
        fuse.submit(|enc| {
            let r = fuse.encode_rope(
                enc,
                Fuse::span(&xb),
                Fuse::span(&cb),
                Fuse::span(&sb),
                Fuse::span(&ob),
                n_heads,
                hd,
                14,
                rows,
            );
            err.set(r.err());
        })
        .expect("submit");
        assert!(
            err.take().is_some(),
            "multi-row rope past table end must be rejected"
        );
    }

    /// `kr_gelu_mul` / `kr_add` / `kr_add_scale` / `kr_copy` vs the CPU helpers.
    #[test]
    fn elementwise_matches_cpu_reference() {
        let Some((device, queue)) = ctx() else {
            eprintln!("skip: no Metal device");
            return;
        };
        let fuse = Fuse::new(&device, &queue).expect("pipelines");
        let mut rng = Rng(0x55AA);
        let n = 20_000usize;
        let g: Vec<f16> = (0..n).map(|_| rng.half()).collect();
        let u: Vec<f16> = (0..n).map(|_| rng.half()).collect();
        let r: Vec<f16> = (0..n).map(|_| rng.half()).collect();
        let gb = fuse.half_buffer(&g);
        let ub = fuse.half_buffer(&u);
        let rb = fuse.half_buffer(&r);
        let ob = fuse.buffer(n * 2);
        let scalar = 0.089355469f32;

        fuse.submit(|enc| {
            fuse.encode_gelu_mul(enc, Fuse::span(&gb), Fuse::span(&ub), Fuse::span(&ob), n)
                .unwrap();
        })
        .unwrap();
        let got = Fuse::read_half(&ob, n);
        let want: Vec<f16> = (0..n)
            .map(|i| {
                let x = g[i].to_f32();
                let k = 0.797_884_56f32;
                f16::from_f32(
                    0.5 * x * (1.0 + (k * (x + 0.044_715 * x * x * x)).tanh()) * u[i].to_f32(),
                )
            })
            .collect();
        let err = max_rel(&want, &got);
        eprintln!("gelu_mul: rel {err:.2e}");
        assert!(err < 1e-3, "gelu rel err {err}");

        let ob2 = fuse.buffer(n * 2);
        fuse.submit(|enc| {
            fuse.encode_add(enc, Fuse::span(&rb), Fuse::span(&ob), Fuse::span(&ob2), n)
                .unwrap();
        })
        .unwrap();
        let got2 = Fuse::read_half(&ob2, n);
        let mut exact = 0usize;
        for i in 0..n {
            let a = r[i];
            let b = got[i];
            let want = f16::from_f64(a.to_f64() + b.to_f64());
            if want == got2[i] {
                exact += 1;
            }
        }
        assert_eq!(exact, n, "f16 add must be bit-exact");

        let ob3 = fuse.buffer(n * 2);
        fuse.submit(|enc| {
            fuse.encode_add_scale(
                enc,
                Fuse::span(&rb),
                Fuse::span(&gb),
                Fuse::span(&ob3),
                n,
                scalar,
            )
            .unwrap();
        })
        .unwrap();
        let got3 = Fuse::read_half(&ob3, n);
        let want3: Vec<f16> = (0..n)
            .map(|i| f16::from_f64((r[i].to_f64() + g[i].to_f64()) * scalar as f64))
            .collect();
        let err = max_rel(&want3, &got3);
        eprintln!("add_scale: rel {err:.2e} (f32 vs f64 product)");
        assert!(err < 2e-3, "add_scale rel err {err}");

        fuse.submit(|enc| {
            fuse.encode_copy(enc, Fuse::span(&gb), Fuse::span(&ob), n)
                .unwrap();
        })
        .unwrap();
        assert_eq!(Fuse::read_half(&ob, n), g, "copy must be exact");
    }

    /// A `nullptr` weight must be accepted (the `k_eq_v` value path).
    #[test]
    fn rms_without_weight_is_the_weightless_norm() {
        let Some((device, queue)) = ctx() else {
            eprintln!("skip: no Metal device");
            return;
        };
        let fuse = Fuse::new(&device, &queue).expect("pipelines");
        let mut rng = Rng(7);
        let (rows, row_len) = (4usize, 256usize);
        let x: Vec<f16> = (0..rows * row_len).map(|_| rng.half()).collect();
        let xb = fuse.half_buffer(&x);
        let ob = fuse.buffer(x.len() * 2);
        fuse.submit(|enc| {
            fuse.encode_rms(
                enc,
                Fuse::span(&xb),
                None,
                Fuse::span(&ob),
                rows,
                row_len,
                1e-6,
            )
            .expect("weightless rms");
        })
        .expect("submit");
        let got = Fuse::read_half(&ob, x.len());
        let mut want = Vec::new();
        for r in 0..rows {
            want.extend(rms_row(&x[r * row_len..(r + 1) * row_len], None, 1e-6));
        }
        let err = max_rel(&want, &got);
        assert!(err < 1e-3, "weightless rms rel err {err}");
    }

    /// A rope table whose span is too short for the requested position (or not
    /// 4-byte aligned) must be rejected, not read past its end.
    #[test]
    fn rejects_bad_rope_tables() {
        let Some((device, queue)) = ctx() else {
            eprintln!("skip: no Metal device");
            return;
        };
        let fuse = Fuse::new(&device, &queue).expect("pipelines");
        let xb = fuse.buffer(1 << 16);
        let err = std::cell::Cell::new(None);
        fuse.submit(|enc| {
            let r = fuse.encode_rope(
                enc,
                Fuse::span(&xb),
                DeviceSpan {
                    buffer: &xb,
                    byte_offset: 2,
                },
                Fuse::span(&xb),
                Fuse::span(&xb),
                1,
                256,
                0,
                1,
            );
            err.set(r.err());
        })
        .expect("submit");
        assert!(err.take().is_some(), "misaligned table must be rejected");

        // Too short for the position requested.
        let err2 = std::cell::Cell::new(None);
        fuse.submit(|enc| {
            let r = fuse.encode_rope(
                enc,
                Fuse::span(&xb),
                Fuse::span(&xb),
                Fuse::span(&xb),
                Fuse::span(&xb),
                1,
                256,
                4096,
                1,
            );
            err2.set(r.err());
        })
        .expect("submit");
        assert!(err2.take().is_some(), "short table must be rejected");
    }

    /// spec/bandwidth-saturate T6: `kr_rms_rope` must match separate `kr_rms` + `kr_rope` bit-for-bit.
    #[test]
    fn rms_rope_matches_separate_passes() {
        let Some((device, queue)) = ctx() else {
            eprintln!("skip: no Metal device");
            return;
        };
        let fuse = Fuse::new(&device, &queue).expect("pipelines");
        let mut rng = Rng(0x8888);
        let max_pos = 16usize;
        let hd = 256usize;
        let hh = hd / 2;
        let mut cos = vec![0f32; max_pos * hh];
        let mut sin = vec![0f32; max_pos * hh];
        for p in 0..max_pos {
            for k in 0..hh {
                let f = p as f32 * (1e4f32.powf(-2.0 * k as f32 / hd as f32));
                cos[p * hh + k] = f.cos();
                sin[p * hh + k] = f.sin();
            }
        }
        let cb = fuse.f32_buffer(&cos);
        let sb = fuse.f32_buffer(&sin);

        for (n_heads, pos, weighted) in [
            (32usize, 0usize, true),
            (32, 5, true),
            (16, 7, true),
            (16, 3, false),
            (1, 10, true),
        ] {
            let x: Vec<f16> = (0..n_heads * hd).map(|_| rng.half()).collect();
            let w: Vec<f32> = (0..hd).map(|_| ((rng.next() % 2001) as f32 / 1000.0) + 0.5).collect();
            let xb = fuse.half_buffer(&x);
            let wb = fuse.f32_buffer(&w);

            let intermediate_b = fuse.buffer(n_heads * hd * 2);
            let want_b = fuse.buffer(n_heads * hd * 2);
            let got_b = fuse.buffer(n_heads * hd * 2);

            let eps = 1e-6f32;

            // Separate: rms -> rope
            fuse.submit(|enc| {
                fuse.encode_rms(
                    enc,
                    Fuse::span(&xb),
                    weighted.then_some(Fuse::span(&wb)),
                    Fuse::span(&intermediate_b),
                    n_heads,
                    hd,
                    eps,
                )
                .unwrap();
                fuse.encode_rope(
                    enc,
                    Fuse::span(&intermediate_b),
                    Fuse::span(&cb),
                    Fuse::span(&sb),
                    Fuse::span(&want_b),
                    n_heads,
                    hd,
                    pos,
                    1,
                )
                .unwrap();
            })
            .expect("separate submit");

            // Fused: rms_rope
            fuse.submit(|enc| {
                fuse.encode_rms_rope(
                    enc,
                    Fuse::span(&xb),
                    weighted.then_some(Fuse::span(&wb)),
                    Fuse::span(&cb),
                    Fuse::span(&sb),
                    Fuse::span(&got_b),
                    n_heads,
                    hd,
                    pos,
                    eps,
                )
                .unwrap();
            })
            .expect("fused submit");

            let want = Fuse::read_half(&want_b, n_heads * hd);
            let got = Fuse::read_half(&got_b, n_heads * hd);

            assert_eq!(want.len(), got.len());
            for i in 0..want.len() {
                assert_eq!(
                    want[i].to_bits(),
                    got[i].to_bits(),
                    "mismatch at index {i} for n_heads={n_heads} pos={pos} weighted={weighted}: want {:#06x} vs got {:#06x}",
                    want[i].to_bits(),
                    got[i].to_bits()
                );
            }
        }
    }

    /// spec/bandwidth-saturate T6.1: `kr_rms_add` and `kr_rms_add_scale` must match separate passes bit-for-bit.
    #[test]
    fn rms_add_and_scale_matches_separate_passes() {
        let Some((device, queue)) = ctx() else {
            eprintln!("skip: no Metal device");
            return;
        };
        let fuse = Fuse::new(&device, &queue).expect("pipelines");
        let mut rng = Rng(0x7777);
        let n = 5376usize;
        let eps = 1e-6f32;
        let scalar = 0.8125f32;

        let x: Vec<f16> = (0..n).map(|_| rng.half()).collect();
        let res: Vec<f16> = (0..n).map(|_| rng.half()).collect();
        let w: Vec<f32> = (0..n).map(|_| ((rng.next() % 2001) as f32 / 1000.0) + 0.5).collect();

        let xb = fuse.half_buffer(&x);
        let resb = fuse.half_buffer(&res);
        let wb = fuse.f32_buffer(&w);

        let norm_b = fuse.buffer(n * 2);
        let want_add_b = fuse.buffer(n * 2);
        let got_add_b = fuse.buffer(n * 2);
        let want_scale_b = fuse.buffer(n * 2);
        let got_scale_b = fuse.buffer(n * 2);

        // Separate: rms -> add, rms -> add_scale
        fuse.submit(|enc| {
            fuse.encode_rms(enc, Fuse::span(&xb), Some(Fuse::span(&wb)), Fuse::span(&norm_b), 1, n, eps).unwrap();
            fuse.encode_add(enc, Fuse::span(&resb), Fuse::span(&norm_b), Fuse::span(&want_add_b), n).unwrap();
            fuse.encode_add_scale(enc, Fuse::span(&resb), Fuse::span(&norm_b), Fuse::span(&want_scale_b), n, scalar).unwrap();
        }).expect("separate submit");

        // Fused: rms_add, rms_add_scale
        fuse.submit(|enc| {
            fuse.encode_rms_add(enc, Fuse::span(&xb), Some(Fuse::span(&wb)), Fuse::span(&resb), Fuse::span(&got_add_b), 1, n, eps).unwrap();
            fuse.encode_rms_add_scale(enc, Fuse::span(&xb), Some(Fuse::span(&wb)), Fuse::span(&resb), Fuse::span(&got_scale_b), 1, n, eps, scalar).unwrap();
        }).expect("fused submit");

        let want_add = Fuse::read_half(&want_add_b, n);
        let got_add = Fuse::read_half(&got_add_b, n);
        let want_scale = Fuse::read_half(&want_scale_b, n);
        let got_scale = Fuse::read_half(&got_scale_b, n);

        for i in 0..n {
            assert_eq!(want_add[i].to_bits(), got_add[i].to_bits(), "rms_add mismatch at {i}: want {:#06x} got {:#06x}", want_add[i].to_bits(), got_add[i].to_bits());
            assert_eq!(want_scale[i].to_bits(), got_scale[i].to_bits(), "rms_add_scale mismatch at {i}: want {:#06x} got {:#06x}", want_scale[i].to_bits(), got_scale[i].to_bits());
        }
    }
}
