//! Host side of the EXL3 Metal kernels: pipeline setup, per-`k` pair tables and
//! grow-on-demand scratch buffers.
//!
//! One `linear_row` call = three dispatches in a single command buffer:
//! `had_in` (x·suh followed by the 128-point Hadamard) → `gemv` (decoded
//! trellis GEMV, f32 partials) → `had_out` (Hadamard × svh → f16).
//!
//! Scope for T2.1: **single row** (decode-shaped) input, f16 `suh`/`svh`,
//! codebook modes `Default`/`Mcg`, even-aligned device spans. Anything else
//! returns [`KernelError::Unsupported`] so the caller can fall back to the CPU
//! reference path instead of producing wrong numbers.

use std::cell::RefCell;
use std::collections::HashMap;

use half::f16;
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

const MSL: &str = include_str!("exl3.metal");

const HAD: usize = 128;
/// Tile edge (samples per row/column of the 16×16 weight tile).
const TILE: usize = 16;
/// Samples in one tile (16×16), as used by the on-disk bitstream length.
const TILE_ELEMS: usize = TILE * TILE;

#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    #[error("metal library compile failed: {0}")]
    Library(String),
    #[error("metal function `{0}` missing: {1}")]
    Function(String, String),
    #[error("pipeline `{0}` failed: {1}")]
    Pipeline(String, String),
    #[error("unsupported for GPU path: {0}")]
    Unsupported(String),
    #[error("command buffer did not complete (status {0:?})")]
    CommandFailed(MTLCommandBufferStatus),
}

/// Tile geometry of one EXL3 linear.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exl3Shape {
    pub in_tiles: usize,
    pub out_tiles: usize,
    pub k: u32,
}

impl Exl3Shape {
    pub fn new(in_features: usize, out_features: usize, k: u32) -> Option<Self> {
        if in_features == 0 || out_features == 0 || !(1..=8).contains(&k) {
            return None;
        }
        if in_features % TILE != 0 || out_features % TILE != 0 {
            return None;
        }
        Some(Self {
            in_tiles: in_features / TILE,
            out_tiles: out_features / TILE,
            k,
        })
    }

    pub fn in_features(&self) -> usize {
        self.in_tiles * TILE
    }

    pub fn out_features(&self) -> usize {
        self.out_tiles * TILE
    }

    /// u16 words per packed tile: `256 * k / 16`.
    pub fn words_per_tile(&self) -> usize {
        TILE * self.k as usize
    }

    /// Words in the circular trellis stream for one tile (`256 * k / 32`).
    pub fn stream_words(&self) -> usize {
        8 * self.k as usize
    }
}

/// A tensor inside a (possibly huge) device buffer.
#[derive(Clone, Copy)]
pub struct DeviceSpan<'a> {
    pub buffer: &'a Buffer,
    pub byte_offset: usize,
}

/// Check that `span` is aligned and large enough for how a kernel will read it.
///
/// `align` is 2 for f16 data and for the trellis (read as u16 pairs): real
/// artifacts place some tensors at offsets that are 2- but not 4-byte aligned,
/// so a blanket 4-byte rule would push the largest single linear back onto the
/// CPU. Shared by the GEMV and attention wrappers.
pub(crate) fn check_span_fits(
    span: &DeviceSpan<'_>,
    bytes_needed: usize,
    what: &str,
    align: usize,
) -> Result<(), KernelError> {
    if span.byte_offset % align != 0 {
        return Err(KernelError::Unsupported(format!(
            "{what}: byte offset {} is not {align}-byte aligned",
            span.byte_offset
        )));
    }
    if (span.buffer.length() as usize) < span.byte_offset + bytes_needed {
        return Err(KernelError::Unsupported(format!(
            "{what}: needs {} bytes at offset {}, buffer is {}",
            bytes_needed,
            span.byte_offset,
            span.buffer.length()
        )));
    }
    Ok(())
}

/// Per-pair funnel-shift descriptor for one `k`, as consumed by the kernel:
/// bits 0..8 = `i0`, 8..16 = `i1`, 16..24 = `s1`.
///
/// Mirrors `unpack_trellis_tile` exactly, including a subtlety that is easy to
/// get wrong: the two words are read at **wrapped** indices (the trellis stream
/// is circular), but `s1` is derived from the **unwrapped** `i1`. Computing `s1`
/// after wrapping yields a negative shift (e.g. k=1, t=0 → -226) and silently
/// garbage output.
pub fn pair_table(k: u32) -> Vec<u32> {
    let stream = (8 * k) as i64;
    let mut out = vec![0u32; 128];
    for (t, slot) in out.iter_mut().enumerate() {
        // NOTE: `TILE_ELEMS` (256 samples per tile), *not* `TILE` (the 16-wide
        // tile edge). Using 16 here silently produced i0 = i1 = 0 for every
        // pair, which the GPU turned into plausible-looking garbage.
        let b0 = t as i64 * 2 * k as i64 + k as i64 - 16 + (TILE_ELEMS as i64) * (k as i64);
        let b2 = b0 + k as i64 + 16;
        let i0_raw = b0 / 32;
        let i1_raw = (b2 - 1) / 32;
        let s1 = (i1_raw + 1) * 32 - b2; // 0..=31, always non-negative
        let i0 = i0_raw.rem_euclid(stream);
        let i1 = i1_raw.rem_euclid(stream);
        debug_assert!((0..32).contains(&s1));
        debug_assert!(i0 < 256 && i1 < 256);
        *slot = (i0 as u32) | ((i1 as u32) << 8) | ((s1 as u32) << 16);
    }
    out
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MslHad {
    row_len: u32,
}

struct Scratch {
    x_stage: Buffer,
    xh: Buffer,
    y: Buffer,
    yh: Buffer,
    x_elems: usize,
    out_elems: usize,
}

/// Owns the pipelines, the per-`k` pair tables and reusable scratch buffers.
pub struct Exl3Gemv {
    device: Device,
    queue: CommandQueue,
    had_in: ComputePipelineState,
    gemv: ComputePipelineState,
    gemm: ComputePipelineState,
    had_out: ComputePipelineState,
    pair_tables: RefCell<HashMap<u32, Buffer>>,
    scratch: RefCell<Option<Scratch>>,
}

impl Exl3Gemv {
    pub fn new(device: &Device, queue: &CommandQueue) -> Result<Self, KernelError> {
        // Precise (IEEE) half conversions and arithmetic: the Hadamard passes are
        // required to be bit-exact against the CPU reference, and Metal's default
        // fast-math allows f32→f16 conversions to round differently.
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
            had_in: build("exl3_had_in")?,
            gemv: build("exl3_gemv")?,
            gemm: build("exl3_gemm")?,
            had_out: build("exl3_had_out")?,
            pair_tables: RefCell::new(HashMap::new()),
            scratch: RefCell::new(None),
        })
    }

    fn pair_buffer(&self, k: u32) -> Buffer {
        if let Some(b) = self.pair_tables.borrow().get(&k) {
            return b.clone();
        }
        let table = pair_table(k);
        let buf = self.device.new_buffer_with_data(
            table.as_ptr() as *const _,
            (table.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        self.pair_tables.borrow_mut().insert(k, buf.clone());
        buf
    }

    /// Shared-mode staging buffer for `n` f16 elements.
    fn new_half_buffer(&self, n: usize) -> Buffer {
        self.device
            .new_buffer((n * 2).max(2) as u64, MTLResourceOptions::StorageModeShared)
    }

    /// Grow the shared scratch to hold `in_elems` f16 inputs and `out_elems`
    /// f32/f16 outputs. Buffers are reused across calls (a token issues ~340
    /// linears), so this allocates at most a handful of times per process.
    fn ensure_scratch(&self, in_elems: usize, out_elems: usize) {
        let mut slot = self.scratch.borrow_mut();
        let needs_grow = match slot.as_ref() {
            Some(s) => s.x_elems < in_elems || s.out_elems < out_elems,
            None => true,
        };
        if needs_grow {
            let keep_in = slot.as_ref().map(|s| s.x_elems).unwrap_or(0).max(in_elems);
            let keep_out = slot
                .as_ref()
                .map(|s| s.out_elems)
                .unwrap_or(0)
                .max(out_elems);
            *slot = Some(Scratch {
                x_stage: self.new_half_buffer(keep_in),
                xh: self.new_half_buffer(keep_in),
                // Metal returns a null object for a zero-length allocation; keep
                // every slot at least one element so unused slots stay valid.
                y: self.device.new_buffer(
                    ((keep_out * 4) as u64).max(4),
                    MTLResourceOptions::StorageModeShared,
                ),
                yh: self.new_half_buffer(keep_out),
                x_elems: keep_in,
                out_elems: keep_out,
            });
        }
    }

    fn with_scratch<R>(
        &self,
        in_elems: usize,
        out_elems: usize,
        f: impl FnOnce(&Scratch) -> R,
    ) -> R {
        self.ensure_scratch(in_elems, out_elems);
        let slot = self.scratch.borrow();
        f(slot.as_ref().unwrap())
    }

    fn encode_had_in(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        suh: DeviceSpan<'_>,
        xh: &Buffer,
        rows: usize,
        row_len: usize,
    ) {
        debug_assert_eq!(row_len % HAD, 0);
        let hp = MslHad {
            row_len: row_len as u32,
        };
        enc.set_compute_pipeline_state(&self.had_in);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(suh.buffer), suh.byte_offset as u64);
        enc.set_buffer(2, Some(xh), 0);
        enc.set_bytes(3, 4, &hp as *const MslHad as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new((row_len / HAD) as u64, rows as u64, 1),
            MTLSize::new(HAD as u64, 1, 1),
        );
    }

    fn encode_gemv(
        &self,
        enc: &ComputeCommandEncoderRef,
        xh: &Buffer,
        trellis: DeviceSpan<'_>,
        pair: &Buffer,
        shape: Exl3Shape,
        mode: u32,
        y: &Buffer,
    ) {
        // The kernel reads the trellis as u32 words (two packed u16s each), so
        // the per-tile stride is half the u16 tile length. `aligned` selects the
        // single-u32 load path (real artifacts: most tensors yes, lm_head no).
        debug_assert_eq!(shape.words_per_tile() % 2, 0);
        let params: [u32; 6] = [
            shape.in_tiles as u32,
            shape.out_tiles as u32,
            shape.k,
            (shape.words_per_tile() / 2) as u32,
            mode,
            u32::from(trellis.byte_offset % 4 == 0),
        ];
        enc.set_compute_pipeline_state(&self.gemv);
        enc.set_buffer(0, Some(xh), 0);
        enc.set_buffer(1, Some(trellis.buffer), trellis.byte_offset as u64);
        enc.set_buffer(2, Some(pair), 0);
        enc.set_buffer(3, Some(y), 0);
        enc.set_bytes(4, 24, params.as_ptr() as *const _);
        // MUST match `WARPS` in exl3.metal (one out-tile per warp). These two
        // constants drifted once and the mismatch looked like a math bug: the
        // kernel indexed shA[] out of bounds and the tests went red.
        const WARPS: usize = 1;
        let groups = shape.out_tiles.div_ceil(WARPS) as u64;
        enc.dispatch_thread_groups(
            MTLSize::new(groups, 1, 1),
            MTLSize::new((32 * WARPS) as u64, 1, 1),
        );
    }

    /// Batched prefill GEMM (T3.2): `y[m] = xh[m] @ decode(trellis)` for
    /// `rows` rows, decoding each weight tile once per out-tile and reusing it
    /// across a chunk of `RGT` (8) rows inside the kernel.
    #[allow(clippy::too_many_arguments)]
    fn encode_gemm(
        &self,
        enc: &ComputeCommandEncoderRef,
        xh: &Buffer,
        trellis: DeviceSpan<'_>,
        pair: &Buffer,
        shape: Exl3Shape,
        mode: u32,
        y: &Buffer,
        rows: usize,
    ) {
        debug_assert!(rows >= 1);
        debug_assert_eq!(shape.words_per_tile() % 2, 0);
        let params: [u32; 7] = [
            shape.in_tiles as u32,
            shape.out_tiles as u32,
            shape.k,
            (shape.words_per_tile() / 2) as u32,
            mode,
            u32::from(trellis.byte_offset % 4 == 0),
            rows as u32,
        ];
        enc.set_compute_pipeline_state(&self.gemm);
        enc.set_buffer(0, Some(xh), 0);
        enc.set_buffer(1, Some(trellis.buffer), trellis.byte_offset as u64);
        enc.set_buffer(2, Some(pair), 0);
        enc.set_buffer(3, Some(y), 0);
        enc.set_bytes(4, 28, params.as_ptr() as *const _);
        // MUST match `RGT` in exl3.metal (rows per threadgroup). 8 is the
        // measured M4 Pro sweet spot; 32 spilled registers (see exl3.metal).
        const RGT: usize = 8;
        enc.dispatch_thread_groups(
            MTLSize::new(shape.out_tiles as u64, rows.div_ceil(RGT) as u64, 1),
            MTLSize::new(32, 1, 1),
        );
    }

    fn encode_had_out(
        &self,
        enc: &ComputeCommandEncoderRef,
        y: &Buffer,
        svh: DeviceSpan<'_>,
        yh: &Buffer,
        rows: usize,
        row_len: usize,
    ) {
        debug_assert_eq!(row_len % HAD, 0);
        let hp = MslHad {
            row_len: row_len as u32,
        };
        enc.set_compute_pipeline_state(&self.had_out);
        enc.set_buffer(0, Some(y), 0);
        enc.set_buffer(1, Some(svh.buffer), svh.byte_offset as u64);
        enc.set_buffer(2, Some(yh), 0);
        enc.set_bytes(3, 4, &hp as *const MslHad as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new((row_len / HAD) as u64, rows as u64, 1),
            MTLSize::new(HAD as u64, 1, 1),
        );
    }

    /// Run `f` with a fresh command buffer and wait for completion.
    fn submit(&self, f: impl FnOnce(&ComputeCommandEncoderRef)) -> Result<(), KernelError> {
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        f(enc);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let status = cb.status();
        if status != MTLCommandBufferStatus::Completed {
            return Err(KernelError::CommandFailed(status));
        }
        Ok(())
    }

    /// Check a device span is big enough and aligned for how the kernel reads it.
    ///
    /// `align` is 2 for both the trellis (read as u16 pairs) and the f16 scales.
    /// The distinction matters: real artifacts place `lm_head.suh` at an offset
    /// that is 2- but not 4-byte aligned, so a blanket 4-byte rule would push
    /// the largest single linear in the model back onto the CPU.
    fn ensure_span_fits(
        span: &DeviceSpan<'_>,
        bytes_needed: usize,
        what: &str,
        align: usize,
    ) -> Result<(), KernelError> {
        check_span_fits(span, bytes_needed, what, align)
    }

    /// Isolated `had_in` pass: `xh = had_r_128(x * suh)` for one row.
    /// Test/bench hook; the production composition is [`Self::linear_row`].
    pub fn had_in_pass(
        &self,
        x: &[f16],
        suh: DeviceSpan<'_>,
        out: &mut Vec<f16>,
    ) -> Result<(), KernelError> {
        let n = x.len();
        if n == 0 || n % HAD != 0 {
            return Err(KernelError::Unsupported(format!(
                "had_in needs a multiple of {HAD}, got {n}"
            )));
        }
        Self::ensure_span_fits(&suh, n * 2, "suh", 2)?;
        out.resize(n, f16::ZERO);
        self.with_scratch(n, 0, |sc| {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    x.as_ptr() as *const u8,
                    sc.x_stage.contents() as *mut u8,
                    n * 2,
                );
            }
            self.submit(|enc| self.encode_had_in(enc, &sc.x_stage, suh, &sc.xh, 1, n))?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    sc.xh.contents() as *const u8,
                    out.as_mut_ptr() as *mut u8,
                    n * 2,
                );
            }
            Ok(())
        })
    }

    /// Isolated GEMV pass: `y = xh @ decode(trellis)` in f32 (no Hadamards).
    /// Test/bench hook.
    pub fn gemv_pass(
        &self,
        xh: &[f16],
        trellis: DeviceSpan<'_>,
        shape: Exl3Shape,
        mode: u32,
        out: &mut Vec<f32>,
    ) -> Result<(), KernelError> {
        if xh.len() != shape.in_features() {
            return Err(KernelError::Unsupported(format!(
                "gemv needs {} activations, got {}",
                shape.in_features(),
                xh.len()
            )));
        }
        if mode > 1 {
            return Err(KernelError::Unsupported(format!("mode {mode} is CPU-only")));
        }
        Self::ensure_span_fits(
            &trellis,
            shape.in_tiles * shape.out_tiles * shape.words_per_tile() * 2,
            "trellis",
            2,
        )?;
        let pair = self.pair_buffer(shape.k);
        out.resize(shape.out_features(), 0.0);
        self.with_scratch(xh.len(), shape.out_features(), |sc| {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    xh.as_ptr() as *const u8,
                    sc.xh.contents() as *mut u8,
                    xh.len() * 2,
                );
            }
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.encode_gemv(enc, &sc.xh, trellis, &pair, shape, mode, &sc.y);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            if cb.status() != MTLCommandBufferStatus::Completed {
                return Err(KernelError::CommandFailed(cb.status()));
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    sc.y.contents() as *const u8,
                    out.as_mut_ptr() as *mut u8,
                    shape.out_features() * 4,
                );
            }
            Ok(())
        })
    }

    /// Isolated `had_out` pass: `out = had_r_128(f16(y)) * svh` for one row.
    /// Test/bench hook.
    pub fn had_out_pass(
        &self,
        y: &[f32],
        svh: DeviceSpan<'_>,
        out: &mut Vec<f16>,
    ) -> Result<(), KernelError> {
        let n = y.len();
        if n == 0 || n % HAD != 0 {
            return Err(KernelError::Unsupported(format!(
                "had_out needs a multiple of {HAD}, got {n}"
            )));
        }
        Self::ensure_span_fits(&svh, n * 2, "svh", 2)?;
        out.resize(n, f16::ZERO);
        self.with_scratch(0, n, |sc| {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    y.as_ptr() as *const u8,
                    sc.y.contents() as *mut u8,
                    n * 4,
                );
            }
            self.submit(|enc| self.encode_had_out(enc, &sc.y, svh, &sc.yh, 1, n))?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    sc.yh.contents() as *const u8,
                    out.as_mut_ptr() as *mut u8,
                    n * 2,
                );
            }
            Ok(())
        })
    }

    /// `y = had_out(had_in(x, suh) @ decode(trellis), svh)` for a single row.
    ///
    /// `out` is resized to `shape.out_features()`. `mode` is the
    /// `CodebookMode` discriminant (0 = Default, 1 = Mcg); Mul1 must not be
    /// routed here.
    pub fn linear_row(
        &self,
        x: &[f16],
        trellis: DeviceSpan<'_>,
        suh: DeviceSpan<'_>,
        svh: DeviceSpan<'_>,
        shape: Exl3Shape,
        mode: u32,
        out: &mut Vec<f16>,
    ) -> Result<(), KernelError> {
        let inn = shape.in_features();
        let outp = shape.out_features();
        if x.len() != inn {
            return Err(KernelError::Unsupported(format!(
                "linear_row expects one row of {inn}, got {}",
                x.len()
            )));
        }
        if mode > 1 {
            return Err(KernelError::Unsupported(format!(
                "codebook mode {mode} is CPU-only"
            )));
        }
        if inn % HAD != 0 || outp % HAD != 0 {
            return Err(KernelError::Unsupported(format!(
                "hadamard needs multiples of {HAD}, got in={inn} out={outp}"
            )));
        }
        // The trellis is addressed as u16 pairs, so 2-byte alignment suffices
        // (real artifacts place lm_head.trellis at a 2-mod-4 offset).
        Self::ensure_span_fits(
            &trellis,
            shape.in_tiles * shape.out_tiles * shape.words_per_tile() * 2,
            "trellis",
            2,
        )?;
        Self::ensure_span_fits(&suh, inn * 2, "suh", 2)?;
        Self::ensure_span_fits(&svh, outp * 2, "svh", 2)?;

        let pair_buf = self.pair_buffer(shape.k);
        out.resize(outp, f16::ZERO);

        self.with_scratch(inn, outp, |sc| {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    x.as_ptr() as *const u8,
                    sc.x_stage.contents() as *mut u8,
                    inn * 2,
                );
            }
            self.submit(|enc| {
                self.encode_had_in(enc, &sc.x_stage, suh, &sc.xh, 1, inn);
                self.encode_gemv(enc, &sc.xh, trellis, &pair_buf, shape, mode, &sc.y);
                self.encode_had_out(enc, &sc.y, svh, &sc.yh, 1, outp);
            })?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    sc.yh.contents() as *const u8,
                    out.as_mut_ptr() as *mut u8,
                    outp * 2,
                );
            }
            Ok(())
        })
    }

    /// `had_in` → `gemv`/`gemm` → `had_out` encoded into **this** command
    /// buffer, for callers that already own device-resident activations (the
    /// T4.2 decode pipeline encodes a whole token this way, so there is one
    /// sync per token instead of one per linear).
    ///
    /// `x` is f16 of `[rows, in_features]`, `out` f16 of `[rows,
    /// out_features]`; both are whole buffers (offset 0). `rows == 1` takes the
    /// tuned single-row GEMV; `rows >= 2` takes the batched GEMM (T3.2), which
    /// decodes each weight tile once per out-tile and reuses it across rows.
    /// The internal `xh`/`y` scratch is shared between calls, which is safe
    /// because dispatches inside one compute encoder run in submission order.
    pub fn encode_linear(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        trellis: DeviceSpan<'_>,
        suh: DeviceSpan<'_>,
        svh: DeviceSpan<'_>,
        out: &Buffer,
        shape: Exl3Shape,
        mode: u32,
        rows: usize,
    ) -> Result<(), KernelError> {
        let inn = shape.in_features();
        let outp = shape.out_features();
        if rows == 0 {
            return Err(KernelError::Unsupported("linear needs rows >= 1".into()));
        }
        if mode > 1 {
            return Err(KernelError::Unsupported(format!(
                "codebook mode {mode} is CPU-only"
            )));
        }
        if inn % HAD != 0 || outp % HAD != 0 {
            return Err(KernelError::Unsupported(format!(
                "hadamard needs multiples of {HAD}, got in={inn} out={outp}"
            )));
        }
        if (x.length() as usize) < rows * inn * 2 {
            return Err(KernelError::Unsupported(format!(
                "x buffer too small: {} < {}",
                x.length(),
                rows * inn * 2
            )));
        }
        if (out.length() as usize) < rows * outp * 2 {
            return Err(KernelError::Unsupported(format!(
                "out buffer too small: {} < {}",
                out.length(),
                rows * outp * 2
            )));
        }
        check_span_fits(
            &trellis,
            shape.in_tiles * shape.out_tiles * shape.words_per_tile() * 2,
            "trellis",
            2,
        )?;
        check_span_fits(&suh, inn * 2, "suh", 2)?;
        check_span_fits(&svh, outp * 2, "svh", 2)?;
        let pair_buf = self.pair_buffer(shape.k);
        self.with_scratch(rows * inn, rows * outp, |sc| {
            self.encode_had_in(enc, x, suh, &sc.xh, rows, inn);
            if rows == 1 {
                self.encode_gemv(enc, &sc.xh, trellis, &pair_buf, shape, mode, &sc.y);
            } else {
                self.encode_gemm(enc, &sc.xh, trellis, &pair_buf, shape, mode, &sc.y, rows);
            }
            self.encode_had_out(enc, &sc.y, svh, out, rows, outp);
        });
        Ok(())
    }
}
