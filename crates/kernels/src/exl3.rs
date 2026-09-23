//! Host side of the EXL3 Metal kernels: pipeline setup and per-`k` pair tables.
//!
//! One `linear_row` call = three dispatches in a single command buffer:
//! `had_in` (x·suh followed by the 128-point Hadamard) → `gemv` (decoded
//! trellis GEMV, f32 partials) → `had_out` (Hadamard × svh → f16).
//!
//! **No retained scratch on `Exl3Gemv`** (user directive 2026-09-23): production
//! decode/prefill pass a caller-owned [`ChunkScratch`] (`encode_linear_with_scratch`);
//! host test/bench hooks allocate a transient scratch per call only.
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

/// Inverse of the tensor-core 16×16 tile permutation: for each row-major
/// tile position, the kernel-order codeword index stored there. Consumed by
/// `exl3_gemm_sg`'s decode precompute (`inv_perm[(2*rp+j)*16 + dcol]`).
///
/// Duplicate of the engine's `exl3::perm::tensor_core_perm_inverse` (the
/// kernels crate must not depend on the engine); same construction as the
/// reference `_inv_perm_u32` / `tensor_core_perm_inverse`
/// (reference/PonyExl3/ponyexl3/ref/perm.py). Unit-tested against the
/// engine's recorded head values below.
fn tensor_core_perm_inverse() -> [u32; 256] {
    let mut perm = [0usize; 256];
    for t in 0..32 {
        let r0 = (t % 4) * 2;
        let c0 = t / 4;
        let base = t * 8;
        perm[base] = r0 * 16 + c0;
        perm[base + 1] = (r0 + 1) * 16 + c0;
        perm[base + 2] = (r0 + 8) * 16 + c0;
        perm[base + 3] = (r0 + 9) * 16 + c0;
        perm[base + 4] = r0 * 16 + c0 + 8;
        perm[base + 5] = (r0 + 1) * 16 + c0 + 8;
        perm[base + 6] = (r0 + 8) * 16 + c0 + 8;
        perm[base + 7] = (r0 + 9) * 16 + c0 + 8;
    }
    let mut inv = [0u32; 256];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i as u32;
    }
    inv
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MslHad {
    row_len: u32,
}

/// Transient host-path scratch (test/bench hooks only — never retained).
struct Scratch {
    x_stage: Buffer,
    xh: Buffer,
    y: Buffer,
    yh: Buffer,
}

/// Per-chunk scratch arena for prefill (rows >= 2). Allocated once per chunk
/// by [`Exl3Gemv::new_chunk_scratch`], passed to every linear in that chunk,
/// dropped after the chunk's command buffer completes. Keeps the shared decode
/// `Scratch`/`partial` bounded at decode-max size instead of growing to
/// prefill rows (spec/prefill-scratch-fix).
pub struct ChunkScratch {
    pub xh: Buffer,
    pub y: Buffer,
    /// T3.10 sg K-split partials `[nsplit, rows_pad, N]` f32. At rows=256
    /// k/v needs 8.4 MB here — shared `partial` must stay gemv/rows=1-sized.
    pub partial: Buffer,
}

/// Owns the pipelines, the per-`k` pair tables and reusable scratch buffers.
///
/// **No production scratch lives here.** Decode and prefill both pass a
/// caller-owned [`ChunkScratch`] (the decoder's fixed arena); the shared
/// grow-on-demand slots were removed so RSS cannot step on first decode
/// or first prefill (spec/prefill-scratch-fix, user directive: no scratch).
/// Host test/bench hooks (`linear_row`, `had_*_pass`, `encode_linear`)
/// still allocate a **transient** scratch per call — never retained.
pub struct Exl3Gemv {
    device: Device,
    queue: CommandQueue,
    had_in: ComputePipelineState,
    gemv: ComputePipelineState,
    /// T2.5: sums the K-split partials of `gemv` back into `y`.
    gemv_reduce: ComputePipelineState,
    gemm: ComputePipelineState,
    /// Simdgroup tensor-core GEMM (v19c port; `rows ≥ 2`, see `sg_eligible`).
    gemm_sg: ComputePipelineState,
    /// T3.10: sums the K-split partials of `gemm_sg` back into `y`.
    gemm_sg_reduce: ComputePipelineState,
    had_out: ComputePipelineState,
    pair_tables: RefCell<HashMap<u32, Buffer>>,
    /// Constant 256×u32 inverse tile permutation, shared by every linear
    /// (the decode geometry is per tile-position, not per layer).
    inv_perm: Buffer,
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
        let inv = tensor_core_perm_inverse();
        let inv_perm = device.new_buffer_with_data(
            inv.as_ptr() as *const _,
            (inv.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            had_in: build("exl3_had_in")?,
            gemv: build("exl3_gemv")?,
            gemv_reduce: build("exl3_gemv_reduce")?,
            gemm: build("exl3_gemm")?,
            gemm_sg: build("exl3_gemm_sg")?,
            gemm_sg_reduce: build("exl3_gemm_sg_reduce")?,
            had_out: build("exl3_had_out")?,
            pair_tables: RefCell::new(HashMap::new()),
            inv_perm,
        })
    }

    /// T2.5 K-split policy: gemv shapes with few out-tiles only fill a small
    /// slice of the GPU's concurrent simdgroups (measured Gw/s tracks wave
    /// depth — lm_head 51 waves = 241, gate 4 = 172, down/o 1 = 149/119), so
    /// split the in-tile range across `grid.y` until the dispatch reaches
    /// ~4096 threadgroups (T2.9 pivot: 1024 left k/v at 94/106 Gw/s with
    /// nsplit=4), capped at 16 splits (partial ≤ ~344 KB).
    fn gemv_splits(out_tiles: usize) -> usize {
        const TARGET_TGS: usize = 4096;
        const MAX_SPLIT: usize = 16;
        TARGET_TGS.div_ceil(out_tiles.max(1)).clamp(1, MAX_SPLIT)
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

    /// Shared-mode f32 buffer of at least `elems` elements (≥1 so slots stay valid).
    fn new_f32_buffer(&self, elems: usize) -> Buffer {
        self.device.new_buffer(
            ((elems * 4) as u64).max(4),
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// **Transient** scratch for one host test/bench call — never retained on
    /// `self`. Production (`encode_linear_with_scratch`) uses the decoder's
    /// fixed arena instead (user directive: no shared scratch buffers).
    fn with_scratch<R>(
        &self,
        in_elems: usize,
        out_elems: usize,
        rows: usize,
        f: impl FnOnce(&Scratch) -> R,
    ) -> R {
        let _ = rows; // retained for call-site clarity; always transient now
        let tmp = Scratch {
            x_stage: self.new_half_buffer(in_elems.max(1)),
            xh: self.new_half_buffer(in_elems.max(1)),
            y: self.new_f32_buffer(out_elems),
            yh: self.new_half_buffer(out_elems.max(1)),
        };
        f(&tmp)
    }

    /// Per-chunk / decoder-owned temps for one linear family at `rows`.
    /// Allocated once by the decoder (fixed, context-independent — not per-chunk).
    /// `max_partial_bytes` = max over linears of [`Self::sg_partial_bytes`] /
    /// [`Self::gemv_partial_bytes`] (≥4 even when no shape splits).
    pub fn new_chunk_scratch(
        &self,
        max_in_elems: usize,
        max_out_elems: usize,
        max_partial_bytes: usize,
    ) -> ChunkScratch {
        ChunkScratch {
            xh: self.new_half_buffer(max_in_elems.max(1)),
            y: self.new_f32_buffer(max_out_elems.max(1)),
            partial: self.device.new_buffer(
                (max_partial_bytes as u64).max(4),
                MTLResourceOptions::StorageModeShared,
            ),
        }
    }

    /// `(xh_elems, y_elems, partial_bytes)` a `ChunkScratch` needs for one
    /// linear at `rows`. Single-sources `xh_rows`/`sg_eligible`/`sg_splits`
    /// (the engine must never re-derive the eligibility heuristic — a
    /// mismatch under-allocates `xh` and corrupts the pad-row reads).
    /// `trellis_offset` = `lin.trellis().byte_offset` (eligibility only needs
    /// the 4-byte alignment; taking the raw offset keeps this policy-testable
    /// without a Metal device).
    pub fn chunk_scratch_needs(
        shape: Exl3Shape,
        trellis_offset: usize,
        rows: usize,
    ) -> (usize, usize, usize) {
        let eligible = rows >= 2 && Self::sg_eligible_at(shape, trellis_offset);
        let xh_rows = if eligible {
            rows.div_ceil(64) * 64
        } else {
            rows
        };
        let xh = xh_rows * shape.in_features();
        let y = rows * shape.out_features();
        (xh, y, Self::sg_partial_bytes(shape, trellis_offset, rows))
    }

    /// Bytes of sg K-split partial a rows>1 call may write
    /// (`nsplit * rows_pad * N * 4`), or 0 when the shape binds `y` directly.
    pub fn sg_partial_bytes(shape: Exl3Shape, trellis_offset: usize, rows: usize) -> usize {
        if rows < 2 || !Self::sg_eligible_at(shape, trellis_offset) {
            return 0;
        }
        let nsplit = Self::sg_splits(shape, rows);
        if nsplit <= 1 {
            return 0;
        }
        let rows_pad = rows.div_ceil(64) * 64;
        nsplit * rows_pad * shape.out_features() * 4
    }

    /// Bytes of rows=1 gemv K-split partial (`nsplit * out_tiles * TILE * 4`),
    /// or 0 when nsplit == 1. Decoder arena sizing must cover both this and
    /// [`Self::sg_partial_bytes`] at `batch_rows`.
    pub fn gemv_partial_bytes(shape: Exl3Shape) -> usize {
        let nsplit = Self::gemv_splits(shape.out_tiles);
        if nsplit <= 1 {
            return 0;
        }
        nsplit * shape.out_tiles * TILE * 4
    }

    /// Encode a linear with caller-owned temps (decoder arena — both decode
    /// `rows==1` and prefill). Does NOT touch any shared scratch (none exists).
    pub fn encode_linear_with_scratch(
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
        scratch: &ChunkScratch,
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
        let x_rows = Self::xh_rows(rows, shape, trellis);
        // Arena is sized to the chunk max across linears — exact equality only
        // for the largest shape; must never be smaller than this call's need.
        debug_assert!(
            (scratch.xh.length() as usize) / 2 >= x_rows * inn,
            "chunk xh under-allocated: {} < {}",
            scratch.xh.length() / 2,
            x_rows * inn
        );
        debug_assert!(
            (scratch.y.length() as usize) >= rows * outp * 4,
            "chunk y under-allocated"
        );
        self.encode_had_in(enc, x, suh, &scratch.xh, rows, inn);
        self.encode_gem(
            enc,
            &scratch.xh,
            trellis,
            &pair_buf,
            shape,
            mode,
            &scratch.y,
            Some(&scratch.partial),
            rows,
        );
        self.encode_had_out(enc, &scratch.y, svh, out, rows, outp);
        Ok(())
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
        ext_partial: Option<&Buffer>,
    ) {
        // The kernel reads the trellis as u32 words (two packed u16s each), so
        // the per-tile stride is half the u16 tile length. `aligned` selects the
        // single-u32 load path (real artifacts: most tensors yes, lm_head no).
        debug_assert_eq!(shape.words_per_tile() % 2, 0);
        let nsplit = Self::gemv_splits(shape.out_tiles);
        let params: [u32; 7] = [
            shape.in_tiles as u32,
            shape.out_tiles as u32,
            shape.k,
            (shape.words_per_tile() / 2) as u32,
            mode,
            u32::from(trellis.byte_offset % 4 == 0),
            nsplit as u32,
        ];
        // T2.5: split shapes write partials `[nsplit, out_tiles, 16]` and are
        // summed by `exl3_gemv_reduce`; nsplit == 1 binds `y` directly, which
        // keeps the old flat path byte-identical (split 0 → offset 0).
        let need = nsplit * shape.out_tiles * TILE * 4;
        // Production always passes the decoder arena's partial (see
        // `encode_linear_with_scratch`). Host hooks without one allocate a
        // transient buffer — never retained on `self`.
        let owned = if nsplit > 1 && ext_partial.is_none() {
            Some(self.new_f32_buffer(need / 4))
        } else {
            None
        };
        let dest: &Buffer = if nsplit == 1 {
            y
        } else if let Some(p) = ext_partial {
            debug_assert!(
                p.length() as usize >= need,
                "gemv partial too small: {} < {}",
                p.length(),
                need
            );
            p
        } else {
            owned.as_ref().expect("transient partial")
        };
        enc.set_compute_pipeline_state(&self.gemv);
        enc.set_buffer(0, Some(xh), 0);
        enc.set_buffer(1, Some(trellis.buffer), trellis.byte_offset as u64);
        enc.set_buffer(2, Some(pair), 0);
        enc.set_buffer(3, Some(dest), 0);
        enc.set_bytes(4, 28, params.as_ptr() as *const _);
        // MUST match `WARPS` in exl3.metal (one out-tile per warp). These two
        // constants drifted once and the mismatch looked like a math bug: the
        // kernel indexed shA[] out of bounds and the tests went red.
        const WARPS: usize = 1;
        let groups = shape.out_tiles.div_ceil(WARPS) as u64;
        enc.dispatch_thread_groups(
            MTLSize::new(groups, nsplit as u64, 1),
            MTLSize::new((32 * WARPS) as u64, 1, 1),
        );
        if nsplit > 1 {
            self.encode_gemv_reduce(enc, dest, y, nsplit, shape.out_tiles);
        }
        // Keep any transient partial alive until after the last set_buffer.
        drop(owned);
    }

    /// T2.5: `y[ot] = Σ_s partial[s, ot]` after a K-split gemv.
    fn encode_gemv_reduce(
        &self,
        enc: &ComputeCommandEncoderRef,
        partial: &Buffer,
        y: &Buffer,
        nsplit: usize,
        out_tiles: usize,
    ) {
        let p: [u32; 2] = [out_tiles as u32, nsplit as u32];
        enc.set_compute_pipeline_state(&self.gemv_reduce);
        enc.set_buffer(0, Some(partial), 0);
        enc.set_buffer(1, Some(y), 0);
        enc.set_bytes(2, 8, p.as_ptr() as *const _);
        enc.dispatch_thread_groups(MTLSize::new(out_tiles as u64, 1, 1), MTLSize::new(32, 1, 1));
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

    /// Simdgroup tensor-core GEMM (v19c port): one threadgroup per 64×64
    /// output tile (4 out-tiles × 64 token rows), f16 MMA with f32
    /// accumulate. Caller must gate via [`Self::sg_eligible`] and have padded
    /// the `xh` scratch to `rows.div_ceil(64) * 64` rows (the kernel loads A
    /// fragments device-direct up to a 64-row boundary and masks the store).
    #[allow(clippy::too_many_arguments)]
    fn encode_gemm_sg(
        &self,
        enc: &ComputeCommandEncoderRef,
        xh: &Buffer,
        trellis: DeviceSpan<'_>,
        shape: Exl3Shape,
        mode: u32,
        y: &Buffer,
        ext_partial: Option<&Buffer>,
        rows: usize,
    ) {
        debug_assert!(rows >= 2);
        debug_assert!(Self::sg_eligible(shape, trellis));
        debug_assert_eq!(shape.words_per_tile() % 2, 0);
        // T3.10: under-waved shapes split the stage range across grid.z.
        const SG_BM: usize = 64;
        const SG_OUT_TILES: usize = 4;
        debug_assert_eq!(shape.out_tiles % SG_OUT_TILES, 0);
        let nsplit = Self::sg_splits(shape, rows);
        let rows_pad = rows.div_ceil(SG_BM) * SG_BM;
        let n = shape.out_tiles * TILE;
        let stride = rows_pad * n; // elements, slice pitch (see Exl3GemmSgParams)
        let params: [u32; 9] = [
            shape.in_tiles as u32,
            shape.out_tiles as u32,
            shape.k,
            (shape.words_per_tile() / 2) as u32,
            mode,
            u32::from(trellis.byte_offset % 4 == 0),
            rows as u32,
            nsplit as u32,
            stride as u32,
        ];
        // T3.10: split shapes write partials `[nsplit, rows_pad, N]` and are
        // summed by `exl3_gemm_sg_reduce`; nsplit == 1 binds `y` directly
        // (tg.z=0 → offset 0), keeping the old path byte-identical.
        let need = nsplit * stride * 4;
        // T3.5→scratch-fix: production always passes the decoder arena's
        // partial; host hooks without one allocate a transient (never retained).
        let owned = if nsplit > 1 && ext_partial.is_none() {
            Some(self.new_f32_buffer(need / 4))
        } else {
            None
        };
        let dest: &Buffer = if nsplit == 1 {
            y
        } else if let Some(p) = ext_partial {
            debug_assert!(
                p.length() as usize >= need,
                "sg partial too small: {} < {}",
                p.length(),
                need
            );
            p
        } else {
            owned.as_ref().expect("transient sg partial")
        };
        enc.set_compute_pipeline_state(&self.gemm_sg);
        enc.set_buffer(0, Some(xh), 0);
        enc.set_buffer(1, Some(trellis.buffer), trellis.byte_offset as u64);
        enc.set_buffer(2, Some(&self.inv_perm), 0);
        enc.set_buffer(3, Some(dest), 0);
        enc.set_bytes(4, 36, params.as_ptr() as *const _);
        // MUST match SG_BM/SG_THREADS in exl3.metal: grid over 64-row ×
        // 4-out-tile blocks, 128 threads (4 simdgroups) per block; z = split.
        enc.dispatch_thread_groups(
            MTLSize::new(
                (shape.out_tiles / SG_OUT_TILES) as u64,
                rows.div_ceil(SG_BM) as u64,
                nsplit as u64,
            ),
            MTLSize::new(128, 1, 1),
        );
        if nsplit > 1 {
            self.encode_gemm_sg_reduce(enc, dest, y, nsplit, stride, n * rows);
        }
        // Keep any transient partial alive until after the last set_buffer.
        drop(owned);
    }

    /// T3.10: `y[i] = Σ_s partial[s, i]` over the live `rows × N` elements.
    fn encode_gemm_sg_reduce(
        &self,
        enc: &ComputeCommandEncoderRef,
        partial: &Buffer,
        y: &Buffer,
        nsplit: usize,
        stride: usize,
        total: usize,
    ) {
        let p: [u32; 3] = [nsplit as u32, stride as u32, total as u32];
        enc.set_compute_pipeline_state(&self.gemm_sg_reduce);
        enc.set_buffer(0, Some(partial), 0);
        enc.set_buffer(1, Some(y), 0);
        enc.set_bytes(2, 12, p.as_ptr() as *const _);
        let groups = total.div_ceil(256) as u64;
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Simdgroup-GEMM eligibility (see `exl3_gemm_sg`): even `in_tiles` (the
    /// k-loop stages over 2 in-tiles), `out_tiles % 4 == 0` (64-wide out
    /// blocks), and a 4-byte-aligned trellis (funnel words read as u32 pairs,
    /// same contract as the v19c reference). lm_head is 2-mod-4 but always
    /// takes the rows=1 GEMV path, so it never needs this gate to fire.
    fn sg_eligible(shape: Exl3Shape, trellis: DeviceSpan<'_>) -> bool {
        Self::sg_eligible_at(shape, trellis.byte_offset)
    }

    /// Alignment-only form of `sg_eligible` (policy helpers + tests take the
    /// raw trellis byte_offset so they don't need a `DeviceSpan`/device).
    fn sg_eligible_at(shape: Exl3Shape, trellis_offset: usize) -> bool {
        shape.in_tiles % 2 == 0 && shape.out_tiles % 4 == 0 && trellis_offset % 4 == 0
    }

    /// T3.10 K-split policy for the sg GEMM: fill under-waved dispatches up
    /// to the measured ~320 concurrent threadgroup slots (gate/k phase
    /// profile: gate@256 = 1344 TGs = 4.2 waves × 2.86 ms = 12.03 ms ✓), leave
    /// ≥1-wave shapes unsplit (`base ≥ 320` → 1 → the old grid + direct-to-y
    /// store, byte-identical for gate/down at normal rows), and never exceed
    /// `stages` (an empty slice would write garbage partials). Rows-aware:
    /// a rows-blind policy splits every shape at small `rows` — including
    /// gate at rows=64, where the split is an exact wave-time wash plus
    /// partial I/O — and does nothing at rows=512 where tiny-out k/v already
    /// got their win from the rows term (`base` grows with `rows.div_ceil(64)`).
    fn sg_splits(shape: Exl3Shape, rows: usize) -> usize {
        const TARGET_TGS: usize = 320;
        const MAX_SPLIT: usize = 8;
        let base = (shape.out_tiles / 4).max(1) * rows.div_ceil(64).max(1);
        let stages = (shape.in_tiles / 2).max(1);
        TARGET_TGS.div_ceil(base).clamp(1, MAX_SPLIT).min(stages)
    }

    /// The one gem-dispatch decision point: rows=1 → tuned single-row GEMV;
    /// sg-eligible multi-row → simdgroup tensor-core GEMM; otherwise the
    /// scalar RGT=8 fallback. Shared by production (`encode_linear_with_scratch`)
    /// and the phase bench so measurements reflect what production runs.
    /// `ext_partial`: decoder arena (`encode_linear_with_scratch`); `None` →
    /// transient per-call partial (host test/bench hooks only).
    #[allow(clippy::too_many_arguments)]
    fn encode_gem(
        &self,
        enc: &ComputeCommandEncoderRef,
        xh: &Buffer,
        trellis: DeviceSpan<'_>,
        pair: &Buffer,
        shape: Exl3Shape,
        mode: u32,
        y: &Buffer,
        ext_partial: Option<&Buffer>,
        rows: usize,
    ) {
        if rows == 1 {
            self.encode_gemv(enc, xh, trellis, pair, shape, mode, y, ext_partial);
        } else if Self::sg_eligible(shape, trellis) {
            self.encode_gemm_sg(enc, xh, trellis, shape, mode, y, ext_partial, rows);
        } else {
            self.encode_gemm(enc, xh, trellis, pair, shape, mode, y, rows);
        }
    }

    /// `xh` rows to reserve: the sg kernel's device-direct A loads read up to
    /// the next 64-row boundary above `rows`; pad rows are multiplied into
    /// outputs the store masks away (`gr < blk_len`). Non-sg paths need no
    /// padding (they only ever address rows < `rows`).
    fn xh_rows(rows: usize, shape: Exl3Shape, trellis: DeviceSpan<'_>) -> usize {
        if rows >= 2 && Self::sg_eligible(shape, trellis) {
            rows.div_ceil(64) * 64
        } else {
            rows
        }
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
        self.with_scratch(n, 0, 1, |sc| {
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
        self.with_scratch(xh.len(), shape.out_features(), 1, |sc| {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    xh.as_ptr() as *const u8,
                    sc.xh.contents() as *mut u8,
                    xh.len() * 2,
                );
            }
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.encode_gemv(enc, &sc.xh, trellis, &pair, shape, mode, &sc.y, None);
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
        self.with_scratch(0, n, 1, |sc| {
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

        self.with_scratch(inn, outp, 1, |sc| {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    x.as_ptr() as *const u8,
                    sc.x_stage.contents() as *mut u8,
                    inn * 2,
                );
            }
            self.submit(|enc| {
                self.encode_had_in(enc, &sc.x_stage, suh, &sc.xh, 1, inn);
                self.encode_gemv(enc, &sc.xh, trellis, &pair_buf, shape, mode, &sc.y, None);
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
        let x_rows = Self::xh_rows(rows, shape, trellis);
        self.with_scratch(x_rows * inn, rows * outp, rows, |sc| {
            self.encode_had_in(enc, x, suh, &sc.xh, rows, inn);
            self.encode_gem(
                enc, &sc.xh, trellis, &pair_buf, shape, mode, &sc.y, None, rows,
            );
            self.encode_had_out(enc, &sc.y, svh, out, rows, outp);
        });
        Ok(())
    }

    /// Microbench: time `had_in`, `gemm`/`gemv`, `had_out` separately for one
    /// linear. Each phase gets its own command buffer + wait (wall-clock per
    /// phase, not overlapped). Returns `(had_in_ms, gem_ms, had_out_ms)`.
    ///
    /// Test/bench hook only — production uses [`Self::encode_linear`].
    #[allow(clippy::too_many_arguments)]
    pub fn bench_linear_phases(
        &self,
        x: &Buffer,
        trellis: DeviceSpan<'_>,
        suh: DeviceSpan<'_>,
        svh: DeviceSpan<'_>,
        out: &Buffer,
        shape: Exl3Shape,
        mode: u32,
        rows: usize,
        reps: usize,
    ) -> Result<(f64, f64, f64), KernelError> {
        let inn = shape.in_features();
        let outp = shape.out_features();
        check_span_fits(
            &trellis,
            shape.in_tiles * shape.out_tiles * shape.words_per_tile() * 2,
            "trellis",
            2,
        )?;
        check_span_fits(&suh, inn * 2, "suh", 2)?;
        check_span_fits(&svh, outp * 2, "svh", 2)?;
        let pair_buf = self.pair_buffer(shape.k);
        let x_rows = Self::xh_rows(rows, shape, trellis);

        let mut result = (0.0f64, 0.0f64, 0.0f64);
        self.with_scratch(x_rows * inn, rows * outp, rows, |sc| {
            // Warm-up (shader specialization / scratch residency).
            for _ in 0..2 {
                self.submit(|enc| {
                    self.encode_had_in(enc, x, suh, &sc.xh, rows, inn);
                    self.encode_gem(enc, &sc.xh, trellis, &pair_buf, shape, mode, &sc.y, None, rows);
                    self.encode_had_out(enc, &sc.y, svh, out, rows, outp);
                })
                .expect("warmup submit");
            }

            let time_in = |gemv: &Exl3Gemv, sc: &Scratch| -> f64 {
                let _ = gemv;
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    gemv.submit(|enc| {
                        gemv.encode_had_in(enc, x, suh, &sc.xh, rows, inn);
                    })
                    .expect("had_in submit");
                }
                t0.elapsed().as_secs_f64() * 1e3 / reps as f64
            };
            let time_gem = |gemv: &Exl3Gemv, sc: &Scratch| -> f64 {
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    gemv.submit(|enc| {
                        gemv.encode_gem(enc, &sc.xh, trellis, &pair_buf, shape, mode, &sc.y, None, rows);
                    })
                    .expect("gem submit");
                }
                t0.elapsed().as_secs_f64() * 1e3 / reps as f64
            };
            let time_out = |gemv: &Exl3Gemv, sc: &Scratch| -> f64 {
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    gemv.submit(|enc| {
                        gemv.encode_had_out(enc, &sc.y, svh, out, rows, outp);
                    })
                    .expect("had_out submit");
                }
                t0.elapsed().as_secs_f64() * 1e3 / reps as f64
            };

            let had_in_ms = time_in(self, sc);
            let gem_ms = time_gem(self, sc);
            let had_out_ms = time_out(self, sc);

            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                self.submit(|enc| {
                    self.encode_had_in(enc, x, suh, &sc.xh, rows, inn);
                    self.encode_gem(enc, &sc.xh, trellis, &pair_buf, shape, mode, &sc.y, None, rows);
                    self.encode_had_out(enc, &sc.y, svh, out, rows, outp);
                })
                .expect("full submit");
            }
            let full = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
            eprintln!(
                "  phases: had_in={had_in_ms:.3} gem={gem_ms:.3} had_out={had_out_ms:.3} full={full:.3} ms \
                 (sum={:.3})",
                had_in_ms + gem_ms + had_out_ms
            );
            result = (had_in_ms, gem_ms, had_out_ms);
        });
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The duplicated inverse permutation must match the engine's recorded
    /// head (crates/engine/src/exl3/perm.rs `perm_inv_head_matches_reference`)
    /// and be a bijection — a wrong inv silently decodes every weight into
    /// the wrong row-major position, which still "looks like" numbers.
    #[test]
    fn inv_perm_head_matches_engine_reference() {
        let inv = tensor_core_perm_inverse();
        let want = [
            0u32, 32, 64, 96, 128, 160, 192, 224, 4, 36, 68, 100, 132, 164, 196, 228,
        ];
        assert_eq!(&inv[..16], &want);
        let mut seen = [false; 256];
        for &v in &inv {
            assert!((v as usize) < 256, "out of range {v}");
            assert!(!seen[v as usize], "duplicate {v}");
            seen[v as usize] = true;
        }
    }

    // -- T3.8 probes: exl3_gemm_sg_ceiling / exl3_gemm_sg_decode_only ------------
    //
    // Test-only pipelines (production Exl3Gemv never binds these). Ceiling =
    // the sg kernel with funnel decode replaced by direct half loads from a
    // pre-decoded w[N, K]; decode_only = the funnel decode + wblk staging +
    // per-stage barriers with the MMA/store skipped. tax = 1 - t(real
    // sg)/t(ceiling) isolates the decode+staging cost of the production path.

    fn probe_ctx() -> Option<(Device, CommandQueue)> {
        let d = Device::system_default()?;
        let q = d.new_command_queue();
        Some((d, q))
    }

    /// T2.5 policy (T2.9 deepened): TARGET 4096 TGs — lm_head (16384 tiles)
    /// stays 1, gate → 4, q → 8, down/o → 13, k/v → 16 (was 4 at 94/106
    /// Gw/s); tiny shapes clamp to 16. Partial scratch ≤ ~344 KB transient.
    #[test]
    fn gemv_splits_policy_covers_shapes() {
        assert_eq!(Exl3Gemv::gemv_splits(16384), 1); // lm_head (already 16k TGs)
        assert_eq!(Exl3Gemv::gemv_splits(4096), 1); // exactly target
        assert_eq!(Exl3Gemv::gemv_splits(1344), 4); // gate (21504 out), was 1
        assert_eq!(Exl3Gemv::gemv_splits(512), 8); // q (8192 out), was 2
        assert_eq!(Exl3Gemv::gemv_splits(336), 13); // down/o (5376 out), was 4
        assert_eq!(Exl3Gemv::gemv_splits(256), 16); // k/v (4096 out), was 4
        assert_eq!(Exl3Gemv::gemv_splits(64), 16); // clamped (want 64)
        assert_eq!(Exl3Gemv::gemv_splits(1), 16);
    }

    /// T3.10 policy: split only under-waved shapes (base_tgs < 320), rows-
    /// aware so gate/down stay unsplit (byte-identical) at normal rows, and
    /// never exceed `stages` (empty slice would write garbage partials).
    #[test]
    fn sg_splits_policy_covers_shapes() {
        let sh = |in_f: usize, out_f: usize| Exl3Shape::new(in_f, out_f, 3).expect("shape");
        let gate = sh(5376, 21504); // out_tiles 1344
        let kv = sh(5376, 4096); // out_tiles 256
        let q = sh(5376, 8192); // out_tiles 512
        let o = sh(5376, 5376); // out_tiles 336
        let tiny_kv = sh(5376, 128); // out_tiles 8 (sliding-window layers)
        let stages_capped = sh(32, 512); // in_tiles 2 → stages 1
        // gate/down: ≥1 wave at every realistic rows → unsplit everywhere.
        for rows in [2usize, 64, 128, 256, 512, 8192] {
            assert_eq!(Exl3Gemv::sg_splits(gate, rows), 1, "gate rows={rows}");
        }
        // k/v: split only while under-waved (base < 320); rows=512 already
        // 512 TGs → exact-wash regime → must stay unsplit.
        assert_eq!(Exl3Gemv::sg_splits(kv, 64), 5); // base 64
        assert_eq!(Exl3Gemv::sg_splits(kv, 256), 2); // base 256
        assert_eq!(Exl3Gemv::sg_splits(kv, 512), 1); // base 512
        assert_eq!(Exl3Gemv::sg_splits(kv, 8192), 1);
        // q: mild split at tiny rows only.
        assert_eq!(Exl3Gemv::sg_splits(q, 64), 3); // base 128
        assert_eq!(Exl3Gemv::sg_splits(q, 256), 1); // base 512
        // o: base 84 at rows=64 → 4; rows=256 → 336 → 1.
        assert_eq!(Exl3Gemv::sg_splits(o, 64), 4);
        assert_eq!(Exl3Gemv::sg_splits(o, 256), 1);
        // tiny-out k/v layers: still starved at rows=512 → full 8.
        assert_eq!(Exl3Gemv::sg_splits(tiny_kv, 512), 8); // base 16
        assert_eq!(Exl3Gemv::sg_splits(tiny_kv, 8192), 2); // base 256
        // stages cap: never split past the slice count (min with stages=1).
        assert_eq!(Exl3Gemv::sg_splits(stages_capped, 64), 1);
    }

    /// T3.5 policy: `chunk_scratch_needs` single-sources `xh_rows` /
    /// `sg_eligible` / `sg_splits` for the per-chunk prefill arena. The sg
    /// K-split partial (up to ~8.4 MB at rows=256 k/v) must land here, never
    /// in the shared decode `partial` (≤ ~344 KB, R4).
    #[test]
    fn chunk_scratch_needs_policy() {
        let sh = |in_f: usize, out_f: usize| Exl3Shape::new(in_f, out_f, 3).expect("shape");
        let gate = sh(5376, 21504); // out_tiles 1344, eligible
        let kv = sh(5376, 4096); // out_tiles 256, eligible
        let lm = sh(21504, 262144); // lm_head, out_tiles 16384
        let aligned = 0usize; // trellis byte_offset % 4 == 0
        let misaligned = 2usize;

        // rows=1 (decode): no 64-row xh pad, no sg path, partial = 0.
        assert_eq!(
            Exl3Gemv::chunk_scratch_needs(kv, aligned, 1),
            (5376, 4096, 0),
            "rows=1 must not pad or split"
        );

        // k/v @ rows=256: eligible + nsplit=2 → partial = 2×256×4096×4.
        assert_eq!(
            Exl3Gemv::chunk_scratch_needs(kv, aligned, 256),
            (256 * 5376, 256 * 4096, 2 * 256 * 4096 * 4),
            "k/v rows=256 arena: xh unpadded (256%64==0), partial 8388608 B"
        );
        assert_eq!(
            Exl3Gemv::chunk_scratch_needs(kv, aligned, 64).2,
            5 * 64 * 4096 * 4
        );
        assert_eq!(
            Exl3Gemv::chunk_scratch_needs(kv, aligned, 512).2,
            0,
            "nsplit=1 @512"
        );

        // gate @ rows=256: eligible (xh pads) but nsplit=1 → partial = 0.
        let (gx, gy, gp) = Exl3Gemv::chunk_scratch_needs(gate, aligned, 256);
        assert_eq!((gx, gy), (256 * 5376, 256 * 21504));
        assert_eq!(gp, 0, "gate must never need sg partial");

        // Ragged rows: xh pads to 64 (100 → 128), y does not.
        let (kx, ky, kp) = Exl3Gemv::chunk_scratch_needs(kv, aligned, 100);
        assert_eq!(kx, 128 * 5376, "xh pads to 64-row boundary");
        assert_eq!(ky, 100 * 4096, "y is rows×N, unpadded");
        assert_eq!(kp, Exl3Gemv::sg_partial_bytes(kv, aligned, 100));

        // Misaligned trellis (byte_offset % 4 != 0): not sg-eligible → no pad,
        // no partial, even for k/v at rows=256.
        assert_eq!(
            Exl3Gemv::chunk_scratch_needs(kv, misaligned, 256),
            (256 * 5376, 256 * 4096, 0)
        );

        // lm_head @ rows=1 (last-chunk fallback uses encode_linear, but the
        // sizes helper must still be well-defined): out%16 ok, rows=1 → no pad.
        assert_eq!(
            Exl3Gemv::chunk_scratch_needs(lm, aligned, 1),
            (1 * 21504, 1 * 262144, 0)
        );
    }

    fn build_probe(device: &Device, name: &str) -> ComputePipelineState {
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let lib = device
            .new_library_with_source(MSL, &opts)
            .unwrap_or_else(|e| panic!("probe MSL compile ({name}): {e}"));
        let f = lib
            .get_function(name, None)
            .unwrap_or_else(|e| panic!("probe function {name}: {e}"));
        device
            .new_compute_pipeline_state_with_function(&f)
            .unwrap_or_else(|e| panic!("probe pipeline {name}: {e}"))
    }

    fn probe_submit(
        queue: &CommandQueue,
        f: impl FnOnce(&ComputeCommandEncoderRef),
    ) -> Result<(), KernelError> {
        let cb = queue.new_command_buffer();
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

    fn dispatch_sg(enc: &ComputeCommandEncoderRef, out_tiles: usize, rows: usize) {
        enc.dispatch_thread_groups(
            MTLSize::new((out_tiles / 4) as u64, rows.div_ceil(64) as u64, 1),
            MTLSize::new(128, 1, 1),
        );
    }

    /// Ceiling kernel vs a CPU f32 matmul on a small sg-eligible shape:
    /// y[row, col] = sum_k xh[row, k] * w[col, k]. Inputs are f16 on both
    /// sides (products exact in f32); only the summation order differs
    /// (MMA k-tiles vs sequential k), so the gate is a tight relative bound.
    #[test]
    fn sg_ceiling_matches_cpu_small() {
        let Some((device, queue)) = probe_ctx() else {
            eprintln!("skip: no Metal device");
            return;
        };
        const K: usize = 64; // in_tiles = 4 (even), stages = 2
        const N: usize = 128; // out_tiles = 8 (% 4 == 0)
        const ROWS: usize = 64; // one full 64-row block, no ragged tail

        let mut rng = 0x9E37_79B9u32;
        let mut next = move || {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (rng >> 16) as u16
        };
        let rand_f16 = |n: &mut dyn FnMut() -> u16| -> f16 {
            // [1, 2) minus 1.5 → [-0.5, 0.5): valid normal f16 throughout.
            f16::from_bits((n() & 0x3BFF) | 0x3800) - f16::from_f32(1.5)
        };

        let mut xh_v = Vec::with_capacity(ROWS * K);
        for _ in 0..ROWS * K {
            xh_v.push(rand_f16(&mut next));
        }
        let mut w_v = Vec::with_capacity(N * K);
        for _ in 0..N * K {
            w_v.push(rand_f16(&mut next));
        }

        let xh = device.new_buffer_with_data(
            xh_v.as_ptr() as *const _,
            (ROWS * K * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let w = device.new_buffer_with_data(
            w_v.as_ptr() as *const _,
            (N * K * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let y = device.new_buffer((ROWS * N * 4) as u64, MTLResourceOptions::StorageModeShared);

        let pipe = build_probe(&device, "exl3_gemm_sg_ceiling");
        let in_tiles = K / TILE;
        let out_tiles = N / TILE;
        let params: [u32; 7] = [in_tiles as u32, out_tiles as u32, 6, 0, 0, 1, ROWS as u32];
        probe_submit(&queue, |enc| {
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(&xh), 0);
            enc.set_buffer(1, Some(&w), 0);
            enc.set_buffer(2, Some(&y), 0);
            enc.set_bytes(3, 28, params.as_ptr() as *const _);
            dispatch_sg(enc, out_tiles, ROWS);
        })
        .expect("ceiling dispatch");

        let got = unsafe { std::slice::from_raw_parts(y.contents() as *const f32, ROWS * N) };
        let mut worst = 0f32;
        let mut scale = 0f32;
        for r in 0..ROWS {
            for c in 0..N {
                let mut acc = 0f32;
                for k in 0..K {
                    acc += xh_v[r * K + k].to_f32() * w_v[c * K + k].to_f32();
                }
                let d = (got[r * N + c] - acc).abs();
                worst = worst.max(d);
                scale = scale.max(acc.abs());
            }
        }
        let rel = if scale > 0.0 { worst / scale } else { worst };
        assert!(
            rel < 1e-3,
            "ceiling vs CPU max_rel={rel:e} (worst_abs={worst:e})"
        );
    }

    /// Gate-shape timing probe (env-gated so the fast kernels suite stays
    /// fast): ceiling TFLOP/s and decode-only ms on rows=256, K=5376,
    /// N=21504 — the real `mlp.gate_proj` shape. For context the engine's
    /// `phase_profile_chunk_budget_slow` measures the production sg kernel
    /// (with funnel decode) at 11.98 ms ≈ 4.91 TFLOP/s on this shape.
    #[test]
    fn sg_ceiling_decode_tax_gate_shape() {
        if std::env::var_os("KAPLUMBA_RUN_SLOW").is_none() {
            eprintln!("skip: set KAPLUMBA_RUN_SLOW=1 for the T3.8 timing probe");
            return;
        }
        let Some((device, queue)) = probe_ctx() else {
            eprintln!("skip: no Metal device");
            return;
        };
        const K: usize = 5376; // in_tiles 336, even
        const N: usize = 21504; // out_tiles 1344, %4==0
        const ROWS: usize = 256; // production chunk rows
        const FLOPS: f64 = 2.0 * ROWS as f64 * K as f64 * N as f64; // 59.19 GF
        let in_tiles = K / TILE;
        let out_tiles = N / TILE;

        let mut rng = 0x1234_5678u32;
        let mut next = move || {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (rng >> 16) as u16
        };

        // w: non-zero normal f16 in [1, 2) — timing-neutral, no denormals.
        let w = {
            let n = N * K;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push((next() & 0x3BFF) | 0x3800);
            }
            device.new_buffer_with_data(
                v.as_ptr() as *const _,
                (n * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let xh = {
            let n = ROWS * K;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push((next() & 0x3BFF) | 0x3800);
            }
            device.new_buffer_with_data(
                v.as_ptr() as *const _,
                (n * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let y = device.new_buffer((ROWS * N * 4) as u64, MTLResourceOptions::StorageModeShared);

        // decode_only: zero trellis (cw=0 is a safe exl3_decode input) at the
        // production twords layout; real inv_perm so the address math matches.
        let twords = 8 * 6; // stream_words for k=6, as production sets it
        let trellis = device.new_buffer(
            (in_tiles * out_tiles * twords * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ); // zero-filled (new_buffer is uninitialized — clear it)
        unsafe {
            std::ptr::write_bytes(trellis.contents() as *mut u8, 0, trellis.length() as usize);
        }
        let inv = tensor_core_perm_inverse();
        let inv_perm = device.new_buffer_with_data(
            inv.as_ptr() as *const _,
            (inv.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let params_ceiling: [u32; 7] = [in_tiles as u32, out_tiles as u32, 6, 0, 0, 1, ROWS as u32];
        let params_dec: [u32; 7] = [
            in_tiles as u32,
            out_tiles as u32,
            6,
            twords as u32,
            0,
            1,
            ROWS as u32,
        ];
        // T3.10: the production sg kernel now reads Exl3GemmSgParams (36 B);
        // nsplit=1 → old unsplit path, stride unused (tg.z=0).
        let mut params_real: [u32; 9] = [0; 9];
        params_real[..7].copy_from_slice(&params_dec);
        params_real[7] = 1;

        let ceil_pipe = build_probe(&device, "exl3_gemm_sg_ceiling");
        let dec_pipe = build_probe(&device, "exl3_gemm_sg_decode_only");
        // Real production kernel on the same synthetic buffers (zero trellis =
        // cw=0, safe decode) so the tax compares gem-only against gem-only —
        // the engine budget's 11.98 ms includes had_in/had_out around the gem.
        let real_pipe = build_probe(&device, "exl3_gemm_sg");

        let run_ceiling = || {
            probe_submit(&queue, |enc| {
                enc.set_compute_pipeline_state(&ceil_pipe);
                enc.set_buffer(0, Some(&xh), 0);
                enc.set_buffer(1, Some(&w), 0);
                enc.set_buffer(2, Some(&y), 0);
                enc.set_bytes(3, 28, params_ceiling.as_ptr() as *const _);
                dispatch_sg(enc, out_tiles, ROWS);
            })
            .expect("ceiling dispatch");
        };
        let run_real = || {
            probe_submit(&queue, |enc| {
                enc.set_compute_pipeline_state(&real_pipe);
                enc.set_buffer(0, Some(&xh), 0);
                enc.set_buffer(1, Some(&trellis), 0);
                enc.set_buffer(2, Some(&inv_perm), 0);
                enc.set_buffer(3, Some(&y), 0);
                enc.set_bytes(4, 36, params_real.as_ptr() as *const _);
                dispatch_sg(enc, out_tiles, ROWS);
            })
            .expect("real sg dispatch");
        };
        let run_decode = || {
            probe_submit(&queue, |enc| {
                enc.set_compute_pipeline_state(&dec_pipe);
                enc.set_buffer(0, Some(&trellis), 0);
                enc.set_buffer(1, Some(&inv_perm), 0);
                enc.set_bytes(2, 28, params_dec.as_ptr() as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new((out_tiles / 4) as u64, ROWS.div_ceil(64) as u64, 1),
                    MTLSize::new(128, 1, 1),
                );
            })
            .expect("decode_only dispatch");
        };

        // Warmup (shader compile lands here), then min of 5 runs each.
        run_ceiling();
        run_real();
        run_decode();
        let mut ceil_best = f64::MAX;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            run_ceiling();
            ceil_best = ceil_best.min(t.elapsed().as_secs_f64());
        }
        let mut real_best = f64::MAX;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            run_real();
            real_best = real_best.min(t.elapsed().as_secs_f64());
        }
        let mut dec_best = f64::MAX;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            run_decode();
            dec_best = dec_best.min(t.elapsed().as_secs_f64());
        }

        let ceil_tf = FLOPS / ceil_best / 1e12;
        let real_tf = FLOPS / real_best / 1e12;
        // Fraction of production gem time that is funnel+staging overhead
        // (ceiling is the same kernel with the funnel replaced by direct loads).
        let tax = 1.0 - ceil_best / real_best;
        eprintln!(
            "[T3.8] real sg (gem-only): {:.3} ms = {:.2} TFLOP/s | ceiling: {:.3} ms = {:.2} \
             TFLOP/s | decode_only: {:.3} ms | funnel tax = {:.1}% of gem time",
            real_best * 1e3,
            real_tf,
            ceil_best * 1e3,
            ceil_tf,
            dec_best * 1e3,
            tax * 100.0
        );
        eprintln!(
            "[T3.8] context: engine budget linear_ms (gem + had_in/out) = 11.98 ms on this \
             shape; 200 pp/s needs ≈11.4 TFLOP/s sustained on linears"
        );
        // Sanity floors: a silent kernel failure shows up as absurd times, not
        // wrong numbers (decode_only writes nothing). 1 TFLOP/s is ~5x below
        // production, so these only trip on a broken dispatch.
        assert!(
            ceil_tf > 1.0,
            "ceiling implausibly slow: {ceil_tf:.3} TFLOP/s — dispatch broken?"
        );
        assert!(
            real_tf > 1.0,
            "real sg implausibly slow: {real_tf:.3} TFLOP/s — dispatch broken?"
        );
        assert!(
            dec_best < ceil_best,
            "decode_only ({:.3} ms) should not exceed ceiling ({:.3} ms)",
            dec_best * 1e3,
            ceil_best * 1e3
        );
    }

    /// Sustained-MMA peak probe (env-gated): wblk filled once, then
    /// PEAK_REPS identical passes over hot fragments — the absolute simdgroup
    /// f16→f32 rate of this GPU for the production tile config, with no
    /// weight streaming or decode. 200 pp/s needs linears ≈ 1065 ms/chunk
    /// ≈ 11.4 TFLOP/s sustained; this number says whether any staging/order
    /// tweak could ever get there.
    #[test]
    fn sg_mma_peak_probe() {
        if std::env::var_os("KAPLUMBA_RUN_SLOW").is_none() {
            eprintln!("skip: set KAPLUMBA_RUN_SLOW=1 for the T3.8 peak probe");
            return;
        }
        let Some((device, queue)) = probe_ctx() else {
            eprintln!("skip: no Metal device");
            return;
        };
        const K: usize = 5376;
        const N: usize = 21504;
        const ROWS: usize = 256;
        const PEAK_REPS: u64 = 32; // MUST match PEAK_REPS in exl3.metal
        let in_tiles = K / TILE;
        let out_tiles = N / TILE;
        let tgs = (out_tiles / 4) as u64 * ROWS.div_ceil(64) as u64;
        // One rep = one 64×64 tile over k2×8 = 32 deep.
        let flops = 2.0 * (64.0 * 64.0 * 32.0) * PEAK_REPS as f64 * tgs as f64;

        let mut rng = 0x8765_4321u32;
        let mut next = move || {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (rng >> 16) as u16
        };
        let w = {
            let n = N * K;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push((next() & 0x3BFF) | 0x3800);
            }
            device.new_buffer_with_data(
                v.as_ptr() as *const _,
                (n * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let xh = {
            let n = ROWS * K;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push((next() & 0x3BFF) | 0x3800);
            }
            device.new_buffer_with_data(
                v.as_ptr() as *const _,
                (n * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let y = device.new_buffer((ROWS * N * 4) as u64, MTLResourceOptions::StorageModeShared);

        let pipe = build_probe(&device, "exl3_gemm_sg_mma_peak");
        let params: [u32; 7] = [in_tiles as u32, out_tiles as u32, 6, 0, 0, 1, ROWS as u32];
        let run = || {
            probe_submit(&queue, |enc| {
                enc.set_compute_pipeline_state(&pipe);
                enc.set_buffer(0, Some(&xh), 0);
                enc.set_buffer(1, Some(&w), 0);
                enc.set_buffer(2, Some(&y), 0);
                enc.set_bytes(3, 28, params.as_ptr() as *const _);
                dispatch_sg(enc, out_tiles, ROWS);
            })
            .expect("peak dispatch");
        };

        run(); // warmup / shader compile
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            run();
            best = best.min(t.elapsed().as_secs_f64());
        }
        let tf = flops / best / 1e12;
        eprintln!(
            "[T3.8] mma_peak: {:.3} ms = {:.2} TFLOP/s (hot fragments, {} reps) | \
             ceiling(streaming) 6.27 TF | production 4.91 TF | 200 pp/s needs ≈11.4 TF \
             sustained on linears",
            best * 1e3,
            tf,
            PEAK_REPS
        );
        assert!(
            tf > 5.0 && tf < 50.0,
            "mma peak implausible: {tf:.3} TFLOP/s — probe broken?"
        );
    }
}
