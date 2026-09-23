//! GPU EXL3 linear dispatch (T2.1).
//!
//! Routes the engine's `linear_forward_reconstruct` calls through the Metal
//! kernels in `kaplumba-kernels` when the tensors are device-resident and the
//! shape/format is one the kernels support. Anything unsupported returns
//! `Ok(None)` so the caller keeps using the CPU reference path — the GPU path
//! is an optimization, never a second source of truth.
//!
//! Eligibility (all must hold):
//! * `trellis` is `[in_tiles, out_tiles, 16*k]` with `1 <= k <= 8`, 2-byte words;
//! * `in_features` and `out_features` are multiples of 128 (Hadamard blocks);
//! * `suh`/`svh` both present and **f16** (the packed sign-bitfield form is
//!   CPU-only for now);
//! * codebook mode `Default` or `Mcg` (`Mul1` is CPU-only);
//! * no `bias`;
//! * a single row of activations (`rows == 1`, the decode shape).

use half::f16;
use kaplumba_kernels::{AttnParams, DeviceSpan, Exl3Attention, Exl3Gemv, Exl3Shape, KernelError};
use metal::{Buffer, MTLResourceOptions};

use crate::exl3::codebook::CodebookMode;
use crate::gemma4::TensorSource;
use crate::load::DType;
use crate::metal_ctx::MetalContext;

#[derive(Debug, thiserror::Error)]
pub enum MetalLinearError {
    #[error(transparent)]
    Kernel(#[from] KernelError),
}

/// Owns the compiled pipelines + scratch for the GPU decode path.
pub struct MetalLinear {
    device: metal::Device,
    queue: metal::CommandQueue,
    gemv: Exl3Gemv,
    attn: Exl3Attention,
    fuse: kaplumba_kernels::Fuse,
    /// Attention staging buffers, grown on demand (T3.5). The engine still hands
    /// attention host slices, so q/k/v are copied in and `out` copied back; the
    /// KV cache (T5.1) will keep K/V device-resident and drop the copies.
    attn_scratch: std::cell::RefCell<Option<AttnScratch>>,
    attn_ns: std::sync::atomic::AtomicU64,
    /// Linears actually dispatched to the GPU (proof that eligibility held for
    /// a real run — a silent fallback would otherwise look like a slow GPU).
    calls: std::sync::atomic::AtomicU64,
    /// Wall time spent inside the GPU dispatches (R7.2: separates kernel cost
    /// from whatever else the forward pass is doing).
    gpu_ns: std::sync::atomic::AtomicU64,
}

struct AttnScratch {
    q: Buffer,
    k: Buffer,
    v: Buffer,
    out: Buffer,
    q_elems: usize,
    kv_elems: usize,
}

/// Owned device handles for one EXL3 linear.
///
/// Owned (`Buffer` clones + byte offsets) rather than borrowed spans so the
/// decode pipeline can resolve all 411 linear weights **once** at construction
/// and keep them, instead of re-validating metadata on every token.
#[derive(Clone)]
pub struct LinearHandles {
    pub name: String,
    pub shape: Exl3Shape,
    pub mode: u32,
    trellis: (Buffer, usize),
    suh: (Buffer, usize),
    svh: (Buffer, usize),
}

impl LinearHandles {
    pub fn trellis(&self) -> DeviceSpan<'_> {
        DeviceSpan {
            buffer: &self.trellis.0,
            byte_offset: self.trellis.1,
        }
    }

    pub fn suh(&self) -> DeviceSpan<'_> {
        DeviceSpan {
            buffer: &self.suh.0,
            byte_offset: self.suh.1,
        }
    }

    pub fn svh(&self) -> DeviceSpan<'_> {
        DeviceSpan {
            buffer: &self.svh.0,
            byte_offset: self.svh.1,
        }
    }
}

impl MetalLinear {
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalLinearError> {
        Ok(Self {
            device: ctx.device().clone(),
            queue: ctx.queue().clone(),
            gemv: Exl3Gemv::new(ctx.device(), ctx.queue())?,
            attn: Exl3Attention::new(ctx.device(), ctx.queue())?,
            fuse: kaplumba_kernels::Fuse::new(ctx.device(), ctx.queue())?,
            attn_scratch: std::cell::RefCell::new(None),
            calls: std::sync::atomic::AtomicU64::new(0),
            gpu_ns: std::sync::atomic::AtomicU64::new(0),
            attn_ns: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Wall time inside GPU attention dispatches, in milliseconds.
    pub fn attn_ms(&self) -> f64 {
        self.attn_ns.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6
    }

    /// Number of linears dispatched to the GPU since construction.
    pub fn calls(&self) -> u64 {
        self.calls.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Total wall time inside GPU dispatches, in milliseconds.
    pub fn gpu_ms(&self) -> f64 {
        self.gpu_ns.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6
    }

    /// Low-level kernel access (per-pass tests and T2.3 microbenchmarks).
    pub fn raw(&self) -> &Exl3Gemv {
        &self.gemv
    }

    /// Why `try_linear(src, name)` would fall back to the CPU reference, or
    /// `None` when the linear is GPU-eligible. Cheap (metadata only) — call it
    /// to prove a real run is not silently CPU-bound (`gpu_linears` in the
    /// bench report only counts dispatches, not the reason).
    pub fn skip_reason(&self, src: &dyn TensorSource, name: &str) -> Option<String> {
        match gather_checked(src, name) {
            Ok(_) => None,
            Err(e) => Some(e),
        }
    }

    /// Single-row `y = had(had(x·suh) @ W)·svh` on the GPU.
    ///
    /// * `Ok(Some(y))` — computed on the GPU, `y.len() == out_features`.
    /// * `Ok(None)` — not GPU-eligible; caller must use the CPU path.
    /// * `Err(_)` — a kernel/dispatch failure (never silently wrong).
    pub fn try_linear(
        &self,
        src: &dyn TensorSource,
        name: &str,
        x: &[f16],
    ) -> Result<Option<Vec<f16>>, MetalLinearError> {
        let Some(g) = linear_handles(src, name).ok() else {
            return Ok(None);
        };
        if x.len() != g.shape.in_features() {
            return Ok(None);
        }
        let mut out = Vec::new();
        let t0 = std::time::Instant::now();
        self.gemv
            .linear_row(x, g.trellis(), g.suh(), g.svh(), g.shape, g.mode, &mut out)?;
        self.gpu_ns.fetch_add(
            t0.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Some(out))
    }
}

impl MetalLinear {
    /// `out = attn(q, k, v)` on the GPU when the shape is supported.
    ///
    /// * `Ok(Some(out))` — computed on the GPU, `out.len() == q.len()`.
    /// * `Ok(None)` — shape not covered by the kernels; caller uses the CPU path.
    /// * `Err(_)` — dispatch/allocation failure (never silently wrong).
    ///
    /// `q` is `[s, n_heads, head_dim]`, `k`/`v` are `[kv_rows, n_kv, head_dim]`,
    /// all f16, with the position bases carried in `p` (`ATTN_ROWS`-batched query
    /// rows per threadgroup, one dispatch over all heads).
    pub fn try_attention(
        &self,
        q: &[f16],
        k: &[f16],
        v: &[f16],
        p: AttnParams,
    ) -> Result<Option<Vec<f16>>, MetalLinearError> {
        if p.part().is_none()
            || q.len() != p.s * p.n_heads * p.head_dim
            || k.len() != p.kv_rows * p.n_kv * p.head_dim
            || v.len() != k.len()
        {
            return Ok(None);
        }

        let q_elems = q.len();
        let kv_elems = k.len();
        let mut slot = self.attn_scratch.borrow_mut();
        let needs_grow = match slot.as_ref() {
            Some(sc) => sc.q_elems < q_elems || sc.kv_elems < kv_elems,
            None => true,
        };
        if needs_grow {
            let q_cap = slot.as_ref().map(|s| s.q_elems).unwrap_or(0).max(q_elems);
            let kv_cap = slot.as_ref().map(|s| s.kv_elems).unwrap_or(0).max(kv_elems);
            let alloc = |n: usize| {
                self.device
                    .new_buffer((n * 2).max(2) as u64, MTLResourceOptions::StorageModeShared)
            };
            *slot = Some(AttnScratch {
                q: alloc(q_cap),
                k: alloc(kv_cap),
                v: alloc(kv_cap),
                out: alloc(q_cap),
                q_elems: q_cap,
                kv_elems: kv_cap,
            });
        }
        let sc = slot.as_ref().expect("just allocated");
        copy_in(&sc.q, q);
        copy_in(&sc.k, k);
        copy_in(&sc.v, v);

        fn span(b: &Buffer) -> DeviceSpan<'_> {
            DeviceSpan {
                buffer: b,
                byte_offset: 0,
            }
        }
        let t0 = std::time::Instant::now();
        self.attn
            .run(span(&sc.q), span(&sc.k), span(&sc.v), span(&sc.out), p)?;
        self.attn_ns.fetch_add(
            t0.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        let mut out = vec![f16::ZERO; q_elems];
        unsafe {
            std::ptr::copy_nonoverlapping(
                sc.out.contents() as *const u8,
                out.as_mut_ptr() as *mut u8,
                q_elems * 2,
            );
        }
        Ok(Some(out))
    }
}

fn copy_in(dst: &Buffer, src: &[f16]) {
    unsafe {
        std::ptr::copy_nonoverlapping(
            src.as_ptr() as *const u8,
            dst.contents() as *mut u8,
            src.len() * 2,
        );
    }
}

/// Device handles for `name`, or the reason it cannot go on the GPU.
///
/// Public (and returning a reason rather than `None`) because a silent CPU
/// fallback is indistinguishable from a slow GPU in a bench report: see
/// `MetalLinear::skip_reason` and the `all_layer_linears_gpu_eligible_slow`
/// guard test.
pub fn linear_handles(src: &dyn TensorSource, name: &str) -> Result<LinearHandles, String> {
    gather_checked(src, name)
}

fn gather_checked(src: &dyn TensorSource, name: &str) -> Result<LinearHandles, String> {
    let trellis_name = format!("{name}.trellis");
    let t = src
        .get(&trellis_name)
        .ok_or_else(|| format!("missing {trellis_name}"))?;
    if t.shape.len() != 3 {
        return Err(format!("{trellis_name}: shape {:?} is not 3-D", t.shape));
    }
    let (in_tiles, out_tiles, packed) = (t.shape[0], t.shape[1], t.shape[2]);
    if packed == 0 || packed % 16 != 0 {
        return Err(format!(
            "trellis packed size {packed} is not a multiple of 16"
        ));
    }
    let k = (packed / 16) as u32;
    let shape = Exl3Shape::new(in_tiles * 16, out_tiles * 16, k)
        .ok_or_else(|| format!("k={k} unsupported for {in_tiles}×{out_tiles} tiles"))?;
    if t.bytes.len() != in_tiles * out_tiles * packed * 2 {
        return Err(format!(
            "trellis byte length {} != {}×{}×{}×2",
            t.bytes.len(),
            in_tiles,
            out_tiles,
            packed
        ));
    }

    let suh_t = src
        .get(&format!("{name}.suh"))
        .ok_or_else(|| format!("missing {name}.suh"))?;
    let svh_t = src
        .get(&format!("{name}.svh"))
        .ok_or_else(|| format!("missing {name}.svh"))?;
    // Packed sign bitfields (I16) and missing scales stay on the CPU.
    if suh_t.dtype != DType::F16 || svh_t.dtype != DType::F16 {
        return Err(format!(
            "sign fields must be F16, got {:?}/{:?} (packed bitfield = CPU-only)",
            suh_t.dtype, svh_t.dtype
        ));
    }
    if suh_t.bytes.len() != shape.in_features() * 2 || svh_t.bytes.len() != shape.out_features() * 2
    {
        return Err(format!(
            "sign field lengths {}/{} != {}/{} halves",
            suh_t.bytes.len() / 2,
            svh_t.bytes.len() / 2,
            shape.in_features(),
            shape.out_features()
        ));
    }
    if src.get(&format!("{name}.bias")).is_some() {
        return Err("has a bias (CPU-only)".to_string());
    }

    let mcg_name = format!("{name}.mcg");
    let mcg = src
        .get(&mcg_name)
        .ok_or_else(|| format!("missing {mcg_name}"))?;
    let mult = mcg
        .first_u32(&mcg_name)
        .map_err(|e| format!("{mcg_name}: {e}"))?;
    let mode = CodebookMode::from_mcg_mult(mult)
        .ok_or_else(|| format!("{mcg_name}: mcg_mult {mult:#x} is not a known codebook"))?;
    let mode = match mode {
        CodebookMode::Default => 0u32,
        CodebookMode::Mcg => 1u32,
        CodebookMode::Mul1 => return Err("codebook Mul1 (CPU-only)".to_string()),
    };

    Ok(LinearHandles {
        name: name.to_string(),
        shape,
        mode,
        trellis: owned_span(src, &trellis_name)
            .ok_or_else(|| format!("{trellis_name}: no device span"))?,
        suh: owned_span(src, &format!("{name}.suh"))
            .ok_or_else(|| format!("{name}.suh: no device span"))?,
        svh: owned_span(src, &format!("{name}.svh"))
            .ok_or_else(|| format!("{name}.svh: no device span"))?,
    })
}

/// `(buffer, byte offset)` for a tensor, owned so it can outlive the borrow.
fn owned_span(src: &dyn TensorSource, name: &str) -> Option<(Buffer, usize)> {
    let (buffer, byte_offset) = src.device_span(name)?;
    Some((buffer.clone(), byte_offset))
}

// ---------------------------------------------------------------------------
// Encoder-level passthroughs (T4.2): these do not submit anything, they add
// passes to a caller-owned command buffer so a whole token is one submission.
// ---------------------------------------------------------------------------

impl MetalLinear {
    pub fn device(&self) -> &metal::Device {
        &self.device
    }

    /// A fresh command buffer from the shared queue (caller commits/waits).
    pub fn new_command_buffer(&self) -> metal::CommandBuffer {
        self.queue.new_command_buffer().to_owned()
    }

    /// Upload a small host f32 vector as a shared device buffer.
    pub fn upload_f32(&self, data: &[f32]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            (data.len() * 4).max(4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// One EXL3 linear, encoded into `enc` (`x`/`out` are whole device buffers
    /// covering `[rows, in/out_features]`; `rows == 1` is the decode GEMV,
    /// `rows >= 2` the batched prefill GEMM).
    pub fn encode_linear(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        lin: &LinearHandles,
        x: &Buffer,
        out: &Buffer,
        rows: usize,
    ) -> Result<(), MetalLinearError> {
        self.gemv.encode_linear(
            enc,
            x,
            lin.trellis(),
            lin.suh(),
            lin.svh(),
            out,
            lin.shape,
            lin.mode,
            rows,
        )?;
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_rms(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        x: DeviceSpan<'_>,
        weight: Option<DeviceSpan<'_>>,
        out: DeviceSpan<'_>,
        rows: usize,
        row_len: usize,
        eps: f32,
    ) -> Result<(), MetalLinearError> {
        self.fuse
            .encode_rms(enc, x, weight, out, rows, row_len, eps)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_rope(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        x: DeviceSpan<'_>,
        cos: DeviceSpan<'_>,
        sin: DeviceSpan<'_>,
        out: DeviceSpan<'_>,
        n_heads: usize,
        head_dim: usize,
        pos: usize,
        rows: usize,
    ) -> Result<(), MetalLinearError> {
        self.fuse
            .encode_rope(enc, x, cos, sin, out, n_heads, head_dim, pos, rows)?;
        Ok(())
    }

    pub fn encode_copy(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        src: DeviceSpan<'_>,
        dst: DeviceSpan<'_>,
        n: usize,
    ) -> Result<(), MetalLinearError> {
        self.fuse.encode_copy(enc, src, dst, n)?;
        Ok(())
    }

    pub fn encode_gelu_mul(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        gate: DeviceSpan<'_>,
        up: DeviceSpan<'_>,
        out: DeviceSpan<'_>,
        n: usize,
    ) -> Result<(), MetalLinearError> {
        self.fuse.encode_gelu_mul(enc, gate, up, out, n)?;
        Ok(())
    }

    pub fn encode_add(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        a: DeviceSpan<'_>,
        b: DeviceSpan<'_>,
        out: DeviceSpan<'_>,
        n: usize,
    ) -> Result<(), MetalLinearError> {
        self.fuse.encode_add(enc, a, b, out, n)?;
        Ok(())
    }

    pub fn encode_add_scale(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        a: DeviceSpan<'_>,
        b: DeviceSpan<'_>,
        out: DeviceSpan<'_>,
        n: usize,
        scalar: f32,
    ) -> Result<(), MetalLinearError> {
        self.fuse.encode_add_scale(enc, a, b, out, n, scalar)?;
        Ok(())
    }

    /// Attention into an existing encoder (the T4.2 pipeline).
    pub fn encode_attention(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        q: DeviceSpan<'_>,
        k: DeviceSpan<'_>,
        v: DeviceSpan<'_>,
        out: DeviceSpan<'_>,
        p: AttnParams,
    ) -> Result<(), MetalLinearError> {
        self.attn.run_in(enc, q, k, v, out, p)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exl3::codebook::CodebookMode as Mode;
    use crate::exl3::forward::{Exl3Linear, linear_forward_reconstruct};
    use crate::exl3::trellis::pack_trellis_tile;
    use crate::gemma4::RawTensor;
    use crate::metal_ctx::MetalContext;
    use std::collections::HashMap;

    /// Deterministic PRNG (no external dep) for synthetic payloads.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
    }

    /// Synthetic EXL3 linear served both as host bytes (CPU reference path) and
    /// as per-tensor Metal buffers (GPU path).
    struct SynthGpuSource {
        host: HashMap<String, Vec<u8>>,
        meta: HashMap<String, (DType, Vec<usize>)>,
        buffers: HashMap<String, Buffer>,
    }

    impl SynthGpuSource {
        fn build(device: &metal::Device, k: u32, in_f: usize, out_f: usize) -> Self {
            let mut rng = Rng(0xC0FFEE_2026);
            let (in_tiles, out_tiles) = (in_f / 16, out_f / 16);
            let words = 16 * k as usize;
            let mut host: HashMap<String, Vec<u8>> = HashMap::new();
            let mut meta: HashMap<String, (DType, Vec<usize>)> = HashMap::new();

            // Packed trellis: random K-bit payloads per tile, artifact byte order.
            let mut trellis = Vec::new();
            let mask = (1u32 << k) - 1;
            for _ in 0..in_tiles * out_tiles {
                let mut enc = [0u16; 256];
                for e in enc.iter_mut() {
                    *e = (rng.next() & mask) as u16;
                }
                for w in pack_trellis_tile(&enc, k).unwrap() {
                    trellis.extend_from_slice(&w.to_le_bytes());
                }
            }
            host.insert("l.trellis".into(), trellis);
            meta.insert(
                "l.trellis".into(),
                (DType::I16, vec![in_tiles, out_tiles, words]),
            );

            // suh/svh in a magnitude range that keeps the f32 Hadamard well behaved.
            for (name, n) in [("l.suh", in_f), ("l.svh", out_f)] {
                let mut bytes = Vec::with_capacity(n * 2);
                for _ in 0..n {
                    let v = ((rng.next() % 2001) as f32 / 1000.0) - 1.0;
                    bytes.extend_from_slice(&f16::from_f32(v).to_le_bytes());
                }
                host.insert(name.into(), bytes);
                meta.insert(name.into(), (DType::F16, vec![n]));
            }

            host.insert("l.mcg".into(), Mode::Mcg.mcg_mult().to_le_bytes().to_vec());
            meta.insert("l.mcg".into(), (DType::I32, vec![1]));

            let buffers = host
                .iter()
                .map(|(name, bytes)| {
                    let buf = device.new_buffer_with_data(
                        bytes.as_ptr() as *const _,
                        bytes.len().max(1) as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                    );
                    (name.clone(), buf)
                })
                .collect();

            Self {
                host,
                meta,
                buffers,
            }
        }

        fn linear(&self) -> Exl3Linear<'_> {
            let trellis_words: &[u16] = unsafe {
                std::slice::from_raw_parts(
                    self.host["l.trellis"].as_ptr() as *const u16,
                    self.host["l.trellis"].len() / 2,
                )
            };
            let shape = &self.meta["l.trellis"].1;
            Exl3Linear {
                trellis: crate::exl3::reconstruct::TrellisView::new(
                    trellis_words,
                    shape[0],
                    shape[1],
                    shape[2],
                )
                .unwrap(),
                suh: Some(&self.host["l.suh"]),
                svh: Some(&self.host["l.svh"]),
                suh_is_f16: true,
                svh_is_f16: true,
                mode: Mode::Mcg,
                bias: None,
            }
        }
    }

    impl TensorSource for SynthGpuSource {
        fn get(&self, name: &str) -> Option<RawTensor<'_>> {
            let (dtype, shape) = self.meta.get(name)?;
            Some(RawTensor {
                bytes: &self.host[name],
                dtype: *dtype,
                shape,
            })
        }

        fn device_span(&self, name: &str) -> Option<(&Buffer, usize)> {
            Some((self.buffers.get(name)?, 0))
        }
    }

    fn ctx() -> Option<MetalContext> {
        match MetalContext::new() {
            Ok(c) => Some(c),
            Err(_) => {
                eprintln!("skip: no Metal device");
                None
            }
        }
    }

    fn max_rel_err(a: &[f16], b: &[f16]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let (x, y) = (x.to_f32(), y.to_f32());
                let scale = 1.0f32.max(x.abs()).max(y.abs());
                (x - y).abs() / scale
            })
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn gather_accepts_eligible_synthetic_linear() {
        let Some(c) = ctx() else { return };
        let src = SynthGpuSource::build(c.device(), 3, 128, 128);
        let g = linear_handles(&src, "l").ok().expect("eligible");
        assert_eq!(g.shape.in_features(), 128);
        assert_eq!(g.shape.out_features(), 128);
        assert_eq!(g.mode, 1, "Mcg");
    }

    #[test]
    fn gather_rejects_ineligible_shapes_and_formats() {
        let Some(c) = ctx() else { return };
        // packed words not 16*k
        let mut src = SynthGpuSource::build(c.device(), 3, 128, 128);
        src.meta.get_mut("l.trellis").unwrap().1[2] = 47;
        assert!(linear_handles(&src, "l").ok().is_none());

        // out_features not a multiple of 128
        let mut src = SynthGpuSource::build(c.device(), 3, 128, 128);
        src.meta.get_mut("l.trellis").unwrap().1[1] = 7;
        assert!(linear_handles(&src, "l").ok().is_none());

        // packed sign bitfield scales stay on the CPU
        let mut src = SynthGpuSource::build(c.device(), 3, 128, 128);
        src.meta.get_mut("l.suh").unwrap().0 = DType::I16;
        assert!(linear_handles(&src, "l").ok().is_none());

        // bias is unsupported on the GPU path
        let mut src = SynthGpuSource::build(c.device(), 3, 128, 128);
        let bias = vec![0u8; 256];
        src.buffers.insert(
            "l.bias".into(),
            c.device().new_buffer_with_data(
                bias.as_ptr() as *const _,
                bias.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ),
        );
        src.host.insert("l.bias".into(), bias);
        src.meta.insert("l.bias".into(), (DType::F16, vec![128]));
        assert!(linear_handles(&src, "l").ok().is_none());

        // unknown codebook multiplier
        let mut src = SynthGpuSource::build(c.device(), 3, 128, 128);
        src.host
            .insert("l.mcg".into(), 0xDEAD_BEEFu32.to_le_bytes().to_vec());
        assert!(linear_handles(&src, "l").ok().is_none());

        // missing device storage
        let mut src = SynthGpuSource::build(c.device(), 3, 128, 128);
        src.buffers.remove("l.svh");
        assert!(linear_handles(&src, "l").ok().is_none());
    }

    /// `had_in` matches the CPU reference within 1 f16 ulp. The transform is
    /// mathematically identical (the butterfly computes the same symmetric
    /// Sylvester matrix the CPU builds explicitly); only the f32 summation order
    /// differs, so a value that lands exactly on an f16 rounding boundary may
    /// round the other way.
    #[test]
    fn gpu_had_in_matches_cpu_reference() {
        let Some(c) = ctx() else { return };
        let src = SynthGpuSource::build(c.device(), 3, 256, 128);
        let Some((buf, off)) = src.device_span("l.suh") else {
            panic!("suh span")
        };
        let suh: Vec<f16> = src.host["l.suh"]
            .chunks_exact(2)
            .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])))
            .collect();
        let mut rng = Rng(0x1111);
        let x: Vec<f16> = (0..256)
            .map(|_| f16::from_f32(((rng.next() % 2001) as f32 / 1000.0) - 1.0))
            .collect();

        let want = crate::exl3::hadamard::had_r_128(&x, 1, 256, Some(&suh), None);
        let mut got = Vec::new();
        MetalLinear::new(&c)
            .unwrap()
            .raw()
            .had_in_pass(
                &x,
                DeviceSpan {
                    buffer: buf,
                    byte_offset: off,
                },
                &mut got,
            )
            .expect("had_in");
        assert_eq!(want.len(), got.len());
        for (i, (w, g)) in want.iter().zip(got.iter()).enumerate() {
            let d = (w.to_bits() as i32 - g.to_bits() as i32).abs();
            assert!(
                d <= 1,
                "had_in[{i}] want {:#06x} got {:#06x} (delta {d} ulp)",
                w.to_bits(),
                g.to_bits()
            );
        }
    }

    /// `had_out` matches the CPU reference within 1 f16 ulp (same reason).
    #[test]
    fn gpu_had_out_matches_cpu_reference() {
        let Some(c) = ctx() else { return };
        let src = SynthGpuSource::build(c.device(), 3, 128, 256);
        let Some((buf, off)) = src.device_span("l.svh") else {
            panic!("svh span")
        };
        let svh: Vec<f16> = src.host["l.svh"]
            .chunks_exact(2)
            .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])))
            .collect();
        let mut rng = Rng(0x2222);
        let y: Vec<f32> = (0..256)
            .map(|_| ((rng.next() % 2001) as f32 / 1000.0) - 1.0)
            .collect();
        let y16: Vec<f16> = y.iter().map(|v| f16::from_f32(*v)).collect();

        let want = crate::exl3::hadamard::had_r_128(&y16, 1, 256, None, Some(&svh));
        let mut got = Vec::new();
        MetalLinear::new(&c)
            .unwrap()
            .raw()
            .had_out_pass(
                &y,
                DeviceSpan {
                    buffer: buf,
                    byte_offset: off,
                },
                &mut got,
            )
            .expect("had_out");

        // f16 inputs/outputs are identical (Metal's f32→f16 is RNE, proven by
        // the bit-exact had_in test). The remaining spread is single f32
        // rounding in the 128-term sum, where the GPU compiler contracts
        // `acc += sh[j] * sign` into an FMA and the CPU reference does not — so
        // the two differ by at most 1 ulp of f16, never more.
        let mut worst = 0i32;
        for (i, (w, g)) in want.iter().zip(got.iter()).enumerate() {
            let d = (w.to_bits() as i32 - g.to_bits() as i32).abs();
            assert!(
                d <= 1,
                "had_out[{i}] want {:#06x} got {:#06x} (delta {d} ulp)",
                w.to_bits(),
                g.to_bits()
            );
            worst = worst.max(d);
        }
        assert!(worst <= 1);
    }

    /// The GEMV alone, against the CPU decode of the very same tile: one 16×16
    /// tile per K, so the perm/funnel/pair-table path is exercised end to end.
    #[test]
    fn gpu_gemv_matches_cpu_decoded_tile() {
        let Some(c) = ctx() else { return };
        for k in 1u32..=8 {
            let src = SynthGpuSource::build(c.device(), k, 16, 16);
            let layer = src.linear();
            let w = layer.trellis.decode_tile(0, 0, Mode::Mcg).unwrap();
            let mut rng = Rng(0x3333_0000u64 + k as u64);
            let xh: Vec<f16> = (0..16)
                .map(|_| f16::from_f32(((rng.next() % 2001) as f32 / 1000.0) - 1.0))
                .collect();
            let mut want = [0f32; 16];
            for (r, xr) in xh.iter().enumerate() {
                for (c_idx, wc) in want.iter_mut().enumerate() {
                    *wc += xr.to_f32() * w[r][c_idx].to_f32();
                }
            }
            let Some((buf, off)) = src.device_span("l.trellis") else {
                panic!("trellis span")
            };
            let shape = Exl3Shape::new(16, 16, k).unwrap();
            let mut got = Vec::new();
            MetalLinear::new(&c)
                .unwrap()
                .raw()
                .gemv_pass(
                    &xh,
                    DeviceSpan {
                        buffer: buf,
                        byte_offset: off,
                    },
                    shape,
                    1,
                    &mut got,
                )
                .expect("gemv");
            for (i, (w, g)) in want.iter().zip(got.iter()).enumerate() {
                let scale = 1.0f32.max(w.abs()).max(g.abs());
                assert!(
                    (w - g).abs() / scale < 1e-3,
                    "k={k} gemv[{i}] want {w} got {g}"
                );
            }
        }
    }

    /// Host-only model of the MSL GEMV's index math: the pair table + funnel
    /// shift must reproduce `unpack_trellis_tile` for every K, and the group →
    /// (row, column) mapping must reproduce `tensor_core_perm`. If this passes
    /// and the GPU still disagrees, the bug is in the kernel, not the model.
    #[test]
    fn pair_table_model_reproduces_reference_unpack() {
        use crate::exl3::perm::tensor_core_perm;
        use crate::exl3::trellis::{pack_trellis_tile, unpack_trellis_tile};
        let perm = tensor_core_perm();
        for k in 1u32..=8 {
            let mask = (1u32 << k) - 1;
            let mut fresh = [0u16; 256];
            let mut state = 0x1234_5678u32 ^ k;
            for f in fresh.iter_mut() {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *f = (state >> 16) as u16 & mask as u16;
            }
            let packed = pack_trellis_tile(&fresh, k).unwrap();
            let want = unpack_trellis_tile(&packed, k).unwrap();
            let words: Vec<u32> = packed
                .chunks_exact(2)
                .map(|p| p[0] as u32 | ((p[1] as u32) << 16))
                .collect();
            let table = kaplumba_kernels::exl3::pair_table(k);

            let mut got = [0u16; 256];
            let mut cols = [(0usize, 0usize); 256];
            for g in 0..32usize {
                for q in 0..4usize {
                    let e = table[4 * g + q];
                    let (i0, i1, s1) = (
                        (e & 0xFF) as usize,
                        ((e >> 8) & 0xFF) as usize,
                        (e >> 16) & 0xFF,
                    );
                    let merged = ((words[i0] as u64) << 32) | words[i1] as u64;
                    let w1 = (merged >> s1) as u32;
                    got[8 * g + 2 * q] = ((w1 >> k) & 0xFFFF) as u16;
                    got[8 * g + 2 * q + 1] = (w1 & 0xFFFF) as u16;
                    // kernel index -> row-major position (what the kernel assumes)
                    let p0 = perm[8 * g + 2 * q] as usize;
                    cols[8 * g + 2 * q] = (p0 / 16, p0 % 16);
                    let p1 = perm[8 * g + 2 * q + 1] as usize;
                    cols[8 * g + 2 * q + 1] = (p1 / 16, p1 % 16);
                }
            }
            for i in 0..256 {
                assert_eq!(got[i], want[i], "k={k} kernel idx {i}");
            }
            // The kernel's hard-coded assumption: pair q of group g feeds rows
            // (r0, r0+1) for q even and (r0+8, r0+9) for q odd, all in column
            // g/4 (q<2) or g/4+8 (q>=2).
            for g in 0..32usize {
                let r0 = (g % 4) * 2;
                for q in 0..4usize {
                    let col = if q < 2 { g / 4 } else { g / 4 + 8 };
                    let (lr, lc) = cols[8 * g + 2 * q];
                    assert_eq!(lc, col, "low col g={g} q={q}");
                    assert_eq!(
                        lr,
                        if q % 2 == 0 { r0 } else { r0 + 8 },
                        "low row g={g} q={q}"
                    );
                    let (hr, hc) = cols[8 * g + 2 * q + 1];
                    assert_eq!(hc, col, "high col g={g} q={q}");
                    assert_eq!(
                        hr,
                        if q % 2 == 0 { r0 + 1 } else { r0 + 9 },
                        "high row g={g} q={q}"
                    );
                }
            }
        }
    }

    /// Per-pass timings on the **real artifact shapes** (T2.3 groundwork).
    /// Gate: `KAPLUMBA_RUN_SLOW=1`; run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_passes_slow() {
        if std::env::var("KAPLUMBA_RUN_SLOW").is_err() {
            eprintln!("skip: set KAPLUMBA_RUN_SLOW=1");
            return;
        }
        let Some(dir) = crate::testutil::artifact_dir() else {
            eprintln!("skip: no artifact");
            return;
        };
        let Some(c) = ctx() else { return };
        let model = crate::load::load_exl3(&c, &dir).expect("load artifact");
        let ml = MetalLinear::new(&c).expect("metal linear");
        let src: &dyn TensorSource = &model;

        let names = [
            ("gate", "model.language_model.layers.0.mlp.gate_proj"),
            ("down", "model.language_model.layers.0.mlp.down_proj"),
            ("o", "model.language_model.layers.0.self_attn.o_proj"),
            ("q", "model.language_model.layers.0.self_attn.q_proj"),
            ("k", "model.language_model.layers.0.self_attn.k_proj"),
            ("v", "model.language_model.layers.0.self_attn.v_proj"),
            ("lm_head", "lm_head"),
        ];
        for (label, name) in names {
            let g = linear_handles(src, name).unwrap_or_else(|e| panic!("{name} eligible: {e}"));
            let (inn, outp) = (g.shape.in_features(), g.shape.out_features());
            let x: Vec<f16> = (0..inn)
                .map(|i| f16::from_f32(((i % 17) as f32 - 8.0) / 64.0))
                .collect();
            const REPS: usize = 20;
            let verbose = std::env::var("KAPLUMBA_BENCH_VERBOSE").is_ok();
            let time = |tag: &str, f: &mut dyn FnMut()| -> (f64, f64) {
                let mut v = Vec::with_capacity(REPS);
                for _ in 0..REPS {
                    let t = std::time::Instant::now();
                    f();
                    v.push(t.elapsed().as_secs_f64() * 1000.0);
                }
                if verbose {
                    eprintln!(
                        "    {tag:8} reps ms: {:?}",
                        v.iter()
                            .map(|x| (x * 100.0).round() / 100.0)
                            .collect::<Vec<_>>()
                    );
                }
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                (v[0], v[REPS / 2])
            };

            let (hi_min, hi_med) = time("had_in", &mut || {
                let mut o = Vec::new();
                ml.raw().had_in_pass(&x, g.suh(), &mut o).unwrap();
            });
            let y: Vec<f32> = (0..outp).map(|i| (i % 13) as f32 * 0.01).collect();
            let (ho_min, ho_med) = time("had_out", &mut || {
                let mut o = Vec::new();
                ml.raw().had_out_pass(&y, g.svh(), &mut o).unwrap();
            });
            let mut xh = vec![f16::ZERO; inn];
            let (gv_min, gv_med) = time("gemv", &mut || {
                let mut o = Vec::new();
                ml.raw()
                    .gemv_pass(&xh, g.trellis(), g.shape, g.mode, &mut o)
                    .unwrap();
            });
            xh.clear();
            let (fu_min, fu_med) = time("full", &mut || {
                let _ = ml.try_linear(src, name, &x).unwrap().unwrap();
            });

            let weights = inn * outp;
            eprintln!(
                "{label:8} {inn}x{outp} k={} off(t{}/s{}/{}) · full {fu_min:7.3}/{fu_med:7.3} · gemv {gv_min:6.3}/{gv_med:6.3} · had_in {hi_min:6.3}/{hi_med:6.3} · had_out {ho_min:6.3}/{ho_med:6.3} ms · {:.0} Gw/s",
                g.shape.k,
                g.trellis().byte_offset % 4,
                g.suh().byte_offset % 4,
                g.svh().byte_offset % 4,
                weights as f64 / (fu_min / 1000.0) / 1e9,
            );
        }
    }

    /// Diagnostic dump of the matrix the GPU sees, one-hot activation per row.
    /// Ignored by default; `cargo test -- --ignored --nocapture debug_gemv_rows`.
    #[test]
    #[ignore]
    fn debug_gemv_rows() {
        let Some(c) = ctx() else { return };
        let k = 1u32;
        let src = SynthGpuSource::build(c.device(), k, 16, 16);
        let layer = src.linear();
        let w = layer.trellis.decode_tile(0, 0, Mode::Mcg).unwrap();
        eprintln!("decoded W (row-major):");
        for r in 0..16 {
            eprintln!(
                "  {:?}",
                w[r].iter().map(|v| v.to_f32()).collect::<Vec<_>>()
            );
        }
        eprintln!(
            "pair_tab k=1 [..4] = {:?}",
            &kaplumba_kernels::exl3::pair_table(1)[..4]
        );
        // One-hot activations reconstruct the matrix the GPU actually sees.
        let (buf, off) = src.device_span("l.trellis").unwrap();
        let ml = MetalLinear::new(&c).unwrap();
        let shape = Exl3Shape::new(16, 16, k).unwrap();
        eprintln!("gpu W (rows) vs cpu W (rows):");
        for r in 0..16 {
            let mut xh = vec![f16::ZERO; 16];
            xh[r] = f16::ONE;
            let mut got = Vec::new();
            ml.raw()
                .gemv_pass(
                    &xh,
                    DeviceSpan {
                        buffer: buf,
                        byte_offset: off,
                    },
                    shape,
                    1,
                    &mut got,
                )
                .unwrap();
            let cpu: Vec<f32> = w[r].iter().map(|v| v.to_f32()).collect();
            eprintln!(
                "  r{r:02} gpu {:?}",
                got.iter()
                    .map(|v| (*v * 1000.0).round() / 1000.0)
                    .collect::<Vec<_>>()
            );
            eprintln!(
                "  r{r:02} cpu {:?}",
                cpu.iter()
                    .map(|v| (*v * 1000.0).round() / 1000.0)
                    .collect::<Vec<_>>()
            );
        }
    }

    /// The GPU linear must reproduce the CPU reference within f32
    /// accumulation-order tolerance (the Hadamard passes themselves are
    /// bit-exact, so the only divergence is the GEMV reduction order).
    #[test]
    fn gpu_linear_matches_cpu_reference() {
        let Some(c) = ctx() else { return };
        for (k, in_f, out_f) in [
            (3usize as u32, 128usize, 128usize),
            (6, 256, 128),
            (3, 128, 256),
        ] {
            let src = SynthGpuSource::build(c.device(), k, in_f, out_f);
            let layer = src.linear();
            let mut rng = Rng(0xABCD_0001);
            let x: Vec<f16> = (0..in_f)
                .map(|_| f16::from_f32(((rng.next() % 2001) as f32 / 1000.0) - 1.0))
                .collect();

            let want = linear_forward_reconstruct(&layer, &x);
            let gpu = MetalLinear::new(&c)
                .unwrap()
                .try_linear(&src, "l", &x)
                .unwrap()
                .expect("gpu path");
            assert_eq!(gpu.len(), out_f);
            let err = max_rel_err(&want, &gpu);
            assert!(err < 5e-3, "k={k} in={in_f} out={out_f} rel err {err}");
        }
    }

    /// Two rows must not silently take the GPU path (decode shape only).
    #[test]
    fn try_linear_falls_back_for_multi_row() {
        let Some(c) = ctx() else { return };
        let src = SynthGpuSource::build(c.device(), 3, 128, 128);
        let x = vec![f16::ZERO; 256];
        let got = MetalLinear::new(&c)
            .unwrap()
            .try_linear(&src, "l", &x)
            .unwrap();
        assert!(got.is_none(), "multi-row input must use the CPU path");
    }

    /// T3.2: the batched GEMM (`encode_linear`, rows ≥ 2) must match both the
    /// CPU reference and M single-row GEMV dispatches on the same weights.
    /// rows=17 > RGT(8) exercises the partial third row-chunk (scalar path);
    /// all three shapes are sg-eligible, so this also covers the simdgroup
    /// tensor-core kernel end to end.
    ///
    /// The batched-vs-GEMV comparison used to assert bit-identity (the scalar
    /// RGT GEMM accumulates each row in exactly the GEMV order). The
    /// simdgroup kernel multiplies in f16 8×8 tiles across k-blocks, so the
    /// f32 sums differ by reassociation noise (~1e-6…1e-5) and the f16
    /// rounding in `had_out` may flip values that sit within that of a
    /// rounding boundary. Products themselves are exact (f16×f16 fits f32),
    /// so the two paths must agree to within an f16 ulp: `max_rel_err < 2e-3`
    /// (divisor ≥ 1, so this is ≤ ~2 ulp for |v| ≤ 1) — tight enough that a
    /// wrong decode/offset (error ~1e-1) still fails loudly.
    #[test]
    fn gpu_gemm_rows_match_per_row_gemv_and_cpu() {
        let Some(c) = ctx() else { return };
        for (k, in_f, out_f, rows) in [
            (3u32, 128usize, 128usize, 5usize),
            (6, 256, 128, 3),
            (3, 128, 256, 17),
        ] {
            let src = SynthGpuSource::build(c.device(), k, in_f, out_f);
            let handles = linear_handles(&src, "l").expect("eligible");
            let layer = src.linear();
            let mut rng = Rng(0x6E_2026);
            let x: Vec<f16> = (0..rows * in_f)
                .map(|_| f16::from_f32(((rng.next() % 2001) as f32 / 1000.0) - 1.0))
                .collect();
            let ml = MetalLinear::new(&c).unwrap();
            let f16_bytes = |v: &[f16]| unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2)
            };
            let new_buf = |bytes: &[u8]| {
                c.device().new_buffer_with_data(
                    bytes.as_ptr() as *const _,
                    bytes.len().max(1) as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                )
            };

            // One batched dispatch over all rows.
            let xb = new_buf(f16_bytes(&x));
            let yb = c.device().new_buffer(
                (rows * out_f * 2) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let cb = ml.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            ml.encode_linear(enc, &handles, &xb, &yb, rows)
                .expect("batched encode_linear");
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let batched: Vec<f16> =
                unsafe { std::slice::from_raw_parts(yb.contents() as *const f16, rows * out_f) }
                    .to_vec();

            for r in 0..rows {
                let row_x = &x[r * in_f..(r + 1) * in_f];
                let want_cpu = linear_forward_reconstruct(&layer, row_x);
                let got = &batched[r * out_f..(r + 1) * out_f];
                let err = max_rel_err(&want_cpu, got);
                assert!(
                    err < 5e-3,
                    "k={k} {rows}x{in_f}x{out_f} row {r} vs CPU rel err {err}"
                );

                // Same row through the single-row GEMV (rows=1).
                let xb1 = new_buf(f16_bytes(row_x));
                let yb1 = c.device().new_buffer(
                    (out_f * 2) as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                let cb1 = ml.new_command_buffer();
                let enc1 = cb1.new_compute_command_encoder();
                ml.encode_linear(enc1, &handles, &xb1, &yb1, 1)
                    .expect("row encode_linear");
                enc1.end_encoding();
                cb1.commit();
                cb1.wait_until_completed();
                let gemv: Vec<f16> =
                    unsafe { std::slice::from_raw_parts(yb1.contents() as *const f16, out_f) }
                        .to_vec();
                let err = max_rel_err(got, &gemv);
                assert!(
                    err < 2e-3,
                    "k={k} {rows}x{in_f}x{out_f} row {r}: batched GEMM vs GEMV rel err {err}"
                );
            }
        }
    }

    /// T3.2 follow-up: where does span prefill time actually go for one linear?
    /// Synthetic weights, real shapes (hidden→q/k/v/o, gate/up, down). Prints
    /// `had_in` / `gem` / `had_out` milliseconds so the next kernel change
    /// targets the dominant phase. Not a correctness test.
    #[test]
    fn phase_profile_prefill_shapes() {
        let Some(c) = ctx() else { return };
        // (name, k, in, out, rows) — Gemma 4 31B linears at batch_rows=256.
        let shapes = [
            ("q_proj", 3u32, 5376usize, 8192usize, 256usize),
            ("k_proj", 3, 5376, 4096, 256),
            ("o_proj", 3, 8192, 5376, 256),
            ("gate", 3, 5376, 21504, 256),
            ("down", 3, 21504, 5376, 256),
            ("gate@512", 3, 5376, 21504, 512),
        ];
        eprintln!("=== phase profile (ms per linear, release-friendly synthetic) ===");
        for (name, k, in_f, out_f, rows) in shapes {
            eprintln!("-- {name}: {rows}x{in_f}x{out_f} k={k}");
            let src = SynthGpuSource::build(c.device(), k, in_f, out_f);
            let handles = linear_handles(&src, "l").expect("eligible");
            let ml = MetalLinear::new(&c).unwrap();
            let mut rng = Rng(0x6E_2026);
            let x: Vec<f16> = (0..rows * in_f)
                .map(|_| f16::from_f32(((rng.next() % 2001) as f32 / 1000.0) - 1.0))
                .collect();
            let xb = c.device().new_buffer_with_data(
                x.as_ptr() as *const _,
                (rows * in_f * 2) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let yb = c.device().new_buffer(
                (rows * out_f * 2) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let (hi, g, ho) = ml
                .raw()
                .bench_linear_phases(
                    &xb,
                    handles.trellis(),
                    handles.suh(),
                    handles.svh(),
                    &yb,
                    handles.shape,
                    handles.mode,
                    rows,
                    5,
                )
                .expect("phase bench");
            // Per-token cost if this linear runs once per 256-row chunk.
            let chunk_ms = hi + g + ho;
            let tokens_per_s_if_only_this = rows as f64 / (chunk_ms / 1e3);
            eprintln!(
                "  => {chunk_ms:.2} ms/chunk ≈ {tokens_per_s_if_only_this:.1} t/s if this were the whole model"
            );
        }
    }

    /// T3.6: attribute one 256-row prefill chunk's wall time to linears /
    /// attention / glue on the real artifact (both layer kinds), then compare
    /// the projected budget against a measured real 256-token span forward.
    /// Every sample is encode + commit + wait on an idle queue, so it is real
    /// GPU time. Not a correctness test.
    #[test]
    fn phase_profile_chunk_budget_slow() {
        if std::env::var("KAPLUMBA_RUN_SLOW").is_err() {
            eprintln!("skip: set KAPLUMBA_RUN_SLOW=1");
            return;
        }
        let Some(dir) = crate::testutil::artifact_dir() else {
            eprintln!("skip: no artifact");
            return;
        };
        let Some(c) = ctx() else { return };
        let model = crate::load::load_exl3(&c, &dir).expect("load artifact");
        let ml = MetalLinear::new(&c).expect("metal linear");
        let src: &dyn TensorSource = &model;
        let rows = 256usize;
        let dev = c.device();
        let mk = |elems: usize| {
            dev.new_buffer(
                (elems * 2) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            )
        };
        let mkf32 = |elems: usize| {
            dev.new_buffer(
                (elems * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            )
        };
        fn sp(b: &metal::Buffer) -> DeviceSpan<'_> {
            DeviceSpan {
                buffer: b,
                byte_offset: 0,
            }
        }
        let time_ms = |f: &dyn Fn(&metal::ComputeCommandEncoderRef)| -> f64 {
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t0 = std::time::Instant::now();
                let cb = ml.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                f(enc);
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                let dt = t0.elapsed().as_secs_f64() * 1e3;
                if dt < best {
                    best = dt;
                }
            }
            best
        };

        // ---- linears (real weights, sg path, encode_linear = had+gem+had) ----
        let linear_ms = |name: &str| -> f64 {
            let handles = linear_handles(src, name).unwrap_or_else(|e| panic!("{name}: {e}"));
            let inn = handles.shape.in_features();
            let out_f = handles.shape.out_features();
            let elig = handles.shape.in_tiles % 2 == 0
                && handles.shape.out_tiles % 4 == 0
                && handles.trellis().byte_offset % 4 == 0;
            eprintln!(
                "    {name}: in_tiles={} out_tiles={} k={} off%4={} sg_eligible={elig}",
                handles.shape.in_tiles,
                handles.shape.out_tiles,
                handles.shape.k,
                handles.trellis().byte_offset % 4,
            );
            let xb = mk(rows * inn);
            let yb = mk(rows * out_f);
            time_ms(&|enc: &metal::ComputeCommandEncoderRef| {
                ml.encode_linear(enc, &handles, &xb, &yb, rows)
                    .expect("encode_linear")
            })
        };
        let prefix = |i: usize, m: &str| format!("model.language_model.layers.{i}.{m}");
        let l0_names = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ];
        let l5_names = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ];
        eprintln!("=== chunk budget profile (rows={rows}, real artifact) ===");
        eprintln!("-- linears (ms, one call each, includes had_in/out)");
        let mut l0_sum = 0.0;
        for m in l0_names {
            let ms = linear_ms(&prefix(0, m));
            l0_sum += ms;
            eprintln!("  L0 sliding {m:24} {ms:8.2}");
        }
        let mut l5_sum = 0.0;
        for m in l5_names {
            let ms = linear_ms(&prefix(5, m));
            l5_sum += ms;
            eprintln!("  L5 full    {m:24} {ms:8.2}");
        }
        eprintln!("  sliding layer sum {l0_sum:.2} ms | full layer sum {l5_sum:.2} ms");

        // lm_head once per prompt (rows=1 gemv).
        let lm_handles = linear_handles(src, "lm_head").expect("lm_head");
        let lm_x = mk(lm_handles.shape.in_features());
        let lm_y = mk(lm_handles.shape.out_features());
        let lm_head_ms = time_ms(&|enc: &metal::ComputeCommandEncoderRef| {
            ml.encode_linear(enc, &lm_handles, &lm_x, &lm_y, 1)
                .expect("lm_head")
        });

        // ---- attention ----
        let attn_ms = |s: usize,
                       kv_rows: usize,
                       q_pos0: usize,
                       kv_pos0: usize,
                       n_heads: usize,
                       n_kv: usize,
                       hd: usize,
                       window: usize|
         -> f64 {
            let q = mk(s * n_heads * hd);
            let kv = mk(kv_rows * n_kv * hd);
            let o = mk(s * n_heads * hd);
            time_ms(&|enc: &metal::ComputeCommandEncoderRef| {
                ml.encode_attention(
                    enc,
                    sp(&q),
                    sp(&kv),
                    sp(&kv),
                    sp(&o),
                    AttnParams {
                        s,
                        kv_rows,
                        q_pos0,
                        kv_pos0,
                        n_heads,
                        n_kv,
                        head_dim: hd,
                        window,
                    },
                )
                .expect("encode_attention")
            })
        };
        eprintln!("-- attention (ms per dispatch)");
        let sl0 = attn_ms(256, 256, 0, 0, 32, 16, 256, 1024);
        let sl1 = attn_ms(256, 512, 256, 0, 32, 16, 256, 1024);
        let sl2 = attn_ms(256, 1024, 7936, 6913, 32, 16, 256, 1024);
        let fl0 = attn_ms(256, 256, 0, 0, 32, 4, 512, 0);
        let fl1 = attn_ms(256, 512, 256, 0, 32, 4, 512, 0);
        let fl4 = attn_ms(256, 4096, 3840, 0, 32, 4, 512, 0);
        let fl8 = attn_ms(256, 8192, 7936, 0, 32, 4, 512, 0);
        eprintln!("  sliding kv=256/512/1024: {sl0:.2} / {sl1:.2} / {sl2:.2}");
        eprintln!("  full    kv=256/512/4096/8192: {fl0:.2} / {fl1:.2} / {fl4:.2} / {fl8:.2}");

        // ---- glue (norms/rope/copy/add/gelu) ----
        let gm = crate::gemma4::Gemma4Model::from_model(&model)
            .expect("cfg")
            .with_gpu(&ml);
        let eps = gm.cfg.eps as f32;
        let xh = [mk(rows * 5376), mk(rows * 5376), mk(rows * 5376)];
        let xn = mk(rows * 5376);
        let o_norm = mk(rows * 5376);
        let gu_gate = mk(rows * 21504);
        let gu_up = mk(rows * 21504);
        let gu = mk(rows * 21504);
        let w_hidden = mkf32(5376);
        let w256 = mkf32(256);
        let w512 = mkf32(512);
        eprintln!("-- glue (ms per dispatch)");

        let rope_glue = |hd: usize, n_heads_q: usize, n_kv: usize, has_v: bool| -> (f64, f64) {
            // returns (hidden-glue ms for this layer kind, head-glue ms)
            let hh = hd / 2;
            let ctab = mkf32((8192 + 512) * hh);
            let stab = mkf32((8192 + 512) * hh);
            let q_raw = mk(rows * n_heads_q * hd);
            let q = mk(rows * n_heads_q * hd);
            let k_raw = mk(rows * n_kv * hd);
            let k_norm = mk(rows * n_kv * hd);
            let v_raw = mk(rows * n_kv * hd);
            let vcache = mk(rows * n_kv * hd);
            let w_qk: &metal::Buffer = if hd == 256 { &w256 } else { &w512 };

            let mut hidden = 0.0;
            // 4 hidden-sized rms per layer.
            for (xin, w) in [
                (&xh[0], Some(&w_hidden)),
                (&o_norm, Some(&w_hidden)),
                (&xh[1], Some(&w_hidden)),
                (&xh[2], Some(&w_hidden)),
            ] {
                hidden += time_ms(&|enc: &metal::ComputeCommandEncoderRef| {
                    ml.encode_rms(enc, sp(xin), w.map(|b| sp(b)), sp(&xn), rows, 5376, eps)
                        .expect("rms hidden")
                });
            }
            // residual add + gelu_mul + add_scale.
            hidden += time_ms(&|enc: &metal::ComputeCommandEncoderRef| {
                ml.encode_add(enc, sp(&xh[0]), sp(&o_norm), sp(&xh[2]), rows * 5376)
                    .expect("add")
            });
            hidden += time_ms(&|enc: &metal::ComputeCommandEncoderRef| {
                ml.encode_gelu_mul(enc, sp(&gu_gate), sp(&gu_up), sp(&gu), rows * 21504)
                    .expect("gelu_mul")
            });
            hidden += time_ms(&|enc: &metal::ComputeCommandEncoderRef| {
                ml.encode_add_scale(enc, sp(&xh[1]), sp(&xn), sp(&xh[0]), rows * 5376, 0.5)
                    .expect("add_scale")
            });

            let mut head = 0.0;
            head += time_ms(&|enc: &metal::ComputeCommandEncoderRef| {
                ml.encode_rms(
                    enc,
                    sp(&q_raw),
                    Some(sp(w_qk)),
                    sp(&q),
                    rows * n_heads_q,
                    hd,
                    eps,
                )
                .expect("rms q_norm")
            });
            head += time_ms(&|enc: &metal::ComputeCommandEncoderRef| {
                ml.encode_rms(
                    enc,
                    sp(&k_raw),
                    Some(sp(w_qk)),
                    sp(&k_norm),
                    rows * n_kv,
                    hd,
                    eps,
                )
                .expect("rms k_norm")
            });
            head += time_ms(&|enc: &metal::ComputeCommandEncoderRef| {
                ml.encode_rope(
                    enc,
                    sp(&q),
                    sp(&ctab),
                    sp(&stab),
                    sp(&q_raw),
                    n_heads_q,
                    hd,
                    7936,
                    rows,
                )
                .expect("rope q")
            });
            head += time_ms(&|enc: &metal::ComputeCommandEncoderRef| {
                ml.encode_rope(
                    enc,
                    sp(&k_norm),
                    sp(&ctab),
                    sp(&stab),
                    sp(&vcache),
                    n_kv,
                    hd,
                    7936,
                    rows,
                )
                .expect("rope k")
            });
            if has_v {
                head += time_ms(&|enc: &metal::ComputeCommandEncoderRef| {
                    ml.encode_copy(enc, sp(&v_raw), sp(&vcache), rows * n_kv * hd)
                        .expect("copy v")
                });
            } else {
                head += time_ms(&|enc: &metal::ComputeCommandEncoderRef| {
                    ml.encode_rms(enc, sp(&k_raw), None, sp(&v_raw), rows * n_kv, hd, eps)
                        .expect("weightless rms v")
                });
            }
            (hidden, head)
        };
        let (sl_hid, sl_head) = rope_glue(256, 32, 16, true);
        let (fl_hid, fl_head) = rope_glue(512, 32, 4, false);
        let sl_glue = sl_hid + sl_head;
        let fl_glue = fl_hid + fl_head;
        eprintln!(
            "  sliding layer: hidden {sl_hid:.2} + head {sl_head:.2} = {sl_glue:.2} ms | \
             full layer: hidden {fl_hid:.2} + head {fl_head:.2} = {fl_glue:.2} ms"
        );

        // ---- real 256-span wall ----
        let mut kv_w = gm.new_kv_cache(264);
        gm.forward_cached(&mut kv_w, &[2, 3, 4, 5, 6, 7, 8, 9])
            .expect("warmup forward");
        let tokens: Vec<u32> = (0..256u32).map(|i| (i % 997) + 2).collect();
        let mut kv = gm.new_kv_cache(264);
        let t0 = std::time::Instant::now();
        let logits = gm.forward_cached(&mut kv, &tokens).expect("256 span");
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        assert_eq!(logits.len(), gm.cfg.vocab);

        let proj = |sl_attn: f64, fl_attn: f64| -> (f64, f64, f64, f64, f64) {
            let lin = 50.0 * l0_sum + 10.0 * l5_sum;
            let glue = 50.0 * sl_glue + 10.0 * fl_glue;
            let attn = 50.0 * sl_attn + 10.0 * fl_attn;
            (lin, glue, attn, lin + glue + attn + lm_head_ms, lm_head_ms)
        };
        // First chunk of any prompt: both kinds see causal kv=256.
        let (p1l, p1g, p1a, p1t, _) = proj(sl0, fl0);
        // 512-prompt average chunk: kv 256 then 512 -> mean of those cases.
        let (p2l, p2g, p2a, p2t, _) = proj((sl0 + sl1) / 2.0, (fl0 + fl1) / 2.0);
        // 8k mid chunk: sliding at window cap, full at kv≈4096 (prompt average).
        let (p3l, p3g, p3a, p3t, lmt) = proj(sl2, fl4);
        eprintln!("-- projections (ms per 256-row chunk, 50 sliding + 10 full)");
        eprintln!(
            "  first chunk:  linears {p1l:.0} + glue {p1g:.0} + attn {p1a:.0} + head {lmt:.1} = {p1t:.0}"
        );
        eprintln!(
            "  512-avg:      linears {p2l:.0} + glue {p2g:.0} + attn {p2a:.0} + head {lmt:.1} = {p2t:.0}"
        );
        eprintln!(
            "  8k mid:       linears {p3l:.0} + glue {p3g:.0} + attn {p3a:.0} + head {lmt:.1} = {p3t:.0}"
        );
        eprintln!(
            "MEASURED real 256-span wall: {wall_ms:.0} ms ({:.1} t/s) | \
             budget first-chunk {p1t:.0} ms | unaccounted {:+.0} ms",
            256000.0 / wall_ms,
            wall_ms - p1t,
        );
    }

    #[test]
    fn shape_derivation_matches_artifact_convention() {
        let shape = Exl3Shape::new(3840, 3072, 3).unwrap();
        assert_eq!(shape.in_tiles, 240);
        assert_eq!(shape.out_tiles, 192);
        assert_eq!(shape.words_per_tile(), 48);
        assert_eq!(shape.stream_words(), 24);
        assert!(Exl3Shape::new(3840, 3072, 0).is_none());
        assert!(Exl3Shape::new(129, 128, 3).is_none());
    }

    /// Real artifact, real f16 scales: one layer's q_proj through both paths.
    /// Gate on `KAPLUMBA_RUN_SLOW=1` (loads 16 GB).
    #[test]
    fn real_artifact_linear_matches_cpu_slow() {
        if std::env::var("KAPLUMBA_RUN_SLOW").is_err() {
            eprintln!("skip: set KAPLUMBA_RUN_SLOW=1");
            return;
        }
        let Some(dir) = crate::testutil::artifact_dir() else {
            eprintln!("skip: no artifact");
            return;
        };
        let Some(c) = ctx() else { return };
        let model = crate::load::load_exl3(&c, &dir).expect("load artifact");
        let g = MetalLinear::new(&c).expect("metal linear");
        let name = "model.language_model.layers.0.self_attn.q_proj";
        let src: &dyn TensorSource = &model;
        let shape = {
            let t = src.get(&format!("{name}.trellis")).unwrap();
            (t.shape[0] * 16, t.shape[1] * 16)
        };
        let mut rng = Rng(0x5EED);
        let x: Vec<f16> = (0..shape.0)
            .map(|_| f16::from_f32(((rng.next() % 2001) as f32 / 1000.0) - 1.0))
            .collect();

        let gm = crate::gemma4::Gemma4Model::from_model(&model).expect("cfg");
        let want = gm.linear_via_cpu(name, &x).expect("cpu linear");
        let gpu = g.try_linear(src, name, &x).unwrap().expect("gpu eligible");
        assert_eq!(gpu.len(), want.len());
        let err = max_rel_err(&want, &gpu);
        assert!(err < 5e-3, "{name}: rel err {err}");
        eprintln!(
            "{name}: {}x{} cpu vs gpu max rel err {err:.2e}",
            shape.0, shape.1
        );
    }

    /// GPU attention vs the engine's CPU reference on the real layer shapes
    /// (sliding: 32q/16kv × 256; full: 32q/4kv × 512) with the config's window.
    #[test]
    fn real_artifact_attention_matches_cpu_slow() {
        if std::env::var("KAPLUMBA_RUN_SLOW").is_err() {
            eprintln!("skip: set KAPLUMBA_RUN_SLOW=1");
            return;
        }
        let Some(dir) = crate::testutil::artifact_dir() else {
            eprintln!("skip: no artifact");
            return;
        };
        let Some(c) = ctx() else { return };
        let model = crate::load::load_exl3(&c, &dir).expect("load artifact");
        let g = MetalLinear::new(&c).expect("metal linear");
        let cfg = crate::gemma4::GemmaCfg::from_text_config(&model.config.text_config)
            .expect("gemma cfg");
        let gm = crate::gemma4::Gemma4Model::from_model(&model).expect("model");

        let mut rng = Rng(0xA77E);
        let mut half = || f16::from_f32(((rng.next() % 2001) as f32 / 1000.0) - 1.0);

        for (i, kind) in [(0usize, "sliding"), (5, "full")] {
            let lc = cfg.layers[i];
            let hd = lc.head_dim;
            let s = 12usize;
            let q: Vec<f16> = (0..s * cfg.n_heads * hd).map(|_| half()).collect();
            let k: Vec<f16> = (0..s * lc.n_kv * hd).map(|_| half()).collect();
            let v: Vec<f16> = (0..s * lc.n_kv * hd).map(|_| half()).collect();
            let want = gm.attention_cpu(&lc, &q, &k, &v, s, 0, 0, s);
            let got = g
                .try_attention(
                    &q,
                    &k,
                    &v,
                    AttnParams {
                        s,
                        kv_rows: s,
                        q_pos0: 0,
                        kv_pos0: 0,
                        n_heads: cfg.n_heads,
                        n_kv: lc.n_kv,
                        head_dim: hd,
                        window: crate::gemma4::layer_window(lc.kind, cfg.sliding_window),
                    },
                )
                .expect("dispatch")
                .expect("gpu eligible");
            let err = max_rel_err(&want, &got);
            eprintln!("layer {i} ({kind}) attention: hd={hd} max rel err {err:.2e}");
            assert!(err < 4e-3, "layer {i}: rel err {err}");
        }
    }

    /// Every decoder linear in the real artifact must be GPU-eligible.
    ///
    /// A single silent CPU fallback costs ~0.5 s/token, and `gpu_linears`
    /// alone cannot tell you *why* a linear was skipped. Gate on
    /// `KAPLUMBA_RUN_SLOW=1` (loads 16 GB, ~5 s — no forward passes).
    #[test]
    fn all_layer_linears_gpu_eligible_slow() {
        if std::env::var("KAPLUMBA_RUN_SLOW").is_err() {
            eprintln!("skip: set KAPLUMBA_RUN_SLOW=1");
            return;
        }
        let Some(dir) = crate::testutil::artifact_dir() else {
            eprintln!("skip: no artifact");
            return;
        };
        let Some(c) = ctx() else { return };
        let model = crate::load::load_exl3(&c, &dir).expect("load artifact");
        let g = MetalLinear::new(&c).expect("metal linear");
        let src: &dyn TensorSource = &model;
        let cfg = crate::gemma4::GemmaCfg::from_text_config(&model.config.text_config)
            .expect("gemma cfg");

        let mut names: Vec<String> = Vec::new();
        for i in 0..cfg.n_layers {
            let p = format!("model.language_model.layers.{i}");
            for m in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj"] {
                names.push(format!("{p}.{m}"));
            }
            if cfg.layers[i].kind == crate::gemma4::LayerKind::Sliding {
                names.push(format!("{p}.self_attn.v_proj"));
            }
            for m in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] {
                names.push(format!("{p}.{m}"));
            }
        }
        names.push("lm_head".to_string());

        let mut skipped = Vec::new();
        for name in &names {
            if let Some(reason) = g.skip_reason(src, name) {
                skipped.push(format!("{name}: {reason}"));
            }
        }
        for s in &skipped {
            eprintln!("CPU fallback -> {s}");
        }
        eprintln!(
            "{} of {} linears GPU-eligible",
            names.len() - skipped.len(),
            names.len()
        );
        assert!(skipped.is_empty(), "{} skipped linears", skipped.len());

        // Eligibility is necessary but not sufficient: the runtime dispatch
        // must actually take the GPU branch. A one-layer forward makes that
        // observable (7 linears + lm_head for a sliding layer).
        let gm = crate::gemma4::Gemma4Model::from_model(&model)
            .expect("cfg")
            .with_gpu(&g);
        let before = g.calls();
        let x = gm.forward_seq_layers(&[2], 1).expect("1-layer forward");
        let _ = gm.logits(&x).expect("logits");
        let dispatched = g.calls() - before;
        eprintln!("1-layer forward + lm_head dispatched {dispatched} GPU linears");
        assert_eq!(
            dispatched, 8,
            "expected 7 layer linears + lm_head on the GPU"
        );
    }

    /// T3.x: batched (`rows ≥ 2` → simdgroup GEMM) vs single-row GEMV on the
    /// REAL artifact trellises — the synthetic `gpu_gemm_rows_...` proves the
    /// math, this proves the real byte offsets / `k` / codebook / shapes.
    /// Isolates kernel correctness from model-level error compounding: a bad
    /// decode or index shows up here as ~1e-1, while summation-order noise
    /// stays within an f16 ulp (2e-3 with the `max_rel_err` divisor ≥ 1).
    /// `rows = 65` covers an exact 64-row block plus a ragged tail (xh pad +
    /// `gr < blk_len` store mask).
    #[test]
    fn real_artifact_batched_linear_matches_gemv_slow() {
        if std::env::var("KAPLUMBA_RUN_SLOW").is_err() {
            eprintln!("skip: set KAPLUMBA_RUN_SLOW=1");
            return;
        }
        let Some(dir) = crate::testutil::artifact_dir() else {
            eprintln!("skip: no artifact");
            return;
        };
        let Some(c) = ctx() else { return };
        let model = crate::load::load_exl3(&c, &dir).expect("load artifact");
        let ml = MetalLinear::new(&c).expect("metal linear");
        let src: &dyn TensorSource = &model;
        let f16_bytes =
            |v: &[f16]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) };
        let new_buf = |bytes: &[u8]| {
            c.device().new_buffer_with_data(
                bytes.as_ptr() as *const _,
                bytes.len().max(1) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            )
        };
        let run = |handles: &LinearHandles, xb: &metal::Buffer, yb: &metal::Buffer, rows: usize| {
            let cb = ml.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            ml.encode_linear(enc, handles, xb, yb, rows)
                .expect("encode_linear");
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            assert_eq!(
                cb.status(),
                metal::MTLCommandBufferStatus::Completed,
                "command buffer failed"
            );
        };

        let rows = 65usize;
        for name in [
            "model.language_model.layers.0.self_attn.q_proj", // 5376×8192
            "model.language_model.layers.0.mlp.gate_proj",    // 5376×21504
            "model.language_model.layers.0.mlp.down_proj",    // 21504×5376
        ] {
            let handles = linear_handles(src, name).expect("eligible");
            let inn = handles.shape.in_features();
            let out_f = handles.shape.out_features();
            let mut rng = Rng(0x5EED);
            let x: Vec<f16> = (0..rows * inn)
                .map(|_| f16::from_f32(((rng.next() % 2001) as f32 / 1000.0) - 1.0))
                .collect();

            let xb = new_buf(f16_bytes(&x));
            let yb = c.device().new_buffer(
                (rows * out_f * 2) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            run(&handles, &xb, &yb, rows);
            let batched: Vec<f16> =
                unsafe { std::slice::from_raw_parts(yb.contents() as *const f16, rows * out_f) }
                    .to_vec();

            let mut worst = 0.0f32;
            for r in 0..rows {
                let xb1 = new_buf(f16_bytes(&x[r * inn..(r + 1) * inn]));
                let yb1 = c.device().new_buffer(
                    (out_f * 2) as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                run(&handles, &xb1, &yb1, 1);
                let gemv: Vec<f16> =
                    unsafe { std::slice::from_raw_parts(yb1.contents() as *const f16, out_f) }
                        .to_vec();
                let err = max_rel_err(&batched[r * out_f..(r + 1) * out_f], &gemv);
                worst = worst.max(err);
                assert!(
                    err < 2e-3,
                    "{name} row {r} (of {rows}): batched sg vs gemv rel err {err}"
                );
            }
            eprintln!("{name}: {rows} rows vs per-row gemv worst rel err {worst:.2e}");
        }
    }

    /// T2.9 precondition: sibling linears share the input Hadamard table
    /// (`suh`) byte-for-byte — only then may one `had_in` feed the group.
    /// RESULT (measured): q/k suh **differ** on this artifact → qkv
    /// shared-had_in fusion is impossible; the test reports every pair so the
    /// gate+up decision is also evidence-based. A non-sibling pair (q vs
    /// gate — different rms inputs) must differ, proving discrimination.
    #[test]
    fn sibling_input_tables_shared_slow() {
        if std::env::var("KAPLUMBA_RUN_SLOW").is_err() {
            eprintln!("skip: set KAPLUMBA_RUN_SLOW=1");
            return;
        }
        let Some(dir) = crate::testutil::artifact_dir() else {
            eprintln!("skip: no artifact");
            return;
        };
        let Some(c) = ctx() else { return };
        let model = crate::load::load_exl3(&c, &dir).expect("load artifact");
        let src: &dyn TensorSource = &model;
        let base = "model.language_model.layers.0";
        let qh = linear_handles(src, &format!("{base}.self_attn.q_proj")).expect("eligible");
        let kh = linear_handles(src, &format!("{base}.self_attn.k_proj")).expect("eligible");
        let gh = linear_handles(src, &format!("{base}.mlp.gate_proj")).expect("eligible");
        let uh = linear_handles(src, &format!("{base}.mlp.up_proj")).expect("eligible");
        let vh = linear_handles(src, &format!("{base}.self_attn.v_proj")).ok();
        let inn = qh.shape.in_features() * 2;
        let same = |a: DeviceSpan<'_>, b: DeviceSpan<'_>| -> bool {
            let (pa, pb) = (a.byte_offset as usize, b.byte_offset as usize);
            let (ba, bb) = (
                unsafe {
                    std::slice::from_raw_parts(a.buffer.contents().add(pa) as *const u8, inn)
                },
                unsafe {
                    std::slice::from_raw_parts(b.buffer.contents().add(pb) as *const u8, inn)
                },
            );
            ba == bb
        };
        let report = |label: &str, eq: bool| {
            eprintln!(
                "suh[{label}] {}",
                if eq {
                    "SHARED → had_in fusable"
                } else {
                    "DIFFER → no share"
                }
            );
            eq
        };
        let q = qh.suh();
        let k = kh.suh();
        let g = gh.suh();
        let u = uh.suh();
        let qk = report("q==k", same(q, k));
        let qu = report("gate==up", same(g, u));
        match vh.as_ref() {
            Some(v) => {
                report("q==v", same(q, v.suh()));
            }
            None => eprintln!("suh[q==v] v_proj not present (k_eq_v?)"),
        }
        assert!(
            !same(q, g),
            "q vs gate suh unexpectedly equal — comparison not discriminating"
        );
        // The fusion callers gate on these exact facts.
        assert_eq!(qk, false, "qkv share suh now? fuse qkv");
        assert_eq!(qu, false, "gate/up share suh now? fuse gate+up");
    }
}
