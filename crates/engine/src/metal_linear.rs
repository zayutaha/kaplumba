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
use kaplumba_kernels::{DeviceSpan, Exl3Gemv, Exl3Shape, KernelError};
use metal::Buffer;

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
    gemv: Exl3Gemv,
    /// Linears actually dispatched to the GPU (proof that eligibility held for
    /// a real run — a silent fallback would otherwise look like a slow GPU).
    calls: std::sync::atomic::AtomicU64,
    /// Wall time spent inside the GPU dispatches (R7.2: separates kernel cost
    /// from whatever else the forward pass is doing).
    gpu_ns: std::sync::atomic::AtomicU64,
}

struct GpuLinear<'a> {
    shape: Exl3Shape,
    mode: u32,
    trellis: DeviceSpan<'a>,
    suh: DeviceSpan<'a>,
    svh: DeviceSpan<'a>,
}

impl MetalLinear {
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalLinearError> {
        Ok(Self {
            gemv: Exl3Gemv::new(ctx.device(), ctx.queue())?,
            calls: std::sync::atomic::AtomicU64::new(0),
            gpu_ns: std::sync::atomic::AtomicU64::new(0),
        })
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
        let Some(g) = gather(src, name) else {
            return Ok(None);
        };
        if x.len() != g.shape.in_features() {
            return Ok(None);
        }
        let mut out = Vec::new();
        let t0 = std::time::Instant::now();
        self.gemv
            .linear_row(x, g.trellis, g.suh, g.svh, g.shape, g.mode, &mut out)?;
        self.gpu_ns.fetch_add(
            t0.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Some(out))
    }
}

/// Collect everything the kernels need, or `None` when not eligible.
fn gather<'a>(src: &'a dyn TensorSource, name: &str) -> Option<GpuLinear<'a>> {
    gather_checked(src, name).ok()
}

/// `gather` with the reason it declined, so a silent CPU fallback is
/// diagnosable (see `skip_reason` and the `all_layer_linears_gpu_eligible_slow`
/// test).
fn gather_checked<'a>(src: &'a dyn TensorSource, name: &str) -> Result<GpuLinear<'a>, String> {
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

    Ok(GpuLinear {
        shape,
        mode,
        trellis: device_span(src, &trellis_name)
            .ok_or_else(|| format!("{trellis_name}: no device span"))?,
        suh: device_span(src, &format!("{name}.suh"))
            .ok_or_else(|| format!("{name}.suh: no device span"))?,
        svh: device_span(src, &format!("{name}.svh"))
            .ok_or_else(|| format!("{name}.svh: no device span"))?,
    })
}

fn device_span<'a>(src: &'a dyn TensorSource, name: &str) -> Option<DeviceSpan<'a>> {
    let (buffer, byte_offset): (&Buffer, usize) = src.device_span(name)?;
    Some(DeviceSpan {
        buffer,
        byte_offset,
    })
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
        let g = gather(&src, "l").expect("eligible");
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
        assert!(gather(&src, "l").is_none());

        // out_features not a multiple of 128
        let mut src = SynthGpuSource::build(c.device(), 3, 128, 128);
        src.meta.get_mut("l.trellis").unwrap().1[1] = 7;
        assert!(gather(&src, "l").is_none());

        // packed sign bitfield scales stay on the CPU
        let mut src = SynthGpuSource::build(c.device(), 3, 128, 128);
        src.meta.get_mut("l.suh").unwrap().0 = DType::I16;
        assert!(gather(&src, "l").is_none());

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
        assert!(gather(&src, "l").is_none());

        // unknown codebook multiplier
        let mut src = SynthGpuSource::build(c.device(), 3, 128, 128);
        src.host
            .insert("l.mcg".into(), 0xDEAD_BEEFu32.to_le_bytes().to_vec());
        assert!(gather(&src, "l").is_none());

        // missing device storage
        let mut src = SynthGpuSource::build(c.device(), 3, 128, 128);
        src.buffers.remove("l.svh");
        assert!(gather(&src, "l").is_none());
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
            ("lm_head", "lm_head"),
        ];
        for (label, name) in names {
            let g = gather(src, name).unwrap_or_else(|| panic!("{name} eligible"));
            let (inn, outp) = (g.shape.in_features(), g.shape.out_features());
            let x: Vec<f16> = (0..inn)
                .map(|i| f16::from_f32(((i % 17) as f32 - 8.0) / 64.0))
                .collect();
            const REPS: usize = 20;
            let verbose = std::env::var("KAPLUMBA_BENCH_VERBOSE").is_ok();
            let mut time = |tag: &str, f: &mut dyn FnMut()| -> (f64, f64) {
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
                ml.raw().had_in_pass(&x, g.suh, &mut o).unwrap();
            });
            let y: Vec<f32> = (0..outp).map(|i| (i % 13) as f32 * 0.01).collect();
            let (ho_min, ho_med) = time("had_out", &mut || {
                let mut o = Vec::new();
                ml.raw().had_out_pass(&y, g.svh, &mut o).unwrap();
            });
            let mut xh = vec![f16::ZERO; inn];
            let (gv_min, gv_med) = time("gemv", &mut || {
                let mut o = Vec::new();
                ml.raw()
                    .gemv_pass(&xh, g.trellis, g.shape, g.mode, &mut o)
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
                g.trellis.byte_offset % 4,
                g.suh.byte_offset % 4,
                g.svh.byte_offset % 4,
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
}
