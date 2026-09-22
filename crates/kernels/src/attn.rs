//! Host side of the bounded-scratch attention kernels (`attn.metal`, T3.5/T3.7).
//!
//! One `run` call = one dispatch: `out = softmax(q·kᵀ) · v` with causal +
//! optional sliding-window masking, GQA head sharing, and **explicit position
//! bases** for keys and queries. The bases are what let one kernel serve every
//! shape the engine has:
//!
//! | case | `s` | `kv_rows` | `q_pos0` | `kv_pos0` |
//! |---|---|---|---|---|
//! | prefill | prompt length | prompt length | 0 | 0 |
//! | cached decode | 1 | `pos + 1` | `pos` | 0 |
//! | windowed SWA cache | 1 | `<= window` | `pos` | `pos + 1 - kv_rows` |
//!
//! The last row is the important one for memory: a sliding layer keeps only its
//! last `window` keys, so key row 0 is *not* absolute position 0.
//!
//! The kernel is a compile-time specialization per head dim (`256` → 8 slices,
//! `512` → 16 slices). Anything else is rejected up front rather than reduced
//! with the wrong simd-group width.
//!
//! Scratch: none beyond the caller's buffers. The online-softmax state lives in
//! registers inside the kernel (see `attn.metal`), so there is nothing to
//! allocate per sequence length — the property T3.7 proves on the CLI side.

use metal::{
    CommandQueue, CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus, MTLSize,
};

use crate::exl3::{DeviceSpan, KernelError, check_span_fits};

const MSL: &str = include_str!("attn.metal");

/// Query rows per threadgroup in `attn.metal` (`AROWS`).
pub const ATTN_ROWS: usize = 8;

/// Shape of one attention pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttnParams {
    /// Query rows in this dispatch.
    pub s: usize,
    /// Key rows available (`kv_pos0 + kv_rows - 1` is the newest).
    pub kv_rows: usize,
    /// Absolute position of query row 0.
    pub q_pos0: usize,
    /// Absolute position of key row 0 (`<= q_pos0`).
    pub kv_pos0: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    /// 0 = full causal, else the sliding window size (in tokens).
    pub window: usize,
}

impl AttnParams {
    /// Simd-group width (threads per query row) for a supported head dim.
    pub fn part(&self) -> Option<usize> {
        match self.head_dim {
            256 => Some(8),
            512 => Some(16),
            _ => None,
        }
    }

    fn validate(&self) -> Result<(), KernelError> {
        if self.s == 0 || self.kv_rows == 0 {
            return Err(KernelError::Unsupported(format!(
                "attention needs s >= 1 and kv_rows >= 1, got s={} kv_rows={}",
                self.s, self.kv_rows
            )));
        }
        if self.kv_pos0 > self.q_pos0 {
            return Err(KernelError::Unsupported(format!(
                "keys cannot start after the first query: kv_pos0={} q_pos0={}",
                self.kv_pos0, self.q_pos0
            )));
        }
        if self.kv_pos0 + self.kv_rows > self.q_pos0 + self.s {
            return Err(KernelError::Unsupported(format!(
                "keys must not run past the last query: {} > {}",
                self.kv_pos0 + self.kv_rows,
                self.q_pos0 + self.s
            )));
        }
        if self.n_kv == 0 || self.n_heads % self.n_kv != 0 {
            return Err(KernelError::Unsupported(format!(
                "GQA ratio: n_heads={} n_kv={}",
                self.n_heads, self.n_kv
            )));
        }
        if self.part().is_none() {
            return Err(KernelError::Unsupported(format!(
                "head_dim {} (kernels cover 256 and 512)",
                self.head_dim
            )));
        }
        Ok(())
    }
}

/// `#[repr(C)]` mirror of the MSL `AttnParams` constant buffer.
#[repr(C)]
#[derive(Clone, Copy)]
struct MslAttnParams {
    s: u32,
    kv_rows: u32,
    q_pos0: u32,
    kv_pos0: u32,
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    window: u32,
}

/// Pipelines + queue for the attention kernels.
pub struct Exl3Attention {
    queue: CommandQueue,
    hd256: ComputePipelineState,
    hd512: ComputePipelineState,
}

impl Exl3Attention {
    pub fn new(device: &Device, queue: &CommandQueue) -> Result<Self, KernelError> {
        // Precise half conversions and `exp`: the flash-style online softmax must
        // land within a couple of f16 ulps of the CPU reference, which Metal's
        // fast-math would not guarantee.
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
            queue: queue.clone(),
            hd256: build("exl3_attn_hd256")?,
            hd512: build("exl3_attn_hd512")?,
        })
    }

    /// `out = attn(q, k, v)` for one dispatch, submitting and waiting.
    ///
    /// All four spans are f16 tensors: `q`/`out` are `[s, n_heads, head_dim]`,
    /// `k`/`v` are `[kv_rows, n_kv, head_dim]`.
    pub fn run(
        &self,
        q: DeviceSpan<'_>,
        k: DeviceSpan<'_>,
        v: DeviceSpan<'_>,
        out: DeviceSpan<'_>,
        p: AttnParams,
    ) -> Result<(), KernelError> {
        p.validate()?;
        let hd = p.head_dim;
        check_span_fits(&q, p.s * p.n_heads * hd * 2, "attn q", 2)?;
        check_span_fits(&k, p.kv_rows * p.n_kv * hd * 2, "attn k", 2)?;
        check_span_fits(&v, p.kv_rows * p.n_kv * hd * 2, "attn v", 2)?;
        check_span_fits(&out, p.s * p.n_heads * hd * 2, "attn out", 2)?;

        let part = p.part().expect("validated");
        let pipeline = if hd == 256 { &self.hd256 } else { &self.hd512 };
        let params = MslAttnParams {
            s: p.s as u32,
            kv_rows: p.kv_rows as u32,
            q_pos0: p.q_pos0 as u32,
            kv_pos0: p.kv_pos0 as u32,
            n_heads: p.n_heads as u32,
            n_kv: p.n_kv as u32,
            head_dim: hd as u32,
            window: p.window as u32,
        };

        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(q.buffer), q.byte_offset as u64);
        enc.set_buffer(1, Some(k.buffer), k.byte_offset as u64);
        enc.set_buffer(2, Some(v.buffer), v.byte_offset as u64);
        enc.set_buffer(3, Some(out.buffer), out.byte_offset as u64);
        enc.set_bytes(
            4,
            std::mem::size_of::<MslAttnParams>() as u64,
            &params as *const MslAttnParams as *const _,
        );
        let groups = p.s.div_ceil(ATTN_ROWS) as u64;
        enc.dispatch_thread_groups(
            MTLSize::new(groups, p.n_heads as u64, 1),
            MTLSize::new(part as u64, ATTN_ROWS as u64, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let status = cb.status();
        if status != MTLCommandBufferStatus::Completed {
            return Err(KernelError::CommandFailed(status));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    /// CPU reference: the semantics the kernel must reproduce (max-subtracted
    /// f32 softmax over the visible keys, f16 output, absolute-position mask).
    fn cpu_attn(q: &[f16], k: &[f16], v: &[f16], p: AttnParams) -> Vec<f16> {
        let hd = p.head_dim;
        let gsize = p.n_heads / p.n_kv;
        let mut out = vec![f16::ZERO; p.s * p.n_heads * hd];
        for r in 0..p.s {
            let pos = p.q_pos0 + r;
            for h in 0..p.n_heads {
                let kvh = h / gsize;
                let qoff = (r * p.n_heads + h) * hd;
                let lo_abs = if p.window == 0 {
                    p.kv_pos0
                } else {
                    // saturating: the window may be wider than the history so far
                    p.kv_pos0.max((pos + 1).saturating_sub(p.window))
                };
                let hi_abs = pos.min(p.kv_pos0 + p.kv_rows - 1);
                let mut sc = Vec::new();
                for a in lo_abs..=hi_abs {
                    let koff = ((a - p.kv_pos0) * p.n_kv + kvh) * hd;
                    let mut acc = 0f32;
                    for d in 0..hd {
                        acc += q[qoff + d].to_f32() * k[koff + d].to_f32();
                    }
                    sc.push((a, acc));
                }
                if !sc.is_empty() {
                    let m = sc.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0f32;
                    for (_, s) in sc.iter_mut() {
                        *s = (*s - m).exp();
                        sum += *s;
                    }
                    let inv = 1.0 / sum;
                    for d in 0..hd {
                        let mut acc = 0f32;
                        for &(a, e) in sc.iter() {
                            acc += e * inv * v[((a - p.kv_pos0) * p.n_kv + kvh) * hd + d].to_f32();
                        }
                        out[qoff + d] = f16::from_f32(acc);
                    }
                }
            }
        }
        out
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
            f16::from_f32(((self.next() % 2001) as f32 / 1000.0) - 1.0)
        }
    }

    fn ctx() -> Option<(metal::Device, CommandQueue)> {
        let d = metal::Device::system_default()?;
        let q = d.new_command_queue();
        Some((d, q))
    }

    fn upload(device: &metal::Device, data: &[f16]) -> metal::Buffer {
        device.new_buffer_with_data(
            data.as_ptr() as *const _,
            (data.len() * 2).max(2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        )
    }

    fn span(b: &metal::Buffer) -> DeviceSpan<'_> {
        DeviceSpan {
            buffer: b,
            byte_offset: 0,
        }
    }

    /// Run one case on the GPU and compare with the CPU reference.
    fn check_case(name: &str, p: AttnParams, tol: f32) -> Vec<f16> {
        let Some((device, queue)) = ctx() else {
            eprintln!("skip: no Metal device");
            return Vec::new();
        };
        let gpu = Exl3Attention::new(&device, &queue).expect("pipelines");
        let mut rng = Rng(0xA77E_2026);
        let q: Vec<f16> = (0..p.s * p.n_heads * p.head_dim)
            .map(|_| rng.half())
            .collect();
        let k: Vec<f16> = (0..p.kv_rows * p.n_kv * p.head_dim)
            .map(|_| rng.half())
            .collect();
        let v: Vec<f16> = (0..p.kv_rows * p.n_kv * p.head_dim)
            .map(|_| rng.half())
            .collect();
        let want = cpu_attn(&q, &k, &v, p);

        let qb = upload(&device, &q);
        let kb = upload(&device, &k);
        let vb = upload(&device, &v);
        let ob = device.new_buffer(
            (want.len() * 2).max(2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        gpu.run(span(&qb), span(&kb), span(&vb), span(&ob), p)
            .expect("attn dispatch");
        let got: Vec<f16> =
            unsafe { std::slice::from_raw_parts(ob.contents() as *const f16, want.len()).to_vec() };

        let mut worst = 0f32;
        let mut scale = 0f32;
        for (a, b) in want.iter().zip(&got) {
            worst = worst.max((a.to_f32() - b.to_f32()).abs());
            scale = scale.max(a.to_f32().abs());
        }
        let rel = if scale > 0.0 { worst / scale } else { worst };
        eprintln!(
            "{name}: s={} kv={} pos0={}/{} hd={} nh={} nkv={} win={} max abs {worst:.3e} rel {rel:.3e}",
            p.s, p.kv_rows, p.q_pos0, p.kv_pos0, p.head_dim, p.n_heads, p.n_kv, p.window
        );
        assert!(rel <= tol, "{name}: rel err {rel} > {tol}");
        got
    }

    /// Prefill / plain-cache shape: keys start at 0 and the queries end the KV
    /// vector (`q_pos0 = kv_rows - s`).
    fn p(
        s: usize,
        kv_rows: usize,
        hd: usize,
        n_heads: usize,
        n_kv: usize,
        window: usize,
    ) -> AttnParams {
        AttnParams {
            s,
            kv_rows,
            q_pos0: kv_rows - s,
            kv_pos0: 0,
            n_heads,
            n_kv,
            head_dim: hd,
            window,
        }
    }

    #[test]
    fn matches_cpu_reference_sliding_gqa() {
        // Sliding layers: head_dim 256, 16 KV heads, 2 query heads per KV head.
        check_case("prefill-window-wide", p(16, 16, 256, 32, 16, 1024), 4e-3);
        // Window smaller than the block: exercises per-row masking inside a tile.
        check_case("prefill-window-narrow", p(32, 32, 256, 32, 16, 4), 4e-3);
        check_case("window-3", p(8, 8, 256, 32, 16, 3), 4e-3);
    }

    #[test]
    fn matches_cpu_reference_full_layers() {
        // Full layers: head_dim 512, n_kv 4, 8 query heads per KV head.
        check_case("full-causal", p(16, 16, 512, 32, 4, 0), 4e-3);
        check_case("full-causal-odd", p(9, 9, 512, 32, 4, 0), 4e-3);
    }

    #[test]
    fn matches_cpu_reference_cached_decode() {
        check_case("decode-kv8", p(1, 8, 256, 2, 1, 0), 4e-3);
        check_case("decode-kv16", p(1, 16, 256, 2, 1, 0), 4e-3);
        check_case("decode-kv17", p(1, 17, 256, 2, 1, 0), 4e-3);
        check_case("decode-kv32", p(1, 32, 256, 2, 1, 0), 4e-3);
        check_case("decode-kv64", p(1, 64, 256, 2, 1, 0), 4e-3);
        check_case("decode-kv256", p(1, 256, 256, 2, 1, 0), 4e-3);
        check_case("decode-sliding", p(1, 512, 256, 32, 16, 1024), 4e-3);
        check_case("decode-sliding-bound", p(1, 1024, 256, 32, 16, 1024), 4e-3);
        check_case("decode-full", p(1, 700, 512, 32, 4, 0), 4e-3);
    }

    #[test]
    fn matches_cpu_reference_boundaries() {
        // Single visible key: softmax must be exactly 1.0 → out == v.
        check_case("kv-1", p(1, 1, 256, 32, 16, 1024), 4e-3);
        // s not a multiple of AROWS (padded rows must not write or trap).
        check_case("padded-rows", p(10, 10, 256, 32, 16, 0), 4e-3);
    }

    /// Minimal non-degenerate case: 2 rows, 2 keys, one head, random q/k/v.
    /// Separates "dot product / slice reduction is wrong" from "multi-head or
    /// masking is wrong".
    #[test]
    fn tiny_random() {
        check_case("tiny-2x2", p(2, 2, 256, 1, 1, 0), 4e-3);
        check_case("tiny-2x2-window1", p(2, 2, 256, 1, 1, 1), 4e-3);
        check_case("2heads", p(2, 2, 256, 2, 1, 0), 4e-3);
        check_case("32heads16kv", p(2, 2, 256, 32, 16, 0), 4e-3);
        check_case("8rows-1head", p(8, 8, 256, 1, 1, 0), 4e-3);
        check_case("16rows-1head", p(16, 16, 256, 1, 1, 0), 4e-3);
        check_case("16rows-2heads", p(16, 16, 256, 2, 1, 0), 4e-3);
    }

    /// Exact, hand-checkable case: uniform scores (q = k = 0) over one-hot `v`.
    ///
    /// Row `r` (position `r`) must return `v[kv] / (r + 1)` summed, i.e. dim `d`
    /// gets `1/(r+1)` for `d <= r` and 0 beyond — no tolerance games, and it
    /// pins the masking direction (causal, not anti-causal) and the row/head
    /// indexing at once.
    #[test]
    fn one_hot_golden() {
        let Some((device, queue)) = ctx() else {
            eprintln!("skip: no Metal device");
            return;
        };
        let gpu = Exl3Attention::new(&device, &queue).expect("pipelines");
        let hd = 256usize;
        let p = AttnParams {
            s: 2,
            kv_rows: 2,
            q_pos0: 0,
            kv_pos0: 0,
            n_heads: 1,
            n_kv: 1,
            head_dim: hd,
            window: 0,
        };
        let zeros = vec![f16::ZERO; 2 * hd];
        let mut v = vec![f16::ZERO; 2 * hd];
        v[0] = f16::ONE; // v[0] is one-hot on dim 0
        v[hd + 1] = f16::ONE; // v[1] is one-hot on dim 1

        let qb = upload(&device, &zeros);
        let kb = upload(&device, &zeros);
        let vb = upload(&device, &v);
        let ob = device.new_buffer(
            (2 * hd * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        gpu.run(span(&qb), span(&kb), span(&vb), span(&ob), p)
            .expect("attn dispatch");
        let got: Vec<f16> =
            unsafe { std::slice::from_raw_parts(ob.contents() as *const f16, 2 * hd).to_vec() };

        let mut got_dims = [[0f32; 4]; 2];
        for r in 0..2 {
            for d in 0..4 {
                got_dims[r][d] = got[r * hd + d].to_f32();
            }
        }
        eprintln!("one-hot rows: {got_dims:?}");
        assert_eq!(got_dims[0], [1.0, 0.0, 0.0, 0.0], "row 0");
        assert_eq!(got_dims[1], [0.5, 0.5, 0.0, 0.0], "row 1");
    }

    /// Windowed SWA cache: key row 0 is *not* absolute position 0.
    ///
    /// A decode step whose cache holds only the last `window` keys must return
    /// exactly what a full-history attention with the same window returns,
    /// because the extra keys are invisible either way. This is the check that
    /// the `q_pos0`/`kv_pos0` bases are honoured: with the earlier
    /// "kv row j is at q_pos0 - s + j" shortcut this case is silently wrong.
    #[test]
    fn windowed_cache_matches_full_history() {
        let Some((device, queue)) = ctx() else {
            eprintln!("skip: no Metal device");
            return;
        };
        let gpu = Exl3Attention::new(&device, &queue).expect("pipelines");
        let (hd, nh, nkv, window) = (256usize, 32usize, 16usize, 4usize);
        let total = 12usize;
        let mut rng = Rng(0xC0DE_1234);
        let q_all: Vec<f16> = (0..total * nh * hd).map(|_| rng.half()).collect();
        let k_all: Vec<f16> = (0..total * nkv * hd).map(|_| rng.half()).collect();
        let v_all: Vec<f16> = (0..total * nkv * hd).map(|_| rng.half()).collect();

        let mut worst = 0f32;
        for pos in 7..total {
            // Full-history reference: all keys 0..=pos, window applied by mask.
            let full = AttnParams {
                s: 1,
                kv_rows: pos + 1,
                q_pos0: pos,
                kv_pos0: 0,
                n_heads: nh,
                n_kv: nkv,
                head_dim: hd,
                window,
            };
            let want = cpu_attn(
                &q_all[pos * nh * hd..(pos + 1) * nh * hd],
                &k_all,
                &v_all,
                full,
            );

            // Windowed cache: only the last `window` keys, at their true positions.
            let kv_rows = window.min(pos + 1);
            let kv_pos0 = pos + 1 - kv_rows;
            let cached = AttnParams {
                s: 1,
                kv_rows,
                q_pos0: pos,
                kv_pos0,
                n_heads: nh,
                n_kv: nkv,
                head_dim: hd,
                window,
            };
            let k_win = &k_all[kv_pos0 * nkv * hd..(pos + 1) * nkv * hd];
            let v_win = &v_all[kv_pos0 * nkv * hd..(pos + 1) * nkv * hd];
            let q = &q_all[pos * nh * hd..(pos + 1) * nh * hd];
            let qb = upload(&device, q);
            let kb = upload(&device, k_win);
            let vb = upload(&device, v_win);
            let ob = device.new_buffer(
                (hd * nh * 2) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            gpu.run(span(&qb), span(&kb), span(&vb), span(&ob), cached)
                .expect("attn dispatch");
            let got: Vec<f16> = unsafe {
                std::slice::from_raw_parts(ob.contents() as *const f16, hd * nh).to_vec()
            };

            for (a, b) in want.iter().zip(&got) {
                worst = worst.max((a.to_f32() - b.to_f32()).abs());
            }
        }
        eprintln!("windowed cache vs full history: max abs {worst:.3e}");
        assert!(worst < 4e-3, "windowed cache mismatch: {worst}");
    }

    #[test]
    fn rejects_unsupported_shapes() {
        let Some((device, queue)) = ctx() else {
            eprintln!("skip: no Metal device");
            return;
        };
        let gpu = Exl3Attention::new(&device, &queue).expect("pipelines");
        let qb = device.new_buffer(4, metal::MTLResourceOptions::StorageModeShared);
        let bad = AttnParams {
            s: 1,
            kv_rows: 1,
            q_pos0: 0,
            kv_pos0: 0,
            n_heads: 4,
            n_kv: 4,
            head_dim: 384,
            window: 0,
        };
        let err = gpu.run(span(&qb), span(&qb), span(&qb), span(&qb), bad);
        assert!(matches!(err, Err(KernelError::Unsupported(_))), "{err:?}");

        // GQA ratio must divide.
        let bad = AttnParams {
            n_heads: 6,
            n_kv: 4,
            head_dim: 256,
            ..bad
        };
        assert!(matches!(
            gpu.run(span(&qb), span(&qb), span(&qb), span(&qb), bad),
            Err(KernelError::Unsupported(_))
        ));
        // Keys must not start after the queries, nor run past them.
        let bad = AttnParams {
            n_heads: 4,
            n_kv: 4,
            head_dim: 256,
            q_pos0: 2,
            kv_pos0: 5,
            ..bad
        };
        assert!(matches!(
            gpu.run(span(&qb), span(&qb), span(&qb), span(&qb), bad),
            Err(KernelError::Unsupported(_))
        ));
    }
}
