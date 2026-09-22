//! EXL3 weight reconstruction: packed trellis → inner f16 matrix
//! and optional public weight (outer Hadamard + suh/svh).

use half::f16;

use super::codebook::CodebookMode;
use super::decode::decode_packed_tile;
use super::hadamard::{HAD_DIM, hadamard_128};
use super::signs::unpack_sign_bitfield;

/// Packed trellis view: `(in_tiles, out_tiles, packed_size)` row-major u16.
#[derive(Debug, Clone, Copy)]
pub struct TrellisView<'a> {
    pub data: &'a [u16],
    pub in_tiles: usize,
    pub out_tiles: usize,
    pub packed_size: usize,
}

impl<'a> TrellisView<'a> {
    pub fn new(
        data: &'a [u16],
        in_tiles: usize,
        out_tiles: usize,
        packed_size: usize,
    ) -> Result<Self, ReconstructError> {
        let expect = in_tiles
            .checked_mul(out_tiles)
            .and_then(|n| n.checked_mul(packed_size))
            .ok_or(ReconstructError::Overflow)?;
        if data.len() != expect {
            return Err(ReconstructError::Shape {
                expected: expect,
                got: data.len(),
            });
        }
        Ok(Self {
            data,
            in_tiles,
            out_tiles,
            packed_size,
        })
    }

    pub fn in_features(&self) -> usize {
        self.in_tiles * 16
    }

    pub fn out_features(&self) -> usize {
        self.out_tiles * 16
    }

    /// Derive trellis `k` from packed size: `packed = 256*k/16`.
    pub fn k(&self) -> u32 {
        (self.packed_size * 16 / 256) as u32
    }

    pub fn tile(&self, tk: usize, tn: usize) -> &'a [u16] {
        debug_assert!(tk < self.in_tiles && tn < self.out_tiles);
        let idx = (tk * self.out_tiles + tn) * self.packed_size;
        &self.data[idx..idx + self.packed_size]
    }

    /// Decode one tile to row-major 16×16 (in-block rows, out-block cols).
    pub fn decode_tile(
        &self,
        tk: usize,
        tn: usize,
        mode: CodebookMode,
    ) -> Result<[[f16; 16]; 16], crate::exl3::trellis::TrellisError> {
        decode_packed_tile(self.tile(tk, tn), self.k(), mode)
    }
}

/// Reconstruct the inner weight matrix `W[in, out]` (before outer Hadamard /
/// signs) as row-major f16.
///
/// Slow and memory-heavy for large heads (lm_head ≈ 2.8 GB) — prefer
/// [`reconstruct_inner_slab`] or streaming forward on big layers.
pub fn reconstruct_inner(
    trellis: &TrellisView<'_>,
    mode: CodebookMode,
) -> Result<Vec<f16>, crate::exl3::trellis::TrellisError> {
    let inn = trellis.in_features();
    let out = trellis.out_features();
    let mut w = vec![f16::ZERO; inn * out];
    for tn in 0..trellis.out_tiles {
        for tk in 0..trellis.in_tiles {
            let tile = trellis.decode_tile(tk, tn, mode)?;
            let r0 = tk * 16;
            let c0 = tn * 16;
            for (i, row) in tile.iter().enumerate() {
                let dst = (r0 + i) * out + c0;
                w[dst..dst + 16].copy_from_slice(row);
            }
        }
    }
    Ok(w)
}

/// Reconstruct output columns `[n_offset, n_offset + n_count)` (n_count % 128 == 0
/// when aligning to Hadamard blocks; any multiple of 16 is valid for the inner matrix).
pub fn reconstruct_inner_slab(
    trellis: &TrellisView<'_>,
    mode: CodebookMode,
    n_offset: usize,
    n_count: usize,
) -> Result<Vec<f16>, crate::exl3::trellis::TrellisError> {
    if n_offset % 16 != 0 || n_count % 16 != 0 {
        return Err(crate::exl3::trellis::TrellisError::BadLen {
            expected: n_offset % 16 + n_count % 16,
            got: 0,
        });
    }
    let out = trellis.out_features();
    let inn = trellis.in_features();
    if n_offset + n_count > out {
        return Err(crate::exl3::trellis::TrellisError::BadLen {
            expected: out,
            got: n_offset + n_count,
        });
    }
    let tn0 = n_offset / 16;
    let tn1 = (n_offset + n_count) / 16;
    let mut w = vec![f16::ZERO; inn * n_count];
    let mut col = 0;
    for tn in tn0..tn1 {
        for tk in 0..trellis.in_tiles {
            let tile = trellis.decode_tile(tk, tn, mode)?;
            let r0 = tk * 16;
            for (i, row) in tile.iter().enumerate() {
                let dst = (r0 + i) * n_count + col;
                w[dst..dst + 16].copy_from_slice(row);
            }
        }
        col += 16;
    }
    Ok(w)
}

fn preapply_had_left(w: &mut [f16], inn: usize, out: usize) {
    assert_eq!(
        inn % HAD_DIM,
        0,
        "in_features must be multiple of {HAD_DIM}"
    );
    let h = hadamard_128();
    let mut block = [0f32; HAD_DIM];
    let mut acc = [0f32; HAD_DIM];
    for r0 in (0..inn).step_by(HAD_DIM) {
        for c in 0..out {
            for i in 0..HAD_DIM {
                block[i] = w[(r0 + i) * out + c].to_f32();
            }
            for (k, &bv) in block.iter().enumerate() {
                if bv == 0.0 {
                    continue;
                }
                let hk = &h[k];
                for (o, a) in acc.iter_mut().enumerate() {
                    *a += bv * hk[o];
                }
            }
            for (o, a) in acc.iter().enumerate() {
                w[(r0 + o) * out + c] = f16::from_f32(*a);
            }
            acc.fill(0.0);
        }
    }
}

fn preapply_had_right(w: &mut [f16], inn: usize, out: usize) {
    assert_eq!(
        out % HAD_DIM,
        0,
        "out_features must be multiple of {HAD_DIM}"
    );
    let h = hadamard_128();
    let mut block = [0f32; HAD_DIM];
    let mut acc = [0f32; HAD_DIM];
    for c0 in (0..out).step_by(HAD_DIM) {
        for r in 0..inn {
            for j in 0..HAD_DIM {
                block[j] = w[r * out + c0 + j].to_f32();
            }
            for (k, &bv) in block.iter().enumerate() {
                if bv == 0.0 {
                    continue;
                }
                let hk = &h[k];
                for (o, a) in acc.iter_mut().enumerate() {
                    *a += bv * hk[o];
                }
            }
            for (o, a) in acc.iter().enumerate() {
                w[r * out + c0 + o] = f16::from_f32(*a);
            }
            acc.fill(0.0);
        }
    }
}

/// Public weight used by `x @ W` (outer Hadamard + suh/svh folded in).
///
/// `suh` / `svh` are either already f16 scales (`in` / `out`) or packed sign
/// bitfields (`in/16` / `out/16` words).
pub fn reconstruct_public_weights(
    trellis: &TrellisView<'_>,
    suh: Option<&[u8]>,
    svh: Option<&[u8]>,
    mode: CodebookMode,
    suh_is_f16: bool,
    svh_is_f16: bool,
) -> Result<Vec<f16>, Box<dyn std::error::Error>> {
    let inn = trellis.in_features();
    let out = trellis.out_features();
    let mut w = reconstruct_inner(trellis, mode)?;
    let su = suh.map(|b| expand_signs(b, inn, suh_is_f16));
    let sv = svh.map(|b| expand_signs(b, out, svh_is_f16));
    if let Some(su) = &su {
        preapply_had_left(&mut w, inn, out);
        for r in 0..inn {
            let s = su[r].to_f32();
            let row = &mut w[r * out..(r + 1) * out];
            for v in row.iter_mut() {
                *v = f16::from_f32(v.to_f32() * s);
            }
        }
    }
    if let Some(sv) = &sv {
        preapply_had_right(&mut w, inn, out);
        for c in 0..out {
            let s = sv[c].to_f32();
            for r in 0..inn {
                let v = w[r * out + c].to_f32();
                w[r * out + c] = f16::from_f32(v * s);
            }
        }
    }
    Ok(w)
}

fn expand_signs(bytes: &[u8], n_elems: usize, is_f16: bool) -> Vec<f16> {
    if is_f16 {
        assert_eq!(bytes.len(), n_elems * 2);
        bytes
            .chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    } else {
        let words: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(words.len() * 16, n_elems);
        unpack_sign_bitfield(&words)
    }
}

/// Read trellis bytes (safetensors i16/u16 LE) as u16 words.
pub fn trellis_bytes_as_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Zero-copy view of trellis bytes as u16 words (little-endian hosts only).
///
/// The loader repacks every tensor at a 64-byte-aligned device offset
/// (`load_exl3`), which implies 2-byte alignment — note the *file* offsets are
/// only 2-byte aligned (alternating `off % 4`), so this relies on the repack,
/// not on the safetensors header. Copies nothing — the forward path relies on
/// this to avoid duplicating 15.5 GB of trellis per token.
pub fn trellis_u16_view(bytes: &[u8]) -> &[u16] {
    assert_eq!(bytes.len() % 2, 0, "trellis byte length must be even");
    assert_eq!(
        bytes.as_ptr() as usize % 2,
        0,
        "trellis byte slice must be 2-byte aligned"
    );
    debug_assert!(
        cfg!(target_endian = "little"),
        "u16 view assumes a little-endian host"
    );
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u16, bytes.len() / 2) }
}

#[derive(Debug, thiserror::Error)]
pub enum ReconstructError {
    #[error("trellis shape overflow")]
    Overflow,
    #[error("trellis data length {got} != expected {expected}")]
    Shape { expected: usize, got: usize },
    #[error(transparent)]
    Trellis(#[from] crate::exl3::trellis::TrellisError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exl3::trellis::pack_trellis_tile;

    fn synth_view(
        k: u32,
        in_tiles: usize,
        out_tiles: usize,
        seed: u32,
    ) -> (Vec<u16>, TrellisView<'static>) {
        // Build packed data, leak into 'static for view simplicity in tests.
        let packed_size = 256 * k as usize / 16;
        let mut data = Vec::with_capacity(in_tiles * out_tiles * packed_size);
        let mut state = seed;
        for _tk in 0..in_tiles {
            for _tn in 0..out_tiles {
                let mask = ((1u32 << k) - 1) as u16;
                let mut fresh = [0u16; 256];
                for f in fresh.iter_mut() {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    *f = (state >> 16) as u16 & mask;
                }
                let packed = pack_trellis_tile(&fresh, k).unwrap();
                data.extend_from_slice(&packed);
            }
        }
        let leaked: &'static [u16] = Box::leak(data.into_boxed_slice());
        let view = TrellisView::new(leaked, in_tiles, out_tiles, packed_size).unwrap();
        (leaked.to_vec(), view)
    }

    #[test]
    fn u16_view_is_bit_identical_to_copy() {
        let words: Vec<u16> = (0..64u16).map(|i| i.wrapping_mul(0x1234)).collect();
        let mut bytes = Vec::with_capacity(words.len() * 2);
        for w in &words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        assert_eq!(trellis_u16_view(&bytes), &words[..]);
        assert_eq!(trellis_bytes_as_u16(&bytes), words);
        // Odd length rejected rather than silently truncated.
        let r = std::panic::catch_unwind(|| trellis_u16_view(&bytes[..7]));
        assert!(r.is_err());
    }

    #[test]
    fn view_shape_and_k() {
        let packed_size = 48; // k=3
        let data = vec![0u16; 2 * 4 * packed_size];
        let v = TrellisView::new(&data, 2, 4, packed_size).unwrap();
        assert_eq!(v.in_features(), 32);
        assert_eq!(v.out_features(), 64);
        assert_eq!(v.k(), 3);
        assert!(TrellisView::new(&data[..10], 2, 4, packed_size).is_err());
    }

    #[test]
    fn slab_matches_full_columns() {
        let k = 3;
        let (_d, view) = synth_view(k, 3, 5, 42);
        let mode = CodebookMode::Mcg;
        let full = reconstruct_inner(&view, mode).unwrap();
        let inn = view.in_features();
        let out = view.out_features();
        let n_offset = 16;
        let n_count = 32;
        let slab = reconstruct_inner_slab(&view, mode, n_offset, n_count).unwrap();
        for r in 0..inn {
            for c in 0..n_count {
                assert_eq!(
                    slab[r * n_count + c],
                    full[r * out + n_offset + c],
                    "r={r} c={c}"
                );
            }
        }
    }

    #[test]
    fn single_tile_reconstruct_equals_decode() {
        use crate::exl3::decode::decode_packed_tile;
        let k = 3;
        let packed_size = 48;
        let mut rng = 0xC0FFEEu32;
        let mask = 7u16;
        let mut fresh = [0u16; 256];
        for f in fresh.iter_mut() {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            *f = (rng >> 16) as u16 & mask;
        }
        let packed = pack_trellis_tile(&fresh, k).unwrap();
        assert_eq!(packed.len(), packed_size);
        let tile = decode_packed_tile(&packed, k, CodebookMode::Mcg).unwrap();
        let view = TrellisView::new(&packed, 1, 1, packed_size).unwrap();
        let w = reconstruct_inner(&view, CodebookMode::Mcg).unwrap();
        assert_eq!(w.len(), 256);
        for i in 0..16 {
            for j in 0..16 {
                assert_eq!(w[i * 16 + j], tile[i][j], "tile[{i}][{j}]");
            }
        }
    }

    #[test]
    fn public_weights_apply_scales() {
        let k = 4;
        // 128×128 so outer Hadamard (dim 128) applies cleanly.
        let (_d, view) = synth_view(k, 8, 8, 7);
        let inn = view.in_features();
        let out = view.out_features();
        assert_eq!(inn, 128);
        assert_eq!(out, 128);
        // f16 suh/svh all +1 → public = had_left(had_right(inner)); +1.0 LE = 0x3C00.
        let mut suh = Vec::with_capacity(inn * 2);
        for _ in 0..inn {
            suh.extend_from_slice(&0x3C00u16.to_le_bytes());
        }
        let mut svh = Vec::with_capacity(out * 2);
        for _ in 0..out {
            svh.extend_from_slice(&0x3C00u16.to_le_bytes());
        }
        let w = reconstruct_inner(&view, CodebookMode::Mcg).unwrap();
        let w_pub = reconstruct_public_weights(
            &view,
            Some(&suh),
            Some(&svh),
            CodebookMode::Mcg,
            true,
            true,
        )
        .unwrap();
        // With +1 scales, only Hadamard changes values — not all-equal to inner.
        assert_eq!(w_pub.len(), w.len());
        let mut differs = false;
        for (a, b) in w.iter().zip(w_pub.iter()) {
            if a.to_f32() != b.to_f32() {
                differs = true;
                break;
            }
        }
        assert!(differs, "hadamard should change weights");
        // Shape preserved; all finite.
        for v in &w_pub {
            assert!(v.to_f32().is_finite());
        }
    }

    #[test]
    fn public_weights_reject_non_multiple_of_128() {
        // 16 features cannot take a 128-point outer Hadamard.
        let packed_size = 48;
        let data = vec![0u16; packed_size]; // 1×1 tile = 16×16
        let view = TrellisView::new(&data, 1, 1, packed_size).unwrap();
        let mut suh = Vec::with_capacity(32);
        for _ in 0..16 {
            suh.extend_from_slice(&0x3C00u16.to_le_bytes());
        }
        let mut svh = Vec::with_capacity(32);
        for _ in 0..16 {
            svh.extend_from_slice(&0x3C00u16.to_le_bytes());
        }
        let r = std::panic::catch_unwind(|| {
            reconstruct_public_weights(&view, Some(&suh), Some(&svh), CodebookMode::Mcg, true, true)
        });
        assert!(r.is_err());
    }
}
