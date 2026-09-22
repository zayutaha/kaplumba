//! Naive EXL3 linear forward via reconstruct (T1.3b) — golden CPU path.
//!
//! Matches reference `linear_forward_reconstruct`:
//! `y = had_r(had_r(x, pre=suh) @ W_inner, post=svh)`.
//! Streams output in 128-column slabs so lm_head never materializes 2.8 GB.

use half::f16;

use super::codebook::CodebookMode;
use super::hadamard::had_r_128;
use super::reconstruct::{TrellisView, reconstruct_inner_slab};

/// One EXL3 linear layer as stored (inner trellis + outer scales).
#[derive(Debug, Clone, Copy)]
pub struct Exl3Linear<'a> {
    pub trellis: TrellisView<'a>,
    /// Raw suh bytes (f16 or packed sign bitfield).
    pub suh: Option<&'a [u8]>,
    pub svh: Option<&'a [u8]>,
    pub suh_is_f16: bool,
    pub svh_is_f16: bool,
    pub mode: CodebookMode,
    pub bias: Option<&'a [u8]>, // f16 LE, out_features
}

impl<'a> Exl3Linear<'a> {
    pub fn in_features(&self) -> usize {
        self.trellis.in_features()
    }

    pub fn out_features(&self) -> usize {
        self.trellis.out_features()
    }
}

fn expand_scales(bytes: Option<&[u8]>, n: usize, is_f16: bool) -> Option<Vec<f16>> {
    let b = bytes?;
    if is_f16 {
        assert_eq!(b.len(), n * 2, "f16 scale len");
        Some(
            b.chunks_exact(2)
                .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
        )
    } else {
        let words: Vec<u16> = b
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(words.len() * 16, n, "packed sign len");
        Some(super::signs::unpack_sign_bitfield(&words))
    }
}

fn expand_bias(bytes: Option<&[u8]>, n: usize) -> Option<Vec<f16>> {
    let b = bytes?;
    assert_eq!(b.len(), n * 2, "bias len");
    Some(
        b.chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
    )
}

/// Naive forward: `x` is row-major `(rows, in_features)` f16.
///
/// Returns `(rows, out_features)` f16.
pub fn linear_forward_reconstruct(layer: &Exl3Linear<'_>, x: &[f16]) -> Vec<f16> {
    let inn = layer.in_features();
    let out = layer.out_features();
    assert!(inn > 0 && out > 0 && inn % 16 == 0 && out % 16 == 0);
    assert_eq!(x.len() % inn, 0, "x rows must be multiple of in_features");
    let rows = x.len() / inn;

    let suh = expand_scales(layer.suh, inn, layer.suh_is_f16);
    let svh = expand_scales(layer.svh, out, layer.svh_is_f16);

    // Input Hadamard + suh → xh (rows, in)
    let xh = had_r_128(x, rows, inn, suh.as_deref(), None);

    let mut y = vec![f16::ZERO; rows * out];

    // Stream output in 128-col slabs (one right-Hadamard block).
    assert_eq!(out % 128, 0, "out_features must be multiple of 128");
    for n0 in (0..out).step_by(128) {
        let n_count = 128;
        let w = reconstruct_inner_slab(&layer.trellis, layer.mode, n0, n_count)
            .expect("slab reconstruct");

        // slab[r, c] = sum_k xh[r, k] * w[k, c]  (f32 accumulate)
        let mut slab = vec![0f32; rows * n_count];
        for tk in 0..layer.trellis.in_tiles {
            let k_off = tk * 16;
            let xh_row_off = |r: usize| -> &[f16] { &xh[r * inn + k_off..r * inn + k_off + 16] };
            for tn_local in 0..8 {
                let col_local = tn_local * 16;
                // w tile rows for this tk: slab of w is (in, 128), tile columns
                // are already placed at col_local within the slab.
                for r in 0..rows {
                    let xr = xh_row_off(r);
                    let dst = r * n_count + col_local;
                    for oc in 0..16 {
                        let mut acc = 0f32;
                        for ic in 0..16 {
                            // w is (inn, 128) row-major; row = k_off+ic, col = col_local+oc
                            acc += xr[ic].to_f32()
                                * w[(k_off + ic) * n_count + col_local + oc].to_f32();
                        }
                        slab[dst + oc] += acc;
                    }
                }
            }
        }

        // Output Hadamard (right) + svh on this slab.
        let slab_f16: Vec<f16> = slab.iter().map(|&v| f16::from_f32(v)).collect();
        let svh_block: Option<&[f16]> = svh.as_ref().map(|s| &s[n0..n0 + 128]);
        let y_block = had_r_128(&slab_f16, rows, 128, None, svh_block);
        for r in 0..rows {
            let dst = r * out + n0;
            y[dst..dst + 128].copy_from_slice(&y_block[r * 128..(r + 1) * 128]);
        }
    }

    if let Some(bias) = expand_bias(layer.bias, out) {
        for r in 0..rows {
            for c in 0..out {
                let v = y[r * out + c].to_f32() + bias[c].to_f32();
                y[r * out + c] = f16::from_f32(v);
            }
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exl3::reconstruct::reconstruct_public_weights;
    use crate::exl3::trellis::pack_trellis_tile;

    fn synth_layer(
        k: u32,
        in_tiles: usize,
        out_tiles: usize,
        seed: u32,
    ) -> (Vec<u16>, Vec<u8>, Vec<u8>, TrellisView<'static>) {
        let packed_size = 256 * k as usize / 16;
        let mut data = Vec::with_capacity(in_tiles * out_tiles * packed_size);
        let mut state = seed;
        for _ in 0..in_tiles * out_tiles {
            let mask = ((1u32 << k) - 1) as u16;
            let mut fresh = [0u16; 256];
            for f in fresh.iter_mut() {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                *f = (state >> 16) as u16 & mask;
            }
            data.extend_from_slice(&pack_trellis_tile(&fresh, k).unwrap());
        }
        let inn = in_tiles * 16;
        let out = out_tiles * 16;
        // Deterministic ±1 f16 scales.
        let mut st = seed ^ 0xA5A5;
        let make_scale = |n: usize, st: &mut u32| -> Vec<u8> {
            let mut v = Vec::with_capacity(n * 2);
            for _ in 0..n {
                *st = st.wrapping_mul(1664525).wrapping_add(1013904223);
                let bits = if (*st >> 31) & 1 == 1 {
                    0xBC00u16
                } else {
                    0x3C00
                };
                v.extend_from_slice(&bits.to_le_bytes());
            }
            v
        };
        let suh = make_scale(inn, &mut st);
        let svh = make_scale(out, &mut st);
        let leaked: &'static [u16] = Box::leak(data.into_boxed_slice());
        let view = TrellisView::new(leaked, in_tiles, out_tiles, packed_size).unwrap();
        // Leak scales too for 'static in this helper — actually return owned and
        // only leak trellis; scales stay in caller.
        let _ = (&suh, &svh);
        (leaked.to_vec(), suh, svh, view)
    }

    fn f16_le(n: usize, seed: u32) -> Vec<f16> {
        let mut st = seed;
        (0..n)
            .map(|_| {
                st = st.wrapping_mul(1664525).wrapping_add(1013904223);
                f16::from_f32(((st >> 16) as i16 as f32) / 512.0)
            })
            .collect()
    }

    #[test]
    fn forward_matches_stepwise_full_reconstruct() {
        let k = 4;
        let in_tiles = 8; // 128 in
        let out_tiles = 8; // 128 out
        let (_td, suh, svh, view) = synth_layer(k, in_tiles, out_tiles, 99);
        let inn = view.in_features();
        let out = view.out_features();
        let layer = Exl3Linear {
            trellis: view,
            suh: Some(&suh),
            svh: Some(&svh),
            suh_is_f16: true,
            svh_is_f16: true,
            mode: CodebookMode::Mcg,
            bias: None,
        };
        let x = f16_le(inn * 3, 1234); // 3 rows
        let y_stream = linear_forward_reconstruct(&layer, &x);

        // Stepwise reference (same algorithm as streaming, full slab = whole matrix):
        // y = had_r(had_r(x, pre=suh) @ W_inner, post=svh)
        let w_inner =
            crate::exl3::reconstruct::reconstruct_inner(&view, CodebookMode::Mcg).unwrap();
        let su: Vec<f16> = suh
            .chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let sv: Vec<f16> = svh
            .chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let rows = 3;
        let xh = had_r_128(&x, rows, inn, Some(&su), None);
        let mut y_mid = vec![f16::ZERO; rows * out];
        for r in 0..rows {
            for c in 0..out {
                let mut acc = 0f32;
                for k_i in 0..inn {
                    acc += xh[r * inn + k_i].to_f32() * w_inner[k_i * out + c].to_f32();
                }
                y_mid[r * out + c] = f16::from_f32(acc);
            }
        }
        let y_step = had_r_128(&y_mid, rows, out, None, Some(&sv));

        // Streaming and stepwise full reconstruct must match exactly (same ops,
        // only tiling order of the inner matmul differs → f32 assoc → tiny slack).
        let mut max_rel = 0f32;
        let mut checked = 0;
        for i in 0..y_stream.len() {
            let a = y_stream[i].to_f32();
            let b = y_step[i].to_f32();
            if a.abs() < 1e-3 && b.abs() < 1e-3 {
                continue;
            }
            let denom = a.abs().max(b.abs()).max(1e-6);
            let rel = (a - b).abs() / denom;
            if rel > max_rel {
                max_rel = rel;
            }
            checked += 1;
        }
        assert!(checked > 50, "too few significant outputs: {checked}");
        assert!(
            max_rel < 1e-3,
            "stream vs stepwise max relative error {max_rel}"
        );
    }

    #[test]
    fn forward_matches_dense_public_weight() {
        let k = 4;
        let in_tiles = 8; // 128 in
        let out_tiles = 8; // 128 out
        let (_td, suh, svh, view) = synth_layer(k, in_tiles, out_tiles, 99);
        let inn = view.in_features();
        let out = view.out_features();
        let layer = Exl3Linear {
            trellis: view,
            suh: Some(&suh),
            svh: Some(&svh),
            suh_is_f16: true,
            svh_is_f16: true,
            mode: CodebookMode::Mcg,
            bias: None,
        };
        let x = f16_le(inn * 3, 1234);
        let y_stream = linear_forward_reconstruct(&layer, &x);

        // Dense path: public weight then x @ W.
        let w = reconstruct_public_weights(
            &view,
            Some(&suh),
            Some(&svh),
            CodebookMode::Mcg,
            true,
            true,
        )
        .unwrap();
        let rows = 3;
        let mut y_dense = vec![f16::ZERO; rows * out];
        for r in 0..rows {
            for c in 0..out {
                let mut acc = 0f32;
                for k_i in 0..inn {
                    acc += x[r * inn + k_i].to_f32() * w[k_i * out + c].to_f32();
                }
                y_dense[r * out + c] = f16::from_f32(acc);
            }
        }

        // Stream applies right-had after an f16 matmul cast; dense folds had into
        // W (f16). Inherent intermediate-rounding gap — require loose agreement.
        let mut max_rel = 0f32;
        let mut checked = 0;
        for i in 0..y_stream.len() {
            let a = y_stream[i].to_f32();
            let b = y_dense[i].to_f32();
            if a.abs() < 1e-3 && b.abs() < 1e-3 {
                continue;
            }
            let denom = a.abs().max(b.abs()).max(1e-6);
            let rel = (a - b).abs() / denom;
            if rel > max_rel {
                max_rel = rel;
            }
            checked += 1;
        }
        assert!(checked > 50, "too few significant outputs: {checked}");
        assert!(
            max_rel < 0.15,
            "stream vs dense max relative error {max_rel}"
        );
    }

    #[test]
    fn forward_single_row_deterministic() {
        let k = 3;
        let (_td, suh, svh, view) = synth_layer(k, 8, 8, 5);
        let layer = Exl3Linear {
            trellis: view,
            suh: Some(&suh),
            svh: Some(&svh),
            suh_is_f16: true,
            svh_is_f16: true,
            mode: CodebookMode::Mcg,
            bias: None,
        };
        let x = f16_le(view.in_features(), 7);
        let y1 = linear_forward_reconstruct(&layer, &x);
        let y2 = linear_forward_reconstruct(&layer, &x);
        assert_eq!(y1, y2, "deterministic");
        assert_eq!(y1.len(), view.out_features());
        for v in &y1 {
            assert!(v.to_f32().is_finite());
        }
    }

    #[test]
    fn forward_rejects_bad_x_length() {
        let (_td, suh, svh, view) = synth_layer(3, 8, 8, 1);
        let layer = Exl3Linear {
            trellis: view,
            suh: Some(&suh),
            svh: Some(&svh),
            suh_is_f16: true,
            svh_is_f16: true,
            mode: CodebookMode::Mcg,
            bias: None,
        };
        let x = f16_le(view.in_features() + 1, 1);
        let r = std::panic::catch_unwind(|| linear_forward_reconstruct(&layer, &x));
        assert!(r.is_err());
    }
}
