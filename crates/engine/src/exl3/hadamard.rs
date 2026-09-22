//! Sylvester Hadamard + right-side 128-point transform used by EXL3.

use half::f16;

pub const HAD_DIM: usize = 128;
/// 1/sqrt(128) as the exact f32 bit pattern (matches NumPy `float32(1/sqrt(128))`).
pub const HAD_SCALE: f32 = f32::from_bits(0x3DB5_04F3);

/// Sylvester construction of order `n` (power of two), scaled by `1/sqrt(n)`.
pub fn hadamard_128() -> [[f32; HAD_DIM]; HAD_DIM] {
    let mut h = [[0f32; HAD_DIM]; HAD_DIM];
    h[0][0] = 1.0;
    let mut n = 1usize;
    while n < HAD_DIM {
        let half = n;
        for i in 0..half {
            for j in 0..half {
                let v = h[i][j];
                h[i][j + half] = v;
                h[i + half][j] = v;
                h[i + half][j + half] = -v;
            }
        }
        n *= 2;
    }
    for v in h.iter_mut().flatten() {
        *v *= HAD_SCALE;
    }
    h
}

/// Right-side 128-point Hadamard on each column block of `x` (rows × features).
///
/// `pre_scale` multiplies each block before the transform (suh);
/// `post_scale` multiplies after (svh). Input/output are f16 (inner math f32).
pub fn had_r_128(
    x: &[f16],
    rows: usize,
    features: usize,
    pre_scale: Option<&[f16]>,
    post_scale: Option<&[f16]>,
) -> Vec<f16> {
    assert_eq!(x.len(), rows * features, "x shape");
    assert_eq!(features % HAD_DIM, 0, "features % 128");
    if let Some(s) = pre_scale {
        assert_eq!(s.len(), features, "pre_scale len");
    }
    if let Some(s) = post_scale {
        assert_eq!(s.len(), features, "post_scale len");
    }

    let had = hadamard_128();
    let mut out = vec![f16::ZERO; rows * features];
    let mut block = [0f32; HAD_DIM];
    let mut acc = [0f32; HAD_DIM];

    for r in 0..rows {
        for j0 in (0..features).step_by(HAD_DIM) {
            let row_off = r * features + j0;
            for (t, b) in block.iter_mut().enumerate() {
                let mut v = x[row_off + t].to_f32();
                if let Some(pre) = pre_scale {
                    v *= pre[j0 + t].to_f32();
                }
                *b = v;
            }
            // y = block @ H  (same accumulation order as reference matmul)
            for (k, &bv) in block.iter().enumerate() {
                if bv == 0.0 {
                    continue;
                }
                let hk = &had[k];
                for (o, a) in acc.iter_mut().enumerate() {
                    *a += bv * hk[o];
                }
            }
            for (t, a) in acc.iter().enumerate() {
                let mut v = *a;
                if let Some(post) = post_scale {
                    v *= post[j0 + t].to_f32();
                }
                out[row_off + t] = f16::from_f32(v);
            }
            acc.fill(0.0);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sylvester_scale_and_involution() {
        let h = hadamard_128();
        assert!((h[0][0] - HAD_SCALE).abs() < 1e-7);
        assert!((h[1][0] - HAD_SCALE).abs() < 1e-7);
        assert!((h[1][1] + HAD_SCALE).abs() < 1e-7);
        // H H = I for orthonormalized Sylvester (H = S/sqrt(n), S S^T = n I).
        // Check diagonal and a few off-diagonals in f32.
        for i in 0..HAD_DIM {
            let mut s = 0f32;
            for k in 0..HAD_DIM {
                s += h[i][k] * h[i][k];
            }
            assert!((s - 1.0).abs() < 1e-4, "row {i} norm {s}");
        }
        for i in 0..8 {
            for j in 0..8 {
                if i == j {
                    continue;
                }
                let mut s = 0f32;
                for k in 0..HAD_DIM {
                    s += h[i][k] * h[j][k];
                }
                assert!(s.abs() < 1e-4, "dot {i},{j} = {s}");
            }
        }
    }

    /// Reference golden: x = 0..127 f16, pre[i] = −1 if i even else +1,
    /// post[i] = −1 if i%4==0 else +1.
    ///
    /// Only the large-magnitude lanes are bit-exact vs Python BLAS; near-zero
    /// lanes are pure cancellation and depend on accumulation order (checked
    /// with an absolute bound instead).
    #[test]
    fn had_r_128_reference_golden() {
        let x: Vec<f16> = (0..128).map(|i| f16::from_f32(i as f32)).collect();
        let pre: Vec<f16> = (0..128)
            .map(|i| if i % 2 == 0 { f16::NEG_ONE } else { f16::ONE })
            .collect();
        let post: Vec<f16> = (0..128)
            .map(|i| if i % 4 == 0 { f16::NEG_ONE } else { f16::ONE })
            .collect();
        let y = had_r_128(&x, 1, 128, Some(&pre), Some(&post));
        assert_eq!(y.len(), 128);

        // Well-conditioned lanes from the Python reference (|y| > 0.01).
        // idx → f16 bits
        let large: [(usize, u16); 8] = [
            (0, 0xc5a8),  // -5.656891
            (1, 0xe19d),  // -718.42053
            (3, 0x49a8),  // 11.313705
            (5, 0x4da8),  // 22.627415
            (9, 0x51a8),  // 45.254833
            (17, 0x55a8), // 90.50966
            (33, 0x59a8), // 181.01932
            (65, 0x5da8), // 362.03867
        ];
        for (i, w) in large {
            let g = y[i].to_bits();
            let d = g.abs_diff(w);
            assert!(d <= 1, "y[{i}] got={g:#06x} want={w:#06x}");
        }

        // Full-vector absolute check vs reference f32 values for the same set.
        let ref_large: [(usize, f32); 8] = [
            (0, -5.656891),
            (1, -718.42053),
            (3, 11.313705),
            (5, 22.627415),
            (9, 45.254833),
            (17, 90.50966),
            (33, 181.01932),
            (65, 362.03867),
        ];
        for (i, w) in ref_large {
            let g = y[i].to_f32();
            assert!(
                (g - w).abs() <= (w.abs() * 1e-3).max(1e-3),
                "y[{i}] got={g} want={w}"
            );
        }

        // Cancellation lanes: true values are ~0; any correct order stays tiny.
        for i in 0..128 {
            if ref_large.iter().any(|(j, _)| *j == i) {
                continue;
            }
            let g = y[i].to_f32();
            // Bound by the natural noise floor of summing 128 f32 terms scaled by HAD.
            assert!(g.abs() < 1.0, "y[{i}] cancellation blew up: {g}");
        }

        // All outputs finite (subnormals OK).
        for v in &y {
            assert!(v.to_f32().is_finite());
        }
    }

    #[test]
    fn had_r_rejects_bad_features() {
        let x = vec![f16::ONE; 10];
        let r = std::panic::catch_unwind(|| had_r_128(&x, 1, 10, None, None));
        assert!(r.is_err());
    }
}
