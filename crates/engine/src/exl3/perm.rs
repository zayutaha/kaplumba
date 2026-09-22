//! Tensor-core 16×16 tile element permutation (kernel order ↔ row-major).

/// Forward permutation: kernel-order index → row-major source index.
pub fn tensor_core_perm() -> [i32; 256] {
    let mut perm = [0i32; 256];
    for t in 0..32 {
        let r0 = (t % 4) * 2;
        let r1 = r0 + 1;
        let r2 = r0 + 8;
        let r3 = r0 + 9;
        let c0 = t / 4;
        let c1 = c0 + 8;
        let base = t * 8;
        perm[base + 0] = (r0 * 16 + c0) as i32;
        perm[base + 1] = (r1 * 16 + c0) as i32;
        perm[base + 2] = (r2 * 16 + c0) as i32;
        perm[base + 3] = (r3 * 16 + c0) as i32;
        perm[base + 4] = (r0 * 16 + c1) as i32;
        perm[base + 5] = (r1 * 16 + c1) as i32;
        perm[base + 6] = (r2 * 16 + c1) as i32;
        perm[base + 7] = (r3 * 16 + c1) as i32;
    }
    perm
}

pub fn tensor_core_perm_inverse() -> [i32; 256] {
    let perm = tensor_core_perm();
    let mut inv = [0i32; 256];
    for (i, &p) in perm.iter().enumerate() {
        inv[p as usize] = i as i32;
    }
    inv
}

/// Reorder 256 kernel-order samples into a row-major 16×16 block.
pub fn kernel_order_to_row_major(values: &[u16; 256]) -> [[u16; 16]; 16] {
    let inv = tensor_core_perm_inverse();
    let mut out = [[0u16; 16]; 16];
    for i in 0..256 {
        let v = values[inv[i] as usize];
        out[i / 16][i % 16] = v;
    }
    out
}

/// f16 variant of [`kernel_order_to_row_major`].
pub fn kernel_order_to_row_major_f16(values: &[half::f16; 256]) -> [[half::f16; 16]; 16] {
    let inv = tensor_core_perm_inverse();
    let mut out = [[half::f16::ZERO; 16]; 16];
    for i in 0..256 {
        let v = values[inv[i] as usize];
        out[i / 16][i % 16] = v;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perm_head_matches_reference() {
        let perm = tensor_core_perm();
        let want = [
            0, 16, 128, 144, 8, 24, 136, 152, 32, 48, 160, 176, 40, 56, 168, 184,
        ];
        assert_eq!(perm[..16], want);
    }

    #[test]
    fn perm_inv_head_matches_reference() {
        let inv = tensor_core_perm_inverse();
        let want = [
            0, 32, 64, 96, 128, 160, 192, 224, 4, 36, 68, 100, 132, 164, 196, 228,
        ];
        assert_eq!(inv[..16], want);
    }

    #[test]
    fn perm_is_bijection() {
        let perm = tensor_core_perm();
        let mut seen = [false; 256];
        for &p in &perm {
            assert!((0..256).contains(&p));
            assert!(!seen[p as usize], "duplicate {p}");
            seen[p as usize] = true;
        }
        let inv = tensor_core_perm_inverse();
        for i in 0..256 {
            assert_eq!(perm[inv[i] as usize], i as i32);
            assert_eq!(inv[perm[i] as usize], i as i32);
        }
    }

    #[test]
    fn kernel_order_row0_matches_reference() {
        let mut vals = [0u16; 256];
        for (i, v) in vals.iter_mut().enumerate() {
            *v = i as u16;
        }
        let rm = kernel_order_to_row_major(&vals);
        let want = [
            0u16, 32, 64, 96, 128, 160, 192, 224, 4, 36, 68, 100, 132, 164, 196, 228,
        ];
        assert_eq!(rm[0], want);
        // Row 1 is row0 shifted by +1 in the source indexing (r0+1 pattern).
        let want1 = [
            1u16, 33, 65, 97, 129, 161, 193, 225, 5, 37, 69, 101, 133, 165, 197, 229,
        ];
        assert_eq!(rm[1], want1);
    }
}
