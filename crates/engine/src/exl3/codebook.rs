//! EXL3 procedural codebook (3-instruction decode) for finite fp16 samples.
//!
//! Artifact uses `codebook: mcg` with `mul1` absent → [`CodebookMode::Mcg`].

use half::f16;

const LOP3_IMM_6A: u32 = 0x6A;
const MCG_MULT: u32 = 0xCBAC_1FED;
const MUL1_MULT: u32 = 0x83DC_D12D;
const DEFAULT_MULT: u32 = 89_226_354;
const DEFAULT_ADD: u32 = 64_248_484;
const K_INV_BITS: u16 = 0x1EEE;
const K_BIAS_BITS: u16 = 0xC931;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodebookMode {
    Default = 0,
    Mcg = 1,
    Mul1 = 2,
}

/// LOP3.b32 with PTX truth-table immediate; `0x6A` = `c ^ (a & b)`.
fn lop3_b32(a: u32, b: u32, c: u32, imm: u32) -> u32 {
    let mut out = 0u32;
    for bit in 0..32 {
        let ab = (a >> bit) & 1;
        let bb = (b >> bit) & 1;
        let cb = (c >> bit) & 1;
        let idx = (ab << 2) | (bb << 1) | cb;
        let ob = (imm >> idx) & 1;
        out |= ob << bit;
    }
    out
}

/// Exact IEEE binary16 add for finite operands (round-to-nearest-even).
///
/// Both halves are exact in f64; `f16::from_f64` is correctly rounded — matches
/// CUDA `__hadd` on finite values without double-rounding through f32.
fn half2_add_u32(x: u32) -> f16 {
    let lo = f16::from_bits((x & 0xFFFF) as u16);
    let hi = f16::from_bits(((x >> 16) & 0xFFFF) as u16);
    f16::from_f64(lo.to_f64() + hi.to_f64())
}

/// PTX `vabsdiff4.u32.u32.u32.add` — sum of per-byte |a-b| plus accumulator.
fn vabsdiff4_add(x: u32, y: u32, acc: u32) -> u32 {
    let mut s = acc;
    for lane in 0..4 {
        let ai = ((x >> (8 * lane)) & 0xFF) as i32;
        let bi = ((y >> (8 * lane)) & 0xFF) as i32;
        s = s.wrapping_add(ai.abs_diff(bi));
    }
    s
}

/// Map a 16-bit trellis codeword to one fp16 weight sample.
pub fn decode_3inst(codeword: u16, mode: CodebookMode) -> f16 {
    let mut x = codeword as u32;
    match mode {
        CodebookMode::Default => {
            x = (x as u64 * DEFAULT_MULT as u64 + DEFAULT_ADD as u64) as u32;
            x = lop3_b32(x, 0x8FFF_8FFF, 0x3B60_3B60, LOP3_IMM_6A);
            half2_add_u32(x)
        }
        CodebookMode::Mcg => {
            x = x.wrapping_mul(MCG_MULT);
            x = lop3_b32(x, 0x8FFF_8FFF, 0x3B60_3B60, LOP3_IMM_6A);
            half2_add_u32(x)
        }
        CodebookMode::Mul1 => {
            x = x.wrapping_mul(MUL1_MULT);
            let summed = vabsdiff4_add(x, 0, 0x6400);
            let h = f16::from_bits((summed & 0xFFFF) as u16);
            let k_inv = f16::from_bits(K_INV_BITS);
            let k_bias = f16::from_bits(K_BIAS_BITS);
            // Matches reference: f32 mul + add then cast to f16 (MUL1 only).
            f16::from_f32(h.to_f32() * k_inv.to_f32() + k_bias.to_f32())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference goldens (MCG): codeword → f16 bit pattern.
    const MCG_CWS: [u32; 10] = [0, 1, 2, 32767, 32768, 65535, 4660, 43981, 255, 65280];
    const MCG_BITS: [u16; 10] = [
        0x3f60, 0x304e, 0xba13, 0x4172, 0xb915, 0x3acd, 0x3e0f, 0x35e7, 0xbd26, 0x3c86,
    ];

    #[test]
    fn mcg_known_codewords() {
        for (cw, want) in MCG_CWS.iter().zip(MCG_BITS.iter()) {
            let got = decode_3inst(*cw as u16, CodebookMode::Mcg);
            assert_eq!(
                got.to_bits(),
                *want,
                "mcg cw={cw:#x} got={:#x} want={:#x}",
                got.to_bits(),
                want
            );
        }
        assert_eq!(decode_3inst(0, CodebookMode::Mcg).to_bits(), 0x3f60);
        assert_eq!(decode_3inst(1, CodebookMode::Mcg).to_bits(), 0x304e);
        assert_eq!(decode_3inst(0xffff, CodebookMode::Mcg).to_bits(), 0x3acd);
        assert_eq!(decode_3inst(0x1234, CodebookMode::Mcg).to_bits(), 0x3e0f);
    }

    #[test]
    fn default_mode_known_codewords() {
        let cases: [(u16, u16); 4] = [(0, 0x3a25), (1, 0xbb5b), (0xffff, 0xb110), (0x1234, 0xc066)];
        for (cw, want) in cases {
            assert_eq!(
                decode_3inst(cw, CodebookMode::Default).to_bits(),
                want,
                "default cw={cw:#x}"
            );
        }
    }

    #[test]
    fn mul1_mode_known_codewords() {
        let cases: [(u16, u16); 4] = [(0, 0xc2e8), (1, 0x3921), (0xffff, 0xb936), (0x1234, 0xb89e)];
        for (cw, want) in cases {
            assert_eq!(
                decode_3inst(cw, CodebookMode::Mul1).to_bits(),
                want,
                "mul1 cw={cw:#x}"
            );
        }
    }

    #[test]
    fn lop3_imm_6a_is_c_xor_a_and_b() {
        // Spot-check all 8 input combos for one bit.
        for a in 0..2u32 {
            for b in 0..2u32 {
                for c in 0..2u32 {
                    let got = lop3_b32(a, b, c, LOP3_IMM_6A) & 1;
                    let want = (c ^ (a & b)) & 1;
                    assert_eq!(got, want, "a={a} b={b} c={c}");
                }
            }
        }
    }

    #[test]
    fn half2_add_matches_f64_path() {
        // 0x3C00 (1.0) + 0x3C00 = 0x4000 (2.0)
        assert_eq!(half2_add_u32(0x3C00_3C00).to_bits(), 0x4000);
        // lo=1.0, hi=0 → 1.0
        assert_eq!(half2_add_u32(0x3C00_0000).to_bits(), 0x3C00);
        // lo=2.0, hi=1.0 → 3.0
        assert_eq!(half2_add_u32(0x3C00_4000).to_bits(), 0x4200);
    }

    #[test]
    fn vabsdiff4_basic() {
        assert_eq!(vabsdiff4_add(0, 0, 0x6400), 0x6400);
        // One byte differs by 1 in lane0.
        assert_eq!(vabsdiff4_add(0x01, 0x00, 0), 1);
        // All four lanes differ by 1.
        assert_eq!(vabsdiff4_add(0x0101_0101, 0, 0), 4);
    }

    #[test]
    fn mcg_output_finite_for_all_u16_spots() {
        for &cw in &MCG_CWS {
            let v = decode_3inst(cw as u16, CodebookMode::Mcg);
            assert!(v.to_f32().is_finite());
        }
        // Dense scan: every codeword must stay finite under MCG+LOP3.
        for cw in (0..=u16::MAX).step_by(17) {
            let v = decode_3inst(cw, CodebookMode::Mcg);
            assert!(v.to_f32().is_finite(), "cw={cw:#x}");
        }
    }
}
