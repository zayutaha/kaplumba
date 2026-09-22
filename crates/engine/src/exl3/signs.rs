//! Packed sign bitfields (suh/svh) → per-element ±1 f16 scales.

use half::f16;

/// Expand packed u16 sign words to `groups * 16` ±1 f16 values.
/// Bit set → −1.0; bit clear → +1.0. One word per 16 channels.
pub fn unpack_sign_bitfield(bitfield: &[u16]) -> Vec<f16> {
    let mut out = Vec::with_capacity(bitfield.len() * 16);
    for &bf in bitfield {
        for b in 0..16 {
            let neg = (bf >> b) & 1 == 1;
            out.push(if neg { f16::NEG_ONE } else { f16::ONE });
        }
    }
    out
}

/// Already-unpacked f16 scales pass through; packed i16/u16 bitfields expand.
///
/// Artifact stores `suh`/`svh` as raw f16 (shape = features), so the common
/// path is the pass-through branch.
pub fn unpack_signs_or_pass(
    values: Option<&[u8]>,
    n_elems: usize,
    is_f16: bool,
) -> Option<Vec<f16>> {
    let bytes = values?;
    if is_f16 {
        assert_eq!(bytes.len(), n_elems * 2, "f16 sign tensor size");
        let mut out = Vec::with_capacity(n_elems);
        for i in 0..n_elems {
            out.push(f16::from_bits(u16::from_le_bytes([
                bytes[2 * i],
                bytes[2 * i + 1],
            ])));
        }
        Some(out)
    } else {
        // Packed sign bitfield: 2 bytes per 16 channels.
        let n_words = bytes.len() / 2;
        assert_eq!(n_words * 16, n_elems, "packed sign length");
        let mut words = Vec::with_capacity(n_words);
        for i in 0..n_words {
            words.push(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]));
        }
        Some(unpack_sign_bitfield(&words))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: words [0x0001, 0x8000, 0xFFFF, 0x0000] → 64 scales.
    #[test]
    fn bitfield_reference_fixture() {
        let bf = [0x0001u16, 0x8000, 0xFFFF, 0x0000];
        let got = unpack_sign_bitfield(&bf);
        assert_eq!(got.len(), 64);
        // word0 bit0 set → first −1; rest of word0 +1
        assert_eq!(got[0], f16::NEG_ONE);
        for i in 1..16 {
            assert_eq!(got[i], f16::ONE, "w0 i={i}");
        }
        // word1 bit15 set → index 16+15 = 31
        for i in 16..31 {
            assert_eq!(got[i], f16::ONE, "w1 i={i}");
        }
        assert_eq!(got[31], f16::NEG_ONE);
        // word2 all ones → 32..48 all −1
        for i in 32..48 {
            assert_eq!(got[i], f16::NEG_ONE, "w2 i={i}");
        }
        // word3 clear → 48..64 all +1
        for i in 48..64 {
            assert_eq!(got[i], f16::ONE, "w3 i={i}");
        }
        // Golden bit patterns from reference: −1 = 0xBC00, +1 = 0x3C00
        assert_eq!(got[0].to_bits(), 0xBC00);
        assert_eq!(got[1].to_bits(), 0x3C00);
        assert_eq!(got[32].to_bits(), 0xBC00);
        assert_eq!(got[48].to_bits(), 0x3C00);
    }

    #[test]
    fn pass_through_f16() {
        let raw = [0x0000u8, 0x3C, 0x00, 0xBC]; // +1.0, −1.0 LE
        let got = unpack_signs_or_pass(Some(&raw), 2, true).unwrap();
        assert_eq!(got[0], f16::ONE);
        assert_eq!(got[1], f16::NEG_ONE);
    }

    #[test]
    fn pass_none() {
        assert!(unpack_signs_or_pass(None, 0, true).is_none());
    }
}
