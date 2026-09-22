//! EXL3 trellis pack/unpack for one 16×16 tile (bit-shift trellis).
//!
//! Packed layout matches on-disk tiles: 16 spans × k-bit values, then a
//! byte-swap of each u16 within every u32 pair (SWAP16).

const TILE: usize = 256;
const SPANS: usize = 16;
const SPAN_LEN: usize = TILE / SPANS; // 16

/// Packed u16 length for one tile at bit-width `k` (must be 1..=8).
pub fn packed_tile_len(k: u32) -> usize {
    debug_assert!((1..=8).contains(&k));
    (TILE * k as usize) / 16
}

/// Pack 256 K-bit codewords into the on-disk bitstream tile.
///
/// `encoded[i]` uses only the low `k` bits. Returns `256*k/16` u16 words
/// with SWAP16 already applied (artifact form).
pub fn pack_trellis_tile(encoded: &[u16], k: u32) -> Result<Vec<u16>, TrellisError> {
    if k < 1 || k > 8 {
        return Err(TrellisError::BadK(k));
    }
    if encoded.len() != TILE {
        return Err(TrellisError::BadLen {
            expected: TILE,
            got: encoded.len(),
        });
    }
    let packed_size = packed_tile_len(k);
    let mut s_packed = vec![0u16; packed_size];
    let mask = (1u32 << k) - 1;

    for t in 0..SPANS {
        let mut i = SPAN_LEN * t;
        let mut j = k as usize * t;
        let mut buf: u32 = 0;
        let mut bit_pos: i32 = 32;
        for _ in 0..SPAN_LEN {
            let v = (encoded[i] as u32) & mask;
            bit_pos -= k as i32;
            buf |= v << bit_pos;
            if bit_pos <= 16 {
                s_packed[j] = (buf >> 16) as u16;
                buf = buf << 16;
                bit_pos += 16;
                j += 1;
            }
            i += 1;
        }
    }

    // SWAP16 per u32 (swap adjacent u16 words).
    for pair in s_packed.chunks_exact_mut(2) {
        pair.swap(0, 1);
    }
    Ok(s_packed)
}

/// Unpack one packed tile to 256 full 16-bit sliding-window codewords.
///
/// `packed` must already be in on-disk form (SWAP16 applied). The funnel-shift
/// loop matches `exl3_dq.cuh` / the CPU reference: each codeword is the 16-bit
/// window ending at that weight's K fresh bits, **not** the low K bits alone.
pub fn unpack_trellis_tile(packed: &[u16], k: u32) -> Result<[u16; TILE], TrellisError> {
    if k < 1 || k > 8 {
        return Err(TrellisError::BadK(k));
    }
    let expected = packed_tile_len(k);
    if packed.len() != expected {
        return Err(TrellisError::BadLen {
            expected,
            got: packed.len(),
        });
    }
    if packed.len() % 2 != 0 {
        return Err(TrellisError::OddPacked(packed.len()));
    }

    let mut words = vec![0u32; packed.len() / 2];
    for (w, pair) in words.iter_mut().zip(packed.chunks_exact(2)) {
        *w = (pair[0] as u32) | ((pair[1] as u32) << 16);
    }
    let mask_words = (k as usize * TILE) / 32;
    debug_assert!(mask_words > 0);

    let mut decoded = [0u32; 128];
    for t in 0..128 {
        let b0 = (t as i64) * 2 * (k as i64) + (k as i64) - 16 + (TILE as i64) * (k as i64);
        let b1 = b0 + k as i64;
        let b2 = b1 + 16;
        let i0 = b0 / 32;
        let i1 = (b2 - 1) / 32;
        let s1 = (i1 + 1) * 32 - b2;
        let a = words[(i0 as usize).rem_euclid(mask_words)];
        let b = words[(i1 as usize).rem_euclid(mask_words)];
        let w1 = funnel_shift_r(b, a, s1 as u32);
        let w0 = (w1 >> k) & 0xFFFF;
        let w1 = w1 & 0xFFFF;
        decoded[t] = (w1 << 16) | w0;
    }

    let mut out = [0u16; TILE];
    for (i, d) in decoded.iter().enumerate() {
        out[2 * i] = (d & 0xFFFF) as u16;
        out[2 * i + 1] = (d >> 16) as u16;
    }
    Ok(out)
}

/// 64-bit merge `(a << 32) | b`, then logical shift right, keep low 32.
fn funnel_shift_r(b: u32, a: u32, shift: u32) -> u32 {
    let merged = ((a as u64) << 32) | (b as u64);
    (merged >> shift) as u32
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TrellisError {
    #[error("trellis K must be in 1..=8, got {0}")]
    BadK(u32),
    #[error("trellis length mismatch: expected {expected}, got {got}")]
    BadLen { expected: usize, got: usize },
    #[error("packed trellis length must be even, got {0}")]
    OddPacked(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    // Goldens from reference CPU pack/unpack (fixtures under /tmp/kaplumba_t13_fixtures
    // when regenerated; values baked in so tests do not depend on /tmp).
    const PACK_K3: [u16; 48] = [
        21260, 9199, 33797, 26669, 26982, 8020, 39698, 53275, 55642, 10335, 13521, 58930, 53498,
        8523, 25703, 60085, 63719, 18978, 45544, 5357, 61983, 13544, 54447, 21982, 55843, 27939,
        19985, 15241, 37181, 29610, 24118, 64700, 31716, 23570, 12786, 50715, 28984, 42077, 64495,
        40344, 54966, 44430, 1026, 62007, 36315, 40048, 17470, 29593,
    ];
    const CW_K3: [u16; 256] = [
        8689, 3976, 31815, 57918, 4599, 36797, 32234, 61267, 31384, 54467, 42520, 12486, 34356,
        12704, 36101, 26669, 16748, 2913, 23304, 55360, 49666, 4116, 32931, 1311, 10490, 18389,
        16040, 62790, 43572, 20901, 36140, 26982, 19254, 22964, 52640, 27905, 26637, 16494, 883,
        7067, 56536, 59076, 13860, 45346, 35092, 18593, 17675, 10335, 17150, 6134, 49074, 64917,
        60589, 25963, 11100, 23270, 55089, 47500, 52324, 25379, 6426, 51411, 18074, 13521, 42633,
        13384, 41538, 4628, 37029, 34095, 10618, 19408, 24199, 62526, 41461, 4014, 32117, 60330,
        23894, 60085, 21931, 44377, 27336, 22086, 45619, 37277, 36073, 26442, 14929, 53896, 37957,
        41519, 4476, 35811, 24348, 63719, 51000, 14789, 52777, 29006, 35446, 21430, 40374, 60849,
        28047, 27770, 25552, 7811, 62490, 41171, 1693, 13544, 42823, 14908, 53732, 36641, 30991,
        51325, 17386, 8021, 64174, 54647, 43965, 24045, 61290, 31570, 55957, 54447, 42363, 11227,
        24282, 63186, 46737, 46223, 42107, 9178, 7889, 63112, 46150, 41523, 4509, 36078, 26481,
        15241, 56394, 57939, 4764, 38113, 42760, 14405, 49710, 4467, 35741, 23786, 59221, 15017,
        54600, 43588, 21031, 37181, 35311, 20351, 31737, 57291, 65118, 62193, 38795, 48222, 58097,
        6029, 48236, 58213, 6958, 55664, 52098, 23570, 57491, 1182, 9463, 10174, 15858, 61331,
        31896, 58566, 9776, 12678, 35894, 25011, 3480, 27847, 26174, 12786, 36757, 31913, 58696,
        10821, 21038, 37237, 35758, 23921, 60297, 23630, 57969, 5001, 40014, 57974, 5043, 40344,
        60615, 26174, 12791, 36798, 32247, 61374, 32245, 61357, 32108, 60259, 23325, 55533, 51051,
        15194, 56022, 54966, 46519, 44476, 28132, 28451, 31003, 51420, 18144, 14084, 47136, 49408,
        2053, 16425, 334, 2673, 21390, 40048, 58244, 7203, 57627, 2269, 18157, 14189, 47982, 56179,
        56220, 56550, 59186, 14740, 52386, 25872, 10375, 17470,
    ];
    const PACK_K6: [u16; 96] = [
        26702, 1614, 23635, 17955, 10963, 9063, 49589, 39530, 49120, 27483, 5466, 19933, 63136,
        22521, 17230, 10062, 27730, 49983, 21089, 17327, 60554, 46570, 2290, 41906, 10865, 57335,
        22954, 44954, 19402, 13361, 16680, 7511, 30488, 26117, 51549, 44690, 35367, 9937, 9166,
        38816, 61676, 51461, 56320, 22004, 51454, 43416, 45864, 21250, 5995, 14956, 1370, 54149,
        24317, 1797, 64157, 29051, 19920, 61484, 3902, 53643, 45329, 23811, 22309, 493, 63953,
        40130, 1573, 62545, 57484, 53049, 53580, 7538, 54119, 56034, 7743, 15269, 53558, 35753,
        28845, 33813, 59050, 64430, 13544, 23505, 9704, 48263, 29314, 26025, 1434, 17851, 33500,
        26403, 33996, 45791, 43145, 38150,
    ];
    const CW_K6: [u16; 256] = [
        8769, 36964, 6457, 20072, 39443, 34020, 14616, 17955, 35031, 13765, 29004, 21283, 51417,
        13938, 40107, 10963, 46310, 14758, 27051, 27329, 45165, 6998, 54701, 27483, 55023, 48126,
        65409, 57421, 4983, 56785, 29781, 5466, 22165, 42367, 24551, 63990, 32168, 27138, 32925,
        10062, 54160, 58420, 3387, 20163, 45263, 13302, 64945, 27730, 5264, 9274, 3773, 44882,
        54424, 9755, 34519, 46570, 31419, 44744, 45610, 35491, 43244, 15136, 51235, 2290, 15543,
        11775, 32732, 63274, 51868, 42778, 50878, 44954, 59030, 42394, 26280, 43572, 36108, 17172,
        50479, 19402, 62087, 41429, 30045, 22337, 53322, 4742, 41368, 26117, 33117, 22385, 56418,
        6318, 11172, 59692, 19237, 51549, 22345, 53869, 39750, 53642, 25225, 41593, 40542, 38816,
        59400, 572, 36667, 52937, 45633, 36959, 6083, 61676, 15125, 50527, 22483, 62684, 14080,
        49162, 678, 43416, 26162, 35983, 9209, 65107, 38080, 12331, 2764, 45864, 51726, 33702,
        59824, 27671, 1498, 30397, 44878, 54149, 57665, 20565, 5480, 23047, 33217, 28757, 5499,
        24317, 48988, 55063, 50671, 31738, 65191, 43487, 30656, 61484, 2835, 50397, 14147, 53457,
        13410, 6320, 11324, 3902, 53143, 58832, 29710, 945, 60484, 4368, 17415, 493, 31573, 54642,
        23702, 9628, 26416, 52271, 3047, 63953, 29821, 8005, 53572, 20742, 16777, 25180, 38716,
        53049, 52856, 40456, 33328, 35869, 1884, 55085, 52037, 53580, 21302, 52654, 27531, 58067,
        46297, 13939, 40174, 15269, 59719, 20963, 30974, 16267, 58090, 47773, 42820, 53558, 19873,
        26689, 4181, 5488, 23595, 2783, 47086, 64430, 60345, 61034, 39593, 43611, 38644, 48403,
        17619, 13544, 14895, 35784, 61980, 34597, 51578, 24198, 41366, 26025, 27228, 38696, 51721,
        33349, 37230, 23472, 60438, 1434, 26265, 42610, 40078, 9090, 57527, 11723, 29387, 45791,
        47073, 63564, 4914, 52373, 9537, 20586, 6818, 43145,
    ];

    #[test]
    fn unpack_k3_matches_reference() {
        let got = unpack_trellis_tile(&PACK_K3, 3).unwrap();
        assert_eq!(got, CW_K3);
    }

    #[test]
    fn unpack_k6_matches_reference() {
        let got = unpack_trellis_tile(&PACK_K6, 6).unwrap();
        assert_eq!(got, CW_K6);
    }

    #[test]
    fn pack_of_codewords_reproduces_packed_tile() {
        let p = pack_trellis_tile(&CW_K3, 3).unwrap();
        assert_eq!(p.as_slice(), &PACK_K3);
        let p6 = pack_trellis_tile(&CW_K6, 6).unwrap();
        assert_eq!(p6.as_slice(), &PACK_K6);
    }

    #[test]
    fn pack_unpack_low_k_roundtrip() {
        // Fresh K-bit payloads: unpack(pack(x)) & mask == x & mask.
        for k in 1u32..=8 {
            let mask = ((1u32 << k) - 1) as u16;
            let mut encoded = [0u16; 256];
            let mut state = 0x1234_5678u32;
            for e in encoded.iter_mut() {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                *e = (state >> 16) as u16 & mask;
            }
            let packed = pack_trellis_tile(&encoded, k).unwrap();
            assert_eq!(packed.len(), packed_tile_len(k));
            let cws = unpack_trellis_tile(&packed, k).unwrap();
            for i in 0..256 {
                assert_eq!(cws[i] & mask, encoded[i], "k={k} i={i}");
            }
        }
    }

    #[test]
    fn rejects_bad_inputs() {
        assert!(matches!(
            pack_trellis_tile(&[0u16; 256], 0),
            Err(TrellisError::BadK(0))
        ));
        assert!(matches!(
            unpack_trellis_tile(&[0u16; 47], 3),
            Err(TrellisError::BadLen { .. })
        ));
        assert!(matches!(
            pack_trellis_tile(&[0u16; 255], 3),
            Err(TrellisError::BadLen { .. })
        ));
    }

    #[test]
    fn funnel_shift_basic() {
        // merged = (a<<32)|b; >> shift; low 32.
        assert_eq!(funnel_shift_r(0, 1, 0), 0);
        assert_eq!(funnel_shift_r(0, 1, 1), 0x8000_0000);
        assert_eq!(funnel_shift_r(1, 0, 0), 1);
        assert_eq!(funnel_shift_r(0xFFFF_0000, 0x1234, 16), 0x1234_FFFF);
        assert_eq!(funnel_shift_r(0xAAAA_5555, 0x1111_2222, 32), 0x1111_2222);
    }
}
