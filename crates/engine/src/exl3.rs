use half::f16;
use thiserror::Error;

pub const TILE_SIZE: usize = 256;
pub const HADAMARD_SIZE: usize = 128;
pub const MCG_MULTIPLIER: u32 = 0xCBAC1FED;
pub const MUL1_MULTIPLIER: u32 = 0x83DCD12D;
pub const FRACTIONAL_MASK: u32 = 0xAAAA;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Exl3Error {
    #[error("unsupported EXL3 bitrate {0}")]
    UnsupportedRate(String),
    #[error("fractional EXL3 bitrate requires the mul1 codebook")]
    FractionalRequiresMul1,
    #[error("packed trellis has {actual} words, expected {expected}")]
    TrellisLength { actual: usize, expected: usize },
    #[error("trellis state {state} does not fit in {bits} bits")]
    StateOutOfRange { state: u16, bits: u8 },
    #[error("state count must be 256, got {0}")]
    StateCount(usize),
    #[error("inconsistent overlapping fractional windows")]
    FractionalWindowConflict,
    #[error("matrix dimensions must be multiples of 16, got {rows}x{columns}")]
    MatrixAlignment { rows: usize, columns: usize },
    #[error("Hadamard dimensions must be multiples of 128, got {rows}x{columns}")]
    HadamardAlignment { rows: usize, columns: usize },
    #[error("sign vector has {actual} values, expected {expected}")]
    SignLength { actual: usize, expected: usize },
    #[error("packed sign vector length is not divisible by 16")]
    PackedSignAlignment,
    #[error("unsupported sign representation")]
    SignRepresentation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codebook {
    Plain,
    Mcg,
    Mul1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitRate {
    Integer(u8),
    Fractional(u8),
}

impl BitRate {
    pub fn from_rate(rate: f32) -> Result<Self, Exl3Error> {
        if !rate.is_finite() {
            return Err(Exl3Error::UnsupportedRate(format!("{rate:?}")));
        }
        if rate.fract() == 0.0 && (1.0..=8.0).contains(&rate) {
            return Ok(Self::Integer(rate as u8));
        }
        if matches!(rate, 1.5 | 2.5 | 3.5) {
            return Ok(Self::Fractional(rate.floor() as u8));
        }
        Err(Exl3Error::UnsupportedRate(rate.to_string()))
    }

    pub fn bits(self) -> u8 {
        match self {
            Self::Integer(bits) => bits,
            Self::Fractional(bits) => bits,
        }
    }

    pub fn is_fractional(self) -> bool {
        matches!(self, Self::Fractional(_))
    }

    pub fn packed_words(self) -> usize {
        match self {
            Self::Integer(bits) => 16 * bits as usize,
            Self::Fractional(bits) => 16 * bits as usize + 8,
        }
    }

    pub fn validate_codebook(self, codebook: Codebook) -> Result<(), Exl3Error> {
        if self.is_fractional() && codebook != Codebook::Mul1 {
            return Err(Exl3Error::FractionalRequiresMul1);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SignVector {
    HalfBits(Vec<u16>),
    PackedBits(Vec<u16>),
}

impl SignVector {
    pub fn from_half_bits(values: Vec<u16>) -> Self {
        Self::HalfBits(values)
    }

    pub fn from_packed_bits(values: Vec<u16>) -> Self {
        Self::PackedBits(values)
    }

    pub fn decode(&self, expected: usize) -> Result<Vec<f32>, Exl3Error> {
        match self {
            Self::HalfBits(values) => {
                if values.len() != expected {
                    return Err(Exl3Error::SignLength {
                        actual: values.len(),
                        expected,
                    });
                }
                Ok(values
                    .iter()
                    .map(|value| f16::from_bits(*value).to_f32())
                    .collect())
            }
            Self::PackedBits(values) => {
                if values.len() * 16 != expected {
                    return Err(Exl3Error::SignLength {
                        actual: values.len() * 16,
                        expected,
                    });
                }
                Ok(unpack_signs(values))
            }
        }
    }
}

pub fn tensor_core_permutation() -> [u16; TILE_SIZE] {
    let mut permutation = [0_u16; TILE_SIZE];
    for t in 0..32 {
        let r0 = (t % 4) * 2;
        let r1 = r0 + 1;
        let r2 = r0 + 8;
        let r3 = r0 + 9;
        let c0 = t / 4;
        let c1 = c0 + 8;
        permutation[t * 8] = (r0 * 16 + c0) as u16;
        permutation[t * 8 + 1] = (r1 * 16 + c0) as u16;
        permutation[t * 8 + 2] = (r2 * 16 + c0) as u16;
        permutation[t * 8 + 3] = (r3 * 16 + c0) as u16;
        permutation[t * 8 + 4] = (r0 * 16 + c1) as u16;
        permutation[t * 8 + 5] = (r1 * 16 + c1) as u16;
        permutation[t * 8 + 6] = (r2 * 16 + c1) as u16;
        permutation[t * 8 + 7] = (r3 * 16 + c1) as u16;
    }
    permutation
}

pub fn permute_tile(values: &[f32; TILE_SIZE]) -> [f32; TILE_SIZE] {
    let permutation = tensor_core_permutation();
    let mut output = [0.0; TILE_SIZE];
    for row_major in 0..TILE_SIZE {
        output[permutation[row_major] as usize] = values[row_major];
    }
    output
}

pub fn unpermute_tile(values: &[f32; TILE_SIZE]) -> [f32; TILE_SIZE] {
    let permutation = tensor_core_permutation();
    let mut output = [0.0; TILE_SIZE];
    for row_major in 0..TILE_SIZE {
        output[row_major] = values[permutation[row_major] as usize];
    }
    output
}

pub fn decode_state(state: u16, codebook: Codebook) -> f32 {
    match codebook {
        Codebook::Plain => decode_three_inst(state as u32, 89_226_354, 64_248_484),
        Codebook::Mcg => decode_three_inst(state as u32, MCG_MULTIPLIER, 0),
        Codebook::Mul1 => decode_mul1(state as u32),
    }
}

fn decode_three_inst(mut state: u32, multiplier: u32, addend: u32) -> f32 {
    state = state.wrapping_mul(multiplier).wrapping_add(addend);
    let transformed = 0x3b60_3b60_u32 ^ (state & 0x8fff_8fff);
    let low = f16::from_bits(transformed as u16);
    let high = f16::from_bits((transformed >> 16) as u16);
    (low + high).to_f32()
}

fn decode_mul1(mut state: u32) -> f32 {
    state = state.wrapping_mul(MUL1_MULTIPLIER);
    let byte_sum = state.to_le_bytes().into_iter().map(u32::from).sum::<u32>() + 0x6400;
    let value = f16::from_bits(byte_sum as u16);
    let inverse = f16::from_bits(0x1eee);
    let bias = f16::from_bits(0xc931);
    f16::from_f32(value.to_f32() * inverse.to_f32() + bias.to_f32()).to_f32()
}

pub fn unpack_trellis(packed: &[u16], rate: BitRate) -> Result<Vec<u16>, Exl3Error> {
    let expected = rate.packed_words();
    if packed.len() != expected {
        return Err(Exl3Error::TrellisLength {
            actual: packed.len(),
            expected,
        });
    }
    match rate {
        BitRate::Integer(bits) => unpack_integer(packed, bits),
        BitRate::Fractional(bits) => unpack_fractional(packed, bits),
    }
}

fn words_from_u16(values: &[u16]) -> Vec<u32> {
    values
        .chunks_exact(2)
        .map(|pair| u32::from(pair[0]) | (u32::from(pair[1]) << 16))
        .collect()
}

fn u16_from_words(words: &[u32], swap_pairs: bool) -> Vec<u16> {
    let mut output = Vec::with_capacity(words.len() * 2);
    for word in words {
        let value = if swap_pairs {
            word.rotate_right(16)
        } else {
            *word
        };
        output.push(value as u16);
        output.push((value >> 16) as u16);
    }
    output
}

fn unpack_integer(packed: &[u16], bits: u8) -> Result<Vec<u16>, Exl3Error> {
    let bits = bits as usize;
    let words = words_from_u16(packed);
    let word_count = words.len();
    let total_bits = 256 * bits;
    let mut output = Vec::with_capacity(TILE_SIZE);
    for index in 0..TILE_SIZE {
        let b0 = index * bits + bits + total_bits - 16;
        let b1 = b0 + 16;
        let i0 = b0 / 32;
        let i1 = (b1 - 1) / 32;
        let shift = (i1 + 1) * 32 - b1;
        let merged = (u64::from(words[i0 % word_count]) << 32) | u64::from(words[i1 % word_count]);
        output.push(((merged >> shift) & 0xffff) as u16);
    }
    Ok(output)
}

fn ring_words(ring: &[bool]) -> Vec<u32> {
    let mut words = vec![0_u32; ring.len() / 32];
    for (position, bit) in ring.iter().enumerate() {
        if *bit {
            words[position / 32] |= 1_u32 << (31 - (position % 32));
        }
    }
    words
}

fn integer_end(index: usize, bits: u8) -> usize {
    (index + 1) * bits as usize
}

fn pack_integer_windows(windows: &[u16], bits: u8) -> Result<Vec<u16>, Exl3Error> {
    if windows.len() != TILE_SIZE {
        return Err(Exl3Error::StateCount(windows.len()));
    }
    let ring_length = 256 * bits as usize;
    let mut ring = vec![false; ring_length];
    let mut assigned = vec![false; ring_length];
    for (index, window) in windows.iter().enumerate() {
        let start = integer_end(index, bits) as isize - 16;
        for bit in 0..16 {
            let position = normalize_ring(start + bit as isize, ring_length);
            let value = (window >> (15 - bit)) & 1 != 0;
            if assigned[position] && ring[position] != value {
                return Err(Exl3Error::FractionalWindowConflict);
            }
            ring[position] = value;
            assigned[position] = true;
        }
    }
    Ok(u16_from_words(&ring_words(&ring), false))
}

pub fn pack_integer_states(states: &[u16], bits: u8) -> Result<(Vec<u16>, Vec<u16>), Exl3Error> {
    if states.len() != TILE_SIZE {
        return Err(Exl3Error::StateCount(states.len()));
    }
    let bits_usize = bits as usize;
    let mask = (1_u32 << bits_usize) - 1;
    let mut packed_values = vec![0_u16; 16 * bits_usize];
    for span in 0..16 {
        let mut state_index = span * 16;
        let mut packed_index = bits_usize * span;
        let mut bit_offset = 32_i32;
        let mut buffer = 0_u32;
        for _ in 0..16 {
            let value = u32::from(states[state_index]);
            if value > mask {
                return Err(Exl3Error::StateOutOfRange {
                    state: states[state_index],
                    bits,
                });
            }
            bit_offset -= bits as i32;
            buffer |= value << bit_offset;
            if bit_offset <= 16 {
                packed_values[packed_index] = (buffer >> 16) as u16;
                buffer <<= 16;
                bit_offset += 16;
                packed_index += 1;
            }
            state_index += 1;
        }
    }
    let packed = u16_from_words(&words_from_u16(&packed_values), true);
    let windows = unpack_integer(&packed, bits)?;
    Ok((packed, windows))
}

pub fn pack_trellis(states: &[u16], rate: BitRate) -> Result<Vec<u16>, Exl3Error> {
    match rate {
        BitRate::Integer(bits) => pack_integer_windows(states, bits),
        BitRate::Fractional(bits) => pack_fractional_windows(states, bits),
    }
}

fn fractional_depth(bits: u8) -> usize {
    16 * bits as usize + 8
}

fn fractional_width(index: usize, bits: u8) -> usize {
    bits as usize + ((FRACTIONAL_MASK >> (index & 15)) & 1) as usize
}

fn fractional_end(index: usize, bits: u8) -> usize {
    let block = (index >> 4) * fractional_depth(bits);
    let remainder = (0..=(index & 15))
        .map(|i| fractional_width(i, bits))
        .sum::<usize>();
    block + remainder
}

fn normalize_ring(position: isize, ring_length: usize) -> usize {
    let mut value = position % ring_length as isize;
    if value < 0 {
        value += ring_length as isize;
    }
    value as usize
}

fn window_from_ring(ring: &[bool], start: isize) -> u16 {
    let mut value = 0_u16;
    for bit in 0..16 {
        value = (value << 1) | u16::from(ring[normalize_ring(start + bit as isize, ring.len())]);
    }
    value
}

fn fractional_windows_from_ring(ring: &[bool], bits: u8) -> Vec<u16> {
    (0..TILE_SIZE)
        .map(|index| {
            let end = fractional_end(index, bits);
            window_from_ring(ring, end as isize - 16)
        })
        .collect()
}

fn pack_fractional_windows(windows: &[u16], bits: u8) -> Result<Vec<u16>, Exl3Error> {
    if windows.len() != TILE_SIZE {
        return Err(Exl3Error::StateCount(windows.len()));
    }
    let ring_length = fractional_depth(bits) * 16;
    let mut ring = vec![false; ring_length];
    let mut assigned = vec![false; ring_length];
    for (index, window) in windows.iter().enumerate() {
        let start = fractional_end(index, bits) as isize - 16;
        for bit in 0..16 {
            let position = normalize_ring(start + bit as isize, ring_length);
            let value = (window >> (15 - bit)) & 1 != 0;
            if assigned[position] && ring[position] != value {
                return Err(Exl3Error::FractionalWindowConflict);
            }
            ring[position] = value;
            assigned[position] = true;
        }
    }
    Ok(u16_from_words(&ring_words(&ring), false))
}

fn unpack_fractional(packed: &[u16], bits: u8) -> Result<Vec<u16>, Exl3Error> {
    let words = words_from_u16(packed);
    let ring_length = fractional_depth(bits) * 16;
    let mut ring = vec![false; ring_length];
    for (position, bit) in ring.iter_mut().enumerate() {
        *bit = (words[position / 32] >> (31 - (position % 32))) & 1 != 0;
    }
    Ok(fractional_windows_from_ring(&ring, bits))
}

pub fn pack_fractional_states(
    states: &[u8],
    rate: BitRate,
) -> Result<(Vec<u16>, Vec<u16>), Exl3Error> {
    let BitRate::Fractional(bits) = rate else {
        return Err(Exl3Error::UnsupportedRate(format!("{rate:?}")));
    };
    if states.len() != TILE_SIZE {
        return Err(Exl3Error::StateCount(states.len()));
    }
    let ring_length = fractional_depth(bits) * 16;
    let mut ring = vec![false; ring_length];
    let mut position = 0;
    for (index, state) in states.iter().enumerate() {
        let width = fractional_width(index, bits);
        let mask = (1_u16 << width) - 1;
        let value = *state;
        if u16::from(value) > mask {
            return Err(Exl3Error::StateOutOfRange {
                state: u16::from(value),
                bits,
            });
        }
        for bit in (0..width).rev() {
            ring[position] = (value >> bit) & 1 != 0;
            position += 1;
        }
    }
    let windows = fractional_windows_from_ring(&ring, bits);
    let packed = u16_from_words(&ring_words(&ring), false);
    Ok((packed, windows))
}

pub fn decode_tile(
    packed: &[u16],
    rate: BitRate,
    codebook: Codebook,
) -> Result<[f32; TILE_SIZE], Exl3Error> {
    rate.validate_codebook(codebook)?;
    let states = unpack_trellis(packed, rate)?;
    let mut lane_values = [0.0; TILE_SIZE];
    for (index, state) in states.iter().enumerate() {
        lane_values[index] = decode_state(*state, codebook);
    }
    Ok(unpermute_tile(&lane_values))
}

pub fn hadamard_128(values: &mut [f32]) -> Result<(), Exl3Error> {
    if values.len() != HADAMARD_SIZE {
        return Err(Exl3Error::HadamardAlignment {
            rows: values.len(),
            columns: HADAMARD_SIZE,
        });
    }
    let mut width = 1;
    while width < HADAMARD_SIZE {
        let mut start = 0;
        while start < HADAMARD_SIZE {
            for offset in 0..width {
                let left = values[start + offset];
                let right = values[start + offset + width];
                values[start + offset] = left + right;
                values[start + offset + width] = left - right;
            }
            start += width * 2;
        }
        width *= 2;
    }
    let scale = 1.0 / (HADAMARD_SIZE as f32).sqrt();
    for value in values {
        *value *= scale;
    }
    Ok(())
}

pub fn apply_hadamard_matrix(
    values: &mut [f32],
    rows: usize,
    columns: usize,
) -> Result<(), Exl3Error> {
    if !rows.is_multiple_of(HADAMARD_SIZE) || !columns.is_multiple_of(HADAMARD_SIZE) {
        return Err(Exl3Error::HadamardAlignment { rows, columns });
    }
    let mut column = [0.0; HADAMARD_SIZE];
    for row_block in (0..rows).step_by(HADAMARD_SIZE) {
        for col in 0..columns {
            for offset in 0..HADAMARD_SIZE {
                column[offset] = values[(row_block + offset) * columns + col];
            }
            hadamard_128(&mut column)?;
            for offset in 0..HADAMARD_SIZE {
                values[(row_block + offset) * columns + col] = column[offset];
            }
        }
    }
    let mut row_values = [0.0; HADAMARD_SIZE];
    for row_index in 0..rows {
        for column_block in (0..columns).step_by(HADAMARD_SIZE) {
            for offset in 0..HADAMARD_SIZE {
                row_values[offset] = values[row_index * columns + column_block + offset];
            }
            hadamard_128(&mut row_values)?;
            for offset in 0..HADAMARD_SIZE {
                values[row_index * columns + column_block + offset] = row_values[offset];
            }
        }
    }
    Ok(())
}

pub fn reconstruct_matrix(
    packed: &[u16],
    tile_rows: usize,
    tile_columns: usize,
    rate: BitRate,
    codebook: Codebook,
    input_signs: &SignVector,
    output_signs: &SignVector,
) -> Result<Vec<f32>, Exl3Error> {
    if tile_rows == 0 || tile_columns == 0 {
        return Err(Exl3Error::MatrixAlignment {
            rows: tile_rows,
            columns: tile_columns,
        });
    }
    let rows = tile_rows * 16;
    let columns = tile_columns * 16;
    let expected_packed = tile_rows * tile_columns * rate.packed_words();
    if packed.len() != expected_packed {
        return Err(Exl3Error::TrellisLength {
            actual: packed.len(),
            expected: expected_packed,
        });
    }
    let input = input_signs.decode(rows)?;
    let output = output_signs.decode(columns)?;
    let mut values = vec![0.0; rows * columns];
    for tile_row in 0..tile_rows {
        for tile_column in 0..tile_columns {
            let offset = (tile_row * tile_columns + tile_column) * rate.packed_words();
            let tile = decode_tile(
                &packed[offset..offset + rate.packed_words()],
                rate,
                codebook,
            )?;
            for row in 0..16 {
                for column in 0..16 {
                    values[(tile_row * 16 + row) * columns + tile_column * 16 + column] =
                        tile[row * 16 + column];
                }
            }
        }
    }
    if rows.is_multiple_of(HADAMARD_SIZE) && columns.is_multiple_of(HADAMARD_SIZE) {
        apply_hadamard_matrix(&mut values, rows, columns)?;
    }
    for row in 0..rows {
        for column in 0..columns {
            values[row * columns + column] *= input[row] * output[column];
        }
    }
    Ok(values)
}

pub fn unpack_signs(packed: &[u16]) -> Vec<f32> {
    let mut output = Vec::with_capacity(packed.len() * 16);
    for value in packed {
        let mut bits = *value;
        for _ in 0..16 {
            output.push(if bits & 1 == 0 { 1.0 } else { -1.0 });
            bits >>= 1;
        }
    }
    output
}

pub fn pack_signs(values: &[f16]) -> Result<Vec<u16>, Exl3Error> {
    if !values.len().is_multiple_of(16) {
        return Err(Exl3Error::PackedSignAlignment);
    }
    let mut output = Vec::with_capacity(values.len() / 16);
    for group in values.chunks_exact(16) {
        let mut packed = 0_u16;
        for value in group {
            packed >>= 1;
            packed |= value.to_bits() & 0x8000;
        }
        output.push(packed);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn integer_states(bits: u8, seed: u32) -> Vec<u16> {
        let mask = (1_u32 << bits) - 1;
        let mut value = seed;
        (0..TILE_SIZE)
            .map(|_| {
                value = value.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (value & mask) as u16
            })
            .collect()
    }

    fn fractional_states(bits: u8, seed: u32) -> Vec<u8> {
        let mut value = seed;
        (0..TILE_SIZE)
            .map(|index| {
                value = value.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let width = fractional_width(index, bits);
                (value & ((1_u32 << width) - 1)) as u8
            })
            .collect()
    }

    #[test]
    fn integer_pack_unpack_all_rates() {
        for bits in 1..=8 {
            let states = integer_states(bits, bits as u32 * 17);
            let (packed, windows) = pack_integer_states(&states, bits).unwrap();
            assert_eq!(packed.len(), 16 * bits as usize);
            assert_eq!(
                unpack_trellis(&packed, BitRate::Integer(bits)).unwrap(),
                windows
            );
        }
    }

    #[test]
    fn fractional_pack_unpack_all_rates() {
        for bits in 1..=3 {
            let states = fractional_states(bits, bits as u32 * 31);
            let (packed, windows) =
                pack_fractional_states(&states, BitRate::Fractional(bits)).unwrap();
            assert_eq!(packed.len(), (16 * bits as usize + 8));
            assert_eq!(
                unpack_trellis(&packed, BitRate::Fractional(bits)).unwrap(),
                windows
            );
        }
    }

    #[test]
    fn codebooks_decode_finite_values() {
        for codebook in [Codebook::Plain, Codebook::Mcg, Codebook::Mul1] {
            for state in [0, 1, 0x1234, 0x8000, 0xffff] {
                assert!(decode_state(state, codebook).is_finite());
            }
        }
    }

    #[test]
    fn tensor_core_permutation_is_invertible() {
        let values: [f32; TILE_SIZE] = core::array::from_fn(|index| index as f32);
        let permuted = permute_tile(&values);
        assert_eq!(unpermute_tile(&permuted), values);
        let permutation = tensor_core_permutation();
        let mut seen = [false; TILE_SIZE];
        for value in permutation {
            assert!(!seen[value as usize]);
            seen[value as usize] = true;
        }
    }

    #[test]
    fn hadamard_is_orthogonal() {
        let mut values = [0.0; HADAMARD_SIZE];
        for (index, value) in values.iter_mut().enumerate() {
            *value = (index as f32 + 1.0) * 0.125;
        }
        let original = values;
        hadamard_128(&mut values).unwrap();
        hadamard_128(&mut values).unwrap();
        for (actual, expected) in values.iter().zip(original) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn packed_and_half_signs_match() {
        let values: Vec<f16> = (0..32)
            .map(|index| f16::from_f32(if index % 3 == 0 { -1.0 } else { 1.0 }))
            .collect();
        let packed = pack_signs(&values).unwrap();
        let half = SignVector::from_half_bits(values.iter().map(|value| value.to_bits()).collect());
        let bits = SignVector::from_packed_bits(packed);
        assert_eq!(half.decode(32).unwrap(), bits.decode(32).unwrap());
    }

    #[test]
    fn reconstructs_integer_variants_with_both_sign_formats() {
        for bits in 1..=8 {
            for codebook in [Codebook::Plain, Codebook::Mcg, Codebook::Mul1] {
                let states = integer_states(bits, bits as u32 * 7 + codebook as u32);
                let (packed, _) = pack_integer_states(&states, bits).unwrap();
                let input_half: Vec<u16> = (0..16)
                    .map(|index| f16::from_f32(if index % 2 == 0 { 1.0 } else { -1.0 }).to_bits())
                    .collect();
                let output_half: Vec<u16> = (0..16)
                    .map(|index| f16::from_f32(if index % 2 == 0 { -1.0 } else { 1.0 }).to_bits())
                    .collect();
                let input_packed = pack_signs(
                    &input_half
                        .iter()
                        .map(|value| f16::from_bits(*value))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                let output_packed = pack_signs(
                    &output_half
                        .iter()
                        .map(|value| f16::from_bits(*value))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                let half = reconstruct_matrix(
                    &packed,
                    1,
                    1,
                    BitRate::Integer(bits),
                    codebook,
                    &SignVector::from_half_bits(input_half),
                    &SignVector::from_half_bits(output_half),
                )
                .unwrap();
                let packed_signs = reconstruct_matrix(
                    &packed,
                    1,
                    1,
                    BitRate::Integer(bits),
                    codebook,
                    &SignVector::from_packed_bits(input_packed),
                    &SignVector::from_packed_bits(output_packed),
                )
                .unwrap();
                assert_eq!(half, packed_signs);
            }
        }
    }

    #[test]
    fn reconstructs_fractional_mul1_tiles() {
        for bits in 1..=3 {
            let states = fractional_states(bits, bits as u32 * 19);
            let (packed, _) = pack_fractional_states(&states, BitRate::Fractional(bits)).unwrap();
            let signs = SignVector::from_half_bits(vec![f16::from_f32(1.0).to_bits(); 16]);
            let result = reconstruct_matrix(
                &packed,
                1,
                1,
                BitRate::Fractional(bits),
                Codebook::Mul1,
                &signs,
                &signs,
            )
            .unwrap();
            assert_eq!(result.len(), 256);
            assert!(result.iter().all(|value| value.is_finite()));
        }
    }
}
