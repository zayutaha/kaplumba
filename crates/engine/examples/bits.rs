//! Determine the exact state -> ring-slot map.
//!
//! The ring is 256 * ka bits and holds 256 codes, so there are exactly 256
//! ka-bit slots. Brute force says every code is a contiguous ka-bit read with no
//! collisions, which means the state -> slot map is a bijection. This finds it.

use kaplumba_engine::exl3::{unpack_trellis, pack_integer_states, BitRate};

fn main() {
    for ka in [1u8, 2, 3, 4, 5, 6, 7, 8] {
        let mut seed = 12345u32;
        let states: Vec<u16> = (0..256)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed & ((1u32 << ka) - 1)) as u16
            })
            .collect();
        let (packed, windows) = pack_integer_states(&states, ka).unwrap();
        assert_eq!(unpack_trellis(&packed, BitRate::Integer(ka)).unwrap(), windows);

        let words: Vec<u32> = packed
            .chunks_exact(2)
            .map(|p| u32::from(p[0]) | (u32::from(p[1]) << 16))
            .collect();
        let ring_bits = 256 * ka as usize;
        let mask = (1u32 << ka) - 1;

        // Slot j = the ka bits at ring bit (OFF + j*ka), most significant first.
        // OFF is the frame the reader's window start sits in: (RING - 16) mod ka.
        let off = (ring_bits - 16) % ka as usize;
        let slot = |j: usize| -> u32 {
            let mut v = 0u32;
            for b in 0..ka as usize {
                let p = (off + j * ka as usize + b) % ring_bits;
                v = (v << 1) | ((words[p / 32] >> (31 - (p % 32))) & 1);
            }
            v
        };
        let slots: Vec<u32> = (0..256).map(slot).collect();
        let mut rotation = None;
        for k in 0..256usize {
            if (0..256).all(|s| slots[(s + k) % 256] == (windows[s] as u32 & mask)) {
                rotation = Some(k);
                break;
            }
        }
        match rotation {
            Some(k) => println!("ka={ka}: frame_off={off:>2}  state s -> slot (s + {k:>3}) mod 256"),
            None => println!("ka={ka}: frame_off={off} NO ROTATION"),
        }
    }
}
