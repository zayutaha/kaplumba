//! Isolated, same-process A/B of the codebook-specialised EXL3 decode matvec.
//!
//! WHAT THIS IS FOR. `decode_state` took the codebook as a runtime argument and
//! branched on it eight times per lane, inside the eight-times-unrolled value
//! loop. That branch, not the decode arithmetic, serialised the kernel. This
//! measures the two kernels back to back in ONE process, on the real Gemma 4
//! shapes, so the comparison cancels clock state, page residency and whatever
//! else the box is doing.
//!
//! METHOD, because every bandwidth harness in this project has been wrong before
//! it was verified:
//!
//!  1. The two kernels are dispatched by NAME through
//!     `gemv_buffer_named`, which is the same code path `gemv_buffer` uses. The
//!     two differ in exactly one token of Metal source, so a difference in the
//!     result is a difference in throughput, not in the work done.
//!  2. Samples are INTERLEAVED A B A B ... never A...A then B...B. Decode on
//!     this box has been measured at 3.7, 3.9, 4.3, 4.8, 6.0 and 6.4 tok/s from
//!     the same binary minutes apart, so a block of A followed by a block of B
//!     reports the drift as an effect.
//!  3. The GPU is warmed to steady state before the first sample. Cold Metal
//!     clocks are up to 7x slower and have produced several false results here.
//!  4. Report MEDIAN, best, worst and spread. A single reading is not a result.
//!  5. Free memory is recorded per configuration. Samples taken while another
//!     process holds the GPU are meaningless, and the box is shared.
//!
//! WHY THERE IS NO CPU REFERENCE HERE. `reconstruct_matrix` folds the Hadamard
//! into the weights, whereas the decode path applies it to the activations. The
//! two are equal mathematically and differ in the last bits, which is why the
//! correctness test for this change is a BIT comparison between the two kernels
//! and lives in `codebook_specialisation.rs`. This file measures speed only.

use half::f16;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{LinearBufferRequest, LinearKernels};
use metal::{Device, MTLResourceOptions};
use std::process::Command;
use std::time::Instant;

/// Per-tile mixed xorshift, so no two tiles share a state stream.
///
/// A period-8 LCG was used before and made every index perturbation that was a
/// multiple of 8 invisible. That does not matter for a throughput measurement,
/// but the same generator is used by the correctness test, so it is the right one
/// and there is only one of them to keep honest.
fn tile_states(bits: u8, tile: u64) -> Vec<u16> {
    let mask = (1_u32 << bits) - 1;
    let mut x = (0x9E37_79B9u32)
        .wrapping_add((tile as u32).wrapping_mul(0x85EB_CA6B))
        .wrapping_add(((tile >> 32) as u32).wrapping_mul(0xC2B2_AE35));
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    (0..256)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x & mask) as u16
        })
        .collect()
}

/// `copies` distinct weight matrices, so every pass reads cold.
///
/// A single 10-43 MB projection is L2-resident on this part. Timing one in
/// isolation measures cache bandwidth, which is exactly what the model does NOT
/// get: 60 layers stream ~11 GB per token and evict everything. Allocating
/// enough copies to exceed L2 and rotating between them is the only way an
/// isolated measurement resembles the real access pattern.
fn build_copies(
    kernels: &LinearKernels,
    in_f: usize,
    out_f: usize,
    bits: u8,
    copies: usize,
) -> Vec<metal::Buffer> {
    let device = Device::system_default().expect("Metal device");
    let tiles = (in_f / 16) * (out_f / 16);
    (0..copies)
        .map(|c| {
            let mut packed = Vec::with_capacity(tiles * 16 * bits as usize);
            for tile in 0..tiles as u64 {
                let (words, _) = kaplumba_engine::exl3::pack_integer_states(
                    &tile_states(bits, tile + c as u64 * 1_000_003),
                    bits,
                )
                .expect("pack");
                packed.extend(words);
            }
            device.new_buffer_with_data(
                packed.as_ptr() as *const std::ffi::c_void,
                (packed.len() * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        })
        .collect()
}

/// `top`'s "unused", parsed correctly. `554M unused` is 0.54 GiB, and stripping
/// non-digits has turned that into 554 GiB before, a 1000x overstatement that
/// stopped agents from running.
fn free_gib() -> f64 {
    let out = Command::new("top")
        .args(["-l", "1", "-n", "0"])
        .output()
        .expect("run top");
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().find(|l| l.contains("PhysMem")).unwrap_or("");
    let before = line.split("unused").next().unwrap_or("").trim_end();
    let head = before
        .rsplit(|c: char| c == ' ' || c == ',')
        .next()
        .unwrap_or("");
    let tok: String = head
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    let unit = head.chars().nth(tok.len()).unwrap_or(' ');
    let v: f64 = tok.parse().unwrap_or(-1.0);
    match unit {
        'T' => v * 1024.0,
        'G' => v,
        'M' => v / 1024.0,
        'K' => v / (1024.0 * 1024.0),
        _ => -1.0,
    }
}

fn one_pass(
    kernels: &LinearKernels,
    kernel: &str,
    weights: &[metal::Buffer],
    input: &[u16],
    input_signs: &[u16],
    output_signs: &[u16],
    in_f: usize,
    out_f: usize,
    bits: u8,
    cursor: &mut usize,
) {
    let packed = &weights[*cursor % weights.len()];
    *cursor += 1;
    kernels
        .gemv_buffer_named(
            kernel,
            LinearBufferRequest {
                packed,
                packed_offset: 0,
                input,
                input_signs,
                output_signs,
                in_features: in_f,
                out_features: out_f,
                rate: Exl3Rate::Integer(bits),
                codebook: Exl3Codebook::Mcg,
                perm_mode: 0,
            },
        )
        .expect("gemv");
}

struct Stat {
    med: f64,
    best: f64,
    worst: f64,
}

impl Stat {
    fn of(bytes: f64, mut t: Vec<f64>) -> Stat {
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Stat {
            med: bytes / t[t.len() / 2] / 1e9,
            best: bytes / t[0] / 1e9,
            worst: bytes / t[t.len() - 1] / 1e9,
        }
    }
    fn spread(&self) -> f64 {
        self.best / self.worst
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let samples: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(9);
    let target_bytes: u64 = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_500_000_000);

    let kernels = LinearKernels::new().expect("kernels");
    println!(
        "# samples {samples}, working-set target {:.1} GB, free {:.1} GiB",
        target_bytes as f64 / 1e9,
        free_gib()
    );
    println!(
        "# {:>22} {:>9} {:>9} {:>9} {:>9} {:>8} {:>7}",
        "shape", "A med", "B med", "B/A", "spreadA", "spreadB", "free"
    );

    for (name, in_f, out_f, bits) in [
        ("q_proj  5376x5376", 5376usize, 5376usize, 3u8),
        ("up_proj 5376x21504", 5376, 21504, 3),
        ("down   21504x5376", 21504, 5376, 3),
        ("lm_head 5376x262144", 5376, 262144, 6),
    ] {
        // The model's lm_head is 6-bit MCG; the decoder linears are 3-bit MCG.
        // Both are dispatched through the MCG family, so both are specialised.
        let per_copy = (in_f / 16) * (out_f / 16) * 16 * bits as usize * 2;
        let copies = ((target_bytes / per_copy as u64).max(2)) as usize;
        let weights = build_copies(&kernels, in_f, out_f, bits, copies);

        let input: Vec<u16> = (0..in_f)
            .map(|i| f16::from_f32(((i as f32) * 0.013).sin()).to_bits())
            .collect();
        let input_signs: Vec<u16> = (0..in_f)
            .map(|i| f16::from_f32(if i % 3 == 0 { -1.0 } else { 1.0 }).to_bits())
            .collect();
        let output_signs: Vec<u16> = (0..out_f)
            .map(|i| f16::from_f32(if i % 5 == 0 { -1.0 } else { 1.0 }).to_bits())
            .collect();
        // Bytes moved per PASS is ONE copy of the weights, not the whole
        // rotating set. Dividing by the working set instead reports a physically
        // impossible figure -- it read 1804 GB/s on a 273 GB/s part, which is the
        // signal that the accounting was wrong rather than that the kernel was
        // fast.
        let bytes = per_copy as f64;

        let generic = format!("linear_gemv_exl3_k{bits}");
        let special = format!("linear_gemv_exl3_k{bits}_mcg");

        let mut cursor = 0usize;
        // Warm to steady state. Cold Metal clocks are up to 7x slower on this
        // part and have produced false results in this project before.
        let warm_until = Instant::now() + std::time::Duration::from_millis(1500);
        while Instant::now() < warm_until {
            one_pass(
                &kernels,
                &generic,
                &weights,
                &input,
                &input_signs,
                &output_signs,
                in_f,
                out_f,
                bits,
                &mut cursor,
            );
            one_pass(
                &kernels,
                &special,
                &weights,
                &input,
                &input_signs,
                &output_signs,
                in_f,
                out_f,
                bits,
                &mut cursor,
            );
        }

        let mut a = Vec::with_capacity(samples);
        let mut b = Vec::with_capacity(samples);
        for _ in 0..samples {
            // Interleaved. Never a block of A then a block of B.
            let t = Instant::now();
            one_pass(
                &kernels,
                &generic,
                &weights,
                &input,
                &input_signs,
                &output_signs,
                in_f,
                out_f,
                bits,
                &mut cursor,
            );
            a.push(t.elapsed().as_secs_f64());
            let t = Instant::now();
            one_pass(
                &kernels,
                &special,
                &weights,
                &input,
                &input_signs,
                &output_signs,
                in_f,
                out_f,
                bits,
                &mut cursor,
            );
            b.push(t.elapsed().as_secs_f64());
        }
        let sa = Stat::of(bytes, a);
        let sb = Stat::of(bytes, b);
        println!(
            "  {:>22} {:>9.1} {:>9.1} {:>8.3}x {:>8.2}x {:>7.2}x {:>6.1}",
            name,
            sa.med,
            sb.med,
            sb.med / sa.med,
            sa.spread(),
            sb.spread(),
            free_gib()
        );
        drop(weights);
    }
}
