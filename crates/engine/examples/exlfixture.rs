//! Validate the CPU EXL3 decode against ishizuki's reference fixture.
use kaplumba_engine::exl3::{decode_tile, BitRate, Codebook};

const FIX: &str = "/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/ishizuki/Tests/IshizukiKitTests/Fixtures/exl3-reference.safetensors";

fn main() {
    let bytes = std::fs::read(FIX).expect("fixture");
    let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let hdr: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let cases: Vec<(&str, BitRate, Codebook)> = vec![
        ("r1c0", BitRate::Integer(1), Codebook::Plain),
        ("r2c1", BitRate::Integer(2), Codebook::Mcg),
        ("r3c2", BitRate::Integer(3), Codebook::Mul1),
        ("r4c0", BitRate::Integer(4), Codebook::Plain),
        ("r5c2", BitRate::Integer(5), Codebook::Mul1),
        ("r6c1", BitRate::Integer(6), Codebook::Mcg),
        ("r7c2", BitRate::Integer(7), Codebook::Mul1),
        ("r8c0", BitRate::Integer(8), Codebook::Plain),
    ];
    println!("{:8} {:>10} {:>10}  verdict", "case", "maxdelta", "mean|exp|");
    for (name, rate, cb) in cases {
        let trellis = read(&bytes, 8 + n, &hdr, &format!("{name}.trellis"));
        let exp = read_f32(&bytes, 8 + n, &hdr, &format!("{name}.expected"));
        let (tr, tc, pw) = shape3(&hdr, &format!("{name}.trellis"));
        let rows = tr * 16;
        let cols = tc * 16;
        let mut got = vec![0.0f32; rows * cols];
        for i in 0..tr {
            for j in 0..tc {
                let off = (i * tc + j) * pw;
                let tile = decode_tile(&trellis[off..off + pw], rate, cb).expect("decode");
                for kr in 0..16 {
                    for kc in 0..16 {
                        got[(i * 16 + kr) * cols + (j * 16 + kc)] = tile[kr * 16 + kc];
                    }
                }
            }
        }
        let mut worst = 0.0f32;
        let mut mag = 0.0f32;
        for (a, b) in got.iter().zip(exp.iter()) {
            worst = worst.max((a - b).abs());
            mag += b.abs();
        }
        mag /= exp.len() as f32;
        println!(
            "{name:8} {worst:10.6} {mag:10.6}  {}",
            if worst == 0.0 { "BIT-EXACT" } else { "MISMATCH" }
        );
    }
}

fn shape3(hdr: &serde_json::Value, name: &str) -> (usize, usize, usize) {
    let s = &hdr[name]["shape"];
    (s[0].as_u64().unwrap() as usize, s[1].as_u64().unwrap() as usize, s[2].as_u64().unwrap() as usize)
}

fn read(bytes: &[u8], base: usize, hdr: &serde_json::Value, name: &str) -> Vec<u16> {
    let s = hdr[name]["data_offsets"][0].as_u64().unwrap() as usize;
    let e = hdr[name]["data_offsets"][1].as_u64().unwrap() as usize;
    bytes[base + s..base + e]
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn read_f32(bytes: &[u8], base: usize, hdr: &serde_json::Value, name: &str) -> Vec<f32> {
    let s = hdr[name]["data_offsets"][0].as_u64().unwrap() as usize;
    let e = hdr[name]["data_offsets"][1].as_u64().unwrap() as usize;
    bytes[base + s..base + e]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
