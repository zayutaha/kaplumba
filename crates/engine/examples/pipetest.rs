//! End-to-end check of the EXL3 linear pipeline (scales + Hadamard + decode)
//! against ishizuki's qwen3 fixture, which ships the original weight.
use kaplumba_engine::exl3::{decode_tile, hadamard_128, BitRate, Codebook};

const FIX: &str = "/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/ishizuki/Tests/IshizukiKitTests/Fixtures/exl3-reference.safetensors";

fn main() {
    let bytes = std::fs::read(FIX).unwrap();
    let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let hdr: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let base = 8 + n;

    // qwen3: trellis [8,8,64] -> 128x128, rate 4, codebook threeInst
    let trellis = read_u16(&bytes, base, &hdr, "qwen3.trellis");
    let orig = read_f16(&bytes, base, &hdr, "qwen3.weight");
    let suh = read_f16(&bytes, base, &hdr, "qwen3.suh");
    let svh = read_f16(&bytes, base, &hdr, "qwen3.svh");

    let (tr, tc, pw) = shape3(&hdr, "qwen3.trellis");
    let k = tr * 16;
    let nn = tc * 16;
    println!("qwen3 tile {tr}x{tc} words={pw} -> {k}x{nn}  (rate {}, cb threeInst)", pw / 16);

    // 1. decode the trellis
    let mut w = vec![0.0f32; k * nn];
    for i in 0..tr {
        for j in 0..tc {
            let off = (i * tc + j) * pw;
            let d = decode_tile(&trellis[off..off + pw], BitRate::Integer(4), Codebook::Plain).unwrap();
            for a in 0..16 {
                for b in 0..16 {
                    w[(i * 16 + a) * nn + (j * 16 + b)] = d[a * 16 + b];
                }
            }
        }
    }

    // 3. full pipeline on a fixed input: H(x*suh) @ W^T -> H -> *svh
    let rows = 4usize;
    let mut x = vec![0.0f32; rows * k];
    for (i, v) in x.iter_mut().enumerate() {
        *v = (((i * 37 + 11) % 101) as f32 / 101.0) - 0.5;
    }
    // transform input
    let mut xh = x.clone();
    for r in 0..rows {
        let mut row = x[r * k..(r + 1) * k].to_vec();
        for i in 0..k { row[i] *= suh[i]; }
        hadamard_128(&mut row).unwrap();
        xh[r * k..(r + 1) * k].copy_from_slice(&row);
    }
    // matmul
    let mut y = vec![0.0f32; rows * nn];
    for r in 0..rows {
        for j in 0..nn {
            let mut acc = 0.0f32;
            for i in 0..k { acc += xh[r * k + i] * w[i * nn + j]; }
            y[r * nn + j] = acc;
        }
    }
    // transform output
    let mut out = y.clone();
    for r in 0..rows {
        let mut row = y[r * nn..(r + 1) * nn].to_vec();
        hadamard_128(&mut row).unwrap();
        for j in 0..nn { row[j] *= svh[j]; }
        out[r * nn..(r + 1) * nn].copy_from_slice(&row);
    }
    // reference: plain matmul against the ORIGINAL weight
    let mut ref_out = vec![0.0f32; rows * nn];
    for r in 0..rows {
        for j in 0..nn {
            let mut acc = 0.0f32;
            for i in 0..k { acc += x[r * k + i] * orig[i * nn + j]; }
            ref_out[r * nn + j] = acc;
        }
    }
    let mut n2 = 0.0f64; let mut d2 = 0.0f64;
    for i in 0..rows * nn {
        let d = out[i] as f64 - ref_out[i] as f64;
        n2 += d * d; d2 += (ref_out[i] as f64) * (ref_out[i] as f64);
    }
    println!("full pipeline vs matmul(original): rel_rmse = {:.4}", (n2 / d2).sqrt());
}

fn shape3(h: &serde_json::Value, name: &str) -> (usize, usize, usize) {
    let s = &h[name]["shape"];
    (s[0].as_u64().unwrap() as usize, s[1].as_u64().unwrap() as usize, s[2].as_u64().unwrap() as usize)
}
fn read_u16(b: &[u8], base: usize, h: &serde_json::Value, name: &str) -> Vec<u16> {
    let s = h[name]["data_offsets"][0].as_u64().unwrap() as usize;
    let e = h[name]["data_offsets"][1].as_u64().unwrap() as usize;
    b[base + s..base + e].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}
fn read_f16(b: &[u8], base: usize, h: &serde_json::Value, name: &str) -> Vec<f32> {
    let s = h[name]["data_offsets"][0].as_u64().unwrap() as usize;
    let e = h[name]["data_offsets"][1].as_u64().unwrap() as usize;
    b[base + s..base + e].chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()
}
