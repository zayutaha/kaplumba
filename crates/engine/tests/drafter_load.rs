//! spec/mtp-40tps T5c: `DrafterWeights` load -> `affine4_gemv` buffers -> result
//! agrees with a host reference on the same (converted) tensors.

use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use kaplumba_engine::drafter::DrafterWeights;
use kaplumba_engine::safetensors::read_shard_header;
use kaplumba_kernels::affine::AffineKernels;

fn dir() -> PathBuf {
    std::env::var("KAPLUMBA_DRAFTER_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(
                "/Users/zayaantaha/.omlx/models/mlx-community/gemma-4-31B-it-qat-assistant-4bit",
            )
        })
}

fn bf16_to_f16(bits: u16) -> u16 {
    half::f16::from_f32(f32::from_bits((bits as u32) << 16)).to_bits()
}

fn tensor_bytes(name: &str) -> Vec<u8> {
    let d = dir();
    let shard = read_shard_header(&d.join("model.safetensors")).expect("header");
    let (start, end) = shard.absolute_range(name).expect("range");
    let mut f = std::fs::File::open(d.join("model.safetensors")).expect("open");
    f.seek(SeekFrom::Start(start)).expect("seek");
    let mut buf = vec![0u8; (end - start) as usize];
    f.read_exact(&mut buf).expect("read");
    buf
}

#[test]
fn drafter_load_gemv_matches_host() {
    let d = dir();
    if !d.join("model.safetensors").exists() {
        eprintln!("drafter checkpoint not present; skipping");
        return;
    }
    let device = metal::Device::system_default().expect("device");
    let loaded = DrafterWeights::load(&d, &device).expect("load");

    let w = tensor_bytes("pre_projection.weight");
    let s = tensor_bytes("pre_projection.scales");
    let b = tensor_bytes("pre_projection.biases");
    let weight: &[u32] = unsafe {
        std::slice::from_raw_parts(w.as_ptr() as *const u32, w.len() / 4)
    };
    let scales_f16: Vec<u16> = s
        .chunks_exact(2)
        .map(|p| bf16_to_f16(u16::from_le_bytes([p[0], p[1]])))
        .collect();
    let biases_f16: Vec<u16> = b
        .chunks_exact(2)
        .map(|p| bf16_to_f16(u16::from_le_bytes([p[0], p[1]])))
        .collect();

    // tensor shapes: weight [1024, 1344], scales/biases [1024, 168], group 64
    let k_dim = 10752;
    let n_rows = 1024;
    let group = 64;

    let mut state: u32 = 42;
    let mut next = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        state
    };
    let mut x = vec![0u16; k_dim];
    for v in x.iter_mut() {
        let f = -1.0 + 2.0 * ((next() % 1000) as f32) / 1000.0;
        *v = half::f16::from_f32(f).to_bits();
    }

    // host reference with the same fp16-converted tensors
    let mut want = vec![0f32; n_rows];
    for n in 0..n_rows {
        let mut acc = 0.0f32;
        for k in 0..k_dim {
            let word = weight[n * (k_dim / 8) + k / 8];
            let nibble = ((word >> (4 * (k % 8))) & 0xF) as f32;
            let g = k / group;
            let scale = half::f16::from_bits(scales_f16[n * (k_dim / group) + g]).to_f32();
            let bias = half::f16::from_bits(biases_f16[n * (k_dim / group) + g]).to_f32();
            let xv = half::f16::from_bits(x[k]).to_f32();
            acc += (nibble * scale + bias) * xv;
        }
        want[n] = acc;
    }

    let kernels = AffineKernels::shared().expect("kernels");
    let yb = kernels.alloc_u16(n_rows).expect("y alloc");
    // upload x
    let xb = unsafe {
        let b = kernels.alloc_u16(k_dim).unwrap();
        std::ptr::copy_nonoverlapping(x.as_ptr(), b.contents() as *mut u16, k_dim);
        b
    };
    kernels
        .gemv_buffers(
            &xb,
            loaded.buffer("pre_projection.weight").unwrap(),
            loaded.buffer("pre_projection.scales").unwrap(),
            loaded.buffer("pre_projection.biases").unwrap(),
            &yb,
            n_rows,
            k_dim,
            group,
        )
        .expect("gemv");
    let src = yb.contents() as *const u16;
    let mut got = vec![0u16; n_rows];
    unsafe {
        std::ptr::copy_nonoverlapping(src, got.as_mut_ptr(), n_rows);
    }
    let mut max_abs = 0.0f32;
    for n in 0..n_rows {
        let g = half::f16::from_bits(got[n]).to_f32();
        max_abs = max_abs.max((g - want[n]).abs());
    }
    assert!(
        max_abs < 5e-3 * want[0].abs().max(1.0),
        "device vs host max abs err {max_abs} (want0={})",
        want[0]
    );
}
