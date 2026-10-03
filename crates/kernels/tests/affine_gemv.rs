//! spec/mtp-40tps T5b: affine-4bit GEMV matches a host reference that uses the
//! same dequant math validated against mx.dequantize (see
//! `crates/engine/tests/mlx_affine_unpack.rs`).

use kaplumba_kernels::affine::AffineKernels;

fn to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

fn host_gemv(
    x: &[u16],
    weight: &[u32],
    scales: &[u16],
    biases: &[u16],
    n_rows: usize,
    k_dim: usize,
    group_size: usize,
) -> Vec<u16> {
    let mut out = vec![0u16; n_rows];
    for n in 0..n_rows {
        let mut acc = 0.0f32;
        for k in 0..k_dim {
            let word = weight[n * (k_dim / 8) + k / 8];
            let nibble = ((word >> (4 * (k % 8))) & 0xF) as f32;
            let g = k / group_size;
            let scale = to_f32(scales[n * (k_dim / group_size) + g]);
            let bias = to_f32(biases[n * (k_dim / group_size) + g]);
            let xv = to_f32(x[k]);
            acc += (nibble * scale + bias) * xv;
        }
        out[n] = half::f16::from_f32(acc).to_bits();
    }
    out
}

#[test]
fn affine4_gemv_matches_host() {
    let n_rows = 64;
    let k_dim = 1024;
    let group = 64;
    // deterministic pseudo-random data; fp16 bit constants for scales/biases/x
    let mut weight = vec![0u32; n_rows * (k_dim / 8)];
    let mut scales = vec![0u16; n_rows * (k_dim / group)];
    let mut biases = vec![0u16; n_rows * (k_dim / group)];
    let mut x = vec![0u16; k_dim];
    let mut state: u32 = 0x12345678;
    let mut next = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        state
    };
    for w in weight.iter_mut() {
        *w = next();
    }
    for i in 0..scales.len() {
        // small positive scales (~0.01), biases near 0, in bf16
        let s = 0.005 + 0.01 * ((next() % 100) as f32) / 100.0;
        let b = -0.02 + 0.04 * ((next() % 100) as f32) / 100.0;
        scales[i] = half::f16::from_f32(s).to_bits();
        biases[i] = half::f16::from_f32(b).to_bits();
    }
    for i in 0..x.len() {
        let v = -1.0 + 2.0 * ((next() % 1000) as f32) / 1000.0;
        x[i] = half::f16::from_f32(v).to_bits();
    }

    // host reference treats scales/biases/x as fp16 patterns (matches the drafter
    // once the engine's bf16->fp16 conversion has run)
    let want = host_gemv(&x, &weight, &scales, &biases, n_rows, k_dim, group);

    let kernels = AffineKernels::shared().expect("affine kernels unavailable");
    let got = kernels
        .gemv(&x, &weight, &scales, &biases, n_rows, k_dim, group)
        .expect("gemv");

    let mut max_abs = 0.0f32;
    for n in 0..n_rows {
        let a = to_f32(got[n]);
        let b = to_f32(want[n]);
        max_abs = max_abs.max((a - b).abs());
    }
    assert!(
        max_abs < 2e-3,
        "device vs host max abs error {max_abs}; got row0={} want row0={}",
        to_f32(got[0]),
        to_f32(want[0])
    );
}
