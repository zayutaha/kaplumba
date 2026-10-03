//! spec/mtp-40tps T6: Rust drafter forward vs MLX golden arrays.

use std::path::PathBuf;

use kaplumba_engine::drafter::DrafterWeights;
use kaplumba_engine::drafter_model::{DrafterModel, DrafterSharedKv, KvSlice};

fn golden_dir() -> PathBuf {
    PathBuf::from("/private/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/drafter_golden")
}

fn read_raw(name: &str) -> Vec<u16> {
    let bytes = std::fs::read(golden_dir().join(name)).expect("raw file");
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn to_bf16_as_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[test]
fn drafter_forward_matches_mlx_golden() {
    let dir = std::env::var("KAPLUMBA_DRAFTER_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(
            "/Users/zayaantaha/.omlx/models/mlx-community/gemma-4-31B-it-qat-assistant-4bit",
        )
    });
    if !dir.join("model.safetensors").exists() || !golden_dir().join("last_hidden.raw").exists() {
        eprintln!("drafter or golden missing; skipping");
        return;
    }
    let device = metal::Device::system_default().expect("device");
    let weights = DrafterWeights::load(&dir, &device).expect("load");
    let model = DrafterModel::new(weights).expect("model");

    let inputs = read_raw("inputs_embeds.raw");
    let want_hidden = read_raw("last_hidden.raw");

    // KV layout in golden: [B, H, T, D]. DrafterModel wants [T, H*D].
    let sliding_k = read_raw("kv_sliding_attention_k.raw");
    let sliding_v = read_raw("kv_sliding_attention_v.raw");
    let full_k = read_raw("kv_full_attention_k.raw");
    let full_v = read_raw("kv_full_attention_v.raw");

    let kv_len = 32usize;
    let n_kv_sliding = 16usize;
    let head_dim_sliding = 256usize;
    let n_kv_full = 4usize;
    let head_dim_full = 512usize;

    let mut sliding_keys = vec![0u16; kv_len * n_kv_sliding * head_dim_sliding];
    let mut sliding_values = vec![0u16; kv_len * n_kv_sliding * head_dim_sliding];
    let mut full_keys = vec![0u16; kv_len * n_kv_full * head_dim_full];
    let mut full_values = vec![0u16; kv_len * n_kv_full * head_dim_full];

    // golden axis order: [B, H, T, D] = element (h, t, d)
    for h in 0..n_kv_sliding {
        for t in 0..kv_len {
            let src = (h * kv_len + t) * head_dim_sliding;
            let dst = (t * n_kv_sliding + h) * head_dim_sliding;
            sliding_keys[dst..dst + head_dim_sliding].copy_from_slice(&sliding_k[src..src + head_dim_sliding]);
            sliding_values[dst..dst + head_dim_sliding].copy_from_slice(&sliding_v[src..src + head_dim_sliding]);
        }
    }
    for h in 0..n_kv_full {
        for t in 0..kv_len {
            let src = (h * kv_len + t) * head_dim_full;
            let dst = (t * n_kv_full + h) * head_dim_full;
            full_keys[dst..dst + head_dim_full].copy_from_slice(&full_k[src..src + head_dim_full]);
            full_values[dst..dst + head_dim_full].copy_from_slice(&full_v[src..src + head_dim_full]);
        }
    }

    let shared = DrafterSharedKv {
        sliding: KvSlice {
            keys: sliding_keys,
            values: sliding_values,
            kv_heads: n_kv_sliding,
            head_dim: head_dim_sliding,
            len: kv_len,
            positions: (0..kv_len as u32).collect(),
        },
        full: KvSlice {
            keys: full_keys,
            values: full_values,
            kv_heads: n_kv_full,
            head_dim: head_dim_full,
            len: kv_len,
            positions: (0..kv_len as u32).collect(),
        },
    };

    let step = model
        .forward(&inputs, &shared, kv_len)
        .expect("forward");
    assert_eq!(step.last_hidden.len(), want_hidden.len());

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (&a, &b) in step.last_hidden.iter().zip(want_hidden.iter()) {
        let af = half::f16::from_bits(a).to_f32();
        let bf = half::f16::from_bits(b).to_f32();
        let ad = (af - bf).abs();
        max_abs = max_abs.max(ad);
        max_rel = max_rel.max(ad / (bf.abs() + 1e-6));
    }
    eprintln!(
        "drafter forward: max abs diff {max_abs}, max rel diff {max_rel}, sample got={} want={}",
        half::f16::from_bits(step.last_hidden[0]).to_f32(),
        half::f16::from_bits(want_hidden[0]).to_f32()
    );
    // bf16-vs-f16: expect a couple of thousandths absolute in the post-projection row
    assert!(max_abs < 0.05, "drafter forward max abs diff {max_abs}");
}
