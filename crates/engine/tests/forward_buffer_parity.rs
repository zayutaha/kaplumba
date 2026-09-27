//! Parity between the host-slice forward path and the fully GPU-resident
//! `forward_buffer` path.
//!
//! `forward_buffer` encodes a whole 60-layer pass into one command buffer and
//! reads back only the final softcapped logits. It reimplements the layer graph
//! against buffer kernels, so it must agree with the already-verified slice path
//! before any throughput number means anything.

use std::ffi::c_void;
use std::mem::size_of;

use half::f16;
use kaplumba_engine::exl3::pack_integer_states;
use kaplumba_engine::model::{
    DecoderLayer, DecoderLayerWeights, LayerConfig, QuantizedLinear, RopeMode, TextModel,
};
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use metal::{Device, MTLResourceOptions};

fn half(value: f32) -> u16 {
    f16::from_f32(value).to_bits()
}

fn value_at(index: usize, offset: f32) -> u16 {
    half(((index as f32 + offset) * 0.071).sin())
}

/// Quantized linear whose weights also live in a Metal buffer.
fn make_gpu_linear(
    device: &Device,
    in_features: usize,
    out_features: usize,
    seed: u32,
) -> QuantizedLinear {
    let mut packed = Vec::new();
    let tile_count = (in_features / 16) * (out_features / 16);
    for tile in 0..tile_count {
        let mut states = vec![0_u16; 256];
        let mut value = seed.wrapping_add(tile as u32 * 7919);
        for state in &mut states {
            value = value.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *state = (value & 0x7) as u16;
        }
        let (words, _) = pack_integer_states(&states, 3).unwrap();
        packed.extend(words);
    }
    let input_signs = vec![half(0.01); in_features];
    let output_signs = vec![half(0.01); out_features];
    let buffer = device.new_buffer_with_data(
        packed.as_ptr() as *const c_void,
        (packed.len() * size_of::<u16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    QuantizedLinear::from_metal_buffer(
        in_features,
        out_features,
        Exl3Rate::Integer(3),
        Exl3Codebook::Mcg,
        buffer,
        0,
        Some(packed.len()),
        input_signs,
        output_signs,
    )
    .unwrap()
}

fn config(full: bool) -> LayerConfig {
    LayerConfig {
        hidden_size: 128,
        intermediate_size: 256,
        q_heads: 2,
        kv_heads: 1,
        head_dim: 128,
        max_sequence_len: 8,
        full_attention: full,
        sliding_window: if full { 0 } else { 3 },
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        rope_mode: RopeMode::Default,
        rope_partial_factor: 1.0,
    }
}

fn gpu_weights(device: &Device, config: &LayerConfig, seed: u32) -> DecoderLayerWeights {
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let q_dim = config.q_heads * config.head_dim;
    let kv_dim = config.kv_heads * config.head_dim;
    DecoderLayerWeights {
        q_proj: make_gpu_linear(device, hidden, q_dim, seed),
        k_proj: make_gpu_linear(device, hidden, kv_dim, seed + 1),
        v_proj: if config.full_attention {
            None
        } else {
            Some(make_gpu_linear(device, hidden, kv_dim, seed + 2))
        },
        o_proj: make_gpu_linear(device, q_dim, hidden, seed + 3),
        gate_proj: make_gpu_linear(device, hidden, inter, seed + 4),
        up_proj: make_gpu_linear(device, hidden, inter, seed + 5),
        down_proj: make_gpu_linear(device, inter, hidden, seed + 6),
        q_norm: (0..config.head_dim)
            .map(|i| value_at(i, seed as f32))
            .collect(),
        k_norm: (0..config.head_dim)
            .map(|i| value_at(i, seed as f32 + 1.0))
            .collect(),
        input_norm: (0..hidden)
            .map(|i| value_at(i, seed as f32 + 2.0))
            .collect(),
        post_attention_norm: (0..hidden)
            .map(|i| value_at(i, seed as f32 + 3.0))
            .collect(),
        pre_ffn_norm: (0..hidden)
            .map(|i| value_at(i, seed as f32 + 4.0))
            .collect(),
        post_ffn_norm: (0..hidden)
            .map(|i| value_at(i, seed as f32 + 5.0))
            .collect(),
        layer_scalar: 0.94,
    }
}

const VOCAB: usize = 256;

fn gpu_model(device: &Device) -> TextModel {
    let embedding: Vec<u16> = (0..VOCAB * 128).map(|i| value_at(i, 0.5)).collect();
    let embedding_buffer = device.new_buffer_with_data(
        embedding.as_ptr() as *const c_void,
        (embedding.len() * size_of::<u16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let layers = vec![
        DecoderLayer::new(config(false), gpu_weights(device, &config(false), 11)).unwrap(),
        DecoderLayer::new(config(true), gpu_weights(device, &config(true), 29)).unwrap(),
    ];
    let final_norm: Vec<u16> = (0..128).map(|i| value_at(i, 7.5)).collect();
    let lm_head = make_gpu_linear(device, 128, VOCAB, 101);
    TextModel::from_metal_parts(embedding_buffer, layers, final_norm, lm_head, 30.0, None).unwrap()
}

fn compare(name: &str, left: &[u16], right: &[u16], tolerance: f32) {
    assert_eq!(left.len(), right.len(), "{name}: length mismatch");
    let mut worst = 0.0_f32;
    let mut worst_index = 0;
    for (index, (a, b)) in left.iter().zip(right.iter()).enumerate() {
        let a = f16::from_bits(*a).to_f32();
        let b = f16::from_bits(*b).to_f32();
        let delta = (a - b).abs();
        if delta > worst {
            worst = delta;
            worst_index = index;
        }
    }
    assert!(
        worst <= tolerance,
        "{name}: worst delta {worst} at {worst_index} exceeds {tolerance}"
    );
}

#[test]
fn buffer_forward_matches_slice_forward_for_prefill() {
    let device = Device::system_default().expect("Metal device");
    let mut slice_model = gpu_model(&device);
    let mut buffer_model = gpu_model(&device);

    let tokens: Vec<u32> = vec![2, 105, 200, 107, 180, 106, 107];
    let positions: Vec<u32> = (0..tokens.len() as u32).collect();

    let slice = slice_model.forward(&tokens, &positions).unwrap();
    let buffer = buffer_model.forward_buffer(&tokens, &positions).unwrap();

    // The slice path emits every position; the buffer path only the sampled row.
    let vocab = slice_model.vocab_size();
    let last = &slice[(tokens.len() - 1) * vocab..];
    compare("prefill", last, &buffer, 5e-2);
}

#[test]
fn buffer_forward_matches_slice_forward_step_by_step() {
    let device = Device::system_default().expect("Metal device");
    let mut slice_model = gpu_model(&device);
    let mut buffer_model = gpu_model(&device);

    // Two full-attention / sliding decode steps exercise cache append,
    // positions, and the single-row path.
    let prompt: Vec<u32> = vec![2, 105, 200];
    for (step, token) in prompt.iter().enumerate() {
        let position = step as u32;
        let slice = slice_model.forward(&[*token], &[position]).unwrap();
        let buffer = buffer_model.forward_buffer(&[*token], &[position]).unwrap();
        compare(&format!("decode step {step}"), &slice, &buffer, 5e-2);
    }
}

#[test]
fn buffer_forward_rejects_bad_positions() {
    let device = Device::system_default().expect("Metal device");
    let mut model = gpu_model(&device);
    assert!(model.forward_buffer(&[1, 2], &[0, 1]).is_ok());
    // Positions must continue from the cache, not restart.
    assert!(model.forward_buffer(&[3], &[0]).is_err());
    // Positions must be contiguous.
    assert!(model.forward_buffer(&[3, 4], &[4, 6]).is_err());
    // Mismatched lengths are rejected.
    assert!(model.forward_buffer(&[3], &[4, 5]).is_err());
    // Out-of-vocabulary ids are rejected before any GPU work.
    assert!(model.forward_buffer(&[9_999], &[2]).is_err());
}
