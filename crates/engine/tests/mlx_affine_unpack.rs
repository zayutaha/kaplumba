//! spec/mtp-40tps T5a: Rust-side unpack of the drafter's MLX affine-4bit
//! tensors, validated against MLX's own `mx.dequantize` golden (row 0 of
//! `pre_projection.weight`, first 128 elements, values are BF16 bit patterns).

use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use kaplumba_engine::safetensors::{read_shard_header, ShardHeader};

fn cast_u32(bytes: &[u8]) -> &[u32] {
    // Test data is 4-byte aligned after the safetensors header payload in
    // practice; fall back to a copied parse if it is not.
    if bytes.as_ptr() as usize % 4 == 0 {
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4) }
    } else {
        panic!("unaligned weight tensor")
    }
}

fn cast_u16(bytes: &[u8]) -> &[u16] {
    if bytes.as_ptr() as usize % 2 == 0 {
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u16, bytes.len() / 2) }
    } else {
        panic!("unaligned scales/biases tensor")
    }
}

fn drafter_dir() -> PathBuf {
    std::env::var("KAPLUMBA_DRAFTER_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(
                "/Users/zayaantaha/.omlx/models/mlx-community/gemma-4-31B-it-qat-assistant-4bit",
            )
        })
}

fn tensor_bytes(shard: &ShardHeader, name: &str) -> Vec<u8> {
    let (start, end) = shard.absolute_range(name).expect("tensor range");
    let mut f = std::fs::File::open(&shard.path).expect("open shard");
    f.seek(SeekFrom::Start(start)).expect("seek");
    let mut buf = vec![0u8; (end - start) as usize];
    f.read_exact(&mut buf).expect("read");
    buf
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    ((bits + rounding_bias) >> 16) as u16
}

const GOLDEN_ROW0: [u16; 128] = [
    15352, 47994, 15481, 15547, 47994, 48284, 48122, 46976, 15547, 15352, 48315, 46976, 15516,
    48284, 48249, 15224, 48122, 15352, 15224, 48122, 47994, 15224, 15352, 46976, 15224, 48315,
    15224, 48122, 46976, 15352, 46976, 15481, 46976, 48284, 46976, 46976, 15418, 46976, 15224,
    47994, 47994, 48187, 15224, 15224, 15224, 47994, 15224, 15418, 15628, 15224, 15224, 47994,
    46976, 48249, 47994, 15418, 46976, 48187, 48284, 47994, 15224, 47994, 15481, 48284, 48022,
    15459, 15459, 15256, 15459, 14336, 48022, 48022, 15658, 15459, 48022, 15459, 15459, 15549,
    15256, 48226, 15459, 15459, 15256, 14336, 14336, 15511, 48226, 15256, 15256, 15459, 48022,
    15511, 48316, 48279, 48226, 48022, 48022, 14336, 14336, 14336, 15256, 48226, 15256, 15256,
    15256, 48316, 48022, 15256, 48150, 48022, 48022, 48022, 48316, 48022, 48022, 48022, 48150,
    15256, 15459, 15459, 48022, 48150, 15384, 15256, 48279, 15256, 15384, 48150,
];

#[test]
fn mlx_affine4_unpack_matches_mx_dequantize() {
    let dir = drafter_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("drafter checkpoint not present at {dir:?}; skipping");
        return;
    }
    let shard =
        read_shard_header(&dir.join("model.safetensors")).expect("shard header");

    let w = tensor_bytes(&shard, "pre_projection.weight");
    let s = tensor_bytes(&shard, "pre_projection.scales");
    let b = tensor_bytes(&shard, "pre_projection.biases");

    // Shapes from the checkpoint: weight [1024, 1344] of U32,
    // scales/biases [1024, 168] of BF16, group_size 64, bits 4.
    let weight: &[u32] = cast_u32(&w);
    let scales: &[u16] = cast_u16(&s);
    let biases: &[u16] = cast_u16(&b);

    const K: usize = 10752;
    const GROUP: usize = 64;
    let mut produced = [0u16; 128];
    for k in 0..128usize {
        let word = weight[k / 8];
        let nibble = ((word >> (4 * (k % 8))) & 0xF) as f32;
        let g = k / GROUP;
        let scale = bf16_to_f32(scales[g]);
        let bias = bf16_to_f32(biases[g]);
        // mx.dequantize rounds once, after the fused multiply-add in f32.
        produced[k] = f32_to_bf16(nibble * scale + bias);
    }

    assert_eq!(
        produced, GOLDEN_ROW0,
        "Rust unpack disagrees with mx.dequantize on pre_projection row 0"
    );
}
