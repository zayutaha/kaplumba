//! spec/mtp-40tps T5c: loader for the MTP drafter checkpoint.
//!
//! The drafter (`Gemma4AssistantForCausalLM`) is stored in MLX affine-4bit
//! (`group_size 64, mode affine`), one `model.safetensors` shard. The engine
//! normally converts the target checkpoint's BF16 tensors to FP16 at load
//! (`weights.rs::copy_tensor`); we mirror that here so the affine GEMV kernel
//! always sees FP16 scales/biases and FP16 activations.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use metal::{Buffer, MTLResourceOptions};

use crate::safetensors::{read_shard_header, DType, ShardHeader};

pub struct DrafterTensor {
    pub buffer: Buffer,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

pub struct DrafterWeights {
    pub dir: PathBuf,
    pub tensors: BTreeMap<String, DrafterTensor>,
}

impl DrafterWeights {
    pub fn load(dir: &Path, device: &metal::Device) -> Result<Self, DraftLoadError> {
        let shard = read_shard_header(&dir.join("model.safetensors"))
            .map_err(|e| DraftLoadError::Shard(e.to_string()))?;
        let mut f = std::fs::File::open(dir.join("model.safetensors"))
            .map_err(|e| DraftLoadError::Io(e.to_string()))?;
        let mut tensors = BTreeMap::new();
        for (name, info) in &shard.tensors {
            let (start, end) = shard
                .absolute_range(name)
                .ok_or_else(|| DraftLoadError::Range(name.clone()))?;
            let mut bytes = vec![0u8; (end - start) as usize];
            f.seek(SeekFrom::Start(start))
                .map_err(|e| DraftLoadError::Io(e.to_string()))?;
            f.read_exact(&mut bytes)
                .map_err(|e| DraftLoadError::Io(e.to_string()))?;

            let convert = info.dtype == DType::BF16;
            if convert {
                if bytes.len() % 2 != 0 {
                    return Err(DraftLoadError::Range(name.clone()));
                }
                for pair in bytes.chunks_exact_mut(2) {
                    let bits = u16::from_le_bytes([pair[0], pair[1]]);
                    let value = half::f16::from_f32(f32::from_bits(u32::from(bits) << 16));
                    pair.copy_from_slice(&value.to_bits().to_le_bytes());
                }
            }

            let buffer = device.new_buffer(
                bytes.len().max(1) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.contents() as *mut u8, bytes.len());
            }
            tensors.insert(
                name.clone(),
                DrafterTensor {
                    buffer,
                    dtype: if convert { DType::F16 } else { info.dtype },
                    shape: info.shape.clone(),
                },
            );
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            tensors,
        })
    }

    pub fn buffer(&self, name: &str) -> Option<&Buffer> {
        self.tensors.get(name).map(|t| &t.buffer)
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|t| t.shape.as_slice())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DraftLoadError {
    #[error("shard header: {0}")]
    Shard(String),
    #[error("io: {0}")]
    Io(String),
    #[error("tensor range: {0}")]
    Range(String),
}
