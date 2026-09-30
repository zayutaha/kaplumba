use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

use crate::config::{
    ConfigError, Gemma4Config, GenerationConfig, MCG_MULTIPLIER, QuantizationConfig, TensorStorage,
    load_config, load_generation_config, load_quantization_config,
};

const MAX_HEADER_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum SafetensorError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid JSON in {path}: {source}")]
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("invalid safetensors metadata in {path}: {message}")]
    Invalid { path: PathBuf, message: String },
    #[error("invalid model metadata: {0}")]
    Metadata(String),
}

#[derive(Debug, Error)]
pub enum MetadataError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error(transparent)]
    Safe(#[from] SafetensorError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Bool,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F16,
    BF16,
    F32,
    F64,
}

impl DType {
    pub fn from_safetensors(value: &str) -> Result<Self, String> {
        match value {
            "BOOL" => Ok(Self::Bool),
            "U8" => Ok(Self::U8),
            "I8" => Ok(Self::I8),
            "U16" => Ok(Self::U16),
            "I16" => Ok(Self::I16),
            "U32" => Ok(Self::U32),
            "I32" => Ok(Self::I32),
            "U64" => Ok(Self::U64),
            "I64" => Ok(Self::I64),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::BF16),
            "F32" => Ok(Self::F32),
            "F64" => Ok(Self::F64),
            other => Err(format!("unsupported safetensors dtype {other}")),
        }
    }

    pub fn from_torch_name(value: &str) -> Result<Self, String> {
        match value {
            "torch.bool" | "bool" => Ok(Self::Bool),
            "torch.uint8" | "u8" => Ok(Self::U8),
            "torch.int8" | "i8" => Ok(Self::I8),
            "torch.uint16" | "u16" => Ok(Self::U16),
            "torch.int16" | "i16" => Ok(Self::I16),
            "torch.uint32" | "u32" => Ok(Self::U32),
            "torch.int32" | "i32" => Ok(Self::I32),
            "torch.uint64" | "u64" => Ok(Self::U64),
            "torch.int64" | "i64" => Ok(Self::I64),
            "torch.float16" | "f16" => Ok(Self::F16),
            "torch.bfloat16" | "bf16" => Ok(Self::BF16),
            "torch.float32" | "f32" => Ok(Self::F32),
            "torch.float64" | "f64" => Ok(Self::F64),
            other => Err(format!("unsupported torch dtype {other}")),
        }
    }

    pub fn size_bytes(self) -> u64 {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub start: u64,
    pub end: u64,
}

impl TensorInfo {
    pub fn byte_len(&self) -> u64 {
        self.end - self.start
    }

    pub fn element_count(&self) -> Option<u64> {
        self.shape.iter().try_fold(1_u64, |count, &dimension| {
            count.checked_mul(dimension as u64)
        })
    }
}

#[derive(Debug, Clone)]
pub struct ShardHeader {
    pub path: PathBuf,
    pub file_size: u64,
    pub header_len: u64,
    pub tensors: BTreeMap<String, TensorInfo>,
}

impl ShardHeader {
    pub fn payload_start(&self) -> u64 {
        8 + self.header_len
    }

    pub fn absolute_range(&self, name: &str) -> Option<(u64, u64)> {
        let tensor = self.tensors.get(name)?;
        Some((
            self.payload_start().checked_add(tensor.start)?,
            self.payload_start().checked_add(tensor.end)?,
        ))
    }
}

#[derive(Debug, Clone)]
pub struct SafeTensorsIndex {
    pub metadata: BTreeMap<String, Value>,
    pub weight_map: BTreeMap<String, String>,
}

impl SafeTensorsIndex {
    pub fn total_size(&self) -> Option<u64> {
        self.metadata.get("total_size").and_then(Value::as_u64)
    }
}

#[derive(Debug, Clone)]
pub struct TextTensorSelection {
    pub index: SafeTensorsIndex,
    pub shards: BTreeMap<String, ShardHeader>,
    pub tensors: BTreeMap<String, TensorInfo>,
    pub tensor_shards: BTreeMap<String, String>,
    pub excluded_vision: Vec<String>,
    pub all_bytes: u64,
    pub text_bytes: u64,
}

impl TextTensorSelection {
    pub fn shard_for(&self, name: &str) -> Option<&str> {
        self.tensor_shards.get(name).map(String::as_str)
    }

    pub fn absolute_range(&self, name: &str) -> Option<(u64, u64)> {
        let shard = self.shards.get(self.shard_for(name)?)?;
        shard.absolute_range(name)
    }
}

#[derive(Debug, Clone)]
pub struct CheckpointMetadata {
    pub config: Gemma4Config,
    pub generation: GenerationConfig,
    pub quantization: QuantizationConfig,
    pub selection: TextTensorSelection,
}

pub fn read_shard_header(path: &Path) -> Result<ShardHeader, SafetensorError> {
    let mut file = File::open(path).map_err(|source| SafetensorError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let file_size = file
        .metadata()
        .map_err(|source| SafetensorError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    let mut length_bytes = [0_u8; 8];
    file.read_exact(&mut length_bytes)
        .map_err(|source| SafetensorError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let header_len = u64::from_le_bytes(length_bytes);
    if !(2..=MAX_HEADER_BYTES).contains(&header_len) {
        return Err(SafetensorError::Invalid {
            path: path.to_path_buf(),
            message: format!("header length {header_len} is outside the allowed range"),
        });
    }
    let payload_start = 8_u64
        .checked_add(header_len)
        .ok_or_else(|| SafetensorError::Invalid {
            path: path.to_path_buf(),
            message: "header length overflows".to_string(),
        })?;
    if payload_start > file_size {
        return Err(SafetensorError::Invalid {
            path: path.to_path_buf(),
            message: "header extends beyond file".to_string(),
        });
    }
    let header_len_usize = usize::try_from(header_len).map_err(|_| SafetensorError::Invalid {
        path: path.to_path_buf(),
        message: "header length does not fit usize".to_string(),
    })?;
    let mut header_bytes = vec![0_u8; header_len_usize];
    file.read_exact(&mut header_bytes)
        .map_err(|source| SafetensorError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let entries: BTreeMap<String, Value> =
        serde_json::from_slice(&header_bytes).map_err(|source| SafetensorError::Json {
            path: path.to_path_buf(),
            source,
        })?;
    let mut tensors = BTreeMap::new();
    for (name, value) in entries {
        if name == "__metadata__" {
            continue;
        }
        let object = value.as_object().ok_or_else(|| SafetensorError::Invalid {
            path: path.to_path_buf(),
            message: format!("tensor {name} is not an object"),
        })?;
        let dtype_name = object.get("dtype").and_then(Value::as_str).ok_or_else(|| {
            SafetensorError::Invalid {
                path: path.to_path_buf(),
                message: format!("tensor {name} has no dtype"),
            }
        })?;
        let dtype =
            DType::from_safetensors(dtype_name).map_err(|message| SafetensorError::Invalid {
                path: path.to_path_buf(),
                message: format!("tensor {name}: {message}"),
            })?;
        let shape = object
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| SafetensorError::Invalid {
                path: path.to_path_buf(),
                message: format!("tensor {name} has no shape"),
            })?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| SafetensorError::Invalid {
                        path: path.to_path_buf(),
                        message: format!("tensor {name} has an invalid shape"),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let offsets = object
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| SafetensorError::Invalid {
                path: path.to_path_buf(),
                message: format!("tensor {name} has no data_offsets"),
            })?;
        if offsets.len() != 2 {
            return Err(SafetensorError::Invalid {
                path: path.to_path_buf(),
                message: format!("tensor {name} must have two data offsets"),
            });
        }
        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| SafetensorError::Invalid {
                path: path.to_path_buf(),
                message: format!("tensor {name} has an invalid start offset"),
            })?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| SafetensorError::Invalid {
                path: path.to_path_buf(),
                message: format!("tensor {name} has an invalid end offset"),
            })?;
        if end < start {
            return Err(SafetensorError::Invalid {
                path: path.to_path_buf(),
                message: format!("tensor {name} has reversed data offsets"),
            });
        }
        let element_count = shape.iter().try_fold(1_u64, |count, &dimension| {
            count.checked_mul(dimension as u64)
        });
        let element_count = element_count.ok_or_else(|| SafetensorError::Invalid {
            path: path.to_path_buf(),
            message: format!("tensor {name} shape overflows"),
        })?;
        let expected_bytes = element_count
            .checked_mul(dtype.size_bytes())
            .ok_or_else(|| SafetensorError::Invalid {
                path: path.to_path_buf(),
                message: format!("tensor {name} byte size overflows"),
            })?;
        if end - start != expected_bytes {
            return Err(SafetensorError::Invalid {
                path: path.to_path_buf(),
                message: format!(
                    "tensor {name} byte size {} does not match shape and dtype size {}",
                    end - start,
                    expected_bytes
                ),
            });
        }
        let absolute_end =
            payload_start
                .checked_add(end)
                .ok_or_else(|| SafetensorError::Invalid {
                    path: path.to_path_buf(),
                    message: format!("tensor {name} end offset overflows"),
                })?;
        if absolute_end > file_size {
            return Err(SafetensorError::Invalid {
                path: path.to_path_buf(),
                message: format!("tensor {name} extends beyond file"),
            });
        }
        tensors.insert(
            name.clone(),
            TensorInfo {
                name,
                dtype,
                shape,
                start,
                end,
            },
        );
    }
    if tensors.is_empty() {
        return Err(SafetensorError::Invalid {
            path: path.to_path_buf(),
            message: "header contains no tensors".to_string(),
        });
    }
    let mut ordered: Vec<&TensorInfo> = tensors.values().collect();
    ordered.sort_by_key(|tensor| (tensor.start, tensor.end));
    for pair in ordered.windows(2) {
        if pair[1].start < pair[0].end {
            return Err(SafetensorError::Invalid {
                path: path.to_path_buf(),
                message: format!(
                    "tensor ranges overlap: {} and {}",
                    pair[0].name, pair[1].name
                ),
            });
        }
    }
    Ok(ShardHeader {
        path: path.to_path_buf(),
        file_size,
        header_len,
        tensors,
    })
}

pub fn read_index(path: &Path) -> Result<SafeTensorsIndex, SafetensorError> {
    let mut file = File::open(path).map_err(|source| SafetensorError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|source| SafetensorError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let raw: RawIndex = serde_json::from_slice(&bytes).map_err(|source| SafetensorError::Json {
        path: path.to_path_buf(),
        source,
    })?;
    if raw.weight_map.is_empty() {
        return Err(SafetensorError::Invalid {
            path: path.to_path_buf(),
            message: "weight_map is empty".to_string(),
        });
    }
    for (tensor, shard) in &raw.weight_map {
        if tensor.is_empty() || !is_safe_shard_name(shard) {
            return Err(SafetensorError::Invalid {
                path: path.to_path_buf(),
                message: format!("invalid weight_map entry {tensor} -> {shard}"),
            });
        }
    }
    Ok(SafeTensorsIndex {
        metadata: raw.metadata,
        weight_map: raw.weight_map,
    })
}

#[derive(Debug, Deserialize)]
struct RawIndex {
    #[serde(default)]
    metadata: BTreeMap<String, Value>,
    weight_map: BTreeMap<String, String>,
}

fn is_safe_shard_name(name: &str) -> bool {
    let path = Path::new(name);
    !name.is_empty()
        && !name.contains('/')
        && !name.contains('\\')
        && path.file_name().and_then(|value| value.to_str()) == Some(name)
        && name != "."
        && name != ".."
}

pub fn load_text_selection(
    model_dir: &Path,
    index: SafeTensorsIndex,
) -> Result<TextTensorSelection, SafetensorError> {
    let shard_names: BTreeSet<String> = index.weight_map.values().cloned().collect();
    let mut shards = BTreeMap::new();
    for shard_name in shard_names {
        let path = model_dir.join(&shard_name);
        shards.insert(shard_name, read_shard_header(&path)?);
    }
    let mut tensors = BTreeMap::new();
    let mut tensor_shards = BTreeMap::new();
    let mut excluded_vision = Vec::new();
    let mut all_bytes = 0_u64;
    let mut text_bytes = 0_u64;
    for (shard_name, shard) in &shards {
        for (name, tensor) in &shard.tensors {
            let mapped_shard = index.weight_map.get(name).ok_or_else(|| {
                SafetensorError::Metadata(format!("tensor {name} is missing from weight_map"))
            })?;
            if mapped_shard != shard_name {
                return Err(SafetensorError::Metadata(format!(
                    "tensor {name} is mapped to {mapped_shard}, found in {shard_name}"
                )));
            }
            all_bytes = all_bytes.checked_add(tensor.byte_len()).ok_or_else(|| {
                SafetensorError::Metadata("tensor byte total overflows".to_string())
            })?;
            if is_vision_tensor_name(name) {
                excluded_vision.push(name.clone());
            } else {
                text_bytes = text_bytes.checked_add(tensor.byte_len()).ok_or_else(|| {
                    SafetensorError::Metadata("text tensor byte total overflows".to_string())
                })?;
                tensors.insert(name.clone(), tensor.clone());
                tensor_shards.insert(name.clone(), shard_name.clone());
            }
        }
    }
    for name in index.weight_map.keys() {
        let shard_name = &index.weight_map[name];
        let shard = shards.get(shard_name).ok_or_else(|| {
            SafetensorError::Metadata(format!(
                "tensor {name} references missing shard {shard_name}"
            ))
        })?;
        if !shard.tensors.contains_key(name) {
            return Err(SafetensorError::Metadata(format!(
                "index tensor {name} is absent from {shard_name}"
            )));
        }
    }
    if let Some(total_size) = index.total_size()
        && total_size != all_bytes
    {
        return Err(SafetensorError::Metadata(format!(
            "index total_size {total_size} does not match tensor bytes {all_bytes}"
        )));
    }
    if tensors.is_empty() {
        return Err(SafetensorError::Metadata(
            "index contains no text tensors".to_string(),
        ));
    }
    Ok(TextTensorSelection {
        index,
        shards,
        tensors,
        tensor_shards,
        excluded_vision,
        all_bytes,
        text_bytes,
    })
}

pub fn is_vision_tensor_name(name: &str) -> bool {
    name.starts_with("model.vision_tower.") || name.starts_with("model.embed_vision.")
}

pub fn validate_quantization_metadata(
    quantization: &QuantizationConfig,
    selection: &TextTensorSelection,
) -> Result<(), SafetensorError> {
    for (module, storage) in &quantization.tensor_storage {
        if is_vision_tensor_name(module) {
            continue;
        }
        validate_storage(module, storage, quantization, selection)?;
    }
    Ok(())
}

fn validate_storage(
    module: &str,
    storage: &TensorStorage,
    quantization: &QuantizationConfig,
    selection: &TextTensorSelection,
) -> Result<(), SafetensorError> {
    for (name, metadata) in &storage.stored_tensors {
        let tensor = selection.tensors.get(name).ok_or_else(|| {
            SafetensorError::Metadata(format!(
                "quantization references missing text tensor {name}"
            ))
        })?;
        let shape_matches = tensor.shape == metadata.shape
            || (quantization.layout.as_deref() == Some("colmajor")
                && tensor.shape.len() == 3
                && metadata.shape.len() == 3
                && tensor.shape[0] == metadata.shape[1]
                && tensor.shape[1] == metadata.shape[0]
                && tensor.shape[2] == metadata.shape[2]);
        if !shape_matches {
            return Err(SafetensorError::Metadata(format!(
                "quantization shape mismatch for {name}"
            )));
        }
        if tensor.byte_len() != metadata.n_bytes {
            return Err(SafetensorError::Metadata(format!(
                "quantization byte count mismatch for {name}"
            )));
        }
        let dtype = DType::from_torch_name(&metadata.dtype).map_err(|message| {
            SafetensorError::Metadata(format!("quantization dtype for {name}: {message}"))
        })?;
        if dtype != tensor.dtype {
            return Err(SafetensorError::Metadata(format!(
                "quantization dtype mismatch for {name}"
            )));
        }
    }
    if storage.quant_format.as_deref() != Some("exl3") {
        return Ok(());
    }
    let bits = storage.bits_per_weight.ok_or_else(|| {
        SafetensorError::Metadata(format!("missing bits_per_weight for {module}"))
    })?;
    let expected_bits = if module == "lm_head" {
        quantization.head_bits as f64
    } else if module.starts_with("model.language_model.layers.") {
        quantization.bits as f64
    } else {
        bits
    };
    if bits != expected_bits {
        return Err(SafetensorError::Metadata(format!(
            "bitrate mismatch for {module}: {bits} versus {expected_bits}"
        )));
    }
    if quantization.codebook == "mcg" && storage.mcg_multiplier != Some(MCG_MULTIPLIER) {
        return Err(SafetensorError::Metadata(format!(
            "mcg multiplier mismatch for {module}"
        )));
    }
    let trellis_name = format!("{module}.trellis");
    let trellis = selection
        .tensors
        .get(&trellis_name)
        .ok_or_else(|| SafetensorError::Metadata(format!("missing text tensor {trellis_name}")))?;
    if trellis.dtype != DType::I16 || trellis.shape.len() != 3 {
        return Err(SafetensorError::Metadata(format!(
            "invalid EXL3 trellis tensor {trellis_name}"
        )));
    }
    for sign_name in [format!("{module}.su"), format!("{module}.suh")] {
        if let Some(sign) = selection.tensors.get(&sign_name)
            && sign.dtype != DType::F16
        {
            return Err(SafetensorError::Metadata(format!(
                "invalid EXL3 input sign tensor {sign_name}"
            )));
        }
    }
    for sign_name in [format!("{module}.sv"), format!("{module}.svh")] {
        if let Some(sign) = selection.tensors.get(&sign_name)
            && sign.dtype != DType::F16
        {
            return Err(SafetensorError::Metadata(format!(
                "invalid EXL3 output sign tensor {sign_name}"
            )));
        }
    }
    if quantization.codebook == "mcg" {
        let mcg_name = format!("{module}.mcg");
        let mcg = selection
            .tensors
            .get(&mcg_name)
            .ok_or_else(|| SafetensorError::Metadata(format!("missing text tensor {mcg_name}")))?;
        if mcg.dtype != DType::I32
            || !(mcg.shape.is_empty() || mcg.shape.len() == 1 && mcg.shape[0] == 1)
        {
            return Err(SafetensorError::Metadata(format!(
                "invalid EXL3 mcg tensor {mcg_name}"
            )));
        }
    }
    Ok(())
}

pub fn validate_text_graph_tensors(
    config: &Gemma4Config,
    quantization: &QuantizationConfig,
    selection: &TextTensorSelection,
) -> Result<(), SafetensorError> {
    let mut expected = BTreeSet::new();
    let mut allowed_modules: BTreeSet<String> = BTreeSet::new();
    let mut expected_quant_modules = BTreeSet::new();

    expect_plain_tensor(
        selection,
        &mut expected,
        "model.language_model.embed_tokens.weight",
        &[config.vocab_size, config.hidden_size],
        DType::BF16,
    )?;
    expect_plain_tensor(
        selection,
        &mut expected,
        "model.language_model.norm.weight",
        &[config.hidden_size],
        DType::BF16,
    )?;
    allowed_modules.insert("model.language_model.embed_tokens".to_string());
    allowed_modules.insert("model.language_model.norm".to_string());

    for (index, kind) in config.layer_types.iter().enumerate() {
        let prefix = format!("model.language_model.layers.{index}");
        let head_dim = if kind.is_full() {
            config.global_head_dim
        } else {
            config.head_dim
        };
        for (suffix, shape) in [
            ("input_layernorm.weight", vec![config.hidden_size]),
            ("self_attn.q_norm.weight", vec![head_dim]),
            ("self_attn.k_norm.weight", vec![head_dim]),
            ("post_attention_layernorm.weight", vec![config.hidden_size]),
            ("pre_feedforward_layernorm.weight", vec![config.hidden_size]),
            (
                "post_feedforward_layernorm.weight",
                vec![config.hidden_size],
            ),
            ("layer_scalar", vec![1]),
        ] {
            expect_plain_tensor(
                selection,
                &mut expected,
                &format!("{prefix}.{suffix}"),
                &shape,
                DType::BF16,
            )?;
        }
        allowed_modules.insert(format!("{prefix}.input_layernorm"));
        allowed_modules.insert(format!("{prefix}.self_attn.q_norm"));
        allowed_modules.insert(format!("{prefix}.self_attn.k_norm"));
        allowed_modules.insert(format!("{prefix}.self_attn.v_norm"));
        allowed_modules.insert(format!("{prefix}.post_attention_layernorm"));
        allowed_modules.insert(format!("{prefix}.pre_feedforward_layernorm"));
        allowed_modules.insert(format!("{prefix}.post_feedforward_layernorm"));
        for linear in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.up_proj",
            "mlp.gate_proj",
            "mlp.down_proj",
        ] {
            if linear == "self_attn.v_proj" && kind.is_full() {
                continue;
            }
            let module = format!("{prefix}.{linear}");
            let storage = quantization.tensor_storage.get(&module).ok_or_else(|| {
                SafetensorError::Metadata(format!("missing quantization storage {module}"))
            })?;
            if storage.quant_format.as_deref() != Some("exl3") {
                return Err(SafetensorError::Metadata(format!(
                    "missing EXL3 storage for {module}"
                )));
            }
            expected_quant_modules.insert(module.clone());
            allowed_modules.insert(module.clone());
            add_quantized_tensor_names(selection, &mut expected, &module, storage)?;
            validate_linear_geometry(config, quantization, &module, storage, selection)?;
        }
    }

    let head_module = "lm_head";
    let head_storage = quantization
        .tensor_storage
        .get(head_module)
        .ok_or_else(|| SafetensorError::Metadata("missing lm_head storage".to_string()))?;
    if head_storage.quant_format.as_deref() != Some("exl3") {
        return Err(SafetensorError::Metadata("lm_head is not EXL3".to_string()));
    }
    expected_quant_modules.insert(head_module.to_string());
    allowed_modules.insert(head_module.to_string());
    add_quantized_tensor_names(selection, &mut expected, head_module, head_storage)?;
    validate_linear_geometry(config, quantization, head_module, head_storage, selection)?;

    for module in quantization.tensor_storage.keys() {
        if !allowed_modules.contains(module) {
            return Err(SafetensorError::Metadata(format!(
                "unexpected quantization module {module}"
            )));
        }
    }
    for module in expected_quant_modules {
        if !quantization.tensor_storage.contains_key(&module) {
            return Err(SafetensorError::Metadata(format!(
                "expected quantization module {module} is missing"
            )));
        }
    }
    for name in selection.tensors.keys() {
        if !expected.contains(name) {
            return Err(SafetensorError::Metadata(format!(
                "unexpected text tensor {name}"
            )));
        }
    }
    Ok(())
}

fn expect_plain_tensor(
    selection: &TextTensorSelection,
    expected: &mut BTreeSet<String>,
    name: &str,
    shape: &[usize],
    dtype: DType,
) -> Result<(), SafetensorError> {
    expected.insert(name.to_string());
    let tensor = selection
        .tensors
        .get(name)
        .ok_or_else(|| SafetensorError::Metadata(format!("missing text tensor {name}")))?;
    if tensor.shape != shape || tensor.dtype != dtype {
        return Err(SafetensorError::Metadata(format!(
            "text tensor shape or dtype mismatch for {name}"
        )));
    }
    Ok(())
}

fn add_quantized_tensor_names(
    selection: &TextTensorSelection,
    expected: &mut BTreeSet<String>,
    module: &str,
    storage: &TensorStorage,
) -> Result<(), SafetensorError> {
    let input_sign = if storage
        .stored_tensors
        .contains_key(&format!("{module}.suh"))
    {
        "suh"
    } else if storage.stored_tensors.contains_key(&format!("{module}.su")) {
        "su"
    } else {
        return Err(SafetensorError::Metadata(format!(
            "missing input sign tensor for {module}"
        )));
    };
    let output_sign = if storage
        .stored_tensors
        .contains_key(&format!("{module}.svh"))
    {
        "svh"
    } else if storage.stored_tensors.contains_key(&format!("{module}.sv")) {
        "sv"
    } else {
        return Err(SafetensorError::Metadata(format!(
            "missing output sign tensor for {module}"
        )));
    };
    for suffix in [input_sign, output_sign, "trellis"] {
        let name = format!("{module}.{suffix}");
        if !selection.tensors.contains_key(&name) {
            return Err(SafetensorError::Metadata(format!(
                "missing quantized text tensor {name}"
            )));
        }
        expected.insert(name);
    }
    if storage
        .stored_tensors
        .contains_key(&format!("{module}.mcg"))
    {
        let name = format!("{module}.mcg");
        if !selection.tensors.contains_key(&name) {
            return Err(SafetensorError::Metadata(format!(
                "missing quantized text tensor {name}"
            )));
        }
        expected.insert(name);
    }
    Ok(())
}

fn validate_linear_geometry(
    config: &Gemma4Config,
    quantization: &QuantizationConfig,
    module: &str,
    storage: &TensorStorage,
    selection: &TextTensorSelection,
) -> Result<(), SafetensorError> {
    let (input, output, bits) =
        linear_dimensions(config, quantization, module).ok_or_else(|| {
            SafetensorError::Metadata(format!("cannot determine dimensions for {module}"))
        })?;
    if input == 0 || output == 0 || !input.is_multiple_of(16) || !output.is_multiple_of(16) {
        return Err(SafetensorError::Metadata(format!(
            "linear dimensions are not tile-aligned for {module}"
        )));
    }
    let input_sign = if storage
        .stored_tensors
        .contains_key(&format!("{module}.suh"))
    {
        "suh"
    } else {
        "su"
    };
    let output_sign = if storage
        .stored_tensors
        .contains_key(&format!("{module}.svh"))
    {
        "svh"
    } else {
        "sv"
    };
    if input_sign == "suh" {
        expect_plain_tensor(
            selection,
            &mut BTreeSet::new(),
            &format!("{module}.suh"),
            &[input],
            DType::F16,
        )?;
    }
    if output_sign == "svh" {
        expect_plain_tensor(
            selection,
            &mut BTreeSet::new(),
            &format!("{module}.svh"),
            &[output],
            DType::F16,
        )?;
    }
    let trellis = selection
        .tensors
        .get(&format!("{module}.trellis"))
        .ok_or_else(|| SafetensorError::Metadata(format!("missing trellis for {module}")))?;
    if bits.fract() == 0.0 {
        let packed_width = (bits as usize * 256) / 16;
        let expected_shape = if quantization.layout.as_deref() == Some("colmajor") && module != "lm_head" {
            vec![output / 16, input / 16, packed_width]
        } else {
            vec![input / 16, output / 16, packed_width]
        };
        if trellis.shape != expected_shape {
            return Err(SafetensorError::Metadata(format!(
                "trellis shape mismatch for {module}"
            )));
        }
    }
    Ok(())
}

fn linear_dimensions(
    config: &Gemma4Config,
    quantization: &QuantizationConfig,
    module: &str,
) -> Option<(usize, usize, f64)> {
    if module == "lm_head" {
        return Some((
            config.hidden_size,
            config.vocab_size,
            quantization.head_bits as f64,
        ));
    }
    let rest = module.strip_prefix("model.language_model.layers.")?;
    let (layer_text, suffix) = rest.split_once('.')?;
    let layer = layer_text.parse::<usize>().ok()?;
    let kind = *config.layer_types.get(layer)?;
    let head_dim = if kind.is_full() {
        config.global_head_dim
    } else {
        config.head_dim
    };
    let kv_heads = if kind.is_full() {
        config.num_global_key_value_heads
    } else {
        config.num_key_value_heads
    };
    let dimensions = match suffix {
        "self_attn.q_proj" => (config.hidden_size, config.num_attention_heads * head_dim),
        "self_attn.k_proj" | "self_attn.v_proj" => (config.hidden_size, kv_heads * head_dim),
        "self_attn.o_proj" => (config.num_attention_heads * head_dim, config.hidden_size),
        "mlp.up_proj" | "mlp.gate_proj" => (config.hidden_size, config.intermediate_size),
        "mlp.down_proj" => (config.intermediate_size, config.hidden_size),
        _ => return None,
    };
    Some((dimensions.0, dimensions.1, quantization.bits as f64))
}

pub fn inspect_model_dir(model_dir: &Path) -> Result<CheckpointMetadata, MetadataError> {
    let config = load_config(&model_dir.join("config.json"))?;
    config.validate_current_checkpoint()?;
    let generation = load_generation_config(&model_dir.join("generation_config.json"))?;
    let quantization = load_quantization_config(&model_dir.join("quantization_config.json"))?;
    config.validate_quantization_summary(&quantization)?;
    let index = read_index(&model_dir.join("model.safetensors.index.json"))?;
    let selection = load_text_selection(model_dir, index)?;
    validate_quantization_metadata(&quantization, &selection)?;
    validate_text_graph_tensors(&config, &quantization, &selection)?;
    Ok(CheckpointMetadata {
        config,
        generation,
        quantization,
        selection,
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::json;

    use super::*;
    use crate::config::LayerKind;

    fn temp_dir(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("kaplumba-{name}-{}-{stamp}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_shard(path: &Path, entries: Vec<(&str, DType, Vec<usize>, Vec<u8>)>) {
        let mut payload = Vec::new();
        let mut header = serde_json::Map::new();
        for (name, dtype, shape, bytes) in entries {
            let start = payload.len() as u64;
            payload.extend_from_slice(&bytes);
            let end = payload.len() as u64;
            header.insert(
                name.to_string(),
                json!({
                    "dtype": match dtype {
                        DType::F32 => "F32",
                        DType::F16 => "F16",
                        DType::BF16 => "BF16",
                        DType::I16 => "I16",
                        DType::I32 => "I32",
                        other => panic!("unsupported test dtype {other:?}"),
                    },
                    "shape": shape,
                    "data_offsets": [start, end]
                }),
            );
        }
        let header_bytes = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut file = fs::File::create(path).unwrap();
        use std::io::Write;
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(&payload).unwrap();
    }

    fn write_shard_with_offsets(path: &Path, entries: Vec<(&str, DType, Vec<usize>, [u64; 2])>) {
        let payload_len = entries
            .iter()
            .map(|(_, _, _, offsets)| offsets[1])
            .max()
            .unwrap_or(0) as usize;
        let mut header = serde_json::Map::new();
        for (name, dtype, shape, offsets) in entries {
            header.insert(
                name.to_string(),
                json!({
                    "dtype": match dtype {
                        DType::I32 => "I32",
                        other => panic!("unsupported test dtype {other:?}"),
                    },
                    "shape": shape,
                    "data_offsets": offsets
                }),
            );
        }
        let header_bytes = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut file = fs::File::create(path).unwrap();
        use std::io::Write;
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(&vec![0_u8; payload_len]).unwrap();
    }

    fn base_config(layer_types: Vec<&str>) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "model_type": "gemma4",
            "image_token_id": 1,
            "audio_token_id": 2,
            "video_token_id": 3,
            "text_config": {
                "model_type": "gemma4_text",
                "dtype": "bfloat16",
                "attention_bias": false,
                "attention_k_eq_v": true,
                "enable_moe_block": false,
                "eos_token_id": 1,
                "final_logit_softcapping": 30.0,
                "global_head_dim": 512,
                "head_dim": 256,
                "hidden_activation": "gelu_pytorch_tanh",
                "hidden_size": 5376,
                "hidden_size_per_layer_input": 0,
                "intermediate_size": 21504,
                "layer_types": layer_types,
                "max_position_embeddings": 8192,
                "num_attention_heads": 32,
                "num_global_key_value_heads": 4,
                "num_hidden_layers": 2,
                "num_key_value_heads": 16,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "full_attention": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional"
                    },
                    "sliding_attention": {
                        "rope_theta": 10000.0,
                        "rope_type": "default"
                    }
                },
                "sliding_window": 1024,
                "tie_word_embeddings": false,
                "use_cache": true,
                "vocab_size": 262144,
                "vocab_size_per_layer_input": 262144,
                "bos_token_id": 2,
                "pad_token_id": 0
            },
            "quantization_config": {
                "quant_method": "exl3",
                "version": "0.0.37",
                "bits": 3.0,
                "head_bits": 6,
                "calibration": {"rows": 250, "cols": 2048},
                "out_scales": "always",
                "codebook": "mcg"
            }
        }))
        .unwrap()
    }

    #[test]
    fn parses_config_and_generation_defaults() {
        let config =
            crate::config::parse_config(&base_config(vec!["sliding_attention", "full_attention"]))
                .unwrap();
        assert_eq!(
            config.layer_types,
            vec![LayerKind::Sliding, LayerKind::Full]
        );
        assert_eq!(config.quantization.head_bits, 6);
        let generation = crate::config::parse_generation_config(
            br#"{"bos_token_id":2,"eos_token_id":[1,106,50],"temperature":1.0,"top_k":64,"top_p":0.95}"#,
        )
        .unwrap();
        assert_eq!(generation.eos_token_ids, vec![1, 106, 50]);
        assert_eq!(generation.top_k, 64);
    }

    #[test]
    fn rejects_layer_count_mismatch() {
        let error = crate::config::parse_config(&base_config(vec!["sliding_attention"]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("layer_types"));
    }

    #[test]
    fn parses_valid_shard_and_rejects_dtype_mismatch() {
        let dir = temp_dir("shard-valid");
        let path = dir.join("model.safetensors");
        write_shard(&path, vec![("weight", DType::F32, vec![2], vec![0; 8])]);
        let header = read_shard_header(&path).unwrap();
        assert_eq!(header.tensors["weight"].byte_len(), 8);
        let bad_path = dir.join("bad.safetensors");
        write_shard(&bad_path, vec![("weight", DType::F32, vec![2], vec![0; 4])]);
        let error = read_shard_header(&bad_path).unwrap_err().to_string();
        assert!(error.contains("byte size"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_overlapping_ranges() {
        let dir = temp_dir("shard-overlap");
        let path = dir.join("model.safetensors");
        write_shard_with_offsets(
            &path,
            vec![
                ("a", DType::I32, vec![1], [0, 4]),
                ("b", DType::I32, vec![1], [2, 6]),
            ],
        );
        let error = read_shard_header(&path).unwrap_err().to_string();
        assert!(error.contains("overlap"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn selects_text_and_excludes_vision() {
        let dir = temp_dir("selection");
        let shard = dir.join("model.safetensors");
        write_shard(
            &shard,
            vec![
                (
                    "model.language_model.norm.weight",
                    DType::BF16,
                    vec![2],
                    vec![0; 4],
                ),
                (
                    "model.vision_tower.weight",
                    DType::BF16,
                    vec![2],
                    vec![0; 4],
                ),
                (
                    "model.embed_vision.weight",
                    DType::BF16,
                    vec![2],
                    vec![0; 4],
                ),
            ],
        );
        let index_path = dir.join("model.safetensors.index.json");
        fs::write(
            &index_path,
            serde_json::to_vec(&json!({
                "metadata": {"total_size": 12},
                "weight_map": {
                    "model.language_model.norm.weight": "model.safetensors",
                    "model.vision_tower.weight": "model.safetensors",
                    "model.embed_vision.weight": "model.safetensors"
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let index = read_index(&index_path).unwrap();
        let selection = load_text_selection(&dir, index).unwrap();
        assert_eq!(selection.tensors.len(), 1);
        assert_eq!(selection.excluded_vision.len(), 2);
        assert_eq!(selection.text_bytes, 4);
        assert!(
            selection
                .shard_for("model.language_model.norm.weight")
                .is_some()
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_missing_index_entry() {
        let dir = temp_dir("selection-missing");
        let shard = dir.join("model.safetensors");
        write_shard(
            &shard,
            vec![(
                "model.language_model.norm.weight",
                DType::BF16,
                vec![2],
                vec![0; 4],
            )],
        );
        let index_path = dir.join("model.safetensors.index.json");
        fs::write(
            &index_path,
            serde_json::to_vec(&json!({
                "metadata": {"total_size": 4},
                "weight_map": {"missing.tensor": "model.safetensors"}
            }))
            .unwrap(),
        )
        .unwrap();
        let error = load_text_selection(&dir, read_index(&index_path).unwrap())
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing from weight_map"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_path_traversal_in_index() {
        let dir = temp_dir("index-path");
        let index_path = dir.join("model.safetensors.index.json");
        fs::write(
            &index_path,
            serde_json::to_vec(&json!({
                "metadata": {"total_size": 0},
                "weight_map": {"tensor": "../outside.safetensors"}
            }))
            .unwrap(),
        )
        .unwrap();
        let error = read_index(&index_path).unwrap_err().to_string();
        assert!(error.contains("invalid weight_map"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_shape_byte_mismatch() {
        let dir = temp_dir("shard-shape");
        let path = dir.join("model.safetensors");
        write_shard(&path, vec![("weight", DType::I32, vec![2], vec![0; 4])]);
        let error = read_shard_header(&path).unwrap_err().to_string();
        assert!(error.contains("byte size"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_index_total_size_mismatch() {
        let dir = temp_dir("selection-size");
        let shard = dir.join("model.safetensors");
        write_shard(
            &shard,
            vec![(
                "model.language_model.norm.weight",
                DType::BF16,
                vec![2],
                vec![0; 4],
            )],
        );
        let index_path = dir.join("model.safetensors.index.json");
        fs::write(
            &index_path,
            serde_json::to_vec(&json!({
                "metadata": {"total_size": 8},
                "weight_map": {"model.language_model.norm.weight": "model.safetensors"}
            }))
            .unwrap(),
        )
        .unwrap();
        let error = load_text_selection(&dir, read_index(&index_path).unwrap())
            .unwrap_err()
            .to_string();
        assert!(error.contains("total_size"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    #[ignore = "requires KAPLUMBA_TEST_MODEL"]
    fn inspects_real_checkpoint() {
        let model_dir = PathBuf::from(
            std::env::var("KAPLUMBA_TEST_MODEL")
                .expect("KAPLUMBA_TEST_MODEL must point to the checkpoint"),
        );
        let metadata = inspect_model_dir(&model_dir).unwrap();
        assert_eq!(metadata.config.num_hidden_layers, 60);
        assert!(metadata.selection.tensors.len() > 1_000);
        assert!(!metadata.selection.excluded_vision.is_empty());
        assert_eq!(metadata.selection.text_bytes, 14_876_563_172);
        assert_eq!(metadata.selection.all_bytes, 16_028_050_244);
    }
}
