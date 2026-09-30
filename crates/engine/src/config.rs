use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;

pub const MCG_MULTIPLIER: u32 = 0xCBAC1FED;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid configuration: {0}")]
    Invalid(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Sliding,
    Full,
}

impl LayerKind {
    pub fn parse(value: &str) -> Result<Self, ConfigError> {
        match value {
            "sliding_attention" => Ok(Self::Sliding),
            "full_attention" => Ok(Self::Full),
            other => Err(ConfigError::Invalid(format!("unknown layer type {other}"))),
        }
    }

    pub fn is_full(self) -> bool {
        matches!(self, Self::Full)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RopeLayerConfig {
    pub rope_type: String,
    pub theta: f32,
    pub partial_rotary_factor: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RopeConfig {
    pub sliding: RopeLayerConfig,
    pub full: RopeLayerConfig,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationSummary {
    pub quant_method: String,
    pub version: String,
    pub bits: f32,
    pub head_bits: u8,
    pub calibration_rows: usize,
    pub calibration_cols: usize,
    pub out_scales: String,
    pub codebook: String,
}

impl Default for QuantizationSummary {
    fn default() -> Self {
        Self {
            quant_method: String::new(),
            version: String::new(),
            bits: 0.0,
            head_bits: 0,
            calibration_rows: 0,
            calibration_cols: 0,
            out_scales: String::new(),
            codebook: String::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4Config {
    pub model_type: String,
    pub text_model_type: String,
    pub dtype: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_global_key_value_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub sliding_window: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub final_logit_softcapping: f32,
    pub hidden_activation: String,
    pub attention_bias: bool,
    pub attention_k_eq_v: bool,
    pub enable_moe_block: bool,
    pub use_cache: bool,
    pub tie_word_embeddings: bool,
    pub hidden_size_per_layer_input: usize,
    pub vocab_size_per_layer_input: usize,
    pub layer_types: Vec<LayerKind>,
    pub rope: RopeConfig,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub image_token_id: Option<u32>,
    pub audio_token_id: Option<u32>,
    pub video_token_id: Option<u32>,
    pub quantization: QuantizationSummary,
}

impl Gemma4Config {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.model_type != "gemma4" {
            return Err(ConfigError::Invalid(format!(
                "model_type must be gemma4, got {}",
                self.model_type
            )));
        }
        if self.text_model_type != "gemma4_text" {
            return Err(ConfigError::Invalid(format!(
                "text model_type must be gemma4_text, got {}",
                self.text_model_type
            )));
        }
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.vocab_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.num_global_key_value_heads == 0
            || self.head_dim == 0
            || self.global_head_dim == 0
        {
            return Err(ConfigError::Invalid(
                "all text dimensions must be non-zero".to_string(),
            ));
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(ConfigError::Invalid(format!(
                "layer_types has {} entries for {} layers",
                self.layer_types.len(),
                self.num_hidden_layers
            )));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(ConfigError::Invalid(
                "num_attention_heads must be divisible by num_key_value_heads".to_string(),
            ));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_global_key_value_heads)
        {
            return Err(ConfigError::Invalid(
                "num_attention_heads must be divisible by num_global_key_value_heads".to_string(),
            ));
        }
        if self.sliding_window == 0 || self.max_position_embeddings == 0 {
            return Err(ConfigError::Invalid(
                "sliding_window and max_position_embeddings must be non-zero".to_string(),
            ));
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(ConfigError::Invalid(
                "rms_norm_eps must be finite and positive".to_string(),
            ));
        }
        if !self.final_logit_softcapping.is_finite() || self.final_logit_softcapping <= 0.0 {
            return Err(ConfigError::Invalid(
                "final_logit_softcapping must be finite and positive".to_string(),
            ));
        }
        if self.attention_bias || !self.attention_k_eq_v {
            return Err(ConfigError::Invalid(
                "the text path requires no attention bias and K=V".to_string(),
            ));
        }
        if self.enable_moe_block {
            return Err(ConfigError::Invalid(
                "MoE text blocks are not supported".to_string(),
            ));
        }
        if self.hidden_size_per_layer_input != 0
            && self.hidden_size_per_layer_input != self.hidden_size
        {
            return Err(ConfigError::Invalid(
                "per-layer hidden input size is not supported".to_string(),
            ));
        }
        if self.vocab_size_per_layer_input != 0
            && self.vocab_size_per_layer_input != self.vocab_size
        {
            return Err(ConfigError::Invalid(
                "per-layer vocabulary input size is not supported".to_string(),
            ));
        }
        validate_rope("sliding_attention", &self.rope.sliding)?;
        validate_rope("full_attention", &self.rope.full)?;
        if !self.dtype.eq_ignore_ascii_case("bfloat16")
            && !self.dtype.eq_ignore_ascii_case("float16")
        {
            return Err(ConfigError::Invalid(format!(
                "unsupported text dtype {}",
                self.dtype
            )));
        }
        Ok(())
    }

    pub fn validate_current_checkpoint(&self) -> Result<(), ConfigError> {
        self.validate()?;
        if self.num_hidden_layers != 60
            || self.hidden_size != 5376
            || self.intermediate_size != 21504
            || self.vocab_size != 262144
            || self.num_attention_heads != 32
            || self.num_key_value_heads != 16
            || self.num_global_key_value_heads != 4
            || self.head_dim != 256
            || self.global_head_dim != 512
            || self.sliding_window != 1024
        {
            return Err(ConfigError::Invalid(
                "checkpoint dimensions do not match the confirmed Gemma 4 graph".to_string(),
            ));
        }
        for (index, kind) in self.layer_types.iter().enumerate() {
            let expected = if index % 6 == 5 {
                LayerKind::Full
            } else {
                LayerKind::Sliding
            };
            if *kind != expected {
                return Err(ConfigError::Invalid(format!(
                    "layer {index} has {:?}, expected {:?}",
                    kind, expected
                )));
            }
        }
        if self.quantization.bits != 3.0
            || self.quantization.head_bits != 6
            || self.quantization.codebook != "mcg"
        {
            return Err(ConfigError::Invalid(
                "current checkpoint must use 3-bit mcg linears and a 6-bit mcg head".to_string(),
            ));
        }
        if self.rope.sliding.rope_type != "default"
            || self.rope.sliding.theta != 10_000.0
            || self.rope.full.rope_type != "proportional"
            || self.rope.full.theta != 1_000_000.0
            || self.rope.full.partial_rotary_factor != Some(0.25)
        {
            return Err(ConfigError::Invalid(
                "checkpoint RoPE parameters do not match the confirmed graph".to_string(),
            ));
        }
        Ok(())
    }

    pub fn validate_quantization_summary(
        &self,
        quantization: &QuantizationConfig,
    ) -> Result<(), ConfigError> {
        if self.quantization.quant_method != quantization.quant_method
            || self.quantization.version != quantization.version
            || self.quantization.bits != quantization.bits
            || self.quantization.head_bits != quantization.head_bits
            || self.quantization.out_scales != quantization.out_scales
            || self.quantization.codebook != quantization.codebook
            || self.quantization.calibration_rows != quantization.calibration.rows
            || self.quantization.calibration_cols != quantization.calibration.cols
        {
            return Err(ConfigError::Invalid(
                "config.json and quantization_config.json disagree".to_string(),
            ));
        }
        Ok(())
    }
}

fn validate_rope(name: &str, rope: &RopeLayerConfig) -> Result<(), ConfigError> {
    if !rope.theta.is_finite() || rope.theta <= 0.0 {
        return Err(ConfigError::Invalid(format!(
            "{name} rope theta must be finite and positive"
        )));
    }
    if let Some(factor) = rope.partial_rotary_factor
        && (!factor.is_finite() || factor <= 0.0 || factor > 1.0)
    {
        return Err(ConfigError::Invalid(format!(
            "{name} partial rotary factor must be in (0, 1]"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerationConfig {
    pub bos_token_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
    pub pad_token_id: Option<u32>,
    pub do_sample: bool,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl GenerationConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.eos_token_ids.is_empty() {
            return Err(ConfigError::Invalid(
                "generation config must contain an EOS token".to_string(),
            ));
        }
        let mut unique = self.eos_token_ids.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != self.eos_token_ids.len() {
            return Err(ConfigError::Invalid(
                "generation EOS IDs must be unique".to_string(),
            ));
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(ConfigError::Invalid(
                "temperature must be finite and non-negative".to_string(),
            ));
        }
        if self.top_k == 0 {
            return Err(ConfigError::Invalid(
                "top_k must be greater than zero".to_string(),
            ));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(ConfigError::Invalid("top_p must be in (0, 1]".to_string()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub quant_method: String,
    pub version: String,
    pub bits: f32,
    pub head_bits: u8,
    pub calibration: CalibrationConfig,
    pub out_scales: String,
    pub codebook: String,
    #[serde(default)]
    pub layout: Option<String>,
    #[serde(default)]
    pub tensor_storage: BTreeMap<String, TensorStorage>,
}

impl QuantizationConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.quant_method != "exl3" {
            return Err(ConfigError::Invalid(format!(
                "unsupported quantization method {}",
                self.quant_method
            )));
        }
        if !is_supported_rate(self.bits) {
            return Err(ConfigError::Invalid(format!(
                "unsupported quantization rate {}",
                self.bits
            )));
        }
        if self.head_bits == 0 || self.head_bits > 8 {
            return Err(ConfigError::Invalid(format!(
                "unsupported head bitrate {}",
                self.head_bits
            )));
        }
        if !matches!(self.codebook.as_str(), "plain" | "mcg" | "mul1") {
            return Err(ConfigError::Invalid(format!(
                "unsupported EXL3 codebook {}",
                self.codebook
            )));
        }
        if self.calibration.rows == 0 || self.calibration.cols == 0 {
            return Err(ConfigError::Invalid(
                "quantization calibration dimensions must be non-zero".to_string(),
            ));
        }
        for (name, storage) in &self.tensor_storage {
            storage.validate(name, &self.codebook)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CalibrationConfig {
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TensorStorage {
    #[serde(default)]
    pub stored_tensors: BTreeMap<String, StoredTensorInfo>,
    pub quant_format: Option<String>,
    pub bits_per_weight: Option<f64>,
    pub mcg_multiplier: Option<u32>,
}

impl TensorStorage {
    pub fn validate(&self, module: &str, codebook: &str) -> Result<(), ConfigError> {
        for (name, tensor) in &self.stored_tensors {
            tensor.validate(name)?;
        }
        let Some(format) = &self.quant_format else {
            return Ok(());
        };
        if format != "exl3" {
            return Err(ConfigError::Invalid(format!(
                "unsupported quant format {format} for {module}"
            )));
        }
        let bits = self
            .bits_per_weight
            .ok_or_else(|| ConfigError::Invalid(format!("missing bits_per_weight for {module}")))?;
        if !is_supported_rate(bits as f32) {
            return Err(ConfigError::Invalid(format!(
                "unsupported EXL3 bitrate {bits} for {module}"
            )));
        }
        if codebook == "mcg" && self.mcg_multiplier != Some(MCG_MULTIPLIER) {
            return Err(ConfigError::Invalid(format!(
                "invalid mcg multiplier for {module}"
            )));
        }
        if !self
            .stored_tensors
            .contains_key(&format!("{module}.trellis"))
        {
            return Err(ConfigError::Invalid(format!(
                "missing trellis tensor for {module}"
            )));
        }
        if codebook == "mcg" && !self.stored_tensors.contains_key(&format!("{module}.mcg")) {
            return Err(ConfigError::Invalid(format!(
                "missing mcg tensor for {module}"
            )));
        }
        let input_signs = [format!("{module}.su"), format!("{module}.suh")];
        let output_signs = [format!("{module}.sv"), format!("{module}.svh")];
        if input_signs
            .iter()
            .filter(|name| self.stored_tensors.contains_key(*name))
            .count()
            != 1
            || output_signs
                .iter()
                .filter(|name| self.stored_tensors.contains_key(*name))
                .count()
                != 1
        {
            return Err(ConfigError::Invalid(format!(
                "expected exactly one input and output sign tensor for {module}"
            )));
        }
        let trellis = &self.stored_tensors[&format!("{module}.trellis")];
        if trellis.shape.len() != 3 || trellis.dtype != "torch.int16" || trellis.n_bytes == 0 {
            return Err(ConfigError::Invalid(format!(
                "invalid trellis metadata for {module}"
            )));
        }
        if trellis.shape[0] == 0
            || trellis.shape[1] == 0
            || trellis.shape[2] == 0
            || !trellis.shape[0].is_multiple_of(16)
            || !trellis.shape[1].is_multiple_of(16)
        {
            return Err(ConfigError::Invalid(format!(
                "trellis dimensions must be non-zero 16x16 tiles for {module}"
            )));
        }
        for sign in [&format!("{module}.su"), &format!("{module}.suh")] {
            if let Some(tensor) = self.stored_tensors.get(sign)
                && (tensor.dtype != "torch.float16" || tensor.n_bytes == 0)
            {
                return Err(ConfigError::Invalid(format!(
                    "invalid input sign metadata for {module}"
                )));
            }
        }
        for sign in [&format!("{module}.sv"), &format!("{module}.svh")] {
            if let Some(tensor) = self.stored_tensors.get(sign)
                && (tensor.dtype != "torch.float16" || tensor.n_bytes == 0)
            {
                return Err(ConfigError::Invalid(format!(
                    "invalid output sign metadata for {module}"
                )));
            }
        }
        if let Some(mcg) = self.stored_tensors.get(&format!("{module}.mcg"))
            && (mcg.dtype != "torch.int32" || mcg.n_bytes != 4 || !mcg.shape.is_empty())
        {
            return Err(ConfigError::Invalid(format!(
                "invalid mcg metadata for {module}"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct StoredTensorInfo {
    pub shape: Vec<usize>,
    pub n_bytes: u64,
    pub dtype: String,
}

impl StoredTensorInfo {
    fn validate(&self, name: &str) -> Result<(), ConfigError> {
        if self.dtype.is_empty() {
            return Err(ConfigError::Invalid(format!("missing dtype for {name}")));
        }
        checked_product(&self.shape)
            .map_err(|_| ConfigError::Invalid(format!("tensor shape overflows for {name}")))?;
        Ok(())
    }
}

fn checked_product(shape: &[usize]) -> Result<u64, ()> {
    shape.iter().try_fold(1_u64, |product, &dimension| {
        product.checked_mul(dimension as u64).ok_or(())
    })
}

pub fn is_supported_rate(rate: f32) -> bool {
    rate.is_finite()
        && ((1.0..=8.0).contains(&rate) && rate.fract() == 0.0
            || rate == 1.5
            || rate == 2.5
            || rate == 3.5)
}

pub fn load_config(path: &Path) -> Result<Gemma4Config, ConfigError> {
    let bytes = fs::read(path).map_err(|source| ConfigError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    parse_config(&bytes)
}

pub fn parse_config(bytes: &[u8]) -> Result<Gemma4Config, ConfigError> {
    let raw: RawConfig = serde_json::from_slice(bytes)?;
    let text = raw.text_config;
    let layer_types = text
        .layer_types
        .iter()
        .map(|value| LayerKind::parse(value))
        .collect::<Result<Vec<_>, _>>()?;
    let config = Gemma4Config {
        model_type: raw.model_type,
        text_model_type: text.model_type,
        dtype: text.dtype,
        hidden_size: text.hidden_size,
        intermediate_size: text.intermediate_size,
        vocab_size: text.vocab_size,
        num_hidden_layers: text.num_hidden_layers,
        num_attention_heads: text.num_attention_heads,
        num_key_value_heads: text.num_key_value_heads,
        num_global_key_value_heads: text.num_global_key_value_heads,
        head_dim: text.head_dim,
        global_head_dim: text.global_head_dim,
        sliding_window: text.sliding_window,
        max_position_embeddings: text.max_position_embeddings,
        rms_norm_eps: text.rms_norm_eps,
        final_logit_softcapping: text.final_logit_softcapping,
        hidden_activation: text.hidden_activation,
        attention_bias: text.attention_bias,
        attention_k_eq_v: text.attention_k_eq_v,
        enable_moe_block: text.enable_moe_block,
        use_cache: text.use_cache,
        tie_word_embeddings: text.tie_word_embeddings,
        hidden_size_per_layer_input: text.hidden_size_per_layer_input,
        vocab_size_per_layer_input: text.vocab_size_per_layer_input,
        layer_types,
        rope: RopeConfig {
            sliding: RopeLayerConfig {
                rope_type: text.rope_parameters.sliding_attention.rope_type,
                theta: text.rope_parameters.sliding_attention.rope_theta,
                partial_rotary_factor: text.rope_parameters.sliding_attention.partial_rotary_factor,
            },
            full: RopeLayerConfig {
                rope_type: text.rope_parameters.full_attention.rope_type,
                theta: text.rope_parameters.full_attention.rope_theta,
                partial_rotary_factor: text.rope_parameters.full_attention.partial_rotary_factor,
            },
        },
        bos_token_id: Some(text.bos_token_id),
        eos_token_id: Some(text.eos_token_id),
        pad_token_id: Some(text.pad_token_id),
        image_token_id: raw.image_token_id,
        audio_token_id: raw.audio_token_id,
        video_token_id: raw.video_token_id,
        quantization: raw
            .quantization_config
            .map(|value| QuantizationSummary {
                quant_method: value.quant_method,
                version: value.version,
                bits: value.bits,
                head_bits: value.head_bits,
                calibration_rows: value.calibration.rows,
                calibration_cols: value.calibration.cols,
                out_scales: value.out_scales,
                codebook: value.codebook,
            })
            .unwrap_or_default(),
    };
    config.validate()?;
    Ok(config)
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    model_type: String,
    #[serde(default)]
    image_token_id: Option<u32>,
    #[serde(default)]
    audio_token_id: Option<u32>,
    #[serde(default)]
    video_token_id: Option<u32>,
    text_config: RawTextConfig,
    #[serde(default)]
    quantization_config: Option<RawQuantizationSummary>,
}

#[derive(Debug, Deserialize)]
struct RawTextConfig {
    model_type: String,
    dtype: String,
    attention_bias: bool,
    attention_k_eq_v: bool,
    enable_moe_block: bool,
    eos_token_id: u32,
    final_logit_softcapping: f32,
    global_head_dim: usize,
    head_dim: usize,
    hidden_activation: String,
    hidden_size: usize,
    hidden_size_per_layer_input: usize,
    intermediate_size: usize,
    layer_types: Vec<String>,
    max_position_embeddings: usize,
    num_attention_heads: usize,
    num_global_key_value_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f32,
    rope_parameters: RawRopeParameters,
    sliding_window: usize,
    tie_word_embeddings: bool,
    use_cache: bool,
    vocab_size: usize,
    vocab_size_per_layer_input: usize,
    bos_token_id: u32,
    pad_token_id: u32,
}

#[derive(Debug, Deserialize)]
struct RawRopeParameters {
    full_attention: RawRopeLayer,
    sliding_attention: RawRopeLayer,
}

#[derive(Debug, Deserialize)]
struct RawRopeLayer {
    rope_type: String,
    rope_theta: f32,
    partial_rotary_factor: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct RawQuantizationSummary {
    quant_method: String,
    version: String,
    bits: f32,
    head_bits: u8,
    calibration: CalibrationConfig,
    out_scales: String,
    codebook: String,
}

#[derive(Debug, Deserialize)]
struct RawGenerationConfig {
    #[serde(default)]
    bos_token_id: Option<u32>,
    #[serde(default)]
    eos_token_id: Option<RawEos>,
    #[serde(default)]
    pad_token_id: Option<u32>,
    #[serde(default = "default_do_sample")]
    do_sample: bool,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default = "default_top_p")]
    top_p: f32,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawEos {
    One(u32),
    Many(Vec<u32>),
}

impl RawEos {
    fn into_ids(self) -> Vec<u32> {
        match self {
            Self::One(id) => vec![id],
            Self::Many(ids) => ids,
        }
    }
}

pub fn load_generation_config(path: &Path) -> Result<GenerationConfig, ConfigError> {
    let bytes = fs::read(path).map_err(|source| ConfigError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    parse_generation_config(&bytes)
}

pub fn parse_generation_config(bytes: &[u8]) -> Result<GenerationConfig, ConfigError> {
    let raw: RawGenerationConfig = serde_json::from_slice(bytes)?;
    let config = GenerationConfig {
        bos_token_id: raw.bos_token_id,
        eos_token_ids: raw
            .eos_token_id
            .map(RawEos::into_ids)
            .unwrap_or_else(|| vec![1]),
        pad_token_id: raw.pad_token_id,
        do_sample: raw.do_sample,
        temperature: raw.temperature,
        top_k: raw.top_k,
        top_p: raw.top_p,
    };
    config.validate()?;
    Ok(config)
}

fn default_do_sample() -> bool {
    true
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_k() -> usize {
    64
}

fn default_top_p() -> f32 {
    0.95
}

pub fn load_quantization_config(path: &Path) -> Result<QuantizationConfig, ConfigError> {
    let bytes = fs::read(path).map_err(|source| ConfigError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    parse_quantization_config(&bytes)
}

pub fn parse_quantization_config(bytes: &[u8]) -> Result<QuantizationConfig, ConfigError> {
    let config: QuantizationConfig = serde_json::from_slice(bytes)?;
    config.validate()?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_structural_config() {
        let bytes = serde_json::to_vec(&json!({
            "model_type": "gemma4",
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
                "layer_types": ["sliding_attention"],
                "max_position_embeddings": 8192,
                "num_attention_heads": 32,
                "num_global_key_value_heads": 4,
                "num_hidden_layers": 1,
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
            }
        }))
        .unwrap();
        let config = parse_config(&bytes).unwrap();
        assert_eq!(config.layer_types, vec![LayerKind::Sliding]);
        config.validate().unwrap();
    }

    #[test]
    fn parses_generation_eos_array_and_defaults() {
        let config =
            parse_generation_config(br#"{"bos_token_id":2,"eos_token_id":[1,106,50]}"#).unwrap();
        assert_eq!(config.eos_token_ids, vec![1, 106, 50]);
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.top_k, 64);
        assert_eq!(config.top_p, 0.95);
    }

    #[test]
    fn rejects_duplicate_eos_ids() {
        let error = parse_generation_config(br#"{"eos_token_id":[1,1]}"#)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unique"));
    }

    #[test]
    fn rejects_invalid_quantization_rate() {
        let bytes = br#"{
            "quant_method":"exl3",
            "version":"0.0.37",
            "bits":4.5,
            "head_bits":6,
            "calibration":{"rows":250,"cols":2048},
            "out_scales":"always",
            "codebook":"mcg",
            "tensor_storage":{}
        }"#;
        let error = parse_quantization_config(bytes).unwrap_err().to_string();
        assert!(error.contains("unsupported quantization rate"));
    }
}
