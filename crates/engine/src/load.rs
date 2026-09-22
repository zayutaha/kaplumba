//! EXL3 artifact loader (T1.1): parse safetensors + configs, stream shards
//! into Metal buffer(s). One buffer per shard (avoids a single ~16 GiB alloc
//! that Metal may reject on 24 GB) — no full-file host mirror retained.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use metal::Buffer;

use crate::metal_ctx::{MetalContext, MetalError};

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("not a model directory (missing {0}): {1}")]
    MissingFile(&'static str, PathBuf),
    #[error("io error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid safetensors header in {path}: {msg}")]
    BadHeader { path: PathBuf, msg: String },
    #[error("invalid JSON in {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("shard listed in index not found: {0}")]
    MissingShard(String),
    #[error("tensor {name} data range out of bounds in {shard}")]
    Range { name: String, shard: String },
    #[error("duplicate tensor name: {0}")]
    Duplicate(String),
    #[error("empty model (no tensors)")]
    Empty,
    #[error("Metal allocation failed for {nbytes} bytes ({what})")]
    Alloc { what: String, nbytes: u64 },
    #[error(transparent)]
    Metal(#[from] MetalError),
}

/// Parsed safetensors tensor entry.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub nbytes: u64,
    /// Offset within this tensor's shard buffer.
    pub data_offset: u64,
    pub shard: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    BF16,
    F16,
    F32,
    I8,
    I16,
    I32,
    U8,
    Bool,
    Other,
}

impl DType {
    pub fn from_safetensors(s: &str) -> Self {
        match s {
            "BF16" => DType::BF16,
            "F16" => DType::F16,
            "F32" => DType::F32,
            "I8" => DType::I8,
            "I16" => DType::I16,
            "I32" => DType::I32,
            "U8" => DType::U8,
            "BOOL" => DType::Bool,
            _ => DType::Other,
        }
    }

    pub fn elem_size(self) -> Option<usize> {
        match self {
            DType::BF16 | DType::F16 | DType::I16 => Some(2),
            DType::F32 | DType::I32 => Some(4),
            DType::I8 | DType::U8 | DType::Bool => Some(1),
            DType::Other => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::BF16 => "BF16",
            DType::F16 => "F16",
            DType::F32 => "F32",
            DType::I8 => "I8",
            DType::I16 => "I16",
            DType::I32 => "I32",
            DType::U8 => "U8",
            DType::Bool => "BOOL",
            DType::Other => "OTHER",
        }
    }
}

/// RoPE variant for one attention flavor (`sliding_attention` / `full_attention`).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeParams {
    pub rope_type: String,
    pub rope_theta: f64,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub num_global_key_value_heads: Option<usize>,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub hidden_activation: Option<String>,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(rename = "attention_k_eq_v", default)]
    pub attention_k_eq_v: bool,
    #[serde(rename = "final_logit_softcapping", default)]
    pub final_logit_softcapping: Option<f64>,
    #[serde(default)]
    pub rope_parameters: HashMap<String, RopeParams>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub text_config: TextConfig,
    #[serde(rename = "eos_token_id", default)]
    pub eos_token_id: Option<serde_json::Value>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct QuantConfig {
    pub quant_method: String,
    pub version: String,
    pub bits: f64,
    pub head_bits: f64,
    pub out_scales: String,
    pub codebook: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct GenerationConfig {
    pub do_sample: bool,
    pub temperature: f64,
    pub top_k: u32,
    pub top_p: f64,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct SafetensorTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

#[derive(Debug, Clone, serde::Deserialize)]
struct SafetensorHeader {
    #[serde(flatten)]
    tensors: HashMap<String, SafetensorTensor>,
}

struct Shard {
    buffer: Buffer,
}

/// Fully loaded EXL3 weights resident in Metal buffers + metadata.
pub struct Exl3Model {
    pub dir: PathBuf,
    pub config: ModelConfig,
    pub quant: QuantConfig,
    pub generation: Option<GenerationConfig>,
    shards: HashMap<String, Shard>,
    tensors: HashMap<String, TensorInfo>,
    pub weights_nbytes: u64,
}

impl Exl3Model {
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    pub fn n_tensors(&self) -> usize {
        self.tensors.len()
    }

    pub fn n_shards(&self) -> usize {
        self.shards.len()
    }

    /// Host pointer into the shared shard buffer (valid while model lives).
    pub fn tensor_bytes(&self, name: &str) -> Option<(&[u8], &TensorInfo)> {
        let info = self.tensors.get(name)?;
        let shard = self.shards.get(&info.shard)?;
        let base = shard.buffer.contents();
        if base.is_null() {
            return None;
        }
        let base = base as *const u8;
        let start = info.data_offset as usize;
        let end = start + info.nbytes as usize;
        if end > shard.buffer.length() as usize {
            return None;
        }
        let slice = unsafe { std::slice::from_raw_parts(base.add(start), info.nbytes as usize) };
        Some((slice, info))
    }

    pub fn resident_bytes(&self) -> u64 {
        self.weights_nbytes
    }

    pub fn resident_gb(&self) -> f64 {
        self.weights_nbytes as f64 / 1e9
    }
}

fn read_json<T: serde::de::DeserializeOwned>(
    path: &Path,
    what: &'static str,
) -> Result<T, LoadError> {
    if !path.is_file() {
        return Err(LoadError::MissingFile(what, path.to_path_buf()));
    }
    let text = std::fs::read_to_string(path).map_err(|source| LoadError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&text).map_err(|source| LoadError::Json {
        path: path.to_path_buf(),
        source,
    })
}

fn parse_safetensors_header(path: &Path) -> Result<HashMap<String, TensorInfo>, LoadError> {
    let mut f = File::open(path).map_err(|source| LoadError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf).map_err(|source| LoadError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    if header_len == 0 || header_len > 512 * 1024 * 1024 {
        return Err(LoadError::BadHeader {
            path: path.to_path_buf(),
            msg: format!("header_len={header_len} out of range"),
        });
    }
    let mut header = vec![0u8; header_len];
    f.read_exact(&mut header).map_err(|source| LoadError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let header: SafetensorHeader =
        serde_json::from_slice(&header).map_err(|e| LoadError::BadHeader {
            path: path.to_path_buf(),
            msg: e.to_string(),
        })?;

    let file_len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let data_start = 8 + header_len as u64;
    let shard_name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();

    let mut out = HashMap::with_capacity(header.tensors.len());
    for (name, t) in header.tensors {
        let (start, end) = (t.data_offsets[0], t.data_offsets[1]);
        if end < start {
            return Err(LoadError::BadHeader {
                path: path.to_path_buf(),
                msg: format!("tensor {name} end<start"),
            });
        }
        if data_start + end > file_len {
            return Err(LoadError::Range {
                name,
                shard: shard_name.clone(),
            });
        }
        let nbytes = end - start;
        let dtype = DType::from_safetensors(&t.dtype);
        if let Some(es) = dtype.elem_size() {
            let elems: usize = t.shape.iter().product();
            if elems as u64 * es as u64 != nbytes && !(t.shape.is_empty() && nbytes <= 16) {
                return Err(LoadError::BadHeader {
                    path: path.to_path_buf(),
                    msg: format!(
                        "tensor {name} size mismatch: shape {:?} * {es} != {nbytes}",
                        t.shape
                    ),
                });
            }
        }
        out.insert(
            name,
            TensorInfo {
                dtype,
                shape: t.shape,
                nbytes,
                data_offset: start,
                shard: shard_name.clone(),
            },
        );
    }
    Ok(out)
}

fn require_file(dir: &Path, name: &'static str) -> Result<PathBuf, LoadError> {
    let p = dir.join(name);
    if !p.is_file() {
        return Err(LoadError::MissingFile(name, p));
    }
    Ok(p)
}

fn read_shard_header_len(path: &Path) -> Result<u64, LoadError> {
    let mut f = File::open(path).map_err(|source| LoadError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf).map_err(|source| LoadError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(u64::from_le_bytes(len_buf))
}

/// Load EXL3 artifact from `dir` into per-shard Metal buffers (shared/UMA).
///
/// Streaming: each shard is read through an 8 MiB staging buffer into its
/// device allocation — no second full-size host copy of the model is retained.
pub fn load_exl3(ctx: &MetalContext, dir: impl AsRef<Path>) -> Result<Exl3Model, LoadError> {
    let dir = dir.as_ref().to_path_buf();
    if !dir.is_dir() {
        return Err(LoadError::MissingFile("model directory", dir.clone()));
    }

    let config: ModelConfig = read_json(&require_file(&dir, "config.json")?, "config.json")?;
    let quant: QuantConfig = read_json(
        &require_file(&dir, "quantization_config.json")?,
        "quantization_config.json",
    )?;
    if quant.quant_method != "exl3" {
        return Err(LoadError::BadHeader {
            path: dir.join("quantization_config.json"),
            msg: format!("quant_method={} (want exl3)", quant.quant_method),
        });
    }
    let generation_path = dir.join("generation_config.json");
    let generation = if generation_path.is_file() {
        Some(read_json(&generation_path, "generation_config.json")?)
    } else {
        None
    };

    let index_path = require_file(&dir, "model.safetensors.index.json")?;
    let index_text = std::fs::read_to_string(&index_path).map_err(|source| LoadError::Io {
        path: index_path.clone(),
        source,
    })?;
    let index: serde_json::Value =
        serde_json::from_str(&index_text).map_err(|source| LoadError::Json {
            path: index_path.clone(),
            source,
        })?;
    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| LoadError::BadHeader {
            path: index_path.clone(),
            msg: "missing weight_map".into(),
        })?;

    let mut by_shard: HashMap<String, Vec<String>> = HashMap::new();
    for (name, shard) in weight_map {
        let shard = shard.as_str().ok_or_else(|| LoadError::BadHeader {
            path: index_path.clone(),
            msg: format!("weight_map[{name}] not a string"),
        })?;
        by_shard
            .entry(shard.to_string())
            .or_default()
            .push(name.clone());
    }
    if by_shard.is_empty() {
        return Err(LoadError::Empty);
    }

    // Parse headers first (fail fast before any large alloc).
    let mut shard_headers: HashMap<String, HashMap<String, TensorInfo>> = HashMap::new();
    let mut shard_sizes: HashMap<String, u64> = HashMap::new();
    let mut tensors: HashMap<String, TensorInfo> = HashMap::new();
    let mut total: u64 = 0;

    let mut shard_names: Vec<String> = by_shard.keys().cloned().collect();
    shard_names.sort();
    for shard in &shard_names {
        let path = dir.join(shard);
        if !path.is_file() {
            return Err(LoadError::MissingShard(shard.clone()));
        }
        let full = parse_safetensors_header(&path)?;
        // Keep only tensors listed in the index for this shard.
        let indexed: HashMap<String, TensorInfo> = full
            .into_iter()
            .filter(|(n, _)| by_shard[shard].iter().any(|x| x == n))
            .collect();
        for name in &by_shard[shard] {
            if !indexed.contains_key(name) {
                return Err(LoadError::MissingShard(format!(
                    "tensor {name} missing from {shard}"
                )));
            }
        }
        let mut size = 0u64;
        for (name, info) in &indexed {
            if tensors.contains_key(name) {
                return Err(LoadError::Duplicate(name.clone()));
            }
            size += info.nbytes;
            total += info.nbytes;
            tensors.insert(name.clone(), info.clone());
        }
        shard_headers.insert(shard.clone(), indexed);
        shard_sizes.insert(shard.clone(), size);
    }

    for name in weight_map.keys() {
        if !tensors.contains_key(name) {
            return Err(LoadError::MissingShard(format!(
                "tensor {name} not in any shard header"
            )));
        }
    }
    if tensors.is_empty() || total == 0 {
        return Err(LoadError::Empty);
    }

    // Allocate + stream each shard.
    const STAGE: usize = 8 * 1024 * 1024;
    let mut stage = vec![0u8; STAGE];
    let mut shards: HashMap<String, Shard> = HashMap::new();

    for shard in &shard_names {
        let path = dir.join(shard);
        let size = shard_sizes[shard];
        let buf = ctx.new_shared_buffer(size as usize);
        if buf.contents().is_null() || buf.length() < size {
            return Err(LoadError::Alloc {
                what: shard.clone(),
                nbytes: size,
            });
        }
        let header_len = read_shard_header_len(&path)?;
        let data_start = 8 + header_len;

        let mut entries: Vec<(&String, &TensorInfo)> =
            shard_headers[shard].iter().map(|(n, i)| (n, i)).collect();
        entries.sort_by_key(|(n, _)| n.as_str());

        let mut f = File::open(&path).map_err(|source| LoadError::Io {
            path: path.clone(),
            source,
        })?;
        let ptr = buf.contents() as *mut u8;

        for (_name, info) in entries {
            f.seek(SeekFrom::Start(data_start + info.data_offset))
                .map_err(|source| LoadError::Io {
                    path: path.clone(),
                    source,
                })?;
            let mut remaining = info.nbytes;
            let mut off = info.data_offset;
            while remaining > 0 {
                let chunk = remaining.min(STAGE as u64) as usize;
                f.read_exact(&mut stage[..chunk])
                    .map_err(|source| LoadError::Io {
                        path: path.clone(),
                        source,
                    })?;
                unsafe {
                    std::ptr::copy_nonoverlapping(stage.as_ptr(), ptr.add(off as usize), chunk);
                }
                off += chunk as u64;
                remaining -= chunk as u64;
            }
        }

        shards.insert(shard.clone(), Shard { buffer: buf });
    }

    // Fix tensor.shard-relative data_offset already equals file data offset
    // which matches buffer layout (packed = original safetensors layout).
    Ok(Exl3Model {
        dir,
        config,
        quant,
        generation,
        shards,
        tensors,
        weights_nbytes: total,
    })
}

pub fn inspect_dir(dir: impl AsRef<Path>) -> Result<ModelConfig, LoadError> {
    read_json(&require_file(dir.as_ref(), "config.json")?, "config.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_safetensors(path: &Path, entries: &[(&str, &[u8])]) {
        let mut map = serde_json::Map::new();
        let mut off = 0u64;
        for (name, data) in entries {
            let mut t = serde_json::Map::new();
            t.insert("dtype".into(), "U8".into());
            t.insert(
                "shape".into(),
                serde_json::Value::Array(vec![data.len().into()]),
            );
            t.insert(
                "data_offsets".into(),
                serde_json::Value::Array(vec![off.into(), (off + data.len() as u64).into()]),
            );
            map.insert(name.to_string(), serde_json::Value::Object(t));
            off += data.len() as u64;
        }
        let header = serde_json::Value::Object(map).to_string();
        let mut f = File::create(path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        for (_, data) in entries {
            f.write_all(data).unwrap();
        }
    }

    #[test]
    fn header_parse_ok() {
        let dir = std::env::temp_dir().join("kaplumba_st_ok");
        std::fs::create_dir_all(&dir).unwrap();
        write_safetensors(
            &dir.join("a.safetensors"),
            &[("t1", b"abcd"), ("t2", b"ef")],
        );
        let got = parse_safetensors_header(&dir.join("a.safetensors")).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got["t1"].nbytes, 4);
        assert_eq!(got["t1"].data_offset, 0);
        assert_eq!(got["t2"].data_offset, 4);
        assert_eq!(got["t1"].dtype, DType::U8);
    }

    #[test]
    fn header_out_of_bounds() {
        let dir = std::env::temp_dir().join("kaplumba_st_oob");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.safetensors");
        let header = r#"{"t":{"dtype":"U8","shape":[100],"data_offsets":[0,100]}}"#;
        let mut f = File::create(&path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        f.write_all(&[0u8; 4]).unwrap();
        let err = parse_safetensors_header(&path).unwrap_err();
        assert!(matches!(err, LoadError::Range { .. }), "got {err:?}");
    }

    #[test]
    fn missing_config_errors() {
        let dir = std::env::temp_dir().join("kaplumba_st_missing");
        std::fs::create_dir_all(&dir).unwrap();
        let err = inspect_dir(&dir).unwrap_err();
        assert!(matches!(err, LoadError::MissingFile("config.json", _)));
    }

    #[test]
    fn bad_header_len_errors() {
        let dir = std::env::temp_dir().join("kaplumba_st_badlen");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("x.safetensors");
        let mut f = File::create(&path).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        let err = parse_safetensors_header(&path).unwrap_err();
        assert!(matches!(err, LoadError::BadHeader { .. }));
    }

    #[test]
    fn load_real_artifact_if_present() {
        let dir = PathBuf::from(std::env::var("KAPLUMBA_MODEL_DIR").unwrap_or_else(|_| {
            let h = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            format!("{h}/.kaplumba/models/Gxma/Gxma")
        }));
        if !dir.join("config.json").is_file() {
            eprintln!("skip: no artifact at {}", dir.display());
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skip: metal {e}");
                return;
            }
        };
        let t0 = std::time::Instant::now();
        let model = load_exl3(&ctx, &dir).expect("load real EXL3 artifact");
        let dt = t0.elapsed();
        assert_eq!(model.quant.quant_method, "exl3");
        assert!(model.n_tensors() > 1000, "n_tensors={}", model.n_tensors());
        assert_eq!(model.n_shards(), 2);
        assert!(
            model.resident_bytes() > 10_000_000_000,
            "too small: {}",
            model.resident_bytes()
        );
        assert!(
            model
                .tensor("model.language_model.embed_tokens.weight")
                .is_some()
        );
        assert!(model.tensor("lm_head.trellis").is_some());
        let (bytes, info) = model
            .tensor_bytes("model.language_model.embed_tokens.weight")
            .expect("embed bytes");
        assert_eq!(bytes.len() as u64, info.nbytes);
        assert_eq!(info.dtype, DType::BF16);
        // Non-zero check: embed first row shouldn't be all zero after load.
        assert!(bytes.iter().any(|&b| b != 0), "embed is all zeros");
        println!(
            "loaded {} tensors / {} shards, {:.3} GB, in {:.1}s",
            model.n_tensors(),
            model.n_shards(),
            model.resident_gb(),
            dt.as_secs_f64()
        );
    }
}
